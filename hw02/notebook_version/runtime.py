"""Готовая техническая часть экспериментов ноутбука."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict
import gc
import gzip
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from hw02.geometry import GeometryRecorder, sample_steps
from hw02.infra import ExperimentConfig, JsonlLogger, build_schedulers, seed_everything, write_json
from hw02.measurement import PaddingStats, RunMeter


LossFn = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def build_optimizers(
    model: nn.Module,
    kind: str,
    config: ExperimentConfig,
    split_parameters: Callable[[nn.Module], tuple[list[tuple[str, nn.Parameter]], list[tuple[str, nn.Parameter]]]],
    apollo_class: type[torch.optim.Optimizer],
    *,
    muon_class: type[torch.optim.Optimizer] | None = None,
) -> list[torch.optim.Optimizer]:
    if kind == "adamw":
        return [torch.optim.AdamW(model.parameters(), lr=config.adamw.lr,
                                  betas=config.adamw.betas, weight_decay=config.adamw.weight_decay)]
    if kind == "adam_mini":
        from hw02.notebook_version.adam_mini import Adam_mini

        return [Adam_mini(model.named_parameters(), lr=config.adam_mini.lr,
                          weight_decay=config.adamw.weight_decay,
                          dim=model.config.hidden_size,
                          n_heads=model.config.num_attention_heads,
                          n_kv_heads=model.config.num_key_value_heads, verbose=False)]
    matrix, other = split_parameters(model)
    matrix_parameters = [parameter for _, parameter in matrix]
    other_parameters = [parameter for _, parameter in other]
    if kind == "muon":
        muon_class = muon_class or torch.optim.Muon
        return [
            muon_class(matrix_parameters, lr=config.muon.lr, momentum=config.muon.momentum,
                             weight_decay=config.muon.weight_decay, ns_steps=config.muon.ns_steps,
                             adjust_lr_fn=config.muon.adjust_lr_fn),
            torch.optim.AdamW(other_parameters, lr=config.muon.fallback_lr,
                              betas=config.adamw.betas, weight_decay=config.adamw.weight_decay),
        ]
    if kind == "apollo":
        return [
            apollo_class(matrix_parameters, lr=config.apollo.lr, betas=config.adamw.betas,
                         weight_decay=config.adamw.weight_decay, seed=config.train.seed),
            torch.optim.AdamW(other_parameters, lr=config.apollo.fallback_lr,
                              betas=config.adamw.betas, weight_decay=config.adamw.weight_decay),
        ]
    raise ValueError(kind)


def _autocast(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if _use_bf16() else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def _use_bf16() -> bool:
    return torch.cuda.is_bf16_supported() and os.environ.get("HW02_FORCE_FP16") != "1"


def _next_batch(batches: Iterable, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(batches)
        return next(iterator), iterator


def _inputs(batch: Mapping[str, torch.Tensor], device: torch.device):
    return {key: value.to(device) for key, value in batch.items() if key != "labels"}


def run_updates(
    model: nn.Module,
    batches: Iterable[Mapping[str, torch.Tensor]],
    optimizers: Sequence[torch.optim.Optimizer],
    schedulers: Sequence[Any],
    loss_fn: LossFn,
    *,
    device: torch.device,
    steps: int,
    accumulation: int,
    max_grad_norm: float,
    meter: Any | None = None,
    on_step: Callable[[dict[str, float | int]], None] | None = None,
    profiler: Any | None = None,
) -> list[dict[str, float | int]]:
    """Выполнить шаги; градиент нормируется по всем таргет-токенам окна."""
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and not _use_bf16(), init_scale=8.0
    )
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    model.train()
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    iterator = iter(batches)
    records: list[dict[str, float | int]] = []
    total_tokens = 0
    for step in range(1, steps + 1):
        if meter is not None:
            meter.begin_step(step)
        window_loss: torch.Tensor | None = None
        window_tokens = 0
        for _ in range(accumulation):
            with torch.profiler.record_function("HW02/data"):
                batch, iterator = _next_batch(batches, iterator)
                window_tokens += int((batch["labels"][:, 1:] != -100).sum())
                labels = batch["labels"].to(device)
            with torch.profiler.record_function("HW02/forward"), _autocast(device):
                output = model(**_inputs(batch, device))
                loss_sum, _ = loss_fn(output.logits, labels)
            with torch.profiler.record_function("HW02/backward"):
                scaler.scale(loss_sum).backward()
            detached = loss_sum.detach()
            window_loss = detached if window_loss is None else window_loss + detached
        if scaler.is_enabled():
            for optimizer in optimizers:
                scaler.unscale_(optimizer)
        with torch.profiler.record_function("HW02/update"):
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.div_(window_tokens)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            for optimizer in optimizers:
                scaler.step(optimizer)
            scaler.update()
            for scheduler in schedulers:
                scheduler.step()
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
        total_tokens += window_tokens
        if meter is not None:
            meter.end_step(step, window_tokens)
        assert window_loss is not None
        loss_value, norm_value = torch.stack(
            (window_loss / window_tokens, grad_norm.detach())
        ).float().cpu().tolist()
        record: dict[str, float | int] = {
            "optimizer_step": step,
            "target_tokens": total_tokens,
            "window_target_tokens": window_tokens,
            "train_loss": loss_value,
            "grad_norm": norm_value,
        }
        records.append(record)
        if on_step is not None:
            on_step(record)
        if profiler is not None:
            profiler.step()
    return records


@torch.no_grad()
def evaluate(
    model: nn.Module,
    batches: Iterable[Mapping[str, torch.Tensor]],
    loss_fn: LossFn,
    *,
    device: torch.device,
) -> tuple[float, int]:
    was_training = model.training
    model.eval()
    total_loss: torch.Tensor | None = None
    total_tokens = 0
    for batch in batches:
        with _autocast(device):
            output = model(**_inputs(batch, device))
            loss_sum, _ = loss_fn(output.logits, batch["labels"].to(device))
        detached = loss_sum.detach()
        total_loss = detached if total_loss is None else total_loss + detached
        total_tokens += int((batch["labels"][:, 1:] != -100).sum())
    model.train(was_training)
    assert total_loss is not None
    return float(total_loss / total_tokens), total_tokens


def _measure_activations(
    model: nn.Module,
    dataset: Any,
    collate: Callable,
    loss_fn: LossFn,
    meter: RunMeter,
    optimizers: Sequence[torch.optim.Optimizer],
    device: torch.device,
    microbatch: int,
) -> None:
    longest = max(dataset, key=lambda row: len(row["input_ids"]))
    batch = collate([longest] * microbatch)
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    model.train()
    with meter.track_saved_tensors(), _autocast(device):
        output = model(**_inputs(batch, device))
        loss_sum, _ = loss_fn(output.logits, batch["labels"].to(device))
    loss_sum.backward()
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    torch.random.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng, device)


def run_experiment(
    run_name: str,
    kind: str,
    *,
    new_model: Callable[[bool], nn.Module],
    train_data: Any,
    eval_data: Any,
    collate: Callable,
    loss_fn: LossFn,
    split_parameters: Callable,
    apollo_class: type[torch.optim.Optimizer],
    config: ExperimentConfig,
    root: str | Path,
    steps: int | None = None,
    checkpointing: bool = False,
    profile: bool = False,
    device: torch.device | None = None,
) -> Path:
    """Один воспроизводимый запуск; сохранённый результат не перезаписывается."""
    root = Path(root)
    run_dir = root / "runs" / run_name
    total_steps = steps or config.train.max_optimizer_steps
    request = {"kind": kind, "steps": total_steps, "checkpointing": checkpointing,
               "profile": profile, "config": asdict(config)}
    request = json.loads(json.dumps(request))
    receipt = run_dir / "request.json"
    if run_dir.exists():
        if (run_dir / "summary.json").is_file() and receipt.is_file() and json.loads(receipt.read_text()) == request:
            if profile and not (run_dir / "profile.json.gz").exists():
                with (run_dir / "profile.json").open("rb") as source, gzip.open(
                    run_dir / "profile.json.gz", "wb", compresslevel=3
                ) as target:
                    shutil.copyfileobj(source, target)
            print(f"{run_name}: используем сохранённый запуск")
            return run_dir
        raise FileExistsError(f"{run_dir}: задайте новое имя запуска; старые результаты не удаляются")
    run_dir.mkdir(parents=True)
    device = device or torch.device("cuda")
    seed_everything(config.train.seed)
    model = new_model(checkpointing).to(device)
    generator = torch.Generator().manual_seed(config.train.seed)
    train_loader = DataLoader(train_data, batch_size=config.train.micro_batch_size,
                              shuffle=True, generator=generator, collate_fn=collate,
                              num_workers=config.train.num_workers)
    eval_loader = DataLoader(eval_data, batch_size=config.train.micro_batch_size,
                             shuffle=False, collate_fn=collate, num_workers=config.train.num_workers)
    padding = PaddingStats()
    optimizers = build_optimizers(model, kind, config, split_parameters, apollo_class)
    warmup = min(total_steps - 1, max(1, round(config.train.warmup_steps * total_steps / config.train.max_optimizer_steps)))
    schedulers = build_schedulers(optimizers, warmup_steps=warmup, total_steps=total_steps)
    meter = RunMeter(model=model, optimizers=optimizers, device=device,
                     warmup_steps=min(config.train.meter_warmup_steps, total_steps - 1))
    _measure_activations(model, eval_data, collate, loss_fn, meter, optimizers, device,
                         config.train.micro_batch_size)
    logger = JsonlLogger(run_dir / "history.jsonl")
    initial_loss, initial_count = evaluate(model, eval_loader, loss_fn, device=device)
    logger.write({"record_type": "eval", "optimizer_step": 0, "target_tokens": 0,
                  "eval_loss": initial_loss, "eval_target_tokens": initial_count,
                  "train_wall_time_seconds": 0.0})

    geometry = None
    target_name = "model.layers.14.self_attn.q_proj.weight"
    if kind in {"adamw", "muon"}:
        target = dict(model.named_parameters())[target_name]
        target_optimizer = next(
            optimizer for optimizer in optimizers
            if any(parameter is target for group in optimizer.param_groups for parameter in group["params"])
        )
        geometry = GeometryRecorder(target_optimizer, target, kind,
                                    sample_steps=sample_steps(total_steps))

    meter.start()
    start = time.perf_counter()
    excluded = 0.0
    last_eval = 0
    profiler = None
    if profile:
        from torch.profiler import ProfilerActivity, profile as torch_profile, schedule

        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        profiler = torch_profile(
            activities=activities,
            schedule=schedule(wait=0, warmup=1, active=1, repeat=1),
            on_trace_ready=lambda item: item.export_chrome_trace(str(run_dir / "profile.json")),
            record_shapes=True,
        )

    def on_step(record: dict[str, float | int]) -> None:
        nonlocal excluded, last_eval
        step = int(record["optimizer_step"])
        if geometry is not None and step in geometry.sample_steps:
            meter.pause()
            tick = time.perf_counter()
            geometry.collect(step)
            excluded += time.perf_counter() - tick
            meter.resume()
        logger.write({"record_type": "train", **record})
        if step == 1 or step % max(1, total_steps // 8) == 0:
            print(f"{run_name}: {step}/{total_steps}, таргет-токенов {record['target_tokens']:,}, лосс {record['train_loss']:.4f}", flush=True)
        if step % config.train.eval_every != 0:
            return
        meter.pause()
        tick = time.perf_counter()
        loss, count = evaluate(model, eval_loader, loss_fn, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        excluded += time.perf_counter() - tick
        train_wall = time.perf_counter() - start - excluded
        meter.resume()
        logger.write({"record_type": "eval", "optimizer_step": step,
                      "target_tokens": record["target_tokens"], "eval_loss": loss,
                      "eval_target_tokens": count, "train_wall_time_seconds": train_wall})
        last_eval = step

    context = profiler if profiler is not None else nullcontext()
    with context:
        records = run_updates(model, padding.wrap(train_loader), optimizers, schedulers, loss_fn,
                              device=device, steps=total_steps,
                              accumulation=config.train.gradient_accumulation_steps,
                              max_grad_norm=config.train.max_grad_norm,
                              meter=meter, on_step=on_step, profiler=profiler)
    if profiler is not None:
        sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
        (run_dir / "profile.txt").write_text(
            profiler.key_averages().table(sort_by=sort_key, row_limit=20), encoding="utf-8"
        )
        with (run_dir / "profile.json").open("rb") as source, gzip.open(
            run_dir / "profile.json.gz", "wb", compresslevel=3
        ) as compressed:
            shutil.copyfileobj(source, compressed)
    if geometry is not None:
        geometry.close()
        torch.save({"parameter": target_name, "optimizer": kind, "snapshots": geometry.snapshots},
                   run_dir / "geometry.pt")
    if last_eval != total_steps:
        meter.pause()
        tick = time.perf_counter()
        loss, count = evaluate(model, eval_loader, loss_fn, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        excluded += time.perf_counter() - tick
        train_wall = time.perf_counter() - start - excluded
        meter.resume()
        logger.write({"record_type": "eval", "optimizer_step": total_steps,
                      "target_tokens": records[-1]["target_tokens"], "eval_loss": loss,
                      "eval_target_tokens": count, "train_wall_time_seconds": train_wall})
    measurement = meter.finish()
    measurement.update(padding.summary())
    write_json(run_dir / "summary.json", {
        "optimizer": kind, "steps": total_steps, "checkpointing": checkpointing,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "measurement": measurement,
    })
    write_json(receipt, request)
    del on_step, records, schedulers, geometry, context, profiler
    if kind in {"adamw", "muon"}:
        del target, target_optimizer
    del model, optimizers, meter
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return run_dir
