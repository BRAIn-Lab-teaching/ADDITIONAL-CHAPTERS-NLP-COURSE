"""Trainer и измерения, предоставленные для экспериментов в ноутбуке."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from functools import wraps
import gc
import gzip
import json
from pathlib import Path
import shutil
import time

import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback, TrainingArguments

from hw02.geometry import GeometryRecorder as BaseGeometryRecorder, sample_steps
from hw02.infra import JsonlLogger, build_schedulers, load_config, seed_everything, write_json
from hw02.measurement import PaddingStats, RunMeter
from hw02.notebook_version.runtime import _autocast, _use_bf16, build_optimizers
from hw02.notebook_version.muon import Muon
from hw02.notebook_version.memory import measure_memory


class GeometryRecorder(BaseGeometryRecorder):
    def begin_step(self, step):
        # Trainer считает и пропущенные FP16-шаги, hook оптимизатора — нет.
        self._step = step - 1


class OptimizerBundle(torch.optim.Optimizer):
    """Один интерфейс для матричного оптимизатора и AdamW остальных весов."""

    def __init__(self, optimizers):
        self.optimizers = optimizers
        super().__init__([group for optimizer in optimizers for group in optimizer.param_groups], {})

    @torch.no_grad()
    def step(self, closure=None):
        with torch.profiler.record_function("HW02/update"):
            for optimizer in self.optimizers:
                optimizer.step()


def _record(name, function):
    @wraps(function)
    def call(*args, **kwargs):
        with torch.profiler.record_function(name):
            return function(*args, **kwargs)

    return call


def _observe_batches(trainer, padding):
    # Читаем счётчик стандартного Trainer; способ вычисления лосса не меняем.
    get_batch_samples = trainer.get_batch_samples

    @wraps(get_batch_samples)
    def sample(*args, **kwargs):
        batches, count = get_batch_samples(*args, **kwargs)
        trainer.window_target_tokens = int(count)
        for batch in batches:
            padding.observe(batch)
        return batches, count

    trainer.get_batch_samples = _record("HW02/data", sample)
    trainer.compute_loss = _record("HW02/forward", trainer.compute_loss)
    trainer.training_step = _record("HW02/microbatch", trainer.training_step)


@torch.no_grad()
def evaluate(model, batches, *, device):
    """Лосс модели, усреднённый по таргет-токенам валидационного набора."""
    was_training = model.training
    model.eval()
    total_loss = torch.zeros((), device=device)
    total_tokens = 0
    for batch in batches:
        count = int((batch["labels"][:, 1:] != -100).sum())
        with _autocast(device):
            loss = model(**{key: value.to(device) for key, value in batch.items()})["loss"]
        total_loss += loss * count
        total_tokens += count
    model.train(was_training)
    return float(total_loss / total_tokens), total_tokens


def _measure_activations(model, dataset, collate, meter, optimizers, device, microbatch):
    longest = max(dataset, key=lambda row: len(row["input_ids"]))
    batch = collate([longest] * microbatch)
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    model.train()
    with meter.track_saved_tensors(), _autocast(device):
        loss = model(**{key: value.to(device) for key, value in batch.items()})["loss"]
    loss.backward()
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    torch.random.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng, device)


def _compress_trace(run_dir):
    with (run_dir / "profile.json").open("rb") as source, gzip.open(
        run_dir / "profile.json.gz", "wb", compresslevel=3
    ) as target:
        shutil.copyfileobj(source, target)


def run_experiment(
    run_name, kind, *, new_model, train_data, eval_data, collate,
    split_parameters, apollo_class, root, config=None, steps=None,
    checkpointing=False, profile=False, device=None,
):
    config = config or load_config(Path(__file__).with_name("config.toml"))
    device = device or torch.device("cuda")
    run_dir = Path(root) / "runs" / run_name
    total_steps = steps or config.train.max_optimizer_steps
    request = json.loads(json.dumps({
        "backend": "trainer-default-loss-5.18", "kind": kind, "steps": total_steps,
        "fp16_scaler": "Trainer default",
        "checkpointing": checkpointing, "profile": profile, "config": asdict(config),
    }))
    receipt = run_dir / "request.json"
    if run_dir.exists():
        if (run_dir / "summary.json").is_file() and receipt.is_file() and json.loads(receipt.read_text()) == request:
            if profile and not (run_dir / "profile.json.gz").exists():
                _compress_trace(run_dir)
            print(f"{run_name}: reusing saved run")
            return run_dir
        raise FileExistsError(f"{run_dir}: choose a new RUN_TAG")
    run_dir.mkdir(parents=True)
    names = {"adamw": "AdamW", "muon": "Muon", "adam_mini": "Adam-mini", "apollo": "APOLLO-Mini"}
    print(f"\n=== {names[kind]} · {run_name} ===", flush=True)
    run_start = time.perf_counter()
    seed_everything(config.train.seed)
    model = new_model(checkpointing).to(device)
    optimizers = build_optimizers(model, kind, config, split_parameters, apollo_class,
                                 muon_class=Muon)
    optimizer = OptimizerBundle(optimizers)
    warmup = min(total_steps - 1, max(1, round(config.train.warmup_steps * total_steps / config.train.max_optimizer_steps)))
    scheduler = build_schedulers([optimizer], warmup_steps=warmup, total_steps=total_steps)[0]
    meter = RunMeter(model=model, optimizers=optimizers, device=device,
                     warmup_steps=min(config.train.meter_warmup_steps, total_steps - 1))
    _measure_activations(model, eval_data, collate, meter, optimizers, device,
                         config.train.micro_batch_size)
    eval_loader = DataLoader(eval_data, batch_size=config.train.micro_batch_size,
                             collate_fn=collate, shuffle=False)
    logger = JsonlLogger(run_dir / "history.jsonl")
    loss, count = evaluate(model, eval_loader, device=device)
    logger.write({"record_type": "eval", "optimizer_step": 0, "target_tokens": 0,
                  "eval_loss": loss, "eval_target_tokens": count, "train_wall_time_seconds": 0.0})

    geometry = None
    target_name = "model.layers.14.self_attn.q_proj.weight"
    if kind in {"adamw", "muon"}:
        target = dict(model.named_parameters())[target_name]
        target_optimizer = next(item for item in optimizers
                                if any(parameter is target for group in item.param_groups for parameter in group["params"]))
        geometry = GeometryRecorder(target_optimizer, target, kind, sample_steps=sample_steps(total_steps))

    profiler = None
    if profile:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
            on_trace_ready=lambda item: item.export_chrome_trace(str(run_dir / "profile.json")),
            record_shapes=True,
        )

    class Measurements(TrainerCallback):
        def __init__(self):
            self.target_tokens = 0
            self.excluded = 0.0
            self.skipped_steps = 0

        def on_train_begin(self, args, state, control, **kwargs):
            self.start = time.perf_counter()
            meter.start()

        def on_step_begin(self, args, state, control, **kwargs):
            meter.begin_step(state.global_step + 1)
            if geometry is not None:
                geometry.begin_step(state.global_step + 1)

        def on_step_end(self, args, state, control, **kwargs):
            step = state.global_step
            window_tokens = trainer.window_target_tokens
            self.target_tokens += window_tokens
            meter.end_step(step, window_tokens)
            skipped = trainer.accelerator.optimizer_step_was_skipped
            self.skipped_steps += int(skipped)
            if geometry is not None and step in geometry.sample_steps and not skipped:
                meter.pause()
                tick = time.perf_counter()
                geometry.collect(step)
                self.excluded += time.perf_counter() - tick
                meter.resume()
            if profiler is not None:
                profiler.step()

        def evaluate(self, step):
            meter.pause()
            tick = time.perf_counter()
            loss, count = evaluate(model, eval_loader, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            self.excluded += time.perf_counter() - tick
            train_wall = time.perf_counter() - self.start - self.excluded
            meter.resume()
            logger.write({"record_type": "eval", "optimizer_step": step,
                          "target_tokens": self.target_tokens, "eval_loss": loss,
                          "eval_target_tokens": count, "train_wall_time_seconds": train_wall})
            metrics = {"eval_loss": loss}
            trainer.log(metrics.copy())
            trainer.control = trainer.callback_handler.on_evaluate(
                trainer.args, trainer.state, trainer.control, metrics=metrics,
            )

        def on_log(self, args, state, control, logs=None, **kwargs):
            if "loss" not in logs:
                return
            record = {"record_type": "train", "optimizer_step": state.global_step,
                      "target_tokens": self.target_tokens,
                      "window_target_tokens": trainer.window_target_tokens,
                      "train_loss": logs["loss"], "grad_norm": logs.get("grad_norm")}
            logger.write(record)
            if state.global_step % config.train.eval_every == 0 or state.global_step == total_steps:
                self.evaluate(state.global_step)

    callback = Measurements()
    args = TrainingArguments(
        output_dir=str(run_dir / "trainer"), use_cpu=device.type == "cpu",
        per_device_train_batch_size=config.train.micro_batch_size,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        max_steps=total_steps, max_grad_norm=config.train.max_grad_norm,
        logging_steps=1, save_strategy="no", report_to="none", disable_tqdm=False,
        remove_unused_columns=False, dataloader_num_workers=config.train.num_workers,
        seed=config.train.seed, data_seed=config.train.seed,
        fp16=device.type == "cuda" and not _use_bf16(),
        bf16=device.type == "cuda" and _use_bf16(),
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_data, data_collator=collate,
        optimizers=(optimizer, scheduler),
        callbacks=[callback],
    )
    padding = PaddingStats()
    _observe_batches(trainer, padding)
    with profiler if profiler is not None else nullcontext():
        trainer.train()
    train_wall = time.perf_counter() - callback.start - callback.excluded
    measurement = meter.finish()
    measurement.update(padding.summary())
    if geometry is not None:
        geometry.close()
        torch.save({"parameter": target_name, "optimizer": kind, "snapshots": geometry.snapshots},
                   run_dir / "geometry.pt")
    if profiler is not None:
        key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
        (run_dir / "profile.txt").write_text(profiler.key_averages().table(sort_by=key, row_limit=20))
        _compress_trace(run_dir)
    if device.type == "cuda":
        longest = max(eval_data, key=lambda row: len(row["input_ids"]))
        memory = measure_memory(model, optimizers,
                                collate([longest] * config.train.micro_batch_size), device)
        memory["origin"] = "trained_model_after_run"
        write_json(run_dir / "memory.json", memory)
    write_json(run_dir / "summary.json", {
        "backend": "Trainer", "loss": "model default", "optimizer": kind, "steps": total_steps,
        "muon_ns_dtype": "float16" if kind == "muon" else None,
        "checkpointing": checkpointing, "skipped_steps": callback.skipped_steps,
        "total_wall_time_seconds": time.perf_counter() - run_start,
        "train_wall_time_seconds": train_wall,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "measurement": measurement,
    })
    write_json(receipt, request)
    del trainer, callback, model, optimizer, optimizers, scheduler, meter, geometry, profiler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return run_dir
