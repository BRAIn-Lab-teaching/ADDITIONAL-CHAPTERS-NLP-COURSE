"""Предоставленная инфраструктура ДЗ 2: конфиг, данные и оптимизаторы."""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn


def run_experiment(
    run_name: str,
    *,
    optimizer: Literal["adamw", "muon", "apollo", "adam_mini", "trainer"],
    config_path: str | Path = "config.toml",
    device: str = "cuda",
    microbatch: int | None = None,
    accumulation: int | None = None,
    max_steps: int | None = None,
    warmup: int | None = None,
    checkpointing: bool = False,
    profile: bool = False,
) -> Path:
    """Запустить готовый CLI из notebook или вернуть результат такого же запуска.

    Обучение идёт отдельным процессом; вывод виден в ячейке. Старые результаты
    не перезаписываются: при изменении кода/настроек выберите другое run_name.
    """
    root = Path(__file__).resolve().parent
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    directory = config_path.parent / config.output.directory / run_name
    script = "train_trainer.py" if optimizer == "trainer" else "train_raw.py"
    command = [
        sys.executable,
        "-u",
        str(root / script),
        "--config",
        str(config_path),
        "--run-name",
        run_name,
        "--device",
        device,
    ]
    if optimizer != "trainer":
        command += ["--optimizer", optimizer]
    for flag, value in (("--microbatch", microbatch), ("--accumulation", accumulation)):
        if value is not None:
            command += [flag, str(value)]
    if optimizer != "trainer":
        for flag, value in (("--max-steps", max_steps), ("--warmup", warmup)):
            if value is not None:
                command += [flag, str(value)]
        if checkpointing:
            command.append("--checkpointing")
        if profile:
            command.append("--profile")
    request = {
        "config": asdict(config),
        "arguments": command[2:],
        "code": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (script, "homework.py", "infra.py", "measurement.py", "geometry.py")
        },
    }
    # JSON normalizes tuples (e.g. AdamW betas) to lists for a stable comparison.
    request = json.loads(json.dumps(request))
    receipt = directory / "launch.json"
    if directory.exists() and any(directory.iterdir()):
        if (
            (directory / "summary.json").is_file()
            and receipt.is_file()
            and json.loads(receipt.read_text()) == request
        ):
            print(f"{run_name}: используем сохранённый результат — {directory}")
            return directory
        raise FileExistsError(
            f"{directory}: другой или незавершённый запуск. Выберите новое run_name "
            "либо вручную перенесите старую папку; автоматического удаления нет."
        )
    print(f"{run_name}: запускаем {optimizer} на {device}", flush=True)
    process = subprocess.Popen(
        command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
        code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    if code:
        raise subprocess.CalledProcessError(code, command)
    if not (directory / "summary.json").is_file():
        raise RuntimeError(f"{run_name}: процесс завершился без summary.json")
    write_json(receipt, request)
    return directory


@dataclass(frozen=True)
class ModelConfig:
    name: str
    gradient_checkpointing: bool


@dataclass(frozen=True)
class DataConfig:
    name: str
    subset: str
    text_column: str
    sequence_length: int
    minimum_length: int
    train_documents: int
    eval_documents: int


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    max_optimizer_steps: int
    eval_every: int
    warmup_steps: int
    max_grad_norm: float
    mixed_precision: bool
    num_workers: int
    meter_warmup_steps: int


@dataclass(frozen=True)
class AdamWConfig:
    lr: float
    betas: tuple[float, float]
    weight_decay: float


@dataclass(frozen=True)
class MuonConfig:
    lr: float
    fallback_lr: float
    momentum: float
    weight_decay: float
    ns_steps: int
    adjust_lr_fn: str


@dataclass(frozen=True)
class ApolloConfig:
    lr: float
    fallback_lr: float
    rank: int


@dataclass(frozen=True)
class AdamMiniConfig:
    lr: float


@dataclass(frozen=True)
class TrainerConfig:
    max_optimizer_steps: int
    logging_steps: int
    eval_every: int


@dataclass(frozen=True)
class OutputConfig:
    directory: str


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    adamw: AdamWConfig
    muon: MuonConfig
    apollo: ApolloConfig
    adam_mini: AdamMiniConfig
    trainer: TrainerConfig
    output: OutputConfig


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)

    config = ExperimentConfig(
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw["data"]),
        train=TrainConfig(**raw["train"]),
        adamw=AdamWConfig(
            lr=raw["adamw"]["lr"],
            betas=tuple(raw["adamw"]["betas"]),
            weight_decay=raw["adamw"]["weight_decay"],
        ),
        muon=MuonConfig(
            lr=raw["muon"]["lr"],
            fallback_lr=raw["muon"]["fallback_lr"],
            momentum=raw["muon"]["momentum"],
            weight_decay=raw["muon"]["weight_decay"],
            ns_steps=raw["muon"]["ns_steps"],
            adjust_lr_fn=raw["muon"]["adjust_lr_fn"],
        ),
        apollo=ApolloConfig(**raw["apollo"]),
        adam_mini=AdamMiniConfig(**raw["adam_mini"]),
        trainer=TrainerConfig(**raw["trainer"]),
        output=OutputConfig(**raw["output"]),
    )
    _validate_config(config)
    return config


def _validate_config(config: ExperimentConfig) -> None:
    if config.data.sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if not 2 <= config.data.minimum_length <= config.data.sequence_length:
        raise ValueError("minimum_length must be between 2 and sequence_length")
    positive = {
        "micro_batch_size": config.train.micro_batch_size,
        "gradient_accumulation_steps": config.train.gradient_accumulation_steps,
        "max_optimizer_steps": config.train.max_optimizer_steps,
        "eval_every": config.train.eval_every,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0 <= config.train.warmup_steps < config.train.max_optimizer_steps:
        raise ValueError("warmup_steps must be in [0, max_optimizer_steps)")
    if config.train.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    if len(config.adamw.betas) != 2 or not all(0 <= beta < 1 for beta in config.adamw.betas):
        raise ValueError("AdamW betas must contain two values in [0, 1)")
    if config.muon.adjust_lr_fn != "match_rms_adamw":
        raise ValueError("this assignment fixes Muon scaling to match_rms_adamw")
    if config.apollo.rank != 1:
        raise ValueError("APOLLO-Mini uses rank 1 in this assignment")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_text_datasets(config: ExperimentConfig):
    """Load fixed train/eval slices with raw text columns."""

    from datasets import load_dataset

    raw = load_dataset(config.data.name, config.data.subset)
    if "train" not in raw:
        raise ValueError("dataset does not contain a train split")
    eval_name = "validation" if "validation" in raw else "test"
    if eval_name not in raw:
        raise ValueError("dataset does not contain validation or test split")

    def select(split: Any, limit: int) -> Any:
        if config.data.text_column not in split.column_names:
            raise ValueError(f"text column {config.data.text_column!r} is missing")
        return split.select(range(min(limit, len(split))))

    return (
        select(raw["train"], config.data.train_documents),
        select(raw[eval_name], config.data.eval_documents),
    )


def load_model_and_tokenizer(config: ExperimentConfig, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # FP32 weights; autocast chooses BF16 or FP16 for operations at runtime.
    # Keeping FP32 weights is required for FP16 GradScaler on T4.
    model = AutoModelForCausalLM.from_pretrained(config.model.name, dtype=torch.float32)
    model.config.use_cache = False
    if config.model.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)
    return model, tokenizer


def build_optimizers(
    model: nn.Module,
    *,
    kind: Literal["adamw", "muon", "apollo", "adam_mini"],
    config: ExperimentConfig,
    learning_rate: float,
    parameter_split: Any | None = None,
) -> list[torch.optim.Optimizer]:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if kind == "adamw":
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        return [
            torch.optim.AdamW(
                parameters,
                lr=learning_rate,
                betas=config.adamw.betas,
                weight_decay=config.adamw.weight_decay,
            )
        ]
    if kind == "adam_mini":
        from adam_mini import Adam_mini

        return [
            Adam_mini(
                model.named_parameters(),
                lr=learning_rate,
                weight_decay=config.adamw.weight_decay,
                dim=model.config.hidden_size,
                n_heads=model.config.num_attention_heads,
                n_kv_heads=model.config.num_key_value_heads,
                verbose=False,
            )
        ]
    if kind not in {"muon", "apollo"}:
        raise ValueError(f"unknown optimizer kind: {kind}")
    if parameter_split is None:
        raise ValueError("hybrid Muon requires an explicit parameter split")
    if kind == "muon" and not hasattr(torch.optim, "Muon"):
        raise RuntimeError("torch.optim.Muon is unavailable; install PyTorch 2.9 or newer")

    muon_parameters = [parameter for _, parameter in parameter_split.muon]
    adamw_parameters = [parameter for _, parameter in parameter_split.adamw]
    if not muon_parameters or not adamw_parameters:
        raise ValueError("hybrid Muon requires non-empty Muon and AdamW groups")
    if kind == "apollo":
        from homework import ApolloMini

        return [
            ApolloMini(
                muon_parameters,
                lr=learning_rate,
                betas=config.adamw.betas,
                weight_decay=config.adamw.weight_decay,
                seed=config.train.seed,
            ),
            torch.optim.AdamW(
                adamw_parameters,
                lr=config.apollo.fallback_lr,
                betas=config.adamw.betas,
                weight_decay=config.adamw.weight_decay,
            ),
        ]
    muon = torch.optim.Muon(  # type: ignore[attr-defined]
        muon_parameters,
        lr=learning_rate,
        weight_decay=config.muon.weight_decay,
        momentum=config.muon.momentum,
        ns_steps=config.muon.ns_steps,
        adjust_lr_fn=config.muon.adjust_lr_fn,
    )
    adamw = torch.optim.AdamW(
        adamw_parameters,
        lr=config.muon.fallback_lr,
        betas=config.adamw.betas,
        weight_decay=config.adamw.weight_decay,
    )
    return [muon, adamw]


def warmup_cosine_factor(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if total_steps <= 0 or not 0 <= warmup_steps < total_steps:
        raise ValueError("require 0 <= warmup_steps < total_steps")
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_schedulers(
    optimizers: list[torch.optim.Optimizer],
    *,
    warmup_steps: int,
    total_steps: int,
) -> list[torch.optim.lr_scheduler.LambdaLR]:
    if not optimizers:
        raise ValueError("at least one optimizer is required")

    def factor(step: int) -> float:
        return warmup_cosine_factor(step, warmup_steps=warmup_steps, total_steps=total_steps)

    return [torch.optim.lr_scheduler.LambdaLR(optimizer, factor) for optimizer in optimizers]


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
