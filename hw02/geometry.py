"""Геометрическая диагностика обновлений из основного эксперимента."""

from __future__ import annotations

import math
from typing import Any

import torch


def sample_steps(total_steps: int) -> tuple[int, ...]:
    """Use nine checkpoints, denser early, scaled to the run length."""
    reference = (1, 2, 5, 10, 15, 25, 40, 55, 75)
    return tuple(sorted({max(1, round(step * total_steps / 75)) for step in reference}))


class GeometryRecorder:
    """Capture one parameter's real optimizer state at selected steps."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        parameter: torch.nn.Parameter,
        kind: str,
        *,
        sample_steps: tuple[int, ...],
    ) -> None:
        self.optimizer = optimizer
        self.parameter = parameter
        self.kind = kind
        self.sample_steps = set(sample_steps)
        self.snapshots: list[dict[str, Any]] = []
        self._step = 0
        self._pending: torch.Tensor | None = None
        self._hook = (
            optimizer.register_step_post_hook(self._after_muon_step) if kind == "muon" else None
        )

    def _after_muon_step(self, optimizer: torch.optim.Optimizer, *args: Any) -> None:
        self._step += 1
        if self._step not in self.sample_steps:
            return
        group = next(
            group
            for group in optimizer.param_groups
            if any(parameter is self.parameter for parameter in group["params"])
        )
        gradient = self.parameter.grad
        assert gradient is not None
        buffer = optimizer.state[self.parameter]["momentum_buffer"]
        self._pending = (
            gradient.lerp(buffer, group["momentum"]) if group["nesterov"] else buffer.clone()
        )

    def collect(self, step: int) -> None:
        if step not in self.sample_steps:
            return
        if self.kind == "muon":
            assert self._pending is not None
            self.snapshots.append({"step": step, "input": self._pending.float().cpu()})
            self._pending = None
            return
        state = self.optimizer.state[self.parameter]
        group = next(
            group
            for group in self.optimizer.param_groups
            if any(parameter is self.parameter for parameter in group["params"])
        )
        self.snapshots.append(
            {
                "step": step,
                "exp_avg": state["exp_avg"].detach().float().cpu(),
                "exp_avg_sq": state["exp_avg_sq"].detach().float().cpu(),
                "betas": group["betas"],
                "eps": group["eps"],
                "state_step": int(state["step"].item()),
            }
        )

    def close(self) -> None:
        if self._hook is not None:
            self._hook.remove()


def adamw_direction(snapshot: dict[str, Any]) -> torch.Tensor:
    """Bias-corrected AdamW direction before LR and weight decay."""
    beta1, beta2 = snapshot["betas"]
    step = snapshot["state_step"]
    mean = snapshot["exp_avg"] / (1 - beta1**step)
    variance = snapshot["exp_avg_sq"] / (1 - beta2**step)
    return mean / (variance.sqrt() + snapshot["eps"])


def muon_direction(matrix: torch.Tensor, ns_steps: int) -> torch.Tensor:
    """Replay PyTorch NS on an input that already includes momentum/Nesterov."""
    candidate = torch.nn.Parameter(torch.zeros_like(matrix, dtype=torch.float32))
    candidate.grad = matrix.detach().float().clone()
    optimizer = torch.optim.Muon(
        [candidate],
        lr=1.0,
        momentum=0.0,
        nesterov=False,
        weight_decay=0.0,
        ns_steps=ns_steps,
        adjust_lr_fn="original",
    )
    optimizer.step()
    ratio = math.sqrt(max(1.0, matrix.shape[0] / matrix.shape[1]))
    return -candidate.detach() / ratio
