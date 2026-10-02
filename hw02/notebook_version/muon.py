"""Предоставленный Muon: Newton–Schulz в FP16, доступном на Colab T4."""

import math

import torch


def newton_schulz(matrix, ns_steps=5, coefficients=(3.4445, -4.775, 2.0315), eps=1e-7):
    a, b, c = coefficients
    # Нормируем в FP32 до перехода в FP16, чтобы не терять масштаб градиента.
    matrix = matrix.float()
    X = (matrix / matrix.norm().clamp_min(eps)).half()
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(ns_steps):
        gram = X @ X.T
        correction = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        X = torch.addmm(X, correction, X, beta=a)
    return (X.T if transposed else X).float()


class Muon(torch.optim.Optimizer):
    """Интерфейс PyTorch Muon с итерациями, не требующими BF16."""

    def __init__(self, params, lr=1e-3, weight_decay=0.1, momentum=0.95,
                 nesterov=True, ns_coefficients=(3.4445, -4.775, 2.0315),
                 eps=1e-7, ns_steps=5, adjust_lr_fn=None):
        super().__init__(params, dict(
            lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
            ns_coefficients=ns_coefficients, eps=eps, ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        ))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                state = self.state[parameter]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(gradient)
                buffer = state["momentum_buffer"]
                buffer.lerp_(gradient, 1 - group["momentum"])
                direction = gradient.lerp(buffer, group["momentum"]) if group["nesterov"] else buffer
                direction = newton_schulz(
                    direction, group["ns_steps"], group["ns_coefficients"], group["eps"]
                )
                m, n = parameter.shape
                if group["adjust_lr_fn"] == "match_rms_adamw":
                    ratio = 0.2 * math.sqrt(max(m, n))
                else:
                    ratio = math.sqrt(max(1.0, m / n))
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(direction, alpha=-group["lr"] * ratio)
        return loss
