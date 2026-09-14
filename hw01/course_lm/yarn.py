"""YaRN rotary embedding exercise.

Shape notation: ``B`` — batch, ``Q`` — sequence length,
``Dh`` — even rotary head dimension.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .modeling import rotate_half


def yarn_find_correction_dim(
    num_rotations: float,
    head_dim: int,
    theta: float,
    original_max_position_embeddings: int,
) -> float:
    """Find the continuous RoPE pair index for a rotation boundary.

    Args:
        num_rotations: Positive number of rotations over the original context.
        head_dim: Positive rotary dimension ``Dh``.
        theta: RoPE frequency base greater than one.
        original_max_position_embeddings: Original context length ``L``.

    Returns:
        Continuous coordinate in the ``head_dim / 2`` frequency grid.
    """
    # STUDENT TODO.
    raise NotImplementedError


def yarn_linear_ramp(
    low: int,
    high: int,
    size: int,
    device: torch.device,
) -> torch.Tensor:
    """Construct a clipped transition over a frequency grid.

    Args:
        low: Last grid index on the lower plateau.
        high: First grid index on the upper plateau.
        size: Positive number of frequency pairs.
        device: Device of the returned tensor.

    Returns:
        Float32 tensor ``[size]`` with values in ``[0,1]``.
    """
    if high <= low:
        high = low + 1
    # STUDENT TODO.
    raise NotImplementedError


def yarn_scaled_inv_freq(
    head_dim: int,
    theta: float,
    factor: float,
    original_max_position_embeddings: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ordinary and YaRN-scaled inverse-frequency grids.

    Args:
        head_dim: Positive even rotary dimension ``Dh``.
        theta: RoPE frequency base greater than one.
        factor: Context extension factor, at least one.
        original_max_position_embeddings: Original context length.
        beta_fast: Rotation boundary for high-frequency components.
        beta_slow: Rotation boundary for low-frequency components.
        device: Optional output device.

    Returns:
        Two float32 tensors of shape ``[Dh/2]``: ordinary and scaled inverse
        frequencies.
    """
    # STUDENT TODO.
    raise NotImplementedError


class YaRNRotaryEmbedding(nn.Module):
    """RoPE with dimension-dependent YaRN frequency interpolation."""

    def __init__(
        self,
        head_dim: int,
        theta: float,
        factor: float,
        original_max_position_embeddings: int,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        attention_factor: float | None = None,
    ) -> None:
        super().__init__()
        _, inv_freq = yarn_scaled_inv_freq(
            head_dim,
            theta,
            factor,
            original_max_position_embeddings,
            beta_fast,
            beta_slow,
        )
        self.attention_factor = (
            0.1 * math.log(factor) + 1.0
            if attention_factor is None
            else attention_factor
        )
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate queries and keys at explicit absolute positions.

        Args:
            q: Floating tensor ``[B,Hq,Q,Dh]``.
            k: Floating tensor ``[B,Hk,Q,Dh]``.
            position_ids: Integer tensor ``[B,Q]``.

        Returns:
            Rotated tensors with the same shapes, dtypes and devices as ``q``
            and ``k`` respectively.

        ``rotate_half`` implements the quarter-turn used in the RoPE formula
        given in the assignment.
        """
        angles = position_ids.float().unsqueeze(-1) * self.inv_freq.float()
        angles = torch.repeat_interleave(angles, 2, dim=-1).unsqueeze(1)
        # angles: [B, 1, Q, Dh]
        # STUDENT TODO.
        raise NotImplementedError
