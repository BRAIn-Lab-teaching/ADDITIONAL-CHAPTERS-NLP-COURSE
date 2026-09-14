import math

import pytest
import torch

from course_lm.modeling import RotaryEmbedding
from course_lm.yarn import (
    YaRNRotaryEmbedding,
    yarn_find_correction_dim,
    yarn_linear_ramp,
    yarn_scaled_inv_freq,
)


def forward_only_yarn(attention_factor: float = 1.0) -> YaRNRotaryEmbedding:
    """Construct the module without depending on the frequency-grid TODOs."""
    layer = YaRNRotaryEmbedding.__new__(YaRNRotaryEmbedding)
    torch.nn.Module.__init__(layer)
    layer.attention_factor = attention_factor
    layer.register_buffer(
        'inv_freq',
        torch.tensor([1.0, 0.1, 0.01, 0.001]),
        persistent=False,
    )
    return layer


@pytest.mark.yarn_correction
def test_correction_dim_is_inverse_of_number_of_rotations() -> None:
    head_dim = 64
    base = 10_000.0
    original_context = 4096
    rotations = 8.0
    dim = yarn_find_correction_dim(rotations, head_dim, base, original_context)
    frequency = base ** (-2 * dim / head_dim)
    recovered = original_context * frequency / (2 * math.pi)
    assert recovered == pytest.approx(rotations)


@pytest.mark.yarn_ramp
def test_linear_ramp_has_exact_plateaus_and_monotone_transition() -> None:
    ramp = yarn_linear_ramp(low=2, high=5, size=8, device=torch.device('cpu'))
    torch.testing.assert_close(
        ramp,
        torch.tensor([0.0, 0.0, 0.0, 1 / 3, 2 / 3, 1.0, 1.0, 1.0]),
    )
    assert torch.all(ramp[1:] >= ramp[:-1])


@pytest.mark.yarn_frequencies
def test_scaled_frequencies_keep_fast_dims_and_interpolate_slow_dims() -> None:
    original, scaled = yarn_scaled_inv_freq(
        head_dim=64,
        theta=10_000.0,
        factor=8.0,
        original_max_position_embeddings=4096,
        beta_fast=32.0,
        beta_slow=1.0,
    )
    assert scaled[0] == original[0]
    assert scaled[-1] == pytest.approx(original[-1] / 8.0)
    ratio = scaled / original
    assert torch.all(ratio[1:] <= ratio[:-1])
    assert torch.all((ratio >= 1 / 8.0) & (ratio <= 1.0))


@pytest.mark.yarn_frequencies
def test_beta_boundaries_determine_frequency_regimes() -> None:
    original, scaled = yarn_scaled_inv_freq(
        head_dim=64,
        theta=10_000.0,
        factor=4.0,
        original_max_position_embeddings=2048,
        beta_fast=32.0,
        beta_slow=1.0,
    )
    ratio = scaled / original

    torch.testing.assert_close(ratio[:9], torch.ones(9))
    assert torch.all((ratio[9:21] < 1.0) & (ratio[9:21] > 0.25))
    torch.testing.assert_close(ratio[21:], torch.full((11,), 0.25))


def test_factor_one_is_exactly_ordinary_rope() -> None:
    torch.manual_seed(0)
    rope = RotaryEmbedding(head_dim=8, theta=10_000.0)
    yarn = YaRNRotaryEmbedding(
        head_dim=8,
        theta=10_000.0,
        factor=1.0,
        original_max_position_embeddings=128,
    )
    q = torch.randn(2, 4, 5, 8)
    k = torch.randn(2, 2, 5, 8)
    positions = torch.tensor([[0, 1, 2, 3, 4], [7, 8, 9, 10, 11]])
    q_ref, k_ref = rope(q, k, positions)
    q_got, k_got = yarn(q, k, positions)
    torch.testing.assert_close(q_got, q_ref, rtol=0, atol=0)
    torch.testing.assert_close(k_got, k_ref, rtol=0, atol=0)


@pytest.mark.yarn_forward
def test_attention_factor_scales_norm_but_rotation_preserves_directional_norm() -> None:
    yarn = forward_only_yarn(attention_factor=1.25)
    q = torch.randn(2, 3, 4, 8)
    k = torch.randn(2, 1, 4, 8)
    positions = torch.arange(4).expand(2, -1)
    q_rotated, k_rotated = yarn(q, k, positions)
    torch.testing.assert_close(q_rotated.norm(dim=-1), 1.25 * q.norm(dim=-1))
    torch.testing.assert_close(k_rotated.norm(dim=-1), 1.25 * k.norm(dim=-1))


@pytest.mark.yarn_forward
def test_yarn_respects_explicit_positions_and_propagates_gradients() -> None:
    yarn = forward_only_yarn()
    q = torch.randn(1, 2, 2, 8, requires_grad=True)
    k = torch.randn(1, 1, 2, 8, requires_grad=True)
    zero_positions = torch.zeros(1, 2, dtype=torch.long)
    offset_positions = torch.tensor([[128, 129]])
    q_zero, _ = yarn(q, k, zero_positions)
    q_offset, k_offset = yarn(q, k, offset_positions)
    assert not torch.allclose(q_zero, q_offset)
    (q_offset.square().sum() + k_offset.square().sum()).backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()


@pytest.mark.yarn_forward
def test_yarn_preserves_bfloat16_on_cpu() -> None:
    yarn = forward_only_yarn()
    q = torch.randn(2, 3, 5, 8, dtype=torch.bfloat16)
    k = torch.randn(2, 1, 5, 8, dtype=torch.bfloat16)
    positions = torch.arange(5).expand(2, -1)

    q_rotated, k_rotated = yarn(q, k, positions)

    assert q_rotated.dtype == torch.bfloat16
    assert k_rotated.dtype == torch.bfloat16
    assert q_rotated.device.type == 'cpu'
    assert k_rotated.device.type == 'cpu'
    assert torch.isfinite(q_rotated).all()
    assert torch.isfinite(k_rotated).all()
