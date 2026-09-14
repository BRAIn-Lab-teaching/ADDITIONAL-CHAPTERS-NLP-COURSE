import torch
from torch.nn import functional as F

from course_lm.configuration import CourseConfig
from course_lm.modeling import RMSNorm, RotaryEmbedding, SwiGLU, repeat_kv


def tiny_config() -> CourseConfig:
    return CourseConfig(
        vocab_size=31,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
    )


def test_rms_norm_matches_definition() -> None:
    norm = RMSNorm(hidden_size=4, eps=1e-6)
    norm.weight.data.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
    x = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
    expected = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + norm.eps)
    expected = expected * norm.weight
    torch.testing.assert_close(norm(x), expected)


def test_rope_uses_explicit_positions_and_preserves_norms() -> None:
    rope = RotaryEmbedding(head_dim=4, theta=10_000.0)
    q = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [2.0, 0.0, -1.0, 3.0]]]])
    k = q.clone()
    positions = torch.tensor([[0, 7]])
    rotated_q, rotated_k = rope(q, k, positions)

    torch.testing.assert_close(rotated_q[..., 0, :], q[..., 0, :])
    torch.testing.assert_close(rotated_k, rotated_q)
    torch.testing.assert_close(rotated_q.norm(dim=-1), q.norm(dim=-1))
    assert not torch.allclose(rotated_q[..., 1, :], q[..., 1, :])


def test_repeat_kv_repeats_each_head_contiguously() -> None:
    x = torch.tensor([[[[1.0]], [[2.0]]]])
    repeated = repeat_kv(x, repeats=3)
    assert repeated.shape == (1, 6, 1, 1)
    torch.testing.assert_close(repeated.flatten(), torch.tensor([1, 1, 1, 2, 2, 2.0]))
    assert repeat_kv(x, repeats=1) is x


def test_swiglu_matches_explicit_formula() -> None:
    layer = SwiGLU(tiny_config())
    x = torch.randn(2, 3, 16)
    expected = layer.down_proj(F.silu(layer.gate_proj(x)) * layer.up_proj(x))
    torch.testing.assert_close(layer(x), expected)
