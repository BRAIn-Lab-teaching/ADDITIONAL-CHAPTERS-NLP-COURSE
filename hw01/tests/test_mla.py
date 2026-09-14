import copy

import pytest
import torch

from course_lm.mla import (
    MLACache,
    MLAConfig,
    MultiHeadLatentAttention,
    mla_cache_num_bytes,
)
from course_lm.yarn import YaRNRotaryEmbedding


def tiny_mla_config() -> MLAConfig:
    return MLAConfig(
        hidden_size=16,
        num_attention_heads=2,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        rope_theta=10_000.0,
    )


def golden_mla_config() -> MLAConfig:
    return MLAConfig(
        hidden_size=4,
        num_attention_heads=2,
        kv_lora_rank=2,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
        rope_theta=10_000.0,
    )


def initialize_golden_mla(layer: MultiHeadLatentAttention) -> None:
    with torch.no_grad():
        layer.q_proj.weight.copy_(
            torch.arange(32, dtype=torch.float32).reshape(8, 4) / 50 - 0.6
        )
        layer.kv_down_proj.weight.copy_(
            torch.arange(16, dtype=torch.float32).reshape(4, 4) / 20 - 0.4
        )
        layer.kv_up_proj.weight.copy_(
            torch.arange(16, dtype=torch.float32).reshape(8, 2) / 25 - 0.3
        )
        layer.o_proj.weight.copy_(
            torch.arange(16, dtype=torch.float32).reshape(4, 4) / 30 - 0.2
        )


def direct_attention_case() -> tuple[
    MultiHeadLatentAttention,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    layer = MultiHeadLatentAttention(golden_mla_config()).eval()
    initialize_golden_mla(layer)
    q_nope = torch.tensor([[[[0.2, -0.1], [0.4, 0.3]], [[-0.2, 0.5], [0.1, -0.4]]]])
    q_rope = torch.tensor([[[[0.3, 0.2], [-0.1, 0.4]], [[0.5, -0.3], [0.2, 0.1]]]])
    kv_latent = torch.tensor([[[0.4, -0.2], [0.1, 0.3], [-0.5, 0.2]]])
    rope_key = torch.tensor([[[0.2, 0.1], [-0.3, 0.4], [0.5, -0.2]]])
    visible = torch.tensor([[[[True, False, False], [True, True, True]]]])
    expected = torch.tensor(
        [
            [
                [0.0024000043, 0.0045333337, 0.0066666631, 0.0087999934],
                [-0.0028907440, 0.0017704134, 0.0064315712, 0.0110927280],
            ]
        ]
    )
    return layer, q_nope, q_rope, kv_latent, rope_key, visible, expected


@pytest.mark.mla_projection
def test_project_current_splits_heads_and_rotates_only_positional_parts() -> None:
    layer = MultiHeadLatentAttention(golden_mla_config()).eval()
    initialize_golden_mla(layer)
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4) / 10 - 0.3
    position_ids = torch.tensor([[2, 5]])

    q_nope, q_rope, latent, rope_key = layer._project_current(
        hidden_states, position_ids
    )

    torch.testing.assert_close(
        q_nope,
        torch.tensor(
            [[[[0.352, 0.304], [-0.560, -0.480]], [[0.160, 0.112], [-0.240, -0.160]]]]
        ),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        q_rope,
        torch.tensor(
            [
                [
                    [[-0.29566750, 0.14622161], [-0.42032066, 0.29279780]],
                    [[-0.04118218, 0.05153670], [-0.02269300, 0.07671396]],
                ]
            ]
        ),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        latent,
        torch.tensor([[[0.22000001, 0.10000000], [-0.29999998, -0.10000000]]]),
    )
    torch.testing.assert_close(
        rope_key,
        torch.tensor([[[0.13562457, 0.04007462], [0.31604347, -0.01079377]]]),
        rtol=1e-5,
        atol=1e-6,
    )


def test_mla_cache_contains_only_latent_and_shared_rope_key() -> None:
    layer = MultiHeadLatentAttention(tiny_mla_config()).eval()
    output, cache = layer(torch.randn(2, 5, 16), use_cache=True)
    assert output.shape == (2, 5, 16)
    assert isinstance(cache, MLACache)
    assert cache.kv_latent.shape == (2, 5, 4)
    assert cache.rope_key.shape == (2, 5, 4)
    assert cache.length == 5
    assert all(tensor.ndim == 3 for tensor in (cache.kv_latent, cache.rope_key))


@pytest.mark.mla_cache
def test_append_cache_concatenates_sequence_axis_and_respects_use_cache() -> None:
    layer = MultiHeadLatentAttention(tiny_mla_config()).eval()
    past = MLACache(
        kv_latent=torch.randn(2, 3, 4),
        rope_key=torch.randn(2, 3, 4),
    )
    current_latent = torch.randn(2, 2, 4)
    current_rope_key = torch.randn(2, 2, 4)

    latent, rope_key, present = layer._append_cache(
        current_latent, current_rope_key, past, use_cache=True
    )
    torch.testing.assert_close(latent, torch.cat((past.kv_latent, current_latent), 1))
    torch.testing.assert_close(
        rope_key, torch.cat((past.rope_key, current_rope_key), 1)
    )
    assert present is not None
    assert present.kv_latent is latent
    assert present.rope_key is rope_key

    _, _, no_present = layer._append_cache(
        current_latent, current_rope_key, past, use_cache=False
    )
    assert no_present is None


@pytest.mark.mla_naive
def test_naive_attention_matches_a_direct_golden_case() -> None:
    layer, q_nope, q_rope, kv_latent, rope_key, visible, expected = (
        direct_attention_case()
    )

    actual = layer._naive_attention(q_nope, q_rope, kv_latent, rope_key, visible)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.mla_absorbed
def test_absorbed_attention_matches_a_direct_golden_case() -> None:
    layer, q_nope, q_rope, kv_latent, rope_key, visible, expected = (
        direct_attention_case()
    )

    actual = layer._absorbed_attention(q_nope, q_rope, kv_latent, rope_key, visible)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_mla_cache_byte_count_is_exact_and_smaller_than_expanded_kv() -> None:
    layer = MultiHeadLatentAttention(tiny_mla_config()).eval()
    _, cache = layer(torch.randn(2, 7, 16), use_cache=True)
    assert cache is not None
    expected = sum(
        t.numel() * t.element_size() for t in (cache.kv_latent, cache.rope_key)
    )
    assert mla_cache_num_bytes(cache) == expected
    expanded_kv_bytes = 2 * 2 * 7 * (4 + 4) * 4
    assert mla_cache_num_bytes(cache) < expanded_kv_bytes


def test_naive_and_absorbed_paths_are_numerically_equivalent() -> None:
    torch.manual_seed(0)
    layer = MultiHeadLatentAttention(tiny_mla_config()).eval()
    x = torch.randn(2, 6, 16)
    naive, _ = layer(x, implementation='naive')
    absorbed, _ = layer(x, implementation='absorbed')
    torch.testing.assert_close(absorbed, naive, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize('implementation', ['naive', 'absorbed'])
def test_mla_full_forward_matches_independent_golden_output(
    implementation: str,
) -> None:
    layer = MultiHeadLatentAttention(golden_mla_config()).eval()
    initialize_golden_mla(layer)
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) / 10 - 0.5
    expected = torch.tensor(
        [
            [
                [-0.012959991, 0.013066667, 0.039093331, 0.065119989],
                [-0.006006690, 0.006200531, 0.018407755, 0.030614972],
                [0.005147954, 0.000619458, -0.003909039, -0.008437535],
            ]
        ]
    )

    actual, _ = layer(hidden_states, implementation=implementation)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_naive_and_absorbed_paths_have_equivalent_gradients() -> None:
    torch.manual_seed(1)
    naive_layer = MultiHeadLatentAttention(tiny_mla_config())
    absorbed_layer = copy.deepcopy(naive_layer)
    x_naive = torch.randn(2, 4, 16, requires_grad=True)
    x_absorbed = x_naive.detach().clone().requires_grad_(True)
    naive_layer(x_naive, implementation='naive')[0].square().sum().backward()
    absorbed_layer(x_absorbed, implementation='absorbed')[0].square().sum().backward()
    torch.testing.assert_close(x_absorbed.grad, x_naive.grad, rtol=4e-5, atol=4e-6)
    for (name_a, parameter_a), (name_b, parameter_b) in zip(
        naive_layer.named_parameters(),
        absorbed_layer.named_parameters(),
    ):
        assert name_a == name_b
        assert parameter_a.grad is not None, name_a
        assert parameter_b.grad is not None, name_b
        torch.testing.assert_close(
            parameter_b.grad, parameter_a.grad, rtol=5e-5, atol=5e-6
        )


@pytest.mark.mla_absorbed
def test_absorbed_path_does_not_call_expanding_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer, q_nope, q_rope, kv_latent, rope_key, visible, _ = direct_attention_case()

    def forbidden(*args, **kwargs):
        raise AssertionError('expanded K/V were materialized')

    monkeypatch.setattr(layer.kv_up_proj, 'forward', forbidden)
    layer._absorbed_attention(q_nope, q_rope, kv_latent, rope_key, visible)
    with pytest.raises(AssertionError, match='materialized'):
        layer._naive_attention(q_nope, q_rope, kv_latent, rope_key, visible)


def test_absorbed_cached_decode_matches_full_forward() -> None:
    torch.manual_seed(2)
    layer = MultiHeadLatentAttention(tiny_mla_config()).eval()
    x = torch.randn(2, 7, 16)
    full, _ = layer(x, implementation='absorbed')
    prefix, cache = layer(x[:, :6], use_cache=True, implementation='absorbed')
    decoded, next_cache = layer(
        x[:, 6:],
        past_key_value=cache,
        use_cache=True,
        implementation='absorbed',
    )
    assert prefix.shape == (2, 6, 16)
    torch.testing.assert_close(decoded, full[:, 6:], rtol=3e-5, atol=3e-6)
    assert next_cache is not None and next_cache.length == 7


def test_visibility_and_padding_masks_change_only_allowed_dependencies() -> None:
    torch.manual_seed(3)
    layer = MultiHeadLatentAttention(tiny_mla_config()).eval()
    x = torch.randn(1, 4, 16)
    changed = x.clone()
    changed[:, 1] += 100
    visibility = torch.eye(4, dtype=torch.bool).unsqueeze(0)
    padding = torch.tensor([[1, 0, 1, 1]])
    output, _ = layer(x, attention_mask=padding, visibility_mask=visibility)
    changed_output, _ = layer(
        changed, attention_mask=padding, visibility_mask=visibility
    )
    torch.testing.assert_close(output[:, [0, 2, 3]], changed_output[:, [0, 2, 3]])


def test_mla_accepts_yarn_only_for_decoupled_rope_subspace() -> None:
    rope = YaRNRotaryEmbedding(4, 10_000.0, 4.0, 128)
    layer = MultiHeadLatentAttention(tiny_mla_config(), rope=rope).eval()
    output, cache = layer(torch.randn(1, 3, 16), use_cache=True)
    assert output.shape == (1, 3, 16)
    assert cache is not None and cache.rope_key.shape[-1] == 4
