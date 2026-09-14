import torch

from course_lm import modeling
from course_lm.configuration import CourseConfig
from course_lm.modeling import GroupedQueryAttention, make_causal_mask


def tiny_config() -> CourseConfig:
    return CourseConfig(
        vocab_size=31,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
    )


def test_causal_mask_accounts_for_past() -> None:
    mask = make_causal_mask(2, 5, past_length=3, device=torch.device('cpu'))
    expected = torch.tensor(
        [
            [True, True, True, True, False],
            [True, True, True, True, True],
        ]
    )
    assert torch.equal(mask, expected)


def test_make_causal_mask_delegates_to_absolute_position_rule(monkeypatch) -> None:
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def sentinel_mask(
        query_positions: torch.Tensor, key_positions: torch.Tensor
    ) -> torch.Tensor:
        calls.append((query_positions.clone(), key_positions.clone()))
        return torch.zeros(
            query_positions.numel(), key_positions.numel(), dtype=torch.bool
        )

    monkeypatch.setattr(modeling, 'causal_mask', sentinel_mask, raising=False)
    actual = modeling.make_causal_mask(
        query_length=2,
        key_length=5,
        past_length=3,
        device=torch.device('cpu'),
    )

    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], torch.tensor([3, 4]))
    torch.testing.assert_close(calls[0][1], torch.arange(5))
    assert not actual.any()


def test_gqa_cache_has_kv_heads_before_repetition() -> None:
    attention = GroupedQueryAttention(tiny_config()).eval()
    x = torch.randn(2, 5, 16)
    output, cache = attention(x, use_cache=True)
    assert output.shape == x.shape
    assert cache is not None
    key, value = cache
    assert key.shape == (2, 2, 5, 4)
    assert value.shape == key.shape


def test_cached_decode_matches_full_attention() -> None:
    torch.manual_seed(0)
    attention = GroupedQueryAttention(tiny_config()).eval()
    x = torch.randn(2, 6, 16)

    full_output, _ = attention(x, use_cache=False)
    _, cache = attention(x[:, :5], use_cache=True)
    decoded_output, next_cache = attention(
        x[:, 5:],
        attention_mask=torch.ones(2, 6, dtype=torch.long),
        past_key_value=cache,
        use_cache=True,
    )

    torch.testing.assert_close(decoded_output, full_output[:, 5:], rtol=1e-5, atol=1e-6)
    assert next_cache is not None and next_cache[0].shape[-2] == 6


def test_padding_mask_hides_changed_key_and_value() -> None:
    torch.manual_seed(1)
    attention = GroupedQueryAttention(tiny_config()).eval()
    x = torch.randn(1, 4, 16)
    changed = x.clone()
    changed[:, 1] += 100
    mask = torch.tensor([[1, 0, 1, 1]])

    output, _ = attention(x, attention_mask=mask)
    changed_output, _ = attention(changed, attention_mask=mask)
    torch.testing.assert_close(
        output[:, 2:], changed_output[:, 2:], rtol=1e-5, atol=1e-5
    )
