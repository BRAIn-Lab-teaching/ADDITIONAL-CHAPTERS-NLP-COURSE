import torch
from torch.nn import functional as F

from course_lm.configuration import CourseConfig
from course_lm.modeling import CourseLM, causal_lm_loss, kv_cache_num_bytes


def tiny_config() -> CourseConfig:
    return CourseConfig(
        vocab_size=31,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
    )


def test_causal_lm_loss_shifts_labels_and_ignores_minus_100() -> None:
    logits = torch.randn(2, 5, 11)
    labels = torch.randint(0, 11, (2, 5))
    labels[0, 3] = -100
    expected = F.cross_entropy(
        logits[:, :-1].reshape(-1, 11),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    torch.testing.assert_close(causal_lm_loss(logits, labels), expected)


def test_model_is_causal() -> None:
    torch.manual_seed(2)
    model = CourseLM(tiny_config()).eval()
    input_ids = torch.randint(0, 31, (1, 7))
    changed = input_ids.clone()
    changed[:, 5:] = torch.randint(0, 31, (1, 2))

    logits, _ = model(input_ids)
    changed_logits, _ = model(changed)
    torch.testing.assert_close(logits[:, :5], changed_logits[:, :5])


def test_model_cache_matches_full_recomputation() -> None:
    torch.manual_seed(3)
    model = CourseLM(tiny_config()).eval()
    input_ids = torch.randint(0, 31, (2, 8))
    full_logits, _ = model(input_ids)

    _, cache = model(input_ids[:, :7], use_cache=True)
    decoded_logits, next_cache = model(
        input_ids[:, 7:],
        attention_mask=torch.ones(2, 8, dtype=torch.long),
        past_key_values=cache,
        use_cache=True,
    )

    torch.testing.assert_close(decoded_logits, full_logits[:, 7:], rtol=1e-5, atol=1e-5)
    assert next_cache is not None and len(next_cache) == tiny_config().num_hidden_layers


def test_greedy_generation_agrees_with_and_without_cache() -> None:
    torch.manual_seed(4)
    model = CourseLM(tiny_config()).eval()
    prompt = torch.randint(0, 31, (2, 5))
    without_cache = model.generate(prompt, max_new_tokens=4, use_cache=False)
    with_cache = model.generate(prompt, max_new_tokens=4, use_cache=True)
    assert torch.equal(without_cache, with_cache)


def test_cache_byte_count_matches_tensor_storage() -> None:
    model = CourseLM(tiny_config()).eval()
    input_ids = torch.randint(0, 31, (2, 5))
    _, cache = model(input_ids, use_cache=True)
    assert cache is not None
    expected = sum(t.numel() * t.element_size() for layer in cache for t in layer)
    assert kv_cache_num_bytes(cache) == expected
