import torch

from course_lm import CourseConfig, CourseLM
from course_lm.mla import MLACache, MultiHeadLatentAttention
from course_lm.moe import SparseMoE
from course_lm.yarn import YaRNRotaryEmbedding


def advanced_config() -> CourseConfig:
    return CourseConfig(
        vocab_size=31,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        max_position_embeddings=32,
        rope_type='yarn',
        yarn_factor=2.0,
        yarn_original_max_position_embeddings=16,
        attention_type='mla',
        mla_kv_lora_rank=4,
        mla_qk_nope_head_dim=4,
        mla_qk_rope_head_dim=4,
        mla_v_head_dim=4,
        ffn_type='moe',
        moe_num_experts=3,
        moe_top_k=2,
        moe_expert_hidden_size=8,
        moe_capacity_factor=10.0,
    )


def test_advanced_config_builds_the_requested_layers() -> None:
    model = CourseLM(advanced_config())
    for block in model.layers:
        assert isinstance(block.attention, MultiHeadLatentAttention)
        assert isinstance(block.attention.rope, YaRNRotaryEmbedding)
        assert isinstance(block.mlp, SparseMoE)


def test_advanced_model_runs_forward_backward_and_exposes_router_loss() -> None:
    torch.manual_seed(0)
    model = CourseLM(advanced_config())
    input_ids = torch.randint(0, 31, (2, 5))
    logits, cache = model(input_ids, use_cache=True)
    assert logits.shape == (2, 5, 31)
    assert cache is not None and all(
        isinstance(layer_cache, MLACache) for layer_cache in cache
    )
    objective = logits.square().mean() + 0.01 * model.router_aux_loss()
    objective.backward()
    for block in model.layers:
        assert block.mlp.router.weight.grad is not None


def test_advanced_cached_decode_matches_full_model() -> None:
    torch.manual_seed(1)
    model = CourseLM(advanced_config()).eval()
    input_ids = torch.randint(0, 31, (2, 6))
    full_logits, _ = model(input_ids)
    _, cache = model(input_ids[:, :5], use_cache=True)
    decoded_logits, next_cache = model(
        input_ids[:, 5:],
        attention_mask=torch.ones(2, 6, dtype=torch.long),
        past_key_values=cache,
        use_cache=True,
    )
    torch.testing.assert_close(decoded_logits, full_logits[:, 5:], rtol=4e-5, atol=4e-6)
    assert next_cache is not None and all(
        layer_cache.length == 6 for layer_cache in next_cache
    )


def test_runtime_visibility_mask_reaches_attention() -> None:
    torch.manual_seed(2)
    model = CourseLM(advanced_config()).eval()
    input_ids = torch.randint(0, 31, (1, 4))
    changed = input_ids.clone()
    changed[:, 1] = (changed[:, 1] + 1) % 31
    diagonal = torch.eye(4, dtype=torch.bool).unsqueeze(0)
    logits, _ = model(input_ids, visibility_mask=diagonal)
    changed_logits, _ = model(changed, visibility_mask=diagonal)
    torch.testing.assert_close(logits[:, [0, 2, 3]], changed_logits[:, [0, 2, 3]])
