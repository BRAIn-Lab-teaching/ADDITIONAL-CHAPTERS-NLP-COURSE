from dataclasses import dataclass


@dataclass
class CourseConfig:
    vocab_size: int = 8_192
    hidden_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    intermediate_size: int = 704
    max_position_embeddings: int = 2_048
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    rope_type: str = 'rope'
    yarn_factor: float = 1.0
    yarn_original_max_position_embeddings: int = 2_048
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_attention_factor: float | None = None
    attention_type: str = 'gqa'
    mla_kv_lora_rank: int = 64
    mla_qk_nope_head_dim: int = 64
    mla_qk_rope_head_dim: int = 32
    mla_v_head_dim: int = 64
    ffn_type: str = 'dense'
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_expert_hidden_size: int = 256
    moe_capacity_factor: float = 1.25
