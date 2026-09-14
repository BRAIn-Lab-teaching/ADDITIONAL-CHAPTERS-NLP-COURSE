"""Complete baseline decoder used by HW01 architecture exercises."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .configuration import CourseConfig

# K and V are deliberately stored before GQA head repetition:
# [batch, num_key_value_heads, cached_length, head_dim].
KVCache = tuple[torch.Tensor, torch.Tensor]
ModelCache = tuple[Any, ...]


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize by root mean square over the last dimension."""
        dtype = x.dtype
        normalized = x.float() * torch.rsqrt(
            x.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Map (..., x0, x1, x2, x3) to (..., -x1, x0, -x3, x2)."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the same position-dependent rotation to Q and K.

        q: [batch, num_heads, query_length, head_dim]
        k: [batch, num_kv_heads, query_length, head_dim]
        position_ids: [batch, query_length]
        """
        angles = position_ids.float().unsqueeze(-1) * self.inv_freq.float()
        angles = torch.repeat_interleave(angles, repeats=2, dim=-1).unsqueeze(1)
        cos, sin = angles.cos(), angles.sin()
        q_rotated = q.float() * cos + rotate_half(q.float()) * sin
        k_rotated = k.float() * cos + rotate_half(k.float()) * sin
        return q_rotated.to(q.dtype), k_rotated.to(k.dtype)


def repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    """Repeat each KV head next to itself for grouped-query attention."""
    return x if repeats == 1 else x.repeat_interleave(repeats, dim=1)


def causal_mask(
    query_positions: torch.Tensor, key_positions: torch.Tensor
) -> torch.Tensor:
    """Return boolean causal visibility ``[Q,S]`` for absolute positions."""
    return key_positions[None, :] <= query_positions[:, None]


def make_causal_mask(
    query_length: int,
    key_length: int,
    past_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a boolean mask [query_length, key_length]; True means visible."""
    query_positions = past_length + torch.arange(query_length, device=device)
    key_positions = torch.arange(key_length, device=device)
    return causal_mask(query_positions, key_positions)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: CourseConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.kv_repeats = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )
        if config.rope_type == 'yarn':
            from .yarn import YaRNRotaryEmbedding

            self.rope = YaRNRotaryEmbedding(
                self.head_dim,
                config.rope_theta,
                config.yarn_factor,
                config.yarn_original_max_position_embeddings,
                config.yarn_beta_fast,
                config.yarn_beta_slow,
                config.yarn_attention_factor,
            )
        else:
            self.rope = RotaryEmbedding(self.head_dim, config.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        visibility_mask: torch.Tensor | None = None,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        """Run causal grouped-query self-attention.

        `attention_mask`, when given, covers cached and current tokens and has shape
        [batch, key_length]. A returned cache contains non-repeated K and V.
        """
        batch_size, query_length, _ = x.shape
        q = (
            self.q_proj(x)
            .view(batch_size, query_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, query_length, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, query_length, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        past_length = 0 if past_key_value is None else past_key_value[0].shape[-2]
        position_ids = torch.arange(
            past_length,
            past_length + query_length,
            device=x.device,
        ).expand(batch_size, -1)
        q, k = self.rope(q, k, position_ids)

        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
        present = (k, v) if use_cache else None

        key_length = k.shape[-2]
        if visibility_mask is None:
            visible = make_causal_mask(
                query_length, key_length, past_length, x.device
            ).unsqueeze(0)
        else:
            visible = visibility_mask.bool()
            if visible.ndim == 2:
                visible = visible.unsqueeze(0)
        if attention_mask is not None:
            visible = visible & attention_mask[:, None, :].bool()
        visible = visible.unsqueeze(1)

        k_heads = repeat_kv(k, self.kv_repeats)
        v_heads = repeat_kv(v, self.kv_repeats)
        scores = torch.matmul(q.float(), k_heads.float().transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )
        scores = scores.masked_fill(~visible, torch.finfo(scores.dtype).min)
        probabilities = scores.softmax(dim=-1).to(v_heads.dtype)
        attended = torch.matmul(probabilities, v_heads)
        attended = (
            attended.transpose(1, 2).contiguous().view(batch_size, query_length, -1)
        )
        return self.o_proj(attended), present


class SwiGLU(nn.Module):
    def __init__(self, config: CourseConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderBlock(nn.Module):
    def __init__(self, config: CourseConfig) -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if config.attention_type == 'mla':
            from .mla import MLAConfig, MultiHeadLatentAttention

            if config.rope_type == 'yarn':
                from .yarn import YaRNRotaryEmbedding

                rope: nn.Module | None = YaRNRotaryEmbedding(
                    config.mla_qk_rope_head_dim,
                    config.rope_theta,
                    config.yarn_factor,
                    config.yarn_original_max_position_embeddings,
                    config.yarn_beta_fast,
                    config.yarn_beta_slow,
                    config.yarn_attention_factor,
                )
            else:
                rope = None
            self.attention = MultiHeadLatentAttention(
                MLAConfig(
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    kv_lora_rank=config.mla_kv_lora_rank,
                    qk_nope_head_dim=config.mla_qk_nope_head_dim,
                    qk_rope_head_dim=config.mla_qk_rope_head_dim,
                    v_head_dim=config.mla_v_head_dim,
                    rope_theta=config.rope_theta,
                ),
                rope=rope,
            )
        else:
            self.attention = GroupedQueryAttention(config)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if config.ffn_type == 'moe':
            from .moe import MoEConfig, SparseMoE

            self.mlp = SparseMoE(
                MoEConfig(
                    hidden_size=config.hidden_size,
                    expert_hidden_size=config.moe_expert_hidden_size,
                    num_experts=config.moe_num_experts,
                    top_k=config.moe_top_k,
                    capacity_factor=config.moe_capacity_factor,
                )
            )
        else:
            self.mlp = SwiGLU(config)
        self.last_router_output = None

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None,
        visibility_mask: torch.Tensor | None,
        past_key_value: Any,
        use_cache: bool,
    ) -> tuple[torch.Tensor, Any]:
        """Apply a pre-norm attention block and a pre-norm MLP block."""
        attention_output, present = self.attention(
            self.input_norm(x),
            attention_mask=attention_mask,
            visibility_mask=visibility_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + attention_output
        mlp_output = self.mlp(self.post_attention_norm(x))
        if isinstance(mlp_output, tuple):
            mlp_output, self.last_router_output = mlp_output
        else:
            self.last_router_output = None
        x = x + mlp_output
        return x, present


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy for next-token prediction, ignoring label -100."""
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


class CourseLM(nn.Module):
    def __init__(self, config: CourseConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        visibility_mask: torch.Tensor | None = None,
        past_key_values: Sequence[Any] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, ModelCache | None]:
        """Return logits and, when requested, one KV pair per decoder layer."""
        if past_key_values is None:
            layer_caches: Sequence[Any] = (None,) * len(self.layers)
        else:
            layer_caches = past_key_values

        hidden_states = self.embed_tokens(input_ids)
        next_cache: list[Any] = []
        for layer, layer_cache in zip(self.layers, layer_caches):
            hidden_states, present = layer(
                hidden_states,
                attention_mask=attention_mask,
                visibility_mask=visibility_mask,
                past_key_value=layer_cache,
                use_cache=use_cache,
            )
            if use_cache:
                assert present is not None
                next_cache.append(present)
        logits = self.lm_head(self.norm(hidden_states))
        return logits, tuple(next_cache) if use_cache else None

    def router_aux_loss(self) -> torch.Tensor:
        """Sum auxiliary router losses produced by the most recent forward pass."""
        losses = [
            layer.last_router_output.aux_loss
            for layer in self.layers
            if layer.last_router_output is not None
        ]
        if not losses:
            return self.embed_tokens.weight.new_zeros(())
        return torch.stack(losses).sum()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        use_cache: bool = True,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Greedy generation, supporting full recomputation and cached decode."""
        generated = input_ids
        current_mask = (
            torch.ones_like(input_ids) if attention_mask is None else attention_mask
        )

        cache: ModelCache | None = None
        for _ in range(max_new_tokens):
            model_input = (
                generated if not use_cache or cache is None else generated[:, -1:]
            )
            logits, cache = self(
                model_input,
                attention_mask=current_mask,
                past_key_values=cache,
                use_cache=use_cache,
            )
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
            current_mask = torch.cat(
                (current_mask, torch.ones_like(next_token, dtype=current_mask.dtype)),
                dim=1,
            )
        return generated


def kv_cache_num_bytes(cache: ModelCache) -> int:
    """Return the exact number of bytes occupied by all K/V cache tensors."""
    total = 0
    for layer in cache:
        if hasattr(layer, 'kv_latent') and hasattr(layer, 'rope_key'):
            tensors = (layer.kv_latent, layer.rope_key)
        else:
            tensors = layer
        total += sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    return total
