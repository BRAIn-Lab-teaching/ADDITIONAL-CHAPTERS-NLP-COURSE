"""Multi-head latent attention with a compressed decode cache.

Shape notation: ``B`` — batch, ``Q`` — current query length, ``S`` — total
key length, ``H`` — heads, ``D`` — hidden size, ``Dn`` — content head
dimension, ``Dr`` — RoPE head dimension, ``Dv`` — value head dimension,
``R`` — latent KV rank.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from .modeling import RotaryEmbedding, causal_mask


@dataclass(frozen=True)
class MLAConfig:
    hidden_size: int
    num_attention_heads: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rope_theta: float = 10_000.0


@dataclass(frozen=True)
class MLACache:
    """Compressed state: ``kv_latent [B,S,R]`` and ``rope_key [B,S,Dr]``."""

    kv_latent: torch.Tensor
    rope_key: torch.Tensor

    @property
    def length(self) -> int:
        return self.kv_latent.shape[1]


class MultiHeadLatentAttention(nn.Module):
    """MLA with decoupled RoPE and explicit or absorbed projections."""

    def __init__(self, config: MLAConfig, rope: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank

        q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * q_head_dim, bias=False
        )
        self.kv_down_proj = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_up_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, config.hidden_size, bias=False
        )
        self.rope = rope or RotaryEmbedding(self.qk_rope_head_dim, config.rope_theta)
        self.scale = q_head_dim**-0.5

    def _project_current(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build queries and the compressed KV state for the current block.

        Args:
            hidden_states: Floating tensor ``[B,Q,D]``.
            position_ids: Integer absolute positions ``[B,Q]``.

        Returns:
            ``q_nope [B,H,Q,Dn]``, ``q_rope [B,H,Q,Dr]``,
            ``current_latent [B,Q,R]`` and ``current_rope_key [B,Q,Dr]``.
            Apply RoPE only to the tensors carrying positional content.
        """
        batch_size, query_length, _ = hidden_states.shape
        q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # STUDENT TODO.
        raise NotImplementedError

    def _append_cache(
        self,
        current_latent: torch.Tensor,
        current_rope_key: torch.Tensor,
        past_key_value: MLACache | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, MLACache | None]:
        """Append ``[B,Q,R]`` and ``[B,Q,Dr]`` to an optional length-``P`` cache.

        Return combined tensors of length ``S=P+Q`` and return their
        ``MLACache`` wrapper only when ``use_cache`` is true. ``use_cache``
        controls only the returned cache: a supplied ``past_key_value`` is
        always part of the attention context.
        """
        kv_latent = current_latent
        rope_key = current_rope_key

        if past_key_value is not None:
            # STUDENT TODO: combine the past and current states with torch.cat.
            raise NotImplementedError

        present = MLACache(kv_latent, rope_key) if use_cache else None
        return kv_latent, rope_key, present

    def _visibility(
        self,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        attention_mask: torch.Tensor | None,
        visibility_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return boolean visibility ``[1,1,Q,S]`` or ``[B,1,Q,S]``."""
        if visibility_mask is None:
            visible = causal_mask(query_positions, key_positions).unsqueeze(0)
        else:
            visible = visibility_mask.bool()
            if visible.ndim == 2:
                visible = visible.unsqueeze(0)
        if attention_mask is not None:
            visible = visible & attention_mask[:, None, :].bool()
        return visible.unsqueeze(1)

    def _masked_softmax(
        self, scores: torch.Tensor, visible: torch.Tensor
    ) -> torch.Tensor:
        """Convert float32 scores ``[B,H,Q,S]`` to masked probabilities."""
        scores = (scores * self.scale).masked_fill(
            ~visible, torch.finfo(scores.dtype).min
        )
        return scores.softmax(dim=-1)

    def _naive_attention(
        self,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        kv_latent: torch.Tensor,
        rope_key: torch.Tensor,
        visible: torch.Tensor,
    ) -> torch.Tensor:
        """Run MLA after materializing ordinary per-head keys and values.

        Args:
            q_nope: Floating tensor ``[B,H,Q,Dn]``.
            q_rope: Floating tensor ``[B,H,Q,Dr]``.
            kv_latent: Floating tensor ``[B,S,R]``.
            rope_key: Floating tensor ``[B,S,Dr]``, shared across heads.
            visible: Boolean tensor broadcastable to ``[B,H,Q,S]``.

        Returns:
            Floating output ``[B,Q,D]``. Attention scores and softmax are
            computed in float32.
        """
        # STUDENT TODO. Apply self.o_proj after merging the heads.
        raise NotImplementedError

    def _absorbed_attention(
        self,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        kv_latent: torch.Tensor,
        rope_key: torch.Tensor,
        visible: torch.Tensor,
    ) -> torch.Tensor:
        """Run MLA without materializing per-head keys or values.

        Use ``kv_up_proj.weight`` and ``o_proj.weight`` directly to absorb the
        content-key projection into queries and the value/output projections
        into one map. Treat heads as a tensor axis; ``einsum`` is convenient
        for the value/output contractions. Inputs and output have the same
        shapes as in ``_naive_attention``.
        """
        # STUDENT TODO.
        raise NotImplementedError

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        visibility_mask: torch.Tensor | None = None,
        past_key_value: MLACache | None = None,
        use_cache: bool = False,
        implementation: Literal['naive', 'absorbed'] = 'absorbed',
    ) -> tuple[torch.Tensor, MLACache | None]:
        """Run MLA prefill or cached decode.

        ``hidden_states`` has shape ``[B,Q,D]``. ``attention_mask`` covers all
        ``S`` keys; ``visibility_mask`` has shape ``[Q,S]`` or ``[B,Q,S]``.
        The returned cache has length ``S`` when ``use_cache`` is true.
        """
        batch_size, query_length, _ = hidden_states.shape
        past_length = 0 if past_key_value is None else past_key_value.length
        query_positions = torch.arange(
            past_length,
            past_length + query_length,
            device=hidden_states.device,
        )
        position_ids = query_positions.expand(batch_size, -1)
        q_nope, q_rope, current_latent, current_rope_key = self._project_current(
            hidden_states, position_ids
        )
        kv_latent, rope_key, present = self._append_cache(
            current_latent, current_rope_key, past_key_value, use_cache
        )
        key_positions = torch.arange(kv_latent.shape[1], device=hidden_states.device)
        visible = self._visibility(
            query_positions, key_positions, attention_mask, visibility_mask
        )

        if implementation == 'naive':
            output = self._naive_attention(q_nope, q_rope, kv_latent, rope_key, visible)
        else:
            output = self._absorbed_attention(
                q_nope, q_rope, kv_latent, rope_key, visible
            )
        return output, present


def mla_cache_num_bytes(cache: MLACache) -> int:
    """Return storage in bytes of ``kv_latent [B,S,R]`` and ``rope_key [B,S,Dr]``."""
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in (cache.kv_latent, cache.rope_key)
    )
