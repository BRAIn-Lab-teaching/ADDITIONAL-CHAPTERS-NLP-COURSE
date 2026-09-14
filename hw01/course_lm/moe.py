"""Classic token-choice sparse mixture of experts.

Shape notation: ``...`` — arbitrary leading token dimensions, ``T`` — their
flattened product, ``D`` — hidden size, ``E`` — number of experts, ``K`` —
``top_k`` routes per token.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MoEConfig:
    """Sparse-MoE dimensions ``D``, expert width, ``E`` and ``K``."""

    hidden_size: int
    expert_hidden_size: int
    num_experts: int
    top_k: int = 2
    capacity_factor: float = 1.25


@dataclass(frozen=True)
class RouterOutput:
    """Routing result.

    ``selected_experts`` is integer ``[T,K]`` and ``accepted_assignments`` is
    boolean ``[T,K]``. Both count tensors are integer ``[E]``. Entropy, load
    CV, dropping fractions and ``aux_loss`` are scalar floating tensors.
    """

    selected_experts: torch.Tensor
    accepted_assignments: torch.Tensor
    expert_counts_before_capacity: torch.Tensor
    expert_counts_after_capacity: torch.Tensor
    capacity: int
    router_entropy: torch.Tensor
    load_cv: torch.Tensor
    dropped_assignment_fraction: torch.Tensor
    dropped_token_fraction: torch.Tensor
    aux_loss: torch.Tensor


class ExpertMLP(nn.Module):
    def __init__(self, hidden_size: int, expert_hidden_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, expert_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, expert_hidden_size, bias=False)
        self.down_proj = nn.Linear(expert_hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map floating token states ``[N,D]`` to outputs ``[N,D]``."""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoE(nn.Module):
    """Evaluate only the top-k selected experts for each token."""

    def __init__(
        self,
        config: MoEConfig,
        expert_factory: Callable[[int], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        factory = expert_factory or (
            lambda _: ExpertMLP(config.hidden_size, config.expert_hidden_size)
        )
        self.experts = nn.ModuleList(
            [factory(index) for index in range(config.num_experts)]
        )

    def _accepted_assignments(
        self, selected_experts: torch.Tensor, capacity: int
    ) -> torch.Tensor:
        """Apply deterministic per-expert capacity.

        Args:
            selected_experts: Integer expert indices ``[T,K]``.
            capacity: Maximum accepted assignments for each expert.

        Returns:
            Boolean acceptance mask ``[T,K]`` in stable token-major order.
        """
        accepted = torch.zeros_like(selected_experts, dtype=torch.bool)
        for expert_index in range(self.config.num_experts):
            assignments = (selected_experts == expert_index).nonzero()
            # ``assignments`` is ``[N_e,2]``. Each row stores
            # ``(token index, top-k position)`` for this expert.
            # STUDENT TODO: keep no more than ``capacity`` assignments.
            raise NotImplementedError
        return accepted

    def _top_k_routing(
        self, router_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Turn router logits into normalized top-k routes.

        Args:
            router_logits: Floating tensor ``[T,E]``.

        Returns:
            Float32 probabilities ``[T,E]``, normalized selected weights
            ``[T,K]`` and integer selected expert indices ``[T,K]``. Normalize
            the weights over the ``K`` routes of each token.
        """
        # STUDENT TODO.
        raise NotImplementedError

    def _dispatch(
        self,
        flat: torch.Tensor,
        selected_experts: torch.Tensor,
        route_weights: torch.Tensor,
        accepted_assignments: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate accepted experts and scatter-add their weighted outputs.

        Args:
            flat: Floating token states ``[T,D]``.
            selected_experts: Integer expert indices ``[T,K]``.
            route_weights: Floating pre-capacity route weights ``[T,K]``.
            accepted_assignments: Boolean capacity mask ``[T,K]``.

        Returns:
            Floating MoE output ``[T,D]``. Surviving route weights are
            renormalized per token; fully dropped tokens produce zeros.
        """
        weights = route_weights * accepted_assignments
        normalizer = weights.sum(dim=-1, keepdim=True)
        weights = weights / normalizer.clamp_min(torch.finfo(weights.dtype).tiny)

        output = torch.zeros_like(flat)
        for expert_index, expert in enumerate(self.experts):
            assigned = (selected_experts == expert_index) & accepted_assignments
            token_indices, route_indices = assigned.nonzero(as_tuple=True)
            if token_indices.numel() == 0:
                # An expert without accepted assignments must not be called.
                continue

            # STUDENT TODO: evaluate the expert on its assigned tokens, apply
            # the corresponding weights and add the rows to ``output``.
            raise NotImplementedError

        return output

    def _routing_statistics(
        self,
        probabilities: torch.Tensor,
        selected_experts: torch.Tensor,
        accepted_assignments: torch.Tensor,
        capacity: int,
    ) -> RouterOutput:
        """Compute routing statistics and the auxiliary loss.

        Args:
            probabilities: Float32 soft router probabilities ``[T,E]``.
            selected_experts: Integer hard routes ``[T,K]``.
            accepted_assignments: Boolean post-capacity mask ``[T,K]``.
            capacity: Maximum assignments accepted by one expert.

        Returns:
            ``RouterOutput`` with hard loads, dropping rates, entropy, load CV
            and differentiable Switch auxiliary loss.
        """
        # All routing metrics and the RouterOutput construction are provided.
        # The only student task in this function is the Switch auxiliary loss.
        num_experts = self.config.num_experts
        counts_before = torch.bincount(
            selected_experts.flatten(), minlength=num_experts
        )
        counts_after = torch.bincount(
            selected_experts[accepted_assignments], minlength=num_experts
        )

        dropped_assignment_fraction = (~accepted_assignments).float().mean()
        dropped_token_fraction = (~accepted_assignments.any(dim=-1)).float().mean()

        loads = counts_before.float()
        load_cv = loads.std(unbiased=False) / loads.mean()
        safe_probabilities = probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        )
        router_entropy = -(probabilities * safe_probabilities.log()).sum(dim=-1).mean()

        load_fractions = counts_before.float() / selected_experts.numel()
        mean_probabilities = probabilities.mean(dim=0)
        # STUDENT TODO: compute the differentiable Switch auxiliary loss from
        # ``load_fractions`` and ``mean_probabilities``.
        raise NotImplementedError

        return RouterOutput(
            selected_experts=selected_experts,
            accepted_assignments=accepted_assignments,
            expert_counts_before_capacity=counts_before,
            expert_counts_after_capacity=counts_after,
            capacity=capacity,
            router_entropy=router_entropy,
            load_cv=load_cv,
            dropped_assignment_fraction=dropped_assignment_fraction,
            dropped_token_fraction=dropped_token_fraction,
            aux_loss=aux_loss,
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, RouterOutput]:
        """Apply token-choice MoE to floating states ``[...,D]``.

        Returns a tensor with the same shape and routing data whose assignment
        fields have shape ``[T,K]``.
        """
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, self.config.hidden_size)
        router_logits = self.router(flat)
        probabilities, route_weights, selected_experts = self._top_k_routing(
            router_logits
        )
        capacity = math.ceil(
            self.config.capacity_factor
            * flat.shape[0]
            * self.config.top_k
            / self.config.num_experts
        )
        accepted = self._accepted_assignments(selected_experts, capacity)
        output = self._dispatch(flat, selected_experts, route_weights, accepted)
        routing = self._routing_statistics(
            probabilities, selected_experts, accepted, capacity
        )
        return output.reshape(original_shape), routing


def moe_parameter_counts(moe: SparseMoE) -> tuple[int, int]:
    """Compare total and per-token active parameter counts.

    Args:
        moe: Sparse MoE layer whose router and experts are counted.

    Returns:
        Pair ``(total, active)``. Both include the router; ``active`` includes
        at most ``top_k`` experts for one token.
    """
    router_parameters = sum(parameter.numel() for parameter in moe.router.parameters())
    expert_parameters = [
        sum(parameter.numel() for parameter in expert.parameters())
        for expert in moe.experts
    ]
    total = router_parameters + sum(expert_parameters)
    active = router_parameters + sum(
        sorted(expert_parameters, reverse=True)[: moe.config.top_k]
    )
    return total, active
