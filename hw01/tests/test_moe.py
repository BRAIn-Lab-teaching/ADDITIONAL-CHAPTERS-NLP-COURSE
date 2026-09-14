import math

import pytest
import torch
from torch import nn

from course_lm.moe import MoEConfig, RouterOutput, SparseMoE, moe_parameter_counts


class ConstantExpert(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value
        self.calls = 0
        self.seen_inputs: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.seen_inputs.append(x.detach().clone())
        return torch.full_like(x, self.value)


class ForbiddenExpert(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise AssertionError('unselected expert was evaluated')


class FixedRouter(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = nn.Parameter(logits.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != self.logits.shape[0]:
            raise ValueError('unexpected number of tokens')
        return self.logits


def config(**overrides) -> MoEConfig:
    values = dict(
        hidden_size=4,
        expert_hidden_size=8,
        num_experts=3,
        top_k=2,
        capacity_factor=10.0,
    )
    values.update(overrides)
    return MoEConfig(**values)


def test_top_k_outputs_are_weighted_and_combined_at_original_tokens() -> None:
    layer = SparseMoE(
        config(), expert_factory=lambda index: ConstantExpert(float(index + 1))
    )
    logits = torch.tensor([[3.0, 2.0, 0.0], [0.0, 1.0, 4.0]])
    layer.router = FixedRouter(logits)
    output, routing = layer(torch.randn(2, 4))
    selected_probabilities = logits.softmax(-1).topk(2, dim=-1).values
    selected_probabilities /= selected_probabilities.sum(-1, keepdim=True)
    expected_values = torch.tensor(
        [
            selected_probabilities[0, 0] * 1 + selected_probabilities[0, 1] * 2,
            selected_probabilities[1, 0] * 3 + selected_probabilities[1, 1] * 2,
        ]
    )
    torch.testing.assert_close(output[:, 0], expected_values)
    torch.testing.assert_close(output, expected_values[:, None].expand_as(output))
    assert isinstance(routing, RouterOutput)
    assert routing.selected_experts.tolist() == [[0, 1], [2, 1]]


@pytest.mark.moe_routing
def test_top_k_routing_returns_full_probabilities_and_normalized_routes() -> None:
    layer = SparseMoE(config())
    logits = torch.tensor([[3.0, 2.0, 0.0], [0.0, 1.0, 4.0]])
    probabilities, route_weights, selected_experts = layer._top_k_routing(logits)

    torch.testing.assert_close(probabilities, logits.float().softmax(-1))
    assert selected_experts.tolist() == [[0, 1], [2, 1]]
    torch.testing.assert_close(route_weights.sum(-1), torch.ones(2))
    expected = probabilities.gather(1, selected_experts)
    expected = expected / expected.sum(-1, keepdim=True)
    torch.testing.assert_close(route_weights, expected)


@pytest.mark.moe_routing
def test_router_math_promotes_low_precision_logits_to_float32() -> None:
    layer = SparseMoE(config())
    logits = torch.tensor([[3.0, 2.0, 0.0], [0.0, 1.0, 4.0]], dtype=torch.bfloat16)

    probabilities, route_weights, selected_experts = layer._top_k_routing(logits)

    assert probabilities.dtype == torch.float32
    assert route_weights.dtype == torch.float32
    assert selected_experts.dtype == torch.int64
    torch.testing.assert_close(route_weights.sum(-1), torch.ones(2))


@pytest.mark.moe_dispatch
def test_dispatch_renormalizes_surviving_routes_and_scatter_adds() -> None:
    layer = SparseMoE(
        config(), expert_factory=lambda index: ConstantExpert(float(index + 1))
    )
    flat = torch.randn(2, 4)
    selected_experts = torch.tensor([[0, 1], [0, 2]])
    route_weights = torch.tensor([[0.75, 0.25], [0.6, 0.4]])
    accepted = torch.tensor([[True, True], [False, True]])

    output = layer._dispatch(flat, selected_experts, route_weights, accepted)
    torch.testing.assert_close(output[0], torch.full((4,), 1.25))
    torch.testing.assert_close(output[1], torch.full((4,), 3.0))
    assert layer.experts[0].calls == 1
    assert layer.experts[1].calls == 1
    assert layer.experts[2].calls == 1
    torch.testing.assert_close(layer.experts[0].seen_inputs[0], flat[[0]])
    torch.testing.assert_close(layer.experts[1].seen_inputs[0], flat[[0]])
    torch.testing.assert_close(layer.experts[2].seen_inputs[0], flat[[1]])


@pytest.mark.moe_dispatch
def test_dispatch_preserves_model_dtype_with_float32_route_weights() -> None:
    layer = SparseMoE(
        config(num_experts=2, top_k=1),
        expert_factory=lambda index: ConstantExpert(float(index + 1)),
    )
    flat = torch.randn(3, 4, dtype=torch.bfloat16)
    selected_experts = torch.tensor([[0], [1], [0]])
    route_weights = torch.ones(3, 1, dtype=torch.float32)
    accepted = torch.ones(3, 1, dtype=torch.bool)

    output = layer._dispatch(flat, selected_experts, route_weights, accepted)

    assert output.dtype == torch.bfloat16
    assert output.shape == flat.shape


@pytest.mark.moe_capacity
def test_accepted_assignments_applies_capacity_in_stable_token_order() -> None:
    layer = SparseMoE(config(num_experts=3, top_k=2))
    selected_experts = torch.tensor([[0, 1], [0, 2], [0, 1]])

    accepted = layer._accepted_assignments(selected_experts, capacity=2)

    expected = torch.tensor(
        [[True, True], [True, True], [False, True]], dtype=torch.bool
    )
    torch.testing.assert_close(accepted, expected)


@pytest.mark.moe_dispatch
def test_dispatch_does_not_call_an_expert_without_assignments() -> None:
    layer = SparseMoE(
        config(num_experts=2, top_k=1),
        expert_factory=lambda _: ConstantExpert(1.0),
    )
    layer.experts[1] = ForbiddenExpert()
    flat = torch.randn(2, 4)

    output = layer._dispatch(
        flat,
        selected_experts=torch.tensor([[0], [0]]),
        route_weights=torch.ones(2, 1),
        accepted_assignments=torch.ones(2, 1, dtype=torch.bool),
    )

    torch.testing.assert_close(output, torch.ones_like(flat))


@pytest.mark.moe_aux
def test_routing_statistics_use_hard_load_and_soft_router_probabilities() -> None:
    layer = SparseMoE(config(num_experts=3, top_k=1))
    probabilities = torch.tensor(
        [[0.7, 0.2, 0.1], [0.6, 0.1, 0.3], [0.2, 0.3, 0.5]],
        requires_grad=True,
    )
    selected_experts = torch.tensor([[0], [0], [2]])
    accepted = torch.tensor([[True], [False], [True]])

    routing = layer._routing_statistics(
        probabilities, selected_experts, accepted, capacity=1
    )
    assert routing.expert_counts_before_capacity.tolist() == [2, 0, 1]
    assert routing.expert_counts_after_capacity.tolist() == [1, 0, 1]
    assert routing.dropped_assignment_fraction.item() == pytest.approx(1 / 3)
    assert routing.dropped_token_fraction.item() == pytest.approx(1 / 3)
    expected_aux = 3 * (torch.tensor([2 / 3, 0.0, 1 / 3]) * probabilities.mean(0)).sum()
    torch.testing.assert_close(routing.aux_loss, expected_aux)
    assert routing.load_cv.item() == pytest.approx(math.sqrt(2 / 3))
    assert routing.router_entropy.item() == pytest.approx(0.9098058, abs=1e-6)
    routing.aux_loss.backward()
    assert probabilities.grad is not None


@pytest.mark.moe_aux
def test_routing_statistics_are_finite_when_probabilities_underflow_to_zero() -> None:
    layer = SparseMoE(config(num_experts=3, top_k=1))
    probabilities = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], requires_grad=True)
    selected_experts = torch.tensor([[0], [1]])
    accepted = torch.ones(2, 1, dtype=torch.bool)

    routing = layer._routing_statistics(
        probabilities, selected_experts, accepted, capacity=1
    )

    assert routing.router_entropy.item() == pytest.approx(0.0)
    assert torch.isfinite(routing.router_entropy)
    assert torch.isfinite(routing.aux_loss)


def test_unselected_expert_is_not_evaluated() -> None:
    layer = SparseMoE(
        config(num_experts=2, top_k=1), expert_factory=lambda _: ConstantExpert(1.0)
    )
    layer.experts[1] = ForbiddenExpert()
    layer.router = FixedRouter(torch.tensor([[5.0, -5.0], [4.0, -3.0]]))
    output, _ = layer(torch.randn(2, 4))
    assert output.shape == (2, 4)


def test_capacity_keeps_first_assignments_in_stable_token_order() -> None:
    layer = SparseMoE(
        config(num_experts=2, top_k=1, capacity_factor=0.5),
        expert_factory=lambda index: ConstantExpert(float(index + 1)),
    )
    layer.router = FixedRouter(torch.tensor([[5.0, 0.0]] * 4))
    output, routing = layer(torch.randn(4, 4))
    assert routing.capacity == 1
    assert routing.accepted_assignments[:, 0].tolist() == [True, False, False, False]
    torch.testing.assert_close(output[0], torch.ones(4))
    torch.testing.assert_close(output[1:], torch.zeros(3, 4))
    assert routing.dropped_assignment_fraction.item() == pytest.approx(0.75)
    assert routing.dropped_token_fraction.item() == pytest.approx(0.75)
    assert routing.expert_counts_before_capacity.tolist() == [4, 0]
    assert routing.expert_counts_after_capacity.tolist() == [1, 0]


def test_surviving_route_weights_are_renormalized_after_partial_drop() -> None:
    layer = SparseMoE(
        config(num_experts=3, top_k=2, capacity_factor=0.5),
        expert_factory=lambda index: ConstantExpert(float(index + 1)),
    )
    layer.router = FixedRouter(
        torch.tensor(
            [
                [5.0, 4.0, 0.0],
                [5.0, 0.0, 4.0],
                [5.0, 0.0, 4.0],
            ]
        )
    )
    output, routing = layer(torch.randn(3, 4))
    assert routing.capacity == 1
    # token 1 loses expert 0 but keeps expert 2, hence its routed output is exactly 3.
    torch.testing.assert_close(output[1], torch.full((4,), 3.0))
    assert routing.accepted_assignments[1].tolist() == [False, True]


def test_auxiliary_loss_matches_definition_and_reaches_router() -> None:
    torch.manual_seed(0)
    layer = SparseMoE(config(num_experts=2, top_k=1))
    x = torch.randn(5, 4)
    _, routing = layer(x)
    probabilities = layer.router(x).softmax(-1)
    counts = torch.bincount(routing.selected_experts.flatten(), minlength=2).float()
    expected = 2 * ((counts / counts.sum()) * probabilities.mean(0)).sum()
    torch.testing.assert_close(routing.aux_loss, expected)
    routing.aux_loss.backward()
    assert layer.router.weight.grad is not None
    assert torch.isfinite(layer.router.weight.grad).all()


def test_main_moe_output_backpropagates_through_route_weights() -> None:
    torch.manual_seed(7)
    layer = SparseMoE(config(num_experts=3, top_k=2, capacity_factor=10.0))
    hidden_states = torch.randn(5, 4)

    output, _ = layer(hidden_states)
    output.square().sum().backward()

    assert layer.router.weight.grad is not None
    assert torch.count_nonzero(layer.router.weight.grad) > 0
    assert torch.isfinite(layer.router.weight.grad).all()


def test_forward_restores_arbitrary_token_leading_dimensions() -> None:
    layer = SparseMoE(config())
    x = torch.randn(2, 3, 5, 4, requires_grad=True)
    output, routing = layer(x)
    assert output.shape == x.shape
    assert routing.selected_experts.shape == (30, 2)
    output.sum().backward()
    assert x.grad is not None


def test_parameter_counts_separate_total_and_per_token_active_weights() -> None:
    layer = SparseMoE(config(num_experts=4, top_k=2))
    total, active = moe_parameter_counts(layer)
    router = sum(parameter.numel() for parameter in layer.router.parameters())
    one_expert = sum(parameter.numel() for parameter in layer.experts[0].parameters())
    assert total == router + 4 * one_expert
    assert active == router + 2 * one_expert
    assert active < total


def test_capacity_formula_counts_assignments_not_only_tokens() -> None:
    layer = SparseMoE(config(num_experts=3, top_k=2, capacity_factor=1.25))
    _, routing = layer(torch.randn(7, 4))
    assert routing.capacity == math.ceil(1.25 * 7 * 2 / 3)
