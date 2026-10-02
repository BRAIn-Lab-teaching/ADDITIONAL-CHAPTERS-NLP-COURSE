"""Короткие проверки упражнений, вызываемые прямо в ноутбуке."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from hw02.measurement import optimizer_state_bytes


def check_causal_loss(
    loss_fn: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
) -> None:
    logits = torch.tensor(
        [[[2.0, 0.0, -1.0], [0.0, 2.0, -1.0], [1.0, 0.0, -1.0]],
         [[0.0, 2.0, -1.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0]]],
        requires_grad=True,
    )
    # 2 одновременно играет роль PAD и настоящего EOS. Значение ID не задаёт маску.
    labels = torch.tensor([[0, 1, 2], [1, 2, -100]])
    total, count = loss_fn(logits, labels)
    assert isinstance(count, torch.Tensor) and count.ndim == 0, (
        "Верните число таргет-токенов скалярным тензором: int(tensor) синхронизирует GPU"
    )
    expected = F.cross_entropy(logits[:, :-1].reshape(-1, 3), labels[:, 1:].reshape(-1),
                               ignore_index=-100, reduction="sum")
    assert count == 3, "Нужно считать незамаскированные таргет-токены после сдвига"
    torch.testing.assert_close(total, expected, rtol=1e-5, atol=1e-6)
    total.backward()
    assert torch.count_nonzero(logits.grad[:, -1]) == 0, "Последний logits не имеет таргета"


def check_causal_collator(collate, *, pad_token_id: int) -> None:
    batch = collate([{"input_ids": [4, 5, pad_token_id]},
                     {"input_ids": [6, pad_token_id]}])
    torch.testing.assert_close(
        batch["input_ids"],
        torch.tensor([[4, 5, pad_token_id], [6, pad_token_id, pad_token_id]]),
    )
    torch.testing.assert_close(batch["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
    torch.testing.assert_close(
        batch["labels"], torch.tensor([[4, 5, pad_token_id], [6, pad_token_id, -100]])
    )


def check_parameter_groups(model, split_parameters) -> None:
    matrix, other = split_parameters(model)
    ids = [id(parameter) for _, parameter in [*matrix, *other]]
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert len(ids) == len(set(ids)) and set(ids) == expected
    output = model.get_output_embeddings()
    output_ids = {id(parameter) for parameter in output.parameters()}
    embedding_ids = {id(parameter) for parameter in model.get_input_embeddings().parameters()}
    matrix_ids = {id(parameter) for _, parameter in matrix}
    assert not matrix_ids & (output_ids | embedding_ids)
    hidden_linear = {
        id(module.weight)
        for module in model.modules()
        if isinstance(module, torch.nn.Linear) and module is not output
    }
    assert matrix_ids == hidden_linear


def check_geometry(
    polar_factor, normalized_singular_values, stable_rank, r01,
    matrix_cosine, relative_frobenius_distance,
) -> None:
    matrix = torch.tensor([[0.0, -3.0], [2.0, 0.0]])
    torch.testing.assert_close(polar_factor(matrix), torch.tensor([[0.0, -1.0], [1.0, 0.0]]))
    diagonal = torch.diag(torch.tensor([3.0, 1.0, 0.0]))
    torch.testing.assert_close(normalized_singular_values(diagonal), torch.tensor([1.0, 1 / 3, 0.0]))
    torch.testing.assert_close(stable_rank(diagonal), torch.tensor(10 / 9))
    assert r01(torch.tensor([1.0, 0.11, 0.09])) == 2
    assert r01(torch.zeros(3)) == 0
    zeros = torch.zeros(3, 2)
    torch.testing.assert_close(normalized_singular_values(zeros), torch.zeros(2))
    torch.testing.assert_close(stable_rank(zeros), torch.tensor(0.0))
    torch.testing.assert_close(matrix_cosine(matrix, 2 * matrix), torch.tensor(1.0))
    torch.testing.assert_close(matrix_cosine(matrix, -matrix), torch.tensor(-1.0))
    torch.testing.assert_close(relative_frobenius_distance(2 * matrix, matrix), torch.tensor(1.0))


def check_apollo_mini(apollo_class) -> None:
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    gradient = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    optimizer = apollo_class([parameter], lr=0.1, betas=(0.0, 0.0), eps=0.0,
                             weight_decay=0.1, scale=1.0, seed=7)
    parameter.grad = gradient.clone()
    projection = torch.randn(2, generator=torch.Generator().manual_seed(7))
    projected = projection @ gradient
    factor = projected.sign().norm() / projected.norm()
    optimizer.step()
    torch.testing.assert_close(parameter, 0.99 - 0.1 * factor * gradient)
    assert optimizer.state[parameter]["exp_avg"].numel() == 2
    assert optimizer.state[parameter]["exp_avg_sq"].numel() == 2

    wide = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    for gradient in (wide, wide.T):
        parameter = torch.nn.Parameter(torch.zeros_like(gradient))
        optimizer = apollo_class([parameter], lr=0.1, betas=(0.0, 0.0), eps=0.0,
                                 weight_decay=0.0, scale=1.0, seed=9)
        parameter.grad = gradient.clone()
        projection = torch.randn(2, generator=torch.Generator().manual_seed(9))
        projected = projection @ wide
        factor = projected.sign().norm() / projected.norm()
        optimizer.step()
        torch.testing.assert_close(parameter, -0.1 * factor * gradient)
        assert optimizer.state[parameter]["projection"].numel() == 2
        assert optimizer.state[parameter]["exp_avg"].numel() == 3
        assert optimizer.state[parameter]["exp_avg_sq"].numel() == 3

    first = torch.nn.Parameter(torch.zeros(2, 2))
    second = torch.nn.Parameter(torch.zeros(2, 2))
    optimizer = apollo_class([first, second], lr=0.1, betas=(0.0, 0.0),
                             weight_decay=0.0, scale=1.0, seed=13)
    first.grad = torch.ones_like(first)
    second.grad = torch.ones_like(second)
    optimizer.step()
    for offset, parameter in enumerate((first, second)):
        expected = torch.randn(2, generator=torch.Generator().manual_seed(13 + offset))
        torch.testing.assert_close(optimizer.state[parameter]["projection"], expected)

    parameter = torch.nn.Parameter(torch.zeros(1, 2))
    optimizer = apollo_class([parameter], lr=0.1, betas=(0.5, 0.5), eps=0.0,
                             weight_decay=0.0, scale=1.0, seed=17)
    first_gradient = torch.tensor([[1.0, 2.0]])
    second_gradient = torch.tensor([[1.0, -2.0]])
    projection_size = torch.randn(1, generator=torch.Generator().manual_seed(17)).abs()
    parameter.grad = first_gradient.clone()
    optimizer.step()
    parameter.grad = second_gradient.clone()
    optimizer.step()
    expected = -0.1 * (
        first_gradient * (2 / 5) ** 0.5 + second_gradient * 2**0.5 / 3
    ) / projection_size
    torch.testing.assert_close(parameter, expected)

    parameter = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = apollo_class([parameter], lr=0.1, weight_decay=0.1, seed=3)
    parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    torch.testing.assert_close(parameter, torch.full_like(parameter, 0.99))

    parameter = torch.nn.Parameter(torch.ones(64, 64))
    optimizer = apollo_class([parameter], lr=0.01, seed=11)
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    assert optimizer_state_bytes([optimizer]) < parameter.numel() * parameter.element_size()

    parameter = torch.nn.Parameter(torch.zeros(2, 2))
    optimizer = apollo_class([parameter], lr=0.1, betas=(0.0, 0.0),
                             weight_decay=0.0, scale=1.0, seed=7)
    projection = torch.randn(2, generator=torch.Generator().manual_seed(7))
    perpendicular = torch.stack((-projection[1], projection[0]))
    row = torch.tensor([1.0, 0.0])
    parameter.grad = torch.outer(projection, row)
    optimizer.step()
    first_norm = parameter.detach().norm()
    before = parameter.detach().clone()
    parameter.grad = torch.outer(perpendicular + 0.01 * projection, row)
    optimizer.step()
    assert (parameter.detach() - before).norm() <= first_norm * 1.011
