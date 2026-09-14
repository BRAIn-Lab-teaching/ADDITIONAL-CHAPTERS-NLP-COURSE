"""Build the required HW01 figures and numerical diagnostics on CPU."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib
import torch

matplotlib.use('Agg')
from matplotlib import pyplot as plt

from course_lm.mla import (
    MLAConfig,
    MultiHeadLatentAttention,
    mla_cache_num_bytes,
)
from course_lm.moe import MoEConfig, SparseMoE, moe_parameter_counts
from course_lm.yarn import yarn_scaled_inv_freq


YARN_HEAD_DIM = 64
YARN_FACTOR = 4.0
YARN_ORIGINAL_CONTEXT = 2_048


def plot_yarn(output_path: Path) -> None:
    original, scaled = yarn_scaled_inv_freq(
        head_dim=YARN_HEAD_DIM,
        theta=10_000.0,
        factor=YARN_FACTOR,
        original_max_position_embeddings=YARN_ORIGINAL_CONTEXT,
    )
    coordinate_pair_index = torch.arange(original.numel()).cpu().numpy()
    original_values = original.detach().cpu().numpy()
    scaled_values = scaled.detach().cpu().numpy()
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].semilogy(coordinate_pair_index, original_values, marker='o', label='RoPE')
    axes[0].semilogy(coordinate_pair_index, scaled_values, marker='o', label='YaRN')
    axes[0].set(
        xlabel=r'индекс пары координат $i$: $(x_{2i}, x_{2i+1})$',
        ylabel='обратная частота',
        title='Частоты RoPE',
    )
    axes[0].legend()
    axes[1].plot(coordinate_pair_index, scaled_values / original_values, marker='o')
    axes[1].axhline(1 / YARN_FACTOR, color='tab:gray', linestyle='--', linewidth=1)
    axes[1].set(
        xlabel=r'индекс пары координат $i$: $(x_{2i}, x_{2i+1})$',
        ylabel=r'$\omega_i^{\mathrm{YaRN}} / \omega_i$',
        title='Профиль интерполяции',
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def collect_yarn_metrics() -> dict[str, object]:
    """Summarize the three frequency regimes shown in the YaRN plot."""
    original, scaled = yarn_scaled_inv_freq(
        head_dim=YARN_HEAD_DIM,
        theta=10_000.0,
        factor=YARN_FACTOR,
        original_max_position_embeddings=YARN_ORIGINAL_CONTEXT,
    )
    ratio = scaled / original
    unchanged = torch.isclose(ratio, torch.ones_like(ratio))
    fully_scaled = torch.isclose(ratio, torch.full_like(ratio, 1 / YARN_FACTOR))
    transition = ~(unchanged | fully_scaled)

    def summarize(name: str, mask: torch.Tensor) -> dict[str, str | float]:
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        values = ratio[indices]
        return {
            'name': name,
            'coordinate_pair_indices': (
                f'{indices[0].item()}\N{EN DASH}{indices[-1].item()}'
            ),
            'ratio_at_first_index': round(values[0].item(), 6),
            'ratio_at_last_index': round(values[-1].item(), 6),
        }

    return {
        'head_dim': YARN_HEAD_DIM,
        'factor': YARN_FACTOR,
        'regimes': [
            summarize('частоты практически не изменены', unchanged),
            summarize('переходная область', transition),
            summarize('частоты масштабированы полностью', fully_scaled),
        ],
    }


def collect_mla_metrics() -> dict[str, float | int]:
    config = MLAConfig(
        hidden_size=32,
        num_attention_heads=4,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
    )
    module = MultiHeadLatentAttention(config).eval()
    hidden_states = torch.randn(2, 12, config.hidden_size)
    with torch.no_grad():
        naive, _ = module(hidden_states, implementation='naive')
        absorbed, cache = module(
            hidden_states,
            implementation='absorbed',
            use_cache=True,
        )
    assert cache is not None

    naive_module = copy.deepcopy(module)
    absorbed_module = copy.deepcopy(module)
    naive_input = hidden_states.clone().requires_grad_(True)
    absorbed_input = hidden_states.clone().requires_grad_(True)
    naive_module(naive_input, implementation='naive')[0].square().sum().backward()
    absorbed_module(absorbed_input, implementation='absorbed')[
        0
    ].square().sum().backward()
    assert naive_input.grad is not None and absorbed_input.grad is not None
    gradient_errors = [(naive_input.grad - absorbed_input.grad).abs().max().item()]
    for naive_parameter, absorbed_parameter in zip(
        naive_module.parameters(),
        absorbed_module.parameters(),
        strict=True,
    ):
        assert naive_parameter.grad is not None and absorbed_parameter.grad is not None
        gradient_errors.append(
            (naive_parameter.grad - absorbed_parameter.grad).abs().max().item()
        )

    element_size = hidden_states.element_size()
    comparable_gqa_num_kv_heads = 2
    comparable_gqa_cache = (
        2
        * hidden_states.shape[0]
        * hidden_states.shape[1]
        * comparable_gqa_num_kv_heads
        * (config.qk_nope_head_dim + config.qk_rope_head_dim)
        * element_size
    )
    return {
        'naive_absorbed_max_error': (naive - absorbed).abs().max().item(),
        'naive_absorbed_max_gradient_error': max(gradient_errors),
        'mla_cache_bytes': mla_cache_num_bytes(cache),
        'comparable_gqa_cache_bytes': comparable_gqa_cache,
        'cache_compression_ratio': comparable_gqa_cache / mla_cache_num_bytes(cache),
    }


@torch.no_grad()
def collect_moe_metrics(
    capacity_factor: float,
) -> dict[str, float | int | list[int]]:
    torch.manual_seed(23)
    config = MoEConfig(
        hidden_size=32,
        expert_hidden_size=48,
        num_experts=8,
        top_k=2,
        capacity_factor=capacity_factor,
    )
    module = SparseMoE(config).eval()
    hidden_states = torch.randn(4, 16, config.hidden_size)
    _, routing = module(hidden_states)
    total_parameters, active_parameters = moe_parameter_counts(module)
    return {
        'total_parameters': total_parameters,
        'active_parameters_per_token': active_parameters,
        'capacity': routing.capacity,
        'expert_counts_before_capacity': routing.expert_counts_before_capacity.tolist(),
        'expert_counts_after_capacity': routing.expert_counts_after_capacity.tolist(),
        'router_entropy': routing.router_entropy.item(),
        'load_cv': routing.load_cv.item(),
        'dropped_assignment_fraction': routing.dropped_assignment_fraction.item(),
        'dropped_token_fraction': routing.dropped_token_fraction.item(),
        'aux_loss': routing.aux_loss.item(),
    }


def _generated_tables(metrics: dict[str, object]) -> dict[str, str]:
    yarn = metrics['yarn']
    mla = metrics['mla']
    moe = metrics['moe']
    assert isinstance(yarn, dict) and isinstance(mla, dict) and isinstance(moe, dict)

    yarn_rows = []
    for regime in yarn['regimes']:
        first = regime['ratio_at_first_index']
        last = regime['ratio_at_last_index']
        ratio = f'{first:.3f}' if first == last else f'{first:.3f} → {last:.3f}'
        yarn_rows.append(
            f'| {regime["name"]} | {regime["coordinate_pair_indices"]} | {ratio} |'
        )
    yarn_table = '\n'.join(
        [
            '| режим | индексы пар координат $i$ | '
            '$\\omega_i^{\\mathrm{YaRN}}/\\omega_i$ при росте $i$ |',
            '|---|---:|---:|',
            *yarn_rows,
        ]
    )

    mla_table = '\n'.join(
        [
            '| величина | значение |',
            '|---|---:|',
            f'| максимальная ошибка outputs | {mla["naive_absorbed_max_error"]:.6g} |',
            '| максимальная ошибка градиентов | '
            f'{mla["naive_absorbed_max_gradient_error"]:.6g} |',
            f'| MLA-cache, байт | {mla["mla_cache_bytes"]} |',
            f'| сопоставимый GQA-cache, байт | {mla["comparable_gqa_cache_bytes"]} |',
            f'| отношение GQA / MLA | {mla["cache_compression_ratio"]:.4f} |',
        ]
    )

    moe_rows = []
    for factor in ('0.5', '1.0', '2.0'):
        run = moe[factor]
        moe_rows.append(
            f'| {factor} | {run["capacity"]} | '
            f'{run["dropped_assignment_fraction"]:.4f} | '
            f'{run["dropped_token_fraction"]:.4f} | {run["load_cv"]:.4f} | '
            f'{run["aux_loss"]:.4f} |'
        )
    parameter_run = moe['1.0']
    moe_table = '\n'.join(
        [
            '| capacity factor | capacity | dropped assignments | dropped tokens | '
            'load CV | auxiliary loss |',
            '|---:|---:|---:|---:|---:|---:|',
            *moe_rows,
            '',
            '| параметры модели | число |',
            '|---|---:|',
            f'| всего | {parameter_run["total_parameters"]} |',
            '| не более чем активно для одного токена | '
            f'{parameter_run["active_parameters_per_token"]} |',
        ]
    )
    return {'YARN': yarn_table, 'MLA': mla_table, 'MOE': moe_table}


def update_report_tables(report_path: Path, metrics: dict[str, object]) -> None:
    """Replace generated table blocks while preserving student-written text."""
    report = report_path.read_text(encoding='utf-8')
    for name, table in _generated_tables(metrics).items():
        begin = f'<!-- BEGIN GENERATED: {name} -->'
        end = f'<!-- END GENERATED: {name} -->'
        if report.count(begin) != 1 or report.count(end) != 1:
            raise ValueError(f'report.md must contain exactly one {name} table block')
        prefix, remainder = report.split(begin)
        _, suffix = remainder.split(end)
        report = f'{prefix}{begin}\n{table}\n{end}{suffix}'
    report_path.write_text(report, encoding='utf-8')


def build_artifacts(
    output_dir: Path, report_path: Path | None = None
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(17)
    plot_yarn(output_dir / 'yarn_frequencies.png')
    metrics = {
        'yarn': collect_yarn_metrics(),
        'mla': collect_mla_metrics(),
        'moe': {str(factor): collect_moe_metrics(factor) for factor in (0.5, 1.0, 2.0)},
    }
    (output_dir / 'metrics.json').write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    if report_path is not None:
        update_report_tables(report_path, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts'))
    parser.add_argument(
        '--report',
        type=Path,
        default=Path(__file__).resolve().parent / 'report.md',
    )
    args = parser.parse_args()
    build_artifacts(args.output_dir, report_path=args.report)
    print(f'Artifacts written to {args.output_dir}; tables updated in {args.report}')


if __name__ == '__main__':
    main()
