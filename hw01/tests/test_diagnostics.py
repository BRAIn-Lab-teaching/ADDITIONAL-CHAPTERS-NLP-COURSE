import json

import pytest
import torch

from diagnostics import build_artifacts, collect_yarn_metrics, update_report_tables


def test_diagnostics_produce_all_expected_artifacts(tmp_path) -> None:
    metrics = build_artifacts(tmp_path)

    for name in ('yarn_frequencies.png', 'metrics.json'):
        path = tmp_path / name
        assert path.is_file()
        assert path.stat().st_size > 500

    stored = json.loads((tmp_path / 'metrics.json').read_text(encoding='utf-8'))
    assert stored == metrics


def test_diagnostics_classify_yarn_coordinate_pairs(monkeypatch) -> None:
    original = torch.ones(6)
    ratio = torch.tensor([1.0, 1.0, 0.8, 0.4, 0.25, 0.25])
    monkeypatch.setattr(
        'diagnostics.yarn_scaled_inv_freq',
        lambda **_: (original, ratio * original),
    )

    metrics = collect_yarn_metrics()

    assert metrics['head_dim'] == 64
    assert metrics['factor'] == 4.0
    unchanged, transition, fully_scaled = metrics['regimes']
    assert unchanged == {
        'name': 'частоты практически не изменены',
        'coordinate_pair_indices': '0\N{EN DASH}1',
        'ratio_at_first_index': 1.0,
        'ratio_at_last_index': 1.0,
    }
    assert transition['name'] == 'переходная область'
    assert transition['coordinate_pair_indices'] == '2\N{EN DASH}3'
    assert transition['ratio_at_first_index'] == pytest.approx(0.8)
    assert transition['ratio_at_last_index'] == pytest.approx(0.4)
    assert fully_scaled == {
        'name': 'частоты масштабированы полностью',
        'coordinate_pair_indices': '4\N{EN DASH}5',
        'ratio_at_first_index': 0.25,
        'ratio_at_last_index': 0.25,
    }


def test_diagnostics_fill_report_tables_without_touching_student_text(tmp_path) -> None:
    report_path = tmp_path / 'report.md'
    report_path.write_text(
        """Введение студента
<!-- BEGIN GENERATED: YARN -->
старый YaRN
<!-- END GENERATED: YARN -->
Мой вывод <!-- FILL -->
<!-- BEGIN GENERATED: MLA -->
старый MLA
<!-- END GENERATED: MLA -->
Мой вывод MLA
<!-- BEGIN GENERATED: MOE -->
старый MoE
<!-- END GENERATED: MOE -->
Заключение студента
""",
        encoding='utf-8',
    )

    metrics = {
        'yarn': {
            'regimes': [
                {
                    'name': 'частоты практически не изменены',
                    'coordinate_pair_indices': '0\N{EN DASH}8',
                    'ratio_at_first_index': 1.0,
                    'ratio_at_last_index': 1.0,
                },
                {
                    'name': 'переходная область',
                    'coordinate_pair_indices': '9\N{EN DASH}20',
                    'ratio_at_first_index': 0.942308,
                    'ratio_at_last_index': 0.307692,
                },
            ]
        },
        'mla': {
            'naive_absorbed_max_error': 1.25e-7,
            'naive_absorbed_max_gradient_error': 2.5e-6,
            'mla_cache_bytes': 1152,
            'comparable_gqa_cache_bytes': 3072,
            'cache_compression_ratio': 8 / 3,
        },
        'moe': {
            factor: {
                'capacity': capacity,
                'dropped_assignment_fraction': dropped_assignments,
                'dropped_token_fraction': dropped_tokens,
                'load_cv': 0.125,
                'aux_loss': 1.03125,
                'total_parameters': 37120,
                'active_parameters_per_token': 9472,
            }
            for factor, capacity, dropped_assignments, dropped_tokens in (
                ('0.5', 8, 0.5, 0.359375),
                ('1.0', 16, 0.125, 0.015625),
                ('2.0', 32, 0.0, 0.0),
            )
        },
    }

    update_report_tables(report_path, metrics)
    first_run = report_path.read_text(encoding='utf-8')

    assert 'Введение студента' in first_run
    assert 'Мой вывод <!-- FILL -->' in first_run
    assert 'Заключение студента' in first_run
    assert 'старый YaRN' not in first_run
    assert 'старый MLA' not in first_run
    assert 'старый MoE' not in first_run
    assert '| частоты практически не изменены | 0\N{EN DASH}8 | 1.000 |' in first_run
    assert '| переходная область | 9\N{EN DASH}20 | 0.942 → 0.308 |' in first_run
    assert '| MLA-cache, байт | 1152 |' in first_run
    assert '| 0.5 | 8 | 0.5000 | 0.3594 |' in first_run
    assert '| всего | 37120 |' in first_run

    update_report_tables(report_path, metrics)
    assert report_path.read_text(encoding='utf-8') == first_run


def test_diagnostics_report_meaningful_architecture_metrics(tmp_path) -> None:
    metrics = build_artifacts(tmp_path)

    assert metrics['mla']['naive_absorbed_max_error'] < 1e-5
    assert metrics['mla']['naive_absorbed_max_gradient_error'] < 1e-4
    assert metrics['mla']['cache_compression_ratio'] > 1
    for run in metrics['moe'].values():
        assert run['total_parameters'] > run['active_parameters_per_token']
        assert len(run['expert_counts_before_capacity']) == 8
        assert 0 <= run['dropped_assignment_fraction'] <= 1
    assert (
        metrics['moe']['0.5']['dropped_assignment_fraction']
        >= metrics['moe']['2.0']['dropped_assignment_fraction']
    )
