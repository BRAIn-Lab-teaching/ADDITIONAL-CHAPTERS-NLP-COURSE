"""Оформление графиков; все числа приходят из логов и студенческих функций."""

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import numpy as np


METHODS = {"adamw": "AdamW", "muon": "Muon", "adam_mini": "Adam-mini", "apollo": "APOLLO-Mini"}
COLORS = {"adamw": "#5276A7", "muon": "#D49338", "adam_mini": "#46856A", "apollo": "#B25B71"}
STYLE = {"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
         "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
         "axes.edgecolor": "#B2B8BF", "axes.labelcolor": "#3C4651",
         "xtick.color": "#59636D", "ytick.color": "#59636D"}


def _grid(axis, *, both=True):
    axis.set_axisbelow(True)
    axis.grid(axis="both" if both else "y", color="#DCE1E6", linewidth=0.7, alpha=0.8)


def plot_quality(histories):
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
        for kind, history in histories.items():
            rows = [row for row in history if row["record_type"] == "eval"]
            tokens = np.array([row["target_tokens"] for row in rows]) / 1000
            times = np.array([row["train_wall_time_seconds"] for row in rows]) / 60
            perplexity = np.exp([row["eval_loss"] for row in rows])
            for axis, x in zip(axes, (tokens, times)):
                axis.plot(x, perplexity, color=COLORS[kind], label=METHODS[kind],
                          marker="o", markersize=5, linewidth=2, markeredgecolor="white")
        axes[0].set(title="Перплексия по числу таргет-токенов", xlabel="Таргет-токены, тыс.", ylabel="Перплексия на валидации")
        axes[1].set(title="Перплексия по времени обучения", xlabel="Время обучения, мин", ylabel="Перплексия на валидации")
        for axis in axes:
            _grid(axis)
            axis.yaxis.set_major_locator(MaxNLocator(6))
        axes[0].legend(frameon=False, fontsize=9)
        figure.suptitle("SmolLM2-135M · перплексия на фиксированном фрагменте WikiText-103", fontsize=15)
        return figure


def plot_memory(comparison, snapshots):
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(1, 3, figsize=(17, 5.5), layout="constrained")
        kinds = list(comparison.index)
        labels = [METHODS[kind] for kind in kinds]
        x = np.arange(len(kinds))
        components = [("parameters", "Веса и buffers", "#526D8A"),
                      ("gradients", "Градиенты", "#94ABBE"),
                      ("optimizer", "Optimizer state", "#C79548"),
                      ("saved_for_backward", "Для backward", "#6D9A86"),
                      ("inputs", "Входной батч", "#B5C8BE"),
                      ("other", "Прочее", "#C4BBCB")]
        if snapshots:
            bottom = np.zeros(len(kinds))
            for key, label, color in components:
                values = np.array([snapshots[kind]["components_bytes"][key] for kind in kinds]) / 2**30
                bars = axes[0].bar(x, values, bottom=bottom, color=color, label=label, width=0.62, edgecolor="white", linewidth=0.5)
                for bar, value in zip(bars, values):
                    if value > 0.15:
                        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_y()+value/2,
                                     f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
                bottom += values
            for index, total in enumerate(bottom):
                axes[0].text(index, total + 0.07, f"{total:.2f}", ha="center")
            axes[0].set_ylim(0, bottom.max() * 1.16)
            axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=8)
        else:
            axes[0].text(0.5, 0.5, "Снимок памяти ещё не снят", transform=axes[0].transAxes, ha="center")
        axes[0].set(title="CUDA-память в конце forward", ylabel="CUDA allocated, ГиБ", xticks=x, xticklabels=labels)
        for index, column in enumerate(("пик CUDA, ГиБ", "состояние, ГиБ")):
            values = comparison[column].to_numpy()
            bars = axes[1].bar(x + (index - 0.5) * 0.32, values, width=0.29,
                               color=("#6A819B", "#C79548")[index], label=("Пик обучения", "Суммарный optimizer state")[index])
            axes[1].bar_label(bars, fmt="%.2f", padding=4, fontsize=8)
        axes[1].set(title="Пик CUDA и память состояния", ylabel="ГиБ", xticks=x, xticklabels=labels)
        axes[1].set_ylim(0, comparison["пик CUDA, ГиБ"].max() * 1.18)
        axes[1].legend(frameon=False, fontsize=9)
        bars = axes[2].barh(labels, comparison["шаг, с"], color=[COLORS[kind] for kind in kinds], height=0.55)
        axes[2].bar_label(bars, fmt="%.3f с", padding=6, fontsize=9)
        axes[2].invert_yaxis()
        axes[2].set(title="Время шага после разогрева GPU", xlabel="Медиана, с", xlim=(0, comparison["шаг, с"].max() * 1.28))
        for axis in axes[:2]:
            _grid(axis, both=False)
        _grid(axes[2])
        figure.suptitle("Память и время шага", fontsize=15)
        return figure


def plot_newton_schulz(table, spectra, *, step):
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
        a, b, c, d = axes.ravel()
        x = table["итераций NS"]
        for axis, column, color in ((a, "относительное расстояние до P", "#5276A7"), (b, "cosine с P", "#46856A")):
            axis.plot(x, table[column], marker="o", color=color, linewidth=2)
            for iteration, value in zip(x, table[column]):
                axis.annotate(f"{value:.3f}", (iteration, value), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
            axis.set(xlabel="Число итераций Newton–Schulz", ylabel=column, xticks=x)
            axis.margins(y=0.2)
        a.set(title="Относительное расстояние до полярного фактора", ylabel="Относительное расстояние (норма Фробениуса)")
        b.set(title="Cosine с полярным фактором", ylabel="Cosine")
        b.axhline(1, color="#929AA3", linestyle="--", linewidth=1)
        colors = plt.get_cmap("viridis")
        ns_labels = [label for label in spectra if label.startswith("NS-")]
        for index, label in enumerate(ns_labels):
            values = spectra[label]
            matrix_label = f"NS: {label[3:]} ит."
            color = colors(0.15 + 0.7 * index / max(1, len(ns_labels)-1))
            ranks = np.arange(1, len(values)+1)
            c.plot(ranks, values, color=color, label=matrix_label, linewidth=1.5)
            d.plot(ranks, values / values[0], color=color, label=matrix_label, linewidth=1.5)
        for axis in (c, d):
            axis.axhline(1, color="#636C75", linestyle="--", label="Точный полярный фактор (SVD)", linewidth=1.2)
            axis.set(xlabel="Номер сингулярного значения", ylabel="Сингулярное значение")
        d.plot(np.arange(1, len(spectra["H"])+1), spectra["H"] / spectra["H"][0], color="#AAB0B7", linestyle=":", label="До Newton–Schulz")
        d.axhline(0.1, color="#B88448", linestyle=":", linewidth=1)
        c.set(title="Спектр update после Newton–Schulz", ylabel=r"$\sigma_i$ (update)")
        d.set(title="Спектры до и после Newton–Schulz", ylabel=r"$\sigma_i\,/\,\sigma_1$", yscale="log", ylim=(1e-4, 1.4))
        c.legend(frameon=False, ncol=3, fontsize=8)
        d.legend(frameon=False, fontsize=8)
        for axis in axes.ravel():
            _grid(axis)
        figure.suptitle(f"Muon · q_proj слоя 14 · шаг {step}", fontsize=14)
        return figure


def plot_geometry(table):
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
        for method in ("Muon", "AdamW"):
            group = table[table["method"] == method].sort_values("step")
            color = COLORS["muon" if method == "Muon" else "adamw"]
            for axis, column in zip(axes[0], ("stable rank", "r01")):
                axis.plot(group["step"], group[column], marker="o", markersize=4, label=method, color=color, linewidth=2)
                axis.annotate(f"{group.iloc[-1][column]:.1f}", (group.iloc[-1]["step"], group.iloc[-1][column]), xytext=(5, 5), textcoords="offset points", color=color)
        axes[0, 0].set(title="Stable rank update", xlabel="Шаг обучения", ylabel="srank(update)")
        axes[0, 1].set(title="r₀.₁ update", xlabel="Шаг обучения", ylabel="r₀.₁(update)")
        for axis in axes[0]:
            _grid(axis)
            axis.margins(x=0.1)
        axes[0, 0].legend(frameon=False)
        for axis, method in zip(axes[1], ("Muon", "AdamW")):
            group = table[table["method"] == method].sort_values("step")
            spectra = np.stack(group["spectrum"])
            image = axis.imshow(spectra, aspect="auto", interpolation="nearest", cmap="viridis", norm=LogNorm(vmin=1e-4, vmax=1), extent=(0.5, spectra.shape[1]+0.5, len(group)-0.5, -0.5))
            axis.set(title=f"{method}: нормированный спектр update", xlabel="Номер сингулярного значения", ylabel="Шаг обучения", yticks=np.arange(len(group)), yticklabels=group["step"].astype(str))
            figure.colorbar(image, ax=axis, label="σᵢ(update) / σ₁(update)", shrink=0.85)
        figure.suptitle("Update для q_proj слоя 14 · без LR и weight decay", fontsize=15)
        return figure


def plot_checkpointing(table):
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
        for axis, column, title, unit in zip(axes,
                ("пик CUDA, ГиБ", "активации, ГиБ", "шаг, с"),
                ("Пик CUDA во время обучения", "Тензоры, сохранённые для backward", "Медиана времени шага"), ("ГиБ", "ГиБ", "с")):
            values = table[column].to_numpy()
            bars = axis.bar(["Без checkpointing", "С checkpointing"], values, color=["#8C9EAF", "#46856A"], width=0.55)
            axis.bar_label(bars, fmt="%.3f", padding=5)
            change = 100 * (values[1]/values[0]-1)
            text = f"{change:+.0f}%".replace("-", "−")
            axis.text(0.97, 0.94, text, transform=axis.transAxes, ha="right", va="top", fontsize=15, color="#46856A" if change < 0 else "#B88448")
            axis.set(title=title, ylabel=unit, ylim=(0, values.max() * 1.32))
            _grid(axis, both=False)
        figure.suptitle("Gradient checkpointing · APOLLO-Mini · память в обмен на вычисления", fontsize=14)
        return figure
