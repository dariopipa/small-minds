from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

STRATEGY_ORDER = ("direct", "self_consistency", "society_of_minds", "role_based_svj")
STRATEGY_COLORS = {
    "direct": "#4D4D4D",
    "self_consistency": "#0072B2",
    "society_of_minds": "#009E73",
    "role_based_svj": "#D55E00",
}
STRATEGY_SHORT_LABELS = {
    "direct": "Direct",
    "self_consistency": "SC",
    "society_of_minds": "SoM",
    "role_based_svj": "SVJ",
}
BENCHMARK_ORDER = ("gsm8k", "arc_challenge_chat", "boolq")
BENCHMARK_LABELS = {
    "gsm8k": "GSM8K",
    "arc_challenge_chat": "ARC-Challenge",
    "boolq": "BoolQ",
}
BENCHMARK_COLORS = {
    "gsm8k": "#0072B2",
    "arc_challenge_chat": "#009E73",
    "boolq": "#D55E00",
}


@dataclass(frozen=True, slots=True)
class StrategyPlotData:
    model: str
    benchmark: str
    strategy: str
    label: str
    mean: float | None
    gain_vs_direct: float | None
    tokens_per_question: float | None


@dataclass(frozen=True, slots=True)
class FigureArtifact:
    stem: str
    title: str
    alt_text: str
    caption: str


def _ordered(values: set[str], preferred: tuple[str, ...]) -> list[str]:
    return [value for value in preferred if value in values] + sorted(
        values - set(preferred)
    )


def _models(values: set[str]) -> list[str]:
    return sorted(
        values,
        key=lambda name: (
            0 if "qwen" in name.lower() else 1 if "llama" in name.lower() else 2,
            name,
        ),
    )


def _model_label(name: str) -> str:
    normalized = name.lower()
    if "qwen2.5" in normalized and "3b" in normalized:
        return "Qwen2.5-3B"
    if "llama3.2" in normalized and "3b" in normalized:
        return "Llama 3.2 3B"
    return name


def _benchmark_label(name: str) -> str:
    return BENCHMARK_LABELS.get(name, name.replace("_", " ").title())


def _color(name: str, index: int, colors: dict[str, str]) -> str:
    fallback = ("#56B4E9", "#CC79A7", "#E69F00", "#000000")
    return colors.get(name, fallback[index % len(fallback)])


def _valid(value: float | None) -> TypeGuard[float]:
    return value is not None and math.isfinite(value)


def _style_axis(axis: plt.Axes) -> None:
    axis.set_axisbelow(True)
    axis.xaxis.grid(False)
    axis.yaxis.grid(color="#D7DCE2", linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _save(figure: Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{stem}.svg", format="svg", facecolor="white")
    figure.savefig(output_dir / f"{stem}.pdf", format="pdf", facecolor="white")
    plt.close(figure)


def pareto_strategies(data: list[StrategyPlotData]) -> set[tuple[str, str, str]]:
    """Return nondominated identities within each observed model/benchmark panel.

    Another strategy dominates a point when it is at least as accurate and uses
    no more tokens, with at least one strict inequality. Missing measurements
    cannot establish either dominance or membership of the observed frontier.
    """
    points = [
        item
        for item in data
        if _valid(item.mean)
        and _valid(item.tokens_per_question)
        and item.tokens_per_question >= 0
    ]
    frontier = set()
    for item in points:
        assert item.mean is not None and item.tokens_per_question is not None
        dominated = any(
            other.model == item.model
            and other.benchmark == item.benchmark
            and other.mean is not None
            and other.tokens_per_question is not None
            and other.mean >= item.mean
            and other.tokens_per_question <= item.tokens_per_question
            and (
                other.mean > item.mean
                or other.tokens_per_question < item.tokens_per_question
            )
            for other in points
        )
        if not dominated:
            frontier.add((item.model, item.benchmark, item.strategy))
    return frontier


def _plot_gain(output_dir: Path, data: list[StrategyPlotData]) -> FigureArtifact:
    strategies = _ordered(
        {item.strategy for item in data if item.strategy != "direct"}, STRATEGY_ORDER
    )
    models = _models({item.model for item in data})
    benchmarks = _ordered({item.benchmark for item in data}, BENCHMARK_ORDER)
    lookup = {(item.model, item.benchmark, item.strategy): item for item in data}
    values = [
        item.gain_vs_direct * 100
        for item in data
        if item.strategy != "direct" and _valid(item.gain_vs_direct)
    ]
    low, high = min(0.0, *values), max(0.0, *values)
    margin = max(0.8, (high - low) * 0.18)
    figure, axes = plt.subplots(
        len(models),
        1,
        figsize=(8.8, 3.0 * len(models)),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    width = 0.76 / len(benchmarks)
    for model, (axis,) in zip(models, axes, strict=True):
        for strategy_index, strategy in enumerate(strategies):
            for benchmark_index, benchmark in enumerate(benchmarks):
                item = lookup.get((model, benchmark, strategy))
                if item is None or not _valid(item.gain_vs_direct):
                    continue
                value = item.gain_vs_direct * 100
                x = (
                    strategy_index
                    + (benchmark_index - (len(benchmarks) - 1) / 2) * width
                )
                axis.bar(
                    x,
                    value,
                    width=width * 0.92,
                    color=_color(benchmark, benchmark_index, BENCHMARK_COLORS),
                    zorder=2,
                )
                axis.annotate(
                    f"{value:+.2f}",
                    (x, value),
                    xytext=(0, 4 if value >= 0 else -4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=8,
                )
        axis.axhline(0, color="#202936", linewidth=0.9)
        axis.set_title(_model_label(model), fontweight="bold")
        axis.set_ylabel("Difference from Direct\n(percentage points)")
        axis.set_ylim(low - margin, high + margin)
        axis.set_xticks(
            range(len(strategies)),
            [
                next(item.label for item in data if item.strategy == strategy)
                for strategy in strategies
            ],
        )
        _style_axis(axis)
    figure.legend(
        handles=[
            Patch(
                facecolor=_color(benchmark, index, BENCHMARK_COLORS),
                label=_benchmark_label(benchmark),
            )
            for index, benchmark in enumerate(benchmarks)
        ],
        loc="outside lower center",
        ncols=len(benchmarks),
        frameon=False,
    )
    stem = "gain_vs_direct"
    _save(figure, output_dir, stem)
    return FigureArtifact(
        stem=stem,
        title="Mean accuracy difference from Direct",
        alt_text="Grouped bars show each strategy's mean accuracy difference from Direct across benchmarks and models.",
        caption="Mean accuracy difference from Direct, in percentage points.",
    )


def _plot_tokens(
    output_dir: Path, data: list[StrategyPlotData]
) -> FigureArtifact | None:
    points = [
        item
        for item in data
        if _valid(item.mean)
        and _valid(item.tokens_per_question)
        and item.tokens_per_question >= 0
    ]
    if not points:
        return None
    models = _models({item.model for item in points})
    benchmarks = _ordered({item.benchmark for item in points}, BENCHMARK_ORDER)
    strategies = _ordered({item.strategy for item in points}, STRATEGY_ORDER)
    frontier = pareto_strategies(points)
    figure, axes = plt.subplots(
        len(benchmarks),
        len(models),
        figsize=(4.6 * len(models), 2.8 * len(benchmarks)),
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    scores = [item.mean * 100 for item in points if item.mean is not None]
    span = max(10.0, max(scores) - min(scores))
    limits = (
        max(0.0, min(scores) - span * 0.15),
        min(100.0, max(scores) + span * 0.15),
    )
    for benchmark_index, benchmark in enumerate(benchmarks):
        token_max = max(
            item.tokens_per_question
            for item in points
            if item.benchmark == benchmark and item.tokens_per_question is not None
        )
        for model_index, model in enumerate(models):
            axis = axes[benchmark_index][model_index]
            panel_points = [
                item
                for item in points
                if item.model == model and item.benchmark == benchmark
            ]
            if not panel_points:
                axis.set_visible(False)
                continue
            for item in panel_points:
                assert item.mean is not None and item.tokens_per_question is not None
                nondominated = (model, benchmark, item.strategy) in frontier
                alpha = 1.0 if nondominated else 0.28
                axis.scatter(
                    item.tokens_per_question,
                    item.mean * 100,
                    s=66,
                    color=_color(
                        item.strategy, strategies.index(item.strategy), STRATEGY_COLORS
                    ),
                    edgecolor="white",
                    linewidth=0.7,
                    alpha=alpha,
                    zorder=3,
                )
                offset = (5, -13) if item.strategy == "society_of_minds" else (5, 6)
                nearby = [
                    (abs(other.mean - item.mean), other.mean)
                    for other in panel_points
                    if other is not item
                    and other.mean is not None
                    and other.tokens_per_question is not None
                    and abs(other.tokens_per_question - item.tokens_per_question)
                    < token_max * 0.2
                    and abs(other.mean - item.mean) * 100 < span * 0.2
                ]
                if nearby:
                    closest_mean = min(nearby)[1]
                    offset = (5, 6) if item.mean >= closest_mean else (5, -13)
                axis.annotate(
                    STRATEGY_SHORT_LABELS.get(item.strategy, item.label),
                    (item.tokens_per_question, item.mean * 100),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=8,
                    color="#202936" if nondominated else "#6C7480",
                )
            axis.set_xlim(-max(1.0, token_max * 0.05), max(1.0, token_max * 1.17))
            axis.set_ylim(*limits)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            axis.set_title(
                f"{_model_label(model)}\n{_benchmark_label(benchmark)}",
                fontweight="bold",
            )
            _style_axis(axis)
    figure.supylabel("Mean exact-match accuracy (%)")
    figure.supxlabel("Prompt and completion tokens per question")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=7,
            color=_color(strategy, index, STRATEGY_COLORS),
            label=next(item.label for item in points if item.strategy == strategy),
        )
        for index, strategy in enumerate(strategies)
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                color="#202936",
                label="Pareto frontier",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                color="#202936",
                alpha=0.28,
                label="Dominated",
            ),
        ]
    )
    figure.legend(
        handles=handles,
        loc="outside upper center",
        ncols=3 if len(models) > 1 else 2,
        frameon=False,
    )
    stem = "accuracy_vs_tokens"
    _save(figure, output_dir, stem)
    return FigureArtifact(
        stem=stem,
        title="Accuracy and token usage",
        alt_text="Accuracy versus prompt and completion tokens, with solid Pareto frontier points and faded dominated points in each model and benchmark panel.",
        caption=(
            "Faded points are dominated: another strategy achieves at least the same "
            "accuracy with no more tokens, improving at least one measure."
        ),
    )


def _plot_mcnemar(
    output_dir: Path, statistical_comparison: dict[str, Any]
) -> FigureArtifact | None:
    tests = statistical_comparison.get("tests", [])
    skipped = statistical_comparison.get("skipped", [])
    if not tests:
        return None
    comparisons = tests + skipped
    models = _models({item["model"] for item in comparisons})
    benchmarks = _ordered({item["benchmark"] for item in comparisons}, BENCHMARK_ORDER)
    strategies = _ordered({item["strategy"] for item in comparisons}, STRATEGY_ORDER)
    repetitions = sorted({item["repetition"] for item in comparisons})
    rows = [
        (strategy, repetition) for strategy in strategies for repetition in repetitions
    ]
    lookup = {
        (item["model"], item["benchmark"], item["strategy"], item["repetition"]): item
        for item in tests
    }
    figure, axes = plt.subplots(
        len(benchmarks),
        len(models),
        figsize=(5.1 * len(models), max(2.8, len(rows) * 0.27 + 0.8) * len(benchmarks)),
        squeeze=False,
        sharex="row",
        layout="constrained",
    )
    for benchmark_index, benchmark in enumerate(benchmarks):
        for model_index, model in enumerate(models):
            axis = axes[benchmark_index, model_index]
            axis.axvline(0, color="#626A73", linewidth=0.9, linestyle="--")
            for row_index, (strategy, repetition) in enumerate(rows):
                item = lookup.get((model, benchmark, strategy, repetition))
                if item is None:
                    axis.text(
                        0.98,
                        row_index,
                        "N/A",
                        ha="right",
                        va="center",
                        transform=axis.get_yaxis_transform(),
                    )
                    continue
                value = item["accuracy_difference_pp"]
                low, high = item["ci95_low_pp"], item["ci95_high_pp"]
                color = _color(strategy, strategies.index(strategy), STRATEGY_COLORS)
                # Draw endpoints directly: BCa intervals need not contain their estimate.
                axis.hlines(row_index, low, high, color=color, linewidth=1.6)
                axis.plot(
                    [low, high], [row_index, row_index], "|", color=color, markersize=5
                )
                axis.plot(value, row_index, "o", color=color, markersize=4)
            axis.set_yticks(
                range(len(rows)),
                [
                    f"{STRATEGY_SHORT_LABELS.get(strategy, strategy)} R{repetition}"
                    for strategy, repetition in rows
                ],
            )
            axis.set_ylim(len(rows) - 0.45, -0.55)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
            axis.xaxis.grid(color="#E4E7EB", linewidth=0.6)
            axis.set_axisbelow(True)
            axis.set_xlabel("Accuracy difference from Direct (pp)")
            axis.set_title(
                f"{_model_label(model)} - {_benchmark_label(benchmark)}",
                fontweight="bold",
            )
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_visible(False)
            axis.tick_params(axis="y", length=0)
            for boundary in range(len(repetitions), len(rows), len(repetitions)):
                axis.axhline(boundary - 0.5, color="#EEF0F3", linewidth=0.7)
            axis.margins(x=0.08)
    stem = "mcnemar_vs_direct"
    _save(figure, output_dir, stem)
    return FigureArtifact(
        stem=stem,
        title="Paired accuracy differences with 95% confidence intervals",
        alt_text="Accuracy differences from Direct for each model, benchmark, strategy and repetition, with horizontal 95% paired bootstrap confidence intervals and a zero reference line.",
        caption=(
            "Points show accuracy differences from Direct; horizontal bars show individual 95% paired BCa confidence intervals. "
            "R1-R3 are separate repetitions. Colors identify strategies. The dashed line marks no difference; "
            "intervals are not adjusted across comparisons."
        ),
    )


def generate_academic_figures(
    output_dir: Path,
    data: list[StrategyPlotData],
    statistical_comparison: dict[str, Any] | None = None,
) -> list[FigureArtifact]:
    """Generate figures from the existing analysis results."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    figures = []
    if any(item.strategy != "direct" and _valid(item.gain_vs_direct) for item in data):
        figures.append(_plot_gain(output_dir, data))
    if efficiency := _plot_tokens(output_dir, data):
        figures.append(efficiency)
    if statistical_comparison and (
        comparison := _plot_mcnemar(output_dir, statistical_comparison)
    ):
        figures.append(comparison)
    return figures
