"""Figures for saved experiment analyses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter

STRATEGY_ORDER = (
    "direct",
    "self_consistency",
    "society_of_minds",
    "role_based_svj",
)
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


@dataclass(frozen=True, slots=True)
class StrategyPlotData:
    model: str
    benchmark: str
    strategy: str
    label: str
    mean: float | None
    repetition_scores: tuple[float, ...]
    gain_vs_direct: float | None
    repetition_gains: tuple[float, ...]
    tokens_per_question: float | None
    end_to_end_latency_per_question: float | None


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


def _benchmark_label(name: str) -> str:
    return BENCHMARK_LABELS.get(name, name.replace("_", " ").title())


def _strategy_label(name: str, fallback: str) -> str:
    return STRATEGY_SHORT_LABELS.get(name, fallback)


def _strategy_color(name: str, index: int) -> str:
    fallback = ("#56B4E9", "#CC79A7", "#E69F00", "#000000")
    return STRATEGY_COLORS.get(name, fallback[index % len(fallback)])


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
    figure.savefig(output_dir / f"{stem}.png", format="png", dpi=300, facecolor="white")
    plt.close(figure)


def _panels(
    data: list[StrategyPlotData],
) -> tuple[Figure, list[tuple[plt.Axes, str, str]]]:
    models = sorted({item.model for item in data})
    benchmarks = _ordered({item.benchmark for item in data}, BENCHMARK_ORDER)
    figure, axes = plt.subplots(
        len(models),
        len(benchmarks),
        figsize=(max(5.4, 3.2 * len(benchmarks)), 3.5 * len(models)),
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    panels = []
    for model_index, model in enumerate(models):
        for benchmark_index, benchmark in enumerate(benchmarks):
            axis = axes[model_index][benchmark_index]
            if any(
                item.model == model and item.benchmark == benchmark for item in data
            ):
                panels.append((axis, model, benchmark))
            else:
                axis.set_visible(False)
    return figure, panels


def _panel_title(model: str, benchmark: str, multiple_models: bool) -> str:
    label = _benchmark_label(benchmark)
    return f"{model}\n{label}" if multiple_models else label


def _legend(
    figure: Figure, data: list[StrategyPlotData], strategies: list[str]
) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            markersize=7,
            linestyle="none",
            markerfacecolor=_strategy_color(strategy, index),
            markeredgecolor="none",
            label=next(
                (item.label for item in data if item.strategy == strategy), strategy
            ),
        )
        for index, strategy in enumerate(strategies)
    ]
    figure.legend(
        handles=handles,
        loc="outside lower center",
        frameon=False,
        ncols=len(handles),
        columnspacing=1.4,
    )


def _plot_accuracy(output_dir: Path, data: list[StrategyPlotData]) -> FigureArtifact:
    strategies = _ordered({item.strategy for item in data}, STRATEGY_ORDER)
    lookup = {(item.model, item.benchmark, item.strategy): item for item in data}
    figure, panels = _panels(data)
    multiple_models = len({item.model for item in data}) > 1

    for axis, model, benchmark in panels:
        items = [
            lookup[(model, benchmark, strategy)]
            for strategy in strategies
            if (model, benchmark, strategy) in lookup
        ]
        for index, item in enumerate(items):
            if item.mean is None:
                continue
            scores = item.repetition_scores or (item.mean,)
            color = _strategy_color(item.strategy, strategies.index(item.strategy))
            axis.bar(index, item.mean * 100, width=0.68, color=color, zorder=2)
            if len(scores) > 1:
                axis.vlines(
                    index,
                    min(scores) * 100,
                    max(scores) * 100,
                    color="#202936",
                    linewidth=1,
                    zorder=3,
                )
                offsets = [
                    0.11 * (position - (len(scores) - 1) / 2)
                    for position in range(len(scores))
                ]
                axis.scatter(
                    [index + offset for offset in offsets],
                    [score * 100 for score in scores],
                    s=14,
                    color="white",
                    edgecolor="#202936",
                    linewidth=0.5,
                    zorder=4,
                )
            axis.text(
                index,
                item.mean * 100 + 2,
                f"{item.mean * 100:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        axis.set_title(
            _panel_title(model, benchmark, multiple_models), fontweight="bold"
        )
        axis.set_xticks(
            range(len(items)),
            [_strategy_label(item.strategy, item.label) for item in items],
        )
        axis.set_ylim(0, 108)
        axis.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        _style_axis(axis)

    panels[0][0].set_ylabel("Exact-match accuracy (%)")
    _legend(figure, data, strategies)
    stem = "accuracy_by_strategy"
    _save(figure, output_dir, stem)
    return FigureArtifact(
        stem=stem,
        title="Accuracy by strategy",
        alt_text="Faceted bar chart comparing each strategy's exact-match accuracy by benchmark.",
        caption=(
            "Bars show mean exact-match accuracy. When an experiment has repeated runs, "
            "white points show individual repetitions and the vertical line shows their observed range."
        ),
    )


def _plot_gain(output_dir: Path, data: list[StrategyPlotData]) -> FigureArtifact:
    strategies = _ordered(
        {item.strategy for item in data if item.strategy != "direct"},
        tuple(strategy for strategy in STRATEGY_ORDER if strategy != "direct"),
    )
    lookup = {(item.model, item.benchmark, item.strategy): item for item in data}
    points = [
        gain * 100
        for item in data
        if item.strategy != "direct"
        for gain in (item.repetition_gains or (item.gain_vs_direct,))
        if gain is not None
    ]
    minimum, maximum = min(points, default=0.0), max(points, default=0.0)
    if minimum >= 0:
        limits = (-max(1.0, maximum * 0.15), max(2.0, maximum * 1.25))
    elif maximum <= 0:
        limits = (min(-2.0, minimum * 1.25), max(1.0, abs(minimum) * 0.15))
    else:
        limit = max(abs(minimum), abs(maximum)) * 1.25
        limits = (-limit, limit)
    figure, panels = _panels(data)
    multiple_models = len({item.model for item in data}) > 1

    for axis, model, benchmark in panels:
        items = [
            lookup[(model, benchmark, strategy)]
            for strategy in strategies
            if (model, benchmark, strategy) in lookup
            and lookup[(model, benchmark, strategy)].gain_vs_direct is not None
        ]
        for index, item in enumerate(items):
            assert item.gain_vs_direct is not None
            gains = item.repetition_gains or (item.gain_vs_direct,)
            value = item.gain_vs_direct * 100
            color = _strategy_color(item.strategy, strategies.index(item.strategy))
            axis.bar(index, value, width=0.68, color=color, zorder=2)
            if len(gains) > 1:
                axis.vlines(
                    index,
                    min(gains) * 100,
                    max(gains) * 100,
                    color="#202936",
                    linewidth=1,
                    zorder=3,
                )
                offsets = [
                    0.11 * (position - (len(gains) - 1) / 2)
                    for position in range(len(gains))
                ]
                axis.scatter(
                    [index + offset for offset in offsets],
                    [gain * 100 for gain in gains],
                    s=14,
                    color="white",
                    edgecolor="#202936",
                    linewidth=0.5,
                    zorder=4,
                )
            offset = 0.35 if value >= 0 else -0.35
            axis.text(
                index,
                value + offset,
                f"{value:+.1f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=7,
            )

        axis.axhline(0, color="#202936", linewidth=0.9)
        axis.set_title(
            _panel_title(model, benchmark, multiple_models), fontweight="bold"
        )
        axis.set_xticks(
            range(len(items)),
            [_strategy_label(item.strategy, item.label) for item in items],
        )
        axis.set_ylim(*limits)
        _style_axis(axis)

    panels[0][0].set_ylabel("Change from Direct (percentage points)")
    _legend(figure, data, strategies)
    stem = "gain_vs_direct"
    _save(figure, output_dir, stem)
    return FigureArtifact(
        stem=stem,
        title="Change from Direct",
        alt_text="Faceted bar chart showing each collaborative strategy's accuracy change from Direct.",
        caption=(
            "Bars show the mean paired accuracy difference from Direct. Positive values favour the strategy; "
            "white points and lines show individual repetitions and their observed range."
        ),
    )


def _plot_efficiency(
    output_dir: Path,
    data: list[StrategyPlotData],
    *,
    field: str,
    title: str,
    stem: str,
    axis_label: str,
    caption: str,
) -> FigureArtifact | None:
    points = [
        item
        for item in data
        if item.mean is not None
        and (value := getattr(item, field)) is not None
        and value > 0
    ]
    if not points:
        return None

    strategies = _ordered({item.strategy for item in points}, STRATEGY_ORDER)
    figure, panels = _panels(points)
    multiple_models = len({item.model for item in points}) > 1

    for axis, model, benchmark in panels:
        panel_points = [
            item
            for item in points
            if item.model == model and item.benchmark == benchmark
        ]
        for item in panel_points:
            value = getattr(item, field)
            assert item.mean is not None and value is not None
            axis.scatter(
                value,
                item.mean * 100,
                s=58,
                color=_strategy_color(item.strategy, strategies.index(item.strategy)),
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
        axis.set_xlabel(axis_label)
        axis.set_title(
            _panel_title(model, benchmark, multiple_models), fontweight="bold"
        )
        axis.set_ylim(0, 105)
        _style_axis(axis)

    panels[0][0].set_ylabel("Exact-match accuracy (%)")
    _legend(figure, points, strategies)
    _save(figure, output_dir, stem)
    return FigureArtifact(
        stem=stem,
        title=title,
        alt_text=f"Scatter plot comparing exact-match accuracy with {axis_label.lower()}.",
        caption=caption,
    )


def generate_academic_figures(
    output_dir: Path,
    data: list[StrategyPlotData],
) -> list[FigureArtifact]:
    """Generate the report figures from one experiment's grouped results."""

    if not data:
        return []
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
    figures = [_plot_accuracy(output_dir, data)]
    if any(item.gain_vs_direct is not None for item in data):
        figures.append(_plot_gain(output_dir, data))
    if efficiency := _plot_efficiency(
        output_dir,
        data,
        field="tokens_per_question",
        title="Accuracy versus token usage",
        stem="accuracy_vs_tokens",
        axis_label="Tokens per question",
        caption=(
            "Each point is one model, benchmark, and strategy. Token cost includes prompt and completion tokens; "
            "points are not connected because this is not a budget sweep."
        ),
    ):
        figures.append(efficiency)
    if latency := _plot_efficiency(
        output_dir,
        data,
        field="end_to_end_latency_per_question",
        title="Accuracy versus end-to-end latency",
        stem="accuracy_vs_end_to_end_latency",
        axis_label="End-to-end latency per question (s)",
        caption=(
            "Each point is one model, benchmark, and strategy. End-to-end latency is the practical time "
            "from starting a strategy for one question until its final answer is available."
        ),
    ):
        figures.append(latency)
    return figures
