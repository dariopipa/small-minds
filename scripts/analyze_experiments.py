from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT_DIR / "outputs" / "experiments"
DEFAULT_PRICING_PRESET = "openai-chatgpt-latest"


@dataclass(frozen=True)
class StrategyAnalysis:
    name: str
    base_strategy: str
    task: str
    repeat_index: int
    repetitions: int
    limit: int
    effective_samples: int
    strict_accuracy: float | None
    flexible_accuracy: float | None
    wall_time_s: float
    strategy_result_count: int
    agent_response_count: int
    prompt_tokens: int
    output_tokens: int
    total_latency_s: float
    estimated_cost_usd: float
    avg_calls_per_sample: float
    avg_tokens_per_sample: float
    avg_latency_per_sample_s: float
    avg_cost_per_sample_usd: float
    flexible_only_correct: int
    unanimous_final_votes: int
    split_final_votes: int
    extraction_failures: int


@dataclass(frozen=True)
class AggregateAnalysis:
    base_strategy: str
    runs: int
    samples_per_run: int
    strict_accuracy_mean: float | None
    strict_accuracy_std: float | None
    flexible_accuracy_mean: float | None
    flexible_accuracy_std: float | None
    avg_calls_per_sample_mean: float
    avg_tokens_per_sample_mean: float
    avg_latency_per_sample_s_mean: float
    avg_cost_per_sample_usd_mean: float
    estimated_cost_usd_total: float


@dataclass(frozen=True)
class PricingPreset:
    label: str
    provider: str
    model: str
    prompt_price_per_1m: float
    output_price_per_1m: float
    source_url: str
    note: str


PRICING_PRESETS = {
    "ollama-local": PricingPreset(
        label="ollama-local",
        provider="Ollama",
        model="local model",
        prompt_price_per_1m=0.0,
        output_price_per_1m=0.0,
        source_url="",
        note="Actual token bill for local Ollama is zero.",
    ),
    "openai-chatgpt-latest": PricingPreset(
        label="openai-chatgpt-latest",
        provider="OpenAI",
        model="ChatGPT chat-latest",
        prompt_price_per_1m=5.0,
        output_price_per_1m=30.0,
        source_url="https://developers.openai.com/api/docs/pricing",
        note="OpenAI specialized ChatGPT API pricing.",
    ),
    "openai-gpt-5-mini": PricingPreset(
        label="openai-gpt-5-mini",
        provider="OpenAI",
        model="GPT-5 mini",
        prompt_price_per_1m=0.25,
        output_price_per_1m=2.0,
        source_url="https://developers.openai.com/api/docs/models/gpt-5-mini",
        note="Affordable OpenAI text model preset.",
    ),
    "openai-gpt-5-nano": PricingPreset(
        label="openai-gpt-5-nano",
        provider="OpenAI",
        model="GPT-5 nano",
        prompt_price_per_1m=0.05,
        output_price_per_1m=0.40,
        source_url="https://openai.com/gpt-5/",
        note="Low-cost OpenAI text model preset.",
    ),
    "openai-gpt-4.1": PricingPreset(
        label="openai-gpt-4.1",
        provider="OpenAI",
        model="GPT-4.1",
        prompt_price_per_1m=2.0,
        output_price_per_1m=8.0,
        source_url="https://developers.openai.com/api/docs/models/gpt-4.1",
        note="OpenAI non-reasoning model preset.",
    ),
    "anthropic-claude-sonnet-5": PricingPreset(
        label="anthropic-claude-sonnet-5",
        provider="Anthropic",
        model="Claude Sonnet 5",
        prompt_price_per_1m=2.0,
        output_price_per_1m=10.0,
        source_url="https://claude.com/pricing",
        note="Introductory Sonnet 5 price through August 31, 2026.",
    ),
    "anthropic-claude-haiku-4.5": PricingPreset(
        label="anthropic-claude-haiku-4.5",
        provider="Anthropic",
        model="Claude Haiku 4.5",
        prompt_price_per_1m=1.0,
        output_price_per_1m=5.0,
        source_url="https://claude.com/pricing",
        note="Anthropic low-cost model preset.",
    ),
    "anthropic-claude-opus-5": PricingPreset(
        label="anthropic-claude-opus-5",
        provider="Anthropic",
        model="Claude Opus 5",
        prompt_price_per_1m=5.0,
        output_price_per_1m=25.0,
        source_url="https://claude.com/pricing",
        note="Anthropic high-capability model preset.",
    ),
    "google-gemini-2.5-flash-lite": PricingPreset(
        label="google-gemini-2.5-flash-lite",
        provider="Google",
        model="Gemini 2.5 Flash-Lite",
        prompt_price_per_1m=0.10,
        output_price_per_1m=0.40,
        source_url="https://ai.google.dev/gemini-api/docs/pricing",
        note="Google low-cost text/image/video token preset.",
    ),
    "google-gemini-2.5-flash": PricingPreset(
        label="google-gemini-2.5-flash",
        provider="Google",
        model="Gemini 2.5 Flash",
        prompt_price_per_1m=0.30,
        output_price_per_1m=2.50,
        source_url="https://ai.google.dev/gemini-api/docs/pricing",
        note="Google price-performance model preset.",
    ),
    "google-gemini-2.5-pro": PricingPreset(
        label="google-gemini-2.5-pro",
        provider="Google",
        model="Gemini 2.5 Pro",
        prompt_price_per_1m=1.25,
        output_price_per_1m=10.0,
        source_url="https://ai.google.dev/gemini-api/docs/pricing",
        note="Standard price for prompts up to 200k tokens.",
    ),
}


def resolve_pricing(
    pricing_preset: str,
    prompt_price_per_1m: float | None = None,
    output_price_per_1m: float | None = None,
) -> PricingPreset:
    if pricing_preset not in PRICING_PRESETS:
        valid_presets = ", ".join(sorted(PRICING_PRESETS))
        raise ValueError(
            f"Unknown pricing preset: {pricing_preset}. Valid presets: {valid_presets}"
        )

    preset = PRICING_PRESETS[pricing_preset]
    prompt_price = (
        preset.prompt_price_per_1m
        if prompt_price_per_1m is None
        else prompt_price_per_1m
    )
    output_price = (
        preset.output_price_per_1m
        if output_price_per_1m is None
        else output_price_per_1m
    )

    if (
        prompt_price == preset.prompt_price_per_1m
        and output_price == preset.output_price_per_1m
    ):
        return preset

    return PricingPreset(
        label=f"{preset.label}-with-overrides",
        provider=preset.provider,
        model=preset.model,
        prompt_price_per_1m=prompt_price,
        output_price_per_1m=output_price,
        source_url=preset.source_url,
        note=f"{preset.note} Manual CLI price override was applied.",
    )


def estimate_cost_usd(
    prompt_tokens: int,
    output_tokens: int,
    pricing: PricingPreset,
) -> float:
    return (
        prompt_tokens / 1_000_000 * pricing.prompt_price_per_1m
        + output_tokens / 1_000_000 * pricing.output_price_per_1m
    )


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stats_from_server_log(
    log_path: Path,
    strategy_config: dict[str, Any],
) -> dict[str, int | float]:
    stats: dict[str, int | float] = {
        "strategy_result_count": 0,
        "agent_response_count": 0,
        "prompt_tokens": 0,
        "output_tokens": 0,
        "total_latency_s": 0.0,
    }

    if not log_path.exists():
        return stats

    text = log_path.read_text(encoding="utf-8", errors="replace")
    response_blocks = re.findall(
        r"\+\+\+ FASTAPI COMPLETION RESPONSE \+\+\+.*?"
        r"prompt tokens: (\d+).*?"
        r"completion tokens: (\d+)",
        text,
        flags=re.DOTALL,
    )
    prompt_tokens = sum(int(prompt) for prompt, _ in response_blocks)
    output_tokens = sum(int(output) for _, output in response_blocks)
    strategy_result_count = len(response_blocks)
    agent_number = strategy_config.get("agent_number", 1)

    if not isinstance(agent_number, int) or agent_number < 1:
        agent_number = 1

    startup_marker = "Application startup complete."
    generation_text = text.split(startup_marker, maxsplit=1)[-1]
    durations = [
        float(duration)
        for duration in re.findall(r"duration seconds: ([0-9.]+)", generation_text)
    ]

    stats["strategy_result_count"] = strategy_result_count
    stats["agent_response_count"] = strategy_result_count * agent_number
    stats["prompt_tokens"] = prompt_tokens
    stats["output_tokens"] = output_tokens
    stats["total_latency_s"] = sum(durations)
    return stats


def latest_experiment_dir() -> Path:
    candidates = [path for path in EXPERIMENTS_DIR.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No experiment runs found in {EXPERIMENTS_DIR}")

    return max(candidates, key=lambda path: path.stat().st_mtime)


def percent(value: float | None) -> str:
    if value is None:
        return "n/a"

    return f"{value * 100:.1f}%"


def number(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"

    if isinstance(value, int):
        return f"{value:,}"

    return f"{value:,.{digits}f}"


def mean(values: list[float]) -> float | None:
    if not values:
        return None

    return sum(values) / len(values)


def sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None

    avg = sum(values) / len(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def mean_std_text(avg: float | None, std: float | None, suffix: str = "") -> str:
    if avg is None:
        return "n/a"

    if std is None:
        return f"{number(avg, 2)}{suffix}"

    return f"{number(avg, 2)} +/- {number(std, 2)}{suffix}"


def money(value: float | None) -> str:
    if value is None:
        return "n/a"

    return f"${value:,.4f}"


def extract_final_answer(text: str | None) -> str | None:
    if text is None:
        return None

    final_answer_matches = re.findall(
        r"The final answer is\s*(-?[$0-9.,]+)",
        text,
        flags=re.IGNORECASE,
    )
    if final_answer_matches:
        return final_answer_matches[-1].replace("$", "").replace(",", "")

    hash_matches = re.findall(r"####\s*(-?[$0-9.,]+)", text)
    if hash_matches:
        return hash_matches[-1].replace("$", "").replace(",", "")

    matches = re.findall(r"-?[$0-9.,]+", text)
    if not matches:
        return None

    return matches[-1].replace("$", "").replace(",", "")


def grouped_samples(samples: list[dict[str, Any]]) -> dict[int, dict[str, dict[str, Any]]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = {}

    for sample in samples:
        doc_id = int(sample["doc_id"])
        grouped.setdefault(doc_id, {})[sample["filter"]] = sample

    return grouped


def task_from_summary(
    summary: dict[str, Any],
    lm_eval_results: dict[str, Any],
) -> str:
    task = summary.get("task")
    if isinstance(task, str):
        return task

    sample_tasks = list(lm_eval_results.get("samples", {}))
    if sample_tasks:
        return sample_tasks[0]

    result_tasks = list(lm_eval_results.get("results", {}))
    if result_tasks:
        return result_tasks[0]

    return "gsm8k"


def sample_key(doc: dict[str, Any]) -> str | None:
    question = doc.get("question")
    if isinstance(question, str):
        return normalize_text(question)

    return None


def grouped_samples_by_prompt_key(
    samples: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}

    for sample in samples:
        key = sample_key(sample["doc"])
        if key is not None:
            filter_name = sample.get("filter", "exact_match")
            grouped.setdefault(key, {})[filter_name] = sample

    return grouped


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def extract_prompt_key(prompt: str) -> str | None:
    if "Question:" not in prompt:
        return None

    question_part = prompt.rsplit("Question:", maxsplit=1)[1]
    question, _, _ = question_part.partition("\nAnswer:")

    return normalize_text(question)


def final_vote_counts(strategy_result: dict[str, Any]) -> Counter[str | None]:
    strategy_config = strategy_result.get("_strategy_config", {})
    agent_number = strategy_config.get("agent_number")
    agent_responses = strategy_result.get("agent_responses", [])

    if isinstance(agent_number, int) and agent_number > 0:
        vote_responses = agent_responses[-agent_number:]
    else:
        vote_responses = agent_responses

    return Counter(response.get("extracted_response") for response in vote_responses)


def wilson_interval(successes: int, total: int) -> tuple[float, float] | None:
    if total == 0:
        return None

    z = 1.96
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    return max(0, center - margin), min(1, center + margin)


def read_strategy_analysis(
    run_dir: Path,
    summary: dict[str, Any],
    pricing: PricingPreset,
) -> tuple[StrategyAnalysis, list[dict[str, Any]]]:
    strategy_name = summary["strategy"]
    base_strategy = summary.get("base_strategy", strategy_name)
    strategy_dir = run_dir / strategy_name
    lm_eval_results = load_json(strategy_dir / "lm_eval_results.json")
    task_name = task_from_summary(summary, lm_eval_results)
    strategy_config = summary.get("strategy_config", {})
    strategy_records = load_jsonl(strategy_dir / "strategy_results.jsonl")

    samples_by_prompt_key = grouped_samples_by_prompt_key(
        lm_eval_results["samples"][task_name],
    )

    per_sample_rows = []
    flexible_only_correct = 0
    unanimous_final_votes = 0
    split_final_votes = 0
    extraction_failures = 0

    for sample_index, record in enumerate(strategy_records):
        strategy_result = record["strategy_result"]
        strategy_result["_strategy_config"] = strategy_config
        prompt_key = extract_prompt_key(strategy_result["prompt"])
        sample = samples_by_prompt_key.get(prompt_key or "", {})
        fallback_sample = next(iter(sample.values()), {})
        strict_sample = sample.get(
            "strict-match",
            sample.get("exact_match", fallback_sample),
        )
        flexible_sample = sample.get("flexible-extract", {})
        strict_correct = strict_sample.get("exact_match")
        flexible_correct = flexible_sample.get("exact_match")
        target = (
            strict_sample.get("target")
            or flexible_sample.get("target")
            or fallback_sample.get("target")
        )
        target_answer = extract_final_answer(target)
        vote_counts = final_vote_counts(strategy_result)

        if flexible_correct == 1.0 and strict_correct == 0.0:
            flexible_only_correct += 1

        if len(vote_counts) <= 1:
            unanimous_final_votes += 1
        else:
            split_final_votes += 1

        extraction_failures += sum(
            1
            for agent_response in strategy_result.get("agent_responses", [])
            if agent_response.get("extracted_response") is None
        )
        if strategy_result.get("extracted_response") is None:
            extraction_failures += 1

        per_sample_rows.append(
            {
                "strategy": strategy_name,
                "base_strategy": base_strategy,
                "task": task_name,
                "repeat_index": summary.get("repeat_index", 1),
                "doc_id": strict_sample.get(
                    "doc_id",
                    flexible_sample.get("doc_id", sample_index),
                ),
                "question": prompt_key,
                "target_answer": target_answer,
                "selected_answer": strategy_result.get("extracted_response"),
                "strict_correct": strict_correct,
                "flexible_correct": flexible_correct,
                "flexible_only_correct": flexible_correct == 1.0
                and strict_correct == 0.0,
                "agent_calls": len(strategy_result.get("agent_responses", [])),
                "prompt_tokens": strategy_result.get("prompt_tokens", 0),
                "output_tokens": strategy_result.get("output_tokens", 0),
                "total_latency_s": strategy_result.get("total_latency_s") or 0,
                "pricing_preset": pricing.label,
                "pricing_provider": pricing.provider,
                "pricing_model": pricing.model,
                "estimated_cost_usd": estimate_cost_usd(
                    strategy_result.get("prompt_tokens", 0)
                    or 0,
                    strategy_result.get("output_tokens", 0)
                    or 0,
                    pricing,
                ),
                "money_spent_usd": estimate_cost_usd(
                    strategy_result.get("prompt_tokens", 0)
                    or 0,
                    strategy_result.get("output_tokens", 0)
                    or 0,
                    pricing,
                ),
                "vote_counts": dict(vote_counts),
            }
        )

    metrics = summary["metrics"]
    stats = dict(summary["strategy_stats"])
    if not strategy_records and stats.get("strategy_result_count", 0) == 0:
        stats = stats_from_server_log(strategy_dir / "server.log", strategy_config)
    effective_samples = metrics["effective_samples"]
    total_tokens = stats["prompt_tokens"] + stats["output_tokens"]
    estimated_cost_usd = estimate_cost_usd(
        stats["prompt_tokens"],
        stats["output_tokens"],
        pricing,
    )

    analysis = StrategyAnalysis(
        name=strategy_name,
        base_strategy=base_strategy,
        task=task_name,
        repeat_index=summary.get("repeat_index", 1),
        repetitions=summary.get("repetitions", 1),
        limit=summary["limit"],
        effective_samples=effective_samples,
        strict_accuracy=metrics["exact_match_strict"],
        flexible_accuracy=metrics["exact_match_flexible"],
        wall_time_s=summary["wall_time_s"],
        strategy_result_count=stats["strategy_result_count"],
        agent_response_count=stats["agent_response_count"],
        prompt_tokens=stats["prompt_tokens"],
        output_tokens=stats["output_tokens"],
        total_latency_s=stats["total_latency_s"],
        estimated_cost_usd=estimated_cost_usd,
        avg_calls_per_sample=stats["agent_response_count"] / effective_samples,
        avg_tokens_per_sample=total_tokens / effective_samples,
        avg_latency_per_sample_s=stats["total_latency_s"] / effective_samples,
        avg_cost_per_sample_usd=estimated_cost_usd / effective_samples,
        flexible_only_correct=flexible_only_correct,
        unanimous_final_votes=unanimous_final_votes,
        split_final_votes=split_final_votes,
        extraction_failures=extraction_failures,
    )

    return analysis, per_sample_rows


def make_aggregate_analyses(
    analyses: list[StrategyAnalysis],
) -> list[AggregateAnalysis]:
    grouped: dict[str, list[StrategyAnalysis]] = {}

    for analysis in analyses:
        grouped.setdefault(analysis.base_strategy, []).append(analysis)

    aggregate_analyses = []
    for base_strategy, strategy_runs in grouped.items():
        strict_values = [
            analysis.strict_accuracy
            for analysis in strategy_runs
            if analysis.strict_accuracy is not None
        ]
        flexible_values = [
            analysis.flexible_accuracy
            for analysis in strategy_runs
            if analysis.flexible_accuracy is not None
        ]
        samples_per_run_values = {
            analysis.effective_samples for analysis in strategy_runs
        }

        aggregate_analyses.append(
            AggregateAnalysis(
                base_strategy=base_strategy,
                runs=len(strategy_runs),
                samples_per_run=(
                    samples_per_run_values.pop()
                    if len(samples_per_run_values) == 1
                    else 0
                ),
                strict_accuracy_mean=mean(strict_values),
                strict_accuracy_std=sample_std(strict_values),
                flexible_accuracy_mean=mean(flexible_values),
                flexible_accuracy_std=sample_std(flexible_values),
                avg_calls_per_sample_mean=sum(
                    analysis.avg_calls_per_sample for analysis in strategy_runs
                )
                / len(strategy_runs),
                avg_tokens_per_sample_mean=sum(
                    analysis.avg_tokens_per_sample for analysis in strategy_runs
                )
                / len(strategy_runs),
                avg_latency_per_sample_s_mean=sum(
                    analysis.avg_latency_per_sample_s for analysis in strategy_runs
                )
                / len(strategy_runs),
                avg_cost_per_sample_usd_mean=sum(
                    analysis.avg_cost_per_sample_usd for analysis in strategy_runs
                )
                / len(strategy_runs),
                estimated_cost_usd_total=sum(
                    analysis.estimated_cost_usd for analysis in strategy_runs
                ),
            )
        )

    return aggregate_analyses


def svg_bar_chart(
    title: str,
    values: list[tuple[str, float]],
    suffix: str = "",
    max_value: float | None = None,
) -> str:
    width = 760
    row_height = 44
    label_width = 160
    chart_width = 500
    height = 62 + row_height * len(values)
    max_bar_value = max_value or max((value for _, value in values), default=1) or 1
    palette = ["#2563eb", "#059669", "#c2410c", "#7c3aed", "#0f766e"]

    rows = [
        f'<text x="20" y="30" font-size="18" font-weight="700">{html.escape(title)}</text>'
    ]

    for index, (label, value) in enumerate(values):
        y = 56 + index * row_height
        bar_width = 0 if max_bar_value == 0 else value / max_bar_value * chart_width
        color = palette[index % len(palette)]
        value_label = money(value) if suffix == "$" else number(value, 2) + suffix
        rows.append(
            f'<text x="20" y="{y + 18}" font-size="13">{html.escape(label)}</text>'
        )
        rows.append(
            f'<rect x="{label_width}" y="{y}" width="{bar_width:.1f}" '
            f'height="24" rx="4" fill="{color}"></rect>'
        )
        rows.append(
            f'<text x="{label_width + bar_width + 8}" y="{y + 17}" '
            f'font-size="13">{html.escape(value_label)}</text>'
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(title)}">{"".join(rows)}</svg>'
    )


def make_interpretation(analyses: list[StrategyAnalysis]) -> list[str]:
    interpretations = []
    by_name = {analysis.name: analysis for analysis in analyses}
    aggregate_analyses = make_aggregate_analyses(analyses)
    by_base_name = {
        aggregate.base_strategy: aggregate for aggregate in aggregate_analyses
    }
    best_strict = max(
        analyses,
        key=lambda analysis: analysis.strict_accuracy
        if analysis.strict_accuracy is not None
        else -1,
    )
    direct = by_name.get("direct")
    direct_aggregate = by_base_name.get("direct")
    min_samples = min(analysis.effective_samples for analysis in analyses)
    has_repetitions = any(aggregate.runs > 1 for aggregate in aggregate_analyses)

    if has_repetitions:
        best_aggregate = max(
            aggregate_analyses,
            key=lambda aggregate: aggregate.strict_accuracy_mean
            if aggregate.strict_accuracy_mean is not None
            else -1,
        )
        interpretations.append(
            "Best mean strict accuracy across repetitions: "
            f"{best_aggregate.base_strategy} at "
            f"{percent(best_aggregate.strict_accuracy_mean)}."
        )
    else:
        interpretations.append(
            f"Best strict accuracy in this run: {best_strict.name} at "
            f"{percent(best_strict.strict_accuracy)}."
        )

    if min_samples < 30:
        step = 1 / min_samples
        interpretations.append(
            f"This run is very small: n={min_samples}. One changed answer moves "
            f"accuracy by {step * 100:.1f} percentage points, so treat ranking as a "
            "smoke test, not evidence."
        )

    if has_repetitions and direct_aggregate is not None:
        for aggregate in aggregate_analyses:
            if aggregate.base_strategy == "direct":
                continue

            if direct_aggregate.avg_tokens_per_sample_mean > 0:
                token_multiplier = (
                    aggregate.avg_tokens_per_sample_mean
                    / direct_aggregate.avg_tokens_per_sample_mean
                )
                interpretations.append(
                    f"{aggregate.base_strategy} used about {token_multiplier:.1f}x "
                    "the tokens per sample compared with direct on average."
                )

            if direct_aggregate.avg_cost_per_sample_usd_mean > 0:
                cost_multiplier = (
                    aggregate.avg_cost_per_sample_usd_mean
                    / direct_aggregate.avg_cost_per_sample_usd_mean
                )
                interpretations.append(
                    f"{aggregate.base_strategy} cost about {cost_multiplier:.1f}x "
                    "more per sample than direct under the configured token prices."
                )

    elif direct is not None:
        for analysis in analyses:
            if analysis.name == "direct":
                continue

            if (
                direct.strict_accuracy is not None
                and analysis.strict_accuracy is not None
                and direct.strict_accuracy > analysis.strict_accuracy
            ):
                interpretations.append(
                    f"{analysis.name} is below direct on strict accuracy here. "
                    "That can happen because direct is deterministic while this "
                    "strategy samples with nonzero temperature, so extra calls can add "
                    "both useful diversity and new mistakes."
                )

            if direct.avg_tokens_per_sample > 0:
                token_multiplier = (
                    analysis.avg_tokens_per_sample / direct.avg_tokens_per_sample
                )
                interpretations.append(
                    f"{analysis.name} used about {token_multiplier:.1f}x the "
                    "tokens per sample compared with direct."
                )

            if direct.avg_cost_per_sample_usd > 0:
                cost_multiplier = (
                    analysis.avg_cost_per_sample_usd
                    / direct.avg_cost_per_sample_usd
                )
                interpretations.append(
                    f"{analysis.name} cost about {cost_multiplier:.1f}x more "
                    "per sample than direct under the configured token prices."
                )

    for analysis in analyses:
        if analysis.flexible_only_correct:
            interpretations.append(
                f"{analysis.name} had {analysis.flexible_only_correct} sample(s) "
                "where flexible extraction was correct but strict matching failed. "
                "That points to formatting, not necessarily reasoning."
            )

        if analysis.split_final_votes:
            interpretations.append(
                f"{analysis.name} had split final votes on "
                f"{analysis.split_final_votes}/{analysis.strategy_result_count} "
                "sample(s), so the multi-agent runs are producing real diversity."
            )

    return interpretations


def make_pricing_scenario_rows(analyses: list[StrategyAnalysis]) -> list[dict[str, Any]]:
    rows = []

    for pricing in PRICING_PRESETS.values():
        for analysis in analyses:
            total_cost = estimate_cost_usd(
                prompt_tokens=analysis.prompt_tokens,
                output_tokens=analysis.output_tokens,
                pricing=pricing,
            )
            rows.append(
                {
                    "pricing_preset": pricing.label,
                    "provider": pricing.provider,
                    "model": pricing.model,
                    "strategy": analysis.name,
                    "samples": analysis.effective_samples,
                    "prompt_price_per_1m": pricing.prompt_price_per_1m,
                    "output_price_per_1m": pricing.output_price_per_1m,
                    "prompt_tokens": analysis.prompt_tokens,
                    "output_tokens": analysis.output_tokens,
                    "money_spent_usd": total_cost,
                    "avg_money_spent_per_sample_usd": total_cost
                    / analysis.effective_samples,
                    "note": pricing.note,
                    "source_url": pricing.source_url,
                }
            )

    return rows


def make_pricing_scenario_table(analyses: list[StrategyAnalysis]) -> str:
    headers = ["Pricing preset", "Provider", "Model"]
    headers.extend(analysis.name for analysis in analyses)
    rows = [
        "<tr>"
        + "".join(f"<th>{html.escape(header)}</th>" for header in headers)
        + "</tr>"
    ]

    for pricing in PRICING_PRESETS.values():
        cells = [
            pricing.label,
            pricing.provider,
            pricing.model,
        ]
        for analysis in analyses:
            cells.append(
                money(
                    estimate_cost_usd(
                        prompt_tokens=analysis.prompt_tokens,
                        output_tokens=analysis.output_tokens,
                        pricing=pricing,
                    )
                )
            )

        rows.append(
            "<tr>"
            + "".join(f"<td>{html.escape(cell)}</td>" for cell in cells)
            + "</tr>"
        )

    return "<table>" + "".join(rows) + "</table>"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_strategy_comparison(
    path: Path,
    analyses: list[StrategyAnalysis],
    pricing: PricingPreset,
) -> None:
    rows = []

    for analysis in analyses:
        strict_successes = (
            round((analysis.strict_accuracy or 0) * analysis.effective_samples)
            if analysis.strict_accuracy is not None
            else 0
        )
        interval = wilson_interval(strict_successes, analysis.effective_samples)
        ci_low = interval[0] if interval else None
        ci_high = interval[1] if interval else None
        rows.append(
            {
                "strategy": analysis.name,
                "base_strategy": analysis.base_strategy,
                "task": analysis.task,
                "repeat_index": analysis.repeat_index,
                "repetitions": analysis.repetitions,
                "samples": analysis.effective_samples,
                "strict_accuracy": analysis.strict_accuracy,
                "flexible_accuracy": analysis.flexible_accuracy,
                "strict_95ci_low": ci_low,
                "strict_95ci_high": ci_high,
                "wall_time_s": analysis.wall_time_s,
                "agent_calls": analysis.agent_response_count,
                "avg_calls_per_sample": analysis.avg_calls_per_sample,
                "prompt_tokens": analysis.prompt_tokens,
                "output_tokens": analysis.output_tokens,
                "avg_tokens_per_sample": analysis.avg_tokens_per_sample,
                "total_latency_s": analysis.total_latency_s,
                "avg_latency_per_sample_s": analysis.avg_latency_per_sample_s,
                "pricing_preset": pricing.label,
                "pricing_provider": pricing.provider,
                "pricing_model": pricing.model,
                "prompt_price_per_1m": pricing.prompt_price_per_1m,
                "output_price_per_1m": pricing.output_price_per_1m,
                "estimated_cost_usd": analysis.estimated_cost_usd,
                "avg_cost_per_sample_usd": analysis.avg_cost_per_sample_usd,
                "money_spent_usd": analysis.estimated_cost_usd,
                "avg_money_spent_per_sample_usd": analysis.avg_cost_per_sample_usd,
                "flexible_only_correct": analysis.flexible_only_correct,
                "split_final_votes": analysis.split_final_votes,
                "extraction_failures": analysis.extraction_failures,
            }
        )

    write_csv(path, rows)


def write_aggregate_comparison(
    path: Path,
    aggregate_analyses: list[AggregateAnalysis],
) -> None:
    rows = []

    for aggregate in aggregate_analyses:
        rows.append(
            {
                "base_strategy": aggregate.base_strategy,
                "runs": aggregate.runs,
                "samples_per_run": aggregate.samples_per_run,
                "strict_accuracy_mean": aggregate.strict_accuracy_mean,
                "strict_accuracy_std": aggregate.strict_accuracy_std,
                "flexible_accuracy_mean": aggregate.flexible_accuracy_mean,
                "flexible_accuracy_std": aggregate.flexible_accuracy_std,
                "avg_calls_per_sample_mean": aggregate.avg_calls_per_sample_mean,
                "avg_tokens_per_sample_mean": aggregate.avg_tokens_per_sample_mean,
                "avg_latency_per_sample_s_mean": aggregate.avg_latency_per_sample_s_mean,
                "avg_cost_per_sample_usd_mean": aggregate.avg_cost_per_sample_usd_mean,
                "estimated_cost_usd_total": aggregate.estimated_cost_usd_total,
            }
        )

    write_csv(path, rows)


def make_aggregate_markdown_table(
    aggregate_analyses: list[AggregateAnalysis],
) -> list[str]:
    if all(aggregate.runs == 1 for aggregate in aggregate_analyses):
        return []

    lines = [
        "",
        "## Repetition Summary",
        "",
        "| Strategy | Runs | Samples/run | Strict mean +/- std | Flexible mean +/- std | Calls/sample | Tokens/sample | Latency/sample | Total money est. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for aggregate in aggregate_analyses:
        lines.append(
            "| "
            f"{aggregate.base_strategy} | "
            f"{aggregate.runs} | "
            f"{aggregate.samples_per_run or 'mixed'} | "
            f"{mean_std_text((aggregate.strict_accuracy_mean or 0) * 100 if aggregate.strict_accuracy_mean is not None else None, (aggregate.strict_accuracy_std or 0) * 100 if aggregate.strict_accuracy_std is not None else None, '%')} | "
            f"{mean_std_text((aggregate.flexible_accuracy_mean or 0) * 100 if aggregate.flexible_accuracy_mean is not None else None, (aggregate.flexible_accuracy_std or 0) * 100 if aggregate.flexible_accuracy_std is not None else None, '%')} | "
            f"{number(aggregate.avg_calls_per_sample_mean, 1)} | "
            f"{number(aggregate.avg_tokens_per_sample_mean, 0)} | "
            f"{number(aggregate.avg_latency_per_sample_s_mean, 2)}s | "
            f"{money(aggregate.estimated_cost_usd_total)} |"
        )

    lines.extend(
        [
            "",
            "The table groups repeated strategy runs by `base_strategy`; individual runs remain available in `strategy_comparison.csv` and `per_sample.csv`.",
        ]
    )
    return lines


def make_markdown(
    run_dir: Path,
    analyses: list[StrategyAnalysis],
    pricing: PricingPreset,
) -> str:
    aggregate_analyses = make_aggregate_analyses(analyses)
    lines = [
        "# Experiment Analysis",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "Cost basis: "
        f"`{pricing.label}` ({pricing.provider} {pricing.model}), "
        f"${pricing.prompt_price_per_1m:g}/1M input tokens and "
        f"${pricing.output_price_per_1m:g}/1M output tokens.",
        "",
        "## Summary",
        "",
        "| Strategy | Samples | Strict | Flexible | Calls/sample | Tokens/sample | Latency/sample | Money spent est. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for analysis in analyses:
        lines.append(
            "| "
            f"{analysis.name} | "
            f"{analysis.effective_samples} | "
            f"{percent(analysis.strict_accuracy)} | "
            f"{percent(analysis.flexible_accuracy)} | "
            f"{analysis.avg_calls_per_sample:.1f} | "
            f"{analysis.avg_tokens_per_sample:,.0f} | "
            f"{analysis.avg_latency_per_sample_s:.2f}s | "
            f"{money(analysis.estimated_cost_usd)} |"
        )

    lines.extend(make_aggregate_markdown_table(aggregate_analyses))

    lines.extend(["", "## Interpretation", ""])
    for point in make_interpretation(analyses):
        lines.append(f"- {point}")

    lines.extend(
        [
            "",
            "## API Cost Scenarios",
            "",
            "| Pricing preset | Provider | Model | "
            + " | ".join(analysis.name for analysis in analyses)
            + " |",
            "|---|---|---|"
            + "|".join("---:" for _ in analyses)
            + "|",
        ]
    )
    for preset in PRICING_PRESETS.values():
        cost_cells = [
            money(
                estimate_cost_usd(
                    prompt_tokens=analysis.prompt_tokens,
                    output_tokens=analysis.output_tokens,
                    pricing=preset,
                )
            )
            for analysis in analyses
        ]
        lines.append(
            f"| {preset.label} | {preset.provider} | {preset.model} | "
            + " | ".join(cost_cells)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `strict` is the benchmark's exact final-answer format match.",
            "- `flexible` is more forgiving extraction of the final number.",
            "- `strategy_results.jsonl` contains every internal agent response when raw strategy logging is available.",
            "- Estimated cost is an API-equivalent token estimate from saved prompt/output token counts. For local Ollama billing, use the `ollama-local` preset.",
        ]
    )

    return "\n".join(lines) + "\n"


def make_html(
    run_dir: Path,
    analyses: list[StrategyAnalysis],
    pricing: PricingPreset,
) -> str:
    aggregate_analyses = make_aggregate_analyses(analyses)
    accuracy_values = [
        (analysis.name, (analysis.strict_accuracy or 0) * 100)
        for analysis in analyses
    ]
    flexible_values = [
        (analysis.name, (analysis.flexible_accuracy or 0) * 100)
        for analysis in analyses
    ]
    token_values = [
        (analysis.name, analysis.avg_tokens_per_sample) for analysis in analyses
    ]
    latency_values = [
        (analysis.name, analysis.avg_latency_per_sample_s) for analysis in analyses
    ]
    call_values = [
        (analysis.name, analysis.avg_calls_per_sample) for analysis in analyses
    ]
    cost_values = [
        (analysis.name, analysis.estimated_cost_usd) for analysis in analyses
    ]

    cards = []
    for analysis in analyses:
        cards.append(
            f"""
            <section class="card">
              <h2>{html.escape(analysis.name)}</h2>
              <dl>
                <dt>Strict accuracy</dt><dd>{percent(analysis.strict_accuracy)}</dd>
                <dt>Flexible accuracy</dt><dd>{percent(analysis.flexible_accuracy)}</dd>
                <dt>Samples</dt><dd>{analysis.effective_samples}</dd>
                <dt>Agent calls</dt><dd>{analysis.agent_response_count}</dd>
                <dt>Avg calls/sample</dt><dd>{analysis.avg_calls_per_sample:.1f}</dd>
                <dt>Avg tokens/sample</dt><dd>{analysis.avg_tokens_per_sample:,.0f}</dd>
                <dt>Avg latency/sample</dt><dd>{analysis.avg_latency_per_sample_s:.2f}s</dd>
                <dt>Money spent est.</dt><dd>{money(analysis.estimated_cost_usd)}</dd>
                <dt>Avg money/sample</dt><dd>{money(analysis.avg_cost_per_sample_usd)}</dd>
                <dt>Split votes</dt><dd>{analysis.split_final_votes}</dd>
              </dl>
            </section>
            """
        )

    interpretation_items = "\n".join(
        f"<li>{html.escape(point)}</li>" for point in make_interpretation(analyses)
    )
    aggregate_table = ""
    if any(aggregate.runs > 1 for aggregate in aggregate_analyses):
        rows = [
            "<tr><th>Strategy</th><th>Runs</th><th>Samples/run</th>"
            "<th>Strict mean +/- std</th><th>Flexible mean +/- std</th>"
            "<th>Calls/sample</th><th>Tokens/sample</th>"
            "<th>Latency/sample</th><th>Total money est.</th></tr>"
        ]
        for aggregate in aggregate_analyses:
            rows.append(
                "<tr>"
                f"<td>{html.escape(aggregate.base_strategy)}</td>"
                f"<td>{aggregate.runs}</td>"
                f"<td>{aggregate.samples_per_run or 'mixed'}</td>"
                f"<td>{html.escape(mean_std_text((aggregate.strict_accuracy_mean or 0) * 100 if aggregate.strict_accuracy_mean is not None else None, (aggregate.strict_accuracy_std or 0) * 100 if aggregate.strict_accuracy_std is not None else None, '%'))}</td>"
                f"<td>{html.escape(mean_std_text((aggregate.flexible_accuracy_mean or 0) * 100 if aggregate.flexible_accuracy_mean is not None else None, (aggregate.flexible_accuracy_std or 0) * 100 if aggregate.flexible_accuracy_std is not None else None, '%'))}</td>"
                f"<td>{number(aggregate.avg_calls_per_sample_mean, 1)}</td>"
                f"<td>{number(aggregate.avg_tokens_per_sample_mean, 0)}</td>"
                f"<td>{number(aggregate.avg_latency_per_sample_s_mean, 2)}s</td>"
                f"<td>{money(aggregate.estimated_cost_usd_total)}</td>"
                "</tr>"
            )
        aggregate_table = (
            "<section class=\"notes\"><h2>Repetition Summary</h2>"
            f"<table>{''.join(rows)}</table></section>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Experiment Analysis</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      color: #1f2937;
      background: #f8fafc;
    }}
    main {{
      max-width: 1060px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    h1, h2 {{
      margin: 0 0 12px;
    }}
    p {{
      line-height: 1.5;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
      margin: 20px 0;
    }}
    .card, .chart, .notes {{
      background: white;
      border: 1px solid #dbe3ef;
      border-radius: 8px;
      padding: 18px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    dl {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px 14px;
      margin: 0;
    }}
    dt {{
      color: #64748b;
    }}
    dd {{
      margin: 0;
      font-weight: 700;
    }}
    svg {{
      width: 100%;
      height: auto;
    }}
    li {{
      margin: 8px 0;
      line-height: 1.5;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid #e2e8f0;
      padding: 8px 6px;
      text-align: left;
    }}
    th:last-child, td:last-child {{
      text-align: right;
    }}
    code {{
      background: #e2e8f0;
      padding: 2px 5px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Experiment Analysis</h1>
    <p>Run directory: <code>{html.escape(str(run_dir))}</code></p>
    <p>Cost basis: <code>{html.escape(pricing.label)}</code>
    ({html.escape(pricing.provider)} {html.escape(pricing.model)}),
    ${pricing.prompt_price_per_1m:g}/1M input tokens and
    ${pricing.output_price_per_1m:g}/1M output tokens. These are API-equivalent
    estimates from saved token counts; local Ollama token billing is zero.</p>

    <div class="grid">
      {''.join(cards)}
    </div>

    {aggregate_table}

    <section class="chart">{svg_bar_chart("Strict Accuracy", accuracy_values, "%", 100)}</section>
    <section class="chart">{svg_bar_chart("Flexible Accuracy", flexible_values, "%", 100)}</section>
    <section class="chart">{svg_bar_chart("Average Tokens Per Sample", token_values)}</section>
    <section class="chart">{svg_bar_chart("Average Latency Per Sample", latency_values, "s")}</section>
    <section class="chart">{svg_bar_chart("Average Agent Calls Per Sample", call_values)}</section>
    <section class="chart">{svg_bar_chart(f"Money Spent Estimate ({pricing.label})", cost_values, "$")}</section>

    <section class="notes">
      <h2>API Cost Scenarios</h2>
      {make_pricing_scenario_table(analyses)}
    </section>

    <section class="notes">
      <h2>Interpretation</h2>
      <ul>{interpretation_items}</ul>
    </section>
  </main>
</body>
</html>
"""


def analyze(
    run_dir: Path,
    pricing_preset: str = DEFAULT_PRICING_PRESET,
    prompt_price_per_1m: float | None = None,
    output_price_per_1m: float | None = None,
) -> Path:
    pricing = resolve_pricing(
        pricing_preset=pricing_preset,
        prompt_price_per_1m=prompt_price_per_1m,
        output_price_per_1m=output_price_per_1m,
    )
    summary = load_json(run_dir / "summary.json")
    analyses = []
    per_sample_rows = []

    for strategy_summary in summary:
        strategy_analysis, strategy_sample_rows = read_strategy_analysis(
            run_dir=run_dir,
            summary=strategy_summary,
            pricing=pricing,
        )
        analyses.append(strategy_analysis)
        per_sample_rows.extend(strategy_sample_rows)

    aggregate_analyses = make_aggregate_analyses(analyses)
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    write_strategy_comparison(
        analysis_dir / "strategy_comparison.csv",
        analyses,
        pricing,
    )
    write_aggregate_comparison(
        analysis_dir / "aggregate_strategy_comparison.csv",
        aggregate_analyses,
    )
    write_csv(analysis_dir / "per_sample.csv", per_sample_rows)
    write_csv(analysis_dir / "pricing_scenarios.csv", make_pricing_scenario_rows(analyses))

    markdown = make_markdown(run_dir, analyses, pricing)
    (analysis_dir / "report.md").write_text(markdown, encoding="utf-8")

    report_html = make_html(run_dir, analyses, pricing)
    (analysis_dir / "report.html").write_text(report_html, encoding="utf-8")

    load_json(run_dir / "summary.json")
    cost_config = {
        "selected_pricing": asdict(pricing),
        "available_presets": {
            label: asdict(preset) for label, preset in PRICING_PRESETS.items()
        },
        "currency": "USD",
        "note": "Estimated from saved token counts. Local Ollama has no API-token bill; use the ollama-local preset for actual local API spend.",
    }
    (analysis_dir / "cost_config.json").write_text(
        json.dumps(cost_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"analysis written to: {analysis_dir}")
    print(f"open report: {analysis_dir / 'report.html'}")

    return analysis_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        help="Experiment output directory. Defaults to the latest run.",
    )
    parser.add_argument(
        "--pricing-preset",
        default=DEFAULT_PRICING_PRESET,
        choices=sorted(PRICING_PRESETS),
        help="Named API price preset used for the headline cost estimate.",
    )
    parser.add_argument(
        "--list-pricing-presets",
        action="store_true",
        help="Print available pricing presets and exit.",
    )
    parser.add_argument(
        "--prompt-price-per-1m",
        type=float,
        default=None,
        help="Override prompt/input token price per 1M tokens in USD.",
    )
    parser.add_argument(
        "--output-price-per-1m",
        type=float,
        default=None,
        help="Override output/completion token price per 1M tokens in USD.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_pricing_presets:
        for preset in PRICING_PRESETS.values():
            print(
                f"{preset.label}: {preset.provider} {preset.model} "
                f"input=${preset.prompt_price_per_1m:g}/1M "
                f"output=${preset.output_price_per_1m:g}/1M"
            )
        return

    run_dir = args.run_dir or latest_experiment_dir()
    analyze(
        run_dir=run_dir.resolve(),
        pricing_preset=args.pricing_preset,
        prompt_price_per_1m=args.prompt_price_per_1m,
        output_price_per_1m=args.output_price_per_1m,
    )


if __name__ == "__main__":
    main()
