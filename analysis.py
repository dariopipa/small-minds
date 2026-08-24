"""Create a static HTML report for one saved experiment directory."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from string import Template
from typing import Any

from analysis_plots import (
    FigureArtifact,
    StrategyPlotData,
    generate_academic_figures,
)

BASELINE_STRATEGY = "direct"
PRIMARY_FILTERS = {
    "gsm8k": "strict-match",
    "arc_challenge_chat": "remove_whitespace",
    "boolq": "none",
}
PRIMARY_METRICS = {
    "gsm8k": "exact_match",
    "arc_challenge_chat": "exact_match",
    "boolq": "exact_match",
}
FLEXIBLE_FILTERS = {"gsm8k": "flexible-extract"}
STRATEGY_LABELS = {
    "direct": "Direct",
    "self_consistency": "Self-Consistency",
    "society_of_minds": "Society of Minds",
    "role_based_svj": "Solver-Verifier-Judge",
}
TEMPLATE_DIR = Path(__file__).with_name("analysis_templates")
BENCHMARK_TEMPLATE = Template(
    (TEMPLATE_DIR / "benchmark.html").read_text(encoding="utf-8")
)
REPORT_TEMPLATE = Template((TEMPLATE_DIR / "report.html").read_text(encoding="utf-8"))


class AnalysisError(ValueError):
    """The saved experiment does not match a supported repository schema."""


@dataclass(slots=True)
class Sample:
    question_id: str
    expected_answer: str | int
    outcome: str
    metric: float
    flexible_metric: float | None
    model_latency: float | None
    end_to_end_latency: float | None
    provider_duration: float | None
    strategy_result: dict[str, Any] | None


@dataclass(slots=True)
class Run:
    path: str
    experiment_id: str
    benchmark: str
    strategy: str
    repetition: int
    status: str
    model: str
    samples: list[Sample]
    saved_score: float | None
    calls: int | None
    prompt_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    model_latency: float | None
    end_to_end_latency: float | None
    provider_duration: float | None
    wall_time: float | None
    cost: float | None

    @property
    def counts(self) -> Counter[str]:
        return Counter(sample.outcome for sample in self.samples)

    @property
    def score(self) -> float | None:
        if self.samples:
            return statistics.fmean(sample.metric for sample in self.samples)
        return self.saved_score

    @property
    def flexible_score(self) -> float | None:
        scores = [sample.flexible_metric for sample in self.samples]
        if not scores or any(score is None for score in scores):
            return None
        return statistics.fmean(score for score in scores if score is not None)


@dataclass(slots=True)
class Group:
    model: str
    benchmark: str
    strategy: str
    label: str
    runs: list[Run]
    unique_questions: int
    evaluated_observations: int
    correct: int
    incorrect: int
    failed: int
    unparseable: int
    mean: float | None
    flexible_mean: float | None
    std: float | None
    minimum: float | None
    maximum: float | None
    calls: int | None
    avg_calls: float | None
    prompt_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    avg_tokens: float | None
    model_latency: float | None
    avg_model_latency: float | None
    end_to_end_latency: float | None
    avg_end_to_end_latency: float | None
    provider_duration: float | None
    avg_provider_duration: float | None
    wall_time: float | None
    avg_wall_time: float | None
    median_model_latency: float | None
    p95_model_latency: float | None
    missing_sample_model_latencies: int
    median_end_to_end_latency: float | None
    p95_end_to_end_latency: float | None
    missing_sample_end_to_end_latencies: int
    cost: float | None
    partial_fields: tuple[str, ...]


@dataclass(slots=True)
class Comparison:
    model: str
    benchmark: str
    strategy: str
    label: str
    direct_score: float | None
    strategy_score: float | None
    gain: float | None
    relative_gain_percent: float | None
    relative_error_reduction_percent: float | None
    verdict: str
    compared_repetitions: int
    fixed: float | None
    regressed: float | None
    both_correct: float | None
    both_wrong: float | None
    matched: float | None
    correction_rate: float | None
    degradation_rate: float | None
    extra_calls_per_question: float | None
    extra_tokens_per_question: float | None
    extra_model_latency_per_question: float | None
    extra_end_to_end_latency_per_question: float | None
    extra_provider_duration_per_question: float | None
    gain_pp_per_additional_call: float | None
    gain_pp_per_additional_1k_tokens: float | None


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"Could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"Expected a JSON object in {path}.")
    return value


def required(data: dict[str, Any], key: str, expected: type, where: str) -> Any:
    value = data.get(key)
    if not isinstance(value, expected):
        raise AnalysisError(f"{where}: {key!r} must be {expected.__name__}.")
    return value


def read_int(
    data: dict[str, Any], key: str, where: str, *, required_field: bool = True
) -> int | None:
    value = data.get(key)
    if value is None and not required_field:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnalysisError(f"{where}: {key!r} must be an integer.")
    return value


def read_number(
    data: dict[str, Any], key: str, where: str, *, required_field: bool = True
) -> float | None:
    value = data.get(key)
    if value is None and not required_field:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{where}: {key!r} must be a number.")
    value = float(value)
    if not math.isfinite(value):
        raise AnalysisError(f"{where}: {key!r} must be finite.")
    return value


def metric_spec(benchmark: str) -> tuple[str, str]:
    try:
        return PRIMARY_FILTERS[benchmark], PRIMARY_METRICS[benchmark]
    except KeyError as exc:
        raise AnalysisError(
            f"No primary metric is configured for benchmark {benchmark!r}."
        ) from exc


def flexible_metric(
    evaluations: dict[str, Any], benchmark: str, where: str
) -> float | None:
    filter_name = FLEXIBLE_FILTERS.get(benchmark)
    if filter_name is None:
        return None
    evaluation = evaluations.get(filter_name)
    if not isinstance(evaluation, dict):
        return None
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return read_number(metrics, "exact_match", where, required_field=False)


def parse_outcome(status: str, metric: float, filtered_response: Any) -> str:
    if status == "failed":
        return "failed"
    if isinstance(filtered_response, list):
        filtered_response = filtered_response[0] if filtered_response else None
    if filtered_response is None or (
        isinstance(filtered_response, str)
        and filtered_response.strip().lower() in {"", "[invalid]", "invalid", "n/a"}
    ):
        return "unparseable"
    return "correct" if metric >= 1.0 - 1e-9 else "incorrect"


def parse_sample(record: dict[str, Any], benchmark: str, where: str) -> Sample:
    question_id = required(record, "question_id", str, where)
    status = required(record, "status", str, where)
    expected_answer = record.get("expected_answer")
    if benchmark == "boolq":
        if type(expected_answer) is not int or expected_answer not in {0, 1}:
            raise AnalysisError(
                f"{where}: 'expected_answer' must be a BoolQ label (0 or 1)."
            )
    else:
        required(record, "expected_answer", str, where)
    assert isinstance(expected_answer, (str, int))
    response = record.get("lm_eval_response")
    if response is not None and not isinstance(response, str):
        raise AnalysisError(f"{where}: 'lm_eval_response' must be a string or null.")

    filter_name, metric_name = metric_spec(benchmark)
    evaluations = required(record, "evaluations", dict, where)
    evaluation = required(evaluations, filter_name, dict, where)
    metrics = required(evaluation, "metrics", dict, where)
    metric = read_number(metrics, metric_name, where)
    assert metric is not None

    model_latency = None
    end_to_end_latency = None
    provider_duration = None
    strategy_result = None
    calls = required(record, "calls", list, where)
    successful_results = [
        call["result"]
        for call in calls
        if isinstance(call, dict) and isinstance(call.get("result"), dict)
    ]
    if successful_results:
        strategy_result = successful_results[-1]
        model_latency = read_number(
            strategy_result, "total_latency_s", where, required_field=False
        )
        end_to_end_latency = read_number(
            strategy_result,
            "end_to_end_latency_s",
            where,
            required_field=False,
        )
        provider_duration = read_number(
            strategy_result,
            "provider_duration_s",
            where,
            required_field=False,
        )

    return Sample(
        question_id=question_id,
        expected_answer=expected_answer,
        outcome=parse_outcome(status, metric, evaluation.get("response")),
        metric=metric,
        flexible_metric=flexible_metric(evaluations, benchmark, where),
        model_latency=model_latency,
        end_to_end_latency=end_to_end_latency,
        provider_duration=provider_duration,
        strategy_result=strategy_result,
    )


def parse_samples(
    path: Path,
    experiment_dir: Path,
    benchmark: str,
    warnings: list[str],
) -> list[Sample]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise AnalysisError(f"Could not read {path}: {exc}") from exc

    samples: list[Sample] = []
    seen_questions: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        where = f"{path.relative_to(experiment_dir)}:{line_number}"
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AnalysisError(f"Malformed JSON at {where}: {exc}") from exc
        if not isinstance(record, dict):
            raise AnalysisError(f"{where}: expected a JSON object.")
        if record.get("record_type") != "sample":
            warnings.append(f"{where}: unscored record excluded from accuracy.")
            continue
        sample = parse_sample(record, benchmark, where)
        if sample.question_id in seen_questions:
            raise AnalysisError(
                f"{where}: duplicate question_id {sample.question_id!r}."
            )
        seen_questions.add(sample.question_id)
        samples.append(sample)
    return samples


def parse_run(
    run_path: Path,
    experiment_dir: Path,
    warnings: list[str],
) -> Run:
    raw = read_object(run_path)
    where = str(run_path.relative_to(experiment_dir))
    benchmark = required(raw, "benchmark", str, where)
    strategy = required(raw, "strategy", str, where)
    repetition = read_int(raw, "repetition", where)
    assert repetition is not None
    status = required(raw, "status", str, where)
    model = required(raw, "model", str, where)

    sample_path = run_path.parent / "samples.jsonl"
    if not sample_path.is_file():
        raise AnalysisError(f"{where}: missing samples.jsonl.")
    samples = parse_samples(sample_path, experiment_dir, benchmark, warnings)
    saved_sample_count = read_int(raw, "sample_count", where)
    if saved_sample_count != len(samples):
        raise AnalysisError(
            f"{where}: sample_count={saved_sample_count}, found {len(samples)} samples."
        )

    saved_score = None
    usage = required(raw, "tokens", dict, where)
    run = Run(
        path=where,
        experiment_id=required(raw, "experiment_id", str, where),
        benchmark=benchmark,
        strategy=strategy,
        repetition=repetition,
        status=status,
        model=model,
        samples=samples,
        saved_score=saved_score,
        calls=read_int(raw, "model_call_count", where, required_field=False),
        prompt_tokens=read_int(usage, "prompt", where, required_field=False),
        output_tokens=read_int(usage, "output", where, required_field=False),
        total_tokens=read_int(usage, "total", where, required_field=False),
        model_latency=read_number(
            raw, "model_latency_seconds", where, required_field=False
        ),
        end_to_end_latency=read_number(
            raw, "end_to_end_latency_seconds", where, required_field=False
        ),
        provider_duration=read_number(
            raw, "provider_duration_seconds", where, required_field=False
        ),
        wall_time=read_number(raw, "wall_time_seconds", where, required_field=False),
        cost=read_number(raw, "cost", where, required_field=False),
    )
    if status != "completed":
        warnings.append(f"{where}: run status is {status!r}.")
    return run


def load_results(
    experiment_dirs: list[Path],
) -> tuple[str, list[Run], list[str]]:
    run_paths: list[tuple[Path, Path]] = []
    seen_paths: set[Path] = set()
    for experiment_dir in experiment_dirs:
        for run_path in sorted(experiment_dir.rglob("run.json")):
            resolved_path = run_path.resolve()
            if resolved_path not in seen_paths:
                seen_paths.add(resolved_path)
                run_paths.append((run_path, experiment_dir))
    if not run_paths:
        raise AnalysisError("No run.json files were found.")
    warnings: list[str] = []
    runs = [
        parse_run(run_path, experiment_dir, warnings)
        for run_path, experiment_dir in run_paths
    ]

    identities = [(run.model, run.benchmark, run.strategy, run.repetition) for run in runs]
    if len(identities) != len(set(identities)):
        duplicates = [item for item, count in Counter(identities).items() if count > 1]
        raise AnalysisError(f"Duplicate run identities: {duplicates}.")

    validate_result_set(runs, warnings)
    label = experiment_dirs[0].name if len(experiment_dirs) == 1 else f"{len(experiment_dirs)} result folders"
    return label, runs, list(dict.fromkeys(warnings))


def validate_result_set(runs: list[Run], warnings: list[str]) -> None:
    runs_by_benchmark_repetition: dict[tuple[str, str, int], list[Run]] = defaultdict(list)
    for run in runs:
        runs_by_benchmark_repetition[(run.model, run.benchmark, run.repetition)].append(
            run
        )
    for (model, benchmark, repetition), matching_runs in (
        runs_by_benchmark_repetition.items()
    ):
        baseline = next(
            (run for run in matching_runs if run.strategy == BASELINE_STRATEGY),
            matching_runs[0],
        )
        baseline_ids = {sample.question_id for sample in baseline.samples}
        for run in matching_runs:
            ids = {sample.question_id for sample in run.samples}
            if ids != baseline_ids:
                warnings.append(
                    f"{model}/{benchmark} repetition {repetition}: {run.strategy} has "
                    f"{len(ids)} question IDs, while {baseline.strategy} has "
                    f"{len(baseline_ids)}."
                )

    question_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
    for run in runs:
        question_ids[(run.model, run.benchmark)].update(
            sample.question_id for sample in run.samples
        )
    for (model, benchmark), ids in question_ids.items():
        if len(ids) < 10:
            warnings.append(
                f"{model}/{benchmark}: only {len(ids)} unique questions were evaluated; "
                "results are suitable for pipeline validation only."
            )


def calculate_statistics(runs: list[Run], warnings: list[str]) -> list[Group]:
    grouped: dict[tuple[str, str, str], list[Run]] = defaultdict(list)
    for run in runs:
        grouped[(run.model, run.benchmark, run.strategy)].append(run)

    groups: list[Group] = []
    for (model, benchmark, strategy), group_runs in grouped.items():
        group_runs.sort(key=lambda run: run.repetition)
        observations = sum(len(run.samples) for run in group_runs)
        scores = [run.score for run in group_runs if run.score is not None]
        flexible_scores = [
            run.flexible_score for run in group_runs if run.flexible_score is not None
        ]
        partial_fields: list[str] = []

        def aggregate(field: str) -> tuple[int | float | None, float | None]:
            covered_runs = [
                run for run in group_runs if getattr(run, field) is not None
            ]
            if not covered_runs:
                return None, None
            if len(covered_runs) != len(group_runs):
                partial_fields.append(field)
                warnings.append(
                    f"{benchmark}/{strategy}: {field} is available for "
                    f"{len(covered_runs)}/{len(group_runs)} runs; totals use available data."
                )
            total = sum(getattr(run, field) for run in covered_runs)
            covered_questions = sum(len(run.samples) for run in covered_runs)
            average = total / covered_questions if covered_questions else None
            return total, average

        calls, avg_calls = aggregate("calls")
        prompt_tokens, _ = aggregate("prompt_tokens")
        output_tokens, _ = aggregate("output_tokens")
        total_tokens, avg_tokens = aggregate("total_tokens")
        model_latency, avg_model_latency = aggregate("model_latency")
        end_to_end_latency, avg_end_to_end_latency = aggregate("end_to_end_latency")
        provider_duration, avg_provider_duration = aggregate("provider_duration")
        wall_time, avg_wall_time = aggregate("wall_time")
        cost, _ = aggregate("cost")

        model_latencies = [
            sample.model_latency
            for run in group_runs
            for sample in run.samples
            if sample.model_latency is not None
        ]
        end_to_end_latencies = [
            sample.end_to_end_latency
            for run in group_runs
            for sample in run.samples
            if sample.end_to_end_latency is not None
        ]
        missing_sample_model_latencies = observations - len(model_latencies)
        if missing_sample_model_latencies and model_latencies:
            warnings.append(
                f"{benchmark}/{strategy}: model-latency percentiles use "
                f"{len(model_latencies)}/{observations} samples."
            )
        missing_sample_end_to_end_latencies = observations - len(end_to_end_latencies)
        if missing_sample_end_to_end_latencies and end_to_end_latencies:
            warnings.append(
                f"{benchmark}/{strategy}: end-to-end latency percentiles use "
                f"{len(end_to_end_latencies)}/{observations} samples."
            )

        ordered_model_latencies = sorted(model_latencies)
        ordered_end_to_end_latencies = sorted(end_to_end_latencies)
        p95_model_latency = (
            ordered_model_latencies[
                max(0, math.ceil(0.95 * len(ordered_model_latencies)) - 1)
            ]
            if ordered_model_latencies
            else None
        )
        p95_end_to_end_latency = (
            ordered_end_to_end_latencies[
                max(0, math.ceil(0.95 * len(ordered_end_to_end_latencies)) - 1)
            ]
            if ordered_end_to_end_latencies
            else None
        )
        counts = Counter(sample.outcome for run in group_runs for sample in run.samples)
        groups.append(
            Group(
                model=model,
                benchmark=benchmark,
                strategy=strategy,
                label=STRATEGY_LABELS.get(strategy, strategy.replace("_", " ").title()),
                runs=group_runs,
                unique_questions=len(
                    {sample.question_id for run in group_runs for sample in run.samples}
                ),
                evaluated_observations=observations,
                correct=counts["correct"],
                incorrect=counts["incorrect"],
                failed=counts["failed"],
                unparseable=counts["unparseable"],
                mean=statistics.fmean(scores) if scores else None,
                flexible_mean=(
                    statistics.fmean(flexible_scores) if flexible_scores else None
                ),
                std=statistics.pstdev(scores) if scores else None,
                minimum=min(scores) if scores else None,
                maximum=max(scores) if scores else None,
                calls=int(calls) if calls is not None else None,
                avg_calls=avg_calls,
                prompt_tokens=(
                    int(prompt_tokens) if prompt_tokens is not None else None
                ),
                output_tokens=(
                    int(output_tokens) if output_tokens is not None else None
                ),
                total_tokens=int(total_tokens) if total_tokens is not None else None,
                avg_tokens=avg_tokens,
                model_latency=model_latency,
                avg_model_latency=avg_model_latency,
                end_to_end_latency=end_to_end_latency,
                avg_end_to_end_latency=avg_end_to_end_latency,
                provider_duration=provider_duration,
                avg_provider_duration=avg_provider_duration,
                wall_time=wall_time,
                avg_wall_time=avg_wall_time,
                median_model_latency=(
                    statistics.median(model_latencies) if model_latencies else None
                ),
                p95_model_latency=p95_model_latency,
                missing_sample_model_latencies=missing_sample_model_latencies,
                median_end_to_end_latency=(
                    statistics.median(end_to_end_latencies)
                    if end_to_end_latencies
                    else None
                ),
                p95_end_to_end_latency=p95_end_to_end_latency,
                missing_sample_end_to_end_latencies=(
                    missing_sample_end_to_end_latencies
                ),
                cost=cost,
                partial_fields=tuple(partial_fields),
            )
        )
    return sorted(groups, key=lambda group: (group.model, group.benchmark, group.strategy))


def compare_strategies(runs: list[Run], groups: list[Group]) -> list[Comparison]:
    group_lookup = {
        (group.model, group.benchmark, group.strategy): group for group in groups
    }
    run_lookup = {
        (run.model, run.benchmark, run.strategy, run.repetition): run for run in runs
    }
    comparisons: list[Comparison] = []
    for group in groups:
        direct = group_lookup.get((group.model, group.benchmark, BASELINE_STRATEGY))
        if direct is None:
            continue
        direct_mean = direct.mean
        gain = (
            group.mean - direct_mean
            if group.mean is not None and direct_mean is not None
            else None
        )
        relative_gain = (
            gain / direct_mean * 100
            if gain is not None and direct_mean is not None and direct_mean != 0
            else None
        )
        relative_error_reduction = (
            gain / (1 - direct_mean) * 100
            if gain is not None
            and direct_mean is not None
            and direct_mean < 1
            else None
        )
        if gain is None:
            verdict = "N/A"
        elif math.isclose(gain, 0.0, abs_tol=1e-12):
            verdict = "tied"
        elif gain > 0:
            verdict = "improved"
        else:
            verdict = "worse"

        per_repetition: list[Counter[str]] = []
        if group.strategy != BASELINE_STRATEGY:
            for run in group.runs:
                direct_run = run_lookup.get(
                    (group.model, group.benchmark, BASELINE_STRATEGY, run.repetition)
                )
                if direct_run is None:
                    continue
                direct_answers = {
                    sample.question_id: sample.outcome == "correct"
                    for sample in direct_run.samples
                }
                strategy_answers = {
                    sample.question_id: sample.outcome == "correct"
                    for sample in run.samples
                }
                counts: Counter[str] = Counter()
                for question_id in direct_answers.keys() & strategy_answers.keys():
                    pair = (direct_answers[question_id], strategy_answers[question_id])
                    category = {
                        (False, True): "fixed",
                        (True, False): "regressed",
                        (True, True): "both_correct",
                        (False, False): "both_wrong",
                    }[pair]
                    counts[category] += 1
                    counts["matched"] += 1
                per_repetition.append(counts)

        def average(category: str) -> float | None:
            return (
                statistics.fmean(counts[category] for counts in per_repetition)
                if per_repetition
                else None
            )

        def average_rate(numerator: str, denominator: tuple[str, str]) -> float | None:
            rates = [
                counts[numerator] / (counts[denominator[0]] + counts[denominator[1]])
                for counts in per_repetition
                if counts[denominator[0]] + counts[denominator[1]] > 0
            ]
            return statistics.fmean(rates) if rates else None

        extra_calls = (
            group.avg_calls - direct.avg_calls
            if group.avg_calls is not None and direct.avg_calls is not None
            else None
        )
        extra_tokens = (
            group.avg_tokens - direct.avg_tokens
            if group.avg_tokens is not None and direct.avg_tokens is not None
            else None
        )
        extra_latency = (
            group.avg_model_latency - direct.avg_model_latency
            if group.avg_model_latency is not None
            and direct.avg_model_latency is not None
            else None
        )
        extra_end_to_end_latency = (
            group.avg_end_to_end_latency - direct.avg_end_to_end_latency
            if group.avg_end_to_end_latency is not None
            and direct.avg_end_to_end_latency is not None
            else None
        )
        extra_provider_duration = (
            group.avg_provider_duration - direct.avg_provider_duration
            if group.avg_provider_duration is not None
            and direct.avg_provider_duration is not None
            else None
        )
        gain_pp = gain * 100 if gain is not None else None
        comparisons.append(
            Comparison(
                model=group.model,
                benchmark=group.benchmark,
                strategy=group.strategy,
                label=group.label,
                direct_score=direct_mean,
                strategy_score=group.mean,
                gain=gain,
                relative_gain_percent=relative_gain,
                relative_error_reduction_percent=relative_error_reduction,
                verdict=verdict,
                compared_repetitions=len(per_repetition),
                fixed=average("fixed"),
                regressed=average("regressed"),
                both_correct=average("both_correct"),
                both_wrong=average("both_wrong"),
                matched=average("matched"),
                correction_rate=average_rate("fixed", ("fixed", "both_wrong")),
                degradation_rate=average_rate(
                    "regressed", ("regressed", "both_correct")
                ),
                extra_calls_per_question=extra_calls,
                extra_tokens_per_question=extra_tokens,
                extra_model_latency_per_question=extra_latency,
                extra_end_to_end_latency_per_question=extra_end_to_end_latency,
                extra_provider_duration_per_question=extra_provider_duration,
                gain_pp_per_additional_call=(
                    gain_pp / extra_calls
                    if gain_pp is not None
                    and extra_calls is not None
                    and extra_calls > 0
                    else None
                ),
                gain_pp_per_additional_1k_tokens=(
                    gain_pp / (extra_tokens / 1000)
                    if gain_pp is not None
                    and extra_tokens is not None
                    and extra_tokens > 0
                    else None
                ),
            )
        )
    return comparisons


def model_label(runs: list[Run]) -> str:
    models = sorted({run.model for run in runs})
    return models[0] if len(models) == 1 else f"{len(models)} models"


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def fmt_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def fmt_pp(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:+.1f} pp"


def fmt_number(value: float | int | None, decimals: int = 1) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int) or float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.{decimals}f}"


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value / 60:.1f} min" if value >= 60 else f"{value:.2f} s"


def render_table(
    headers: list[str], rows: list[list[str]], row_classes: list[str] | None = None
) -> str:
    header_html = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = []
    for index, row in enumerate(rows):
        row_class = row_classes[index] if row_classes else ""
        cells = "".join(f"<td>{cell}</td>" for cell in row)
        body.append(f'<tr class="{escape(row_class)}">{cells}</tr>')
    return f"""
<div class="table-wrap">
  <table>
    <thead><tr>{header_html}</tr></thead>
    <tbody>{"".join(body)}</tbody>
  </table>
</div>"""


def build_plot_data(groups: list[Group]) -> list[StrategyPlotData]:
    """Reduce analysis objects to the stable data contract used by the figures."""

    group_lookup = {
        (group.model, group.benchmark, group.strategy): group for group in groups
    }
    plot_data: list[StrategyPlotData] = []
    for group in groups:
        repetition_gains: list[float] = []
        if group.strategy != BASELINE_STRATEGY:
            direct = group_lookup.get(
                (group.model, group.benchmark, BASELINE_STRATEGY)
            )
            direct_runs = (
                {run.repetition: run for run in direct.runs}
                if direct is not None
                else {}
            )
            for run in group.runs:
                direct_run = direct_runs.get(run.repetition)
                if direct_run is None:
                    continue
                direct_scores = {
                    sample.question_id: sample.metric for sample in direct_run.samples
                }
                strategy_scores = {
                    sample.question_id: sample.metric for sample in run.samples
                }
                shared_ids = direct_scores.keys() & strategy_scores.keys()
                if shared_ids:
                    repetition_gains.append(
                        statistics.fmean(
                            strategy_scores[question_id] - direct_scores[question_id]
                            for question_id in shared_ids
                        )
                    )

        plot_data.append(
            StrategyPlotData(
                model=group.model,
                benchmark=group.benchmark,
                strategy=group.strategy,
                label=group.label,
                mean=group.mean,
                repetition_scores=tuple(
                    run.score for run in group.runs if run.score is not None
                ),
                gain_vs_direct=(
                    statistics.fmean(repetition_gains) if repetition_gains else None
                ),
                repetition_gains=tuple(repetition_gains),
                tokens_per_question=group.avg_tokens,
                end_to_end_latency_per_question=group.avg_end_to_end_latency,
            )
        )
    return plot_data


def render_academic_figures(artifacts: list[FigureArtifact]) -> str:
    if not artifacts:
        return ""
    cards = []
    for artifact in artifacts:
        stem = escape(artifact.stem)
        cards.append(
            f"""
<figure class="academic-figure">
  <a href="figures/{stem}.pdf" title="Open the thesis-ready PDF">
    <img src="figures/{stem}.svg" alt="{escape(artifact.alt_text)}" loading="lazy">
  </a>
  <figcaption><strong>{escape(artifact.title)}</strong><br>
    {escape(artifact.caption)}
    <span class="figure-links">Download:
      <a href="figures/{stem}.pdf">PDF</a> ·
      <a href="figures/{stem}.svg">SVG</a> ·
      <a href="figures/{stem}.png">PNG</a>
    </span>
  </figcaption>
</figure>"""
        )
    return f"""
<section id="academic-figures">
  <h2>Academic figures</h2>
  <p class="muted">These plots summarize only comparisons supported by this fixed
  experiment. Observed repetition ranges are descriptive and must not be read as
  confidence intervals.</p>
  <div class="figure-grid">{"".join(cards)}</div>
</section>"""


def render_reference(reference: dict[str, Any] | None) -> str:
    if reference is None:
        return """
<div class="reference">
  <strong>Official reference: N/A</strong><br>
  No verified exact model/benchmark reference is configured.
</div>"""
    return f"""
<div class="reference">
  <strong>Official reference: {fmt_percent(reference["score"])}</strong>
  — {escape(reference["model"])} on {escape(reference["benchmark"])}
  ({escape(reference["metric"])}).<br>
  <a href="{escape(reference["source"])}">Official score source</a> ·
  <a href="{escape(reference["model_identity_source"])}">Ollama model identity source</a>
  <p>{escape(reference["evaluation_notes"])}</p>
</div>"""


GSM8K_TARGET_PATTERN = re.compile(r"####\s*(-?\$?[0-9][0-9,]*(?:\.[0-9]+)?)")


def agent_responses(sample: Sample) -> list[dict[str, Any]]:
    if sample.strategy_result is None:
        return []
    responses = sample.strategy_result.get("agent_responses")
    return responses if isinstance(responses, list) else []


def expected_extracted_answer(sample: Sample, benchmark: str) -> str | None:
    if benchmark == "gsm8k":
        matches = GSM8K_TARGET_PATTERN.findall(str(sample.expected_answer))
        return matches[-1].replace("$", "").replace(",", "") if matches else None
    if benchmark == "arc_challenge_chat":
        return str(sample.expected_answer).strip().upper()
    if benchmark == "boolq" and sample.expected_answer in {0, 1}:
        return "yes" if sample.expected_answer else "no"
    return None


def answer_is_correct(sample: Sample, benchmark: str, answer: Any) -> bool:
    expected = expected_extracted_answer(sample, benchmark)
    return expected is not None and answer == expected


def fmt_rate(count: int, denominator: int) -> str:
    if denominator == 0:
        return "N/A"
    return f"{count}/{denominator} ({count / denominator:.1%})"


def render_self_consistency_analysis(groups: list[Group]) -> str:
    group = next(
        (group for group in groups if group.strategy == "self_consistency"), None
    )
    if group is None:
        return ""

    all_agree = clear_majority = ties = 0
    strengths: list[float] = []
    correct_strengths: list[float] = []
    incorrect_strengths: list[float] = []
    observations = 0
    for sample in (sample for run in group.runs for sample in run.samples):
        answers = [
            response.get("extracted_response")
            for response in agent_responses(sample)
            if response.get("extracted_response") is not None
        ]
        if not answers:
            continue
        observations += 1
        votes = Counter(answers)
        top_count = max(votes.values())
        strength = top_count / len(answers)
        strengths.append(strength)
        if len(votes) == 1:
            all_agree += 1
        elif list(votes.values()).count(top_count) == 1:
            clear_majority += 1
        else:
            ties += 1
        if sample.outcome == "correct":
            correct_strengths.append(strength)
        else:
            incorrect_strengths.append(strength)

    if observations == 0:
        return ""
    return """
  <h3>Self-Consistency voting</h3>
  <p class="muted">Agreement strength is the share of valid candidate answers in the most common vote.</p>
""" + render_table(
        [
            "Strategy",
            "Questions",
            "All candidates agree",
            "Clear non-unanimous majority",
            "Tied top vote",
            "Average agreement strength",
            "Agreement when final correct",
            "Agreement when final not correct",
        ],
        [
            [
                escape(group.label),
                fmt_number(observations),
                fmt_rate(all_agree, observations),
                fmt_rate(clear_majority, observations),
                fmt_rate(ties, observations),
                fmt_percent(statistics.fmean(strengths)),
                fmt_percent(
                    statistics.fmean(correct_strengths) if correct_strengths else None
                ),
                fmt_percent(
                    statistics.fmean(incorrect_strengths)
                    if incorrect_strengths
                    else None
                ),
            ]
        ],
    )


def render_society_of_minds_analysis(groups: list[Group], benchmark: str) -> str:
    group = next(
        (group for group in groups if group.strategy == "society_of_minds"), None
    )
    if group is None:
        return ""

    observations = initial_agreement = final_agreement = 0
    changed_agents = comparable_agents = 0
    moves_toward_final = final_answer_cases = 0
    minority_correct_spread = minority_correct_cases = 0
    for sample in (sample for run in group.runs for sample in run.samples):
        rounds: dict[int, dict[int, str | None]] = defaultdict(dict)
        for response in agent_responses(sample):
            agent_id = response.get("agent_id")
            round_id = response.get("round_id")
            if isinstance(agent_id, int) and isinstance(round_id, int):
                rounds[round_id][agent_id] = response.get("extracted_response")
        if 1 not in rounds or len(rounds) < 2:
            continue
        observations += 1
        initial = rounds[1]
        final = rounds[max(rounds)]
        initial_answers = [answer for answer in initial.values() if answer is not None]
        final_answers = [answer for answer in final.values() if answer is not None]
        if initial_answers and len(set(initial_answers)) == 1:
            initial_agreement += 1
        if final_answers and len(set(final_answers)) == 1:
            final_agreement += 1
        for agent_id in initial.keys() & final.keys():
            initial_answer = initial[agent_id]
            final_answer = final[agent_id]
            if initial_answer is not None and final_answer is not None:
                comparable_agents += 1
                changed_agents += initial_answer != final_answer

        strategy_answer = (
            sample.strategy_result.get("extracted_response")
            if sample.strategy_result is not None
            else None
        )
        if strategy_answer is not None:
            final_answer_cases += 1
            moves_toward_final += sum(
                answer == strategy_answer for answer in final_answers
            ) > sum(answer == strategy_answer for answer in initial_answers)

        initial_correct = sum(
            answer_is_correct(sample, benchmark, answer) for answer in initial_answers
        )
        final_correct = sum(
            answer_is_correct(sample, benchmark, answer) for answer in final_answers
        )
        if initial_answers and 0 < initial_correct * 2 < len(initial_answers):
            minority_correct_cases += 1
            minority_correct_spread += final_correct > initial_correct

    if observations == 0:
        return ""
    return """
  <h3>Society of Minds answer changes</h3>
  <p class="muted">A minority-correct answer is a correct initial answer held by fewer than half of the agents.</p>
""" + render_table(
        [
            "Strategy",
            "Questions with revisions",
            "Initial agreement",
            "Final agreement",
            "Agents changing answer",
            "Group moves toward final answer",
            "Minority-correct answer spreads",
        ],
        [
            [
                escape(group.label),
                fmt_number(observations),
                fmt_rate(initial_agreement, observations),
                fmt_rate(final_agreement, observations),
                fmt_rate(changed_agents, comparable_agents),
                fmt_rate(moves_toward_final, final_answer_cases),
                fmt_rate(minority_correct_spread, minority_correct_cases),
            ]
        ],
    )


def render_svj_analysis(groups: list[Group], benchmark: str) -> str:
    group = next(
        (group for group in groups if group.strategy == "role_based_svj"), None
    )
    if group is None:
        return ""

    counts: Counter[str] = Counter()
    changed_answers = comparable_answers = 0
    for sample in (sample for run in group.runs for sample in run.samples):
        solver = next(
            (
                response
                for response in agent_responses(sample)
                if response.get("agent_role") == "solver"
            ),
            None,
        )
        if solver is None:
            continue
        solver_answer = solver.get("extracted_response")
        solver_is_correct = answer_is_correct(sample, benchmark, solver_answer)
        final_correct = sample.outcome == "correct"
        counts[
            ("solver_correct" if solver_is_correct else "solver_wrong")
            + ("_final_correct" if final_correct else "_final_wrong")
        ] += 1
        final_answer = (
            sample.strategy_result.get("extracted_response")
            if sample.strategy_result is not None
            else None
        )
        if solver_answer is not None and final_answer is not None:
            comparable_answers += 1
            changed_answers += solver_answer != final_answer

    observations = sum(counts.values())
    if observations == 0:
        return ""
    solver_wrong = counts["solver_wrong_final_correct"] + counts["solver_wrong_final_wrong"]
    solver_correct = counts["solver_correct_final_correct"] + counts["solver_correct_final_wrong"]
    return """
  <h3>Solver-Verifier-Judge behaviour</h3>
""" + render_table(
        [
            "Strategy",
            "Solver correct -> final correct",
            "Solver wrong -> final correct",
            "Solver correct -> final wrong",
            "Solver wrong -> final wrong",
            "Correction rate",
            "Damage rate",
            "Final answer differs from Solver",
        ],
        [
            [
                escape(group.label),
                fmt_number(counts["solver_correct_final_correct"]),
                fmt_number(counts["solver_wrong_final_correct"]),
                fmt_number(counts["solver_correct_final_wrong"]),
                fmt_number(counts["solver_wrong_final_wrong"]),
                fmt_rate(counts["solver_wrong_final_correct"], solver_wrong),
                fmt_rate(counts["solver_correct_final_wrong"], solver_correct),
                fmt_rate(changed_answers, comparable_answers),
            ]
        ],
    )


def render_strategy_analyses(groups: list[Group], benchmark: str) -> str:
    return "".join(
        [
            render_self_consistency_analysis(groups),
            render_society_of_minds_analysis(groups, benchmark),
            render_svj_analysis(groups, benchmark),
        ]
    )


def render_benchmark_section(
    model: str,
    benchmark: str,
    groups: list[Group],
    comparisons: list[Comparison],
    reference: dict[str, Any] | None,
) -> str:
    comparison_lookup = {comparison.strategy: comparison for comparison in comparisons}
    filter_name, metric_name = metric_spec(benchmark)
    has_flexible_diagnostic = benchmark in FLEXIBLE_FILTERS

    performance_rows = []
    repetition_rows = []
    efficiency_rows = []
    for group in groups:
        comparison = comparison_lookup.get(group.strategy)
        performance_row = [
            escape(group.label),
            fmt_number(group.unique_questions),
            fmt_number(group.evaluated_observations),
            fmt_number(group.correct),
            fmt_number(group.incorrect),
            fmt_number(group.failed + group.unparseable),
            fmt_percent(group.mean),
        ]
        if has_flexible_diagnostic:
            performance_row.append(fmt_percent(group.flexible_mean))
        performance_row.append(fmt_pp(comparison.gain if comparison else None))
        performance_rows.append(performance_row)

        repetition_row = [
            escape(group.label),
            ", ".join(
                f"R{run.repetition}: {fmt_percent(run.score)}" for run in group.runs
            ),
        ]
        if has_flexible_diagnostic:
            repetition_row.extend(
                [
                ", ".join(
                    f"R{run.repetition}: {fmt_percent(run.flexible_score)}"
                    for run in group.runs
                ),
                fmt_percent(group.mean),
                fmt_percent(group.flexible_mean),
                ]
            )
        else:
            repetition_row.append(fmt_percent(group.mean))
        repetition_row.extend(
            [
                fmt_percent(group.std),
                fmt_percent(group.minimum),
                fmt_percent(group.maximum),
            ]
        )
        repetition_rows.append(repetition_row)
        efficiency_rows.append(
            [
                escape(group.label),
                fmt_number(group.avg_calls, 2),
                fmt_number(
                    group.prompt_tokens / group.evaluated_observations
                    if group.prompt_tokens is not None and group.evaluated_observations
                    else None,
                    1,
                ),
                fmt_number(
                    group.output_tokens / group.evaluated_observations
                    if group.output_tokens is not None and group.evaluated_observations
                    else None,
                    1,
                ),
                fmt_number(group.avg_tokens, 1),
                fmt_seconds(group.avg_end_to_end_latency),
                fmt_seconds(group.avg_provider_duration),
            ]
        )

    non_direct = [
        comparison
        for comparison in comparisons
        if comparison.strategy != BASELINE_STRATEGY
    ]
    question_rows = [
        [
            escape(comparison.label),
            fmt_number(comparison.fixed, 2),
            fmt_number(comparison.regressed, 2),
            fmt_number(comparison.both_correct, 2),
            fmt_number(comparison.both_wrong, 2),
            fmt_percent(comparison.correction_rate),
            fmt_percent(comparison.degradation_rate),
            fmt_number(
                comparison.fixed - comparison.regressed
                if comparison.fixed is not None and comparison.regressed is not None
                else None,
                2,
            ),
        ]
        for comparison in non_direct
    ]
    tradeoff_rows = [
        [
            escape(comparison.label),
            fmt_percent(comparison.direct_score),
            fmt_percent(comparison.strategy_score),
            fmt_pp(comparison.gain),
            (
                "N/A"
                if comparison.relative_gain_percent is None
                else f"{comparison.relative_gain_percent:+.2f}%"
            ),
            (
                "N/A"
                if comparison.relative_error_reduction_percent is None
                else f"{comparison.relative_error_reduction_percent:+.2f}%"
            ),
            fmt_number(comparison.extra_calls_per_question, 2),
            fmt_number(comparison.extra_tokens_per_question, 1),
            fmt_seconds(comparison.extra_end_to_end_latency_per_question),
            fmt_seconds(comparison.extra_provider_duration_per_question),
        ]
        for comparison in non_direct
    ]

    performance_headers = [
        "Strategy",
        "Unique Qs",
        "Observations",
        "Correct",
        "Incorrect",
        "Failed / unparseable",
        "Mean accuracy",
    ]
    repetition_headers = ["Strategy", "Strict score per repetition"]
    if has_flexible_diagnostic:
        performance_headers.append("Flexible accuracy (diagnostic)")
        repetition_headers.extend(
            [
            "Flexible score per repetition (diagnostic)",
            "Strict mean",
            "Flexible mean (diagnostic)",
            ]
        )
    else:
        repetition_headers.append("Strict mean")
    repetition_headers.extend(["Population std", "Min", "Max"])
    performance_table = render_table(
        performance_headers + ["Δ vs Direct"], performance_rows
    )
    repetition_table = (
        render_table(repetition_headers, repetition_rows)
        if any(len(group.runs) > 1 for group in groups)
        else "<p class=\"muted\">Repetition variability requires at least two repetitions.</p>"
    )
    question_table = (
        render_table(
            [
                "Strategy",
                "Direct wrong -> correct",
                "Direct correct -> wrong",
                "Correct in both",
                "Wrong in both",
                "Correction rate",
                "Degradation rate",
                "Net fixes",
            ],
            question_rows,
        )
        if question_rows
        else '<p class="muted">Question-level comparison: N/A</p>'
    )
    efficiency_table = render_table(
        [
            "Strategy",
            "Calls / Q",
            "Prompt tokens / Q",
            "Completion tokens / Q",
            "Tokens / Q",
            "End-to-end latency / Q",
            "Ollama duration / Q",
        ],
        efficiency_rows,
    )
    tradeoff_table = render_table(
        [
            "Strategy",
            "Direct accuracy",
            "Strategy accuracy",
            "Absolute improvement",
            "Relative accuracy gain",
            "Relative error reduction",
            "Δ calls / Q",
            "Δ tokens / Q",
            "Δ end-to-end latency / Q",
            "Δ Ollama duration / Q",
        ],
        tradeoff_rows,
    )

    return BENCHMARK_TEMPLATE.substitute(
        benchmark=escape(f"{model} — {benchmark}"),
        metric_name=escape(metric_name),
        filter_name=escape(filter_name),
        flexible_note=(
            "GSM8K also shows flexible extraction as a diagnostic; the strict "
            "saved metric remains the primary result."
            if benchmark in FLEXIBLE_FILTERS
            else ""
        ),
        reference=render_reference(reference),
        performance_table=performance_table,
        repetition_table=repetition_table,
        question_table=question_table,
        efficiency_table=efficiency_table,
        tradeoff_table=tradeoff_table,
        strategy_analysis=render_strategy_analyses(groups, benchmark),
    )


def render_report(
    experiment_label: str,
    output_dir: Path,
    models_label: str,
    groups: list[Group],
    comparisons: list[Comparison],
    warnings: list[str],
    references: dict[str, Any],
    figure_artifacts: list[FigureArtifact],
) -> str:
    benchmarks = sorted({group.benchmark for group in groups})
    models = sorted({group.model for group in groups})
    strategy_order = list(STRATEGY_LABELS)
    strategies = sorted(
        {group.strategy for group in groups},
        key=lambda strategy: (
            strategy_order.index(strategy)
            if strategy in strategy_order
            else len(strategy_order)
        ),
    )
    group_lookup = {
        (group.model, group.benchmark, group.strategy): group for group in groups
    }
    comparison_lookup = {
        (comparison.model, comparison.benchmark, comparison.strategy): comparison
        for comparison in comparisons
    }

    overall_rows = []
    overall_classes = []
    for model in models:
        for benchmark in benchmarks:
            benchmark_groups = [
                group
                for group in groups
                if group.model == model and group.benchmark == benchmark
            ]
            scores = [group.mean for group in benchmark_groups if group.mean is not None]
            best = max(scores) if scores else None
            for group in benchmark_groups:
                comparison = comparison_lookup.get((model, benchmark, group.strategy))
                overall_rows.append(
                    [
                        escape(model),
                        escape(benchmark),
                        escape(group.label),
                        fmt_percent(group.mean),
                        fmt_pp(comparison.gain if comparison else None),
                        fmt_seconds(group.wall_time),
                        fmt_number(group.calls),
                        fmt_number(group.total_tokens),
                    ]
                )
                overall_classes.append(
                    "best-row"
                    if best is not None
                    and group.mean is not None
                    and math.isclose(group.mean, best)
                    else ""
                )
    overall_table = render_table(
        [
            "Model",
            "Benchmark",
            "Strategy",
            "Mean accuracy",
            "Δ vs Direct",
            "Total wall time",
            "Model calls",
            "Tokens",
        ],
        overall_rows,
        overall_classes,
    )
    matrix_rows = []
    for model in models:
        for benchmark in benchmarks:
            row = [escape(model), escape(benchmark)]
            for strategy in strategies:
                matrix_group = group_lookup.get((model, benchmark, strategy))
                row.append(fmt_percent(matrix_group.mean) if matrix_group else "N/A")
            matrix_rows.append(row)
    matrix_table = render_table(
        ["Model", "Benchmark"]
        + [STRATEGY_LABELS.get(name, name) for name in strategies],
        matrix_rows,
    )

    benchmark_sections = "".join(
        render_benchmark_section(
            model,
            benchmark,
            [
                group
                for group in groups
                if group.model == model and group.benchmark == benchmark
            ],
            [
                comparison
                for comparison in comparisons
                if comparison.model == model and comparison.benchmark == benchmark
            ],
            references.get(model, {}).get(benchmark),
        )
        for model in models
        for benchmark in benchmarks
    )

    quality_rows = [
        [
            escape(group.model),
            escape(group.benchmark),
            escape(group.label),
            fmt_number(run.repetition),
            escape(run.status),
            fmt_number(len(run.samples)),
            fmt_number(run.counts["failed"]),
            fmt_number(run.counts["unparseable"]),
            fmt_percent(run.score),
        ]
        for group in groups
        for run in group.runs
    ]
    quality_table = render_table(
        [
            "Model",
            "Benchmark",
            "Strategy",
            "Repetition",
            "Status",
            "Evaluated",
            "Failed",
            "Unparseable",
            "Score",
        ],
        quality_rows,
    )
    warning_details = (
        '<div class="warning"><strong>Data-quality warnings</strong><ul>'
        + "".join(f"<li>{escape(warning)}</li>" for warning in warnings)
        + "</ul></div>"
        if warnings
        else '<div class="success">No data-quality warnings were found.</div>'
    )
    warning_summary = (
        '<div class="warning warning-summary"><strong>'
        f"{len(warnings)} data-quality warning(s).</strong> "
        '<a href="#data-quality">View details</a>.</div>'
        if warnings
        else '<div class="success">No data-quality warnings were found.</div>'
    )

    unique_questions = sum(
        max(
            group.unique_questions
            for group in groups
            if group.model == model and group.benchmark == benchmark
        )
        for model in models
        for benchmark in benchmarks
    )
    repetitions = max((len(group.runs) for group in groups), default=0)
    observations = sum(group.evaluated_observations for group in groups)
    return REPORT_TEMPLATE.substitute(
        experiment_id=escape(experiment_label),
        generated_at=escape(datetime.now(UTC).isoformat()),
        experiment_folder=escape(output_dir),
        folder_label=escape(output_dir.name),
        model=escape(models_label),
        model_card_label="Model" if len(models) == 1 else "Models",
        benchmark_count=len(benchmarks),
        strategy_count=len(strategies),
        repetitions=repetitions,
        unique_questions=unique_questions,
        observations=observations,
        warning_count=len(warnings),
        warning_summary=warning_summary,
        overall_table=overall_table,
        matrix_table=matrix_table,
        academic_figures=render_academic_figures(figure_artifacts),
        benchmark_sections=benchmark_sections,
        quality_table=quality_table,
        warning_details=warning_details,
    )


def export_group(group: Group) -> dict[str, Any]:
    data = asdict(group)
    data.pop("runs")
    data["per_run"] = [
        {
            "repetition": run.repetition,
            "status": run.status,
            "evaluated": len(run.samples),
            "correct": run.counts["correct"],
            "incorrect": run.counts["incorrect"],
            "failed": run.counts["failed"],
            "unparseable": run.counts["unparseable"],
            "score": run.score,
            "flexible_score": run.flexible_score,
            "calls": run.calls,
            "prompt_tokens": run.prompt_tokens,
            "output_tokens": run.output_tokens,
            "total_tokens": run.total_tokens,
            "model_latency": run.model_latency,
            "end_to_end_latency": run.end_to_end_latency,
            "provider_duration": run.provider_duration,
            "wall_time": run.wall_time,
            "cost": run.cost,
        }
        for run in group.runs
    ]
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze one or more existing experiment result folders."
    )
    parser.add_argument("experiment_folders", nargs="+", type=Path)
    args = parser.parse_args()
    experiment_dirs = [path.resolve() for path in args.experiment_folders]
    for experiment_dir in experiment_dirs:
        if not experiment_dir.is_dir():
            parser.error(f"not a directory: {experiment_dir}")

    try:
        experiment_label, runs, warnings = load_results(experiment_dirs)
        groups = calculate_statistics(runs, warnings)
        warnings = list(dict.fromkeys(warnings))
        comparisons = compare_strategies(runs, groups)
        models = model_label(runs)
        references = read_object(Path(__file__).with_name("analysis_references.json"))
    except AnalysisError as exc:
        parser.error(str(exc))

    output_dir = (
        experiment_dirs[0] / "analysis"
        if len(experiment_dirs) == 1
        else experiment_dirs[0].parent / "analysis-comparison"
    )
    output_dir.mkdir(exist_ok=True)
    figure_artifacts = generate_academic_figures(
        output_dir / "figures",
        build_plot_data(groups),
    )
    report_path = output_dir / "report.html"
    aggregate_path = output_dir / "aggregates.json"
    report_path.write_text(
        render_report(
            experiment_label,
            output_dir,
            models,
            groups,
            comparisons,
            warnings,
            references,
            figure_artifacts,
        ),
        encoding="utf-8",
    )
    official_references = []
    for model_name in sorted({group.model for group in groups}):
        for benchmark_name in sorted({group.benchmark for group in groups}):
            reference = references.get(model_name, {}).get(benchmark_name)
            if reference is not None:
                official_references.append(reference)
    aggregate_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": datetime.now(UTC).isoformat(),
                "experiment_folders": [str(path) for path in experiment_dirs],
                "model": models,
                "results": [export_group(group) for group in groups],
                "comparisons": [asdict(comparison) for comparison in comparisons],
                "official_references": official_references,
                "warnings": warnings,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Report: {report_path}")
    print(f"Aggregates: {aggregate_path}")
    print(f"Figures: {output_dir / 'figures'} ({len(figure_artifacts)})")
    print(f"Warnings: {len(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
