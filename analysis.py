"""Create a static HTML report for one saved experiment directory."""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from string import Template
from typing import Any

BASELINE_STRATEGY = "direct"
PRIMARY_FILTERS = {
    "gsm8k": "strict-match",
    "arc_challenge_chat": "remove_whitespace",
}
PRIMARY_METRICS = {
    "gsm8k": "exact_match",
    "arc_challenge_chat": "exact_match",
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
    outcome: str
    metric: float
    flexible_metric: float | None
    latency: float | None


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
    wall_time: float | None
    avg_wall_time: float | None
    median_latency: float | None
    p95_latency: float | None
    missing_sample_latencies: int
    cost: float | None
    partial_fields: tuple[str, ...]


@dataclass(slots=True)
class Comparison:
    benchmark: str
    strategy: str
    label: str
    direct_score: float | None
    strategy_score: float | None
    gain: float | None
    relative_gain_percent: float | None
    verdict: str
    compared_repetitions: int
    fixed: float | None
    regressed: float | None
    both_correct: float | None
    both_wrong: float | None
    matched: float | None
    extra_calls_per_question: float | None
    extra_tokens_per_question: float | None
    extra_model_latency_per_question: float | None
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


def parse_v1_sample(
    record: dict[str, Any], benchmark: str, where: str, warnings: list[str]
) -> Sample:
    question_id = required(record, "question_id", str, where)
    status = required(record, "status", str, where)
    expected = required(record, "reference_extracted_answer", str, where).strip()
    response = record.get("final_response")
    if response is not None and not isinstance(response, str):
        raise AnalysisError(f"{where}: 'final_response' must be a string or null.")
    predicted = record.get("final_extracted_answer")
    if predicted is not None and not isinstance(predicted, str):
        raise AnalysisError(
            f"{where}: 'final_extracted_answer' must be a string or null."
        )

    filter_name, metric_name = metric_spec(benchmark)
    correctness = required(record, "correctness", dict, where)
    evaluation = required(correctness, filter_name, dict, where)
    metrics = required(evaluation, "metrics", dict, where)
    metric = read_number(metrics, metric_name, where)
    assert metric is not None
    filtered_response = evaluation.get("filtered_response")

    if not expected:
        raise AnalysisError(f"{where}: expected answer is empty.")
    if (response is None or not response.strip()) and status != "failed":
        raise AnalysisError(f"{where}: predicted response is empty.")
    if predicted is not None:
        predicted = predicted.strip()
        if predicted == expected and metric < 1.0 - 1e-9:
            warnings.append(
                f"{where}: question {question_id!r} has matching saved extracted "
                f"answers but {filter_name}/{metric_name}={metric:g}; the benchmark "
                "metric is retained (usually an output-format difference)."
            )

    return Sample(
        question_id=question_id,
        outcome=parse_outcome(status, metric, filtered_response),
        metric=metric,
        flexible_metric=flexible_metric(correctness, benchmark, where),
        latency=read_number(record, "latency_seconds", where, required_field=False),
    )


def parse_runner_sample(record: dict[str, Any], benchmark: str, where: str) -> Sample:
    question_id = required(record, "question_id", str, where)
    status = required(record, "status", str, where)
    required(record, "expected_answer", str, where)
    response = record.get("lm_eval_response")
    if response is not None and not isinstance(response, str):
        raise AnalysisError(f"{where}: 'lm_eval_response' must be a string or null.")

    filter_name, metric_name = metric_spec(benchmark)
    evaluations = required(record, "evaluations", dict, where)
    evaluation = required(evaluations, filter_name, dict, where)
    metrics = required(evaluation, "metrics", dict, where)
    metric = read_number(metrics, metric_name, where)
    assert metric is not None

    latency = None
    calls = required(record, "calls", list, where)
    successful_results = [
        call["result"]
        for call in calls
        if isinstance(call, dict) and isinstance(call.get("result"), dict)
    ]
    if successful_results:
        latency = read_number(
            successful_results[-1], "total_latency_s", where, required_field=False
        )

    return Sample(
        question_id=question_id,
        outcome=parse_outcome(status, metric, evaluation.get("response")),
        metric=metric,
        flexible_metric=flexible_metric(evaluations, benchmark, where),
        latency=latency,
    )


def parse_samples(
    path: Path,
    experiment_dir: Path,
    schema: str,
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
        sample = (
            parse_v1_sample(record, benchmark, where, warnings)
            if schema == "v1"
            else parse_runner_sample(record, benchmark, where)
        )
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
    schema: str,
    benchmark_tasks: dict[str, str],
    warnings: list[str],
) -> Run:
    raw = read_object(run_path)
    where = str(run_path.relative_to(experiment_dir))
    benchmark = required(raw, "benchmark", str, where)
    strategy = required(raw, "strategy", str, where)
    if benchmark not in benchmark_tasks:
        raise AnalysisError(f"{where}: unknown benchmark {benchmark!r}.")
    repetition = read_int(raw, "repetition", where)
    assert repetition is not None
    status = required(raw, "status", str, where)
    model = required(raw, "model", str, where)

    sample_path = run_path.parent / "samples.jsonl"
    if not sample_path.is_file():
        raise AnalysisError(f"{where}: missing samples.jsonl.")
    samples = parse_samples(sample_path, experiment_dir, schema, benchmark, warnings)
    saved_sample_count = read_int(raw, "sample_count", where)
    if saved_sample_count != len(samples):
        raise AnalysisError(
            f"{where}: sample_count={saved_sample_count}, found {len(samples)} samples."
        )

    filter_name, metric_name = metric_spec(benchmark)
    metric_key = f"{metric_name},{filter_name}"
    if status == "failed":
        saved_score = None
    elif schema == "v1":
        metrics = required(raw, "metrics", dict, where)
        saved_score = read_number(metrics, metric_key, where)
    else:
        evaluation = required(raw, "evaluation", dict, where)
        results = required(evaluation, "results", dict, where)
        task_name = benchmark_tasks[benchmark]
        task_results = required(results, task_name, dict, where)
        saved_score = read_number(task_results, metric_key, where)

    usage_key = "token_usage" if schema == "v1" else "tokens"
    usage = required(raw, usage_key, dict, where)
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
        wall_time=read_number(raw, "wall_time_seconds", where, required_field=False),
        cost=read_number(raw, "cost", where, required_field=False),
    )
    if status != "completed":
        warnings.append(f"{where}: run status is {status!r}.")
    if (
        run.score is not None
        and saved_score is not None
        and not math.isclose(run.score, saved_score, abs_tol=1e-9)
    ):
        raise AnalysisError(
            f"{where}: saved score {saved_score:g} differs from sample score "
            f"{run.score:g}."
        )
    return run


def load_results(
    experiment_dir: Path,
) -> tuple[dict[str, Any], list[Run], list[str]]:
    metadata = read_object(experiment_dir / "experiment.json")
    schema_version = metadata.get("schema_version")
    # Existing result folders use v1; the repository's current runner is unversioned.
    if type(schema_version) is int and schema_version == 1:
        schema = "v1"
    elif schema_version is None:
        schema = "runner"
    else:
        raise AnalysisError(f"Unsupported schema_version: {schema_version!r}.")

    configuration = required(metadata, "configuration", dict, "experiment.json")
    benchmark_config = required(configuration, "benchmarks", dict, "experiment.json")
    benchmark_tasks = {
        name: required(config, "task", str, f"experiment.json benchmark {name}")
        for name, config in benchmark_config.items()
        if isinstance(config, dict)
    }
    if len(benchmark_tasks) != len(benchmark_config):
        raise AnalysisError(
            "experiment.json: every benchmark config must be an object."
        )

    run_paths = sorted(experiment_dir.rglob("run.json"))
    if not run_paths:
        raise AnalysisError("No run.json files were found.")
    warnings: list[str] = []
    runs = [
        parse_run(path, experiment_dir, schema, benchmark_tasks, warnings)
        for path in run_paths
    ]

    identities = [(run.benchmark, run.strategy, run.repetition) for run in runs]
    if len(identities) != len(set(identities)):
        duplicates = [item for item, count in Counter(identities).items() if count > 1]
        raise AnalysisError(f"Duplicate run identities: {duplicates}.")
    experiment_id = required(metadata, "experiment_id", str, "experiment.json")
    for run in runs:
        if run.experiment_id != experiment_id:
            raise AnalysisError(f"{run.path}: experiment_id does not match metadata.")

    validate_result_set(metadata, runs, warnings)
    return metadata, runs, list(dict.fromkeys(warnings))


def validate_result_set(
    metadata: dict[str, Any], runs: list[Run], warnings: list[str]
) -> None:
    configuration = required(metadata, "configuration", dict, "experiment.json")
    matrix = required(configuration, "matrix", dict, "experiment.json")
    run_config = required(configuration, "run", dict, "experiment.json")
    benchmarks = required(matrix, "benchmarks", list, "experiment.json")
    strategies = required(matrix, "strategies", list, "experiment.json")
    repetitions = read_int(run_config, "repetitions", "experiment.json")
    assert repetitions is not None

    identities = {(run.benchmark, run.strategy, run.repetition) for run in runs}
    for benchmark in benchmarks:
        for strategy in strategies:
            for repetition in range(1, repetitions + 1):
                if (benchmark, strategy, repetition) not in identities:
                    warnings.append(
                        f"Missing configured run: {benchmark}/{strategy}/"
                        f"run-{repetition:03d}."
                    )

    runs_by_benchmark_repetition: dict[tuple[str, int], list[Run]] = defaultdict(list)
    for run in runs:
        runs_by_benchmark_repetition[(run.benchmark, run.repetition)].append(run)
    for (benchmark, repetition), matching_runs in runs_by_benchmark_repetition.items():
        baseline = next(
            (run for run in matching_runs if run.strategy == BASELINE_STRATEGY),
            matching_runs[0],
        )
        baseline_ids = {sample.question_id for sample in baseline.samples}
        for run in matching_runs:
            ids = {sample.question_id for sample in run.samples}
            if ids != baseline_ids:
                warnings.append(
                    f"{benchmark} repetition {repetition}: {run.strategy} has "
                    f"{len(ids)} question IDs, while {baseline.strategy} has "
                    f"{len(baseline_ids)}."
                )

    unique_questions = {
        (run.benchmark, sample.question_id) for run in runs for sample in run.samples
    }
    if len(unique_questions) < 10:
        warnings.append(
            f"Only {len(unique_questions)} unique benchmark question(s) were evaluated; "
            "accuracy and strategy differences are not statistically meaningful."
        )


def calculate_statistics(runs: list[Run], warnings: list[str]) -> list[Group]:
    grouped: dict[tuple[str, str], list[Run]] = defaultdict(list)
    for run in runs:
        grouped[(run.benchmark, run.strategy)].append(run)

    groups: list[Group] = []
    for (benchmark, strategy), group_runs in grouped.items():
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
        wall_time, avg_wall_time = aggregate("wall_time")
        cost, _ = aggregate("cost")

        sample_latencies = [
            sample.latency
            for run in group_runs
            for sample in run.samples
            if sample.latency is not None
        ]
        missing_sample_latencies = observations - len(sample_latencies)
        if missing_sample_latencies and sample_latencies:
            warnings.append(
                f"{benchmark}/{strategy}: latency percentiles use "
                f"{len(sample_latencies)}/{observations} samples."
            )
        ordered_latencies = sorted(sample_latencies)
        p95_latency = (
            ordered_latencies[max(0, math.ceil(0.95 * len(ordered_latencies)) - 1)]
            if ordered_latencies
            else None
        )
        counts = Counter(sample.outcome for run in group_runs for sample in run.samples)
        groups.append(
            Group(
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
                wall_time=wall_time,
                avg_wall_time=avg_wall_time,
                median_latency=(
                    statistics.median(sample_latencies) if sample_latencies else None
                ),
                p95_latency=p95_latency,
                missing_sample_latencies=missing_sample_latencies,
                cost=cost,
                partial_fields=tuple(partial_fields),
            )
        )
    return sorted(groups, key=lambda group: (group.benchmark, group.strategy))


def compare_strategies(runs: list[Run], groups: list[Group]) -> list[Comparison]:
    group_lookup = {(group.benchmark, group.strategy): group for group in groups}
    run_lookup = {(run.benchmark, run.strategy, run.repetition): run for run in runs}
    comparisons: list[Comparison] = []
    for group in groups:
        direct = group_lookup.get((group.benchmark, BASELINE_STRATEGY))
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
                    (group.benchmark, BASELINE_STRATEGY, run.repetition)
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
        gain_pp = gain * 100 if gain is not None else None
        comparisons.append(
            Comparison(
                benchmark=group.benchmark,
                strategy=group.strategy,
                label=group.label,
                direct_score=direct_mean,
                strategy_score=group.mean,
                gain=gain,
                relative_gain_percent=relative_gain,
                verdict=verdict,
                compared_repetitions=len(per_repetition),
                fixed=average("fixed"),
                regressed=average("regressed"),
                both_correct=average("both_correct"),
                both_wrong=average("both_wrong"),
                matched=average("matched"),
                extra_calls_per_question=extra_calls,
                extra_tokens_per_question=extra_tokens,
                extra_model_latency_per_question=extra_latency,
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


def model_from_metadata(metadata: dict[str, Any]) -> str:
    configuration = required(metadata, "configuration", dict, "experiment.json")
    application = required(configuration, "application", dict, "experiment.json")
    provider = required(application, "provider", dict, "experiment.json")
    return required(provider, "model", str, "experiment.json")


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


def render_bars(groups: list[Group]) -> str:
    scored = [group for group in groups if group.mean is not None]
    if not scored:
        return '<p class="muted">Accuracy chart: N/A</p>'
    best = max(group.mean for group in scored if group.mean is not None)
    rows = []
    for group in scored:
        assert group.mean is not None
        width = max(0.0, min(100.0, group.mean * 100))
        best_class = "best" if math.isclose(group.mean, best) else ""
        rows.append(
            f"""
<div class="bar-row">
  <div class="bar-label">{escape(group.label)}</div>
  <div class="bar-track"><div class="bar {best_class}" style="width:{width:.3f}%"></div></div>
  <div class="bar-value">{fmt_percent(group.mean)}</div>
</div>"""
        )
    return f'<div class="bar-chart">{"".join(rows)}</div>'


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


def render_benchmark_section(
    benchmark: str,
    groups: list[Group],
    comparisons: list[Comparison],
    reference: dict[str, Any] | None,
) -> str:
    comparison_lookup = {comparison.strategy: comparison for comparison in comparisons}
    filter_name, metric_name = metric_spec(benchmark)

    performance_rows = []
    repetition_rows = []
    efficiency_rows = []
    for group in groups:
        comparison = comparison_lookup.get(group.strategy)
        performance_rows.append(
            [
                escape(group.label),
                fmt_number(group.unique_questions),
                fmt_number(group.evaluated_observations),
                fmt_number(group.correct),
                fmt_number(group.incorrect),
                fmt_number(group.failed + group.unparseable),
                fmt_percent(group.mean),
                fmt_percent(group.flexible_mean),
                fmt_pp(comparison.gain if comparison else None),
            ]
        )
        repetition_rows.append(
            [
                escape(group.label),
                ", ".join(
                    f"R{run.repetition}: {fmt_percent(run.score)}" for run in group.runs
                ),
                ", ".join(
                    f"R{run.repetition}: {fmt_percent(run.flexible_score)}"
                    for run in group.runs
                ),
                fmt_percent(group.mean),
                fmt_percent(group.flexible_mean),
                fmt_percent(group.std),
                fmt_percent(group.minimum),
                fmt_percent(group.maximum),
            ]
        )
        efficiency_rows.append(
            [
                escape(group.label),
                fmt_number(group.calls),
                fmt_number(group.avg_calls, 2),
                fmt_number(group.prompt_tokens),
                fmt_number(group.output_tokens),
                fmt_number(group.total_tokens),
                fmt_number(group.avg_tokens, 1),
                fmt_seconds(group.wall_time),
                fmt_seconds(group.avg_wall_time),
                fmt_seconds(group.median_latency),
                fmt_seconds(group.p95_latency),
                "N/A" if group.cost is None else fmt_number(group.cost, 4),
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
            fmt_pp(comparison.gain),
            (
                "N/A"
                if comparison.relative_gain_percent is None
                else f"{comparison.relative_gain_percent:+.2f}%"
            ),
            escape(comparison.verdict),
            fmt_number(comparison.extra_calls_per_question, 2),
            fmt_number(comparison.extra_tokens_per_question, 1),
            fmt_seconds(comparison.extra_model_latency_per_question),
            fmt_number(comparison.gain_pp_per_additional_call, 3),
            fmt_number(comparison.gain_pp_per_additional_1k_tokens, 3),
        ]
        for comparison in non_direct
    ]

    performance_table = render_table(
        [
            "Strategy",
            "Unique Qs",
            "Observations",
            "Correct",
            "Incorrect",
            "Failed / unparseable",
            "Mean accuracy",
            "Flexible accuracy (diagnostic)",
            "Δ vs Direct",
        ],
        performance_rows,
    )
    repetition_table = render_table(
        [
            "Strategy",
            "Strict score per repetition",
            "Flexible score per repetition (diagnostic)",
            "Strict mean",
            "Flexible mean (diagnostic)",
            "Population std",
            "Min",
            "Max",
        ],
        repetition_rows,
    )
    question_table = (
        render_table(
            [
                "Strategy",
                "Direct wrong → correct",
                "Direct correct → wrong",
                "Correct in both",
                "Wrong in both",
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
            "Calls",
            "Calls / Q",
            "Prompt tokens",
            "Completion tokens",
            "Total tokens",
            "Tokens / Q",
            "Wall time",
            "Wall / Q",
            "Median model latency",
            "P95 model latency",
            "Cost",
        ],
        efficiency_rows,
    )
    tradeoff_table = render_table(
        [
            "Strategy",
            "Accuracy gain",
            "Relative gain",
            "Result",
            "Extra calls / Q",
            "Extra tokens / Q",
            "Extra model latency / Q",
            "Gain pp / extra call",
            "Gain pp / extra 1K tokens",
        ],
        tradeoff_rows,
    )

    return BENCHMARK_TEMPLATE.substitute(
        benchmark=escape(benchmark),
        metric_name=escape(metric_name),
        filter_name=escape(filter_name),
        flexible_note=(
            "GSM8K also shows flexible extraction as a diagnostic; the strict "
            "saved metric remains the primary result."
            if benchmark in FLEXIBLE_FILTERS
            else ""
        ),
        accuracy_bars=render_bars(groups),
        reference=render_reference(reference),
        performance_table=performance_table,
        repetition_table=repetition_table,
        question_table=question_table,
        efficiency_table=efficiency_table,
        tradeoff_table=tradeoff_table,
    )


def render_report(
    experiment_dir: Path,
    metadata: dict[str, Any],
    model: str,
    groups: list[Group],
    comparisons: list[Comparison],
    warnings: list[str],
    references: dict[str, Any],
) -> str:
    experiment_id = required(metadata, "experiment_id", str, "experiment.json")
    benchmarks = sorted({group.benchmark for group in groups})
    strategy_order = list(STRATEGY_LABELS)
    strategies = sorted(
        {group.strategy for group in groups},
        key=lambda strategy: (
            strategy_order.index(strategy)
            if strategy in strategy_order
            else len(strategy_order)
        ),
    )
    group_lookup = {(group.benchmark, group.strategy): group for group in groups}
    comparison_lookup = {
        (comparison.benchmark, comparison.strategy): comparison
        for comparison in comparisons
    }

    overall_rows = []
    overall_classes = []
    for benchmark in benchmarks:
        benchmark_groups = [group for group in groups if group.benchmark == benchmark]
        scores = [group.mean for group in benchmark_groups if group.mean is not None]
        best = max(scores) if scores else None
        for group in benchmark_groups:
            comparison = comparison_lookup.get((benchmark, group.strategy))
            overall_rows.append(
                [
                    escape(benchmark),
                    escape(group.label),
                    fmt_percent(group.mean),
                    fmt_percent(group.flexible_mean),
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
            "Benchmark",
            "Strategy",
            "Mean accuracy",
            "Flexible accuracy (diagnostic)",
            "Δ vs Direct",
            "Total wall time",
            "Model calls",
            "Tokens",
        ],
        overall_rows,
        overall_classes,
    )
    matrix_rows = []
    for benchmark in benchmarks:
        row = [escape(benchmark)]
        for strategy in strategies:
            matrix_group = group_lookup.get((benchmark, strategy))
            row.append(fmt_percent(matrix_group.mean) if matrix_group else "N/A")
        matrix_rows.append(row)
    matrix_table = render_table(
        ["Benchmark"] + [STRATEGY_LABELS.get(name, name) for name in strategies],
        matrix_rows,
    )

    model_references = references.get(model, {})
    benchmark_sections = "".join(
        render_benchmark_section(
            benchmark,
            [group for group in groups if group.benchmark == benchmark],
            [
                comparison
                for comparison in comparisons
                if comparison.benchmark == benchmark
            ],
            model_references.get(benchmark),
        )
        for benchmark in benchmarks
    )

    quality_rows = [
        [
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
        max(group.unique_questions for group in groups if group.benchmark == benchmark)
        for benchmark in benchmarks
    )
    repetitions = max((len(group.runs) for group in groups), default=0)
    observations = sum(group.evaluated_observations for group in groups)
    folder_label = str(Path(experiment_dir.parent.name) / experiment_dir.name)

    return REPORT_TEMPLATE.substitute(
        experiment_id=escape(experiment_id),
        generated_at=escape(datetime.now(UTC).isoformat()),
        experiment_folder=escape(experiment_dir),
        folder_label=escape(folder_label),
        model=escape(model),
        benchmark_count=len(benchmarks),
        strategy_count=len(strategies),
        repetitions=repetitions,
        unique_questions=unique_questions,
        observations=observations,
        warning_count=len(warnings),
        warning_summary=warning_summary,
        overall_table=overall_table,
        matrix_table=matrix_table,
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
            "wall_time": run.wall_time,
            "cost": run.cost,
        }
        for run in group.runs
    ]
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze one existing experiment result folder."
    )
    parser.add_argument("experiment_folder", type=Path)
    args = parser.parse_args()
    experiment_dir = args.experiment_folder.resolve()
    if not experiment_dir.is_dir():
        parser.error(f"not a directory: {experiment_dir}")

    try:
        metadata, runs, warnings = load_results(experiment_dir)
        groups = calculate_statistics(runs, warnings)
        warnings = list(dict.fromkeys(warnings))
        comparisons = compare_strategies(runs, groups)
        model = model_from_metadata(metadata)
        if {run.model for run in runs} != {model}:
            raise AnalysisError(
                "Run model fields do not consistently match the experiment metadata."
            )
        references = read_object(Path(__file__).with_name("analysis_references.json"))
    except AnalysisError as exc:
        parser.error(str(exc))

    output_dir = experiment_dir / "analysis"
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / "report.html"
    aggregate_path = output_dir / "aggregates.json"
    report_path.write_text(
        render_report(
            experiment_dir,
            metadata,
            model,
            groups,
            comparisons,
            warnings,
            references,
        ),
        encoding="utf-8",
    )
    aggregate_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": datetime.now(UTC).isoformat(),
                "experiment_folder": str(experiment_dir),
                "model": model,
                "results": [export_group(group) for group in groups],
                "comparisons": [asdict(comparison) for comparison in comparisons],
                "official_references": [
                    reference
                    for benchmark in sorted({group.benchmark for group in groups})
                    if (reference := references.get(model, {}).get(benchmark))
                    is not None
                ],
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
    print(f"Warnings: {len(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
