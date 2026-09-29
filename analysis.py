from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from string import Template
from typing import Any

import numpy as np
import scipy  # type: ignore[import-untyped]
import statsmodels  # type: ignore[import-untyped]
from scipy.stats import bootstrap  # type: ignore[import-untyped]
from statsmodels.stats.contingency_tables import mcnemar  # type: ignore[import-untyped]

from analysis_behavior import calculate_strategy_behavior
from analysis_plots import (
    BENCHMARK_LABELS,
    BENCHMARK_ORDER,
    STRATEGY_ORDER,
    FigureArtifact,
    StrategyPlotData,
    generate_academic_figures,
    pareto_strategies,
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
STRATEGY_LABELS = {
    "direct": "Direct",
    "self_consistency": "Self-Consistency",
    "society_of_minds": "Society of Minds",
    "role_based_svj": "SVJ",
}

MODEL_LABELS = {"qwen2.5:3b": "Qwen2.5-3B", "llama3.2:3b": "Llama 3.2 3B"}


def model_order(name: str) -> tuple[int, str]:
    return (
        list(MODEL_LABELS).index(name) if name in MODEL_LABELS else len(MODEL_LABELS),
        name,
    )


def benchmark_order(name: str) -> tuple[int, str]:
    return (
        BENCHMARK_ORDER.index(name)
        if name in BENCHMARK_ORDER
        else len(BENCHMARK_ORDER),
        name,
    )


def strategy_order(name: str) -> tuple[int, str]:
    return (
        STRATEGY_ORDER.index(name) if name in STRATEGY_ORDER else len(STRATEGY_ORDER),
        name,
    )


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
    strategy_result: dict[str, Any] | None
    document: dict[str, Any] | None = None
    document_hash: str | None = None
    target_hash: str | None = None


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
    calls: int | None
    prompt_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    end_to_end_latency: float | None
    seed: int | None = None

    @property
    def counts(self) -> Counter[str]:
        return Counter(sample.outcome for sample in self.samples)

    @property
    def score(self) -> float | None:
        return (
            statistics.fmean(sample.metric for sample in self.samples)
            if self.samples
            else None
        )


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
    std: float | None
    calls: int | None
    avg_calls: float | None
    prompt_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    avg_tokens: float | None
    end_to_end_latency: float | None
    avg_end_to_end_latency: float | None
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
    verdict: str
    latency_ratio: float | None


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
    filter_name, metric_name = metric_spec(benchmark)
    evaluations = required(record, "evaluations", dict, where)
    evaluation = required(evaluations, filter_name, dict, where)
    metric = read_number(
        required(evaluation, "metrics", dict, where), metric_name, where
    )
    if metric not in (0.0, 1.0):
        raise AnalysisError(f"{where}: the primary score must be binary (0 or 1).")
    outcome = parse_outcome(status, metric, evaluation.get("response"))
    if outcome in {"failed", "unparseable"} and metric != 0.0:
        raise AnalysisError(
            f"{where}: a failed or unparseable answer has a nonzero score."
        )
    document = record.get("document")
    if document is not None and not isinstance(document, dict):
        raise AnalysisError(f"{where}: 'document' must be an object or null.")
    hashes = record.get("hashes") or {}
    if not isinstance(hashes, dict):
        raise AnalysisError(f"{where}: 'hashes' must be an object or null.")
    calls = required(record, "calls", list, where)
    successful_results = [
        call["result"]
        for call in calls
        if isinstance(call, dict) and isinstance(call.get("result"), dict)
    ]
    strategy_result = None
    if successful_results:
        result = successful_results[-1]
        # Keep the intermediate fields needed for strategy comparisons.
        strategy_result = {
            key: result[key]
            for key in ("extracted_response", "initial_extracted_response")
            if key in result
        }
        responses = result.get("agent_responses", [])
        if not isinstance(responses, list) or any(
            not isinstance(r, dict) for r in responses
        ):
            raise AnalysisError(f"{where}: agent_responses must be a list of objects.")
        strategy_result["agent_responses"] = [
            {
                key: response.get(key)
                for key in (
                    "agent_id",
                    "round_id",
                    "agent_role",
                    "extracted_response",
                    "response",
                )
            }
            for response in responses
        ]
    return Sample(
        question_id=question_id,
        expected_answer=expected_answer,
        outcome=outcome,
        metric=metric,
        strategy_result=strategy_result,
        document=document,
        document_hash=hashes.get("document"),
        target_hash=hashes.get("target"),
    )


def parse_samples(
    path: Path, experiment_dir: Path, benchmark: str, warnings: list[str]
) -> list[Sample]:
    samples: list[Sample] = []
    seen_questions: set[str] = set()
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
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
    except (OSError, UnicodeError) as exc:
        raise AnalysisError(f"Could not read {path}: {exc}") from exc
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
        calls=read_int(raw, "model_call_count", where, required_field=False),
        prompt_tokens=read_int(usage, "prompt", where, required_field=False),
        output_tokens=read_int(usage, "output", where, required_field=False),
        total_tokens=read_int(usage, "total", where, required_field=False),
        end_to_end_latency=read_number(
            raw, "end_to_end_latency_seconds", where, required_field=False
        ),
        seed=read_int(raw, "seed", where, required_field=False),
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

    identities = [
        (run.model, run.benchmark, run.strategy, run.repetition) for run in runs
    ]
    if len(identities) != len(set(identities)):
        duplicates = [item for item, count in Counter(identities).items() if count > 1]
        raise AnalysisError(f"Duplicate run identities: {duplicates}.")

    validate_result_set(runs, warnings)
    label = (
        experiment_dirs[0].name
        if len(experiment_dirs) == 1
        else f"{len(experiment_dirs)} result folders"
    )
    return label, runs, list(dict.fromkeys(warnings))


def validate_result_set(runs: list[Run], warnings: list[str]) -> None:
    runs_by_benchmark_repetition: dict[tuple[str, str, int], list[Run]] = defaultdict(
        list
    )
    for run in runs:
        runs_by_benchmark_repetition[(run.model, run.benchmark, run.repetition)].append(
            run
        )
    for (
        model,
        benchmark,
        repetition,
    ), matching_runs in runs_by_benchmark_repetition.items():
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
    groups = []
    for (model, benchmark, strategy), group_runs in grouped.items():
        group_runs.sort(key=lambda run: run.repetition)
        observations = sum(len(run.samples) for run in group_runs)
        scores = [run.score for run in group_runs if run.score is not None]
        partial_fields: list[str] = []

        def aggregate(field: str) -> tuple[Any, float | None]:
            if any(getattr(run, field) is None for run in group_runs):
                partial_fields.append(field)
                warnings.append(
                    f"{model}/{benchmark}/{strategy}: incomplete {field}; aggregate omitted."
                )
                return None, None
            total = sum(getattr(run, field) for run in group_runs)
            return total, total / observations if observations else None

        calls, avg_calls = aggregate("calls")
        prompt_tokens, _ = aggregate("prompt_tokens")
        output_tokens, _ = aggregate("output_tokens")
        total_tokens, avg_tokens = aggregate("total_tokens")
        latency, avg_latency = aggregate("end_to_end_latency")
        counts = Counter(sample.outcome for run in group_runs for sample in run.samples)
        groups.append(
            Group(
                model=model,
                benchmark=benchmark,
                strategy=strategy,
                label=STRATEGY_LABELS.get(strategy, strategy),
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
                std=statistics.stdev(scores) if len(scores) > 1 else None,
                calls=calls,
                avg_calls=avg_calls,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                avg_tokens=avg_tokens,
                end_to_end_latency=latency,
                avg_end_to_end_latency=avg_latency,
                partial_fields=tuple(partial_fields),
            )
        )
    return sorted(
        groups,
        key=lambda group: (
            model_order(group.model),
            benchmark_order(group.benchmark),
            strategy_order(group.strategy),
        ),
    )


def compare_strategies(groups: list[Group]) -> list[Comparison]:
    lookup = {(group.model, group.benchmark, group.strategy): group for group in groups}
    comparisons = []
    for group in groups:
        direct = lookup.get((group.model, group.benchmark, BASELINE_STRATEGY))
        if direct is None:
            continue
        gain = (
            group.mean - direct.mean
            if group.mean is not None and direct.mean is not None
            else None
        )
        verdict = (
            "N/A"
            if gain is None
            else "tied"
            if math.isclose(gain, 0, abs_tol=1e-12)
            else "improved"
            if gain > 0
            else "worse"
        )
        latency_ratio = (
            group.avg_end_to_end_latency / direct.avg_end_to_end_latency
            if group.avg_end_to_end_latency is not None
            and direct.avg_end_to_end_latency
            else None
        )
        comparisons.append(
            Comparison(
                group.model,
                group.benchmark,
                group.strategy,
                group.label,
                direct.mean,
                group.mean,
                gain,
                verdict,
                latency_ratio,
            )
        )
    return comparisons


def paired_accuracy_difference(
    direct: np.ndarray, strategy: np.ndarray, axis: int = -1
) -> np.ndarray:
    """Accuracy difference in percentage points, retaining question pairs."""
    return 100 * (np.mean(strategy, axis=axis) - np.mean(direct, axis=axis))


def paired_confidence_interval(
    direct: np.ndarray,
    strategy: np.ndarray,
    identity: tuple[str, str, str, int],
) -> dict[str, Any]:
    """Return a reproducible 95% BCa interval for one paired comparison."""
    differences = strategy - direct
    if differences.size < 2 or np.all(differences == differences[0]):
        raise AnalysisError(
            f"Cannot estimate a BCa interval for constant paired differences: {identity}"
        )
    seed = int.from_bytes(
        hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).digest()[
            :8
        ],
        "little",
    )
    result = bootstrap(
        (direct, strategy),
        paired_accuracy_difference,
        paired=True,
        vectorized=True,
        n_resamples=20_000,
        batch=256,
        confidence_level=0.95,
        method="BCa",
        rng=np.random.default_rng(seed),
    )
    low, high = (
        float(result.confidence_interval.low),
        float(result.confidence_interval.high),
    )
    if not (math.isfinite(low) and math.isfinite(high) and low <= high):
        raise AnalysisError(f"Invalid paired BCa confidence interval: {identity}")
    return {
        "ci95_low_pp": low,
        "ci95_high_pp": high,
        "ci_seed": seed,
        "ci_method": "BCa",
        "ci_resamples": 20_000,
        "ci_level": 0.95,
        "ci_excludes_zero": low > 0 or high < 0,
    }


def calculate_mcnemar(runs: list[Run], warnings: list[str]) -> dict[str, Any]:
    """Compare complete question pairs separately within each repetition."""
    lookup = {
        (run.model, run.benchmark, run.repetition, run.strategy): run for run in runs
    }
    contexts = sorted({key[:3] for key in lookup})
    strategies = sorted({run.strategy for run in runs} - {BASELINE_STRATEGY})
    tests: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for model, benchmark, repetition in contexts:
        for strategy in strategies:
            identity = {
                "model": model,
                "benchmark": benchmark,
                "strategy": strategy,
                "repetition": repetition,
            }
            direct = lookup.get((model, benchmark, repetition, BASELINE_STRATEGY))
            other = lookup.get((model, benchmark, repetition, strategy))
            reason = None
            if direct is None or other is None:
                reason = "Direct or strategy run is missing."
            elif direct.status != "completed" or other.status != "completed":
                reason = "Both runs must be completed."
            elif direct.seed is None or other.seed is None or direct.seed != other.seed:
                reason = "Master seeds are missing or different."
            else:
                direct_samples = {
                    sample.question_id: sample for sample in direct.samples
                }
                other_samples = {sample.question_id: sample for sample in other.samples}
                if len(direct_samples) != len(direct.samples) or len(
                    other_samples
                ) != len(other.samples):
                    reason = "Duplicate question IDs prevent one-to-one pairing."
                elif (
                    not direct_samples or direct_samples.keys() != other_samples.keys()
                ):
                    reason = "The complete question ID sets are empty or different."
                else:
                    counts: Counter[tuple[bool, bool]] = Counter()
                    for question_id, baseline in direct_samples.items():
                        candidate = other_samples[question_id]
                        if (
                            baseline.document is None
                            or candidate.document is None
                            or baseline.document != candidate.document
                            or baseline.expected_answer != candidate.expected_answer
                            or baseline.document_hash != candidate.document_hash
                            or baseline.target_hash != candidate.target_hash
                        ):
                            reason = f"Question or reference data differ/missing: {question_id}."
                            break
                        if any(
                            sample.metric not in (0.0, 1.0)
                            or (
                                sample.outcome in {"failed", "unparseable"}
                                and sample.metric != 0.0
                            )
                            for sample in (baseline, candidate)
                        ):
                            reason = (
                                f"Inconsistent binary correctness score: {question_id}."
                            )
                            break
                        counts[(baseline.metric == 1.0, candidate.metric == 1.0)] += 1
                    if reason is None:
                        both_correct = counts[(True, True)]
                        losses = counts[(True, False)]
                        gains = counts[(False, True)]
                        both_wrong = counts[(False, False)]
                        # Rows: Direct correct/incorrect; columns: strategy correct/incorrect.
                        result = mcnemar(
                            [[both_correct, losses], [gains, both_wrong]], exact=True
                        )
                        question_ids = sorted(direct_samples)
                        interval = paired_confidence_interval(
                            np.array(
                                [direct_samples[q].metric for q in question_ids],
                                dtype=np.int8,
                            ),
                            np.array(
                                [other_samples[q].metric for q in question_ids],
                                dtype=np.int8,
                            ),
                            (model, benchmark, strategy, repetition),
                        )
                        tests.append(
                            {
                                **identity,
                                "seed": direct.seed,
                                "paired_questions": len(direct_samples),
                                "both_correct": both_correct,
                                "direct_only_correct": losses,
                                "strategy_only_correct": gains,
                                "both_incorrect": both_wrong,
                                "accuracy_difference_pp": 100
                                * (gains - losses)
                                / len(direct_samples),
                                "p_raw": float(result.pvalue),
                                **interval,
                            }
                        )
            if reason is not None:
                skipped.append({**identity, "reason": reason})
                warnings.append(
                    f"McNemar skipped: {model}/{benchmark}/{strategy}, "
                    f"repetition {repetition}: {reason}"
                )

    for test in tests:
        significant = test["p_raw"] <= 0.05
        test["significant"] = significant
        test["outcome"] = (
            "Significant increase"
            if significant and test["accuracy_difference_pp"] > 0
            else "Significant decrease"
            if significant
            else "Not significant"
        )
    return {
        "method": "Two-sided exact McNemar test",
        "correction": None,
        "alpha": 0.05,
        "library": f"statsmodels {statsmodels.__version__}",
        "confidence_interval": {
            "method": "Paired BCa bootstrap",
            "confidence_level": 0.95,
            "resamples": 20_000,
            "library": f"SciPy {scipy.__version__}",
            "scope": "Individual comparison within each repetition; no multiple-comparison adjustment",
        },
        "comparison_count": len(tests),
        "summary": dict(Counter(test["outcome"] for test in tests)),
        "tests": tests,
        "skipped": skipped,
    }


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def fmt_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}%"


def fmt_pp(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:+.2f} pp"


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
    lookup = {(g.model, g.benchmark, g.strategy): g for g in groups}
    data = []
    for group in groups:
        direct = lookup.get((group.model, group.benchmark, BASELINE_STRATEGY))
        data.append(
            StrategyPlotData(
                model=group.model,
                benchmark=group.benchmark,
                strategy=group.strategy,
                label=group.label,
                mean=group.mean,
                gain_vs_direct=group.mean - direct.mean
                if direct and group.mean is not None and direct.mean is not None
                else None,
                tokens_per_question=group.avg_tokens,
            )
        )
    return data


def render_academic_figures(artifacts: list[FigureArtifact]) -> str:
    if not artifacts:
        return ""
    cards = []
    for artifact in artifacts:
        stem = escape(artifact.stem)
        cards.append(
            f"""
<figure class="academic-figure">
  <a href="figures/{stem}.pdf" title="Open PDF">
    <img src="figures/{stem}.svg" alt="{escape(artifact.alt_text)}" loading="lazy">
  </a>
  <figcaption><strong>{escape(artifact.title)}</strong><br>
    {escape(artifact.caption)}
    <span class="figure-links">Download:
      <a href="figures/{stem}.pdf">PDF</a> ·
      <a href="figures/{stem}.svg">SVG</a>
    </span>
  </figcaption>
</figure>"""
        )
    return f"""
<section id="academic-figures">
  <h2>Figures</h2>
  <div class="figure-grid">{"".join(cards)}</div>
</section>"""


def render_benchmark_section(
    benchmark: str, groups: list[Group], comparisons: list[Comparison]
) -> str:
    lookup = {(c.model, c.strategy): c for c in comparisons}
    best = {
        model: max(
            (g.mean for g in groups if g.model == model and g.mean is not None),
            default=None,
        )
        for model in {g.model for g in groups}
    }
    rows, classes, summaries = [], [], []
    frontier = pareto_strategies(build_plot_data(groups))
    for group in groups:
        comparison = lookup.get((group.model, group.strategy))
        ratio = comparison.latency_ratio if comparison else None
        is_best = group.mean is not None and group.mean == best[group.model]
        rows.append(
            [
                escape(MODEL_LABELS.get(group.model, group.model)),
                escape(group.label),
                fmt_percent(group.mean),
                "N/A" if group.std is None else f"{group.std * 100:.2f} pp",
                "-"
                if group.strategy == BASELINE_STRATEGY
                else fmt_pp(comparison.gain if comparison else None),
                "N/A" if group.avg_calls is None else f"{group.avg_calls:.3f}",
                fmt_number(group.avg_tokens, 0),
                fmt_seconds(group.avg_end_to_end_latency),
                "N/A" if ratio is None else f"{ratio:.2f}x",
                "N/A"
                if group.mean is None or group.avg_tokens is None
                else "Frontier"
                if (group.model, group.benchmark, group.strategy) in frontier
                else "Dominated",
            ]
        )
        classes.append("best-row" if is_best else "")
    for model in sorted(best, key=model_order):
        winners = [
            g.label
            for g in groups
            if g.model == model and g.mean is not None and g.mean == best[model]
        ]
        if winners:
            summaries.append(
                f"{MODEL_LABELS.get(model, model)}: {', '.join(winners)} "
                f"{'leads' if len(winners) == 1 else 'tie'} at {fmt_percent(best[model])}."
            )
    return BENCHMARK_TEMPLATE.substitute(
        benchmark=escape(BENCHMARK_LABELS.get(benchmark, benchmark)),
        summary=escape(" ".join(summaries)),
        results_table=render_table(
            [
                "Model",
                "Strategy",
                "Accuracy",
                "SD",
                "Gain vs Direct",
                "Calls / Q",
                "Tokens / Q",
                "Latency / Q",
                "Latency vs Direct",
                "Token trade-off",
            ],
            rows,
            classes,
        ),
    )


def render_mcnemar_section(data: dict[str, Any]) -> str:
    tests, counts = data["tests"], data["summary"]
    overview = []
    for strategy in sorted({t["strategy"] for t in tests}, key=strategy_order):
        outcomes = Counter(t["outcome"] for t in tests if t["strategy"] == strategy)
        overview.append(
            [
                escape(STRATEGY_LABELS.get(strategy, strategy)),
                *[
                    str(outcomes[key])
                    for key in (
                        "Significant increase",
                        "Significant decrease",
                        "Not significant",
                    )
                ],
            ]
        )
    details = render_table(
        [
            "Model",
            "Benchmark",
            "Strategy",
            "Run",
            "Gain vs Direct",
            "95% CI",
            "McNemar p",
            "McNemar result",
        ],
        [
            [
                escape(MODEL_LABELS.get(t["model"], t["model"])),
                escape(BENCHMARK_LABELS.get(t["benchmark"], t["benchmark"])),
                escape(STRATEGY_LABELS.get(t["strategy"], t["strategy"])),
                str(t["repetition"]),
                f"{t['accuracy_difference_pp']:+.2f} pp",
                f"[{t['ci95_low_pp']:+.2f}, {t['ci95_high_pp']:+.2f}] pp",
                f"{t['p_raw']:.4g}",
                escape(t["outcome"]),
            ]
            for t in tests
        ],
        [
            "significant-increase"
            if t["outcome"] == "Significant increase"
            else "significant-decrease"
            if t["outcome"] == "Significant decrease"
            else ""
            for t in tests
        ],
    )
    summary = (
        f"{len(tests)} paired comparisons. Exact McNemar tests show {counts.get('Significant increase', 0)} significant increases, "
        f"{counts.get('Significant decrease', 0)} decreases, {counts.get('Not significant', 0)} not significant."
    )
    return f"""<section id="statistical-comparison"><h2>Statistical comparison with Direct</h2>
      <p>{summary}</p>
      <p class="muted">We calculate a 95% confidence interval for each accuracy difference using a paired BCa bootstrap with 20,000 resamples.
      Questions are resampled with the two strategies' outcomes kept together. McNemar p-values use a {data["alpha"]:g} significance threshold.
      These intervals and tests describe individual comparisons within each repetition, not the mean across repetitions.
      No adjustment is applied across the {data["comparison_count"]} comparisons. {len(data["skipped"])} comparisons skipped.</p>
      {render_table(["Strategy", "Significant increase", "Significant decrease", "Not significant"], overview)}
      <details><summary>Paired results, confidence intervals and p-values</summary>{details}</details>
      <p><a href="aggregates.json">Analysis data, including paired counts, confidence intervals and p-values (JSON)</a></p></section>"""


def summarize_findings(
    comparisons: list[Comparison], behavior: dict[str, Any], cross: dict[str, Any]
) -> list[str]:
    findings = []
    if cross["strategies"]:
        leader = max(
            cross["strategies"],
            key=lambda r: (r["improved"], r["highest_accuracy"], r["mean_gain"]),
        )
        findings.append(
            f"{STRATEGY_LABELS.get(leader['strategy'], leader['strategy'])} improves on Direct in "
            f"{leader['improved']}/{leader['configurations']} configurations and has the highest accuracy "
            f"in {leader['highest_accuracy']}/{leader['configurations']} (including ties)."
        )
    candidates = [
        c for c in comparisons if c.strategy != BASELINE_STRATEGY and c.gain is not None
    ]
    for label, eligible, choose in (
        (
            "Largest improvement",
            [c for c in candidates if c.gain is not None and c.gain > 0],
            max,
        ),
        (
            "Largest decline",
            [c for c in candidates if c.gain is not None and c.gain < 0],
            min,
        ),
    ):
        if eligible:
            c = choose(eligible, key=lambda c: c.gain if c.gain is not None else 0)
            latency = (
                f"; {c.latency_ratio:.2f}x Direct latency"
                if c.latency_ratio is not None
                else ""
            )
            findings.append(
                f"{label}: {c.label} on {MODEL_LABELS.get(c.model, c.model)} / "
                f"{BENCHMARK_LABELS.get(c.benchmark, c.benchmark)}, {fmt_pp(c.gain)}{latency}."
            )
    revised = [
        r for r in behavior["society_of_minds_accuracy"] if r["gain"] is not None
    ]
    if revised:
        findings.append(
            f"Society of Minds revision improves accuracy in {sum(r['gain'] > 0 for r in revised)}/{len(revised)} "
            f"configurations and reduces it in {sum(r['gain'] < 0 for r in revised)}/{len(revised)}; "
            "these comparisons cover only revised attempts."
        )
    return findings


def render_report(
    experiment_label: str,
    output_dir: Path,
    groups: list[Group],
    comparisons: list[Comparison],
    warnings: list[str],
    figure_artifacts: list[FigureArtifact],
    statistical_comparison: dict[str, Any],
    behavior: dict[str, Any],
    summary: dict[str, Any],
    cross: dict[str, Any],
    findings: list[str],
) -> str:
    benchmarks = sorted({g.benchmark for g in groups}, key=benchmark_order)
    models = sorted({g.model for g in groups}, key=model_order)
    benchmark_sections = "".join(
        render_benchmark_section(
            benchmark,
            [g for g in groups if g.benchmark == benchmark],
            [c for c in comparisons if c.benchmark == benchmark],
        )
        for benchmark in benchmarks
    )
    quality_rows = [
        [
            escape(MODEL_LABELS.get(g.model, g.model)),
            escape(BENCHMARK_LABELS.get(g.benchmark, g.benchmark)),
            escape(g.label),
            str(run.repetition),
            escape(run.status),
            fmt_number(len(run.samples)),
            fmt_number(run.counts["failed"]),
            fmt_number(run.counts["unparseable"]),
            fmt_percent(run.score),
        ]
        for g in groups
        for run in g.runs
    ]
    warning_details = (
        '<div class="warning"><strong>Data-quality warnings</strong><ul>'
        + "".join(f"<li>{escape(warning)}</li>" for warning in warnings)
        + "</ul></div>"
        if warnings
        else '<p class="success">No data-quality warnings were found.</p>'
    )
    completion = (
        f"{summary['completed_runs']:,} of {summary['runs']:,} loaded runs completed. "
        f"{summary['failed']:,} observations failed; {summary['unparseable']:,} "
        f"({summary['unparseable_rate']:.2%}) were unparseable and scored zero."
    )
    return REPORT_TEMPLATE.substitute(
        experiment_id=escape(experiment_label),
        generated_at=escape(datetime.now(UTC).isoformat()),
        experiment_folder=escape(output_dir),
        folder_label=escape(output_dir.name),
        model=escape(", ".join(MODEL_LABELS.get(model, model) for model in models)),
        benchmark_count=len(benchmarks),
        strategy_count=len({g.strategy for g in groups}),
        repetitions=escape(
            ", ".join(str(n) for n in sorted({len(g.runs) for g in groups}))
        ),
        unique_questions=f"{summary['unique_questions']:,}",
        observations=f"{summary['observations']:,}",
        completion_summary=escape(completion),
        warning_details=warning_details,
        key_findings=(
            '<section id="key-findings"><h2>Key findings</h2><ul>'
            + "".join(f"<li>{escape(finding)}</li>" for finding in findings)
            + "</ul></section>"
        ),
        benchmark_sections=benchmark_sections,
        strategy_behavior=render_behavior(behavior),
        cross_benchmark=render_cross_benchmark(cross),
        academic_figures=render_academic_figures(figure_artifacts),
        statistical_comparison=render_mcnemar_section(statistical_comparison),
        quality_table=render_table(
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
        ),
    )


def export_group(group: Group) -> dict[str, Any]:
    # Avoid copying all sample traces just to discard the runs afterwards.
    data = {
        field.name: getattr(group, field.name)
        for field in fields(group)
        if field.name != "runs"
    }
    data["per_run"] = [
        {
            "repetition": run.repetition,
            "seed": run.seed,
            "status": run.status,
            "evaluated": len(run.samples),
            "correct": run.counts["correct"],
            "incorrect": run.counts["incorrect"],
            "failed": run.counts["failed"],
            "unparseable": run.counts["unparseable"],
            "score": run.score,
            "calls": run.calls,
            "prompt_tokens": run.prompt_tokens,
            "output_tokens": run.output_tokens,
            "total_tokens": run.total_tokens,
            "end_to_end_latency": run.end_to_end_latency,
        }
        for run in group.runs
    ]
    return data


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def completion_summary(runs: list[Run]) -> dict[str, Any]:
    counts = Counter(sample.outcome for run in runs for sample in run.samples)
    observations = sum(counts.values())
    return {
        "runs": len(runs),
        "completed_runs": sum(run.status == "completed" for run in runs),
        "unique_questions": len(
            {
                (run.benchmark, sample.question_id)
                for run in runs
                for sample in run.samples
            }
        ),
        "observations": observations,
        "failed": counts["failed"],
        "unparseable": counts["unparseable"],
        "unparseable_rate": counts["unparseable"] / observations if observations else 0,
    }


def calculate_cross_benchmark(
    groups: list[Group], comparisons: list[Comparison]
) -> dict[str, Any]:
    non_direct = [
        c for c in comparisons if c.strategy != BASELINE_STRATEGY and c.gain is not None
    ]
    best: dict[tuple[str, str], float] = {}
    for group in groups:
        if group.mean is not None:
            key = (group.model, group.benchmark)
            best[key] = max(best.get(key, group.mean), group.mean)
    strategies = []
    for strategy in sorted({c.strategy for c in non_direct}, key=strategy_order):
        selected = [c for c in non_direct if c.strategy == strategy]
        strategies.append(
            {
                "strategy": strategy,
                "configurations": len(selected),
                "improved": sum(c.verdict == "improved" for c in selected),
                "highest_accuracy": sum(
                    c.strategy_score == best[(c.model, c.benchmark)] for c in selected
                ),
                "mean_gain": statistics.fmean(
                    c.gain for c in selected if c.gain is not None
                ),
            }
        )
    breakdown = {}
    for dimension in ("benchmark", "model"):
        order = benchmark_order if dimension == "benchmark" else model_order
        breakdown[dimension] = [
            {
                dimension: value,
                "comparisons": len(selected),
                "improved": sum(c.verdict == "improved" for c in selected),
                "mean_gain": statistics.fmean(
                    c.gain for c in selected if c.gain is not None
                ),
            }
            for value in sorted({getattr(c, dimension) for c in non_direct}, key=order)
            if (selected := [c for c in non_direct if getattr(c, dimension) == value])
        ]
    frontier = pareto_strategies(build_plot_data(groups))
    pareto = [
        {
            "model": g.model,
            "benchmark": g.benchmark,
            "strategy": g.strategy,
            "on_frontier": (g.model, g.benchmark, g.strategy) in frontier,
        }
        for g in groups
        if g.mean is not None and g.avg_tokens is not None
    ]
    return {"strategies": strategies, **breakdown, "token_pareto": pareto}


def render_cross_benchmark(data: dict[str, Any]) -> str:
    rows = [
        [
            escape(STRATEGY_LABELS.get(r["strategy"], r["strategy"])),
            f"{r['improved']}/{r['configurations']}",
            f"{r['highest_accuracy']}/{r['configurations']}",
            fmt_pp(r["mean_gain"]),
        ]
        for r in data["strategies"]
    ]
    trends = []
    for dimension, labels in (("benchmark", BENCHMARK_LABELS), ("model", MODEL_LABELS)):
        if data[dimension]:
            trends.append(
                f"Mean gain by {dimension}: "
                + ", ".join(
                    f"{labels.get(r[dimension], r[dimension])} {fmt_pp(r['mean_gain'])}"
                    for r in data[dimension]
                )
                + "."
            )
    return (
        '<section id="cross-benchmark"><h2>Strategy comparison</h2>'
        + render_table(
            [
                "Strategy",
                "Improved over Direct",
                "Highest accuracy (incl. ties)",
                "Mean gain vs Direct",
            ],
            rows,
        )
        + f"<p>{escape(' '.join(trends))}</p></section>"
    )


def render_behavior(data: dict[str, Any]) -> str:
    def table(headers: list[str], rows: list[dict[str, Any]], cells: Any) -> str:
        return render_table(
            ["Model", "Benchmark"] + headers,
            [
                [
                    escape(MODEL_LABELS.get(r["model"], r["model"])),
                    escape(BENCHMARK_LABELS.get(r["benchmark"], r["benchmark"])),
                ]
                + cells(r)
                for r in rows
            ],
        )

    sections = ['<section id="strategy-behavior"><h2>Strategy behavior</h2>']
    sc = data["self_consistency_accuracy"]
    if sc:
        gains = [r["gain"] for r in sc if r["gain"] is not None]
        sections.append("<h3>Self-Consistency voting</h3>")
        if gains:
            sections.append(
                f"<p>Voting changed candidate accuracy by {fmt_pp(min(gains))} to {fmt_pp(max(gains))} across {len(gains)} configurations.</p>"
            )
        sections.append(
            table(
                ["Initial-six accuracy", "Final accuracy", "Voting gain"],
                sc,
                lambda r: [
                    fmt_percent(r["individual_accuracy"]),
                    fmt_percent(r["final_accuracy"]),
                    fmt_pp(r["gain"]),
                ],
            )
        )
        sections.append(
            "<details><summary>Initial agreement and final accuracy</summary>"
            + render_table(
                ["Agreement among six initial answers", "Attempts", "Final accuracy"],
                [
                    [
                        escape(r["label"]),
                        fmt_number(r["observations"]),
                        fmt_percent(r["final_accuracy"]),
                    ]
                    for r in data["self_consistency_agreement"]
                ],
            )
            + "</details>"
        )
    revisions = {
        (r["model"], r["benchmark"]): r for r in data["society_of_minds_revision"]
    }
    som = data["society_of_minds_accuracy"]
    if som:

        def revision_cells(r: dict[str, Any]) -> list[str]:
            revision = revisions[(r["model"], r["benchmark"])]
            return [
                f"{revision['revised']:,} ({fmt_percent(revision['revision_rate'])})",
                fmt_percent(revision["agreement_rate"]),
                fmt_percent(r["before_accuracy"]),
                fmt_percent(r["after_accuracy"]),
                fmt_pp(r["gain"]),
                fmt_number(r["corrected"]),
                fmt_number(r["damaged"]),
            ]

        sections.append("<h3>Society of Minds revision</h3>")
        sections.append(
            table(
                [
                    "Revised attempts",
                    "All three agree after revision",
                    "Before",
                    "After",
                    "Change",
                    "Corrections",
                    "Degradations",
                ],
                som,
                revision_cells,
            )
        )
    svj = data["svj_transitions"]
    if svj:
        sections.append("<h3>Solver-Verifier-Judge changes</h3>")
        sections.append(
            table(
                [
                    "Wrong Solver → correct final",
                    "Missing Solver → correct final",
                    "Correct Solver → wrong final",
                ],
                svj,
                lambda r: [
                    fmt_number(r[k])
                    for k in (
                        "wrong_to_correct",
                        "missing_to_correct",
                        "correct_to_wrong",
                    )
                ],
            )
        )
        sections.append(
            "<details><summary>Final selection when Solver and Verifier disagree</summary>"
            + table(
                [
                    "Disagreements",
                    "Matched Solver",
                    "Matched Verifier",
                    "Matched neither",
                ],
                data["svj_disagreement"],
                lambda r: [
                    fmt_number(r[k])
                    for k in (
                        "disagreements",
                        "matched_solver",
                        "matched_verifier",
                        "matched_neither",
                    )
                ],
            )
            + "</details>"
        )
    sections.append(
        '<p class="muted">Counts are repeated attempts. Revision rate uses all analyzed attempts; agreement and before/after accuracy use revised attempts. Agreement requires three valid answers; final correctness uses saved benchmark scores.</p></section>'
    )
    return "".join(sections) if sc or som or svj else ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze accuracy, cost and strategy behavior from saved experiments."
    )
    parser.add_argument("experiment_folders", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Report directory (defaults to analysis or analysis-comparison).",
    )
    args = parser.parse_args()
    experiment_dirs = [path.resolve() for path in args.experiment_folders]
    for experiment_dir in experiment_dirs:
        if not experiment_dir.is_dir():
            parser.error(f"not a directory: {experiment_dir}")
    try:
        experiment_label, runs, warnings = load_results(experiment_dirs)
        groups = calculate_statistics(runs, warnings)
        comparisons = compare_strategies(groups)
        statistical_comparison = calculate_mcnemar(runs, warnings)
        behavior = calculate_strategy_behavior(groups)
        warnings.extend(behavior["warnings"])
        warnings = list(dict.fromkeys(warnings))
        summary = completion_summary(runs)
        cross = calculate_cross_benchmark(groups, comparisons)
        findings = summarize_findings(comparisons, behavior, cross)
    except AnalysisError as exc:
        parser.error(str(exc))
    output_dir = (
        args.output.resolve()
        if args.output
        else (
            experiment_dirs[0] / "analysis"
            if len(experiment_dirs) == 1
            else experiment_dirs[0].parent / "analysis-comparison"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_artifacts = generate_academic_figures(
        output_dir / "figures", build_plot_data(groups), statistical_comparison
    )
    report_path = output_dir / "report.html"
    report_path.write_text(
        render_report(
            experiment_label,
            output_dir,
            groups,
            comparisons,
            warnings,
            figure_artifacts,
            statistical_comparison,
            behavior,
            summary,
            cross,
            findings,
        ),
        encoding="utf-8",
    )
    tests = statistical_comparison["tests"]
    write_json(
        output_dir / "aggregates.json",
        {
            "schema_version": 2,
            "generated_at": datetime.now(UTC).isoformat(),
            "experiment_folders": [str(path) for path in experiment_dirs],
            "accuracy_metric": "Saved primary exact-match score; failures and unparseable answers score zero",
            "standard_deviation": "Sample SD across repetitions (ddof=1)",
            "summary": summary,
            "findings": findings,
            "results": [export_group(group) for group in groups],
            "comparisons": [asdict(comparison) for comparison in comparisons],
            "strategy_behavior": behavior,
            "cross_benchmark": cross,
            "statistical_comparison": statistical_comparison,
            "warnings": warnings,
        },
    )
    print(f"Report: {report_path}")
    print(f"Aggregates: {output_dir / 'aggregates.json'}")
    print(f"Figures: {output_dir / 'figures'} ({len(figure_artifacts)})")
    print(f"Warnings: {len(warnings)}")
    print(
        f"Paired comparisons: {len(tests)} exact McNemar tests with 95% BCa confidence intervals, {len(statistical_comparison['skipped'])} skipped"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
