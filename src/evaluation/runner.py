import logging
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from configs.config_loader import load_experiments
from configs.experiments import ApplicationSettings, Experiment, ExperimentConfig
from evaluation.api_process import start_api, stop_api
from evaluation.lm_eval_config import build_llm_eval_config
from evaluation.lm_eval_harness import LLMEvalHarness
from evaluation.result_store import (
    completed_run_exists,
    prepare_experiment_directory,
    write_json,
    write_jsonl,
)
from evaluation.sample_results import build_samples, read_calls

logger = logging.getLogger(__name__)


def run_evaluation(
    repetitions: int | None = None,
    question_limit: int | None = None,
    benchmark: str | None = None,
    experiment_dir: Path | None = None,
) -> dict[str, Any]:
    settings, config, experiments = load_experiments(benchmark)
    if repetitions is not None:
        config.run.repetitions = repetitions
    if question_limit is not None:
        config.run.questions = question_limit
    output_dir, experiment_id = prepare_experiment_directory(
        config.run.output_dir, datetime.now(UTC), experiment_dir
    )
    pending_runs = [
        (experiment, repetition)
        for experiment in experiments
        for repetition in range(1, config.run.repetitions + 1)
        if not completed_run_exists(output_dir, experiment, repetition)
    ]

    logger.info("Experiment %s started; results=%s", experiment_id, output_dir)
    statuses = []
    if pending_runs:
        with tempfile.TemporaryDirectory(prefix="master-thesis-") as temp_dir:
            call_log = Path(temp_dir) / "calls.jsonl"
            process = start_api(settings, call_log)
            try:
                for experiment, repetition in pending_runs:
                    statuses.append(
                        _run_once(
                            output_dir,
                            experiment_id,
                            experiment,
                            settings,
                            config,
                            repetition,
                            call_log,
                        )
                    )
            finally:
                stop_api(process)

    skipped = len(experiments) * config.run.repetitions - len(pending_runs)
    summary = {
        "experiment_id": experiment_id,
        "output_dir": str(output_dir),
        "run_count": len(statuses) + skipped,
        "executed": len(statuses),
        "skipped": skipped,
        "completed": statuses.count("completed") + skipped,
        "completed_with_errors": statuses.count("completed_with_errors"),
        "failed": statuses.count("failed"),
    }
    logger.info("Experiment finished: %s", summary)
    return summary


def _run_once(
    output_dir: Path,
    experiment_id: str,
    experiment: Experiment,
    settings: ApplicationSettings,
    config: ExperimentConfig,
    repetition: int,
    call_log: Path,
) -> str:
    run_id = f"{experiment_id}:{experiment.name}:{repetition:03d}"
    run_dir = (
        output_dir
        / experiment.benchmark_name
        / experiment.strategy_name
        / f"run-{repetition:03d}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    call_log.unlink(missing_ok=True)
    started_at = datetime.now(UTC)
    timer = time.perf_counter()

    logger.info(
        "Running benchmark=%s strategy=%s repetition=%d",
        experiment.benchmark_name,
        experiment.strategy_name,
        repetition,
    )
    results = None
    error = None
    try:
        evaluation_config = build_llm_eval_config(
            settings,
            experiment,
            config.run.questions,
            repetition,
            config.run.repetition_seeds[repetition - 1],
        )
        results = LLMEvalHarness(evaluation_config).evaluate()
        raw_samples = results["samples"][experiment.benchmark.task]
        if not raw_samples:
            raise RuntimeError("lm-eval returned no samples")
    except Exception as exc:
        logger.exception("Run failed: %s", run_id)
        error = str(exc)

    calls = read_calls(call_log)
    if error is None:
        samples = build_samples(experiment, raw_samples, calls)
        status = (
            "completed"
            if all(sample["status"] == "completed" for sample in samples)
            else "completed_with_errors"
        )
    else:
        samples = [
            {
                "record_type": "unscored_calls",
                "status": "unscored",
                "calls": calls,
            }
        ]
        status = "failed"

    strategy_results = [call["result"] for call in calls if "result" in call]
    prompt_tokens = sum(result["prompt_tokens"] for result in strategy_results)
    output_tokens = sum(result["output_tokens"] for result in strategy_results)
    write_jsonl(run_dir / "samples.jsonl", samples)
    write_json(
        run_dir / "run.json",
        {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "status": status,
            "benchmark": experiment.benchmark_name,
            "strategy": experiment.strategy_name,
            "repetition": repetition,
            "question_limit": config.run.questions,
            "model": settings.provider.model,
            "seed": config.run.repetition_seeds[repetition - 1],
            "sample_count": sum(
                sample["record_type"] == "sample" for sample in samples
            ),
            "error_count": sum(sample["status"] != "completed" for sample in samples),
            "model_call_count": sum(
                len(result["agent_responses"]) for result in strategy_results
            ),
            "tokens": {
                "prompt": prompt_tokens,
                "output": output_tokens,
                "total": prompt_tokens + output_tokens,
            },
            "model_latency_seconds": sum(
                result["total_latency_s"] or 0 for result in strategy_results
            ),
            "end_to_end_latency_seconds": sum(
                result["end_to_end_latency_s"] or 0 for result in strategy_results
            ),
            "provider_duration_seconds": sum(
                result["provider_duration_s"] or 0 for result in strategy_results
            ),
            "wall_time_seconds": time.perf_counter() - timer,
            "cost": None,
            "evaluation": None
            if results is None
            else {key: value for key, value in results.items() if key != "samples"},
            "error": error,
        },
    )
    return status
