import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from common.exceptions import ConfigurationError
from configs.context import build_llm_eval_config, load_experiments
from configs.experiments import ApplicationSettings, Experiment, ExperimentConfig
from evaluation.lm_eval_harness import LLMEvalHarness

logger = logging.getLogger(__name__)

SOURCE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_DIR.parent


def run_evaluation(
    repetitions: int | None = None,
    question_limit: int | None = None,
) -> dict[str, Any]:
    settings, config, experiments = load_experiments()

    if repetitions is not None:
        config.run.repetitions = repetitions
    if question_limit is not None:
        config.run.questions = question_limit

    if not settings.evaluation.log_samples:
        raise ConfigurationError("evaluation.log_samples must be true")

    started_at = datetime.now(UTC)
    experiment_id = f"{started_at:%Y-%m-%d_%H%M%SZ}-{uuid.uuid4().hex[:8]}"

    output_root = Path(config.run.output_dir)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    output_dir = output_root / experiment_id
    output_dir.mkdir(parents=True)

    _write_json(
        output_dir / "experiment.json",
        {
            "experiment_id": experiment_id,
            "created_at": started_at.isoformat(),
            "selected_experiments": [experiment.name for experiment in experiments],
            "configuration": {
                "application": settings.model_dump(mode="json"),
                "run": config.run.model_dump(mode="json"),
                "matrix": config.matrix.model_dump(mode="json"),
                "strategies": {
                    experiment.strategy_name: experiment.strategy.model_dump(
                        mode="json"
                    )
                    for experiment in experiments
                },
                "benchmarks": {
                    experiment.benchmark_name: experiment.benchmark.model_dump(
                        mode="json"
                    )
                    for experiment in experiments
                },
            },
            "cost": None,
        },
    )

    logger.info("Experiment %s started; results=%s", experiment_id, output_dir)
    statuses: list[str] = []

    with tempfile.TemporaryDirectory(prefix="master-thesis-") as temp_dir:
        call_log = Path(temp_dir) / "calls.jsonl"

        try:
            api_process = _start_api(settings, call_log)
        except Exception as exc:
            _write_json(
                output_dir / "error.json",
                {
                    "stage": "api_startup",
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            raise

        try:
            for experiment in experiments:
                for repetition in range(1, config.run.repetitions + 1):
                    statuses.append(
                        _run_once(
                            output_dir=output_dir,
                            experiment_id=experiment_id,
                            experiment=experiment,
                            settings=settings,
                            config=config,
                            repetition=repetition,
                            call_log=call_log,
                        )
                    )
        finally:
            _stop_api(api_process)

    summary = {
        "experiment_id": experiment_id,
        "output_dir": str(output_dir),
        "run_count": len(statuses),
        "completed": statuses.count("completed"),
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
    run_dir.mkdir(parents=True)
    call_log.unlink(missing_ok=True)

    started_at = datetime.now(UTC)
    timer = time.perf_counter()

    logger.info(
        "Running benchmark=%s strategy=%s repetition=%d",
        experiment.benchmark_name,
        experiment.strategy_name,
        repetition,
    )

    results: dict[str, Any] | None = None
    error: dict[str, str] | None = None

    try:
        evaluation_config = build_llm_eval_config(
            settings,
            experiment,
            question_limit=config.run.questions,
        )
        results = LLMEvalHarness(evaluation_config).evaluate()

        if not results.get("samples", {}).get(experiment.benchmark.task):
            raise RuntimeError("lm-eval returned no samples")

    except Exception as exc:
        logger.exception("Run failed: %s", run_id)
        error = {
            "stage": "evaluation",
            "type": type(exc).__name__,
            "message": str(exc),
        }

    calls = _read_calls(call_log)

    if results is not None and error is None:
        samples = _build_samples(
            experiment,
            results["samples"][experiment.benchmark.task],
            calls,
        )
    elif calls:
        samples = [
            {
                "record_type": "unscored_calls",
                "status": "unscored",
                "calls": calls,
            }
        ]
    else:
        samples = []

    error_count = sum(sample.get("status") != "completed" for sample in samples)

    if error is not None:
        status = "failed"
    elif error_count:
        status = "completed_with_errors"
    else:
        status = "completed"

    strategy_results = [call["result"] for call in calls if "result" in call]
    prompt_tokens = sum(result.get("prompt_tokens", 0) for result in strategy_results)
    output_tokens = sum(result.get("output_tokens", 0) for result in strategy_results)

    evaluation = None
    if results:
        evaluation = {key: value for key, value in results.items() if key != "samples"}

    _write_jsonl(run_dir / "samples.jsonl", samples)
    _write_json(
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
            "model": settings.provider.model,
            "seed": experiment.strategy.generation.seed,
            "sample_count": sum(
                sample.get("record_type") == "sample" for sample in samples
            ),
            "error_count": error_count,
            "model_call_count": sum(
                len(result.get("agent_responses", [])) for result in strategy_results
            ),
            "tokens": {
                "prompt": prompt_tokens,
                "output": output_tokens,
                "total": prompt_tokens + output_tokens,
            },
            "model_latency_seconds": sum(
                result.get("total_latency_s") or 0 for result in strategy_results
            ),
            "wall_time_seconds": time.perf_counter() - timer,
            "cost": None,
            "evaluation": evaluation,
            "error": error,
        },
    )

    return status


def _build_samples(
    experiment: Experiment,
    raw_samples: list[dict[str, Any]],
    calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    samples_by_id: dict[str, dict[str, Any]] = {}

    for raw in raw_samples:
        sample_id = str(raw.get("doc_id"))

        if sample_id not in samples_by_id:
            samples_by_id[sample_id] = {
                "sample_id": raw.get("doc_id"),
                "question_id": f"{experiment.benchmark_name}:{raw.get('doc_id')}",
                "document": raw.get("doc"),
                "prompt": str(_first_value(raw.get("arguments"))),
                "expected_answer": raw.get("target"),
                "lm_eval_response": _first_value(raw.get("resps")),
                "hashes": {
                    "document": raw.get("doc_hash"),
                    "prompt": raw.get("prompt_hash"),
                    "target": raw.get("target_hash"),
                },
                "evaluations": {},
            }

        samples_by_id[sample_id]["evaluations"][raw.get("filter", "none")] = {
            "response": raw.get("filtered_resps"),
            "metrics": {metric: raw.get(metric) for metric in raw.get("metrics", [])},
        }

    calls_by_prompt: dict[str, list[dict[str, Any]]] = {}
    unmatched_calls: list[dict[str, Any]] = []

    for call in calls:
        prompt = call.get("prompt")
        if isinstance(prompt, str):
            calls_by_prompt.setdefault(prompt, []).append(call)
        else:
            unmatched_calls.append(call)

    records: list[dict[str, Any]] = []

    for sample in samples_by_id.values():
        matching_calls = calls_by_prompt.pop(sample["prompt"], [])

        has_result = any("result" in call for call in matching_calls)
        has_error = any("error" in call for call in matching_calls)

        if not has_result:
            status = "failed"
        elif has_error:
            status = "completed_with_retries"
        else:
            status = "completed"

        records.append(
            {
                "record_type": "sample",
                "status": status,
                **sample,
                "calls": matching_calls,
            }
        )

    for remaining_calls in calls_by_prompt.values():
        unmatched_calls.extend(remaining_calls)

    if unmatched_calls:
        records.append(
            {
                "record_type": "unscored_calls",
                "status": "unscored",
                "calls": unmatched_calls,
            }
        )

    return records


def _first_value(value: Any) -> Any:
    while isinstance(value, (list, tuple)) and value:
        value = value[0]
    return value


def _read_calls(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _start_api(
    settings: ApplicationSettings,
    call_log: Path,
) -> subprocess.Popen[bytes]:
    environment = os.environ.copy()
    environment["STRATEGY_RESULTS_PATH"] = str(call_log)

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            settings.server.host,
            "--port",
            str(settings.server.port),
        ],
        cwd=SOURCE_DIR,
        env=environment,
    )

    host = settings.server.host
    if host == "0.0.0.0":
        host = "127.0.0.1"

    url = f"http://{host}:{settings.server.port}/openapi.json"
    deadline = time.monotonic() + max(30, settings.evaluation.timeout)

    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"API exited with code {process.returncode}")

            try:
                if httpx.get(url, timeout=1).is_success:
                    return process
            except httpx.HTTPError:
                pass

            time.sleep(0.25)

        raise TimeoutError(f"API did not start before timeout: {url}")

    except Exception:
        _stop_api(process)
        raise


def _stop_api(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return

    process.terminate()

    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as output:
        json.dump(
            value,
            output,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )
        output.write("\n")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for value in values:
            output.write(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    default=_json_default,
                )
                + "\n"
            )


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    return str(value)
