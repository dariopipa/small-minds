from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import socket
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
CONFIG_DIR = SRC_DIR / "configs"
PROVIDER_CONFIG_PATH = CONFIG_DIR / "provider.yaml"
EVAL_CONFIG_PATH = CONFIG_DIR / "llm_eval_harness.yaml"
ACTIVE_STRATEGY_CONFIG_PATH = CONFIG_DIR / "strategy.yaml"
STRATEGY_CONFIG_DIR = CONFIG_DIR / "strategies"
DEFAULT_MAX_OUTPUT_TOKENS = 1024

sys.path.insert(0, str(SRC_DIR))


@dataclass(frozen=True)
class Experiment:
    name: str
    strategy_config_file: str
    provider_options: dict[str, Any]


EXPERIMENTS = [
    Experiment(
        name="direct",
        strategy_config_file="direct.yaml",
        provider_options={
            "seed": 42,
            "temperature": 0,
            "top_p": None,
        },
    ),
    Experiment(
        name="self-consistency",
        strategy_config_file="self_consistency.yaml",
        provider_options={
            "seed": None,
            "temperature": 0.5,
            "top_p": 0.9,
        },
    ),
    # Experiment(
    #     name="society-of-minds",
    #     strategy_config_file="society_of_minds.yaml",
    #     provider_options={
    #         "seed": None,
    #         "temperature": 0.5,
    #         "top_p": 0.9,
    #     },
    # ),
]


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")


def find_free_port(start_port: int) -> int:
    port = start_port

    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                port += 1
                continue

        return port


def wait_for_server(process: subprocess.Popen, url: str, log_path: Path) -> None:
    deadline = time.monotonic() + 120

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Server exited early with code {process.returncode}. Check {log_path}."
            )

        try:
            with urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except URLError:
            time.sleep(1)

    raise RuntimeError(f"Server did not become ready. Check {log_path}.")


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return

    process.terminate()

    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=20)


def provider_config_for_experiment(
    base_provider_config: dict[str, Any],
    experiment: Experiment,
) -> dict[str, Any]:
    provider_config = deepcopy(base_provider_config)
    options = dict(provider_config.get("options", {}))
    options.update(experiment.provider_options)
    current_max_output_tokens = options.get("max_output_tokens")

    if (
        current_max_output_tokens is None
        or current_max_output_tokens < DEFAULT_MAX_OUTPUT_TOKENS
    ):
        options["max_output_tokens"] = DEFAULT_MAX_OUTPUT_TOKENS

    provider_config["options"] = options
    return provider_config


def eval_config_for_experiment(
    base_eval_config: dict[str, Any],
    limit: int,
    base_url: str,
) -> dict[str, Any]:
    eval_config = deepcopy(base_eval_config)
    eval_config["limit"] = limit
    eval_config["log_samples"] = True
    eval_config["write_out"] = False
    eval_config["bootstrap_iters"] = 0
    eval_config["batch_size"] = 1

    model_args = dict(eval_config["model_args"])
    model_args["base_url"] = base_url
    model_args["num_concurrent"] = 1
    model_args["timeout"] = max(int(model_args.get("timeout", 180)), 300)
    eval_config["model_args"] = model_args

    return eval_config


def strategy_result_stats(strategy_results_path: Path) -> dict[str, int | float]:
    stats: dict[str, int | float] = {
        "strategy_result_count": 0,
        "agent_response_count": 0,
        "prompt_tokens": 0,
        "output_tokens": 0,
        "total_latency_s": 0.0,
    }

    if not strategy_results_path.exists():
        return stats

    with strategy_results_path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            strategy_result = record["strategy_result"]
            agent_responses = strategy_result.get("agent_responses", [])

            stats["strategy_result_count"] += 1
            stats["agent_response_count"] += len(agent_responses)
            stats["prompt_tokens"] += strategy_result.get("prompt_tokens", 0)
            stats["output_tokens"] += strategy_result.get("output_tokens", 0)
            stats["total_latency_s"] += strategy_result.get("total_latency_s") or 0

    return stats


def extract_task_summary(results: dict[str, Any], task_name: str) -> dict[str, Any]:
    task_results = results.get("results", {}).get(task_name, {})
    samples = results.get("n-samples", {}).get(task_name, {})

    return {
        "exact_match_strict": task_results.get("exact_match,strict-match"),
        "exact_match_flexible": task_results.get("exact_match,flexible-extract"),
        "exact_match_strict_stderr": task_results.get(
            "exact_match_stderr,strict-match"
        ),
        "exact_match_flexible_stderr": task_results.get(
            "exact_match_stderr,flexible-extract"
        ),
        "effective_samples": samples.get("effective"),
        "original_samples": samples.get("original"),
    }


def start_server(
    port: int,
    experiment_dir: Path,
    strategy_results_path: Path,
) -> subprocess.Popen:
    log_path = experiment_dir / "server.log"
    env = os.environ.copy()
    env["STRATEGY_RESULTS_PATH"] = str(strategy_results_path)

    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "main:app",
        "--app-dir",
        str(SRC_DIR),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "info",
    ]

    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=ROOT_DIR,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    process._codex_log_file = log_file  # type: ignore[attr-defined]
    return process


def close_server_log(process: subprocess.Popen) -> None:
    log_file = getattr(process, "_codex_log_file", None)
    if log_file is not None:
        log_file.close()


def configure_project_logging() -> None:
    main_module = importlib.import_module("main")
    main_module.configure_logging()


def evaluate(eval_config: dict[str, Any]) -> dict[str, Any]:
    config_module = importlib.import_module("evaluation.lm_eval_config")
    harness_module = importlib.import_module("evaluation.lm_eval_harness")

    harness_config = config_module.LLMEvalHarnessConfig.model_validate(eval_config)
    return harness_module.LLMEvalHarness(config=harness_config).evaluate()


def run_experiment(
    experiment: Experiment,
    run_name: str,
    repeat_index: int,
    repetitions: int,
    base_provider_config: dict[str, Any],
    base_eval_config: dict[str, Any],
    output_dir: Path,
    limit: int,
    start_port: int,
) -> dict[str, Any]:
    experiment_dir = output_dir / run_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    port = find_free_port(start_port)
    base_url = f"http://127.0.0.1:{port}/v1/completions"
    strategy_results_path = experiment_dir / "strategy_results.jsonl"
    server_log_path = experiment_dir / "server.log"

    if strategy_results_path.exists():
        raise FileExistsError(
            f"Refusing to append to existing strategy results: {strategy_results_path}"
        )

    provider_config = provider_config_for_experiment(
        base_provider_config,
        experiment,
    )
    strategy_config = load_yaml(STRATEGY_CONFIG_DIR / experiment.strategy_config_file)
    eval_config = eval_config_for_experiment(
        base_eval_config,
        limit=limit,
        base_url=base_url,
    )

    write_yaml(PROVIDER_CONFIG_PATH, provider_config)
    write_yaml(EVAL_CONFIG_PATH, eval_config)
    write_yaml(ACTIVE_STRATEGY_CONFIG_PATH, strategy_config)
    write_json(
        experiment_dir / "experiment_config.json",
        {
            "strategy": run_name,
            "base_strategy": experiment.name,
            "repeat_index": repeat_index,
            "repetitions": repetitions,
            "strategy_config_file": experiment.strategy_config_file,
            "provider_config": provider_config,
            "eval_config": eval_config,
            "strategy_config": strategy_config,
            "strategy_results_path": str(strategy_results_path),
            "server_log_path": str(server_log_path),
        },
    )

    started_at = utc_now()
    wall_start = time.perf_counter()
    server_process = start_server(
        port=port,
        experiment_dir=experiment_dir,
        strategy_results_path=strategy_results_path,
    )

    try:
        wait_for_server(
            process=server_process,
            url=f"http://127.0.0.1:{port}/docs",
            log_path=server_log_path,
        )

        print(f"\n=== Running {run_name} ===")
        print(f"base strategy: {experiment.name}")
        print(f"repeat: {repeat_index}/{repetitions}")
        print(f"limit: {limit}")
        print(f"base_url: {base_url}")
        print(f"server log: {server_log_path}")
        print(f"strategy results: {strategy_results_path}")

        results = evaluate(eval_config)
    finally:
        stop_server(server_process)
        close_server_log(server_process)

    wall_time_s = time.perf_counter() - wall_start
    finished_at = utc_now()

    write_json(experiment_dir / "lm_eval_results.json", results)

    strategy_stats = strategy_result_stats(strategy_results_path)
    task_name = eval_config["tasks"][0]
    metrics = extract_task_summary(results, task_name)
    summary = {
        "strategy": run_name,
        "base_strategy": experiment.name,
        "repeat_index": repeat_index,
        "repetitions": repetitions,
        "task": task_name,
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_time_s": wall_time_s,
        "limit": limit,
        "metrics": metrics,
        "strategy_stats": strategy_stats,
        "provider_options": provider_config.get("options", {}),
        "strategy_config": strategy_config,
        "paths": {
            "experiment_dir": str(experiment_dir),
            "lm_eval_results": str(experiment_dir / "lm_eval_results.json"),
            "strategy_results": str(strategy_results_path),
            "server_log": str(server_log_path),
        },
    }
    write_json(experiment_dir / "summary.json", summary)

    print(f"finished: {run_name}")
    print(f"metrics: {metrics}")
    print(f"strategy stats: {strategy_stats}")

    return summary


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fieldnames = [
        "strategy",
        "base_strategy",
        "task",
        "repeat_index",
        "repetitions",
        "limit",
        "effective_samples",
        "exact_match_strict",
        "exact_match_flexible",
        "wall_time_s",
        "strategy_result_count",
        "agent_response_count",
        "prompt_tokens",
        "output_tokens",
        "total_latency_s",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for summary in summaries:
            metrics = summary["metrics"]
            strategy_stats = summary["strategy_stats"]
            writer.writerow(
                {
                    "strategy": summary["strategy"],
                    "base_strategy": summary.get("base_strategy", summary["strategy"]),
                    "task": summary.get("task"),
                    "repeat_index": summary.get("repeat_index", 1),
                    "repetitions": summary.get("repetitions", 1),
                    "limit": summary["limit"],
                    "effective_samples": metrics["effective_samples"],
                    "exact_match_strict": metrics["exact_match_strict"],
                    "exact_match_flexible": metrics["exact_match_flexible"],
                    "wall_time_s": f"{summary['wall_time_s']:.3f}",
                    "strategy_result_count": strategy_stats["strategy_result_count"],
                    "agent_response_count": strategy_stats["agent_response_count"],
                    "prompt_tokens": strategy_stats["prompt_tokens"],
                    "output_tokens": strategy_stats["output_tokens"],
                    "total_latency_s": f"{strategy_stats['total_latency_s']:.3f}",
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=2,
        help=(
            "Run each strategy this many times. Defaults to 2 true repeats per "
            "strategy."
        ),
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR
        / "outputs"
        / "experiments"
        / datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Do not generate the analysis report after the experiments finish.",
    )
    parser.add_argument(
        "--pricing-preset",
        default="openai-chatgpt-latest",
        help="Named API price preset used by the analysis report.",
    )
    parser.add_argument(
        "--prompt-price-per-1m",
        type=float,
        default=None,
        help="Override prompt/input token price per 1M tokens in USD for cost analysis.",
    )
    parser.add_argument(
        "--output-price-per-1m",
        type=float,
        default=None,
        help="Override output/completion token price per 1M tokens in USD for cost analysis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repetitions < 1:
        raise ValueError("--repetitions must be at least 1")

    configure_project_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    original_provider_config = load_yaml(PROVIDER_CONFIG_PATH)
    original_eval_config = load_yaml(EVAL_CONFIG_PATH)
    original_strategy_config = load_yaml(ACTIVE_STRATEGY_CONFIG_PATH)
    summaries = []

    write_json(
        args.output_dir / "original_configs.json",
        {
            "provider_config": original_provider_config,
            "eval_config": original_eval_config,
            "strategy_config": original_strategy_config,
        },
    )

    try:
        for repeat_index in range(1, args.repetitions + 1):
            for experiment in EXPERIMENTS:
                run_name = (
                    experiment.name
                    if args.repetitions == 1
                    else f"{experiment.name}-run-{repeat_index}"
                )
                summaries.append(
                    run_experiment(
                        experiment=experiment,
                        run_name=run_name,
                        repeat_index=repeat_index,
                        repetitions=args.repetitions,
                        base_provider_config=original_provider_config,
                        base_eval_config=original_eval_config,
                        output_dir=args.output_dir,
                        limit=args.limit,
                        start_port=args.port,
                    )
                )
    finally:
        write_yaml(PROVIDER_CONFIG_PATH, original_provider_config)
        write_yaml(EVAL_CONFIG_PATH, original_eval_config)
        write_yaml(ACTIVE_STRATEGY_CONFIG_PATH, original_strategy_config)

    write_json(args.output_dir / "summary.json", summaries)
    write_summary_csv(args.output_dir / "summary.csv", summaries)

    if not args.skip_analysis:
        from analyze_experiments import analyze

        analyze(
            run_dir=args.output_dir.resolve(),
            pricing_preset=args.pricing_preset,
            prompt_price_per_1m=args.prompt_price_per_1m,
            output_price_per_1m=args.output_price_per_1m,
        )

    print("\n=== All experiments finished ===")
    print(f"output dir: {args.output_dir}")
    print(f"summary: {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
