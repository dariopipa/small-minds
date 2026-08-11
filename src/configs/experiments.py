from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError

from common.exceptions import ConfigurationError


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Provider(ConfigModel):
    name: Literal["ollama"]
    model: str
    context_window: PositiveInt
    max_output_tokens: PositiveInt
    keep_alive: str


class Evaluation(ConfigModel):
    backend: Literal["local-completions"]
    batch_size: int | str = 1
    concurrency: PositiveInt = 1
    timeout: PositiveInt = 180
    log_samples: bool = False
    write_out: bool = False
    bootstrap_iters: int = Field(default=0, ge=0)


class Server(ConfigModel):
    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)


class ProviderConfig(ConfigModel):
    provider: Provider
    evaluation: Evaluation
    server: Server


class Benchmark(ConfigModel):
    task: str = Field(min_length=1)
    prompt: str | None = Field(default=None, min_length=1)
    num_fewshot: int = Field(default=0, ge=0)


class BenchmarkConfig(ConfigModel):
    benchmarks: dict[str, Benchmark] = Field(min_length=1)


class Run(ConfigModel):
    questions: PositiveInt
    repetitions: PositiveInt


class Generation(ConfigModel):
    seed: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None


class Strategy(ConfigModel):
    generation: Generation = Field(default_factory=Generation)
    params: dict[str, int | str | float | bool | None] = Field(default_factory=dict)


class Matrix(ConfigModel):
    benchmarks: list[str] = Field(min_length=1)
    strategies: list[str] = Field(min_length=1)


class ExperimentConfig(ConfigModel):
    run: Run
    strategies: dict[str, Strategy] = Field(min_length=1)
    matrix: Matrix


@dataclass(frozen=True)
class Experiment:
    name: str
    benchmark_name: str
    benchmark: Benchmark
    strategy_name: str
    strategy: Strategy


def load_provider_config(path: Path) -> ProviderConfig:
    try:
        return ProviderConfig.model_validate(load_yaml(path))
    except ValidationError as exc:
        raise ConfigurationError(
            f"Invalid provider configuration in {path}:\n{exc}"
        ) from exc


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    try:
        return BenchmarkConfig.model_validate(load_yaml(path))
    except ValidationError as exc:
        raise ConfigurationError(
            f"Invalid benchmark configuration in {path}:\n{exc}"
        ) from exc


def load_experiment_config(path: Path) -> ExperimentConfig:
    try:
        return ExperimentConfig.model_validate(load_yaml(path))
    except ValidationError as exc:
        raise ConfigurationError(
            f"Invalid experiment configuration in {path}:\n{exc}"
        ) from exc


def load_yaml(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as source:
            data = yaml.safe_load(source)
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Configuration file does not exist: {path}") from exc
    except OSError as exc:
        raise ConfigurationError(
            f"Could not read configuration file {path}: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in configuration file: {path}") from exc

    if data is None:
        raise ConfigurationError(f"Configuration file is empty: {path}")

    return data


def validate_config(experiment: ExperimentConfig, benchmarks: BenchmarkConfig) -> None:
    duplicate_benchmarks = find_duplicates(experiment.matrix.benchmarks)
    if duplicate_benchmarks:
        raise ConfigurationError(
            "Duplicate benchmark matrix entries: " + ", ".join(duplicate_benchmarks)
        )

    duplicate_strategies = find_duplicates(experiment.matrix.strategies)
    if duplicate_strategies:
        raise ConfigurationError(
            "Duplicate strategy matrix entries: " + ", ".join(duplicate_strategies)
        )

    missing_benchmarks = set(experiment.matrix.benchmarks) - set(benchmarks.benchmarks)
    if missing_benchmarks:
        available = "\n".join(f"  {name}" for name in sorted(benchmarks.benchmarks))
        raise ConfigurationError(
            "Unknown benchmarks in experiment matrix: "
            + ", ".join(sorted(missing_benchmarks))
            + f"\n\nAvailable benchmarks:\n{available}"
        )

    missing_strategies = set(experiment.matrix.strategies) - set(experiment.strategies)
    if missing_strategies:
        available = "\n".join(f"  {name}" for name in sorted(experiment.strategies))
        raise ConfigurationError(
            "Unknown strategies in experiment matrix: "
            + ", ".join(sorted(missing_strategies))
            + f"\n\nAvailable configured strategies:\n{available}"
        )


def iter_experiments(
    experiment: ExperimentConfig,
    benchmarks: BenchmarkConfig,
) -> Iterator[Experiment]:
    validate_config(experiment, benchmarks)

    for benchmark_name in experiment.matrix.benchmarks:
        benchmark = benchmarks.benchmarks[benchmark_name]
        for strategy_name in experiment.matrix.strategies:
            yield Experiment(
                name=f"{strategy_name}_{benchmark_name}",
                benchmark_name=benchmark_name,
                benchmark=benchmark,
                strategy_name=strategy_name,
                strategy=experiment.strategies[strategy_name],
            )


def resolve_experiment(
    experiment: ExperimentConfig,
    benchmarks: BenchmarkConfig,
    experiment_name: str | None,
) -> Experiment:
    experiments = list(iter_experiments(experiment, benchmarks))
    if experiment_name is None:
        return experiments[0]

    for item in experiments:
        if item.name == experiment_name:
            return item

    available = ", ".join(item.name for item in experiments)
    raise ConfigurationError(
        f"Experiment is not defined: {experiment_name}. "
        f"Available experiments: {available}"
    )


def find_duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    repeated: set[str] = set()
    for value in values:
        if value in seen:
            repeated.add(value)
        seen.add(value)
    return sorted(repeated)
