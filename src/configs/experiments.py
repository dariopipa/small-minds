from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError

from common.exceptions import ConfigurationError

T = TypeVar("T", bound=BaseModel)


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProviderSettings(ConfigModel):
    name: Literal["ollama"]
    model: str
    context_window: PositiveInt
    max_output_tokens: PositiveInt
    keep_alive: str


class EvaluationSettings(ConfigModel):
    backend: Literal["local-completions"]
    batch_size: int | str = 1
    concurrency: PositiveInt = 1
    timeout: PositiveInt = 180
    log_samples: bool = False
    write_out: bool = False
    bootstrap_iters: int = Field(default=0, ge=0)


class ServerSettings(ConfigModel):
    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)


class ApplicationSettings(ConfigModel):
    provider: ProviderSettings
    evaluation: EvaluationSettings
    server: ServerSettings


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
    benchmark_name: str
    benchmark: Benchmark
    strategy_name: str
    strategy: Strategy

    @property
    def name(self) -> str:
        return f"{self.strategy_name}_{self.benchmark_name}"


def load_config(path: Path, model: type[T]) -> T:
    try:
        return model.model_validate(load_yaml(path))
    except ValidationError as exc:
        raise ConfigurationError(
            f"Invalid {model.__name__} configuration in {path}:\n{exc}"
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
    validate_names(experiment.matrix.benchmarks, benchmarks.benchmarks, "benchmark")
    validate_names(experiment.matrix.strategies, experiment.strategies, "strategy")


def validate_names(
    selected: list[str], available: Mapping[str, object], kind: str
) -> None:
    duplicates = find_duplicates(selected)
    if duplicates:
        raise ConfigurationError(
            f"Duplicate {kind} matrix entries: " + ", ".join(duplicates)
        )

    missing = sorted(set(selected) - set(available))
    if missing:
        available_names = "\n".join(f"  {name}" for name in sorted(available))
        raise ConfigurationError(
            f"Unknown {kind}s in experiment matrix: {', '.join(missing)}"
            f"\n\nAvailable {kind}s:\n{available_names}"
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
    experiments = iter_experiments(experiment, benchmarks)
    if experiment_name is None:
        return next(experiments)

    available: list[str] = []
    for item in experiments:
        if item.name == experiment_name:
            return item
        available.append(item.name)

    raise ConfigurationError(
        f"Experiment is not defined: {experiment_name}. "
        f"Available experiments: {', '.join(available)}"
    )


def find_duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    repeated: set[str] = set()
    for value in values:
        if value in seen:
            repeated.add(value)
        seen.add(value)
    return sorted(repeated)
