from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar

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
    repetition_seeds: list[int]
    output_dir: str = Field(default="results", min_length=1)


class Generation(ConfigModel):
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None


class Strategy(ConfigModel):
    generation: Generation = Field(default_factory=Generation)
    num_fewshot: int | None = Field(default=None, ge=0)
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
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if data is None:
            raise ConfigurationError(f"Configuration file is empty: {path}")
        return model.model_validate(data)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise ConfigurationError(f"Invalid configuration in {path}: {exc}") from exc


def validate_matrix(names: list[str], available: dict, kind: str) -> None:
    if len(names) != len(set(names)):
        raise ConfigurationError(f"Duplicate {kind}s in the experiment matrix.")

    unknown = [name for name in names if name not in available]
    if unknown:
        raise ConfigurationError(f"Unknown {kind}s: {', '.join(unknown)}")


def iter_experiments(
    experiment: ExperimentConfig,
    benchmarks: BenchmarkConfig,
) -> Iterator[Experiment]:
    validate_matrix(experiment.matrix.benchmarks, benchmarks.benchmarks, "benchmark")
    validate_matrix(experiment.matrix.strategies, experiment.strategies, "strategy")

    for benchmark_name in experiment.matrix.benchmarks:
        benchmark = benchmarks.benchmarks[benchmark_name]
        for strategy_name in experiment.matrix.strategies:
            yield Experiment(
                benchmark_name=benchmark_name,
                benchmark=benchmark,
                strategy_name=strategy_name,
                strategy=experiment.strategies[strategy_name],
            )
