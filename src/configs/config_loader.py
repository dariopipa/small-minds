from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from common.exceptions import ConfigurationError
from configs.experiments import (
    ApplicationSettings,
    BenchmarkConfig,
    Experiment,
    ExperimentConfig,
    iter_experiments,
    load_config,
)
from strategies.models import SUPPORTED_STRATEGY_NAMES, StrategyConfig

CONFIG_DIR = Path(__file__).resolve().parent
PROVIDER_CONFIG_PATH = CONFIG_DIR / "provider.yaml"
BENCHMARKS_CONFIG_PATH = CONFIG_DIR / "benchmarks.yaml"
EXPERIMENT_CONFIG_PATH = CONFIG_DIR / "experiments.yaml"


def load_experiments(
    benchmark_name: str | None = None,
    strategy_name: str | None = None,
) -> tuple[ApplicationSettings, ExperimentConfig, list[Experiment]]:
    settings = load_config(PROVIDER_CONFIG_PATH, ApplicationSettings)
    benchmarks = load_config(BENCHMARKS_CONFIG_PATH, BenchmarkConfig)
    config = load_config(EXPERIMENT_CONFIG_PATH, ExperimentConfig)
    experiments = list(iter_experiments(config, benchmarks))
    if benchmark_name is not None:
        experiments = [
            experiment
            for experiment in experiments
            if experiment.benchmark_name == benchmark_name
        ]
        if not experiments:
            available = ", ".join(config.matrix.benchmarks)
            raise ConfigurationError(
                f"Unknown or unconfigured benchmark: {benchmark_name}. "
                f"Available benchmarks: {available}"
            )
    if strategy_name is not None:
        experiments = [
            experiment
            for experiment in experiments
            if experiment.strategy_name == strategy_name
        ]
        if not experiments:
            available = ", ".join(config.matrix.strategies)
            raise ConfigurationError(
                f"Unknown or unconfigured strategy: {strategy_name}. "
                f"Available strategies: {available}"
            )
    return settings, config, experiments


def build_strategy_config(experiment: Experiment) -> StrategyConfig:
    try:
        return TypeAdapter(StrategyConfig).validate_python(
            {"name": experiment.strategy_name, **experiment.strategy.params}
        )
    except ValidationError as exc:
        available = "\n".join(f"  {name}" for name in SUPPORTED_STRATEGY_NAMES)
        raise ConfigurationError(
            f"Invalid or unsupported strategy '{experiment.strategy_name}' "
            f"in {EXPERIMENT_CONFIG_PATH}.\n\n"
            f"Implemented strategies:\n{available}\n\n"
            f"Details:\n{exc}"
        ) from exc
