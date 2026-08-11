import logging
import os
from pathlib import Path
from urllib.parse import urlencode

from pydantic import TypeAdapter, ValidationError

from common.exceptions import ConfigurationError
from configs.experiments import (
    ApplicationSettings,
    BenchmarkConfig,
    Experiment,
    ExperimentConfig,
    iter_experiments,
    load_config,
    resolve_experiment,
)
from evaluation.lm_eval_config import LLMEvalHarnessConfig, LocalCompletionsModelArgs
from llm.ollama.config import OllamaConfig, OllamaModelOptions, OllamaProviderConfig
from prompts import read_prompt
from strategies.models import SUPPORTED_STRATEGY_NAMES, StrategyConfig

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent
SOURCE_DIR = CONFIG_DIR.parent
PROVIDER_CONFIG_PATH = CONFIG_DIR / "provider.yaml"
BENCHMARKS_CONFIG_PATH = CONFIG_DIR / "benchmarks.yaml"
EXPERIMENT_CONFIG_PATH = CONFIG_DIR / "experiments.yaml"


def load_experiments() -> tuple[
    ApplicationSettings, ExperimentConfig, list[Experiment]
]:
    settings = load_config(PROVIDER_CONFIG_PATH, ApplicationSettings)
    benchmarks = load_config(BENCHMARKS_CONFIG_PATH, BenchmarkConfig)
    config = load_config(EXPERIMENT_CONFIG_PATH, ExperimentConfig)
    experiment_name = os.getenv("EXPERIMENT")
    experiments = (
        [resolve_experiment(config, benchmarks, experiment_name)]
        if experiment_name is not None
        else list(iter_experiments(config, benchmarks))
    )
    return settings, config, experiments


def build_ollama_config(
    settings: ApplicationSettings,
    experiment: Experiment,
) -> OllamaProviderConfig:
    generation_options = experiment.strategy.generation.model_dump(exclude_none=True)
    logger.debug(
        "Resolved provider config for experiment=%s strategy=%s",
        experiment.name,
        experiment.strategy_name,
    )
    return OllamaProviderConfig(
        provider=settings.provider.name,
        model_name=settings.provider.model,
        options=OllamaModelOptions(
            num_ctx=settings.provider.context_window,
            num_predict=settings.provider.max_output_tokens,
            **generation_options,
        ),
        config=OllamaConfig(
            stream=False,
            think=False,
            keep_alive=settings.provider.keep_alive,
        ),
    )


def build_llm_eval_config(
    settings: ApplicationSettings,
    experiment: Experiment,
    question_limit: int,
) -> LLMEvalHarnessConfig:
    system_instruction = (
        read_prompt(SOURCE_DIR / experiment.benchmark.prompt)
        if experiment.benchmark.prompt
        else None
    )
    logger.debug(
        "Resolved evaluation config for experiment=%s benchmark=%s",
        experiment.name,
        experiment.benchmark_name,
    )
    return LLMEvalHarnessConfig(
        backend=settings.evaluation.backend,
        model_args=LocalCompletionsModelArgs(
            base_url=evaluation_base_url(settings, experiment.name),
            tokenizer_backend="none",
            tokenized_requests=False,
            eos_string="<|im_end|>",
            num_concurrent=settings.evaluation.concurrency,
            timeout=settings.evaluation.timeout,
        ),
        system_instruction=system_instruction,
        tasks=[experiment.benchmark.task],
        num_fewshot=experiment.benchmark.num_fewshot,
        batch_size=settings.evaluation.batch_size,
        limit=question_limit,
        log_samples=settings.evaluation.log_samples,
        write_out=settings.evaluation.write_out,
        bootstrap_iters=settings.evaluation.bootstrap_iters,
    )


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


def evaluation_base_url(
    settings: ApplicationSettings,
    experiment_name: str | None = None,
) -> str:
    base_url = f"http://{settings.server.host}:{settings.server.port}/v1/completions"
    if experiment_name is None:
        return base_url
    return f"{base_url}?{urlencode({'experiment': experiment_name})}"
