import logging
import os
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from common.exceptions import ConfigurationError
from configs.experiments import (
    BenchmarkConfig,
    Experiment,
    ExperimentConfig,
    ProviderConfig,
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


def load_experiment() -> tuple[ProviderConfig, ExperimentConfig, Experiment]:
    provider = load_config(PROVIDER_CONFIG_PATH, ProviderConfig)
    benchmarks = load_config(BENCHMARKS_CONFIG_PATH, BenchmarkConfig)
    config = load_config(EXPERIMENT_CONFIG_PATH, ExperimentConfig)
    selected = resolve_experiment(config, benchmarks, os.getenv("EXPERIMENT"))
    return provider, config, selected


def build_ollama_config(
    provider: ProviderConfig,
    experiment: Experiment,
) -> OllamaProviderConfig:
    generation_options = experiment.strategy.generation.model_dump(exclude_none=True)
    logger.debug(
        "Resolved provider config for experiment=%s strategy=%s",
        experiment.name,
        experiment.strategy_name,
    )
    return OllamaProviderConfig(
        provider=provider.provider.name,
        model_name=provider.provider.model,
        options=OllamaModelOptions(
            num_ctx=provider.provider.context_window,
            num_predict=provider.provider.max_output_tokens,
            **generation_options,
        ),
        config=OllamaConfig(
            stream=False,
            think=False,
            keep_alive=provider.provider.keep_alive,
        ),
    )


def build_llm_eval_config(
    provider: ProviderConfig,
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
        backend=provider.evaluation.backend,
        model_args=LocalCompletionsModelArgs(
            base_url=evaluation_base_url(provider),
            tokenizer_backend="none",
            tokenized_requests=False,
            eos_string="<|im_end|>",
            num_concurrent=provider.evaluation.concurrency,
            timeout=provider.evaluation.timeout,
        ),
        system_instruction=system_instruction,
        tasks=[experiment.benchmark.task],
        num_fewshot=experiment.benchmark.num_fewshot,
        batch_size=provider.evaluation.batch_size,
        limit=question_limit,
        log_samples=provider.evaluation.log_samples,
        write_out=provider.evaluation.write_out,
        bootstrap_iters=provider.evaluation.bootstrap_iters,
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


def evaluation_base_url(provider: ProviderConfig) -> str:
    return f"http://{provider.server.host}:{provider.server.port}/v1/completions"
