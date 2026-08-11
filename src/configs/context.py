import logging
import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from common.exceptions import ConfigurationError
from configs.experiments import (
    Experiment,
    ExperimentConfig,
    ProviderConfig,
    load_benchmark_config,
    load_experiment_config,
    load_provider_config,
    resolve_experiment,
)
from evaluation.lm_eval_config import LLMEvalHarnessConfig
from llm.ollama.config import OllamaProviderConfig
from prompts import read_prompt
from strategies.models import SUPPORTED_STRATEGY_NAMES, StrategyConfig

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent
SOURCE_DIR = CONFIG_DIR.parent
PROVIDER_CONFIG_PATH = CONFIG_DIR / "provider.yaml"
BENCHMARKS_CONFIG_PATH = CONFIG_DIR / "benchmarks.yaml"
EXPERIMENT_CONFIG_PATH = CONFIG_DIR / "experiments.yaml"


@dataclass(frozen=True)
class ExperimentContext:
    provider_settings: ProviderConfig
    experiment_config: ExperimentConfig
    selected: Experiment


def selected_experiment_name() -> str | None:
    return os.getenv("EXPERIMENT")


def load_experiment_context() -> ExperimentContext:
    settings = load_provider_config(PROVIDER_CONFIG_PATH)
    benchmarks = load_benchmark_config(BENCHMARKS_CONFIG_PATH)
    experiment = load_experiment_config(EXPERIMENT_CONFIG_PATH)
    selected = resolve_experiment(experiment, benchmarks, selected_experiment_name())

    return ExperimentContext(
        provider_settings=settings,
        experiment_config=experiment,
        selected=selected,
    )


def load_ollama_provider_config(context: ExperimentContext) -> OllamaProviderConfig:
    generation_options = context.selected.strategy.generation.model_dump(
        exclude_none=True
    )
    data = {
        "provider": context.provider_settings.provider.name,
        "model_name": context.provider_settings.provider.model,
        "options": {
            "context_window": context.provider_settings.provider.context_window,
            "max_output_tokens": context.provider_settings.provider.max_output_tokens,
            **generation_options,
        },
        "config": {
            "stream": False,
            "think": False,
            "keep_alive": context.provider_settings.provider.keep_alive,
        },
    }
    logger.debug(
        "Resolved provider config for experiment=%s strategy=%s",
        context.selected.name,
        context.selected.strategy_name,
    )
    return OllamaProviderConfig.model_validate(data)


def load_llm_eval_config(context: ExperimentContext) -> LLMEvalHarnessConfig:
    system_instruction = None
    if context.selected.benchmark.prompt is not None:
        system_instruction = read_prompt(SOURCE_DIR / context.selected.benchmark.prompt)

    data = {
        "backend": context.provider_settings.evaluation.backend,
        "model_args": {
            "base_url": evaluation_base_url(context.provider_settings),
            "tokenizer_backend": "none",
            "tokenized_requests": False,
            "eos_string": "<|im_end|>",
            "num_concurrent": context.provider_settings.evaluation.concurrency,
            "timeout": context.provider_settings.evaluation.timeout,
        },
        "system_instruction": system_instruction,
        "tasks": [context.selected.benchmark.task],
        "num_fewshot": context.selected.benchmark.num_fewshot,
        "batch_size": context.provider_settings.evaluation.batch_size,
        "limit": context.experiment_config.run.questions,
        "log_samples": context.provider_settings.evaluation.log_samples,
        "write_out": context.provider_settings.evaluation.write_out,
        "bootstrap_iters": context.provider_settings.evaluation.bootstrap_iters,
    }
    logger.debug(
        "Resolved evaluation config for experiment=%s benchmark=%s",
        context.selected.name,
        context.selected.benchmark_name,
    )
    return LLMEvalHarnessConfig.model_validate(data)


def load_strategy_config(context: ExperimentContext) -> StrategyConfig:
    data = {
        "name": context.selected.strategy_name,
        **context.selected.strategy.params,
    }
    try:
        return TypeAdapter(StrategyConfig).validate_python(data)
    except ValidationError as exc:
        available = "\n".join(f"  {name}" for name in SUPPORTED_STRATEGY_NAMES)
        raise ConfigurationError(
            f"Invalid or unsupported strategy '{context.selected.strategy_name}' "
            f"in {EXPERIMENT_CONFIG_PATH}.\n\n"
            f"Implemented strategies:\n{available}\n\n"
            f"Details:\n{exc}"
        ) from exc


def evaluation_base_url(settings: ProviderConfig) -> str:
    match settings.evaluation.backend:
        case "local-completions":
            return (
                f"http://{settings.server.host}:{settings.server.port}/v1/completions"
            )
