import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from pydantic import TypeAdapter, ValidationError

from agents.agent import Agent
from agents.factory import AgentFactory
from agents.models import AgentConfig
from api.routes import routes
from common.exceptions import (
    ConfigurationError,
    ModelLoadException,
    ModelNotFoundException,
)
from common.logging_config import configure_logging
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
from evaluation.lm_eval_harness import LLMEvalHarness
from extractors.base import AnswerExtractor
from extractors.factory import create_extractor
from llm.base import LLMClient
from llm.factory import LLMClientFactory
from llm.ollama.config import OllamaProviderConfig
from prompts import read_prompt
from strategies.base import Strategy
from strategies.factory import StrategyFactory
from strategies.models import SUPPORTED_STRATEGY_NAMES, StrategyConfig

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
SOURCE_DIR = CONFIG_DIR.parent
PROVIDER_CONFIG_PATH = CONFIG_DIR / "provider.yaml"
BENCHMARKS_CONFIG_PATH = CONFIG_DIR / "benchmarks.yaml"
EXPERIMENT_CONFIG_PATH = CONFIG_DIR / "experiments.yaml"


def selected_experiment_name() -> str | None:
    return os.getenv("EXPERIMENT")


def load_ollama_provider_config() -> OllamaProviderConfig:
    settings, _, selected = load_selected_experiment()
    generation_options = selected.strategy.generation.model_dump(exclude_none=True)
    data = {
        "provider": settings.provider.name,
        "model_name": settings.provider.model,
        "options": {
            "context_window": settings.provider.context_window,
            "max_output_tokens": settings.provider.max_output_tokens,
            **generation_options,
        },
        "config": {
            "stream": False,
            "think": False,
            "keep_alive": settings.provider.keep_alive,
        },
    }
    logger.debug(
        "Resolved provider config for experiment=%s strategy=%s",
        selected.name,
        selected.strategy_name,
    )
    return OllamaProviderConfig.model_validate(data)


def load_llm_eval_config() -> LLMEvalHarnessConfig:
    settings, plan, selected = load_selected_experiment()

    system_instruction = None
    if selected.benchmark.prompt is not None:
        system_instruction = read_prompt(SOURCE_DIR / selected.benchmark.prompt)

    data = {
        "backend": settings.evaluation.backend,
        "model_args": {
            "base_url": evaluation_base_url(settings),
            "tokenizer_backend": "none",
            "tokenized_requests": False,
            "eos_string": "<|im_end|>",
            "num_concurrent": settings.evaluation.concurrency,
            "timeout": settings.evaluation.timeout,
        },
        "system_instruction": system_instruction,
        "tasks": [selected.benchmark.task],
        "num_fewshot": selected.benchmark.num_fewshot,
        "batch_size": settings.evaluation.batch_size,
        "limit": plan.run.questions,
        "log_samples": settings.evaluation.log_samples,
        "write_out": settings.evaluation.write_out,
        "bootstrap_iters": settings.evaluation.bootstrap_iters,
    }
    logger.debug(
        "Resolved evaluation config for experiment=%s benchmark=%s",
        selected.name,
        selected.benchmark_name,
    )
    return LLMEvalHarnessConfig.model_validate(data)


def load_strategy_config() -> StrategyConfig:
    _, _, selected = load_selected_experiment()
    data = {"name": selected.strategy_name, **selected.strategy.params}
    try:
        return TypeAdapter(StrategyConfig).validate_python(data)
    except ValidationError as exc:
        available = "\n".join(f"  {name}" for name in SUPPORTED_STRATEGY_NAMES)
        raise ConfigurationError(
            f"Invalid or unsupported strategy '{selected.strategy_name}' "
            f"in {EXPERIMENT_CONFIG_PATH}.\n\n"
            f"Implemented strategies:\n{available}\n\n"
            f"Details:\n{exc}"
        ) from exc


def load_selected_experiment() -> tuple[
    ProviderConfig,
    ExperimentConfig,
    Experiment,
]:
    settings = load_provider_config(PROVIDER_CONFIG_PATH)
    benchmarks = load_benchmark_config(BENCHMARKS_CONFIG_PATH)
    experiment = load_experiment_config(EXPERIMENT_CONFIG_PATH)
    selected = resolve_experiment(experiment, benchmarks, selected_experiment_name())

    return settings, experiment, selected


def evaluation_base_url(settings: ProviderConfig) -> str:
    match settings.evaluation.backend:
        case "local-completions":
            return (
                f"http://{settings.server.host}:{settings.server.port}/v1/completions"
            )


def create_llm_client(provider_config: OllamaProviderConfig) -> LLMClient:
    return LLMClientFactory.create(provider_config)


def create_answer_extractor() -> AnswerExtractor:
    _, _, selected = load_selected_experiment()
    return create_extractor(selected.benchmark.task)


def create_agent(
    agent_config: AgentConfig,
    llm_client: LLMClient,
    answer_extractor: AnswerExtractor,
) -> Agent:
    return AgentFactory.create(
        agent_config=agent_config,
        llm_client=llm_client,
        answer_extractor=answer_extractor,
    )


def create_strategy(strategy_config: StrategyConfig, agent: Agent) -> Strategy:
    return StrategyFactory.create_strategy(strategy_config=strategy_config, agent=agent)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()

    try:
        provider_config = load_ollama_provider_config()
        eval_config = load_llm_eval_config()
        strategy_config = load_strategy_config()
        _, _, selected = load_selected_experiment()
        agent_config = AgentConfig(name=strategy_config.name, role="solver")

        logger.info(
            "Starting API: experiment=%s provider=%s model=%s strategy=%s tasks=%s",
            selected.name,
            provider_config.provider,
            provider_config.model_name,
            strategy_config.name,
            ",".join(eval_config.tasks),
        )

        llm_client = create_llm_client(provider_config)
        await llm_client.ensure_model_ready()

        answer_extractor = create_answer_extractor()
        agent = create_agent(
            agent_config=agent_config,
            llm_client=llm_client,
            answer_extractor=answer_extractor,
        )
        strategy = create_strategy(strategy_config=strategy_config, agent=agent)

        app.state.strategy = strategy

        logger.info(
            "API ready: endpoint=/v1/completions strategy=%s",
            strategy_config.name,
        )
    except (ConfigurationError, ModelLoadException, ModelNotFoundException) as exc:
        logger.error("API startup failed: %s", exc)
        raise

    yield


app = FastAPI(lifespan=lifespan)
app.include_router(router=routes)


def main():
    configure_logging()

    try:
        wall_start = time.perf_counter()
        cpu_start = time.process_time()

        eval_config = load_llm_eval_config()
        _, _, selected = load_selected_experiment()
        logger.info(
            "Evaluation started: experiment=%s backend=%s tasks=%s fewshot=%d limit=%s",
            selected.name,
            eval_config.backend,
            ",".join(eval_config.tasks),
            eval_config.num_fewshot,
            eval_config.limit,
        )
        eval_harness = LLMEvalHarness(config=eval_config)

        results = eval_harness.evaluate()

        wall_end = time.perf_counter()
        cpu_end = time.process_time()

        wall_time = wall_end - wall_start
        cpu_time = cpu_end - cpu_start
        wait_time = wall_time - cpu_time

        logger.info(
            "Evaluation finished: tasks=%s wall=%.2fs cpu=%.2fs wait=%.2fs",
            ",".join(results.get("results", {})),
            wall_time,
            cpu_time,
            wait_time,
        )

    except KeyboardInterrupt:
        sys.exit(1)
    except ConfigurationError as exc:
        logger.error("%s", exc)
        sys.exit(2)
    except (ModelLoadException, ModelNotFoundException) as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
