import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI
from pydantic import TypeAdapter

from agents.agent import Agent
from agents.factory import AgentFactory
from agents.models import AgentConfig
from api.routes import routes
from common.logging_config import configure_logging
from evaluation.lm_eval_config import LLMEvalHarnessConfig
from evaluation.lm_eval_harness import LLMEvalHarness
from extractors.base import AnswerExtractor
from extractors.factory import create_extractor
from llm.base import LLMClient
from llm.factory import LLMClientFactory
from llm.ollama.config import OllamaProviderConfig
from strategies.base import Strategy
from strategies.factory import StrategyFactory
from strategies.models import StrategyConfig

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def load_yaml_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_provider_config() -> OllamaProviderConfig:
    data = load_yaml_config(CONFIG_DIR / "provider.yaml")
    return OllamaProviderConfig.model_validate(data)


def load_llm_eval_config() -> LLMEvalHarnessConfig:
    data = load_yaml_config(CONFIG_DIR / "llm_eval_harness.yaml")
    return LLMEvalHarnessConfig.model_validate(data)


def load_strategy_config() -> StrategyConfig:
    data = load_yaml_config(CONFIG_DIR / "strategy.yaml")
    return TypeAdapter(StrategyConfig).validate_python(data)


def create_llm_client(provider_config: OllamaProviderConfig) -> LLMClient:
    return LLMClientFactory.create(provider_config)


def create_answer_extractor(config: LLMEvalHarnessConfig) -> AnswerExtractor:
    return create_extractor(config.tasks)


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

    provider_config = load_provider_config()
    eval_config = load_llm_eval_config()
    strategy_config = load_strategy_config()
    agent_config = AgentConfig(name=strategy_config.name, role="solver")

    logger.info(
        "Starting API: provider=%s model=%s strategy=%s tasks=%s",
        provider_config.provider,
        provider_config.model_name,
        strategy_config.name,
        ",".join(eval_config.tasks),
    )

    llm_client = create_llm_client(provider_config)
    await llm_client.ensure_model_ready()

    answer_extractor = create_answer_extractor(eval_config)
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
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(router=routes)


def main():
    configure_logging()

    try:
        wall_start = time.perf_counter()
        cpu_start = time.process_time()

        eval_config = load_llm_eval_config()
        logger.info(
            "Evaluation started: backend=%s tasks=%s fewshot=%d limit=%s",
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


if __name__ == "__main__":
    main()
