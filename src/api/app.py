import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agents.factory import AgentFactory
from api.routes import routes
from common.exceptions import (
    ConfigurationError,
    ModelLoadException,
    ModelNotFoundException,
)
from common.logging_config import configure_logging
from configs.context import (
    load_experiment_context,
    load_llm_eval_config,
    load_ollama_provider_config,
    load_strategy_config,
)
from extractors.factory import create_extractor
from llm.factory import LLMClientFactory
from strategies.base import Strategy
from strategies.factory import StrategyFactory

logger = logging.getLogger(__name__)


async def build_selected_strategy() -> tuple[Strategy, str]:
    context = load_experiment_context()
    provider_config = load_ollama_provider_config(context)
    eval_config = load_llm_eval_config(context)
    strategy_config = load_strategy_config(context)
    logger.info(
        "Starting API: experiment=%s provider=%s model=%s strategy=%s tasks=%s",
        context.selected.name,
        provider_config.provider,
        provider_config.model_name,
        strategy_config.name,
        ",".join(eval_config.tasks),
    )

    llm_client = LLMClientFactory.create(provider_config)
    await llm_client.ensure_model_ready()

    answer_extractor = create_extractor(context.selected.benchmark.task)
    strategy = StrategyFactory.create_strategy(
        strategy_config=strategy_config,
        agent_factory=AgentFactory(
            llm_client=llm_client,
            answer_extractor=answer_extractor,
        ),
    )

    return strategy, strategy_config.name


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()

    try:
        strategy, strategy_name = await build_selected_strategy()
        app.state.strategy = strategy

        logger.info(
            "API ready: endpoint=/v1/completions strategy=%s",
            strategy_name,
        )
    except (ConfigurationError, ModelLoadException, ModelNotFoundException) as exc:
        logger.error("API startup failed: %s", exc)
        raise

    yield


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(router=routes)
    return app
