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
from configs.config_loader import (
    build_strategy_config,
    load_experiments,
)
from extractors.factory import create_extractor
from llm.factory import LLMClientFactory
from llm.ollama.config import build_ollama_config
from strategies.base import Strategy
from strategies.factory import StrategyFactory

logger = logging.getLogger(__name__)


async def build_strategies() -> dict[str, Strategy]:
    settings, _, experiments = load_experiments()
    strategies: dict[str, Strategy] = {}
    # The provider model is shared by every configured experiment.
    model_warmed = False

    for experiment in experiments:
        provider_config = build_ollama_config(settings, experiment)
        strategy_config = build_strategy_config(experiment)
        logger.info(
            "Loading experiment=%s provider=%s model=%s strategy=%s task=%s",
            experiment.name,
            provider_config.provider,
            provider_config.model_name,
            strategy_config.name,
            experiment.benchmark.task,
        )

        llm_client = LLMClientFactory.create(provider_config)
        if not model_warmed:
            await llm_client.ensure_model_ready()
            model_warmed = True

        strategies[experiment.name] = StrategyFactory.create_strategy(
            strategy_config=strategy_config,
            agent_factory=AgentFactory(
                llm_client=llm_client,
                answer_extractor=create_extractor(experiment.benchmark.task),
                base_seed=provider_config.options.seed,
                base_temperature=provider_config.options.temperature,
            ),
        )

    return strategies


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()

    try:
        strategies = await build_strategies()
        app.state.strategies = strategies
        app.state.default_experiment = next(iter(strategies))

        logger.info(
            "API ready: endpoint=/v1/completions experiments=%s",
            ", ".join(strategies),
        )
    except (ConfigurationError, ModelLoadException, ModelNotFoundException) as exc:
        logger.error("API startup failed: %s", exc)
        raise

    yield


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(router=routes)
    return app
