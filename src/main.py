import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI

from agents.agent import Agent
from api.routes import routes
from evaluation.lm_eval_config import LLMEvalHarnessConfig
from evaluation.lm_eval_harness import LLMEvalHarness
from extractors.answer_extractor_interface import AnswerExtractorI
from extractors.extractor_factory import create_extractor
from llm.client_factory import LLMClientFactory
from llm.client_interface import LLMClientI
from llm.ollama.config import OllamaProviderConfig
from strategies.strategy_factory import StrategyFactory
from strategies.strategy_interface import StrategyI

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def load_provider_config() -> OllamaProviderConfig:
    with open(CONFIG_DIR / "provider.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return OllamaProviderConfig.model_validate(data)


def load_llm_eval_config() -> LLMEvalHarnessConfig:
    with open(CONFIG_DIR / "llm_eval_harness.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return LLMEvalHarnessConfig.model_validate(data)


def create_llm_client(provider_config: OllamaProviderConfig) -> LLMClientI:
    return LLMClientFactory.create(provider_config)


def create_answer_extractor(config: LLMEvalHarnessConfig) -> AnswerExtractorI:
    return create_extractor(config.tasks)


def create_agent(
    client: LLMClientI,
    answer_extractor: AnswerExtractorI,
) -> Agent:
    return Agent(
        llm_client=client,
        answer_extractor=answer_extractor,
    )


def run_strategy(
    strategy_name: str,
    agent: Agent,
) -> StrategyI:
    return StrategyFactory.create_strategy(strategy_config=strategy_name, agent=agent)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()

    provider_config = load_provider_config()
    eval_config = load_llm_eval_config()

    # TODO: REMOVE THIS LATER ON, DEBUGGING PURPOSES
    logger.info(
        "\n=================================================\n"
        "+++ FASTAPI SERVER STARTUP +++\n"
        f"provider: {provider_config.provider}\n"
        f"model: {provider_config.model_name}\n"
        f"provider options: {provider_config.options.to_dict()}\n"
        f"generate config: {provider_config.config.to_generate_kwargs()}\n"
        f"eval tasks: {eval_config.tasks}\n"
        f"eval endpoint: {eval_config.model_args.base_url}\n"
        "====================================================\n",
    )

    client = create_llm_client(provider_config)
    await client.ensure_model_ready()

    answer_extractor = create_answer_extractor(eval_config)
    agent = create_agent(client, answer_extractor)
    strategy = run_strategy(strategy_name="direct", agent=agent)

    app.state.strategy = strategy

    # TODO: REMOVE THIS LATER ON, DEBUGGING PURPOSES
    logger.info(
        "\n======================================================\n"
        "+++ FASTAPI SERVER READY +++\n"
        "agent loaded: yes\n"
        f"answer extractor tasks: {eval_config.tasks}\n"
        "completion endpoint: /v1/completions\n"
        "========================================================\n",
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
        # TODO: REMOVE THIS LATER ON, DEBUGGING PURPOSES
        logger.info(
            "\n========================================================================================\n"
            "+++ EVALUATION START +++\n"
            f"backend: {eval_config.backend}\n"
            f"base_url: {eval_config.model_args.base_url}\n"
            f"tasks: {eval_config.tasks}\n"
            f"num_fewshot: {eval_config.num_fewshot}\n"
            f"batch_size: {eval_config.batch_size}\n"
            f"limit: {eval_config.limit}\n"
            f"log_samples: {eval_config.log_samples}\n"
            f"write_out: {eval_config.write_out}\n"
            f"bootstrap_iters: {eval_config.bootstrap_iters}\n"
            "========================================================================================\n",
        )
        eval_harness = LLMEvalHarness(config=eval_config)

        results = eval_harness.evaluate()

        wall_end = time.perf_counter()
        cpu_end = time.process_time()

        wall_time = wall_end - wall_start
        cpu_time = cpu_end - cpu_start
        wait_time = wall_time - cpu_time

        gsm8k = results["results"]["gsm8k"]
        samples = results["n-samples"]["gsm8k"]
        n_shot = results["n-shot"]["gsm8k"]

        # TODO: REMOVE THIS LATER ON, DEBUGGING PURPOSES
        logger.info(
            "\n========================================================================================\n"
            "+++ GSM8K EVALUATION RESULTS +++\n"
            f"samples evaluated: {samples['effective']} / {samples['original']}\n"
            f"few-shot examples: {n_shot}\n"
            f"exact match strict: {float(gsm8k['exact_match,strict-match']):.4f}\n"
            "exact match flexible: "
            f"{float(gsm8k['exact_match,flexible-extract']):.4f}\n"
            f"strict stderr: {gsm8k['exact_match_stderr,strict-match']}\n"
            f"flexible stderr: {gsm8k['exact_match_stderr,flexible-extract']}\n"
            "========================================================================================\n",
        )

        # TODO: REMOVE THIS LATER ON, DEBUGGING PURPOSES
        logger.info(
            "\n========================================================================================\n"
            "+++ EVALUATION TIMING +++\n"
            f"wall time: {wall_time:.2f} seconds\n"
            f"cpu time: {cpu_time:.2f} seconds\n"
            f"wait time: {wait_time:.2f} seconds\n"
            "========================================================================================\n",
        )

    except KeyboardInterrupt:
        sys.exit(1)


if __name__ == "__main__":
    main()
