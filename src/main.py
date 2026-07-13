from contextlib import asynccontextmanager
import sys
import time

from fastapi import FastAPI
import yaml

from agents.agent import Agent
from api.routes import routes
from evaluation.lm_eval_config import LLMEvalHarnessConfig
from evaluation.lm_eval_harness import LLMEvalHarness
from extractors.answer_extractor_interface import AnswerExtractorI
from extractors.extractor_factory import create_extractor
from llm.client_interface import LLMClientI
from llm.client_factory import LLMClientFactory
from llm.ollama.config import OllamaProviderConfig


def load_provider_config() -> OllamaProviderConfig:
    with open("..\\src\\configs\\provider.yaml") as f:
        data = yaml.safe_load(f)

    return OllamaProviderConfig.model_validate(data)


def load_llm_eval_config() -> LLMEvalHarnessConfig:
    with open("..\\src\\configs\\llm_eval_harness.yaml") as f:
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    provider_config = load_provider_config()
    eval_config = load_llm_eval_config()

    client = create_llm_client(provider_config)
    await client.ensure_model_ready()

    answer_extractor = create_answer_extractor(eval_config)
    agent = create_agent(client, answer_extractor)

    app.state.agent = agent
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(router=routes)


def main():
    try:
        print("MAIN CALLED")
        wall_start = time.perf_counter()
        cpu_start = time.process_time()

        eval_config = load_llm_eval_config()
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

        print("\n=== GSM8K EVALUATION ===")
        print(f"Samples evaluated: {samples['effective']} / {samples['original']}")
        print(f"Few-shot examples: {n_shot}")

        print(f"Exact match strict:   {float(gsm8k['exact_match,strict-match']):.4f}")
        print(
            f"Exact match flexible: {float(gsm8k['exact_match,flexible-extract']):.4f}"
        )

        print(f"Strict stderr:   {gsm8k['exact_match_stderr,strict-match']}")
        print(f"Flexible stderr: {gsm8k['exact_match_stderr,flexible-extract']}")

        print("\n=== TIMING ===")
        print(f"Wall time: {wall_time:.2f} seconds")
        print(f"CPU time:  {cpu_time:.2f} seconds")
        print(f"Wait time: {wait_time:.2f} seconds")

    except KeyboardInterrupt:
        sys.exit(1)


if __name__ == "__main__":
    main()
