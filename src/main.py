from contextlib import asynccontextmanager
import sys
import time

from fastapi import FastAPI
import yaml

from evaluation.lm_eval_config import (
    LLMEvalHarnessConfig,
)
from llm.client_interface import LLMClientI
from llm.factories import LLMClientFactory
from llm.ollama.config import OllamaProviderConfig
from evaluation.lm_eval_harness import LLMEvalHarness
from api.routes import routes


MODEL_NAME = "qwen2.5:3b"
OLLAMA_CHAT_URL = "http://localhost:11434/v1"

# THIS URL WILL POINT IT TO OUR LOCAL MODEL, 
# WHICH WILL ALLOW US TO RUN WHATEVER STRATEGY WE WANT AND ALSO SAVE BENCHMARK INFORMATION
EXPOSED_FASTAPI_URL = "http://127.0.0.1:8000/v1/completions"


def load_provider_config() -> OllamaProviderConfig:
    with open("..\\src\\configs\\provider.yaml") as f:
        data = yaml.safe_load(f)

    return OllamaProviderConfig.model_validate(data)


def load_llm_eval_config() -> LLMEvalHarnessConfig:
    with open("..\\src\\configs\\llm_eval_harness.yaml") as f:
        data = yaml.safe_load(f)

    return LLMEvalHarnessConfig.model_validate(data)


def create_client() -> LLMClientI:
    return LLMClientFactory.create(load_provider_config())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run at startup
    Initialize the Client and add it to request.state
    """
    client = create_client()
    await client.ensure_model_ready()
    app.state.llm_client = client
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(router=routes)


def main():
    try:
        # TODO: Remove
        print("MAIN CALLED")
        wall_start = time.perf_counter()
        cpu_start = time.process_time()

        eval_harness = LLMEvalHarness(config=load_llm_eval_config())

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
