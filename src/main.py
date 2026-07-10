from contextlib import asynccontextmanager
import sys
import time

from fastapi import FastAPI

from evaluation.lm_eval_config_models import (
    LLMEvalHarnessConfig,
    LocalCompletionsModelArgs,
)
from llm.factories import LLMClientFactory
from llm.ollama.config import OllamaConfig, OllamaModelOptions, OllamaProviderConfig
from evaluation.lm_eval_harness import LLMEvalHarness
from api.routes import routes


# TODO: REMOVE AND CENTRALIZE THE CONFIGURATION

MODEL_NAME = "qwen2.5:3b"
OLLAMA_CHAT_URL = "http://localhost:11434/v1"
BASE_URL = "http://127.0.0.1:8000/v1/completions"


def create_client():
    return LLMClientFactory.create(
        config=OllamaProviderConfig(
            provider="ollama",
            model_name=MODEL_NAME,
            options=OllamaModelOptions(seed=42, temperature=0),
            config=OllamaConfig(stream=False, think=False, keep_alive="60m"),
        )
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run at startup
    Initialize the Client and add it to request.state
    """
    client = create_client()
    # await client.ensure_model_ready()
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

        eval_harness = LLMEvalHarness(
            config=LLMEvalHarnessConfig(
                backend="local-completions",
                model_args=LocalCompletionsModelArgs(
                    model=MODEL_NAME,
                    base_url=BASE_URL,
                    tokenizer_backend="none",
                    tokenized_requests=False,
                    eos_string="<|im_end|>",
                    num_concurrent=2,
                    timeout=180,
                ),
                system_instruction=(
                    "Solve the math problem step by step.\n"
                    "At the end, output the final answer on its own line in exactly this format:\n"
                    "#### <number>\n"
                    "Do not put anything after the final answer line.\n\n"
                ),
                tasks=["gsm8k"],
                num_fewshot=2,
                batch_size=1,
                limit=1,
                log_samples=False,
                write_out=False,
                bootstrap_iters=10,
            )
        )

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
