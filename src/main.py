import sys
import time

from fastapi import FastAPI

from evaluation.lm_eval_config_models import (
    GenerationKwargs,
    LLMEvalHarnessConfig,
    LocalCompletionsModelArgs,
)
from src.evaluation.lm_eval_harness import LLMEvalHarness
from src.api.routes import routes


# TODO: REMOVE AND CENTRALIZE THE CONFIGURATION

MODEL_NAME = "qwen2.5:3b"
OLLAMA_CHAT_URL = "http://localhost:11434/v1"
BASE_URL = "http://127.0.0.1:8000/v1/completions"
app = FastAPI()

app.include_router(router=routes)


def main():
    try:
        # TODO: Remove
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
                num_fewshot=5,
                batch_size=1,
                limit=50,
                log_samples=False,
                write_out=False,
                bootstrap_iters=10,
                gen_kwargs=GenerationKwargs(
                    temperature=0.0,
                    max_tokens=256,
                ),
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
