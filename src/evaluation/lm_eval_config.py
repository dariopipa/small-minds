from pathlib import Path
from typing import Literal
from urllib.parse import urlencode

from pydantic import BaseModel, ConfigDict, Field

from configs.experiments import ApplicationSettings, Experiment
from prompts import load_prompt

SOURCE_DIR = Path(__file__).resolve().parents[1]


class LocalCompletionsModelArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str
    tokenizer_backend: Literal["none"] = "none"
    tokenized_requests: bool = False
    eos_string: str | None = None
    num_concurrent: int = Field(default=1, ge=1)
    timeout: int = Field(default=180, ge=1)


class LLMEvalHarnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["local-completions"] = "local-completions"
    model_args: LocalCompletionsModelArgs

    system_instruction: str | None = None

    tasks: list[str] = Field(min_length=1)
    num_fewshot: int = Field(default=0, ge=0)
    batch_size: int | str = 1
    limit: int | None = Field(default=None, ge=1)

    write_out: bool = False
    bootstrap_iters: int = Field(default=0, ge=0)
    cache_requests: bool = True


def build_llm_eval_config(
    settings: ApplicationSettings,
    experiment: Experiment,
    question_limit: int,
    repetition: int = 1,
    repetition_seed: int | None = None,
) -> LLMEvalHarnessConfig:
    system_instruction = (
        f"{load_prompt(SOURCE_DIR / experiment.benchmark.prompt)}\n\n"
        if experiment.benchmark.prompt
        else None
    )
    return LLMEvalHarnessConfig(
        backend=settings.evaluation.backend,
        model_args=LocalCompletionsModelArgs(
            base_url=evaluation_base_url(
                settings,
                experiment.name,
                repetition,
                repetition_seed,
            ),
            tokenizer_backend="none",
            tokenized_requests=False,
            eos_string="<|im_end|>",
            num_concurrent=settings.evaluation.concurrency,
            timeout=settings.evaluation.timeout,
        ),
        system_instruction=system_instruction,
        tasks=[experiment.benchmark.task],
        num_fewshot=(
            experiment.strategy.num_fewshot
            if experiment.strategy.num_fewshot is not None
            else experiment.benchmark.num_fewshot
        ),
        batch_size=settings.evaluation.batch_size,
        limit=question_limit,
        write_out=settings.evaluation.write_out,
        bootstrap_iters=settings.evaluation.bootstrap_iters,
    )


def evaluation_base_url(
    settings: ApplicationSettings,
    experiment_name: str | None = None,
    repetition: int = 1,
    repetition_seed: int | None = None,
) -> str:
    base_url = f"http://{settings.server.host}:{settings.server.port}/v1/completions"
    if experiment_name is None:
        return base_url
    query = {"experiment": experiment_name, "repetition": repetition}
    if repetition_seed is not None:
        query["repetition_seed"] = repetition_seed
    return f"{base_url}?{urlencode(query)}"
