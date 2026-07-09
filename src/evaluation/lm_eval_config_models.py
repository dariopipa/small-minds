from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class LocalCompletionsModelArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    base_url: str
    tokenizer_backend: Literal["none"] = "none"
    tokenized_requests: bool = False
    eos_string: str | None = None
    num_concurrent: int = Field(default=1, ge=1)
    timeout: int = Field(default=180, ge=1)


# class GenerationKwargs(BaseModel):
#     model_config = ConfigDict(extra="forbid")

#     temperature: float = Field(default=0.0, ge=0.0)
#     max_tokens: int = Field(default=256, ge=1)


class LLMEvalHarnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["local-completions", "local-chat-completions"] = (
        "local-completions"
    )
    model_args: LocalCompletionsModelArgs

    system_instruction: str | None = None

    tasks: list[str] = Field(min_length=1)
    num_fewshot: int = Field(default=0, ge=0)
    batch_size: int | str = 1
    limit: int | None = Field(default=None, ge=1)

    log_samples: bool = False
    write_out: bool = False
    bootstrap_iters: int = Field(default=0, ge=0)

    # gen_kwargs: GenerationKwargs = Field(default_factory=GenerationKwargs)
