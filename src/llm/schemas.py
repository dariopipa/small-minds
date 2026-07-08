from dataclasses import asdict, dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# todo: make this a pydantic model
@dataclass
class LLMResponse:
    response: str
    model: str | None = None
    thinking: str | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_duration_ns: int = 0

    @property
    def duration_s(self) -> float:
        return self.total_duration_ns / 1_000_000_000


# todo: make this a pydantic model
@dataclass
class ModelOptions:
    seed: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    stop: list[str] | None = None
    num_ctx: int | None = None
    num_predict: int | None = None

    def to_dict(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}


# todo: bound to change to add more fields,template etc
class OllamaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stream: Literal[False] | None = None
    think: bool | Literal["low", "medium", "high"] | None = None
    keep_alive: float | str | None = None

    def to_generate_kwargs(self) -> dict:
        return self.model_dump(exclude_none=True)
