from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class OllamaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stream: Literal[False] | None = None
    think: bool | Literal["low", "medium", "high"] | None = None
    keep_alive: str | None = None

    def to_generate_kwargs(self) -> dict:
        return self.model_dump(exclude_none=True)


class OllamaModelOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    stop: list[str] | None = None
    num_ctx: int | None = Field(
        default=None,
        validation_alias="context_window",
    )
    num_predict: int | None = Field(
        default=None,
        validation_alias="max_output_tokens",
    )

    def to_dict(self) -> dict:
        return self.model_dump(exclude_none=True)


class OllamaProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["ollama"] = "ollama"
    model_name: str
    options: OllamaModelOptions = Field(default_factory=OllamaModelOptions)
    config: OllamaConfig = Field(default_factory=OllamaConfig)
