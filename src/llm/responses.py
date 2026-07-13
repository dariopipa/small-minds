from pydantic import BaseModel, ConfigDict


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: str
    model: str
    thinking: str | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_duration_ns: int = 0

    @property
    def duration_s(self) -> float:
        return self.total_duration_ns / 1_000_000_000
