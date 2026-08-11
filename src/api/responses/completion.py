from typing import Any, Literal

from pydantic import BaseModel


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    logprobs: dict[str, Any] | None = None
    finish_reason: Literal["stop"] = "stop"


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage
