from typing import Literal

from pydantic import BaseModel, Field


class BaseCompletionRequest(BaseModel):
    model: str
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    stop: list[str] | str | None = None
    seed: int | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class CompletionRequest(BaseCompletionRequest):
    prompt: str


class ChatCompletionRequest(BaseCompletionRequest):
    messages: list[ChatMessage]
