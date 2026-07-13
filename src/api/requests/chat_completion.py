from typing import Literal

from pydantic import BaseModel


class BaseCompletionRequest(BaseModel):
    stop: list[str] | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class CompletionRequest(BaseCompletionRequest):
    prompt: str


class ChatCompletionRequest(BaseCompletionRequest):
    messages: list[ChatMessage]
