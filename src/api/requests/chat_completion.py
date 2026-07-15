from pydantic import BaseModel


class CompletionRequest(BaseModel):
    prompt: str
    stop: list[str] | None = None
