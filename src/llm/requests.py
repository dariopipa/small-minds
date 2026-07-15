from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str
    stop: list[str] | None = None
