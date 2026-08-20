from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str
    stop: list[str] | None = None
    seed: int | None = None
    temperature: float | None = None
    repetition: int = 1
