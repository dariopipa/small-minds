from abc import ABC, abstractmethod

from llm.responses import LLMResponse


class LLMClientI(ABC):
    @abstractmethod
    async def ensure_model_ready(self) -> None:
        raise NotImplementedError

    # Generate sends a single prompt and returns a single response.
    async def generate(self, prompt: str, stop: list[str] | None) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    async def chat_bot(self) -> str:
        raise NotImplementedError
