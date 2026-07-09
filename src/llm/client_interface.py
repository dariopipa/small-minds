from abc import ABC, abstractmethod

from llm.schemas import LLMResponse


class LLMClientI(ABC):
    @abstractmethod
    async def load_model(self) -> None:
        raise NotImplementedError

    # Generate sends a single prompt and returns a single response.
    async def generate(self, prompt: str) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    async def chat_bot(self) -> str:
        raise NotImplementedError
