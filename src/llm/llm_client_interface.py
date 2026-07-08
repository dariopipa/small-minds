from abc import ABC, abstractmethod

from src.llm.schemas import LLMResponse


class LLMClientI(ABC):
    @abstractmethod
    async def load_model(self, model_name: str) -> str:
        raise NotImplementedError

    # Generate sends a single prompt and returns a single response.
    @abstractmethod
    async def generate(self, prompt: str) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    async def chat_bot(self) -> str:
        raise NotImplementedError
