from abc import ABC, abstractmethod

from llm.requests import GenerateRequest
from llm.responses import LLMResponse


class LLMClientI(ABC):
    @abstractmethod
    async def ensure_model_ready(self) -> None:
        raise NotImplementedError

    # Generate sends a single prompt and returns a single response.
    async def generate(self, generation_request: GenerateRequest) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    async def chat_bot(self) -> str:
        raise NotImplementedError
