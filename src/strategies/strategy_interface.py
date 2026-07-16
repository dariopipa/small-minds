from abc import ABC, abstractmethod

from llm.requests import GenerateRequest


class StrategyI(ABC):
    @abstractmethod
    async def run(self, generation_request: GenerateRequest):
        raise NotImplementedError
