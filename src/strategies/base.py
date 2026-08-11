from abc import ABC, abstractmethod

from llm.requests import GenerateRequest
from strategies.models import StrategyResult


class Strategy(ABC):
    @abstractmethod
    async def run(self, generation_request: GenerateRequest) -> StrategyResult:
        raise NotImplementedError
