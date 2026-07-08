from abc import ABC, abstractmethod
from typing import Any


class EvaluatorI(ABC):
    @abstractmethod
    def evaluate(self) -> dict[str, Any]:
        raise NotImplementedError
