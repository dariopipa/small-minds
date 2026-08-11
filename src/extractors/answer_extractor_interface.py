from abc import ABC, abstractmethod


class AnswerExtractorI(ABC):
    @abstractmethod
    def extract(self, text: str) -> str | None:
        raise NotImplementedError

    def normalize_final_response(self, text: str) -> str:
        return text
