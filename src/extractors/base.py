from abc import ABC, abstractmethod


class AnswerExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> str | None:
        raise NotImplementedError

    def normalize_final_response(self, text: str) -> str:
        return text

    def prepare_prompt(self, prompt: str) -> str:
        return prompt

    def prepare_followup_context(self, prompt: str) -> str:
        return prompt

    def prepare_stop(self, stop: list[str] | None) -> list[str] | None:
        return stop
