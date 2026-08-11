import re

from extractors.base import AnswerExtractor

GSM8K_ANSWER_PATTERN = re.compile(r"####\s*(-?\$?[0-9][0-9,]*(?:\.[0-9]+)?)")
NUMBER_PATTERN = re.compile(r"-?\$?[0-9][0-9,]*(?:\.[0-9]+)?")


def normalize_number(text: str) -> str:
    return text.replace("$", "").replace(",", "")


class GSM8KAnswerExtractor(AnswerExtractor):
    def extract(self, text: str) -> str | None:
        answer_matches = GSM8K_ANSWER_PATTERN.findall(text)
        if answer_matches:
            return normalize_number(answer_matches[-1])

        matches = NUMBER_PATTERN.findall(text)
        if not matches:
            return None

        return normalize_number(matches[-1])
