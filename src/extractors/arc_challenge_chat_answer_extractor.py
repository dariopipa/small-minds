import re

from extractors.base import AnswerExtractor


class ARCChallengeChatAnswerExtractor(AnswerExtractor):
    def extract(self, text: str) -> str | None:
        stripped = text.strip()
        single_letter = re.fullmatch(r"\(?([ABCD])\)?[.!]?", stripped, re.IGNORECASE)
        if single_letter:
            return single_letter.group(1).upper()

        labeled_answers = re.findall(
            r"(?:the\s+best\s+answer\s+is|final[_\s]+answer|answer)"
            r"\s*[:=-]?\s*\(?([ABCD])\)?\b",
            stripped,
            re.IGNORECASE,
        )
        return labeled_answers[-1].upper() if labeled_answers else None

    def normalize_final_response(self, text: str) -> str:
        return self.extract(text) or text
