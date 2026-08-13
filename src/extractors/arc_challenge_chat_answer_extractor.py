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

    def prepare_prompt(self, prompt: str) -> str:
        reminder = "Final answer reminder: return only one letter: A, B, C, or D."
        answer_cue = "The best answer is"
        if prompt.rstrip().endswith(answer_cue):
            return f"{prompt.rstrip()[: -len(answer_cue)]}{reminder}\n{answer_cue}"
        return f"{prompt}\n\n{reminder}"
