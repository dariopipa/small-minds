import re

from extractors.base import AnswerExtractor

FINAL_ANSWER_PATTERN = re.compile(
    r"^final answer:\s*(yes|no)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
ANSWER_STATEMENT_PATTERN = re.compile(
    r"\b(?:final\s+)?answer\s+(?:is|:)\s*(yes|no)\b",
    re.IGNORECASE,
)


class BoolQAnswerExtractor(AnswerExtractor):
    def extract(self, text: str) -> str | None:
        answers = FINAL_ANSWER_PATTERN.findall(text)
        if not answers:
            answers = ANSWER_STATEMENT_PATTERN.findall(text)
        return answers[-1].lower() if answers else None

    def normalize_final_response(self, text: str) -> str:
        answer = self.extract(text)
        # boolq-seq2seq's choices are literally " no" and " yes". Its exact
        # match metric does not remove whitespace, so the continuation needs
        # the same single leading delimiter while remaining answer-only.
        return f" {answer}" if answer is not None else ""

    def prepare_stop(self, stop: list[str] | None) -> list[str] | None:
        # boolq-seq2seq stops at the first newline. That would cut off an
        # internal rationale before its final answer.
        newline_stops = {"\n", "\n\n", "\r\n", "\r\n\r\n"}
        remaining = [value for value in stop or [] if value not in newline_stops]
        return remaining or None

    def prepare_followup_context(self, prompt: str) -> str:
        prompt = re.sub(r"\bAnswer:\s*$", "", prompt.rstrip(), flags=re.IGNORECASE)
        return (
            "Required final-answer format: `Final answer: yes` or "
            "`Final answer: no` on its own line, with nothing after it.\n\n"
            f"{prompt.rstrip()}"
        )

    def prepare_prompt(self, prompt: str) -> str:
        prompt = re.sub(r"\bAnswer:\s*$", "", prompt.rstrip(), flags=re.IGNORECASE)
        reminder = (
            "Reason briefly using only the passage, then end on its own line with "
            "exactly `Final answer: yes` or `Final answer: no`. Do not write "
            "anything after that line."
        )
        return f"{prompt.rstrip()}\n\n{reminder}"
