import re

from extractors.base import AnswerExtractor

FINAL_ANSWER_PATTERN = re.compile(
    r"^final answer:\s*([ABCD])\s*$",
    re.IGNORECASE | re.MULTILINE,
)


class ARCChallengeChatAnswerExtractor(AnswerExtractor):
    def extract(self, text: str) -> str | None:
        answers = FINAL_ANSWER_PATTERN.findall(text)
        return answers[-1].upper() if answers else None

    def normalize_final_response(self, text: str) -> str:
        return self.extract(text) or ""

    def prepare_stop(self, stop: list[str] | None) -> list[str] | None:
        # lm-eval's letter-only ARC task stops at punctuation/blank lines. Those
        # stops would cut off our internal rationale before its final answer.
        remaining = [value for value in stop or [] if value not in {".", "\n\n"}]
        return remaining or None

    def prepare_prompt(self, prompt: str) -> str:
        reminder = (
            "Reason concisely, check the strongest competing option, then end with "
            "`Final answer: <A, B, C, or D>` on its own line. Do not write "
            "anything after that line."
        )
        answer_cue = "The best answer is"
        if prompt.rstrip().endswith(answer_cue):
            prompt = prompt.rstrip()[: -len(answer_cue)].rstrip()
        return f"{prompt}\n\n{reminder}"
