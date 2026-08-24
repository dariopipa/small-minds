import re

from extractors.base import AnswerExtractor

GSM8K_ANSWER_PATTERN = re.compile(r"####\s*(-?\$?(?a:\d[\d,]*(?:\.\d+)?))")
NUMBER_PATTERN = re.compile(r"-?\$?(?a:\d[\d,]*(?:\.\d+)?)")


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

    def prepare_prompt(self, prompt: str) -> str:
        reminder = (
            "Final answer reminder: after your reasoning, end with exactly "
            "`#### <number>` on its own line. Do not use \\boxed{} and do not "
            "write anything after that line."
        )
        if prompt.rstrip().endswith("Answer:"):
            return f"{prompt.rstrip()[: -len('Answer:')]}{reminder}\nAnswer:"
        return f"{prompt}\n\n{reminder}"
