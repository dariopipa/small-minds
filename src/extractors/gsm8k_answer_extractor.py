import re
from decimal import Decimal

from extractors.base import AnswerExtractor

GSM8K_ANSWER_PATTERN = re.compile(r"####\s*(-?\$?(?a:\d[\d,]*(?:\.\d+)?))")
NUMBER_PATTERN = re.compile(r"-?\$?(?a:\d[\d,]*(?:\.\d+)?)")


def normalize_number(text: str) -> str:
    value = Decimal(text.replace("$", "").replace(",", ""))
    if value == 0:
        return "0"
    normalized = format(value, "f")
    return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized


def normalize_number_for_response(text: str) -> str:
    normalized = normalize_number(text)
    if "$" not in text:
        return normalized
    return f"-${normalized[1:]}" if normalized.startswith("-") else f"${normalized}"


class GSM8KAnswerExtractor(AnswerExtractor):
    def extract(self, text: str) -> str | None:
        answer_matches = GSM8K_ANSWER_PATTERN.findall(text)
        if answer_matches:
            # lm-eval's GSM8K strict-match filter selects the first marked answer.
            return normalize_number(answer_matches[0])

        matches = NUMBER_PATTERN.findall(text)
        if not matches:
            return None

        return normalize_number(matches[-1])

    def normalize_final_response(self, text: str) -> str:
        marked_answers = list(GSM8K_ANSWER_PATTERN.finditer(text))
        if marked_answers:
            # Canonicalize the number without changing marker placement or
            # surrounding text, so malformed formatting remains observable.
            for match in reversed(marked_answers):
                start, end = match.span(1)
                replacement = normalize_number_for_response(match.group(1))
                text = f"{text[:start]}{replacement}{text[end:]}"
            return text

        matches = list(NUMBER_PATTERN.finditer(text))
        if not matches:
            return text

        # Keep a missing marker missing, but canonicalize the fallback number
        # used by lm-eval's flexible extractor.
        match = matches[-1]
        start, end = match.span()
        replacement = normalize_number_for_response(match.group())
        return f"{text[:start]}{replacement}{text[end:]}"

    def prepare_followup_context(self, prompt: str) -> str:
        marker = "\nQuestion:"
        question_start = prompt.rfind(marker)
        if question_start < 0:
            return prompt

        final_question = prompt[question_start + 1 :].strip()
        return (
            "Required final-answer format: `#### <number>` on its own line, "
            "with nothing after it.\n\n"
            f"{final_question}"
        )
