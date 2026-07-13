import re

from extractors.answer_extractor_interface import AnswerExtractorI


class GSM8KAnswerExtractor(AnswerExtractorI):
    def extract(self, text: str) -> str | None:
        # Strict GSM8K format: use the first match.
        strict = re.search(r"#### (\-?[0-9\.\,]+)", text)
        if strict:
            return strict.group(1)

        # Flexible format: use the last number-like match.
        matches = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", text)
        if not matches:
            return None

        first_group, second_group = matches[-1]

        return first_group or second_group
