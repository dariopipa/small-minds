from extractors.gsm8k_answer_extractor import GSM8KAnswerExtractor
from extractors.answer_extractor_interface import AnswerExtractorI


def create_extractor(task_name: list[str]) -> AnswerExtractorI:
    match task_name[0]:
        case "gsm8k":
            return GSM8KAnswerExtractor()
        case _:
            raise ValueError(f"Unsupported answer extractor: {task_name}")
