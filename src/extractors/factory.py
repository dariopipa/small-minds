from extractors.arc_challenge_chat_answer_extractor import (
    ARCChallengeChatAnswerExtractor,
)
from extractors.base import AnswerExtractor
from extractors.gsm8k_answer_extractor import GSM8KAnswerExtractor


def create_extractor(task_name: list[str]) -> AnswerExtractor:
    match task_name[0]:
        case "gsm8k":
            return GSM8KAnswerExtractor()
        case "arc_challenge_chat":
            return ARCChallengeChatAnswerExtractor()
        case _:
            raise ValueError(f"Unsupported answer extractor: {task_name}")
