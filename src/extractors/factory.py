from common.exceptions import ConfigurationError
from extractors.arc_challenge_chat_answer_extractor import (
    ARCChallengeChatAnswerExtractor,
)
from extractors.base import AnswerExtractor
from extractors.gsm8k_answer_extractor import GSM8KAnswerExtractor


def create_extractor(name: str) -> AnswerExtractor:
    match name:
        case "gsm8k":
            return GSM8KAnswerExtractor()
        case "arc_challenge_chat":
            return ARCChallengeChatAnswerExtractor()
        case _:
            raise ConfigurationError(
                f"Unsupported answer extractor for task '{name}'. "
                "Add an explicit mapping in extractors.factory.create_extractor "
                "before adding this task to benchmarks.yaml."
            )
