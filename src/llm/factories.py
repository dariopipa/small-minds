from llm.client_interface import LLMClientI
from llm.ollama.client import OllamaClient
from llm.ollama.config import OllamaProviderConfig


class LLMClientFactory:
    @staticmethod
    def create(config: OllamaProviderConfig) -> LLMClientI:
        match config.provider:
            case "ollama":
                return OllamaClient(config=config)
            case _:
                raise ValueError(f"Unsupported provider: {config.provider}")
