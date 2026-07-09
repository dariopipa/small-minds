from typing import Any

import httpx
import ollama
from ollama import AsyncClient

from llm.schemas import LLMResponse
from llm.ollama.config import OllamaProviderConfig
from common.exceptions import ModelLoadException, ModelNotFoundException
from llm.client_interface import LLMClientI


class OllamaClient(LLMClientI):
    def __init__(self, config: OllamaProviderConfig):
        self.model_name = config.model_name
        self.options = config.options
        self.config = config.config
        self.client = AsyncClient()

    async def load_model(self) -> None:
        model_exists = await self._check_model_existence()

        if model_exists is False:
            raise ModelNotFoundException(f"Model '{self.model_name}' was not found.")

        await self._pre_load_model()

    async def generate(self, prompt: str) -> LLMResponse:
        response = await self._generate_call(prompt=prompt)

        return LLMResponse(
            response=response.response,
            model=response.model or self.model_name,
            thinking=getattr(response, "thinking", None),
            prompt_tokens=response.prompt_eval_count or 0,
            output_tokens=response.eval_count or 0,
            total_duration_ns=response.total_duration or 0,
        )

    async def chat_bot(self):
        return "chaaaaaaaaaaaaaaat-boooooooooooot"

    async def _check_model_existence(self) -> bool:
        try:
            response = await self.client.list()
            models = response.models

            for model in models:
                if self.model_name == model.model:
                    return True

            return False

        except (ConnectionError, httpx.ConnectError) as e:
            raise ModelLoadException(
                "Cannot connect to Ollama. Is Ollama running?"
            ) from e

        except ollama.RequestError as e:
            raise ModelLoadException(
                "Cannot communicate with Ollama. Is Ollama running?"
            ) from e

    async def _pre_load_model(self) -> bool:
        try:
            # To load a model in OLLAMA an empty prompt must be sent.
            await self._generate_call(prompt="")
            return True

        except ollama.RequestError as e:
            raise ModelLoadException("Could not communicate with Ollama.") from e

        except ollama.ResponseError as e:
            if e.status_code == 404:
                raise ModelNotFoundException(
                    f"Model '{self.model_name}' was not found while preloading."
                ) from e

            raise ModelLoadException(
                f"Could not preload model '{self.model_name}'."
            ) from e

    async def _generate_call(self, prompt: str = "") -> Any:
        return await self.client.generate(
            model=self.model_name,
            prompt=prompt,
            options=self.options.to_dict(),
            **self.config.to_generate_kwargs(),
        )
