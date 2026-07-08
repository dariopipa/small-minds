from typing import Any

import httpx
import ollama
from ollama import AsyncClient

from src.common.exceptions import ModelLoadException, ModelNotFoundException
from src.llm.llm_client_interface import LLMClientI
from src.llm.schemas import LLMResponse, ModelOptions, OllamaConfig


class OllamaClient(LLMClientI):
    def __init__(
        self, model_name: str, model_options: ModelOptions, ollama_config: OllamaConfig
    ):
        self.model_name = model_name
        self.model_options = model_options
        self.client = AsyncClient()
        self.config = ollama_config

    async def load_model(self) -> None:
        model_exists = await self._check_model_existence()

        if model_exists is False:
            raise ModelNotFoundException(f"Model '{self.model_name}' was not found.")

        await self._pre_load_model()

    async def generate(self, prompt: str) -> LLMResponse:
        response = await self._generate_call(prompt=prompt)

        return LLMResponse(
            response=response.response,
            model=response.model,
            thinking=getattr(response, "thinking", None),
            prompt_tokens=response.prompt_eval_count or 0,
            output_tokens=response.eval_count or 0,
            total_duration_ns=response.total_duration or 0,
        )

    def chat_bot(self):
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

    # To load a model in Ollama, an empty prompt must be sent.
    async def _pre_load_model(self) -> bool:
        try:
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

    # todo: ADD A OLLAMA CONFIG that can be passed in the generate()
    async def _generate_call(self, prompt: str = "") -> Any:
        generate_kwargs = {
            "model": self.model_name,
            "prompt": prompt,
            "options": self.model_options.to_dict(),
            **self.config.to_generate_kwargs(),
        }

        print("\n=== OLLAMA GENERATE KWARGS ===")
        for key, value in generate_kwargs.items():
            if key == "prompt":
                print("prompt_chars:", len(value))
            else:
                print(f"{key}: {value}")

        print("\n==== DEBUGGG ===")

        response = await self.client.generate(
            model=self.model_name,
            prompt=prompt,
            options=self.model_options.to_dict(),
            **self.config.to_generate_kwargs(),
        )

        # TODO: Remove
        print("\n=== OLLAMA RESPONSE ===")
        print("prompt_eval_count:", response.prompt_eval_count)
        print("eval_count:", response.eval_count)
        print("total_duration_s:", (response.total_duration or 0) / 1e9)
        print("load_duration_s:", (getattr(response, "load_duration", 0) or 0) / 1e9)
        print("prompt_eval_duration_s:", (response.prompt_eval_duration or 0) / 1e9)
        print("eval_duration_s:", (response.eval_duration or 0) / 1e9)

        return response
