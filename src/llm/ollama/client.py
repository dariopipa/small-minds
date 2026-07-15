from typing import Any

import httpx
import ollama
from ollama import AsyncClient

from llm.requests import GenerateRequest
from llm.responses import LLMResponse
from llm.ollama.config import OllamaProviderConfig
from common.exceptions import ModelLoadException, ModelNotFoundException
from llm.client_interface import LLMClientI


class OllamaClient(LLMClientI):
    def __init__(self, config: OllamaProviderConfig):
        self.model_name = config.model_name
        self.options = config.options
        self.config = config.config
        self.client = AsyncClient()

    async def ensure_model_ready(self) -> None:
        # TODO: REMOVE THIS LATER ON, DEBUGGING PURPOSES
        print(
            "\n========================================================================================\n"
            "+++ OLLAMA MODEL CHECK +++\n"
            f"model: {self.model_name}\n"
            "checking local Ollama model list\n"
            "========================================================================================\n",
            flush=True,
        )
        model_exists = await self._model_exists()

        if not model_exists:
            raise ModelNotFoundException(f"Model '{self.model_name}' was not found.")

        try:
            # To load a model in OLLAMA an empty prompt must be sent.
            await self._generate(
                generation_request=GenerateRequest(prompt="", stop=None)
            )
            # TODO: REMOVE THIS LATER ON, DEBUGGING PURPOSES
            print(
                "\n========================================================================================\n"
                "+++ OLLAMA MODEL READY +++\n"
                f"model: {self.model_name}\n"
                f"default options: {self.options.to_dict()}\n"
                f"generate config: {self.config.to_generate_kwargs()}\n"
                "========================================================================================\n",
                flush=True,
            )

        except ollama.RequestError as e:
            raise ModelLoadException("Could not communicate with Ollama.") from e

        except ollama.ResponseError as e:
            if e.status_code == 404:
                raise ModelNotFoundException(
                    f"Model '{self.model_name}' was not found while warming up."
                ) from e

            raise ModelLoadException(
                f"Could not warm up model '{self.model_name}'."
            ) from e

    async def generate(self, generation_request: GenerateRequest) -> LLMResponse:
        try:
            # TODO: REMOVE THIS LATER ON, DEBUGGING PURPOSES
            print(
                "\n========================================================================================\n"
                "+++ OLLAMA GENERATE REQUEST +++\n"
                f"model: {self.model_name}\n"
                f"prompt chars: {len(generation_request.prompt)}\n"
                f"effective options: {self._generation_options(generation_request)}\n"
                f"generate config: {self.config.to_generate_kwargs()}\n"
                "========================================================================================\n",
                flush=True,
            )
            response = await self._generate(generation_request=generation_request)

        except (ConnectionError, httpx.ConnectError) as e:
            raise ModelLoadException("Cannot connect to Ollama.") from e

        except ollama.RequestError as e:
            raise ModelLoadException("Cannot communicate with Ollama.") from e

        except ollama.ResponseError as e:
            if e.status_code == 404:
                raise ModelNotFoundException(
                    f"Model '{self.model_name}' was not found while generating."
                ) from e

            raise ModelLoadException(
                f"Could not generate with model '{self.model_name}'."
            ) from e

        llm_response = LLMResponse(
            response=response.response,
            model=response.model or self.model_name,
            thinking=getattr(response, "thinking", None),
            prompt_tokens=response.prompt_eval_count or 0,
            output_tokens=response.eval_count or 0,
            total_duration_ns=response.total_duration or 0,
        )

        # TODO: REMOVE THIS LATER ON, DEBUGGING PURPOSES
        print(
            "\n========================================================================================\n"
            "+++ OLLAMA GENERATE RESPONSE +++\n"
            f"model: {llm_response.model}\n"
            f"prompt tokens: {llm_response.prompt_tokens}\n"
            f"completion tokens: {llm_response.output_tokens}\n"
            f"duration seconds: {llm_response.duration_s:.3f}\n"
            "========================================================================================\n",
            flush=True,
        )

        return llm_response

    async def chat_bot(self):
        return "chaaaaaaaaaaaaaaat-boooooooooooot"

    async def _model_exists(self) -> bool:
        try:
            response = await self.client.list()
            models = response.models

            for model in models:
                if self.model_name == model.model:
                    return True

            return False

        except (ConnectionError, httpx.ConnectError) as e:
            raise ModelLoadException("Cannot connect to Ollama.") from e

        except ollama.RequestError as e:
            raise ModelLoadException("Cannot communicate with Ollama.") from e

    async def _generate(self, generation_request: GenerateRequest) -> Any:
        return await self.client.generate(
            model=self.model_name,
            prompt=generation_request.prompt,
            options=self._generation_options(generation_request),
            **self.config.to_generate_kwargs(),
        )

    def _generation_options(
        self, generation_request: GenerateRequest
    ) -> dict[str, Any]:
        options = self.options.to_dict()

        if generation_request.stop is not None:
            options["stop"] = generation_request.stop

        return options
