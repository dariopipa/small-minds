import logging
from typing import Any

import httpx
import ollama
from ollama import AsyncClient

from common.exceptions import ModelLoadException, ModelNotFoundException
from llm.base import LLMClient
from llm.ollama.config import OllamaProviderConfig
from llm.requests import GenerateRequest
from llm.responses import LLMResponse

logger = logging.getLogger(__name__)


class OllamaClient(LLMClient):
    def __init__(self, config: OllamaProviderConfig):
        self.model_name = config.model_name
        self.options = config.options
        self.config = config.config
        self.client = AsyncClient()

    async def ensure_model_ready(self) -> None:
        logger.info(
            "Checking Ollama model readiness: model=%s",
            self.model_name,
        )
        model_exists = await self._model_exists()

        if not model_exists:
            raise ModelNotFoundException(f"Model '{self.model_name}' was not found.")

        try:
            # To load a model in OLLAMA an empty prompt must be sent.
            await self._generate(
                generation_request=GenerateRequest(prompt="", stop=None)
            )
            logger.info(
                "Ollama model ready: model=%s",
                self.model_name,
            )

        except ollama.RequestError as e:
            logger.exception("Could not communicate with Ollama during warmup")
            raise ModelLoadException(
                f"Could not communicate with Ollama while warming up "
                f"model '{self.model_name}'."
            ) from e

        except ollama.ResponseError as e:
            if e.status_code == 404:
                logger.exception("Ollama model was not found during warmup")
                raise ModelNotFoundException(
                    f"Model '{self.model_name}' was not found while warming up."
                ) from e

            logger.exception("Could not warm up Ollama model")
            raise ModelLoadException(
                f"Could not warm up model '{self.model_name}'."
            ) from e

    async def generate(self, generation_request: GenerateRequest) -> LLMResponse:
        try:
            response = await self._generate(generation_request=generation_request)

        except (ConnectionError, httpx.ConnectError) as e:
            logger.exception("Cannot connect to Ollama")
            raise ModelLoadException(
                f"Cannot connect to Ollama while generating with "
                f"model '{self.model_name}'."
            ) from e

        except ollama.RequestError as e:
            logger.exception("Cannot communicate with Ollama")
            raise ModelLoadException(
                f"Cannot communicate with Ollama while generating with "
                f"model '{self.model_name}'."
            ) from e

        except ollama.ResponseError as e:
            if e.status_code == 404:
                logger.exception("Ollama model was not found during generation")
                raise ModelNotFoundException(
                    f"Model '{self.model_name}' was not found while generating."
                ) from e

            logger.exception("Could not generate with Ollama model")
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

        return llm_response

    async def _model_exists(self) -> bool:
        try:
            response = await self.client.list()
            models = response.models

            for model in models:
                if self.model_name == model.model:
                    return True

            return False

        except (ConnectionError, httpx.ConnectError) as e:
            logger.exception("Cannot connect to Ollama while listing models")
            raise ModelLoadException(
                f"Cannot connect to Ollama while checking whether "
                f"model '{self.model_name}' is installed."
            ) from e

        except ollama.RequestError as e:
            logger.exception("Cannot communicate with Ollama while listing models")
            raise ModelLoadException(
                f"Cannot communicate with Ollama while checking whether "
                f"model '{self.model_name}' is installed."
            ) from e

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
