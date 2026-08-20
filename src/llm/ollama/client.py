import logging
from typing import Any, cast

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
        logger.info("Warming up Ollama model: model=%s", self.model_name)
        await self._generate(GenerateRequest(prompt="", stop=None))
        logger.info("Ollama model ready: model=%s", self.model_name)

    async def generate(self, generation_request: GenerateRequest) -> LLMResponse:
        response = await self._generate(generation_request)
        return LLMResponse(
            response=cast(str, response.response),
            model=response.model or self.model_name,
            thinking=getattr(response, "thinking", None),
            prompt_tokens=response.prompt_eval_count or 0,
            output_tokens=response.eval_count or 0,
            total_duration_ns=response.total_duration or 0,
        )

    async def _generate(
        self, generation_request: GenerateRequest
    ) -> ollama.GenerateResponse:
        try:
            return await self.client.generate(
                model=self.model_name,
                prompt=generation_request.prompt,
                options=self._generation_options(generation_request),
                **self.config.to_generate_kwargs(),
            )
        except ollama.ResponseError as error:
            if error.status_code == 404:
                raise ModelNotFoundException(
                    f"Model '{self.model_name}' was not found."
                ) from error
            logger.exception("Ollama rejected a request: model=%s", self.model_name)
            raise ModelLoadException(
                f"Ollama could not process a request for model '{self.model_name}'."
            ) from error
        except (ConnectionError, httpx.HTTPError, ollama.RequestError) as error:
            logger.exception(
                "Cannot communicate with Ollama: model=%s", self.model_name
            )
            raise ModelLoadException(
                f"Cannot communicate with Ollama for model '{self.model_name}'."
            ) from error

    def _generation_options(
        self, generation_request: GenerateRequest
    ) -> dict[str, Any]:
        options = self.options.to_dict()

        if generation_request.seed is not None:
            options["seed"] = generation_request.seed

        if generation_request.temperature is not None:
            options["temperature"] = generation_request.temperature

        if generation_request.stop is not None:
            options["stop"] = generation_request.stop

        return options
