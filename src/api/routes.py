import time
from typing import Annotated
import uuid

from fastapi import APIRouter, Depends, Request
from api.responses.chat_completion import (
    CompletionChoice,
    CompletionResponse,
    CompletionUsage,
)
from llm.client_interface import LLMClientI
from api.requests.chat_completion import (
    CompletionRequest,
)

routes = APIRouter()


def get_llm_client(request: Request) -> LLMClientI:
    return request.app.state.llm_client


@routes.post("/v1/completions")
async def completions(
    request: CompletionRequest,
    client: Annotated[LLMClientI, Depends(get_llm_client)],
) -> CompletionResponse:

    # add options in the generate function
    result = await client.generate(prompt=request.prompt, stop=request.stop)

    # change the response.
    response = CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=str(result.model),
        choices=[
            CompletionChoice(
                text=result.response,
                index=0,
                logprobs=None,
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.output_tokens,
            total_tokens=result.prompt_tokens + result.output_tokens,
        ),
    )

    # todo: strategy_service.call(clientI,prompt,options)

    return response
