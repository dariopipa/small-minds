import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from api.requests.chat_completion import (
    CompletionRequest,
)
from api.responses.chat_completion import (
    CompletionChoice,
    CompletionResponse,
    CompletionUsage,
)
from llm.requests import GenerateRequest
from strategies.strategy_interface import StrategyI

routes = APIRouter()


def get_strategy(request: Request) -> StrategyI:
    return request.app.state.strategy


@routes.post("/v1/completions")
async def completions(
    completion_request: CompletionRequest,
    strategy: Annotated[StrategyI, Depends(get_strategy)],
) -> CompletionResponse:

    completion_id = f"cmpl-{uuid.uuid4().hex}"
    generation_request = GenerateRequest(
        prompt=completion_request.prompt,
        stop=completion_request.stop,
    )

    result = await strategy.run(generation_request=generation_request)

    # TODO: REMOVE THIS LATER ON, DEBUGGING PURPOSES
    print(
        "\n========================================================================================\n"
        "+++ FASTAPI COMPLETION REQUEST +++\n"
        f"id: {completion_id}\n"
        f"prompt chars: {len(completion_request.prompt)}\n"
        f"stop: {completion_request.stop}\n"
        "prompt preview first 700 chars:\n"
        f"{completion_request.prompt[:700]}\n"
        "========================================================================================\n",
        flush=True,
    )

    # result = await agent.run(generation_request=generation_request)

    # TODO: REMOVE THIS LATER ON, DEBUGGING PURPOSES
    print(
        "\n========================================================================================\n"
        "+++ FASTAPI COMPLETION RESPONSE +++\n"
        f"id: {completion_id}\n"
        f"model: {result.model}\n"
        f"prompt tokens: {result.prompt_tokens}\n"
        f"completion tokens: {result.output_tokens}\n"
        f"extracted answer: {result.extracted_response}\n"
        "========================================================================================\n",
        flush=True,
    )

    # change the response.
    response = CompletionResponse(
        id=completion_id,
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

    return response
