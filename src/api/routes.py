import time
from typing import Annotated
import uuid

from fastapi import APIRouter, Depends, Request
from agents.agent import Agent
from api.responses.chat_completion import (
    CompletionChoice,
    CompletionResponse,
    CompletionUsage,
)
from api.requests.chat_completion import (
    CompletionRequest,
)

routes = APIRouter()


def get_agent(request: Request) -> Agent:
    return request.app.state.agent


@routes.post("/v1/completions")
async def completions(
    request: CompletionRequest,
    agent: Annotated[Agent, Depends(get_agent)],
) -> CompletionResponse:

    result = await agent.run(request.prompt, stop=request.stop)

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

    return response
