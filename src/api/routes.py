import json
import logging
import os
import time
import uuid
from pathlib import Path
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
from strategies.models import StrategyResult
from strategies.strategy_interface import StrategyI

logger = logging.getLogger(__name__)

routes = APIRouter()


def get_strategy(request: Request) -> StrategyI:
    return request.app.state.strategy


def save_strategy_result(
    path: str | None,
    completion_id: str,
    created: int,
    completion_request: CompletionRequest,
    result: StrategyResult,
) -> None:
    if path is None:
        return

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "id": completion_id,
        "created": created,
        "prompt_chars": len(completion_request.prompt),
        "stop": completion_request.stop,
        "strategy_result": result.model_dump(mode="json"),
    }

    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@routes.post("/v1/completions")
async def completions(
    completion_request: CompletionRequest,
    strategy: Annotated[StrategyI, Depends(get_strategy)],
) -> CompletionResponse:

    completion_id = f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    generation_request = GenerateRequest(
        prompt=completion_request.prompt,
        stop=completion_request.stop,
    )

    result = await strategy.run(generation_request=generation_request)
    save_strategy_result(
        path=os.getenv("STRATEGY_RESULTS_PATH"),
        completion_id=completion_id,
        created=created,
        completion_request=completion_request,
        result=result,
    )

    # change the response.
    response = CompletionResponse(
        id=completion_id,
        created=created,
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
