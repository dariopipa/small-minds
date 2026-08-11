import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from api.requests.completion import CompletionRequest
from api.responses.completion import (
    CompletionChoice,
    CompletionResponse,
    CompletionUsage,
)
from llm.requests import GenerateRequest
from strategies.base import Strategy

logger = logging.getLogger(__name__)

routes = APIRouter()


def get_strategy(request: Request) -> Strategy:
    experiment_name = request.query_params.get(
        "experiment", request.app.state.default_experiment
    )
    strategy = request.app.state.strategies.get(experiment_name)
    if strategy is None:
        available = ", ".join(request.app.state.strategies)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown experiment '{experiment_name}'. "
                f"Available experiments: {available}"
            ),
        )
    return strategy


def save_call(record: dict) -> None:
    path = os.getenv("STRATEGY_RESULTS_PATH")
    if path is None:
        return

    output_path = Path(path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        raise RuntimeError(
            f"Could not write strategy result to {output_path}. "
            "Check STRATEGY_RESULTS_PATH."
        ) from exc


@routes.post("/v1/completions")
async def completions(
    completion_request: CompletionRequest,
    strategy: Annotated[Strategy, Depends(get_strategy)],
) -> CompletionResponse:

    completion_id = f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    generation_request = GenerateRequest(
        prompt=completion_request.prompt,
        stop=completion_request.stop,
    )
    try:
        result = await strategy.run(generation_request=generation_request)
    except Exception as exc:
        try:
            save_call(
                {
                    "created": created,
                    "prompt": completion_request.prompt,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            )
        except Exception:
            logger.exception(
                "Could not persist failed completion: id=%s", completion_id
            )
        raise

    logger.info(
        "Completion finished: id=%s strategy=%s model=%s calls=%d tokens=%d",
        completion_id,
        result.strategy_name,
        result.model,
        len(result.agent_responses),
        result.prompt_tokens + result.output_tokens,
    )
    save_call(
        {
            "created": created,
            "prompt": completion_request.prompt,
            "result": result.model_dump(mode="json"),
        }
    )

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
