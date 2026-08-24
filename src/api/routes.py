import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from api.requests.completion import CompletionRequest
from api.responses.completion import (
    CompletionChoice,
    CompletionResponse,
    CompletionUsage,
)
from llm.requests import GenerateRequest
from strategies.base import Strategy
from strategies.models import StrategyResult

logger = logging.getLogger(__name__)

routes = APIRouter()
results_path = os.getenv("STRATEGY_RESULTS_PATH")
RESULTS_PATH = Path(results_path) if results_path else None


class CompletionRecord(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    created: int = Field(default_factory=lambda: int(time.time()))
    prompt: str
    result: StrategyResult | None = None
    error: str | None = None


def get_strategy(request: Request, experiment_name: str | None) -> Strategy:
    name = experiment_name or request.app.state.default_experiment
    strategy = request.app.state.strategies.get(name)

    if strategy is None:
        available = ", ".join(request.app.state.strategies)
        raise HTTPException(
            status_code=400,
            detail=(f"Unknown experiment '{name}'. Available experiments: {available}"),
        )

    return strategy


def save_record(
    prompt: str,
    result: StrategyResult | None = None,
    error: str | None = None,
) -> CompletionRecord:
    record = CompletionRecord(prompt=prompt, result=result, error=error)

    if RESULTS_PATH is not None:
        with RESULTS_PATH.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    record.model_dump(mode="json", exclude_none=True),
                    ensure_ascii=False,
                )
                + "\n"
            )

    return record


@routes.post(
    "/v1/completions",
    responses={400: {"description": "Unknown experiment"}},
)
async def completions(
    completion_request: CompletionRequest,
    request: Request,
    experiment: Annotated[str | None, Query()] = None,
    repetition: Annotated[int, Query(ge=1)] = 1,
) -> CompletionResponse:
    strategy = get_strategy(request, experiment)

    try:
        started_at = time.perf_counter()
        result = await strategy.run(
            GenerateRequest(
                prompt=completion_request.prompt,
                stop=completion_request.stop,
                repetition=repetition,
            )
        )
        result.end_to_end_latency_s = time.perf_counter() - started_at
    except Exception as exc:
        save_record(completion_request.prompt, error=str(exc))
        raise

    record = save_record(
        completion_request.prompt,
        result=result,
    )

    total_tokens = result.prompt_tokens + result.output_tokens

    logger.info(
        "Completion finished: id=%s strategy=%s model=%s calls=%d tokens=%d end_to_end_latency_s=%.3f",
        record.id,
        result.strategy_name,
        result.model,
        len(result.agent_responses),
        total_tokens,
        result.end_to_end_latency_s,
    )

    return CompletionResponse(
        id=record.id,
        created=record.created,
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
            total_tokens=total_tokens,
        ),
    )
