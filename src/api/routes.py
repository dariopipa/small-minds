import json
import time
import uuid
from typing import Any

from fastapi import APIRouter

from src.api.requests.chat_completion import ChatCompletionRequest, CompletionRequest
from src.llm.ollama_client import OllamaClient
from src.llm.schemas import ModelOptions, OllamaConfig


routes = APIRouter()


@routes.post("/v1/completions")
async def completions(request: CompletionRequest) -> dict[str, Any]:
    request_start = time.perf_counter()

    print("\n" + "=" * 100)
    print("LM-EVAL REQUEST /v1/completions")
    print("model:", request.model)
    print("temperature:", request.temperature)
    print("max_tokens:", request.max_tokens)
    print("stop:", request.stop)
    print("prompt_chars:", len(request.prompt))
    print("=" * 100 + "\n")

    model_options = ModelOptions(
        temperature=request.temperature,
        num_predict=request.max_tokens,
        stop=request.stop,
    )

    print("MODEL OPTIONS SENT TO OLLAMA:")
    print(model_options.to_dict())

    ollama_provider = OllamaClient(
        model_name=request.model,
        model_options=model_options,
        ollama_config=OllamaConfig(stream=False, keep_alive="30m"),
    )

    result = await ollama_provider.generate(prompt=request.prompt)

    request_end = time.perf_counter()

    print("\nOLLAMA RESULT")
    print("response_chars:", len(result.response))
    print("prompt_tokens:", result.prompt_tokens)
    print("output_tokens:", result.output_tokens)
    print("total_tokens:", result.prompt_tokens + result.output_tokens)
    print("ollama_total_duration_s:", result.total_duration_ns / 1e9)
    print("fastapi_request_wall_s:", request_end - request_start)
    print("=" * 100 + "\n")

    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": result.model,
        "choices": [
            {
                "text": result.response,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.output_tokens,
            "total_tokens": result.prompt_tokens + result.output_tokens,
        },
    }


@routes.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
    body = request.model_dump(mode="json")

    print("\n" + "=" * 100)
    print("LM-EVAL SENT THIS REQUEST TO /v1/chat/completions:")
    print(json.dumps(body, indent=2, ensure_ascii=False))
    print("=" * 100 + "\n")

    prompt = "\n".join(
        f"{message.role}: {message.content}" for message in request.messages
    )

    model_options = ModelOptions(
        temperature=request.temperature,
        num_predict=request.max_tokens,
        stop=request.stop,
    )

    ollama_provider = OllamaClient(
        model_name=request.model,
        model_options=model_options,
    )

    result = await ollama_provider.generate(prompt=prompt)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.response,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.output_tokens,
            "total_tokens": result.prompt_tokens + result.output_tokens,
        },
    }
