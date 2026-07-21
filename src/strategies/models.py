from typing import Annotated, Literal

from pydantic import BaseModel, Field, PositiveInt

from agents.models import AgentResponse


class StrategyResult(BaseModel):
    model: str | None = None
    strategy_name: str
    prompt: str
    response: str
    extracted_response: str | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_latency_s: float | None = None
    agent_responses: list[AgentResponse]


class DirectStrategyConfig(BaseModel):
    name: Literal["direct"] = "direct"


class MultiStrategyConfig(BaseModel):
    name: Literal["multi"] = "multi"
    agent_number: PositiveInt


StrategyConfig = Annotated[
    DirectStrategyConfig | MultiStrategyConfig,
    Field(discriminator="name"),
]
