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


class SelfConsistencyConfig(BaseModel):
    name: Literal["self-consistency"] = "self-consistency"
    agent_number: PositiveInt


class SocietyOfMindsConfig(BaseModel):
    name: Literal["society-of-minds"] = "society-of-minds"
    agent_number: PositiveInt
    debate_rounds: PositiveInt


StrategyConfig = Annotated[
    DirectStrategyConfig | SelfConsistencyConfig | SocietyOfMindsConfig,
    Field(discriminator="name"),
]
