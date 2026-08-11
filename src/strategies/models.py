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


class RoleBasedSVJStrategyConfig(BaseModel):
    name: Literal["role_based_svj"] = "role_based_svj"


class SelfConsistencyConfig(BaseModel):
    name: Literal["self_consistency"] = "self_consistency"
    agent_number: PositiveInt


class SocietyOfMindsConfig(BaseModel):
    name: Literal["society_of_minds"] = "society_of_minds"
    agent_number: PositiveInt
    debate_rounds: PositiveInt


StrategyConfig = Annotated[
    DirectStrategyConfig
    | RoleBasedSVJStrategyConfig
    | SelfConsistencyConfig
    | SocietyOfMindsConfig,
    Field(discriminator="name"),
]
