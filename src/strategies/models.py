from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from agents.models import AgentResponse

SUPPORTED_STRATEGY_NAMES = (
    "direct",
    "role_based_svj",
    "self_consistency",
    "society_of_minds",
)


class StrategyConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StrategyResult(BaseModel):
    model: str | None = None
    strategy_name: str
    prompt: str
    response: str
    extracted_response: str | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_latency_s: float | None = None
    initial_extracted_response: str | None = None
    agent_responses: list[AgentResponse]


class DirectStrategyConfig(StrategyConfigModel):
    name: Literal["direct"] = "direct"


class RoleBasedSVJStrategyConfig(StrategyConfigModel):
    name: Literal["role_based_svj"] = "role_based_svj"


class SelfConsistencyConfig(StrategyConfigModel):
    name: Literal["self_consistency"] = "self_consistency"
    agent_number: PositiveInt


class SocietyOfMindsConfig(StrategyConfigModel):
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
