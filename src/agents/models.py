from pydantic import BaseModel


class AgentConfig(BaseModel):
    name: str
    role: str
    system_prompt: str | None = None
    base_seed: int | None = None
    base_temperature: float | None = None


class AgentResponse(BaseModel):
    agent_name: str
    agent_role: str
    model: str
    prompt: str
    response: str
    extracted_response: str | None
    prompt_tokens: int
    output_tokens: int
    seed: int | None = None
    temperature: float | None = None
    latency_s: float | None = None
    provider_duration_s: float | None = None
    agent_id: int | None = None
    round_id: int | None = None
