from pydantic import BaseModel


class AgentConfig(BaseModel):
    name: str
    role: str
    system_prompt: str | None = None


class AgentResponse(BaseModel):
    agent_name: str
    agent_role: str
    model: str
    prompt: str
    response: str
    extracted_response: str | None
    prompt_tokens: int
    output_tokens: int
    latency_s: float | None = None
