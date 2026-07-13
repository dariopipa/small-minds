from typing import Any

from pydantic import BaseModel

from extractors.answer_extractor_interface import AnswerExtractorI
from llm.client_interface import LLMClientI


class AgentResponse(BaseModel):
    response: Any
    extracted_response: Any
    model: Any
    prompt_tokens: Any
    output_tokens: Any


class Agent:
    def __init__(self, llm_client: LLMClientI, answer_extractor: AnswerExtractorI):
        self.llm_client = llm_client
        self.answer_extractor = answer_extractor

    async def run(self, prompt: str, stop: list[str] | None = None) -> AgentResponse:
        llm_response = await self.llm_client.generate(prompt=prompt, stop=stop)

        return AgentResponse(
            response=llm_response.response,
            extracted_response=self.answer_extractor.extract(llm_response.response),
            model=llm_response.model,
            prompt_tokens=llm_response.prompt_tokens,
            output_tokens=llm_response.output_tokens,
        )
