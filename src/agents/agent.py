from typing import Any

from pydantic import BaseModel

from extractors.answer_extractor_interface import AnswerExtractorI
from llm.client_interface import LLMClientI
from llm.requests import GenerateRequest


# todo: remove from here, remove Any and make it type specific.
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

    async def run(self, generation_request: GenerateRequest) -> AgentResponse:
        llm_response = await self.llm_client.generate(
            generation_request=generation_request
        )

        return AgentResponse(
            response=llm_response.response,
            extracted_response=self.answer_extractor.extract(llm_response.response),
            model=llm_response.model,
            prompt_tokens=llm_response.prompt_tokens,
            output_tokens=llm_response.output_tokens,
        )
