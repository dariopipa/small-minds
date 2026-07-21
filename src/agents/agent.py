from agents.models import AgentConfig, AgentResponse
from common.latency_measure import Timer
from extractors.answer_extractor_interface import AnswerExtractorI
from llm.client_interface import LLMClientI
from llm.requests import GenerateRequest


class Agent:
    def __init__(
        self,
        llm_client: LLMClientI,
        answer_extractor: AnswerExtractorI,
        agent_config: AgentConfig,
    ):
        self.llm_client = llm_client
        self.answer_extractor = answer_extractor
        self.agent_config = agent_config

    async def run(self, generation_request: GenerateRequest) -> AgentResponse:
        request = self._build_generation_request(generation_request)

        with Timer() as t:
            llm_response = await self.llm_client.generate(generation_request=request)

        return AgentResponse(
            agent_name=self.agent_config.name,
            agent_role=self.agent_config.role,
            prompt=request.prompt,
            response=llm_response.response,
            extracted_response=self.answer_extractor.extract(llm_response.response),
            model=llm_response.model,
            prompt_tokens=llm_response.prompt_tokens,
            output_tokens=llm_response.output_tokens,
            latency_s=t.elapsed,
        )

    def _build_generation_request(
        self, generation_request: GenerateRequest
    ) -> GenerateRequest:
        if self.agent_config.system_prompt is None:
            return generation_request

        prompt = (
            f"{self.agent_config.system_prompt}\n\nTask:\n{generation_request.prompt}"
        )
        return generation_request.model_copy(update={"prompt": prompt})
