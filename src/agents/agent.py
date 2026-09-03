from agents.models import AgentConfig, AgentResponse
from common.latency_measure import Timer
from common.seeding import derive_seed
from extractors.base import AnswerExtractor
from llm.base import LLMClient
from llm.requests import GenerateRequest


class Agent:
    def __init__(
        self,
        llm_client: LLMClient,
        answer_extractor: AnswerExtractor,
        agent_config: AgentConfig,
    ):
        self.llm_client = llm_client
        self.answer_extractor = answer_extractor
        self.agent_config = agent_config

    async def run(
        self,
        generation_request: GenerateRequest,
        seed_key: str,
        agent_id: int | None = None,
        round_id: int | None = None,
        prepare_prompt: bool = True,
    ) -> AgentResponse:
        request = self._build_generation_request(
            generation_request,
            seed_key,
            prepare_prompt,
        )

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
            seed=request.seed,
            temperature=request.temperature,
            latency_s=t.elapsed,
            provider_duration_s=llm_response.duration_s,
            agent_id=agent_id,
            round_id=round_id,
        )

    def _build_generation_request(
        self,
        generation_request: GenerateRequest,
        seed_key: str,
        prepare_prompt: bool,
    ) -> GenerateRequest:
        prompt = generation_request.prompt
        if prepare_prompt:
            prompt = self.answer_extractor.prepare_prompt(prompt)
        if self.agent_config.system_prompt is not None:
            prompt = f"{self.agent_config.system_prompt}\n\nTask:\n{prompt}"

        seed = generation_request.seed
        if seed is not None:
            seed = derive_seed(
                seed,
                generation_request.prompt,
                seed_key,
            )

        temperature = generation_request.temperature
        if temperature is None:
            temperature = self.agent_config.base_temperature

        return generation_request.model_copy(
            update={
                "prompt": prompt,
                "stop": self.answer_extractor.prepare_stop(generation_request.stop),
                "seed": seed,
                "temperature": temperature,
            }
        )
