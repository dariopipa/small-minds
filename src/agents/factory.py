from agents.agent import Agent
from agents.models import AgentConfig
from extractors.base import AnswerExtractor
from llm.base import LLMClient


class AgentFactory:
    def __init__(
        self,
        llm_client: LLMClient,
        answer_extractor: AnswerExtractor,
        base_seed: int | None = None,
        base_temperature: float | None = None,
    ):
        self.llm_client = llm_client
        self.answer_extractor = answer_extractor
        self.base_seed = base_seed
        self.base_temperature = base_temperature

    def create(
        self,
        name: str,
        role: str,
        system_prompt: str | None = None,
    ) -> Agent:
        return Agent(
            llm_client=self.llm_client,
            answer_extractor=self.answer_extractor,
            agent_config=AgentConfig(
                name=name,
                role=role,
                system_prompt=system_prompt,
                base_seed=self.base_seed,
                base_temperature=self.base_temperature,
            ),
        )
