from agents.agent import Agent
from agents.models import AgentConfig
from extractors.base import AnswerExtractor
from llm.base import LLMClient


class AgentFactory:
    def __init__(
        self,
        llm_client: LLMClient,
        answer_extractor: AnswerExtractor,
    ):
        self.llm_client = llm_client
        self.answer_extractor = answer_extractor

    def create(self, name: str, role: str) -> Agent:
        return Agent(
            llm_client=self.llm_client,
            answer_extractor=self.answer_extractor,
            agent_config=AgentConfig(name=name, role=role),
        )
