from agents.agent import Agent
from agents.models import AgentConfig
from extractors.base import AnswerExtractor
from llm.base import LLMClient


class AgentFactory:
    @staticmethod
    def create(
        agent_config: AgentConfig,
        llm_client: LLMClient,
        answer_extractor: AnswerExtractor,
    ) -> Agent:
        return Agent(
            llm_client=llm_client,
            answer_extractor=answer_extractor,
            agent_config=agent_config,
        )
