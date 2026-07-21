from agents.agent import Agent
from agents.models import AgentConfig
from extractors.answer_extractor_interface import AnswerExtractorI
from llm.client_interface import LLMClientI


class AgentFactory:
    @staticmethod
    def create(
        agent_config: AgentConfig,
        llm_client: LLMClientI,
        answer_extractor: AnswerExtractorI,
    ) -> Agent:
        return Agent(
            llm_client=llm_client,
            answer_extractor=answer_extractor,
            agent_config=agent_config,
        )
