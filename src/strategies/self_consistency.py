from collections import Counter

from agents.agent import Agent
from llm.requests import GenerateRequest
from strategies.models import StrategyResult
from strategies.strategy_interface import StrategyI


class SelfConsistencyStrategy(StrategyI):
    def __init__(self, agent: Agent, agent_number: int):
        self.agent = agent
        self.agent_number = agent_number

    async def run(self, generation_request: GenerateRequest) -> StrategyResult:
        agent_responses = []

        for _ in range(self.agent_number):
            agent_responses.append(await self.agent.run(generation_request))

        selected_answer = Counter(
            response.extracted_response for response in agent_responses
        ).most_common(1)[0][0]

        selected_response = next(
            agent_response
            for agent_response in agent_responses
            if agent_response.extracted_response == selected_answer
        )

        return StrategyResult(
            model=selected_response.model,
            strategy_name="self-consistency",
            prompt=generation_request.prompt,
            response=selected_response.response,
            extracted_response=selected_answer,
            prompt_tokens=sum(
                agent_response.prompt_tokens for agent_response in agent_responses
            ),
            output_tokens=sum(
                agent_response.output_tokens for agent_response in agent_responses
            ),
            total_latency_s=sum(
                agent_response.latency_s or 0 for agent_response in agent_responses
            ),
            agent_responses=agent_responses,
        )
