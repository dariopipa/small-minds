from agents.agent import Agent
from llm.requests import GenerateRequest
from strategies.models import StrategyResult
from strategies.strategy_interface import StrategyI


class DirectStrategy(StrategyI):
    def __init__(self, agent: Agent):
        self.agent = agent

    async def run(self, generation_request: GenerateRequest) -> StrategyResult:
        agent_response = await self.agent.run(generation_request)

        return StrategyResult(
            model=agent_response.model,
            strategy_name="direct",
            prompt=generation_request.prompt,
            response=agent_response.response,
            extracted_response=agent_response.extracted_response,
            prompt_tokens=agent_response.prompt_tokens,
            output_tokens=agent_response.output_tokens,
            total_latency_s=agent_response.latency_s,
            agent_responses=[agent_response],
        )
