from agents.agent import Agent
from llm.requests import GenerateRequest
from strategies.strategy_interface import StrategyI


class DirectStrategy(StrategyI):
    def __init__(self, agent: Agent):
        self.agent = agent

    async def run(self, generation_request: GenerateRequest):
        return await self.agent.run(generation_request)
