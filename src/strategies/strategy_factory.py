from agents.agent import Agent
from strategies.direct_strategy import DirectStrategy
from strategies.strategy_interface import StrategyI


class StrategyFactory:
    @staticmethod
    def create_strategy(strategy_config: str, agent: Agent) -> StrategyI:
        match strategy_config:
            case "direct":
                return DirectStrategy(agent=agent)
            case _:
                raise ValueError(f"Unsupported strategy: {strategy_config}")
