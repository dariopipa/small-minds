from agents.agent import Agent
from strategies.direct_strategy import DirectStrategy
from strategies.models import StrategyConfig
from strategies.strategy_interface import StrategyI


class StrategyFactory:
    @staticmethod
    def create_strategy(strategy_config: StrategyConfig, agent: Agent) -> StrategyI:
        match strategy_config.name:
            case "direct":
                return DirectStrategy(agent=agent)
            case _:
                raise ValueError(f"Unsupported strategy: {strategy_config}")
