from agents.agent import Agent
from strategies.direct import DirectStrategy
from strategies.models import StrategyConfig
from strategies.self_consistency import SelfConsistencyStrategy
from strategies.strategy_interface import StrategyI


# mypy: disable-error-code=union-attr
class StrategyFactory:
    @staticmethod
    def create_strategy(strategy_config: StrategyConfig, agent: Agent) -> StrategyI:
        match strategy_config.name:
            case "direct":
                return DirectStrategy(agent=agent)
            case "self-consistency":
                return SelfConsistencyStrategy(
                    agent=agent,
                    agent_number=strategy_config.agent_number,
                )
            case _:
                raise ValueError(f"Unsupported strategy: {strategy_config}")
