from agents.factory import AgentFactory
from strategies.base import Strategy
from strategies.direct import DirectStrategy
from strategies.models import StrategyConfig
from strategies.role_based_svj import RoleBasedSVJStrategy
from strategies.self_consistency import SelfConsistencyStrategy
from strategies.society_of_minds import SocietyOfMindsStrategy


# mypy: disable-error-code=union-attr
class StrategyFactory:
    @staticmethod
    def create_strategy(
        strategy_config: StrategyConfig,
        agent_factory: AgentFactory,
    ) -> Strategy:
        match strategy_config.name:
            case "direct":
                return DirectStrategy(
                    agent=agent_factory.create(
                        name=strategy_config.name,
                        role="solver",
                    )
                )
            case "role_based_svj":
                return RoleBasedSVJStrategy(
                    solver=agent_factory.create(
                        name=strategy_config.name,
                        role="solver",
                    ),
                    verifier=agent_factory.create(
                        name="role_based_svj_verifier",
                        role="verifier",
                    ),
                    judge=agent_factory.create(
                        name="role_based_svj_judge",
                        role="judge",
                    ),
                )
            case "self_consistency":
                return SelfConsistencyStrategy(
                    agent=agent_factory.create(
                        name=strategy_config.name,
                        role="solver",
                    ),
                    agent_number=strategy_config.agent_number,
                )
            case "society_of_minds":
                return SocietyOfMindsStrategy(
                    agent=agent_factory.create(
                        name=strategy_config.name,
                        role="solver",
                    ),
                    agent_number=strategy_config.agent_number,
                    debate_rounds=strategy_config.debate_rounds,
                )
            case _:
                raise ValueError(f"Unsupported strategy: {strategy_config}")
