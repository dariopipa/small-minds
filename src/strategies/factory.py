from agents.factory import AgentFactory
from prompts import load_prompt
from strategies.base import Strategy
from strategies.direct import DirectStrategy
from strategies.models import StrategyConfig
from strategies.role_based_svj import RoleBasedSVJStrategy
from strategies.self_consistency import SelfConsistencyStrategy
from strategies.society_of_minds import SocietyOfMindsStrategy

SOLVER_PROMPT = "shared/solver"


# mypy: disable-error-code=union-attr
class StrategyFactory:
    @staticmethod
    def create_strategy(
        strategy_config: StrategyConfig,
        agent_factory: AgentFactory,
        benchmark_name: str,
    ) -> Strategy:

        match strategy_config.name:
            case "direct":
                return DirectStrategy(
                    agent=agent_factory.create(
                        name=strategy_config.name,
                        role="solver",
                        system_prompt=load_prompt(SOLVER_PROMPT),
                    )
                )

            case "role_based_svj":
                prompt_directory = f"benchmarks/{benchmark_name}/role_based_svj"
                return RoleBasedSVJStrategy(
                    solver=agent_factory.create(
                        name=strategy_config.name,
                        role="solver",
                        system_prompt=(
                            f"{load_prompt(SOLVER_PROMPT)}\n\n"
                            f"{load_prompt(f'{prompt_directory}/solver_system')}"
                        ),
                    ),
                    verifier=agent_factory.create(
                        name="role_based_svj_verifier",
                        role="verifier",
                        system_prompt=load_prompt(
                            f"{prompt_directory}/verifier_system"
                        ),
                    ),
                    judge=agent_factory.create(
                        name="role_based_svj_judge",
                        role="judge",
                    ),
                    prompt_directory=prompt_directory,
                )

            case "self_consistency":
                return SelfConsistencyStrategy(
                    agent=agent_factory.create(
                        name=strategy_config.name,
                        role="solver",
                        system_prompt=load_prompt(SOLVER_PROMPT),
                    ),
                    agent_number=strategy_config.agent_number,
                )

            case "society_of_minds":
                return SocietyOfMindsStrategy(
                    agent=agent_factory.create(
                        name=strategy_config.name,
                        role="solver",
                        system_prompt=load_prompt(SOLVER_PROMPT),
                    ),
                    agent_number=strategy_config.agent_number,
                    debate_rounds=strategy_config.debate_rounds,
                    revision_prompt=load_prompt(
                        f"benchmarks/{benchmark_name}/society_of_minds/revision"
                    ),
                )

            case _:
                raise ValueError(f"Unsupported strategy: {strategy_config}")
