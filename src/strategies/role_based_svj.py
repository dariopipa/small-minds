from agents.agent import Agent
from llm.requests import GenerateRequest
from prompts import load_prompt
from strategies.base import Strategy
from strategies.models import StrategyResult


class RoleBasedSVJStrategy(Strategy):
    def __init__(self, solver: Agent, verifier: Agent, judge: Agent):
        self.solver = solver
        self.verifier = verifier
        self.judge = judge
        self.solver_prompt = load_prompt("strategies", "role_based_svj", "solver")
        self.verifier_prompt = load_prompt("strategies", "role_based_svj", "verifier")
        self.judge_prompt = load_prompt("strategies", "role_based_svj", "judge")

    async def run(self, generation_request: GenerateRequest) -> StrategyResult:
        question = generation_request.prompt
        solver_response = await self.solver.run(
            generation_request.model_copy(
                update={
                    "prompt": self.solver_prompt.format(question=question),
                    "stop": None,
                }
            )
        )
        verifier_response = await self.verifier.run(
            generation_request.model_copy(
                update={
                    "prompt": self.verifier_prompt.format(
                        question=question,
                        solver_output=solver_response.response,
                    ),
                    "stop": None,
                }
            )
        )
        judge_response = await self.judge.run(
            generation_request.model_copy(
                update={
                    "prompt": self.judge_prompt.format(
                        question=question,
                        solver_output=solver_response.response,
                        verifier_output=verifier_response.response,
                    )
                }
            )
        )

        agent_responses = [solver_response, verifier_response, judge_response]
        return StrategyResult(
            model=judge_response.model,
            strategy_name="role_based_svj",
            prompt=question,
            response=self.judge.answer_extractor.normalize_final_response(
                judge_response.response
            ),
            extracted_response=judge_response.extracted_response,
            prompt_tokens=sum(response.prompt_tokens for response in agent_responses),
            output_tokens=sum(response.output_tokens for response in agent_responses),
            total_latency_s=sum(
                response.latency_s or 0.0 for response in agent_responses
            ),
            agent_responses=agent_responses,
        )
