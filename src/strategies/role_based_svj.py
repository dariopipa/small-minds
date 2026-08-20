from agents.agent import Agent
from common.seeding import derive_seed
from llm.requests import GenerateRequest
from prompts import load_prompt
from strategies.base import Strategy
from strategies.models import StrategyResult


class RoleBasedSVJStrategy(Strategy):
    def __init__(self, solver: Agent, verifier: Agent, judge: Agent):
        self.solver = solver
        self.verifier = verifier
        self.judge = judge
        self.judge_prompt = load_prompt("strategies/role_based_svj/judge")

    async def run(self, generation_request: GenerateRequest) -> StrategyResult:
        solver_response = await self.solver.run(
            generation_request,
            seed_key="solver",
        )

        verifier_response = await self.verifier.run(
            generation_request,
            seed_key="verifier",
        )

        responses = [solver_response, verifier_response]

        if (
            solver_response.extracted_response is not None
            and solver_response.extracted_response
            == verifier_response.extracted_response
        ):
            # Solver and verifier agree, so no judge is needed.
            return StrategyResult(
                model=solver_response.model,
                strategy_name="role_based_svj",
                prompt=generation_request.prompt,
                response=self.solver.answer_extractor.normalize_final_response(
                    solver_response.response
                ),
                extracted_response=solver_response.extracted_response,
                prompt_tokens=sum(r.prompt_tokens for r in responses),
                output_tokens=sum(r.output_tokens for r in responses),
                total_latency_s=sum(r.latency_s or 0 for r in responses),
                agent_responses=responses,
            )

        candidate_a = solver_response.response
        candidate_b = verifier_response.response

        seed = derive_seed(
            self.judge.agent_config.base_seed or 0,
            generation_request.repetition,
            "svj_candidate_order",
        )

        if seed % 2:
            candidate_a, candidate_b = candidate_b, candidate_a

        judge_response = await self.judge.run(
            generation_request.model_copy(
                update={
                    "prompt": self.judge_prompt.format(
                        question=generation_request.prompt,
                        candidate_a=candidate_a,
                        candidate_b=candidate_b,
                    ),
                    "stop": None,
                    "temperature": 0.0,
                }
            ),
            seed_key="judge",
        )

        responses.append(judge_response)

        return StrategyResult(
            model=judge_response.model,
            strategy_name="role_based_svj",
            prompt=generation_request.prompt,
            response=self.judge.answer_extractor.normalize_final_response(
                judge_response.response
            ),
            extracted_response=judge_response.extracted_response,
            prompt_tokens=sum(r.prompt_tokens for r in responses),
            output_tokens=sum(r.output_tokens for r in responses),
            total_latency_s=sum(r.latency_s or 0 for r in responses),
            agent_responses=responses,
        )
