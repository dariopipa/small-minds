from agents.agent import Agent
from llm.requests import GenerateRequest
from prompts import load_prompt
from strategies.base import Strategy
from strategies.models import StrategyResult


class RoleBasedSVJStrategy(Strategy):
    def __init__(
        self,
        solver: Agent,
        verifier: Agent,
        judge: Agent,
        prompt_directory: str,
    ):
        self.solver = solver
        self.verifier = verifier
        self.judge = judge
        self.verifier_task_prompt = load_prompt(f"{prompt_directory}/verifier_task")
        self.judge_task_prompt = load_prompt(f"{prompt_directory}/judge_task")

    async def run(self, generation_request: GenerateRequest) -> StrategyResult:
        followup_context = self.solver.answer_extractor.prepare_followup_context(
            generation_request.prompt
        )
        solver_response = await self.solver.run(
            generation_request,
            seed_key="direct:solver",
        )
        solver_answer = solver_response.extracted_response

        verifier_response = await self.verifier.run(
            generation_request.model_copy(
                update={
                    "prompt": self.verifier_task_prompt.format(
                        question=followup_context,
                        solver_response=solver_response.response,
                        solver_answer=solver_answer or "UNPARSEABLE",
                    ),
                    "stop": None,
                    "temperature": 0.0,
                }
            ),
            seed_key="role_based_svj:verifier",
            prepare_prompt=False,
        )
        verifier_answer = verifier_response.extracted_response

        responses = [solver_response, verifier_response]

        judge_response = await self.judge.run(
            generation_request.model_copy(
                update={
                    "prompt": self.judge_task_prompt.format(
                        question=followup_context,
                        solver_response=solver_response.response,
                        solver_answer=solver_answer or "UNPARSEABLE",
                        verifier_response=verifier_response.response,
                        verifier_answer=verifier_answer or "UNPARSEABLE",
                    ),
                    "stop": None,
                    "temperature": 0.0,
                }
            ),
            seed_key="role_based_svj:judge",
            prepare_prompt=False,
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
            provider_duration_s=sum(r.provider_duration_s or 0 for r in responses),
            agent_responses=responses,
        )
