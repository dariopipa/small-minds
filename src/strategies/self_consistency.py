from collections import Counter

from agents.agent import Agent
from llm.requests import GenerateRequest
from strategies.base import Strategy
from strategies.models import StrategyResult


def _get_answers(responses) -> list[str]:
    return [
        response.extracted_response
        for response in responses
        if response.extracted_response is not None
    ]


def _is_tied(answers: list[str]) -> bool:
    top = Counter(answers).most_common(2)
    return len(top) > 1 and top[0][1] == top[1][1]


class SelfConsistencyStrategy(Strategy):
    def __init__(self, agent: Agent, agent_number: int):
        self.agent = agent
        self.agent_number = agent_number

    async def run(self, generation_request: GenerateRequest) -> StrategyResult:
        responses = []

        for i in range(self.agent_number):
            responses.append(
                await self.agent.run(
                    generation_request,
                    seed_key=f"self_consistency:candidate:{i}",
                )
            )

        answers = _get_answers(responses)

        if _is_tied(answers):
            responses.append(
                await self.agent.run(
                    generation_request,
                    seed_key=f"self_consistency:candidate:{self.agent_number}",
                )
            )
            answers = _get_answers(responses)

        selected_answer = Counter(answers).most_common(1)[0][0] if answers else None

        selected_response = next(
            (
                response
                for response in responses
                if response.extracted_response == selected_answer
            ),
            responses[0],
        )

        return StrategyResult(
            model=selected_response.model,
            strategy_name="self_consistency",
            prompt=generation_request.prompt,
            response=self.agent.answer_extractor.normalize_final_response(
                selected_response.response
            ),
            extracted_response=selected_answer,
            prompt_tokens=sum(r.prompt_tokens for r in responses),
            output_tokens=sum(r.output_tokens for r in responses),
            total_latency_s=sum(r.latency_s or 0 for r in responses),
            provider_duration_s=sum(r.provider_duration_s or 0 for r in responses),
            agent_responses=responses,
        )
