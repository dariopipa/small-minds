from collections import Counter

from agents.agent import Agent
from llm.requests import GenerateRequest
from prompts import load_prompt
from strategies.base import Strategy
from strategies.models import StrategyResult


class SocietyOfMindsStrategy(Strategy):
    def __init__(self, agent: Agent, agent_number: int, debate_rounds: int):
        self.agent = agent
        self.agent_number = agent_number
        self.debate_rounds = debate_rounds
        self.revision_prompt = load_prompt("strategies/society_of_minds/revision")

    async def run(self, generation_request: GenerateRequest) -> StrategyResult:
        agent_responses = []
        current_responses = []

        for agent_index in range(self.agent_number):
            agent_response = await self.agent.run(
                generation_request,
                seed_key=f"candidate:{agent_index}",
                agent_id=agent_index + 1,
                round_id=1,
            )
            agent_responses.append(agent_response)
            current_responses.append(agent_response)

        initial_answer = self._majority_answer(current_responses)

        # The reference implementation counts the independent generation as
        # round one. Later rounds update each answer using the other agents'
        # responses from the preceding round.
        for round_index in range(1, self.debate_rounds):
            previous_responses = current_responses
            if self._has_consensus(previous_responses):
                break

            current_responses = []

            for agent_index, own_response in enumerate(previous_responses):
                other_indices = [
                    index
                    for index in range(len(previous_responses))
                    if index != agent_index
                ]
                other_responses = "\n\n".join(
                    f"Candidate {candidate_index + 1}:\n"
                    f"{previous_responses[other_index].response}"
                    for candidate_index, other_index in enumerate(other_indices)
                )
                debate_prompt = self.revision_prompt.format(
                    question=generation_request.prompt,
                    own_response=own_response.response,
                    other_responses=other_responses,
                )

                debate_request = generation_request.model_copy(
                    update={"prompt": debate_prompt, "stop": None}
                )
                agent_response = await self.agent.run(
                    debate_request,
                    seed_key=f"revision:{round_index}:{agent_index}",
                    agent_id=agent_index + 1,
                    round_id=round_index + 1,
                )
                agent_responses.append(agent_response)
                current_responses.append(agent_response)

        selected_answer = self._majority_answer(current_responses)

        selected_response = next(
            (
                agent_response
                for agent_response in current_responses
                if agent_response.extracted_response == selected_answer
            ),
            current_responses[0],
        )

        return StrategyResult(
            model=selected_response.model,
            strategy_name="society_of_minds",
            prompt=generation_request.prompt,
            response=self.agent.answer_extractor.normalize_final_response(
                selected_response.response
            ),
            extracted_response=selected_answer,
            prompt_tokens=sum(
                agent_response.prompt_tokens for agent_response in agent_responses
            ),
            output_tokens=sum(
                agent_response.output_tokens for agent_response in agent_responses
            ),
            total_latency_s=sum(
                agent_response.latency_s or 0 for agent_response in agent_responses
            ),
            provider_duration_s=sum(
                agent_response.provider_duration_s or 0
                for agent_response in agent_responses
            ),
            initial_extracted_response=initial_answer,
            agent_responses=agent_responses,
        )

    @staticmethod
    def _majority_answer(responses) -> str | None:
        valid_answers = [
            response.extracted_response
            for response in responses
            if response.extracted_response is not None
        ]
        return Counter(valid_answers).most_common(1)[0][0] if valid_answers else None

    @staticmethod
    def _has_consensus(responses) -> bool:
        answers = [response.extracted_response for response in responses]
        return bool(answers) and answers[0] is not None and len(set(answers)) == 1
