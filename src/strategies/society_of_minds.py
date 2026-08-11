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
        self.revision_prompt = load_prompt(
            "strategies",
            "society_of_minds",
            "revision",
        )

    async def run(self, generation_request: GenerateRequest) -> StrategyResult:
        agent_responses = []
        current_responses = []

        for _ in range(self.agent_number):
            agent_response = await self.agent.run(generation_request)
            agent_responses.append(agent_response)
            current_responses.append(agent_response)

        # The reference implementation counts the independent generation as
        # round one. Later rounds update each answer using the other agents'
        # responses from the preceding round.
        for _ in range(1, self.debate_rounds):
            previous_responses = current_responses
            current_responses = []

            for agent_index, own_response in enumerate(previous_responses):
                other_agents = previous_responses[:agent_index]
                other_agents += previous_responses[agent_index + 1 :]
                other_responses = "\n\n".join(
                    other_response.response for other_response in other_agents
                )
                debate_prompt = self.revision_prompt.format(
                    question=generation_request.prompt,
                    own_response=own_response.response,
                    other_responses=other_responses,
                )

                debate_request = generation_request.model_copy(
                    update={"prompt": debate_prompt, "stop": None}
                )
                agent_response = await self.agent.run(debate_request)
                agent_responses.append(agent_response)
                current_responses.append(agent_response)

        answer_counts = Counter(
            response.extracted_response for response in current_responses
        )
        selected_answer = answer_counts.most_common(1)[0][0]

        selected_response = next(
            agent_response
            for agent_response in current_responses
            if agent_response.extracted_response == selected_answer
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
            agent_responses=agent_responses,
        )
