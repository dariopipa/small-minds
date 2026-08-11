from collections import Counter

from agents.agent import Agent
from llm.requests import GenerateRequest
from prompts import load_prompt
from strategies.models import StrategyResult
from strategies.strategy_interface import StrategyI


# society should be changed into something else like interaction with verification
class SocietyOfMindsStrategy(StrategyI):
    def __init__(self, agent: Agent, agent_number: int, debate_rounds: int):
        self.agent = agent
        self.agent_number = agent_number
        self.debate_rounds = debate_rounds
        self.revision_prompt = load_prompt("society_of_minds", "revision")

    async def run(self, generation_request: GenerateRequest) -> StrategyResult:
        agent_responses = []
        current_responses = []

        print("\n" + "=" * 80)
        print("SOCIETY OF MINDS START")
        print(f"agents: {self.agent_number}")
        print(f"debate rounds: {self.debate_rounds}")
        print("=" * 80)

        for agent_index in range(self.agent_number):
            agent_response = await self.agent.run(generation_request)
            agent_responses.append(agent_response)
            current_responses.append(agent_response)

            print(
                "INITIAL ANSWER "
                f"agent={agent_index + 1} "
                f"extracted={agent_response.extracted_response}"
            )

        for round_index in range(self.debate_rounds):
            print("-" * 80)
            print(f"SOCIETY OF MINDS ROUND {round_index + 1}")
            print("-" * 80)

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

                print(
                    "UPDATED ANSWER "
                    f"round={round_index + 1} "
                    f"agent={agent_index + 1} "
                    f"previous={own_response.extracted_response} "
                    f"updated={agent_response.extracted_response}"
                )

        answer_counts = Counter(
            response.extracted_response for response in current_responses
        )
        selected_answer = answer_counts.most_common(1)[0][0]

        selected_response = next(
            agent_response
            for agent_response in current_responses
            if agent_response.extracted_response == selected_answer
        )

        print("=" * 80)
        print("SOCIETY OF MINDS FINAL VOTE")
        print(f"votes: {dict(answer_counts)}")
        print(f"selected answer: {selected_answer}")
        print("-" * 80)
        print("SOCIETY OF MINDS SELECTED RESPONSE")
        print(selected_response.response)
        print("=" * 80)

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
