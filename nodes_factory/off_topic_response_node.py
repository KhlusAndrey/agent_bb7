from .base_node import BaseNode
from states.state import AgentState


class OffTopicResponseNode(BaseNode):
    def __call__(self, state: AgentState):
        return self.process(state)

    def process(self, state: AgentState):
        question = state["question"]

        system_message = self._prompts.get_off_topic_prompt(user_question=question)
        prompt = self.create_prompt(system_message, f"User question: {question}")
        formatted_prompt = prompt.format_messages()
        response = self.llm.invoke(formatted_prompt)
        state["agent_response"] = response.content
        state.get("messages", []).append(response)
        return state
