from .base_node import BaseNode
from states.state import AgentState


class WriteAnalyticReportNode(BaseNode):
    def __call__(self, state: AgentState):
        return self.process(state)

    def process(self, state: AgentState):
        question = state["question"]
        plan = state["plan"]

        system_message = self._prompts.get_report_writing_prompt(
            user_question=question, plan=plan
        )
        prompt = self.create_prompt(system_message, f"User question: {question}")
        formatted_prompt = prompt.format_messages()
        response = self.llm.invoke(formatted_prompt)

        state["report"] = response.content
        state["agent_response"] = response.content
        state.get("messages", []).append(response)

        return state
