from .base_node import BaseNode
from states.state import AgentState, GradeQuestion


class JudgmentReportNode(BaseNode):
    def __call__(self, state: AgentState):
        return self.process(state)

    def process(self, state: AgentState):
        question = state["question"]
        report = state["report"]
        plan = state["plan"]

        system_message = self._prompts.get_report_validation_prompt(
            report=report, plan=plan, user_question=question
        )
        prompt = self.create_prompt(system_message, f"User question: {question}")
        structured_llm = self.llm.with_structured_output(GradeQuestion)
        grader_llm = prompt | structured_llm
        response = grader_llm.invoke({"question": question})

        state["should_report"] = response.score.strip().upper() == "YES"
        state.get("messages", []).append(response)

        return state
