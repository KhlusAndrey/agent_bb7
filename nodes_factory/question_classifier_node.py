# nodes/question_classifier_node.py
from .base_node import BaseNode
from states.state import AgentState, GradeQuestion


class QuestionClassifierNode(BaseNode):
    def __call__(self, state: AgentState):
        return self.process(state)

    def process(self, state: AgentState):
        question = state["question"]

        system_message = self._prompts.get_analysis_prompt()
        prompt = self.create_prompt(system_message, f"User question: {question}")

        structured_llm = self.llm.with_structured_output(GradeQuestion)
        grader_llm = prompt | structured_llm
        response = grader_llm.invoke({"question": question})

        state["on_topic"] = response.score.strip().upper() == "YES"
        state.get("messages", []).append(response)
        return state
