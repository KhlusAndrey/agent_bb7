# nodes/question_classifier_node.py
from .base_node import BaseNode
from states.state import AgentState, GradeQuestion

class QuestionClassifierNode(BaseNode):
    def process(self, state: AgentState):
        question = state["question"]
        
        system_message = self._prompts.get_analysis_prompt()
        prompt = self.create_prompt(system_message, f"User question: {question}")
        
        response = self.invoke(prompt, structured_output=GradeQuestion, input_data={"question": question})
        
        state["on_topic"] = response.score.strip().upper() == "YES"
        state["messages"].append([response])
        return state