from helper.llm_client import LLMClient
from prompts.prompts import Prompts


class BaseNode:
    def __init__(self, model_name="gpt-4o-mini", temperature=0):
        self.llm_client = LLMClient(model_name=model_name, temperature=temperature)
        self._prompts = Prompts()

    def create_prompt(self, system_message: str, user_message: str):
        """Creates the prompt for the LLM."""
        return [
            ("system", system_message),
            ("human", user_message),
        ]

    def invoke(self, prompt, structured_output=None, input_data=None):
        """
        Invokes the LLM with the given prompt and input data.

        Args:
            prompt: Prompt or template for the LLM.
            structured_output Type: Class or structure of the structured output.
            input_data dict: Input data for the LLM.

        Returns:
            Response: Response from the LLM text or structured output.
        """
        if structured_output:
            structured_llm = self.llm_client.with_structured_output(structured_output)
            grader_llm = prompt | structured_llm
            response = grader_llm.invoke(input_data or {})
        else:
            formatted_prompt = prompt.format_messages()
            response = self.llm_client.invoke(formatted_prompt)

        return response
