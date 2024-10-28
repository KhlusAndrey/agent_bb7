from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from typing import Type

class LLMClient:
    def __init__(self, model_name="gpt-4o-mini", temperature=0):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.structured_output_type = None

    def with_structured_output(self, output_type: Type):
        """
        Define the type of the structured output of the LLM.
        
        Args:
            output_type (Type): Class or structure of the structured output.
            
        Returns:
            LLMClient: Copy of the LLMClient with the new structured output type.
        """
        client_with_structure = LLMClient(model_name=self.model_name, temperature=self.temperature)
        client_with_structure.structured_output_type = output_type
        return client_with_structure

    def invoke(self, prompt: str | ChatPromptTemplate, input_data: dict = None):
        """
        Invoke the LLM with the given prompt and input data.
        
        Args:
            prompt (str | ChatPromptTemplate): Prompt or template for the LLM.
            input_data dict: Input data for the LLM.
            
        Returns:
            Text or structured output format of the LLM.
        """
        if self.structured_output_type:
    
            structured_llm = self.llm.with_structured_output(self.structured_output_type)
            
            response = structured_llm.invoke(input_data or {})
        else:
            formatted_prompt = prompt if isinstance(prompt, str) else prompt.format_messages()
            
            response = self.llm.invoke(formatted_prompt)
        
        return response