# from helper.llm_client import LLMClient
from langchain_core.prompts import ChatPromptTemplate
from prompts.prompts import Prompts
from langchain_openai import ChatOpenAI


class BaseNode:
    def __init__(self, model_name="gpt-4o-mini", temperature=0):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self._prompts = Prompts()

    def create_prompt(self, system_message: str, user_message: str):
        """Creates the prompt for the LLM."""
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("human", user_message),
            ]
        )
