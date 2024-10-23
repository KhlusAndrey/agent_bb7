from langchain_openai import ChatOpenAI
import os
import dotenv

_ = dotenv.load_dotenv(dotenv.find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_openai_model(temperature=0, model='gpt-4o-mini'):
    llm = ChatOpenAI(
    model=model,
    temperature = temperature,
)
    return llm

def get_openai_json_model(temperature=0, model='gpt-4o-mini'):
    llm = ChatOpenAI(
    model=model,
    temperature = temperature,
    model_kwargs={"response_format": {"type": "json_object"}},
)
    return llm