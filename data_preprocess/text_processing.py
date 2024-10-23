from transformers import pipeline
from langdetect import detect
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import dotenv
import os

_ = dotenv.load_dotenv(dotenv.find_dotenv())

"""
This script is used to preprocess comments from the user and extracts the sentiment and emotion of the comment.
"""


def classify_comment_roberta(comment: str) -> tuple[str, int, str]:
    sentiment_pipeline = pipeline("sentiment-analysis")
    emotions_pipeline = pipeline(
        task="text-classification",
        model="SamLowe/roberta-base-go_emotions",
        token=os.getenv("HF_TOKEN"),
    )
    language = detect(comment)
    sentiment = sentiment_pipeline(comment)[0]["label"]
    emotion = emotions_pipeline(comment)[0]["label"]
    return (sentiment, emotion, language)


class Classification(BaseModel):
    sentiment: str = Field(
        ..., enum=["awesome", "happy", "neutral", "sad", "angry", "confused"]
    )
    emotion: str = Field(
        ...,
        description="describe in one word what emotion the comment is",
            )
    language: str = Field(
        ...,
        enum=[
            "spanish",
            "english",
            "french",
            "german",
            "italian",
            "portuguese",
            "finish",
            "dutch",
        ],
    )


def analyze_comment_langchain(comment: str) -> tuple[str, str, str]:
    tagging_prompt = ChatPromptTemplate.from_template(
        """
    Extract the desired information from the following passage.

    Only extract the properties mentioned in the 'Classification' function.

    Passage:
    {input}
    """
    )
    llm = ChatOpenAI(temperature=0, 
                     model="gpt-4o-mini"
                     ).with_structured_output(Classification)
    chain = tagging_prompt | llm
    response = chain.invoke({"input": comment}).dict()
    sentiment = response["sentiment"]
    emotion = response["emotion"]
    language = response["language"]

    return (sentiment, emotion, language)


if __name__ == "__main__":
    comment = "This product is terrible and I'm very disappointed!"
    sentiment, emotion, language = classify_comment_roberta(comment)
    print(f"Comment: {comment}")
    print(f"Sentiment: {sentiment}, Emotion: {emotion}, Language: {language}")
