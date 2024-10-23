from typing import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel


class GradeQuestion(BaseModel):
    score: str  # "YES" or "NO"


class AgentState(TypedDict):
    question: str
    messages: list[BaseMessage, add_messages]
    on_topic: bool
    should_report: bool
    plan: str
    report: str
    agent_response: str
    tool_calls: list[str]
