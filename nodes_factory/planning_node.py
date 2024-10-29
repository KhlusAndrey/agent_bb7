from tools.fake_tools import (
    get_fake_relevant_documents_chromadb,
    get_fake_sql_query_result,
)
from tools.tavily_search import load_tavily_search_tool
from .base_node import BaseNode
from states.state import AgentState
from langchain_core.messages import ToolMessage


class PlanningNode(BaseNode):
    def __call__(self, state: AgentState):
        return self.process(state)

    def process(self, state: AgentState):
        question = state["question"]
        tavily_search_results_json = load_tavily_search_tool(
            tavily_search_max_results=2
        )

        system_message = self._prompts.get_research_planning_prompt(
            user_question=question,
            tool_names="get_fake_sql_query_result, get_fake_relevant_documents_chromadb, tavily_search_results_json",
        )
        prompt = self.create_prompt(system_message, f"User question: {question}")
        formatted_prompt = prompt.format_messages()
        initial_response = self.llm.invoke(formatted_prompt)

        formatted_prompt.append(initial_response)

        llm_with_tools = self.llm.bind_tools(
            [
                get_fake_sql_query_result,
                get_fake_relevant_documents_chromadb,
                tavily_search_results_json,
            ]
        )
        response_with_tools = llm_with_tools.invoke(formatted_prompt)

        tool_messages = []
        for tool_call in response_with_tools.tool_calls:
            tool_name = tool_call["name"].lower()
            args = tool_call["args"]

            if tool_name == "get_fake_sql_query_result":
                tool_output = get_fake_sql_query_result.invoke(args)
            elif tool_name == "get_fake_relevant_documents_chromadb":
                tool_output = get_fake_relevant_documents_chromadb.invoke(args)
            elif tool_name == "tavily_search_results_json":
                tool_output = tavily_search_results_json.invoke(args)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

            tool_messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

        formatted_prompt.extend(tool_messages)

        state["plan"] = initial_response.content
        state.get("messages", []).append(response_with_tools)
        state.get("tool_calls", []).append(tool_messages)

        return state
