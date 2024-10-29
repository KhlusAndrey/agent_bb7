from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from states.state import AgentState, GradeQuestion
from prompts.prompts import Prompts
from tools.fake_tools import (
    get_fake_relevant_documents_chromadb,
    get_fake_sql_query_result,
)
from tools.tavily_search import load_tavily_search_tool
from langchain_core.messages import ToolMessage
from rich import print

prompt_getter = Prompts()
tavily_search_results_json = load_tavily_search_tool(tavily_search_max_results=2)


def question_classifier_node(state: AgentState):
    print(f"{'QUESTION CLASSIFIER NODE STATE':-^80}")
    print(state, "question_classifier_node")
    question = state["question"]

    system = prompt_getter.get_analysis_prompt()

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", f"User question: {question}"),
        ]
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    structured_llm = llm.with_structured_output(GradeQuestion)
    grader_llm = grade_prompt | structured_llm
    response = grader_llm.invoke({"question": question})

    state["on_topic"] = response.score.strip().upper() == "YES"
    return state


def judgment_report_node(state: AgentState):
    print(f"{'JUDGMENT REPORT NODE STATE':-^80}")
    print(state)
    question = state["question"]
    report = state["report"]
    plan = state["plan"]
    system = prompt_getter.get_report_validation_prompt(
        report=report, plan=plan, user_question=question
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", f"User question: {question}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(GradeQuestion)
    grader_llm = prompt | structured_llm
    response = grader_llm.invoke({"question": question})

    state["should_report"] = response.score.strip().upper() == "YES"
    return state


def planning_node(state: AgentState):
    print(f"{'PLANNING NODE STATE':-^80}")
    print(state)
    question = state["question"]
    system = prompt_getter.get_research_planning_prompt(
        user_question=question,
        tool_names="get_fake_sql_query_result, get_fake_relevant_documents_chromadb, tavily_search_results_json",
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", f"User question: {question}"),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    message = prompt.format_messages()
    response = llm.invoke(message)
    message.append(response)

    llm_with_tools = llm.bind_tools(
        [
            get_fake_sql_query_result,
            get_fake_relevant_documents_chromadb,
            tavily_search_results_json,
        ]
    )

    final_plan = llm_with_tools.invoke(message)

    tool_messages = []
    for tool_call in final_plan.tool_calls:
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
    message.extend(tool_messages)
    state["plan"] = response.content
    state["messages"] = message
    state["tool_calls"] = tool_messages
    return state


def off_topic_response_node(state: AgentState):
    print(f"{'OFF TOPIC RESPONSE NODE STATE':-^80}")
    print(state)
    question = state["question"]
    system = prompt_getter.get_off_topic_prompt(user_question=question)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", f"User question: {question}"),
        ]
    )
    formatted_prompt = prompt.format_messages()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(formatted_prompt)
    state["agent_response"] = response.content
    return state


def write_analytic_report_node(state: AgentState):
    print(f"{'WRITE ANALYTIC REPORT NODE STATE':-^80}")
    print(state)
    question = state["question"]
    plan = state["plan"]

    system = prompt_getter.get_report_writing_prompt(user_question=question, plan=plan)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", f"User question: {question}"),
        ]
    )

    formatted_prompt = prompt.format_messages()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(formatted_prompt)
    state["report"] = response.content
    state["agent_response"] = response.content
    return state


def ask_to_rewrite_question_node(state: AgentState):
    print(f"{'ASK TO REWRITE QUESTION NODE STATE':-^80}")
    print(state)
    question = state["question"]
    plan = state["plan"]
    system = prompt_getter.get_clarification_question_prompt(
        user_question=question, plan=plan
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", f"User question: {question}"),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    formatted_prompt = prompt.format_messages()
    response = llm.invoke(formatted_prompt)
    state["agent_response"] = response.content
    return state
