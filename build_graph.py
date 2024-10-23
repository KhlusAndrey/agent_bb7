from langgraph.graph import StateGraph, END, START
from routers.routers import should_sent_router, ask_rewrite_router, on_topic_router
from nodes.nodes import (
    question_classifier_node,
    off_topic_response_node,
    planning_node,
    write_analytic_report_node,
    judgment_report_node,
    ask_to_rewrite_question_node,
)
from rich import print

from states.state import AgentState


def build_graph():
    """
    Builds an agent decision-making graph by combining an LLM with various tools
    and defining the flow of interactions between them for analitic reporting .
    """

    graph = StateGraph(AgentState)

    graph.add_node("question_classifier_node", question_classifier_node)
    graph.add_node("off_topic_response_node", off_topic_response_node)
    graph.add_node("planning_node", planning_node)
    graph.add_node("writer_node", write_analytic_report_node)
    graph.add_node("judgment_report_node", judgment_report_node)
    graph.add_node("ask_rewrite_node", ask_to_rewrite_question_node)

    graph.add_edge(START, "question_classifier_node")
    graph.add_edge("off_topic_response_node", END)
    graph.add_edge("planning_node", "writer_node")
    graph.add_edge("ask_rewrite_node", END)
    graph.add_edge("judgment_report_node", END)

    graph.add_conditional_edges(
        "question_classifier_node",
        on_topic_router,
        {"on_topic": "planning_node", "off_topic": "off_topic_response_node"},
    )

    graph.add_conditional_edges(
        "judgment_report_node",
        should_sent_router,
        {"should_sent": END, "not_sent": "planning_node"},
    )
    graph.add_conditional_edges(
        "writer_node",
        ask_rewrite_router,
        {
            "ask_user_rewrite": "ask_rewrite_node",
            "send_to_judgment": "judgment_report_node",
        },
    )

    return graph


if __name__ == "__main__":
    graph = build_graph().compile()
    print(graph.get_graph().draw_ascii())
    # result = graph.invoke({"question": "Who Is Favored To Win The 2024 Presidential Election?"})
    # result = graph.invoke({"question": "what are people so mad about that we have so many negative comments?"})
    # print(result["agent_response"])
