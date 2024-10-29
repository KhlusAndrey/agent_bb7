from langgraph.graph import StateGraph, END, START
from nodes_factory import create_node
from routers.routers import Routers

from states.state import AgentState

def build_graph():
    graph = StateGraph(AgentState)
    nodes = {
        "question_classifier_node": create_node("question_classifier_node"),
        "off_topic_response_node": create_node("off_topic_response_node"),
        "planning_node": create_node("planning_node"),
        "writer_node": create_node("write_analytic_report_node"),
        "judgment_report_node": create_node("judgment_report_node"),
        "ask_rewrite_node": create_node("ask_to_rewrite_question_node"),
    }

    for node_name, node in nodes.items():
        graph.add_node(node_name, node)

    graph.add_edge(START, "question_classifier_node")
    graph.add_edge("off_topic_response_node", END)
    graph.add_edge("planning_node", "writer_node")
    graph.add_edge("ask_rewrite_node", END)
    graph.add_edge("judgment_report_node", END)

    graph.add_conditional_edges("question_classifier_node", Routers.on_topic_router, {
        "on_topic": "planning_node", "off_topic": "off_topic_response_node"
    })
    graph.add_conditional_edges("judgment_report_node", Routers.should_sent_router, {
        "should_sent": END, "not_sent": "planning_node"
    })
    graph.add_conditional_edges("writer_node", Routers.ask_rewrite_router, {
        "ask_user_rewrite": "ask_rewrite_node", "send_to_judgment": "judgment_report_node"
    })

    return graph