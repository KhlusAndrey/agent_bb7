from states.state import AgentState


def on_topic_router(state: AgentState):
    if state["on_topic"]:
        return "on_topic"
    return "off_topic"


def should_sent_router(state: AgentState):
    if state["should_report"]:
        return "should_sent"
    return "not_sent"


def ask_rewrite_router(state: AgentState):
    if state["report"].upper() == "ASK_TO_REWRITE_QUESTION":
        return "ask_user_rewrite"
    return "send_to_judgment"
