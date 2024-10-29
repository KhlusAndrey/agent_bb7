from .question_classifier_node import QuestionClassifierNode
from .judgment_report_node import JudgmentReportNode
from .planning_node import PlanningNode
from .off_topic_response_node import OffTopicResponseNode
from .write_analytic_report_node import WriteAnalyticReportNode
from .ask_to_rewrite_question_node import AskToRewriteQuestionNode

NODE_CLASSES = {
    "question_classifier_node": QuestionClassifierNode,
    "judgment_report_node": JudgmentReportNode,
    "planning_node": PlanningNode,
    "off_topic_response_node": OffTopicResponseNode,
    "write_analytic_report_node": WriteAnalyticReportNode,
    "ask_to_rewrite_question_node": AskToRewriteQuestionNode,
}

def create_node(node_name):
    node_class = NODE_CLASSES.get(node_name)
    if not node_class:
        raise ValueError(f"Unknown node: {node_name}")
    return node_class()