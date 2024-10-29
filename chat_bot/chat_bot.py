from graph_builder.build_graph_nodes_factory import build_graph
from helper.memory import Memory


class ChatBot:
    """
    Class for handling chatbot responses.
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.graph = build_graph()
        self.config = {"configurable": {"thread_id": user_id}}

    def respond(self, chatbot_history: list, message: str) -> tuple:
        """
        ОArgs:
            chatbot_history (List): Chatbot history.
            message (str): User input.

        Returns:
            tuple: Tuple containing the response and the updated chatbot history.
        """

        events = self.graph.compile().stream(
            {"question": message}, self.config, stream_mode="values"
        )
        response = list(events)[-1]["agent_response"]

        chatbot_history.append((message, response))

        Memory.write_chat_history_to_file(
            gradio_chatbot=chatbot_history, folder_path="./db", thread_id=self.user_id
        )

        return "", chatbot_history