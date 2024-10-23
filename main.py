import gradio as gr
from build_graph import build_graph
from helper.memory import Memory
from rich import print
from helper.UI_like import UISettings


graph = build_graph()
config = {"configurable": {"thread_id": "bb_1"}}


class ChatBot:
    """
    Class for handling chatbot responses.
    """

    @staticmethod
    def respond(chatbot_history: list, message: str) -> tuple:
        """
        Args:
            chatbot_history (List): Chatbot history.
            message (str): User input.

        Returns:
            tuple: Tuple containing the response and the updated chatbot history.
        """

        events = graph.compile().stream(
            {"question": message}, config, stream_mode="values"
        )
        response = list(events)[-1]["agent_response"]

        chatbot_history.append((message, response))

        Memory.write_chat_history_to_file(
            gradio_chatbot=chatbot_history, folder_path="./db", thread_id="bb_1"
        )

        return "", chatbot_history


with gr.Blocks(theme="soft", title="Agent_BB7") as chat_ui:
    with gr.Tabs():
        with gr.TabItem("Agent_BB7"):
            with gr.Row() as row_one:
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    bubble_full_width=False,
                    height=500,
                    avatar_images=("helper/User-Avatar.png", "helper/bb7.png"),
                )
                chatbot.like(UISettings.feedback, None, None)

            with gr.Row():
                input_txt = gr.Textbox(
                    lines=3,
                    scale=8,
                    placeholder="Enter text and press enter",
                    container=False,
                    elem_id="input_box",
                )

            with gr.Row() as row_two:
                text_submit_btn = gr.Button(value="Submit text")
                clear_button = gr.ClearButton([input_txt, chatbot])

            txt_msg = input_txt.submit(
                fn=ChatBot.respond,
                inputs=[chatbot, input_txt],
                outputs=[input_txt, chatbot],
                queue=False,
            ).then(lambda: gr.Textbox(interactive=True), None, [input_txt], queue=False)

            txt_msg = text_submit_btn.click(
                fn=ChatBot.respond,
                inputs=[chatbot, input_txt],
                outputs=[input_txt, chatbot],
                queue=False,
            ).then(lambda: gr.Textbox(interactive=True), None, [input_txt], queue=False)


if __name__ == "__main__":
    chat_ui.launch(debug=True, share=True)
