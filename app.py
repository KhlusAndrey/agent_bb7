import gradio as gr
from chat_bot.chat_bot import ChatBot
from helper.UI_like import UISettings
import random

def get_random_chatbot_instance():
    user_id = f"user_{random.randint(0, 1000)}"  # Could be `session_id` or UUID
    return ChatBot(user_id)

chatbot_instance = get_random_chatbot_instance()

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
                fn=chatbot_instance.respond,  
                inputs=[chatbot, input_txt],
                outputs=[input_txt, chatbot],
                queue=False,
            ).then(lambda: gr.Textbox(interactive=True), None, [input_txt], queue=False)

            txt_msg = text_submit_btn.click(
                fn=chatbot_instance.respond,
                inputs=[chatbot, input_txt],
                outputs=[input_txt, chatbot],
                queue=False,
            ).then(lambda: gr.Textbox(interactive=True), None, [input_txt], queue=False)

if __name__ == "__main__":
    chat_ui.launch(debug=True, share=True)