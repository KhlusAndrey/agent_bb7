import os
import pandas as pd
from typing import List
from datetime import datetime, date


class Memory:
    """
    A class for handling the storage of chatbot conversation history by writing chat logs to a CSV file.

    Methods:
        write_chat_history_to_file(gradio_chatbot: List, thread_id: str, folder_path: str) -> None:
            Writes the most recent chatbot interaction (user query and bot response) to a CSV file.
            The chat log is saved with the current date as the filename, and the interaction is
            timestamped.
    """

    @staticmethod
    def write_chat_history_to_file(
        gradio_chatbot: List, thread_id: str, folder_path: str
    ) -> None:
        """
        Writes the most recent chatbot interaction (user query and response) to a CSV file. The log includes
        the thread ID and timestamp of the interaction. The file for each day is saved with the current date as the filename.

        Args:
            gradio_chatbot (List): A list containing tuples of user queries and chatbot responses.
                                   The most recent interaction is appended to the log.
            thread_id (str): The unique identifier for the chat session (or thread).
            folder_path (str): The directory path where the chat log CSV files should be stored.

        Returns:
            None

        File Structure:
            - The chat log for each day is saved as a separate CSV file in the specified folder.
            - The CSV file is named using the current date in 'YYYY-MM-DD' format.
            - Each row in the CSV file contains the following columns: 'thread_id', 'timestamp', 'user_query', 'response'.
        """
        tmp_list = list(gradio_chatbot[-1])

        today_str = date.today().strftime("%Y-%m-%d")
        tmp_list.insert(0, thread_id)

        current_time_str = datetime.now().strftime("%H:%M:%S")
        tmp_list.insert(1, current_time_str)

        file_path = os.path.join(folder_path, f"{today_str}.csv")

        new_df = pd.DataFrame(
            [tmp_list], columns=["thread_id", "timestamp", "user_query", "response"]
        )

        if os.path.exists(file_path):
            new_df.to_csv(file_path, mode="a", header=False, index=False)
        else:
            new_df.to_csv(file_path, mode="w", header=True, index=False)
