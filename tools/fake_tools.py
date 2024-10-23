from langchain_core.tools import tool
from langchain.schema import Document
from random import choices, randint


with open("tools/comments_data.txt", "r") as f:
    comments_data = f.read().splitlines()


@tool
def get_fake_sql_query_result(query: str) -> str:
    """
    A tool for querying the SQL database and retrieving information from a table comments in the client database. The input should be a search query, 
    which will be converted into an SQL query by the language model, executed, and the results processed into a user-friendly answer.
    Database table: Comments 
    Table columns: id, comment_text, sentiment, emotion, language, categories, tags, timestamp, user_id

    Given the following user question, corresponding SQL query, and SQL result, answer the user question.\n
            Question: {question}\n
            SQL Query: {query}\n
            SQL Result: {result}\n
    """
    f"""Return a list of fake SQL {query} results."""

    return choices(comments_data, k=randint(3, 25))


@tool
def get_fake_relevant_documents_chromadb(query: str) -> str:
    """
    Retrieve the most relevant documents based on the user's query using the RAG (Retrieval-Augmented Generation) approach.
    
    Parameters:
    - query (str): The user's search query in natural language.
    
    Returns:
    - list[Document]: The content of the top-k retrieved documents, concatenated as a single string.
    """
    docs = [
            Document(
                page_content="The chart shows a 6% increase in sales in the second quarter, a 12% decrease in negative reviews, and an 8% increase in spam message detection.",
                metadata={"source": "chart1.pdf"},
            ),
            Document(
                page_content="The graph contains information for September 2024, the indicators on the diagram are positive. Range of values from $15k to $35k",
                metadata={"source": "chart_2.pdf"},
            )]
    return docs

