from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_openai import ChatOpenAI


class SQLAgentTool:
    """
    A specialized tool for interacting with an SQL database using a language model (LLM) to generate, execute, and interpret SQL queries.

    This tool allows users to ask questions related to the data stored in an SQL database. The language model converts 
    these natural language questions into valid SQL queries, executes them against the specified database, 
    and processes the query results to provide a meaningful answer to the user.
    
    Responsibilities:
    - Interpret the user’s natural language query and generate an appropriate SQL query.
    - Execute the generated SQL query on the given database.
    - Interpret the SQL results and return a concise and accurate response to the user’s question.
    """

    def __init__(self, llm: str, sqldb_directory: str, llm_temperature: float) -> None:
        # Initialize the language model and database connection
        self.sql_agent_llm = ChatOpenAI(
            model=llm, temperature=llm_temperature)
        
        # System role for the agent, guiding the model in converting SQL results into meaningful answers.
        self.system_role = """You are an expert SQL assistant. Given a user's question, the corresponding SQL query, and the SQL result, 
            your task is to provide a clear, concise, and correct answer to the user's question based on the SQL result.

            Format:
            - Question: The user's original question in natural language.
            - SQL Query: The SQL query generated from the question.
            - SQL Result: The result of executing the query on the database.
            
            You should process the SQL result and provide a final answer to the user in clear and understandable terms. If the query result is empty, explain why no data was returned.

            Example:
            Question: {question}
            SQL Query: {query}
            SQL Result: {result}
            Answer: 
            """

        self.db = SQLDatabase.from_uri(f"sqlite:///{sqldb_directory}")
        
        print(self.db.get_usable_table_names())

        execute_query = QuerySQLDataBaseTool(db=self.db)
        write_query = create_sql_query_chain(self.sql_agent_llm, self.db)

        answer_prompt = PromptTemplate.from_template(self.system_role)

        answer = answer_prompt | self.sql_agent_llm | StrOutputParser()
        self.chain = (
            RunnablePassthrough.assign(query=write_query).assign(
                result=itemgetter("query") | execute_query
            )
            | answer
        )


@tool
def query_sqldb(query: str) -> str:
    """
    A tool for querying the SQL database and retrieving information. The input should be a natural language search query, 
    which will be converted into an SQL query by the language model, executed, and the results processed into a user-friendly answer.
    """
    agent = SQLAgentTool(
        llm='gpt-4o-mini',
        sqldb_directory="../db/brandbastion.db",
        llm_temperature=0
    )
    
    response = agent.chain.invoke({"question": query})
    return response