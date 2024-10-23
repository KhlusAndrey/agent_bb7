from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool


class RAGTool:
    """
    A tool for retrieving relevant documents using a Retrieval-Augmented Generation (RAG) approach.
    
    This tool leverages vector embeddings to find semantically relevant documents in a vector database. 
    The user provides a query in natural language, and the tool converts this query into vector space using a specified 
    embedding model. The vector database (Chroma) is then queried for similar vectors to retrieve documents relevant 
    to the user's query.

    Main functionalities:
    - Convert the user's query into a vector representation using the provided embedding model.
    - Perform similarity search in the vector database to retrieve the most semantically relevant documents.
    - Return the top-k relevant documents based on the similarity scores.
    """

    def __init__(self, embedding_model: str, vectordb_dir: str, k: int, collection_name: str) -> None:
        """
        Initialize the RAGTool with the necessary components for document retrieval.

        Parameters:
        - embedding_model (str): The name of the embedding model used to convert queries into vector representations.
        - vectordb_dir (str): Directory where the vector database (Chroma) is stored.
        - k (int): The number of top relevant documents to retrieve from the database.
        - collection_name (str): The name of the collection in the vector database where the documents are stored.
        """

        self.embedding_model = embedding_model
        self.vectordb_dir = vectordb_dir
        self.k = k
        self.vectordb = Chroma(
            collection_name=collection_name,
            persist_directory=self.vectordb_dir,
            embedding_function=OpenAIEmbeddings(model=self.embedding_model)
        )
        
        print("Number of vectors in vectordb:", self.vectordb._collection.count(), "\n\n")


@tool
def retrieve_tool(query: str) -> str:
    """
    Retrieve the most relevant documents based on the user's query using the RAG (Retrieval-Augmented Generation) approach.
    
    Parameters:
    - query (str): The user's search query in natural language.
    
    Returns:
    - str: The content of the top-k retrieved documents, concatenated as a single string.
    """

    rag_tool = RAGTool(
        embedding_model="text-embedding-3-large", 
        vectordb_dir="./Chromadb", 
        k=5, 
        collection_name="comments"
    )
    
   
    docs = rag_tool.vectordb.similarity_search(query, k=rag_tool.k)
    
    return "\n\n".join([doc.page_content for doc in docs])