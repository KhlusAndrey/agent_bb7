import os
import dotenv
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from FlagEmbedding import BGEM3FlagModel


_ = dotenv.load_dotenv(dotenv.find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_image_embedding_sentence_transformers(
    image: Image.Image, image_id: str
) -> tuple[str, torch.Tensor]:
    """Get image embedding using sentence transformer"""
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    image_embedding = embed_model.encode([image], convert_to_tensor=True)
    return (image_id, image_embedding)


def get_image_embedding_blip(
    image: Image.Image, image_id: str
) -> tuple[str, torch.Tensor]:
    """Get image embedding using blip"""
    embed_model = SentenceTransformer("blip-image-captioning-base")
    image_embedding = embed_model.encode([image], convert_to_tensor=True)
    return (image_id, image_embedding)


def get_text_embedding_langchain_openai(
    text: str, model: str = "text-embedding-3-large"
) -> list[float]:
    """Get text embedding using langchain openai"""
    embedding_openai = OpenAIEmbeddings(
        model="text-embedding-3-large", deployment=model
    )
    embedding = embedding_openai.embed_query(text)
    return embedding


def get_text_embedding_openai(
    text: str, model: str = "text-embedding-3-large"
) -> list[float]:
    """Get text embedding using openai (small = 1536, large = 3072 dimension model)"""
    client = OpenAI()
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def get_text_embedding_bge_m3(text: str) -> list[float]:
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    # Setting use_fp16 to True speeds up computation with a slight performance degradation
    embeddings = model.encode(text, batch_size=12, max_length=8192)["dense_vecs"]
    # If you don't need such a long length, you can set a smaller value max_length to speed up the encoding process.
    return embeddings


if __name__ == "__main__":
    # emb = get_text_embedding_langchain_openai("Get text embedding test")
    # emb = get_text_embedding_openai("Get text embedding test")
    emb = get_text_embedding_bge_m3("Get text embedding test")
    print(len(emb))
