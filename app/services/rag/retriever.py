# app/services/rag/retriever.py
from pinecone import Pinecone, ServerlessSpec
from core.config import settings
from services.embeddings.gemini import embed_text

pc = Pinecone(
    api_key=(settings.PINECONE_API_KEY)
)
index = pc.Index(settings.PINECONE_INDEX)


def fetch_context(query: str, top_k: int = 6):
    query_vec = embed_text([query])[0]
    res = index.query(vector=query_vec, top_k=top_k, include_metadata=True)
    chunks = [m["metadata"]["text"] for m in res["matches"]]
    return chunks
