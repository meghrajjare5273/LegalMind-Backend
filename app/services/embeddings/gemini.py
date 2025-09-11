    # app/services/embeddings/gemini.py
import google.generativeai as genai
from core.config import settings

embedding_client = genai
embedding_client.configure(api_key=settings.GEMINI_API_KEY)

def embed_text(texts: list[str]) -> list[list[float]]:
    model = embedding_client.get_model(settings.EMB_MODEL)
    return model.embed_content(texts)["embedding"]
