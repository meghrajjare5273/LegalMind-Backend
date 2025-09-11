# app/services/rag/chat.py
from typing import List
from core.config import settings
import google.generativeai as genai
from .retriever import fetch_context

genai.configure(api_key=settings.GEMINI_API_KEY)
llm = genai.GenerativeModel(settings.LLM_MODEL)

def answer_question(question: str) -> tuple[str, List[str]]:
    context_chunks = fetch_context(question)
    prompt = (
        "Answer the question strictly using the CONTEXT. "
        "If context is insufficient say you don't know.\n\n"
        f"CONTEXT:\n{''.join(context_chunks)}\n\n"
        f"QUESTION:\n{question}"
    )
    completion = llm.generate_content(prompt, safety_settings={"temperature": 0.2})
    return completion.text, context_chunks
