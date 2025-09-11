# agents/tools.py
from langchain.tools import tool
from contracts.analyzer import HybridContractAnalyzer
from pdf_utils.pdf_functions import extract_text_from_pdf
import httpx, json
import asyncio

analyzer = HybridContractAnalyzer()

@tool
def risk_analyze(text: str) -> dict:
    """Return detailed risk analysis of the given legal text."""
    return asyncio.run(analyzer.analyze_contract(text))

@tool
def search_cases(query: str, k: int = 5) -> list[dict]:
    """Search Indian case-law vector DB and return top-k cases."""
    # call your vector search here
    return vectordb.similarity_search(query, k=k)

@tool
def search_ipc(section_keywords: str) -> list[str]:
    """Return IPC sections relevant to the keywords."""
    # dummy — replace with real db
    return ipc_db.lookup(section_keywords)

@tool
def simplify_legalese(passage: str, grade: str = "10") -> str:
    """Translate legal jargon into plain language for grade-8 reader."""
    from langchain_community.llms import Ollama
    llm = Ollama(model="llama2")
    return llm.invoke(f"Explain in plain English (grade {grade}): {passage}")
