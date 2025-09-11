# agents/orchestrator.py
from .base_agent import BaseAgent
from langchain_core.prompts import PromptTemplate
# from .legal_demystifier import LegalDocumentDemystifierAgent
# from .precedent_research import PrecedentResearchAgent
# from .ipc_crossref import IPCCrossRefAgent
# from .risk_analysis_agent import RiskAnalysisAgent
# from .redline_negotiation import RedlineNegotiationAgent

SPECIALISTS = {
    # "demystify": LegalDocumentDemystifierAgent(),
    # "precedent": PrecedentResearchAgent(),
    # "ipc": IPCCrossRefAgent(),
    # "risk": RiskAnalysisAgent(),
    # "redline": RedlineNegotiationAgent()
}

template = """
    Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""


prompt = PromptTemplate.from_template(template)

SYSTEM_PROMPT = """
You are Orchestrator-GPT. Decide which specialists to call
and merge their JSON replies into a final answer. 
Always return JSON with keys:
 analyses, precedents, ipc_refs, redlines, demystified, overall_summary
User query: {query}
"""

class OrchestratorAgent(BaseAgent):
    def __init__(self):
        super().__init__("orchestrator", [], template)

    async def run(self, query: str, pdf_text: str):
        # 1. Naïve plan (enhance with LLM later)
        tasks = ["risk", "demystify"]
        if "IPC" in query or "criminal" in query:
            tasks.append("ipc")
        if "precedent" in query or "case" in query:
            tasks.append("precedent")
        if "redline" in query or "negotiate" in query:
            tasks.append("redline")

        results = {}
        for t in tasks:
            results[t] = await SPECIALISTS[t].run(query=query, pdf_text=pdf_text)

        # 2. Compose overall summary
        results["overall_summary"] = (
            f"Completed {', '.join(tasks)} tasks. "
            "See keys for details."
        )
        # Flatten for API
        return {
            "analyses": results.get("risk", {}).get("analyses", []),
            "summary": results.get("risk", {}).get("summary", {}),
            "sections": results.get("risk", {}).get("sections", []),
            "recommendations": results.get("risk", {}).get("recommendations", []),
            "precedents": results.get("precedent", {}).get("cases", []),
            "ipc_refs": results.get("ipc", {}).get("sections", []),
            "redlines": results.get("redline", {}).get("edits", []),
            "demystified": results.get("demystify", {}).get("plain_explanation", ""),
            "overall_summary": results["overall_summary"],
            "complexity_score": results.get("risk", {}).get("complexity_score", 0.3),
            "power_balance": results.get("risk", {}).get("power_balance", 0.5)
        }
