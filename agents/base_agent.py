# agents/base_agent.py
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

class BaseAgent:
    def __init__(self, name: str, tools: list, system_prompt: str):
        llm = OllamaLLM(model="llama2")
        prompt = PromptTemplate.from_template(system_prompt)
        agent = create_react_agent(llm, tools, prompt)
        self.exec = AgentExecutor(agent=agent, tools=tools, verbose=False)
        self.name = name

    async def run(self, **kwargs):
        return await self.exec.arun(kwargs)
