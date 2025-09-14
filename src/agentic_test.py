import asyncio

from browser_use import Agent
from browser_use.mcp.server import main as mcp_main
from dotenv import load_dotenv

from mistral.chat import ChatMistral

load_dotenv()


async def main():
    llm = ChatMistral("mistral-medium-latest")
    task = "Trouve moi des billets d'avions pas cheres pour madagascar"
    agent = Agent(task=task, llm=llm)
    await agent.run()


if __name__ == "__main__":
    asyncio.run(mcp_main())
