from browser_use import Agent, ChatGroq
from dotenv import load_dotenv
import asyncio

from mistral_connector import ChatMistral

load_dotenv()

async def main():
    llm = ChatMistral("mistral-medium-latest")
    task = "Find the Mistral AI MCP Hackathon"
    agent = Agent(task=task, llm=llm)
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())