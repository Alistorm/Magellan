from browser_use import Agent
from dotenv import load_dotenv
import asyncio

from mistral_connector import ChatMistral

load_dotenv()

async def main():
    llm = ChatMistral(model="mistral-medium-latest")
    task = "Find Mistral AI and Cerebral Valley MCP Hackathon and give me all informations about it"
    agent = Agent(task=task, llm=llm)
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())