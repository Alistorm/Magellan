# Project Magellan 🧭

*An instrument of navigation for Mistral AI agents, built for the Mistral AI MCP Hackathon 2025.*

[![Hackathon](https://img.shields.io/badge/Mistral%20AI%20MCP%20Hackathon-2025-blue)](https://mistral.ai/)
[![Status](https://img.shields.io/badge/status-in%20development-yellow)](https://github.com/your-username/magellan)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---
## About The Project

Le Chat and other Mistral AI-powered assistants are powerful language models, but their capabilities are confined to the digital world of text. They cannot interact with the most significant information interface ever created: the World Wide Web.

Popular browser automation libraries like `browser-use` provide the tools for AI agents to control a web browser, but they lack native, first-class support for Mistral's powerful family of models.

**Magellan solves this.**

Magellan is a custom-built, high-performance **Model Context Protocol (MCP) server** that acts as the missing bridge. It introduces a brand new, purpose-built **Mistral LLM connector** for the `browser-use` library, empowering Mistral agents to navigate, understand, and interact with any website.

Our project's goal is to give Le Chat the "eyes" to see the web and the "hands" to use it, all orchestrated through the intelligence of Mistral AI.
## Core Features

*   **First-Class Mistral Integration:** A custom-built `ChatMistral` class that allows the `browser-use` agent to be powered by Mistral models (`mistral-large-latest`, etc.).
*   **Interactive Visual Browsing in Le Chat:** Magellan doesn't just automate; it returns visual screenshots and a list of interactive elements, allowing the user to guide the agent in a "show, don't just tell" experience.
*   **Custom MCP Server:** Built from the ground up in Python to be lightweight, extensible, and tailored specifically for our Mistral-powered agent.
*   **Pydantic V2 Schemas:** All agent components and data structures are defined with Pydantic for robust type-safety and clarity, ensuring our implementation is production-grade.

## Architecture: How It Works

Magellan orchestrates a seamless flow from user request to web interaction:

1.  **User Prompt:** A user asks Le Chat to perform a web-based task (e.g., "Find the latest news on Alpic Cloud resources").
2.  **MCP Request:** Le Chat dispatches the task to the Magellan MCP server.
3.  **Mistral-Powered Decision:** Magellan receives the task and uses our custom `ChatMistral` connector. It sends the current state and goal to the Mistral API to determine the next best action (e.g., `browser_navigate`, `browser_type`).
4.  **Browser Execution:** The `browser-use` library executes the action determined by the Mistral model in a headless browser session.
5.  **Visual Feedback Loop:** Magellan captures the new page state (screenshot and interactive DOM elements).
6.  **Return to Le Chat:** The server sends the visual context back to the user in Le Chat, showing them what the agent sees and what it can do next.

## Getting Started

To get a local copy up and running, follow these simple steps using [uv](https://github.com/astral-sh/uv), the high-performance Python package manager.

### Prerequisites

*   [uv](https://github.com/astral-sh/uv) installed
*   Python 3.10+
*   An API key from [Mistral AI](https://mistral.ai/)

### Installation

1.  Clone the repository:
    ```sh
    git clone https://github.com/Alistorm/Magellan.git
    cd Magellan
    ```
2.  Create a virtual environment and install dependencies with `uv`:
    ```sh    
    # Install packages from requirements.txt into the virtual environment
    uv sync
    ```
3.  Set up your environment variables by creating a `.env` file:
    ```.env
    MISTRAL_API_KEY="your-mistral-api-key-here"
    ```
4.  Run the Magellan MCP server using `uv`:
    ```sh
    # 'uv run' executes the command within the managed virtual environment
    uv run ./src/main.py
    ```

## Usage Example

Once the Magellan server is running, configure your MCP-compatible client (like Le Chat) to connect to it.

**Example configuration for an MCP client:**
```json
{
  "mcpServers": {
    "magellan": {
      "command": "uv",
      "args": ["run", "magellan_server.py"],
      "env": {
        "MISTRAL_API_KEY": "your-mistral-api-key-here"
      }
    }
  }
}
```

You can then ask Le Chat to perform tasks:
> "Hey Magellan, go to the Mistral AI Cookbook on GitHub and find an example of how to use function calling."
>  "Hey Magellan, help me play Akinator live"

## Built With

*   [Mistral AI](https://mistral.ai/)
*   [Python](https://www.python.org/)
*   [uv](https://github.com/astral-sh/uv)
*   [browser-use](https://browser-use.com/)
*   [Pydantic V2](https://docs.pydantic.dev/latest/)
*   *(Potentially)* [Qdrant](https://qdrant.tech/) for agent memory
*   *(Potentially)* [Alpic Cloud](https://alpic.cloud/) for hosting

---
**Project Magellan** is proudly developed by Wilfred/Mohamed Ali/Adrian and their AI teammates for the **Mistral AI MCP Hackathon 2025**.

## Retrofitting MCP connectors with the Akinator example (Wilfred part)

### What we learned
- MCP protocol: how capabilities/tools are exposed and invoked.
- Implementing an MCP server in Python with fastmcp, focusing on type annotations and JSON schemas.
- Deploying an MCP service to production with Alpik.
- Configuring Mistral Le Chat to connect to our MCP server.
- Integrating a third‑party API (Akinator).
- Delivering a concrete, end‑to‑end project.

### What we built
- A working MCP server in Python (fastmcp) deployed on Alpik and consumable from Mistral Le Chat.
- Demonstrations using the Akinator API to validate end‑to‑end capability exposure and dialogue.

### Next steps and ideas
- Automatically generate MCP connectors by “retrofitting” existing API specs:
  - Swagger/OpenAPI 3
  - GraphQL
- Enable conversations with real‑world devices for maintenance and Industry 4.0:
  - Automotive CAN bus
  - Other IIoT devices

### Differentiator to support
- AsyncAPI (event‑driven API spec) as a key differentiator.
  - Vision: allow users to issue a request in Le Chat and receive results asynchronously via notifications.
  - Implication: would require evolving the turn‑based MCP protocol to support asynchronous events.

### Relationship between our two tracks
-  Browser automation for retrofitting existing devices complements the MCP connector generation:
  - Browser automation bridges legacy or UI‑only systems.
  - MCP connectors provide a standardized interface to both modern APIs and retrofitted endpoints.
  - Together, they expand coverage from web‑only/legacy devices to event‑driven and device‑level integrations.
  Pattern illustrated by Akinator

The Akinator example showcases cooperation between an expert system and a generative AI.
Division of roles:
Expert system (Akinator): domain heuristics, state management, and deterministic questioning/answer evaluation.
Generative model (Le Chat/Mistral): natural‑language interaction, explanation, and strategy suggestions to improve user experience.


MCP acts as the orchestration layer: the LLM invokes Akinator’s capabilities, receives structured results, and turns them into conversational guidance.
Outcome: a seamless, human‑friendly dialogue powered by a reliable expert engine—demonstrating how classical AI and generative AI can complement each other end‑to‑end.
