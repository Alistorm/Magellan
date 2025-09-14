"""
fastmcp_server.py

A professional, full-featured, Mistral-native MCP Server.
Implements the complete browser-use toolset on a robust, lifespan-managed architecture.
Built for the Mistral AI MCP Hackathon 2025.
"""

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from browser_use import Agent, BrowserProfile, BrowserSession, Tools
from browser_use.config import get_default_profile, load_browser_use_config

# We need more imports for the advanced tools
from browser_use.filesystem.file_system import FileSystem
from dotenv import load_dotenv
from pydantic import Field, create_model

from mistral.chat import ChatMistral, MistralModel

load_dotenv()
# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

try:
    from mcp.server.fastmcp import FastMCP

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.error("MCP SDK not installed. Install with: pip install mcp")
    sys.exit(1)


class FastServerState:
    def __init__(self, default_model: MistralModel = "mistral-medium-latest", session_timeout_minutes: int = 10):
        self.config = load_browser_use_config()
        self.default_model: MistralModel = default_model
        self.agent: Agent | None = None
        self.browser_session: BrowserSession | None = None
        self.tools: Tools | None = None
        self.llm: ChatMistral | None = None
        self.file_system: FileSystem | None = None
        self.start_time = time.time()

        # Session management
        self.active_sessions: dict[str, dict[str, Any]] = {}  # session_id -> session info
        self.session_timeout_minutes = session_timeout_minutes
        self.cleanup_task: Any = None


state = FastServerState()


@asynccontextmanager
async def lifespan(app: FastMCP):
    """
    Manages the lifecycle of shared resources (browser, DB connections, etc.).
    This is the modern way to handle startup and shutdown logic in FastAPI/FastMCP.
    """
    session_start = time.time()
    logger.info("Session is starting up...")

    # --- Startup Logic ---
    # Initialize the browser session session on startup.
    await init_browser_session()

    # Start the background task for cleaning up idle sessions.
    if state.cleanup_task is None:
        state.cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info("Background session cleanup task started.")

    yield

    # --- Shutdown Logic ---
    logger.info("Session is shutting down...")
    logger.info(f"Session lasted: {time.time() - session_start:.2f} seconds.")


# --- MCP Server Initialization ---
mcp = FastMCP(
    "Mistral Browser Agent (A full-featured browsing agent powered by Browser Use and Mistral AI.)",
    lifespan=lifespan,
    stateless_http=True,
    debug=False,
    port=7000,
)


async def init_browser_session(allowed_domains: list[str] | None = None, **kwargs):
    """Initialize browser session using config"""
    if state.browser_session:
        return

    logger.debug("Initializing browser session...")

    # Get profile config
    profile_config = get_default_profile(state.config)

    # Merge profile config with defaults and overrides
    profile_data = {
        "downloads_path": str(Path.home() / "Downloads" / "browser-use-mcp"),
        "minimum_wait_page_load_time": 0.1,
        "wait_between_actions": 0.1,
        "keep_alive": True,
        "user_data_dir": "~/.config/browseruse/profiles/default",
        "device_scale_factor": 1.0,
        "disable_security": False,
        "headless": False,
        **profile_config,  # Config values override defaults
    }

    # Tool parameter overrides (highest priority)
    if allowed_domains is not None:
        profile_data["allowed_domains"] = allowed_domains

    # Merge any additional kwargs that are valid BrowserProfile fields
    for key, value in kwargs.items():
        profile_data[key] = value

    # Create browser profile
    profile = BrowserProfile(**profile_data)

    # Create browser session
    state.browser_session = BrowserSession(browser_profile=profile)
    await state.browser_session.start()

    # Track the session for management
    track_session(state.browser_session)

    # Create tools for direct actions
    state.tools = Tools()

    # Initialize LLM
    state.llm = ChatMistral(model=state.default_model, api_key=os.environ.get("MISTRAL_API_KEY"))

    # Initialize FileSystem for extraction actions
    file_system_path = profile_config.get("file_system_path", "~/.browser-use-mcp")
    state.file_system = FileSystem(base_dir=Path(file_system_path).expanduser())

    logger.debug("Browser session initialized")


# --- Tool Definitions ---


@mcp.tool(title="Navigate to URL")
async def navigate(
    url: str = Field(description="The full URL to navigate to."),
    new_tab: bool = Field(False, description="Whether to open the URL in a new tab."),
) -> str:
    """Navigate to a URL in the browser"""
    if not state.browser_session:
        return "Error: No browser session active"

    # Update session activity
    update_session_activity(state.browser_session.id)

    from browser_use.browser.events import NavigateToUrlEvent

    if new_tab:
        event = state.browser_session.event_bus.dispatch(NavigateToUrlEvent(url=url, new_tab=True))
        await event
        return f"Opened new tab with URL: {url}"
    else:
        event = state.browser_session.event_bus.dispatch(NavigateToUrlEvent(url=url))
        await event
        return f"Navigated to: {url}"


@mcp.tool(title="Click Element")
async def click(
    index: int = Field(description="The index of the link or element to click (from get_state)."),
    new_tab: bool = Field(False, description="If the element is a link, attempt to open it in a new tab."),
) -> str:
    """Click an element on the page by its index. Can open links in a new tab."""
    if not state.browser_session:
        return "Error: No browser session active"
    update_session_activity(state.browser_session.id)

    # Get the element
    element = await state.browser_session.get_dom_element_by_index(index)
    if not element:
        return f"Element with index {index} not found"

    if new_tab:
        # For links, extract href and open in new tab
        href = element.attributes.get("href")
        if href:
            # Convert relative href to absolute URL
            page_state = await state.browser_session.get_browser_state_summary()
            current_url = page_state.url
            if href.startswith("/"):
                # Relative URL - construct full URL
                from urllib.parse import urlparse

                parsed = urlparse(current_url)
                full_url = f"{parsed.scheme}://{parsed.netloc}{href}"
            else:
                full_url = href

            # Open link in new tab
            from browser_use.browser.events import NavigateToUrlEvent

            event = state.browser_session.event_bus.dispatch(NavigateToUrlEvent(url=full_url, new_tab=True))
            await event
            return f"Clicked element {index} and opened in new tab {full_url[:20]}..."
        else:
            # For non-link elements, just do a normal click
            # Opening in new tab without href is not reliably supported
            from browser_use.browser.events import ClickElementEvent

            event = state.browser_session.event_bus.dispatch(ClickElementEvent(node=element))
            await event
            return f"Clicked element {index} (new tab not supported for non-link elements)"
    else:
        # Normal click
        from browser_use.browser.events import ClickElementEvent

        event = state.browser_session.event_bus.dispatch(ClickElementEvent(node=element))
        await event
        return f"Clicked element {index}"


@mcp.tool(title="Type Text")
async def type_text(
    index: int = Field(description="The index of the link or element to click (from get_state)."),
    text: str = Field(description="The text to type into the element."),
) -> str:
    """Type text into an element."""
    if not state.browser_session:
        return "Error: No browser session active"

    element = await state.browser_session.get_dom_element_by_index(index)
    if not element:
        return f"Element with index {index} not found"

    from browser_use.browser.events import TypeTextEvent

    event = state.browser_session.event_bus.dispatch(TypeTextEvent(node=element, text=text))
    await event
    return f"Typed '{text}' into element {index}"


@mcp.tool(title="Get Browser State")
async def get_browser_state(
    include_screenshot: bool = Field(
        False, description="Whether to include a base64 encoded screenshot of the current page."
    ),
) -> dict:
    """Get current browser state."""
    if not state.browser_session:
        # Returning a dict to be consistent with the success case return type
        return {"error": "No browser session active"}

    page_state = await state.browser_session.get_browser_state_summary(cache_clickable_elements_hashes=False)

    result = {
        "url": page_state.url,
        "title": page_state.title,
        "tabs": [{"url": tab.url, "title": tab.title} for tab in page_state.tabs],
        "interactive_elements": [],
    }

    # Add interactive elements with their indices
    for index, element in page_state.dom_state.selector_map.items():
        elem_info = {
            "index": index,
            "tag": element.tag_name,
            "text": element.get_all_children_text(max_depth=2)[:100],
        }
        if placeholder := element.attributes.get("placeholder"):
            elem_info["placeholder"] = placeholder
        if href := element.attributes.get("href"):
            elem_info["href"] = href
        result["interactive_elements"].append(elem_info)

    if include_screenshot and page_state.screenshot:
        result["screenshot"] = page_state.screenshot

    return result


@mcp.tool(title="Extract Structured Content")
async def extract_content(
    query: str = Field(description="A detailed query of what information to extract from the page."),
    extract_links: bool = Field(False, description="Whether to include source links in the extraction."),
) -> dict:
    """Uses an LLM to extract structured content from the current page based on a query."""
    if not state.llm:
        return {"error": "LLM not initialized"}
    if not state.file_system:
        return {"error": "FileSystem not initialized"}
    if not state.browser_session:
        return {"error": "No browser session active"}
    if not state.tools:
        # This should be initialized in lifespan, but we check just in case.
        return {"error": "Tools service not initialized"}

    # Dynamically create the Pydantic model that the 'act' method expects.
    ExtractAction = create_model(
        "ExtractAction",
        extract_structured_data=(dict, Field(..., query=query, extract_links=extract_links)),
    )

    # Call the tools service to perform the extraction.
    action_result = await state.tools.act(
        action=ExtractAction(),
        browser_session=state.browser_session,
        page_extraction_llm=state.llm,
        file_system=state.file_system,
    )

    return action_result.extracted_content or {"message": "No content was extracted for the given query."}


@mcp.tool(
    title="Scroll Page",
)
async def scroll(
    direction: str = Field("down", description="The direction to scroll ('up', 'down', 'left', 'right')."),
) -> str:
    """Scroll the current page up or down."""
    if not state.browser_session:
        return "Error: No browser session active"

    from browser_use.browser.events import ScrollEvent

    # Scroll by a standard amount (500 pixels)
    event = state.browser_session.event_bus.dispatch(
        ScrollEvent(
            direction=direction,  # type: ignore
            amount=500,
        )
    )
    await event
    return f"Scrolled {direction}"


@mcp.tool(title="Go Back")
async def go_back() -> str:
    """Go back in browser history."""
    if not state.browser_session:
        return "Error: No browser session active"

    from browser_use.browser.events import GoBackEvent

    event = state.browser_session.event_bus.dispatch(GoBackEvent())
    await event
    return "Navigated back"


@mcp.tool(title="Close Browser")
async def close_browser() -> str:
    """Close the browser session."""
    if state.browser_session:
        from browser_use.browser.events import BrowserStopEvent

        event = state.browser_session.event_bus.dispatch(BrowserStopEvent())
        await event
        state.browser_session = None
        state.tools = None
        return "Browser closed"
    return "No browser session to close"


@mcp.tool(title="List Browser Tabs")
async def list_tabs() -> dict:
    """List all open tabs."""
    if not state.browser_session:
        return "Error: No browser session active"

    tabs_info = await state.browser_session.get_tabs()
    tabs = []
    for i, tab in enumerate(tabs_info):
        tabs.append({"tab_id": tab.target_id[-4:], "url": tab.url, "title": tab.title or ""})
    return tabs


@mcp.tool(title="Switch Tab")
async def switch_tab(tab_id: str = Field(description="The 4-character ID of the tab to switch to.")) -> str:
    """Switch to a different tab."""
    if not state.browser_session:
        return "Error: No browser session active"

    from browser_use.browser.events import SwitchTabEvent

    target_id = await state.browser_session.get_target_id_from_tab_id(tab_id)
    event = state.browser_session.event_bus.dispatch(SwitchTabEvent(target_id=target_id))
    await event
    page_state = await state.browser_session.get_browser_state_summary()
    return f"Switched to tab {tab_id}: {page_state.url}"


@mcp.tool(title="Close Tab")
async def close_tab(tab_id: str = Field(description="The 4-character ID of the tab to close.")) -> str:
    """Close a specific tab."""
    if not state.browser_session:
        return "Error: No browser session active"

    from browser_use.browser.events import CloseTabEvent

    target_id = await state.browser_session.get_target_id_from_tab_id(tab_id)
    event = state.browser_session.event_bus.dispatch(CloseTabEvent(target_id=target_id))
    await event
    current_url = await state.browser_session.get_current_page_url()
    return f"Closed tab # {tab_id}, now on {current_url}"


def track_session(session: BrowserSession) -> None:
    """Track a browser session for management."""
    state.active_sessions[session.id] = {
        "session": session,
        "created_at": time.time(),
        "last_activity": time.time(),
        "url": getattr(session, "current_url", None),
    }


def update_session_activity(session_id: str) -> None:
    """Update the last activity time for a session."""
    if session_id in state.active_sessions:
        state.active_sessions[session_id]["last_activity"] = time.time()


@mcp.tool(title="Get Session Status")
async def get_session_status() -> dict:
    """List all active browser sessions."""
    if not state.active_sessions:
        return {"status": "Inactive", "session_id": None, "current_url": None}

    sessions_info = []
    for session_id, session_data in state.active_sessions.items():
        session = session_data["session"]
        created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(session_data["created_at"]))
        last_activity = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(session_data["last_activity"]))

        # Check if session is still active
        is_active = hasattr(session, "cdp_client") and session.cdp_client is not None

        sessions_info.append(
            {
                "session_id": session_id,
                "created_at": created_at,
                "last_activity": last_activity,
                "active": is_active,
                "current_url": session_data.get("url", "Unknown"),
                "age_minutes": (time.time() - session_data["created_at"]) / 60,
            }
        )

    return sessions_info


@mcp.tool(title="Close a Session")
async def close_session(session_id: str = Field(description="The session ID of the browser session to close.")) -> str:
    """Close a specific browser session."""
    if session_id not in state.active_sessions:
        return f"Session {session_id} not found"

    session_data = state.active_sessions[session_id]
    session = session_data["session"]

    try:
        # Close the session
        if hasattr(session, "kill"):
            await session.kill()
        elif hasattr(session, "close"):
            await session.close()

        # Remove from tracking
        del state.active_sessions[session_id]

        # If this was the current session, clear it
        if state.browser_session and state.browser_session.id == session_id:
            state.browser_session = None
            state.tools = None

        return f"Successfully closed session {session_id}"
    except Exception as e:
        return f"Error closing session {session_id}: {str(e)}"


@mcp.tool(title="Close All Sessions")
async def close_all_sessions() -> str:
    """Close all active browser sessions."""
    if not state.active_sessions:
        return "No active sessions to close"

    closed_count = 0
    errors = []

    for session_id in list(state.active_sessions.keys()):
        try:
            result = await close_session(session_id)
            if "Successfully closed" in result:
                closed_count += 1
            else:
                errors.append(f"{session_id}: {result}")
        except Exception as e:
            errors.append(f"{session_id}: {str(e)}")

    # Clear current session references
    state.browser_session = None
    state.tools = None

    result = f"Closed {closed_count} sessions"
    if errors:
        result += f". Errors: {'; '.join(errors)}"

    return result


async def cleanup_expired_sessions() -> None:
    """Background task to clean up expired sessions."""
    current_time = time.time()
    timeout_seconds = state.session_timeout_minutes * 60

    expired_sessions = []
    for session_id, session_data in state.active_sessions.items():
        last_activity = session_data["last_activity"]
        if current_time - last_activity > timeout_seconds:
            expired_sessions.append(session_id)

    for session_id in expired_sessions:
        try:
            await close_session(session_id)
            logger.info(f"Auto-closed expired session {session_id}")
        except Exception as e:
            logger.error(f"Error auto-closing session {session_id}: {e}")


async def cleanup_loop():
    """Start the background cleanup task."""
    while True:
        try:
            await cleanup_expired_sessions()
            # Check every 2 minutes
            await asyncio.sleep(120)
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(120)


@mcp.tool(title="Run Mistral Agent")
async def run_mistral_agent(
    task: str = Field(
        description="""A clear, detailed description of the high-level goal for the autonomous agent to achieve.
        For example: '
        1. Go to reddit https://www.reddit.com/search/?q=browser+agent&type=communities 
        2. Click directly on the first 5 communities to open each in new tabs
        3. Find out what the latest post is about, and switch directly to the next tab
        4. Return the latest post summary for each page'."""
    ),
    max_steps: int = Field(20, description="The maximum number of steps the agent can take before stopping."),
) -> str:
    """
    Hands control of the browser session to an autonomous agent to perform a complex task.
    """
    if not state.browser_session:
        return "Error: No browser session active"

    logger.info(f"Agent is taking control of the browser session (ID: {state.browser_session.id})")

    # Create the agent, ensuring vision is enabled for best performance.
    agent = Agent(task=task, llm=state.llm, browser_session=state.browser_session)

    # Run the agent's task, using the new max_steps parameter.
    history = await agent.run(max_steps=int(max_steps * 1.3))
    final_result = history.final_result() or "Task completed without a specific final output."

    await agent.close()
    logger.info("Agent has finished its task. The browser session remains active.")

    return final_result


def fast_server_run():
    if not MCP_AVAILABLE:
        print("MCP SDK is required. Install with: pip install mcp", file=sys.stderr)
        sys.exit(1)
    try:
        mcp.run(transport="streamable-http")
    finally:
        duration = time.time() - state.start_time
        print(f"Server lasted: {duration}")
