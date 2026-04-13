import asyncio
import yaml
from pathlib import Path

from agent.models.local_model import LocalModel
from agent.core.memory import Memory
from agent.core.agent import Agent
from agent.tools.base import ToolRegistry
from agent.tools.search import SearchTool
from agent.tools.file_system import FileSystemTool
from agent.tools.code_executor import CodeExecutorTool
from agent.tools.browser import BrowserTool
from agent.tools.memory_tools import RememberTool, RecallTool, ForgetTool, CoreMemoryUpdateTool, SearchHistoryTool, GraphRecallTool
from agent.tools.research_tool import ResearchTool
from agent.tools.code_review_tool import CodeReviewTool
from agent.ui.cli import chat_loop


def load_config() -> dict:
    cfg_path = Path(__file__).parent / "configs" / "agent.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


async def main():
    cfg = load_config()

    # Model
    model = LocalModel()

    # Check model health
    if not await model.health_check():
        print("[!] llama-server non accessible sur", model.base_url)
        print("    Lance d'abord : scripts/start_server.bat (avec --jinja)")
        return

    # Memory
    memory = Memory(cfg, model=model)
    await memory.connect()
    await memory.init_table()

    # Tools
    tools = ToolRegistry()
    tools.register(SearchTool(cfg))
    tools.register(FileSystemTool(cfg))
    tools.register(CodeExecutorTool(cfg))
    tools.register(BrowserTool())
    tools.register(RememberTool(memory))
    tools.register(RecallTool(memory))
    tools.register(ForgetTool(memory))
    tools.register(CoreMemoryUpdateTool(memory))
    tools.register(SearchHistoryTool(memory, cfg))
    tools.register(GraphRecallTool(memory, cfg))

    # Compound tools (ont besoin du model + tools registry)
    tools.register(ResearchTool(tools, model, cfg))
    tools.register(CodeReviewTool(tools, model, cfg))

    # Agent
    agent = Agent(
        model=model,
        tools=tools,
        memory=memory,
        max_iterations=cfg["agent"]["max_iterations"],
        tool_timeout=cfg["agent"]["tool_timeout"],
        context_config=cfg.get("context_manager", {}),
        config=cfg,
    )

    try:
        await chat_loop(agent)
    finally:
        await model.close()
        await memory.close()
        browser_tool = tools.get("browser")
        if browser_tool and hasattr(browser_tool, "close"):
            await browser_tool.close()


if __name__ == "__main__":
    asyncio.run(main())
