"""Setharkk Web Interface -- FastAPI + WebSocket.

Architecture :
- LocalModel est partage (stateless HTTP client)
- Memory, Agent, ToolRegistry sont crees PAR WebSocket (isolation multi-session)
- Chaque client a son propre _session_id et sa propre boucle tool-calling
"""

from __future__ import annotations

import json
import yaml
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

from agent.models.local_model import LocalModel
from agent.core.memory import Memory
from agent.core.agent import Agent
from agent.core.planner import AgentStep
from agent.tools.base import ToolRegistry
from agent.tools.search import SearchTool
from agent.tools.file_system import FileSystemTool
from agent.tools.code_executor import CodeExecutorTool
from agent.tools.browser import BrowserTool
from agent.tools.memory_tools import RememberTool, RecallTool, ForgetTool, CoreMemoryUpdateTool, SearchHistoryTool, GraphRecallTool
from agent.tools.research_tool import ResearchTool
from agent.tools.code_review_tool import CodeReviewTool


WEB_DIR = Path(__file__).parent
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "agent.yaml"

# Shared state — model et tools sont thread-safe (stateless HTTP + outil)
_model: LocalModel | None = None
_tools: ToolRegistry | None = None
_cfg: dict | None = None


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _build_tools(memory: Memory, model: LocalModel) -> ToolRegistry:
    """Cree un ToolRegistry avec les tools lies a cette session."""
    tools = ToolRegistry()
    tools.register(SearchTool(_cfg))
    tools.register(FileSystemTool(_cfg))
    tools.register(CodeExecutorTool(_cfg))
    tools.register(BrowserTool())
    tools.register(RememberTool(memory))
    tools.register(RecallTool(memory))
    tools.register(ForgetTool(memory))
    tools.register(CoreMemoryUpdateTool(memory))
    tools.register(SearchHistoryTool(memory, _cfg))
    tools.register(GraphRecallTool(memory, _cfg))
    # Compound tools (ont besoin du model + tools registry)
    tools.register(ResearchTool(tools, model, _cfg))
    tools.register(CodeReviewTool(tools, model, _cfg))
    return tools


async def _create_session_agent() -> tuple[Agent, Memory]:
    """Cree un Agent + Memory isoles pour une session WebSocket."""
    memory = Memory(_cfg, model=_model)
    await memory.connect()
    await memory.init_table()

    tools = _build_tools(memory, _model)

    agent = Agent(
        model=_model,
        tools=tools,
        memory=memory,
        max_iterations=_cfg["agent"]["max_iterations"],
        tool_timeout=_cfg["agent"]["tool_timeout"],
        context_config=_cfg.get("context_manager", {}),
        config=_cfg,
    )

    return agent, memory


async def _init_shared():
    """Initialise les composants partages au demarrage."""
    global _model, _cfg
    _cfg = _load_config()
    _model = LocalModel()
    if not await _model.health_check():
        raise RuntimeError("llama-server non accessible")


async def _shutdown_shared():
    if _model:
        await _model.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _init_shared()
    yield
    await _shutdown_shared()


app = FastAPI(title="Setharkk", lifespan=lifespan)


# ── HTML ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/logo.png")
async def logo():
    return FileResponse(WEB_DIR / "logo.png", media_type="image/png")


# ── REST endpoints ───────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    ok = _model is not None and await _model.health_check()
    return {"status": "ok" if ok else "down"}


@app.get("/api/config")
async def get_config():
    return {
        "agent_name": _cfg["agent"]["name"],
        "max_iterations": _cfg["agent"]["max_iterations"],
        "tool_timeout": _cfg["agent"]["tool_timeout"],
        "tools": _cfg.get("tools", {}).get("enabled", []),
    }


@app.post("/api/file-read")
async def file_read(upload: UploadFile = File(...)):
    """Lit un fichier uploade et retourne son contenu texte."""
    try:
        content = await upload.read()
        text = content.decode("utf-8", errors="replace")
        return {"filename": upload.filename, "content": text, "size": len(text)}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


# ── WebSocket ────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    # Chaque WebSocket a son propre Agent + Memory (isolation)
    agent, memory = await _create_session_agent()

    try:
        # Envoyer le message de bienvenue
        await ws.send_json({
            "type": "welcome",
            "content": _cfg["agent"]["name"],
            "tools": [s["name"] for s in agent.tools.list_schemas()],
        })

        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            user_input = msg.get("message", "").strip()

            if not user_input:
                continue

            # ── Slash commands ──

            if user_input == "/clear":
                await memory.new_session()
                agent._session_initialized = False
                await ws.send_json({"type": "system", "content": "Nouvelle session (rien supprime en base)."})
                continue

            if user_input == "/tools":
                schemas = agent.tools.list_schemas()
                tools_data = [{"name": s["name"], "description": s["description"]} for s in schemas]
                await ws.send_json({"type": "tools_list", "tools": tools_data})
                continue

            if user_input == "/memory":
                msgs = await memory.get_messages()
                await ws.send_json({
                    "type": "memory_info",
                    "count": len(msgs),
                    "recent": [{"role": m["role"], "content": m["content"][:120]} for m in msgs[-5:]],
                })
                continue

            if user_input == "/status":
                core = await memory.get_core_prompt()
                tools_list = [s["name"] for s in agent.tools.list_schemas()]
                await ws.send_json({
                    "type": "status_info",
                    "core_memory": core,
                    "tools": tools_list,
                    "session_id": memory._session_id,
                })
                continue

            # ── File prompt (via message) ──
            if user_input.startswith("/file "):
                file_content = msg.get("file_content", "")
                if file_content:
                    user_input = file_content
                    await ws.send_json({"type": "system", "content": f"Prompt charge ({len(user_input)} chars)"})
                else:
                    await ws.send_json({"type": "error", "content": "Aucun contenu de fichier recu. Utilise le bouton upload."})
                    continue

            # ── Run agent ──
            async for step in agent.run(user_input):
                payload = {"type": step.kind, "content": step.content}
                if step.tool_name:
                    payload["tool_name"] = step.tool_name
                if step.tool_args:
                    payload["tool_args"] = step.tool_args
                await ws.send_json(payload)

            await ws.send_json({"type": "done"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "content": str(e)})
        except Exception:
            pass
    finally:
        # Cleanup de la session
        await memory.close()
