from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    name: str
    description: str
    parameters: dict  # JSON Schema
    last_structured: Any = None  # Donnees structurees du dernier appel

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        ...

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def to_openai_schema(self) -> dict:
        """Format OpenAI pour le param 'tools' de chat_with_tools."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def openai_schemas(self) -> list[dict]:
        """Schemas au format OpenAI pour chat_with_tools."""
        return [t.to_openai_schema() for t in self._tools.values()]

    def get_structured(self, name: str) -> Any:
        """Recupere les donnees structurees du dernier appel d'un tool."""
        tool = self._tools.get(name)
        if tool is None:
            return None
        return tool.last_structured

    async def execute(self, name: str, exec_timeout: float = 30, **kwargs: Any) -> str:
        """Execute un tool par nom. exec_timeout = timeout asyncio (pas passe au tool)."""
        tool = self._tools.get(name)
        if tool is None:
            return f"[ERREUR] Tool '{name}' introuvable. Disponibles : {self.names()}"
        try:
            return await asyncio.wait_for(tool.execute(**kwargs), timeout=exec_timeout)
        except asyncio.TimeoutError:
            return f"[ERREUR] Tool '{name}' timeout apres {exec_timeout}s"
        except Exception as e:
            return f"[ERREUR] Tool '{name}' a echoue : {type(e).__name__}: {e}"
