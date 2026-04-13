from __future__ import annotations

import asyncio

from agent.tools.base import Tool
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


def _search_sync(query: str, max_results: int) -> list[dict]:
    """Appel DDGS synchrone -- execute dans un thread via asyncio.to_thread."""
    return DDGS().text(query, max_results=max_results, region="fr-fr")


class SearchTool(Tool):
    name = "search"
    description = "Recherche sur le web via DuckDuckGo. Retourne les resultats les plus pertinents."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "La requete de recherche"},
            "max_results": {"type": "integer", "description": "Nombre max de resultats"},
        },
        "required": ["query"],
    }

    def __init__(self, config: dict | None = None) -> None:
        defaults = (config or {}).get("tools", {}).get("defaults", {})
        self._default_max_results: int = defaults.get("search_max_results", 10)

    async def execute(self, query: str, max_results: int = 0, **_) -> str:
        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = 0
        if max_results <= 0:
            max_results = self._default_max_results
        results = await asyncio.to_thread(_search_sync, query, max_results)
        if not results:
            self.last_structured = []
            return "Aucun resultat trouve."
        self.last_structured = results
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r['title']}")
            lines.append(f"    URL: {r['href']}")
            lines.append(f"    {r['body']}")
            lines.append("")
        return "\n".join(lines)
