"""Compound tool : recherche approfondie (search + browse + analyse LLM).

Encapsule le pipeline search -> browse top pages -> synthese LLM
en un seul appel de tool pour le LLM principal.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from agent.tools.base import Tool, ToolRegistry
from agent.models.local_model import LocalModel

_DEFAULT_MAX_BROWSE_PAGES = 4
_DEFAULT_MAX_BROWSE_CHARS = 15000
_DEFAULT_SEARCH_TIMEOUT = 30
_DEFAULT_BROWSE_TIMEOUT = 20
_DEFAULT_COMPOUND_TIMEOUT = 120


class ResearchTool(Tool):
    """Recherche approfondie : search web + browse pages + synthese LLM."""

    name = "research"
    description = (
        "Recherche approfondie sur un sujet. Cherche sur le web, "
        "browse les pages les plus pertinentes, et produit une synthese structuree. "
        "Utilise ce tool pour des sujets necessitant plusieurs sources."
    )
    parameters = {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "Le sujet a rechercher"},
            "max_results": {
                "type": "integer",
                "description": "Nombre de resultats de recherche (defaut: 5)",
            },
        },
        "required": ["topic"],
    }

    def __init__(self, tools: ToolRegistry, model: LocalModel, config: dict | None = None) -> None:
        self._tools = tools
        self._model = model
        rcfg = (config or {}).get("tools", {}).get("research", {})
        self._max_browse_pages: int = rcfg.get("max_browse_pages", _DEFAULT_MAX_BROWSE_PAGES)
        self._max_browse_chars: int = rcfg.get("max_browse_chars", _DEFAULT_MAX_BROWSE_CHARS)
        self._search_timeout: int = rcfg.get("search_timeout", _DEFAULT_SEARCH_TIMEOUT)
        self._browse_timeout: int = rcfg.get("browse_timeout", _DEFAULT_BROWSE_TIMEOUT)
        self.timeout: int = rcfg.get("compound_timeout", _DEFAULT_COMPOUND_TIMEOUT)

    async def execute(self, topic: str, max_results: int = 5, **_: Any) -> str:
        """Execute la recherche approfondie et retourne une synthese texte."""
        current_year = str(date.today().year)
        search_query = f"{topic} {current_year}"

        # Etape 1 : Recherche web
        search_text = await self._tools.execute(
            "search", exec_timeout=self._search_timeout, query=search_query, max_results=max_results
        )
        structured = self._tools.get_structured("search") or []
        self.last_structured = structured

        # Etape 2 : Browse les pages les plus pertinentes
        browsed_content = ""
        urls = [r.get("href", "") for r in structured if r.get("href")]
        if not urls:
            urls = _extract_urls(search_text)

        for url in urls[:self._max_browse_pages]:
            try:
                await self._tools.execute("browser", exec_timeout=self._browse_timeout, action="navigate", url=url)
                text = await self._tools.execute("browser", exec_timeout=self._browse_timeout, action="extract_text")
                if text and not text.startswith("[ERREUR]"):
                    browsed_content += f"\n--- {url} ---\n{text[:self._max_browse_chars]}\n"
            except Exception:
                pass

        # Etape 3 : Synthese LLM
        full_context = f"Resultats de recherche:\n{search_text}"
        if browsed_content:
            full_context += f"\n\nContenu extrait des pages:\n{browsed_content}"

        try:
            result = await self._model.chat_with_tools(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Analyse ces resultats de recherche et le contenu extrait. "
                            "Identifie les points cles et produis une synthese structuree. "
                            "En francais."
                        ),
                    },
                    {"role": "user", "content": f"Sujet: {topic}\n\n{full_context}"},
                ],
                tools=[],
                tool_choice="none",
            )
            synthesis = result.get("content") or search_text
        except Exception:
            synthesis = search_text

        # Validation : la synthese doit mentionner le sujet
        if synthesis and topic.lower() not in synthesis.lower():
            # La synthese ne mentionne meme pas le sujet -- fallback
            synthesis = search_text

        # Formater la sortie
        source_count = len(structured)
        browse_count = min(len(urls), self._max_browse_pages)
        header = f"[Recherche approfondie : {source_count} sources, {browse_count} pages browsees]\n\n"
        return header + synthesis


def _extract_urls(search_text: str) -> list[str]:
    """Extrait les URLs des resultats de recherche texte."""
    urls: list[str] = []
    for line in search_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("URL:"):
            url = stripped[4:].strip()
            if url:
                urls.append(url)
    return urls
