from __future__ import annotations

from typing import Any

from agent.tools.base import Tool


class RememberTool(Tool):
    name = "remember"
    description = "Stocke un fait, une decision, une preference ou une competence dans ta memoire long terme. Utilise-le quand tu apprends quelque chose d'important."
    parameters = {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Le fait a retenir"},
            "category": {
                "type": "string",
                "enum": ["fact", "decision", "preference", "skill", "error"],
                "description": "Type de souvenir",
            },
        },
        "required": ["content", "category"],
    }

    def __init__(self, memory):
        self._memory = memory

    async def execute(self, content: str, category: str = "fact", **_) -> str:
        await self._memory.store(content, {"type": "knowledge", "category": category})
        return f"Memorise [{category}] : {content}"


class RecallTool(Tool):
    name = "recall"
    description = "Cherche dans ta memoire long terme. Utilise-le pour retrouver des faits, decisions, preferences ou competences que tu as appris dans des sessions precedentes."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Ce que tu cherches dans ta memoire"},
        },
        "required": ["query"],
    }

    def __init__(self, memory):
        self._memory = memory

    async def execute(self, query: str, **_) -> str:
        memories = await self._memory.recall_archival(query)
        if not memories:
            return "Aucun souvenir pertinent trouve."
        lines: list[str] = []
        for i, m in enumerate(memories, 1):
            meta = m.get("metadata", {})
            cat = meta.get("category", meta.get("type", "?"))
            score = m.get("score", m.get("similarity", 0))
            lines.append(f"[{i}] [{cat}] (score:{score:.2f}) {m['content']}")
        return "\n".join(lines)


class CoreMemoryUpdateTool(Tool):
    name = "core_memory_update"
    description = "Met a jour ta memoire centrale (identite, profil utilisateur, regles). Ces informations sont toujours presentes dans ton contexte."
    parameters = {
        "type": "object",
        "properties": {
            "section": {
                "type": "string",
                "enum": ["persona", "user_profile", "system_rules"],
                "description": "Section a modifier",
            },
            "key": {"type": "string", "description": "Cle a creer ou modifier"},
            "value": {"type": "string", "description": "Nouvelle valeur"},
        },
        "required": ["section", "key", "value"],
    }

    def __init__(self, memory):
        self._memory = memory

    async def execute(self, section: str, key: str, value: str, **_) -> str:
        await self._memory.core.set(section, key, value)
        return f"Core memory mise a jour : [{section}] {key} = {value}"


class ForgetTool(Tool):
    name = "forget"
    description = "Supprime un souvenir de ta memoire long terme par similarite. Utilise-le quand une information n'est plus vraie."
    parameters = {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Le fait a oublier (sera cherche par similarite et supprime)"},
        },
        "required": ["content"],
    }

    def __init__(self, memory):
        self._memory = memory

    async def execute(self, content: str, **_: Any) -> str:
        deleted = await self._memory.delete_by_similarity(content, threshold=0.85)
        if deleted > 0:
            return f"Oublie : {deleted} souvenir(s) supprime(s) pour '{content}'"
        return f"Aucun souvenir similaire trouve pour : {content}"


class SearchHistoryTool(Tool):
    """Recherche cross-session : combine recall textuel + archival semantique."""

    name = "search_history"
    description = (
        "Recherche dans l'historique complet des conversations passees (toutes sessions). "
        "Combine recherche textuelle et recherche semantique."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Ce que tu cherches dans l'historique des conversations",
            },
            "max_results": {
                "type": "integer",
                "description": "Nombre maximum de resultats (defaut: 10)",
            },
        },
        "required": ["query"],
    }

    def __init__(self, memory: Any, config: dict | None = None) -> None:
        self._memory = memory
        defaults = (config or {}).get("tools", {}).get("defaults", {})
        self._default_max: int = defaults.get("history_max_results", 15)

    async def execute(self, query: str, max_results: int = 0, **_: Any) -> str:
        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = 0
        if max_results <= 0:
            max_results = self._default_max
        # 1. Recherche textuelle cross-session (recall_memory)
        text_hits = await self._memory.recall.search_text(query, limit=max_results)

        # 2. Recherche semantique (archival_memory)
        archival_hits = await self._memory.recall_archival(query, k=max_results)

        # 3. Merge avec dedup
        lines: list[str] = []
        seen_snippets: set[str] = set()

        # Recall hits d'abord (match textuel exact)
        for hit in text_hits:
            snippet = (hit.get("content") or "")[:80]
            if snippet in seen_snippets:
                continue
            seen_snippets.add(snippet)
            sid = hit.get("session_id", "?")
            role = hit.get("role", "?")
            ts = hit.get("created_at", "")
            date_str = str(ts)[:10] if ts else "?"
            lines.append(f"[recall] [session:{sid}] [{date_str}] {role}: {hit.get('content', '')[:200]}")

        # Archival hits ensuite (match semantique)
        for hit in archival_hits:
            snippet = (hit.get("content") or "")[:80]
            if snippet in seen_snippets:
                continue
            seen_snippets.add(snippet)
            score = hit.get("score", hit.get("similarity", 0))
            lines.append(f"[archival] (score:{score:.2f}) {hit.get('content', '')[:200]}")

        if not lines:
            return f"Aucun resultat dans l'historique pour : {query}"

        # Limiter le nombre total
        lines = lines[:max_results]
        return "## Historique des conversations\n" + "\n".join(f"[{i+1}] {l}" for i, l in enumerate(lines))


class GraphRecallTool(Tool):
    """Explore le graphe de connaissances unifie (Neo4j)."""

    name = "graph_recall"
    description = (
        "Explore le graphe de connaissances : trouve des entites (personnes, projets, "
        "technologies, concepts) et leurs relations. Combine vector search et graph traversal."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Ce que tu cherches dans le graphe de connaissances",
            },
            "depth": {
                "type": "integer",
                "description": "Profondeur de traversee du graphe (defaut: 1, max: 3)",
            },
        },
        "required": ["query"],
    }

    def __init__(self, memory: Any, config: dict | None = None) -> None:
        self._memory = memory
        defaults = (config or {}).get("tools", {}).get("defaults", {})
        self._default_depth: int = defaults.get("graph_depth", 1)
        self._max_depth: int = defaults.get("graph_max_depth", 3)

    async def execute(self, query: str, depth: int = 0, **_: Any) -> str:
        try:
            depth = int(depth)
        except (TypeError, ValueError):
            depth = 0
        if depth <= 0:
            depth = self._default_depth
        depth = min(max(depth, 1), self._max_depth)
        result = await self._memory.recall_with_graph(query, k=5, graph_depth=depth)

        lines: list[str] = []

        # Entites
        entities = result.get("entities", [])
        if entities:
            lines.append("## Entites")
            for e in entities:
                imp = e.get("importance", 0)
                desc = e.get("description", "") or ""
                lines.append(f"- [{e.get('entity_type', '?')}] {e.get('name', '?')} (imp:{imp:.1f}) {desc[:100]}")

        # Relations
        rels = result.get("relationships", [])
        if rels:
            lines.append("## Relations")
            for r in rels:
                types = r.get("types", [])
                type_str = ", ".join(str(t) for t in types) if types else "?"
                lines.append(f"- {r.get('source', '?')} --[{type_str}]--> {r.get('target', '?')}")

        # Faits lies aux entites
        graph_mems = result.get("graph_memories", [])
        if graph_mems:
            lines.append("## Faits lies")
            for i, m in enumerate(graph_mems[:5], 1):
                lines.append(f"[{i}] {m.get('content', '')[:200]}")

        # Souvenirs vector search
        memories = result.get("memories", [])
        if memories:
            lines.append("## Souvenirs (vector)")
            for i, m in enumerate(memories[:5], 1):
                score = m.get("score", m.get("similarity", 0))
                lines.append(f"[{i}] (score:{score:.2f}) {m['content'][:200]}")

        if not lines:
            return f"Aucun resultat dans le graphe pour : {query}"

        return "\n".join(lines)
