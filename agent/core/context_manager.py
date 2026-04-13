"""Context Manager Intelligent -- inspire ReMe + Mastra.

Gere le budget tokens du contexte LLM :
1. Compte les chars des messages de session
2. Si > compact_threshold : compresse les vieux messages en resume structure
3. Injecte [system + resume + N messages recents] au lieu de tout l'historique
4. Compresse les tool results dans la boucle run_simple quand ca deborde
5. Sauvegarde les resumes en archival pour les futures sessions

La compaction est faite COTE SERVEUR (Python), pas par le LLM.
Le LLM ne gere pas sa memoire -- il recoit un contexte deja optimise.
"""

from __future__ import annotations

from agent.models.local_model import LocalModel
from agent.core.memory import Memory


class ContextManager:
    def __init__(self, model: LocalModel, memory: Memory, config: dict | None = None):
        self.model = model
        self.memory = memory
        cfg = config or {}
        self.compact_threshold = cfg.get("compact_threshold", 20000)
        self.keep_recent = cfg.get("keep_recent", 6)
        self.summary_max_chars = cfg.get("summary_max_chars", 2000)
        self.tool_result_threshold = cfg.get("tool_result_threshold", 30000)
        self._old_tool_truncation: int = cfg.get("old_tool_truncation", 3000)
        self._compact_keep_last: int = cfg.get("compact_keep_last", 4)
        self._current_summary: str | None = None

    async def prepare_context(self, system_prompt: str) -> list[dict]:
        """Prepare le contexte optimal pour le LLM.

        Sous le seuil : retourne tout (comportement identique a l'actuel).
        Au-dessus : compacte les vieux messages, garde les recents.

        Retourne : [system_msg, (summary_msg), recent_msgs...]
        """
        messages = await self.memory.recall.get_session_unlimited()
        total_chars = sum(len(m.get("content", "") or "") for m in messages)

        result = [{"role": "system", "content": system_prompt}]

        # Sous le seuil : pas de compaction
        if total_chars <= self.compact_threshold:
            result.extend(messages)
            return result

        # Au-dessus : compacter les vieux, garder les recents
        if len(messages) <= self.keep_recent:
            result.extend(messages)
            return result

        old_messages = messages[:-self.keep_recent]
        recent_messages = messages[-self.keep_recent:]

        # Compacter les vieux messages en resume
        summary = await self._compact(old_messages)
        self._current_summary = summary

        # Injecter le resume DANS le system prompt (pas un 2eme message system)
        # Qwen --jinja exige un seul message system au debut
        if summary:
            result[0]["content"] += f"\n\n## Resume de la conversation precedente :\n{summary}"

        result.extend(recent_messages)
        return result

    async def _compact(self, messages: list[dict]) -> str:
        """Compresse une liste de messages en resume structure CPU-only."""
        if not messages:
            return ""

        summary = self._fallback_summary(messages)

        # Sauvegarder le resume en archival pour les futures sessions
        if summary:
            await self.memory.store(
                f"[SUMMARY] {summary}",
                {"type": "summary", "message_count": len(messages)},
                source_session_id=self.memory._session_id,
            )

        return summary

    def _fallback_summary(self, messages: list[dict]) -> str:
        """Resume structure sans LLM -- extraction deterministe."""
        user_msgs: list[str] = []
        assistant_actions: list[str] = []
        tool_calls: list[str] = []

        for m in messages:
            role = m.get("role", "")
            content = (m.get("content") or "")[:150]
            if role == "user" and content:
                user_msgs.append(content)
            elif role == "assistant" and content:
                assistant_actions.append(content)
            elif role == "tool":
                tool_name = m.get("tool_name", "?")
                tool_calls.append(tool_name)

        parts: list[str] = []
        if user_msgs:
            parts.append("OBJECTIF: " + " | ".join(user_msgs[-5:]))
        if tool_calls:
            parts.append("ACTIONS: " + ", ".join(tool_calls[-10:]))
        if assistant_actions:
            parts.append("PROGRES: " + " | ".join(assistant_actions[-3:]))

        return "\n".join(parts) if parts else ""

    async def compact_tool_results(self, messages: list[dict]) -> list[dict]:
        """Compresse les resultats de tools pendant la boucle run_simple.

        Tronque les vieux tool results intermediaires pour eviter
        que le contexte explose pendant les boucles multi-tour.
        """
        total = sum(len(m.get("content", "") or "") for m in messages)
        if total <= self.tool_result_threshold:
            return messages

        # Garder les N derniers messages intacts (current tool exchange)
        # Tronquer les tool results anterieurs
        compacted = []
        cutoff = len(messages) - self._compact_keep_last

        for i, m in enumerate(messages):
            if m.get("role") == "tool" and i < cutoff:
                content = m.get("content", "") or ""
                if len(content) > self._old_tool_truncation:
                    m = dict(m)
                    m["content"] = content[:self._old_tool_truncation] + "\n[... resultat tronque pour economiser le contexte ...]"
            compacted.append(m)

        return compacted
