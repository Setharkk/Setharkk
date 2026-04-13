"""Agent Setharkk -- Router + Tool Calling Natif.

Flux :
- DIRECT_RECALL / DIRECT_FILE_READ / DIRECT_FILE_LIST : zero LLM, execution directe
- Tout le reste : le LLM decide via tool calling natif (--jinja)

Le ContextManager gere le budget tokens automatiquement.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from agent.models.local_model import LocalModel
from agent.models.prompts import build_system_prompt
from agent.tools.base import ToolRegistry
from agent.core.memory import Memory
from agent.core.router import Router, IntentClass
from agent.core.planner import PlanAndExecute, AgentStep
from agent.core.context_manager import ContextManager
from agent.core.data_collector import DataCollector

logger = logging.getLogger("setharkk.agent")


class Agent:
    """Agent principal Setharkk."""

    def __init__(
        self,
        model: LocalModel,
        tools: ToolRegistry,
        memory: Memory,
        max_iterations: int = 15,
        tool_timeout: float = 30,
        context_config: dict | None = None,
        config: dict | None = None,
    ):
        self.model = model
        self.tools = tools
        self.memory = memory
        self.max_iterations = max_iterations
        self.tool_timeout = tool_timeout
        self.collector = DataCollector()
        self.router = Router()
        self.context_manager = ContextManager(model, memory, context_config or {})

        # Boot config
        boot_cfg = (config or {}).get("boot", {})
        self._boot_identity_k: int = boot_cfg.get("identity_k", 3)
        self._boot_context_k: int = boot_cfg.get("context_k", 5)
        self._boot_graph_k: int = boot_cfg.get("graph_k", 2)
        self._boot_graph_depth: int = boot_cfg.get("graph_depth", 1)
        self._boot_max_merged: int = boot_cfg.get("max_merged", 7)

        # Resume config
        resume_cfg = (config or {}).get("resume", {})
        self._resume_min_messages: int = resume_cfg.get("min_messages", 5)
        self._resume_last_n: int = resume_cfg.get("last_n_messages", 10)

        # Reflection config (for _check_task_completion)
        refl_cfg = (config or {}).get("reflection", {})
        self._task_completion_min_chars: int = refl_cfg.get("task_completion_min_chars", 300)

        self.planner = PlanAndExecute(model, tools, memory, self.context_manager, config)
        self._session_initialized = False

    _ACTION_WORDS: list[str] = [
        "analyse", "compare", "explique", "liste", "detaille", "resume",
        "decris", "examine", "investigue", "synthetise", "evalue",
        "analyze", "compare", "explain", "list", "detail", "describe",
    ]

    def _check_task_completion(self, user_input: str, response: str) -> str | None:
        """Verifie si la tache est complete. Retourne une critique ou None.

        Heuristique pure, pas de LLM. Calibre pour Qwen 3.5 9B
        qui s'arrete souvent a ~300 tokens sans finir.
        """
        if not response:
            return "La reponse est vide"

        is_complex = any(w in user_input.lower() for w in Agent._ACTION_WORDS)
        if is_complex and len(response) < self._task_completion_min_chars:
            return "La question demande une analyse detaillee mais la reponse est trop courte"

        return None

    async def _build_system_prompt(self) -> str:
        """Construit le system prompt avec core memory + tools dynamiques."""
        core_block = await self.memory.get_core_prompt()
        tool_schemas = self.tools.list_schemas()
        return build_system_prompt(core_memory_block=core_block, tool_schemas=tool_schemas)

    async def _resume_last_session(self) -> str:
        """Charge le contexte de la derniere session pour reprise.

        Cherche la session la plus recente (autre que la courante) dans recall_memory,
        extrait les derniers echanges, et construit un resume.
        """
        current_sid = self.memory._session_id
        try:
            async with self.memory._pool.acquire() as conn:
                # Prendre la session recente la plus substantielle (>= 5 messages)
                row = await conn.fetchrow(
                    "SELECT session_id, COUNT(*) AS cnt "
                    "FROM recall_memory "
                    "WHERE session_id != $1 "
                    "GROUP BY session_id "
                    "HAVING COUNT(*) >= $2 "
                    "ORDER BY MAX(created_at) DESC LIMIT 1",
                    current_sid, self._resume_min_messages,
                )
                if not row:
                    return ""

                prev_sid = row["session_id"]
                msgs = await conn.fetch(
                    "SELECT role, content, tool_name FROM recall_memory "
                    "WHERE session_id = $1 AND role IN ('user', 'assistant') "
                    "AND (tool_name IS NULL OR tool_name != 'tool_calling') "
                    "AND content != '' "
                    "ORDER BY created_at DESC LIMIT $2",
                    prev_sid, self._resume_last_n,
                )

            if not msgs:
                return ""

            # Construire le resume (du plus recent au plus ancien, puis inverser)
            lines: list[str] = []
            for m in reversed(msgs):
                role = m["role"]
                content = (m["content"] or "")[:200]
                if content:
                    prefix = "Toi" if role == "user" else "Moi"
                    lines.append(f"  {prefix}: {content}")

            return (
                f"## Reprise de la session precedente ({prev_sid})\n"
                f"Voici les derniers echanges :\n"
                + "\n".join(lines)
                + "\n\nJe reprends a partir de la."
            )
        except Exception as exc:
            logger.warning("Resume session failed: %s", exc)
            return ""

    async def _boot_recall(self, system_prompt: str, user_input: str | None = None) -> str:
        """Recall initial au demarrage de session. 2 phases : identite + contexte adaptatif.

        Phase 1 (toujours) : rappel identite/regles (query anglaise pour bge-small-en-v1.5).
        Phase 2 (si user_input fourni) : rappel adaptatif base sur le message utilisateur.
        Retourne le prompt enrichi.
        """
        if self._session_initialized:
            return system_prompt

        self._session_initialized = True

        # Phase 1 : identite (query anglaise -- bge-small-en-v1.5 est anglophone)
        identity_memories = await self.memory.recall_archival(
            "agent identity rules creator preferences", k=self._boot_identity_k
        )

        # Phase 2 : contexte adaptatif base sur le message utilisateur
        context_memories: list[dict] = []
        if user_input:
            context_memories = await self.memory.recall_archival(user_input, k=self._boot_context_k)

        # Fusion + dedup par contenu exact, cap a max_merged
        seen_contents: set[str] = set()
        merged: list[dict] = []
        for m in identity_memories + context_memories:
            content = m["content"]
            if content not in seen_contents and len(merged) < self._boot_max_merged:
                seen_contents.add(content)
                merged.append(m)

        if merged:
            boot_lines = [f"- {m['content']}" for m in merged]
            system_prompt += "\n\n## Souvenirs de sessions precedentes :\n" + "\n".join(boot_lines)

        # Phase 3 : contexte du knowledge graph (entites liees au message)
        if user_input and self.memory.archival:
            try:
                subgraph = await self.memory.archival.get_subgraph(user_input, k=self._boot_graph_k, depth=self._boot_graph_depth)
                graph_lines: list[str] = []
                for entity in subgraph.get("entities", [])[:5]:
                    desc = entity.get("description", "") or ""
                    if desc:
                        graph_lines.append(
                            f"[{entity.get('entity_type', '?')}] "
                            f"{entity.get('name', '?')}: {desc}"
                        )
                if graph_lines:
                    system_prompt += "\n\n## Contexte du graphe de connaissances :\n" + "\n".join(
                        f"- {line}" for line in graph_lines
                    )
            except Exception as exc:
                logger.debug("Boot graph recall failed: %s", exc)

        return system_prompt

    async def run(self, user_input: str) -> AsyncIterator[AgentStep]:
        """Point d'entree principal."""

        # Stocker le message utilisateur
        await self.memory.add_message("user", user_input)

        # Construire le contexte
        system_prompt = await self._build_system_prompt()
        system_prompt = await self._boot_recall(system_prompt, user_input)

        # Router : shortcuts deterministes
        intent, params = self.router.classify(user_input)

        # -- DIRECT RESUME : reprise de session precedente --
        if intent == IntentClass.DIRECT_RESUME:
            resume_ctx = await self._resume_last_session()
            if resume_ctx:
                await self.memory.add_message("assistant", resume_ctx)
                yield AgentStep("response", resume_ctx)
            else:
                yield AgentStep("response", "Aucune session precedente trouvee.")
            return

        # -- DIRECT RECALL : zero LLM --
        if intent == IntentClass.DIRECT_RECALL:
            query = params.get("query", user_input)
            results = await self.memory.recall_archival(query)
            if results:
                lines = [f"[{i+1}] (sim:{m['similarity']:.2f}) {m['content']}" for i, m in enumerate(results)]
                response = "Souvenirs trouves :\n" + "\n".join(lines)
            else:
                response = "Aucun souvenir trouve pour cette requete."
            await self.memory.add_message("assistant", response)
            yield AgentStep("response", response)
            return

        # -- DIRECT FILE READ : zero LLM --
        if intent == IntentClass.DIRECT_FILE_READ:
            path = params.get("path", "")
            if path:
                result = await self.tools.execute("file_system", exec_timeout=self.tool_timeout, action="read", path=path)
                yield AgentStep("act", f"file_system(read, {path})", "file_system", {"action": "read", "path": path})
                yield AgentStep("observe", result)
                await self.memory.add_message("assistant", result)
                yield AgentStep("response", result)
            else:
                yield AgentStep("response", "Quel fichier veux-tu lire ? Donne-moi le chemin.")
            return

        # -- DIRECT FILE LIST : zero LLM --
        if intent == IntentClass.DIRECT_FILE_LIST:
            path = params.get("path", ".")
            result = await self.tools.execute("file_system", exec_timeout=self.tool_timeout, action="list", path=path)
            yield AgentStep("act", f"file_system(list, {path})", "file_system", {"action": "list", "path": path})
            yield AgentStep("observe", result)
            await self.memory.add_message("assistant", result)
            yield AgentStep("response", result)
            return

        # -- TOUT LE RESTE : le LLM decide via tool calling natif --
        messages = await self.context_manager.prepare_context(system_prompt)
        tools_schemas = self.tools.openai_schemas()

        # Collecter la derniere reponse pour la verification de completude
        last_response = ""
        async for step in self.planner.run_simple(messages, tools_schemas, max_rounds=self.max_iterations):
            if step.kind == "response":
                last_response = step.content
            yield step

        # Verification de completude de tache (heuristique, max 1 relance)
        critique = self._check_task_completion(user_input, last_response)
        if critique:
            logger.info("Task completion check failed: %s", critique)
            await self.memory.add_message("user", f"[reflexion] {critique}. Complete ta reponse.")
            messages = await self.context_manager.prepare_context(system_prompt)
            async for step in self.planner.run_simple(messages, tools_schemas, max_rounds=5):
                yield step

        # Sauvegarder l'interaction pour fine-tuning
        try:
            system_text = messages[0]["content"] if messages else ""
            post_msgs = await self.memory.get_messages()
            self.collector.save_interaction(system_text, post_msgs)
        except Exception as e:
            logger.warning("DataCollector save failed: %s", e)
