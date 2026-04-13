"""State Machine Planner -- boucle tool-calling en graphe d'etats.

Inspire de LangGraph (state machine + transitions) et Letta V1 (model-native control).
Adapte a Qwen 3.5 9B : XML tool call fallback, presence_penalty, reponses courtes.

Etats :
  CALL_LLM       -> appel modele
  PARSE_RESPONSE -> analyse reponse (XML fallback, tool_calls detection)
  EXECUTE_TOOLS  -> execution des tool calls
  CHECK_RESPONSE -> decide : continuer, corriger, ou terminer
  RESPOND        -> reponse finale
  FORCE_RESPOND  -> limite atteinte, force synthesis
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator

from agent.models.local_model import LocalModel
from agent.tools.base import ToolRegistry
from agent.core.memory import Memory

logger = logging.getLogger("setharkk.planner")

# -- Constantes --
DEFAULT_TOOL_TIMEOUT = 30
RETRY_INDICATORS = ["timeout", "[ERREUR]", "ConnectionError", "ReadTimeout"]
NO_RETRY_INDICATORS = ["introuvable", "not found", "Action inconnue", "unknown"]
FILE_EXTENSIONS = [".py", ".yaml", ".yml", ".json", ".md", ".txt", ".js", ".html", ".css", ".toml"]

# Etats
CALL_LLM = "CALL_LLM"
PARSE_RESPONSE = "PARSE_RESPONSE"
EXECUTE_TOOLS = "EXECUTE_TOOLS"
CHECK_RESPONSE = "CHECK_RESPONSE"
REFLECT = "REFLECT"
RESPOND = "RESPOND"
FORCE_RESPOND = "FORCE_RESPOND"
END = "END"


@dataclass
class PlanStep:
    """Etape de plan (conserve pour compatibilite)."""
    name: str
    type: str
    tool_name: str | None = None
    tool_args: dict | None = None


@dataclass
class Plan:
    """Plan d'execution structure."""
    task: str
    steps: list[PlanStep] = field(default_factory=list)


class AgentStep:
    """Etape d'execution emise par l'agent (compatible CLI + web)."""

    def __init__(self, kind: str, content: str, tool_name: str = "", tool_args: dict | None = None):
        self.kind = kind  # "think", "act", "observe", "response", "plan"
        self.content = content
        self.tool_name = tool_name
        self.tool_args = tool_args or {}


@dataclass
class PlannerState:
    """Etat partage entre les nodes de la state machine."""
    messages: list[dict]
    tools_schemas: list[dict]
    recent_tools: list[str] = field(default_factory=list)
    step_count: int = 0
    max_steps: int = 50
    grounded_check_done: bool = False
    reflection_done: bool = False
    original_user_input: str = ""
    current_phase: str = CALL_LLM
    llm_result: dict | None = None
    pending_steps: list[AgentStep] = field(default_factory=list)


class PlanAndExecute:
    """State machine planner -- chaque etat est un node, les transitions sont explicites."""

    def __init__(self, model: LocalModel, tools: ToolRegistry, memory: Memory, context_manager: object | None = None, config: dict | None = None) -> None:
        self.model = model
        self.tools = tools
        self.memory = memory
        self.context_manager = context_manager
        refl = (config or {}).get("reflection", {})
        self._completude_ratio: float = refl.get("completude_ratio", 0.05)
        self._completude_min_chars: int = refl.get("completude_min_chars", 500)
        self._grounding_min_matches: int = refl.get("grounding_min_matches", 2)
        self._grounding_word_min_len: int = refl.get("grounding_word_min_len", 6)
        self._pertinence_word_min_len: int = refl.get("pertinence_word_min_len", 4)

    async def run_simple(
        self, messages: list[dict], tools_schemas: list[dict], max_rounds: int = 50
    ) -> AsyncIterator[AgentStep]:
        """Point d'entree public. API identique a l'ancienne version."""
        # Extraire la question originale (premier message user)
        user_input = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_input = msg.get("content") or ""
                break

        state = PlannerState(
            messages=messages,
            tools_schemas=tools_schemas,
            max_steps=max_rounds,
            original_user_input=user_input,
        )

        current = CALL_LLM

        while current != END:
            state.current_phase = current
            logger.debug("Planner state: %s (step %d/%d)", current, state.step_count, state.max_steps)

            if current == CALL_LLM:
                current = await self._node_call_llm(state)
            elif current == PARSE_RESPONSE:
                current = self._node_parse_response(state)
            elif current == EXECUTE_TOOLS:
                current = await self._node_execute_tools(state)
            elif current == CHECK_RESPONSE:
                current = self._node_check_response(state)
            elif current == REFLECT:
                current = self._node_reflect(state)
            elif current == RESPOND:
                await self._node_respond(state)
                current = END
            elif current == FORCE_RESPOND:
                await self._node_force_respond(state)
                current = END
            else:
                logger.error("Unknown planner state: %s", current)
                current = END

            # Emettre les steps accumules par les nodes
            for step in state.pending_steps:
                yield step
            state.pending_steps.clear()

    # ── Nodes ──────────────────────────────────────────────────────────

    async def _node_call_llm(self, state: PlannerState) -> str:
        """Appelle le LLM. Retourne PARSE_RESPONSE ou FORCE_RESPOND."""
        state.step_count += 1

        if state.step_count > state.max_steps:
            return FORCE_RESPOND

        state.llm_result = await self.model.chat_with_tools(
            state.messages, state.tools_schemas
        )
        return PARSE_RESPONSE

    def _node_parse_response(self, state: PlannerState) -> str:
        """Analyse la reponse LLM. Retourne EXECUTE_TOOLS ou CHECK_RESPONSE."""
        result = state.llm_result
        if not result:
            return CHECK_RESPONSE

        # Emettre le raisonnement si present
        reasoning = result.get("reasoning")
        if reasoning:
            state.pending_steps.append(AgentStep("think", reasoning))

        # Si pas de tool_calls natif, chercher du XML dans le content (Qwen 3.5 quirk)
        if not result.get("tool_calls"):
            content = result.get("content") or ""
            if content and "<tool_call>" in content and "<function=" in content:
                parsed = self._parse_xml_tool_call(content)
                if parsed:
                    result["tool_calls"] = [parsed]

        if result.get("tool_calls"):
            return EXECUTE_TOOLS

        return CHECK_RESPONSE

    async def _node_execute_tools(self, state: PlannerState) -> str:
        """Execute les tool calls. Retourne toujours CALL_LLM."""
        result = state.llm_result
        tool_calls = result.get("tool_calls", [])

        # Persister le message assistant avec ses tool_calls
        await self.memory.add_assistant_tool_calls(tool_calls)

        for tc in tool_calls:
            func = tc["function"]
            tool_name = func["name"]
            tool_args = func["arguments"]
            state.recent_tools.append(tool_name)

            state.pending_steps.append(
                AgentStep("act", f"{tool_name}({tool_args})", tool_name, tool_args)
            )

            # Timeout adaptatif
            tool_obj = self.tools.get(tool_name)
            timeout = getattr(tool_obj, "timeout", DEFAULT_TOOL_TIMEOUT) if tool_obj else DEFAULT_TOOL_TIMEOUT

            # Executer avec retry si erreur transitoire
            tool_result = await self.tools.execute(
                tool_name, exec_timeout=timeout, **tool_args
            )
            if self._is_retriable(tool_result):
                tool_result = await self.tools.execute(
                    tool_name, exec_timeout=timeout, **tool_args
                )

            state.pending_steps.append(AgentStep("observe", tool_result))

            # Persister dans recall memory
            await self.memory.add_tool_result(tc["id"], tool_name, tool_result)

            # Ajouter aux messages pour le prochain tour
            state.messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tool_name, "arguments": json.dumps(tool_args)},
                }],
            })
            state.messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": tool_result,
            })

        # Compresser les tool results si le contexte grossit
        if self.context_manager:
            state.messages = await self.context_manager.compact_tool_results(state.messages)

        return CALL_LLM

    def _node_check_response(self, state: PlannerState) -> str:
        """Decide quoi faire de la reponse. Retourne le prochain etat."""
        result = state.llm_result
        content = (result.get("content") or "") if result else ""

        # Content vide apres des tool calls → forcer le LLM a synthetiser
        if not content and state.recent_tools:
            state.messages.append({
                "role": "user",
                "content": "Tu as lu les informations demandees. Donne ta reponse maintenant.",
            })
            return CALL_LLM

        # Reponse tronquee → continuer
        if content and result and result.get("truncated"):
            state.pending_steps.append(AgentStep("response", content))
            state.messages.append({"role": "assistant", "content": content})
            state.messages.append({
                "role": "user",
                "content": "Continue ta reponse exactement la ou tu t'es arrete.",
            })
            return CALL_LLM

        # Grounded check (une seule fois par run)
        if content and not state.grounded_check_done:
            nudge = self._grounded_check(content, state.recent_tools)
            if nudge:
                state.grounded_check_done = True
                state.messages.append({"role": "assistant", "content": content})
                state.messages.append({"role": "user", "content": nudge})
                return CALL_LLM

        # Reponse candidate -- passe par la reflexion avant emission
        if content:
            return REFLECT

        # Rien du tout et limite atteinte
        if state.step_count >= state.max_steps:
            return FORCE_RESPOND

        # Rien du tout, pas de tools recents → on tente encore
        return CALL_LLM

    def _node_reflect(self, state: PlannerState) -> str:
        """Reflexion heuristique avant emission. Adapte au Qwen 3.5 9B."""
        # Si reflexion deja faite, pas de boucle infinie
        if state.reflection_done:
            return RESPOND

        result = state.llm_result
        content = (result.get("content") or "") if result else ""
        if not content:
            return RESPOND

        # Check 1 -- Completude (ratio reponse / tool results)
        # Le 9B s'arrete souvent a 305 tokens meme avec du contexte riche
        tool_results_size = sum(
            len(m.get("content") or "")
            for m in state.messages if m.get("role") == "tool"
        )
        if tool_results_size > 0:
            ratio = len(content) / tool_results_size
            if ratio < self._completude_ratio and len(content) < self._completude_min_chars:
                state.reflection_done = True
                state.messages.append({"role": "assistant", "content": content})
                state.messages.append({
                    "role": "user",
                    "content": (
                        "[reflexion] Ta reponse est trop courte par rapport "
                        "aux informations collectees. Developpe ta reponse "
                        "en utilisant les donnees que tu as lues."
                    ),
                })
                logger.info("REFLECT: completude echouee (ratio=%.2f, content=%d, tools=%d)",
                            ratio, len(content), tool_results_size)
                return CALL_LLM

        # Check 2 -- Grounding des tool results
        # La reponse doit mentionner des mots-cles des resultats de tools
        tool_words: set[str] = set()
        for m in state.messages:
            if m.get("role") == "tool":
                for word in (m.get("content") or "").split():
                    if len(word) > self._grounding_word_min_len:
                        tool_words.add(word.lower())
        if tool_words:
            response_lower = content.lower()
            matches = sum(1 for w in list(tool_words)[:20] if w in response_lower)
            if matches < self._grounding_min_matches:
                state.reflection_done = True
                state.messages.append({"role": "assistant", "content": content})
                state.messages.append({
                    "role": "user",
                    "content": (
                        "[reflexion] Ta reponse ne semble pas utiliser "
                        "les informations des tools. Integre les resultats "
                        "dans ta reponse."
                    ),
                })
                logger.info("REFLECT: grounding echoue (matches=%d, tool_words=%d)",
                            matches, len(tool_words))
                return CALL_LLM

        # Check 3 -- Pertinence vs question originale
        if state.original_user_input:
            question_words = [
                w.lower() for w in state.original_user_input.split()
                if len(w) > self._pertinence_word_min_len
            ]
            if question_words:
                response_lower = content.lower()
                matches = sum(1 for w in question_words if w in response_lower)
                if matches == 0:
                    state.reflection_done = True
                    state.messages.append({"role": "assistant", "content": content})
                    state.messages.append({
                        "role": "user",
                        "content": (
                            "[reflexion] Ta reponse ne semble pas repondre "
                            "a la question posee. Relis la question et reponds."
                        ),
                    })
                    logger.info("REFLECT: pertinence echouee (aucun mot-cle de la question)")
                    return CALL_LLM

        # Check 4 -- Format validation (XML orphelins, specifique Qwen 3.5 9B)
        has_open_xml = "<tool_call>" in content or "<function=" in content
        has_close_xml = "</tool_call>" in content or "</function>" in content
        if has_open_xml and not has_close_xml:
            state.reflection_done = True
            state.messages.append({"role": "assistant", "content": content})
            state.messages.append({
                "role": "user",
                "content": (
                    "[reflexion] Ta reponse contient un appel de tool mal forme. "
                    "Reformule ta reponse ou appelle le tool correctement."
                ),
            })
            logger.info("REFLECT: format XML orphelin detecte")
            return CALL_LLM

        # Tous les checks passent
        return RESPOND

    async def _node_respond(self, state: PlannerState) -> None:
        """Emet la reponse finale et la persiste."""
        content = (state.llm_result.get("content") or "") if state.llm_result else ""
        await self.memory.add_message("assistant", content)
        state.pending_steps.append(AgentStep("response", content))

    async def _node_force_respond(self, state: PlannerState) -> None:
        """Force le LLM a synthetiser quand la limite est atteinte."""
        try:
            state.messages.append({
                "role": "user",
                "content": "Tu as atteint la limite de tours. Donne ta reponse finale maintenant avec ce que tu as.",
            })
            final = await self.model.chat_with_tools(
                state.messages, state.tools_schemas, tool_choice="none"
            )
            content = final.get("content") or "[Limite de tours atteinte]"
            await self.memory.add_message("assistant", content)
            state.pending_steps.append(AgentStep("response", content))
        except Exception:
            state.pending_steps.append(AgentStep("response", "[Limite de tours atteinte]"))

    # ── Fonctions statiques (inchangees) ───────────────────────────────

    @staticmethod
    def _is_retriable(tool_result: str) -> bool:
        """Verifie si le resultat indique une erreur transitoire retentable."""
        lower = tool_result.lower()
        for no_retry in NO_RETRY_INDICATORS:
            if no_retry.lower() in lower:
                return False
        for indicator in RETRY_INDICATORS:
            if indicator.lower() in lower:
                return True
        return False

    @staticmethod
    def _grounded_check(response: str, recent_tool_names: list[str]) -> str | None:
        """Verifie que le LLM a utilise les tools avant d'affirmer."""
        mentions_file = any(ext in response for ext in FILE_EXTENSIONS)
        used_file_tool = "file_system" in recent_tool_names or "review_code" in recent_tool_names
        if mentions_file and not used_file_tool:
            return (
                "[systeme] Tu mentionnes un fichier dans ta reponse mais tu ne l'as pas lu. "
                "Utilise file_system pour lire le fichier d'abord, puis reponds."
            )

        web_indicators = ["selon ", "d'apres ", "source:", "http"]
        mentions_web = any(ind in response.lower() for ind in web_indicators)
        used_search = "search" in recent_tool_names or "research" in recent_tool_names
        if mentions_web and not used_search:
            return (
                "[systeme] Tu mentionnes des sources web mais tu n'as pas fait de recherche. "
                "Utilise search ou research pour verifier d'abord."
            )

        return None

    @staticmethod
    def _parse_xml_tool_call(content: str) -> dict | None:
        """Parse un tool call XML que Qwen 3.5 met parfois dans le content."""
        tc_start = content.find("<tool_call>")
        tc_end = content.find("</tool_call>")
        if tc_start == -1 or tc_end == -1:
            return None

        block = content[tc_start:tc_end]

        fn_start = block.find("<function=")
        if fn_start == -1:
            return None
        fn_name_start = fn_start + len("<function=")
        fn_name_end = block.find(">", fn_name_start)
        if fn_name_end == -1:
            return None
        func_name = block[fn_name_start:fn_name_end].strip()

        arguments: dict = {}
        search_from = fn_name_end
        while True:
            p_start = block.find("<parameter=", search_from)
            if p_start == -1:
                break
            p_name_start = p_start + len("<parameter=")
            p_name_end = block.find(">", p_name_start)
            if p_name_end == -1:
                break
            param_name = block[p_name_start:p_name_end].strip()

            p_val_start = p_name_end + 1
            p_val_end = block.find("</parameter>", p_val_start)
            if p_val_end == -1:
                break
            param_value = block[p_val_start:p_val_end].strip()

            # Convertir les valeurs numeriques (le XML donne tout en string)
            if param_value.isdigit():
                arguments[param_name] = int(param_value)
            else:
                arguments[param_name] = param_value
            search_from = p_val_end + len("</parameter>")

        if not func_name:
            return None

        return {
            "id": f"xml_{func_name}",
            "function": {
                "name": func_name,
                "arguments": arguments,
            },
        }
