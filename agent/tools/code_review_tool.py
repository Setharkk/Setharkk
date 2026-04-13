"""Compound tool : review de code (analyse statique + LLM par batch).

Encapsule la lecture des fichiers, le batching par budget de caracteres,
l'analyse statique et la synthese LLM en un seul appel de tool.
"""
from __future__ import annotations

from typing import Any

from agent.tools.base import Tool, ToolRegistry
from agent.models.local_model import LocalModel

# -- Constantes --
_DEFAULT_BATCH_CHARS_BUDGET = 60000
MAX_LINE_LENGTH = 120
_DEFAULT_MAX_FILE_LINES = 300
_DEFAULT_FILE_READ_TIMEOUT = 30
EXCLUDE_DIRS = ["llama.cpp", "llama-bin", "__pycache__", ".venv", "training", "node_modules"]
SEVERITY_WEIGHTS = {"critical": 20, "high": 10, "medium": 5, "low": 2, "info": 0}

_DEFAULT_COMPOUND_TIMEOUT = 180  # Repertoires = plusieurs batches LLM


class CodeReviewTool(Tool):
    """Analyse statique + LLM review d'un fichier ou repertoire Python."""

    name = "review_code"
    description = (
        "Analyse de code Python : analyse statique (bugs, style) + review LLM. "
        "Accepte un fichier ou un repertoire. Pour les repertoires, "
        "groupe les fichiers par batch pour gerer les gros projets."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Chemin du fichier ou repertoire a analyser",
            },
        },
        "required": ["path"],
    }

    def __init__(self, tools: ToolRegistry, model: LocalModel, config: dict | None = None) -> None:
        self._tools = tools
        self._model = model
        crcfg = (config or {}).get("tools", {}).get("code_review", {})
        self._batch_chars_budget: int = crcfg.get("batch_chars_budget", _DEFAULT_BATCH_CHARS_BUDGET)
        self._max_file_lines: int = crcfg.get("max_file_lines", _DEFAULT_MAX_FILE_LINES)
        self._file_read_timeout: int = crcfg.get("file_read_timeout", _DEFAULT_FILE_READ_TIMEOUT)
        self.timeout: int = crcfg.get("compound_timeout", _DEFAULT_COMPOUND_TIMEOUT)

    async def execute(self, path: str, **_: Any) -> str:
        """Analyse un fichier ou repertoire et retourne un rapport texte."""
        # Detecter si c'est un fichier ou un repertoire
        if path.endswith("/") or ("/" not in path and "." not in path):
            return await self._review_directory(path)

        # Verifier si c'est un fichier avec extension
        has_ext = any(path.endswith(ext) for ext in [".py", ".js", ".yaml", ".json", ".md", ".txt"])
        if has_ext:
            return await self._review_file(path)

        # Tenter comme repertoire par defaut
        return await self._review_directory(path)

    # -- Review d'un fichier unique --

    async def _review_file(self, file_path: str) -> str:
        """Analyse un fichier unique : statique + LLM."""
        content = await self._tools.execute(
            "file_system", exec_timeout=self._file_read_timeout, action="read", path=file_path
        )
        if content.startswith("[ERREUR]"):
            return content

        # Analyse statique
        issues = _static_analysis(content, self._max_file_lines)
        issues_text = _format_issues(issues)

        # Analyse LLM
        try:
            result = await self._model.chat_with_tools(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Review ce code Python. Identifie les bugs, "
                            "problemes de securite, et suggestions d'amelioration. "
                            "Sois concis et precis. En francais."
                        ),
                    },
                    {"role": "user", "content": f"Fichier: {file_path}\n\n{content}"},
                ],
                tools=[],
                tool_choice="none",
            )
            llm_review = result.get("content") or ""
        except Exception as e:
            llm_review = f"[Analyse LLM indisponible: {e}]"

        # Score
        score = _calculate_score(issues)

        # Formater la sortie
        parts = [f"## Review : {file_path} (score: {score:.0f}/100)\n"]
        if issues_text:
            parts.append(f"### Analyse statique\n{issues_text}")
        if llm_review:
            parts.append(f"### Analyse LLM\n{llm_review}")
        return "\n".join(parts)

    # -- Review d'un repertoire par batch --

    async def _review_directory(self, dir_path: str) -> str:
        """Analyse un repertoire complet par batch dynamique."""
        # Lister les fichiers Python
        files_result = await self._tools.execute(
            "file_system", exec_timeout=self._file_read_timeout, action="search", path=dir_path, pattern="*.py"
        )
        files = [
            f.strip() for f in files_result.strip().split("\n")
            if f.strip() and not any(ex in f for ex in EXCLUDE_DIRS)
        ]

        if not files:
            return f"Aucun fichier .py trouve dans {dir_path}"

        # Lire TOUS les fichiers complets
        all_files: list[dict] = []
        for filepath in files:
            try:
                content = await self._tools.execute(
                    "file_system", exec_timeout=self._file_read_timeout, action="read", path=filepath
                )
                all_files.append({"path": filepath, "content": content, "chars": len(content)})
            except Exception:
                all_files.append({"path": filepath, "content": "[ERREUR LECTURE]", "chars": 0})

        # Grouper en batches par budget de chars
        batches = _batch_by_chars(all_files, self._batch_chars_budget)

        # Analyser chaque batch avec le LLM
        batch_analyses: list[str] = []
        for i, batch in enumerate(batches):
            batch_text = "\n\n".join(
                f"=== {f['path']} ({f['chars']} chars) ===\n{f['content']}" for f in batch
            )
            user_content = f"Batch {i + 1}/{len(batches)} : {len(batch)} fichiers\n\n{batch_text}"

            try:
                result = await self._model.chat_with_tools(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Analyse ce lot de code source COMPLET. "
                                "Pour chaque fichier : bugs reels, qualite, structure. "
                                "2-3 lignes par fichier max. En francais."
                            ),
                        },
                        {"role": "user", "content": user_content},
                    ],
                    tools=[],
                    tool_choice="none",
                )
                batch_analyses.append(result.get("content") or "Analyse non disponible")
            except Exception as e:
                batch_analyses.append(f"Erreur batch {i + 1}: {e}")

        # Si un seul batch, retourner directement
        if len(batch_analyses) == 1:
            header = f"## Review : {dir_path} ({len(files)} fichiers)\n\n"
            return header + batch_analyses[0]

        # Plusieurs batches : synthetiser
        combined = "\n\n---\n\n".join(
            f"## Batch {i + 1}\n{a}" for i, a in enumerate(batch_analyses)
        )
        try:
            synth_result = await self._model.chat_with_tools(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Synthetise ces analyses de code en UN rapport final structure. "
                            "Regroupe par themes (bugs, architecture, qualite). En francais."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Projet: {dir_path}\n"
                            f"{len(files)} fichiers analyses en {len(batch_analyses)} lots.\n\n"
                            f"{combined}"
                        ),
                    },
                ],
                tools=[],
                tool_choice="none",
            )
            final = synth_result.get("content") or combined
        except Exception:
            final = combined

        header = f"## Review : {dir_path} ({len(files)} fichiers, {len(batches)} batches)\n\n"
        return header + final


# -- Fonctions utilitaires (hors classe) --

def _batch_by_chars(files: list[dict], budget: int) -> list[list[dict]]:
    """Groupe les fichiers en batches par budget de caracteres."""
    batches: list[list[dict]] = []
    current_batch: list[dict] = []
    current_chars = 0

    for f in files:
        if current_chars + f["chars"] > budget and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_chars = 0
        current_batch.append(f)
        current_chars += f["chars"]

    if current_batch:
        batches.append(current_batch)
    return batches


def _static_analysis(content: str, max_file_lines: int = 300) -> list[dict]:
    """Analyse statique du code Python -- checks utiles sans regex."""
    issues: list[dict] = []
    lines = content.split("\n")

    if len(lines) > max_file_lines:
        issues.append({
            "line": 1, "type": "style", "severity": "low",
            "message": f"Fichier long ({len(lines)} lignes, seuil {max_file_lines})",
        })

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        if len(line) > MAX_LINE_LENGTH:
            issues.append({
                "line": i, "type": "style", "severity": "low",
                "message": f"Ligne de {len(line)} chars (max {MAX_LINE_LENGTH})",
            })

        if stripped == "except:" or stripped == "except Exception:":
            next_stripped = lines[i].strip() if i < len(lines) else ""
            if next_stripped == "pass" or not next_stripped:
                issues.append({
                    "line": i, "type": "bug", "severity": "high",
                    "message": "Bare except avec pass silencieux",
                })

        if stripped.startswith("from ") and "import *" in stripped:
            issues.append({
                "line": i, "type": "style", "severity": "medium",
                "message": "Wildcard import (import *)",
            })

        if "open(" in stripped and "with " not in stripped and not stripped.startswith("#"):
            issues.append({
                "line": i, "type": "bug", "severity": "medium",
                "message": "open() sans with statement",
            })

    return issues


def _format_issues(issues: list[dict]) -> str:
    """Formate les issues en texte lisible."""
    if not issues:
        return ""
    lines = []
    for issue in issues:
        lines.append(f"  L{issue['line']} [{issue['severity']}] {issue['message']}")
    return "\n".join(lines)


def _calculate_score(issues: list[dict]) -> float:
    """Score de qualite base sur les issues trouvees."""
    weight = sum(SEVERITY_WEIGHTS.get(i["severity"], 0) for i in issues)
    return max(0.0, 100.0 - weight)
