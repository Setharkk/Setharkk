"""Router deterministe -- shortcuts sans LLM pour les cas evidents.

Seuls 3 cas sont routes directement (zero LLM) :
- DIRECT_RECALL : "rappelle toi", "memoire"
- DIRECT_FILE_READ : "lis le fichier X.py"
- DIRECT_FILE_LIST : "liste les fichiers"

Tout le reste -> SIMPLE_TOOL : le LLM decide avec tool calling natif.
"""

from __future__ import annotations

from enum import Enum


class IntentClass(Enum):
    DIRECT_RECALL = "direct_recall"
    DIRECT_FILE_READ = "direct_read"
    DIRECT_FILE_LIST = "direct_list"
    DIRECT_RESUME = "direct_resume"
    SIMPLE_TOOL = "simple_tool"


_RESUME_PATTERNS = [
    "continu", "continue", "reprends", "reprend", "on en etait ou",
    "ou on en etait", "resume", "la suite", "poursuit", "poursuis",
]

_RECALL_PATTERNS = [
    "dans ta memoire", "en base de donnee", "sessions passees",
    "dans tes souvenirs", "en base",
    "in your memory", "from memory", "past sessions", "in database",
    "do you remember", "recall",
]

_FILE_READ_PATTERNS = [
    "lis le fichier", "lire le fichier", "montre le fichier",
    "affiche le fichier", "contenu de", "contenu du fichier",
    "cat ", "ouvre le fichier",
    "read file", "read the file", "show file", "show the file",
    "open file", "content of", "display file",
]

_FILE_LIST_PATTERNS = [
    "liste les fichiers", "ls ", "quels fichiers", "arborescence",
    "structure du projet", "tree",
    "list files", "list the files", "which files",
    "directory structure", "project structure", "show files",
]

_UPDATE_WORDS = [
    "mets a jour", "met a jour", "ajoute", "modifie", "change",
    "update", "retiens", "souviens-toi", "note que",
    "remember that", "note that", "keep in mind",
]


class Router:
    """Shortcuts deterministes pour les cas evidents. Tout le reste -> LLM."""

    def classify(self, user_input: str) -> tuple[IntentClass, dict]:
        low = user_input.lower().strip()
        params: dict = {}

        # /commands explicites
        if low.startswith("/recall ") or low.startswith("/memory"):
            params["query"] = user_input.split(" ", 1)[-1] if " " in user_input else ""
            return IntentClass.DIRECT_RECALL, params

        # Resume de session precedente
        for p in _RESUME_PATTERNS:
            if low == p or low.startswith(p + " ") or low.startswith(p + ","):
                return IntentClass.DIRECT_RESUME, params

        if low.startswith("/tools") or low.startswith("/status") or low.startswith("/clear"):
            return IntentClass.SIMPLE_TOOL, params

        # Recall direct (mais PAS si c'est un update/remember)
        is_update = any(w in low for w in _UPDATE_WORDS)
        if not is_update:
            for p in _RECALL_PATTERNS:
                if p in low:
                    params["query"] = user_input
                    return IntentClass.DIRECT_RECALL, params

        # Lecture fichier direct (si chemin detecte)
        for p in _FILE_READ_PATTERNS:
            if p in low:
                path = _extract_path(user_input)
                if path:
                    params["path"] = path
                    return IntentClass.DIRECT_FILE_READ, params
                break

        # Listing fichiers direct
        for p in _FILE_LIST_PATTERNS:
            if p in low:
                path = _extract_path(user_input) or "."
                params["path"] = path
                return IntentClass.DIRECT_FILE_LIST, params

        # Tout le reste : le LLM decide
        return IntentClass.SIMPLE_TOOL, params


def _extract_path(text: str) -> str | None:
    """Extrait un chemin de fichier depuis le texte."""
    for word in text.split():
        if "/" in word or "\\" in word:
            return word.strip("'\"")
        if "." in word and any(word.endswith(ext) for ext in [
            ".py", ".yaml", ".yml", ".json", ".md", ".txt", ".js",
            ".html", ".css", ".toml", ".cfg", ".log", ".bat", ".sh",
        ]):
            return word.strip("'\"")
    return None
