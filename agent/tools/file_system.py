from __future__ import annotations

import os
from pathlib import Path

from agent.tools.base import Tool


class FileSystemTool(Tool):
    name = "file_system"
    description = "Operations sur le systeme de fichiers : lire, ecrire, lister, chercher des fichiers."
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "write", "list", "search", "replace"],
                "description": "Action a effectuer. 'replace' remplace old_text par content dans le fichier.",
            },
            "path": {"type": "string", "description": "Chemin du fichier ou dossier"},
            "content": {"type": "string", "description": "Contenu a ecrire (pour 'write') ou nouveau texte (pour 'replace')"},
            "old_text": {"type": "string", "description": "Texte a remplacer (pour action 'replace')"},
            "pattern": {"type": "string", "description": "Pattern de recherche (pour action 'search')"},
        },
        "required": ["action", "path"],
    }

    def __init__(self, config: dict | None = None) -> None:
        cfg = (config or {}).get("tools", {}).get("file_system", {})
        self._read_max_chars: int = cfg.get("read_max_chars", 50_000)
        self._list_max_entries: int = cfg.get("list_max_entries", 50)
        self._search_max_matches: int = cfg.get("search_max_matches", 30)

    async def execute(self, action: str, path: str, content: str = "", old_text: str = "", pattern: str = "", **_) -> str:
        p = Path(path).expanduser()

        if action == "read":
            if not p.is_file():
                return f"[ERREUR] Fichier introuvable : {p}"
            text = p.read_text(encoding="utf-8", errors="replace")
            if len(text) > self._read_max_chars:
                return text[:self._read_max_chars] + f"\n\n... (tronque, {len(text)} chars total)"
            return text

        if action == "write":
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Fichier ecrit : {p} ({len(content)} chars)"

        if action == "list":
            if not p.is_dir():
                return f"[ERREUR] Dossier introuvable : {p}"
            entries = sorted(p.iterdir())[:self._list_max_entries]
            lines = []
            for e in entries:
                kind = "DIR " if e.is_dir() else "FILE"
                size = e.stat().st_size if e.is_file() else 0
                lines.append(f"  {kind}  {e.name}  ({size} bytes)" if size else f"  {kind}  {e.name}")
            return f"Contenu de {p} ({len(entries)} entrees) :\n" + "\n".join(lines)

        if action == "search":
            if not p.is_dir():
                return f"[ERREUR] Dossier introuvable : {p}"
            matches = list(p.rglob(pattern or "*"))[:self._search_max_matches]
            if not matches:
                return f"Aucun fichier trouvé pour '{pattern}' dans {p}"
            return "\n".join(str(m) for m in matches)

        if action == "replace":
            if not p.is_file():
                return f"[ERREUR] Fichier introuvable : {p}"
            if not old_text:
                return "[ERREUR] old_text requis pour action 'replace'"
            text = p.read_text(encoding="utf-8", errors="replace")
            if old_text not in text:
                return f"[ERREUR] Texte a remplacer introuvable dans {p}"
            new_text = text.replace(old_text, content, 1)
            p.write_text(new_text, encoding="utf-8")
            return f"Remplacement effectue dans {p}"

        return f"[ERREUR] Action inconnue : {action}"
