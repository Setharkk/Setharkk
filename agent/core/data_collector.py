"""Collecte automatique des interactions réussies pour le fine-tuning LoRA."""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

DATA_DIR = Path(__file__).resolve().parents[2] / "training" / "data"
DATASET_FILE = DATA_DIR / "interactions.jsonl"


class DataCollector:
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def save_interaction(self, system_prompt: str, messages: list[dict], success: bool = True, kind: str = "response"):
        """Sauvegarde une interaction complète au format ChatML pour fine-tuning.

        kind: "response", "tool_call", "free_text", "unknown"
        """
        if not success:
            return

        # Construire l'exemple d'entraînement
        training_messages = [{"role": "system", "content": system_prompt}]

        for msg in messages:
            training_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

        entry = {
            "messages": training_messages,
            "kind": kind,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with open(DATASET_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def count(self) -> int:
        if not DATASET_FILE.exists():
            return 0
        with open(DATASET_FILE, encoding="utf-8") as f:
            return sum(1 for _ in f)
