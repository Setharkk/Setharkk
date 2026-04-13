"""Nettoie les donnees corrompues dans pgvector.

Cible : les entries qui contiennent des resultats de recherche character-by-character
(le bug de _normalize_search_results). Ces entries ont des snippets d'un seul caractere
stockes comme des objets JSON individuels, ce qui donne des contenus de 100K+ chars
qui polluent le recall et font exploser le contexte du LLM.

Usage : python scripts/cleanup_pgvector.py [--dry-run]
"""

import asyncio
import sys
import asyncpg
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "agent.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


async def main():
    dry_run = "--dry-run" in sys.argv
    cfg = load_config()["memory"]["postgres"]

    pool = await asyncpg.create_pool(
        host=cfg["host"],
        port=cfg["port"],
        database=cfg["database"],
        user=cfg["user"],
        password=cfg["password"],
    )

    async with pool.acquire() as conn:
        # Stats avant nettoyage
        total = await conn.fetchval("SELECT COUNT(*) FROM agent_memory")
        print(f"Total entries en base : {total}")

        # 1. Trouver les entries avec snippets character-by-character
        # Pattern : contiennent "'snippet': '" suivi d'un seul caractere puis "'"
        # Ces entries ont des centaines de ces patterns
        char_by_char = await conn.fetch("""
            SELECT id, LENGTH(content) as len,
                   LEFT(content, 100) as preview,
                   metadata->>'type' as type
            FROM agent_memory
            WHERE content LIKE '%''snippet'': ''_''%'
              AND LENGTH(content) > 10000
            ORDER BY id
        """)
        print(f"\nEntries avec snippets char-by-char (>10K chars) : {len(char_by_char)}")

        for row in char_by_char:
            print(f"  id={row['id']} len={row['len']:,} type={row['type']} preview={row['preview'][:60]}...")

        # 2. Trouver les entries anormalement longues (>20K chars)
        huge = await conn.fetch("""
            SELECT id, LENGTH(content) as len,
                   LEFT(content, 100) as preview,
                   metadata->>'type' as type
            FROM agent_memory
            WHERE LENGTH(content) > 20000
              AND id NOT IN (SELECT id FROM agent_memory WHERE content LIKE '%''snippet'': ''_''%' AND LENGTH(content) > 10000)
            ORDER BY LENGTH(content) DESC
        """)
        print(f"\nAutres entries >20K chars : {len(huge)}")
        for row in huge:
            print(f"  id={row['id']} len={row['len']:,} type={row['type']} preview={row['preview'][:60]}...")

        # 3. Supprimer les entries corrompues
        if char_by_char:
            ids_to_delete = [row["id"] for row in char_by_char]
            if dry_run:
                print(f"\n[DRY RUN] Aurait supprime {len(ids_to_delete)} entries corrompues")
            else:
                deleted = await conn.execute(
                    "DELETE FROM agent_memory WHERE id = ANY($1::int[])",
                    ids_to_delete,
                )
                print(f"\nSupprime : {deleted}")

        # 4. Supprimer les entries >50K chars (probablement des hooks avec char-by-char dans les resultats)
        if not dry_run:
            deleted_huge = await conn.execute(
                "DELETE FROM agent_memory WHERE LENGTH(content) > 50000"
            )
            print(f"Supprime entries >50K chars : {deleted_huge}")

        # Stats apres nettoyage
        remaining = await conn.fetchval("SELECT COUNT(*) FROM agent_memory")
        print(f"\nEntries restantes : {remaining}")

        # Taille moyenne
        avg_len = await conn.fetchval("SELECT AVG(LENGTH(content))::int FROM agent_memory")
        max_len = await conn.fetchval("SELECT MAX(LENGTH(content)) FROM agent_memory")
        print(f"Taille moyenne : {avg_len:,} chars")
        print(f"Taille max : {max_len:,} chars")

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
