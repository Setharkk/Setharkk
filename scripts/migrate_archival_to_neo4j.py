"""Migration des donnees archival_memory de PostgreSQL vers Neo4j.

Usage:
    python scripts/migrate_archival_to_neo4j.py
    python scripts/migrate_archival_to_neo4j.py --dry-run
"""

from __future__ import annotations

import asyncio
import json
import sys
import yaml
import asyncpg
from pathlib import Path
from neo4j import AsyncGraphDatabase

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "agent.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


async def migrate(dry_run: bool = False) -> None:
    cfg = _load_config()
    mem = cfg["memory"]
    pg = mem["postgres"]
    neo = mem.get("neo4j", {})

    # Connect PostgreSQL
    pool = await asyncpg.create_pool(
        host=pg["host"], port=pg["port"], database=pg["database"],
        user=pg["user"], password=pg["password"], min_size=1, max_size=2,
    )

    # Connect Neo4j
    driver = AsyncGraphDatabase.driver(
        neo.get("uri", "bolt://127.0.0.1:7687"),
        auth=(neo.get("user", "neo4j"), neo.get("password", "")),
    )

    # Check Neo4j connection
    async with driver.session() as session:
        await session.run("RETURN 1")
    print("[OK] Neo4j connected")

    # Fetch all archival_memory rows
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT content, embedding, metadata, importance_score, access_count, "
            "last_accessed, source_session_id, created_at FROM archival_memory ORDER BY created_at"
        )
    print(f"[OK] Found {len(rows)} rows in archival_memory")

    if dry_run:
        for r in rows[:5]:
            print(f"  [{r['importance_score']:.1f}] {r['content'][:80]}...")
        if len(rows) > 5:
            print(f"  ... and {len(rows) - 5} more")
        print("[DRY RUN] No changes made")
        await pool.close()
        await driver.close()
        return

    # Migrate each row to Neo4j
    migrated = 0
    async with driver.session() as session:
        for r in rows:
            content = r["content"]
            meta_raw = r["metadata"]
            metadata = json.loads(meta_raw) if isinstance(meta_raw, str) else (meta_raw or {})
            importance = float(r["importance_score"]) if r["importance_score"] is not None else 0.5
            access_count = int(r["access_count"]) if r["access_count"] is not None else 0
            source_sid = r["source_session_id"] or ""

            # Convert pgvector embedding to list of floats
            emb_raw = r["embedding"]
            if emb_raw is None:
                print(f"  [SKIP] No embedding for: {content[:50]}...")
                continue

            # pgvector returns a string like "[0.1,0.2,...]" or a native type
            if isinstance(emb_raw, str):
                embedding = [float(x) for x in emb_raw.strip("[]").split(",")]
            elif hasattr(emb_raw, "__iter__"):
                embedding = [float(x) for x in emb_raw]
            else:
                print(f"  [SKIP] Unknown embedding type {type(emb_raw)} for: {content[:50]}...")
                continue

            await session.run(
                "CREATE (m:Memory {"
                "  content: $content, embedding: $vec, metadata: $meta,"
                "  importance_score: $imp, source_session_id: $sid,"
                "  access_count: $ac, last_accessed: datetime(), created_at: datetime()"
                "})",
                content=content, vec=embedding, meta=json.dumps(metadata),
                imp=importance, sid=source_sid, ac=access_count,
            )
            migrated += 1

    print(f"[OK] Migrated {migrated}/{len(rows)} memories to Neo4j")
    print("[INFO] PostgreSQL archival_memory table NOT dropped (kept as backup)")

    await pool.close()
    await driver.close()


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    asyncio.run(migrate(dry_run=dry))
