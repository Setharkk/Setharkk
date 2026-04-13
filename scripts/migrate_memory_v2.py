"""Migration memoire v2 : flat pgvector -> 3 tiers (core + recall + archival).

- Renomme agent_memory -> archival_memory (zero perte)
- Cree core_memory (section/key/value, petit, toujours en contexte)
- Cree recall_memory (conversation log SQL, pas de vector)
- Seed core_memory depuis les knowledge existantes

Usage : python scripts/migrate_memory_v2.py
Rollback : python scripts/migrate_memory_v2.py --rollback
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


async def migrate():
    cfg = load_config()["memory"]["postgres"]
    pool = await asyncpg.create_pool(
        host=cfg["host"], port=cfg["port"],
        database=cfg["database"], user=cfg["user"], password=cfg["password"],
    )

    async with pool.acquire() as conn:
        # Verifier si deja migre
        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'archival_memory')"
        )
        if exists:
            print("Migration deja appliquee (archival_memory existe).")
            await pool.close()
            return

        print("=== Migration memoire v2 ===")

        # 1. Renommer agent_memory -> archival_memory
        await conn.execute("ALTER TABLE agent_memory RENAME TO archival_memory")
        print("[OK] agent_memory -> archival_memory")

        # 2. Creer core_memory
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS core_memory (
                id SERIAL PRIMARY KEY,
                section VARCHAR(50) NOT NULL,
                key VARCHAR(100) NOT NULL,
                value TEXT NOT NULL,
                updated_at TIMESTAMPTZ DEFAULT now(),
                UNIQUE(section, key)
            )
        """)
        print("[OK] core_memory cree")

        # 3. Creer recall_memory
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS recall_memory (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(16) NOT NULL,
                role VARCHAR(20) NOT NULL,
                content TEXT NOT NULL,
                tool_name VARCHAR(50),
                tool_args JSONB,
                created_at TIMESTAMPTZ DEFAULT now()
            )
        """)
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_recall_session ON recall_memory(session_id, created_at)"
        )
        print("[OK] recall_memory cree + index session")

        # 4. Seed core_memory depuis les knowledge existantes dans archival
        knowledge = await conn.fetch(
            "SELECT content FROM archival_memory WHERE metadata->>'type' = 'knowledge'"
        )
        print(f"[INFO] {len(knowledge)} knowledge entries trouvees")

        # Parser les knowledge pour extraire les infos d'identite
        seeds = [
            ("persona", "name", "Setharkk"),
            ("persona", "nature", "Agent IA autonome, local, independant"),
            ("persona", "mission", "Aider son createur Samir Ben Jaddi a atteindre l'AGI"),
            ("persona", "language", "Francais par defaut"),
            ("user_profile", "name", "Samir Ben Jaddi"),
            ("user_profile", "role", "Createur de Setharkk"),
            ("system_rules", "no_regex", "Jamais import re - utiliser methodes string"),
            ("system_rules", "no_unicode", "ASCII uniquement dans print et code"),
            ("system_rules", "file_replace", "Toujours replace, jamais reecrire un fichier entier"),
            ("system_rules", "ask_before_edit", "Toujours demander confirmation avant modifier un fichier"),
            ("system_rules", "code_analysis", "Pour analyser du code, utiliser file_system read, jamais recall"),
        ]

        for section, key, value in seeds:
            await conn.execute(
                """INSERT INTO core_memory (section, key, value)
                   VALUES ($1, $2, $3)
                   ON CONFLICT (section, key) DO UPDATE SET value = $3, updated_at = now()""",
                section, key, value,
            )
        print(f"[OK] {len(seeds)} entries core_memory seedees")

        # Stats
        archival_count = await conn.fetchval("SELECT COUNT(*) FROM archival_memory")
        core_count = await conn.fetchval("SELECT COUNT(*) FROM core_memory")
        print(f"\n=== Resultat ===")
        print(f"  archival_memory : {archival_count} entries (ex agent_memory)")
        print(f"  core_memory     : {core_count} entries")
        print(f"  recall_memory   : 0 entries (nouveau)")
        print("Migration terminee.")

    await pool.close()


async def rollback():
    cfg = load_config()["memory"]["postgres"]
    pool = await asyncpg.create_pool(
        host=cfg["host"], port=cfg["port"],
        database=cfg["database"], user=cfg["user"], password=cfg["password"],
    )

    async with pool.acquire() as conn:
        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'archival_memory')"
        )
        if not exists:
            print("Rien a rollback (archival_memory n'existe pas).")
            await pool.close()
            return

        await conn.execute("ALTER TABLE archival_memory RENAME TO agent_memory")
        await conn.execute("DROP TABLE IF EXISTS core_memory")
        await conn.execute("DROP TABLE IF EXISTS recall_memory")
        print("Rollback termine : agent_memory restaure, core/recall supprimes.")

    await pool.close()


if __name__ == "__main__":
    if "--rollback" in sys.argv:
        asyncio.run(rollback())
    else:
        asyncio.run(migrate())
