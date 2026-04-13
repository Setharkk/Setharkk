"""Initialise la table agent_memory dans PostgreSQL + pgvector."""
import asyncio
import asyncpg


async def main():
    conn = await asyncpg.connect(
        host="127.0.0.1",
        port=5434,
        database="pieuvre",
        user="pieuvre",
        password="pieuvre2026",
    )
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_memory (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            embedding vector(384),
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT now()
        )
    """)
    count = await conn.fetchval("SELECT count(*) FROM agent_memory")
    print(f"Table agent_memory prete. {count} entrees existantes.")
    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
