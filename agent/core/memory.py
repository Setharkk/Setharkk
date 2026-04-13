"""Memoire hybride PostgreSQL + Neo4j.

- CoreMemory : PostgreSQL, petit, structure, toujours dans le system prompt
- RecallMemory : PostgreSQL, log conversation SQL, recherche par session/texte
- ArchivalMemory : Neo4j, knowledge long terme (vector search + knowledge graph)
- Memory : facade unifiee
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
import yaml
import asyncpg
from pathlib import Path
from fastembed import TextEmbedding
from neo4j import AsyncGraphDatabase

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "agent.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


class CoreMemory:
    """Petit, structure, toujours injecte dans le system prompt.

    Stocke l'identite de l'agent, le profil utilisateur, et les regles.
    Le LLM peut l'editer via un tool dedier.
    """

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def get_all(self) -> dict[str, dict[str, str]]:
        """Retourne toute la core memory groupee par section."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT section, key, value FROM core_memory ORDER BY section, key")
        result: dict[str, dict[str, str]] = {}
        for r in rows:
            result.setdefault(r["section"], {})[r["key"]] = r["value"]
        return result

    async def get_section(self, section: str) -> dict[str, str]:
        """Retourne une section specifique."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT key, value FROM core_memory WHERE section = $1 ORDER BY key", section
            )
        return {r["key"]: r["value"] for r in rows}

    async def set(self, section: str, key: str, value: str):
        """Cree ou met a jour une entree."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO core_memory (section, key, value)
                   VALUES ($1, $2, $3)
                   ON CONFLICT (section, key) DO UPDATE SET value = $3, updated_at = now()""",
                section, key, value,
            )

    async def delete(self, section: str, key: str):
        """Supprime une entree."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM core_memory WHERE section = $1 AND key = $2", section, key
            )

    async def to_prompt_block(self) -> str:
        """Formate la core memory en bloc texte pour le system prompt."""
        data = await self.get_all()
        if not data:
            return ""
        parts = []
        for section, entries in data.items():
            lines = [f"  {k}: {v}" for k, v in entries.items()]
            parts.append(f"[{section}]\n" + "\n".join(lines))
        return "\n\n".join(parts)


class RecallMemory:
    """Log de conversation SQL. Pas de vector -- recherche par session et texte."""

    def __init__(self, pool: asyncpg.Pool, max_per_session_search: int = 3) -> None:
        self._pool = pool
        self._session_id: str = ""
        self._max_per_session_search: int = max_per_session_search

    async def add_message(self, role: str, content: str, tool_name: str | None = None, tool_args: dict | None = None):
        """Ajoute un message a la conversation courante."""
        args_json = json.dumps(tool_args) if tool_args else None
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO recall_memory (session_id, role, content, tool_name, tool_args)
                   VALUES ($1, $2, $3, $4, $5::jsonb)""",
                self._session_id, role, content, tool_name, args_json,
            )

    async def add_tool_result(self, tool_call_id: str, tool_name: str, content: str) -> None:
        """Persiste un message role=tool avec son tool_call_id."""
        args_json = json.dumps({"tool_call_id": tool_call_id})
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO recall_memory (session_id, role, content, tool_name, tool_args)
                   VALUES ($1, $2, $3, $4, $5::jsonb)""",
                self._session_id, "tool", content, tool_name, args_json,
            )

    async def add_assistant_tool_calls(self, tool_calls: list[dict]) -> None:
        """Persiste un message assistant contenant des tool_calls."""
        # Serialiser les tool_calls dans tool_args pour reconstruction ulterieure
        args_json = json.dumps({"tool_calls": tool_calls})
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO recall_memory (session_id, role, content, tool_name, tool_args)
                   VALUES ($1, $2, $3, $4, $5::jsonb)""",
                self._session_id, "assistant", "", "tool_calling", args_json,
            )

    @staticmethod
    def _reconstruct_openai(rows: list) -> list[dict]:
        """Reconstruit les messages DB au format OpenAI pour le LLM.

        - assistant + tool_name="tool_calling" -> message avec tool_calls
        - tool + tool_args.tool_call_id -> message avec tool_call_id
        - autres -> message standard {role, content}
        """
        messages: list[dict] = []
        for r in rows:
            role = r["role"]
            content = r["content"] or ""
            tool_name = r["tool_name"]
            raw_args = r["tool_args"]
            tool_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args

            # Message assistant avec tool_calls serialises
            if role == "assistant" and tool_name == "tool_calling" and tool_args:
                stored_calls = tool_args.get("tool_calls", [])
                openai_calls = []
                for tc in stored_calls:
                    func = tc.get("function", {})
                    openai_calls.append({
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": func.get("name", ""),
                            "arguments": json.dumps(func.get("arguments", {})),
                        },
                    })
                msg = {"role": "assistant", "content": content or None, "tool_calls": openai_calls}
                messages.append(msg)
                continue

            # Message tool avec tool_call_id
            if role == "tool" and tool_args and "tool_call_id" in tool_args:
                msg = {
                    "role": "tool",
                    "tool_call_id": tool_args["tool_call_id"],
                    "content": content,
                }
                messages.append(msg)
                continue

            # Message standard (user, assistant sans tool_calls)
            messages.append({"role": role, "content": content})

        return messages

    async def get_session(self, limit: int = 50) -> list[dict]:
        """Recupere les messages de la session courante au format OpenAI."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT role, content, tool_name, tool_args
                   FROM recall_memory
                   WHERE session_id = $1
                   ORDER BY created_at
                   LIMIT $2""",
                self._session_id, limit,
            )
        return self._reconstruct_openai(rows)

    async def get_session_unlimited(self) -> list[dict]:
        """Recupere TOUS les messages au format OpenAI sans limite de taille.

        Le ContextManager gere la taille, pas la memoire.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT role, content, tool_name, tool_args
                   FROM recall_memory
                   WHERE session_id = $1
                   ORDER BY created_at""",
                self._session_id,
            )
        return self._reconstruct_openai(rows)

    async def search_text(self, query: str, limit: int = 20) -> list[dict]:
        """Recherche cross-session par texte. Max 3 resultats par session pour diversifier."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT session_id, role, content, created_at
                   FROM recall_memory
                   WHERE content ILIKE '%' || $1 || '%'
                   ORDER BY created_at DESC
                   LIMIT $2""",
                query, limit * 2,  # Fetcher plus pour compenser le dedup
            )

        # Dedup par session : max N messages par session_id
        per_session: dict[str, int] = {}
        results: list[dict] = []
        for r in rows:
            sid = r["session_id"]
            count = per_session.get(sid, 0)
            if count >= self._max_per_session_search:
                continue
            per_session[sid] = count + 1
            results.append(dict(r))
            if len(results) >= limit:
                break

        return results


class ArchivalMemory:
    """Memoire long terme Neo4j : vector search + knowledge graph unifie.

    Noeuds :Memory pour les faits archivaux (avec vector index 384d).
    Noeuds :Entity pour les entites extraites (personnes, projets, etc.).
    Relations :MENTIONS (Memory->Entity) et :RELATES (Entity->Entity).
    """

    # Poids par categorie pour le scoring d'importance
    _CATEGORY_WEIGHTS: dict[str, float] = {
        "error": 0.9, "decision": 0.8, "skill": 0.7,
        "preference": 0.6, "fact": 0.5, "summary": 0.4,
    }
    _ENTITY_TYPE_WEIGHTS: dict[str, float] = {
        "error": 0.9, "rule": 0.8, "person": 0.7, "project": 0.7,
        "technology": 0.6, "file": 0.5, "concept": 0.5,
    }
    _IMPORTANCE_KEYWORDS: list[str] = [
        "important", "critical", "decision", "error", "bug", "rule",
    ]

    def __init__(
        self,
        driver: AsyncGraphDatabase.driver,
        embedder: TextEmbedding,
        top_k: int = 15,
        scoring_config: dict | None = None,
    ):
        self._driver = driver
        self._embedder = embedder
        self.top_k = top_k
        self._logger = logging.getLogger("setharkk.archival")
        sc = scoring_config or {}
        self._w_similarity: float = sc.get("weight_similarity", 0.55)
        self._w_importance: float = sc.get("weight_importance", 0.15)
        self._w_recency: float = sc.get("weight_recency", 0.15)
        self._w_frequency: float = sc.get("weight_frequency", 0.15)
        self._freq_cap: int = sc.get("frequency_cap", 20)
        self._min_similarity: float = sc.get("min_similarity", 0.3)

    def _embed(self, text: str) -> list[float]:
        return list(self._embedder.embed([text]))[0].tolist()

    def _compute_importance(self, content: str, metadata: dict) -> float:
        """Score d'importance deterministe (0.0-1.0). Pas de LLM, pas de regex."""
        if content.startswith("[SUMMARY]") or content.startswith("[SESSION"):
            return 0.4
        category = metadata.get("category", metadata.get("type", ""))
        score = self._CATEGORY_WEIGHTS.get(category, 0.5)
        if len(content) < 20:
            score -= 0.1
        elif len(content) > 500:
            score += 0.05
        content_lower = content.lower()
        for kw in self._IMPORTANCE_KEYWORDS:
            if kw in content_lower:
                score += 0.1
                break
        return max(0.0, min(1.0, score))

    async def init_schema(self) -> None:
        """Cree les contraintes et index Neo4j (idempotent)."""
        queries = [
            "CREATE CONSTRAINT entity_unique IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name_lower, e.entity_type) IS UNIQUE",
            "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
        ]
        async with self._driver.session() as session:
            for q in queries:
                await session.run(q)
        # Vector index (syntaxe separee car peut echouer si deja existant)
        try:
            async with self._driver.session() as session:
                await session.run(
                    "CREATE VECTOR INDEX memory_embedding IF NOT EXISTS "
                    "FOR (m:Memory) ON (m.embedding) "
                    "OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}"
                )
        except Exception as exc:
            self._logger.debug("Vector index creation skipped: %s", exc)

    # ── Store / Recall / Delete (API identique a l'ancienne) ───────────

    async def store(self, content: str, metadata: dict | None = None, source_session_id: str = "") -> str:
        """Stocke un fait comme noeud :Memory. Retourne l'elementId du noeud cree."""
        meta = metadata or {}
        importance = self._compute_importance(content, meta)
        vec = self._embed(content)
        meta_json = json.dumps(meta)

        async def _tx(tx: object) -> str:
            result = await tx.run(
                "CREATE (m:Memory {"
                "  content: $content, embedding: $vec, metadata: $meta,"
                "  importance_score: $importance, source_session_id: $sid,"
                "  access_count: 0, last_accessed: datetime(), created_at: datetime()"
                "}) RETURN elementId(m) AS eid",
                content=content, vec=vec, meta=meta_json,
                importance=importance, sid=source_session_id,
            )
            record = await result.single()
            return record["eid"] if record else ""

        async with self._driver.session() as session:
            return await session.execute_write(_tx)

    async def recall(self, query: str, k: int | None = None, min_similarity: float | None = None) -> list[dict]:
        """Recherche par scoring composite : similarite + importance + recency + frequence."""
        k = k or self.top_k
        min_sim = min_similarity if min_similarity is not None else self._min_similarity
        vec = self._embed(query)

        async with self._driver.session() as session:
            # Vector search + scoring composite
            result = await session.run(
                "CALL db.index.vector.queryNodes('memory_embedding', $k_fetch, $vec) "
                "YIELD node, score AS similarity "
                "WHERE similarity > $min_sim "
                "WITH node, similarity, "
                "  ($w_sim * similarity "
                "   + $w_imp * COALESCE(node.importance_score, 0.5) "
                "   + $w_rec * (1.0 / (1.0 + duration.between(node.last_accessed, datetime()).seconds / 86400.0)) "
                "   + $w_freq * (toFloat(CASE WHEN node.access_count > $cap THEN $cap ELSE node.access_count END) / $cap) "
                "  ) AS final_score "
                "RETURN node.content AS content, node.metadata AS metadata, "
                "  similarity, final_score AS score, node.access_count AS access_count, "
                "  elementId(node) AS node_id "
                "ORDER BY final_score DESC LIMIT $k",
                vec=vec, k_fetch=k * 2, k=k, min_sim=min_sim,
                w_sim=self._w_similarity, w_imp=self._w_importance,
                w_rec=self._w_recency, w_freq=self._w_frequency,
                cap=float(self._freq_cap),
            )
            records = [record.data() async for record in result]

            # Tracker les acces
            if records:
                node_ids = [r["node_id"] for r in records]
                await session.run(
                    "UNWIND $ids AS nid "
                    "MATCH (m:Memory) WHERE elementId(m) = nid "
                    "SET m.access_count = m.access_count + 1, m.last_accessed = datetime()",
                    ids=node_ids,
                )

        return [
            {
                "content": r["content"],
                "metadata": json.loads(r["metadata"]) if isinstance(r["metadata"], str) else (r["metadata"] or {}),
                "similarity": float(r["similarity"]),
                "score": float(r["score"]),
                "access_count": int(r.get("access_count") or 0),
            }
            for r in records
        ]

    async def delete_by_similarity(self, query: str, threshold: float = 0.85) -> int:
        """Supprime les noeuds :Memory similaires au-dessus du seuil."""
        vec = self._embed(query)
        async with self._driver.session() as session:
            result = await session.run(
                "CALL db.index.vector.queryNodes('memory_embedding', 100, $vec) "
                "YIELD node, score "
                "WHERE score >= $threshold "
                "DETACH DELETE node "
                "RETURN count(*) AS deleted",
                vec=vec, threshold=threshold,
            )
            record = await result.single()
            return record["deleted"] if record else 0

    # ── Knowledge Graph (entites + relations) ──────────────────────────

    async def add_entity(
        self, name: str, entity_type: str, description: str = "",
        properties: dict | None = None, source_session_id: str = "",
    ) -> None:
        """Upsert un noeud :Entity. Sur conflit, garde la description la plus longue."""
        importance = self._ENTITY_TYPE_WEIGHTS.get(entity_type, 0.5)
        async with self._driver.session() as session:
            await session.run(
                "MERGE (e:Entity {name_lower: $nl, entity_type: $et}) "
                "ON CREATE SET e.name = $name, e.description = $desc, "
                "  e.importance = $imp, e.source_session_id = $sid, e.created_at = datetime() "
                "ON MATCH SET e.description = CASE WHEN size($desc) > size(COALESCE(e.description, '')) "
                "  THEN $desc ELSE e.description END, e.updated_at = datetime()",
                nl=name.lower(), et=entity_type, name=name, desc=description,
                imp=importance, sid=source_session_id,
            )

    async def add_relationship(
        self, source_name: str, source_type: str,
        target_name: str, target_type: str, rel_type: str,
        context: str = "", source_session_id: str = "",
    ) -> None:
        """Upsert une relation :RELATES. Sur conflit, incremente le poids."""
        async with self._driver.session() as session:
            await session.run(
                "MERGE (s:Entity {name_lower: $snl, entity_type: $st}) "
                "ON CREATE SET s.name = $sn, s.created_at = datetime() "
                "MERGE (t:Entity {name_lower: $tnl, entity_type: $tt}) "
                "ON CREATE SET t.name = $tn, t.created_at = datetime() "
                "MERGE (s)-[r:RELATES {type: $rt}]->(t) "
                "ON CREATE SET r.weight = 1.0, r.context = $ctx, "
                "  r.source_session_id = $sid, r.created_at = datetime() "
                "ON MATCH SET r.weight = r.weight + 1.0, r.context = $ctx, r.updated_at = datetime()",
                snl=source_name.lower(), st=source_type, sn=source_name,
                tnl=target_name.lower(), tt=target_type, tn=target_name,
                rt=rel_type, ctx=context, sid=source_session_id,
            )

    async def link_memory_to_entities(self, memory_element_id: str, entity_names: list[str]) -> None:
        """Lie un noeud :Memory aux :Entity mentionnees."""
        if not entity_names:
            return
        async with self._driver.session() as session:
            await session.run(
                "MATCH (m:Memory) WHERE elementId(m) = $mid "
                "UNWIND $names AS name "
                "MATCH (e:Entity) WHERE e.name_lower = toLower(name) "
                "MERGE (m)-[:MENTIONS]->(e)",
                mid=memory_element_id, names=entity_names,
            )

    async def search_entities(self, query: str, k: int = 5) -> list[dict]:
        """Recherche textuelle d'entites par nom ou description."""
        async with self._driver.session() as session:
            result = await session.run(
                "MATCH (e:Entity) "
                "WHERE toLower(e.name) CONTAINS toLower($q) "
                "   OR toLower(COALESCE(e.description, '')) CONTAINS toLower($q) "
                "RETURN e.name AS name, e.entity_type AS entity_type, "
                "  e.description AS description, e.importance AS importance "
                "ORDER BY e.importance DESC LIMIT $k",
                q=query, k=k,
            )
            return [record.data() async for record in result]

    async def get_neighbors(self, entity_name: str, depth: int = 1) -> list[dict]:
        """Traversee du graphe a partir d'une entite. Retourne les voisins."""
        async with self._driver.session() as session:
            result = await session.run(
                "MATCH (start:Entity {name_lower: $nl}) "
                "MATCH path = (start)-[:RELATES*1.." + str(depth) + "]-(neighbor:Entity) "
                "WITH DISTINCT neighbor, length(path) AS dist, "
                "  [r IN relationships(path) | r.type] AS rel_types, "
                "  [r IN relationships(path) | r.weight] AS weights "
                "RETURN neighbor.name AS name, neighbor.entity_type AS entity_type, "
                "  neighbor.description AS description, dist AS depth, "
                "  rel_types, weights "
                "ORDER BY dist, neighbor.importance DESC LIMIT 20",
                nl=entity_name.lower(),
            )
            return [record.data() async for record in result]

    async def get_subgraph(self, query: str, k: int = 3, depth: int = 1) -> dict:
        """Recherche entites + traversee + faits lies. Retourne un sous-graphe complet."""
        seeds = await self.search_entities(query, k)
        all_entities: list[dict] = list(seeds)
        all_relationships: list[dict] = []
        all_memories: list[dict] = []
        seen_names: set[str] = {e["name"] for e in seeds}

        for seed in seeds:
            neighbors = await self.get_neighbors(seed["name"], depth)
            for n in neighbors:
                if n["name"] not in seen_names:
                    seen_names.add(n["name"])
                    all_entities.append(n)
                all_relationships.append({
                    "source": seed["name"],
                    "target": n["name"],
                    "types": n.get("rel_types", []),
                    "weights": n.get("weights", []),
                })

        # Fetch les :Memory lies aux entites trouvees
        if seen_names:
            async with self._driver.session() as session:
                result = await session.run(
                    "UNWIND $names AS name "
                    "MATCH (e:Entity {name_lower: toLower(name)})<-[:MENTIONS]-(m:Memory) "
                    "RETURN DISTINCT m.content AS content, m.importance_score AS importance "
                    "ORDER BY m.importance_score DESC LIMIT 10",
                    names=list(seen_names),
                )
                all_memories = [record.data() async for record in result]

        return {"entities": all_entities, "relationships": all_relationships, "memories": all_memories}


class SessionExtractor:
    """Extraction hybride : deterministe + LLM (knowledge graph).

    1. Deterministe : scanne recall_memory, extrait Q/R, erreurs, fichiers, recherches
    2. LLM (si disponible) : appelle Qwen 3.5 9B via chat_with_tools pour extraire
       entites et relations vers le knowledge graph Neo4j
    """

    _DEFAULT_DEDUP_THRESHOLD: float = 0.85
    _DEFAULT_MAX_ENTITIES: int = 5
    _DEFAULT_MIN_ENTITY_NAME: int = 2
    _DEFAULT_MAX_ENTITY_NAME: int = 100
    _DEFAULT_MIN_FACT_LENGTH: int = 30

    _EXTRACT_TOOL: dict = {
        "type": "function",
        "function": {
            "name": "extract_knowledge",
            "description": "Extract entities and relationships from conversation facts",
            "parameters": {
                "type": "object",
                "properties": {
                    "entities": {"type": "array", "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string",
                                     "enum": ["person", "project", "technology", "concept", "file", "error", "rule"]},
                            "description": {"type": "string"},
                        }, "required": ["name", "type"],
                    }},
                    "relationships": {"type": "array", "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "target": {"type": "string"},
                            "type": {"type": "string",
                                     "enum": ["uses", "creates", "depends_on", "causes", "solves", "contains", "relates_to"]},
                            "description": {"type": "string"},
                        }, "required": ["source", "target", "type"],
                    }},
                },
                "required": ["entities", "relationships"],
            },
        },
    }

    def __init__(
        self,
        pool: asyncpg.Pool,
        archival: ArchivalMemory,
        model: object | None = None,
        extractor_config: dict | None = None,
    ) -> None:
        self._pool = pool
        self._archival = archival
        self._model = model
        self._logger = logging.getLogger("setharkk.extractor")
        ecfg = extractor_config or {}
        self._dedup_threshold: float = ecfg.get("dedup_threshold", self._DEFAULT_DEDUP_THRESHOLD)
        self._max_entities: int = ecfg.get("max_entities_per_session", self._DEFAULT_MAX_ENTITIES)
        self._min_entity_name: int = ecfg.get("min_entity_name", self._DEFAULT_MIN_ENTITY_NAME)
        self._max_entity_name: int = ecfg.get("max_entity_name", self._DEFAULT_MAX_ENTITY_NAME)
        self._min_fact_length: int = ecfg.get("min_fact_length", self._DEFAULT_MIN_FACT_LENGTH)

    async def extract_and_save(self, session_id: str) -> int:
        """Extrait les faits importants d'une session et les stocke en archival.

        Retourne le nombre de faits sauvegardes.
        """
        if not session_id:
            return 0

        try:
            return await self._do_extract(session_id)
        except Exception as exc:
            self._logger.warning("SessionExtractor failed for session %s: %s", session_id, exc)
            return 0

    async def _do_extract(self, session_id: str) -> int:
        """Logique interne d'extraction. Peut lever des exceptions."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT role, content, tool_name, tool_args, created_at
                   FROM recall_memory
                   WHERE session_id = $1
                   ORDER BY created_at""",
                session_id,
            )

        if not rows:
            return 0

        facts: list[tuple[str, dict]] = []  # (content, metadata)
        user_messages: list[str] = []
        user_count = 0

        i = 0
        while i < len(rows):
            r = rows[i]
            role = r["role"]
            content = r["content"] or ""
            tool_name = r["tool_name"] or ""
            raw_args = r["tool_args"]
            tool_args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})

            # Skip les appels remember et core_memory_update (deja persistes)
            if role == "assistant" and tool_name == "tool_calling" and tool_args:
                stored_calls = tool_args.get("tool_calls", [])
                for tc in stored_calls:
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "")
                    fn_args = fn.get("arguments", {})

                    if fn_name in ("remember", "core_memory_update"):
                        pass  # Skip, deja persiste
                    elif fn_name == "file_system":
                        action = fn_args.get("action", "")
                        path = fn_args.get("path", "")
                        if action in ("write", "replace") and path:
                            facts.append((
                                f"Fichier modifie: {path}",
                                {"type": "auto_extract", "category": "skill"},
                            ))
                    elif fn_name in ("search", "research"):
                        query = fn_args.get("query", fn_args.get("topic", ""))
                        if query:
                            facts.append((
                                f"Recherche: {query[:100]}",
                                {"type": "auto_extract", "category": "fact"},
                            ))
                i += 1
                continue

            # Erreurs tools
            if role == "tool" and "[ERREUR]" in content:
                facts.append((
                    f"Erreur {tool_name}: {content[:200]}",
                    {"type": "auto_extract", "category": "error"},
                ))
                i += 1
                continue

            # Paires Q/R substantielles
            if role == "user" and content:
                user_count += 1
                user_messages.append(content[:150])
                # Chercher la prochaine reponse assistant
                j = i + 1
                while j < len(rows):
                    if rows[j]["role"] == "assistant" and rows[j]["tool_name"] != "tool_calling":
                        a_content = rows[j]["content"] or ""
                        if len(a_content) > 100:
                            facts.append((
                                f"Q: {content[:150]} -> R: {a_content[:300]}",
                                {"type": "auto_extract", "category": "fact"},
                            ))
                        break
                    elif rows[j]["role"] == "user":
                        break  # Pas de reponse trouvee
                    j += 1

            i += 1

        # Resume de session si > 4 messages user
        if user_count > 4:
            summary = self._build_session_summary(rows)
            if summary:
                facts.append((
                    f"[SESSION {session_id}] {summary}",
                    {"type": "session_summary"},
                ))

        # Validation + dedup + stockage
        saved = 0
        for fact_content, fact_meta in facts:
            # Validation : fait trop court = bruit
            if len(fact_content) < self._min_fact_length:
                continue
            # Validation : path de fichier seul sans contexte
            if fact_content.startswith("Fichier modifie:") and len(fact_content) < 50:
                continue
            if await self._is_duplicate(fact_content):
                continue
            await self._archival.store(fact_content, fact_meta, source_session_id=session_id)
            saved += 1

        if saved > 0:
            self._logger.info("Session %s: %d facts auto-saved", session_id, saved)

        # LLM knowledge graph extraction (si model disponible)
        if self._model and facts:
            try:
                kg_saved = await asyncio.wait_for(
                    self._extract_knowledge_llm(session_id, facts), timeout=30.0,
                )
                if kg_saved > 0:
                    self._logger.info("Session %s: %d KG items extracted", session_id, kg_saved)
            except asyncio.TimeoutError:
                self._logger.warning("KG extraction timeout for session %s", session_id)
            except Exception as exc:
                self._logger.warning("KG extraction failed for session %s: %s", session_id, exc)

        return saved

    async def _extract_knowledge_llm(self, session_id: str, facts: list[tuple[str, dict]]) -> int:
        """Appelle le LLM pour extraire entites et relations des faits de session."""
        # Construire le prompt avec les 15 premiers faits
        fact_lines = [f"- {fc[:200]}" for fc, _ in facts[:15]]
        messages = [
            {"role": "system", "content": (
                "You extract ONLY named entities and their relationships from conversation facts.\n"
                "STRICT RULES:\n"
                "- Entity names MUST be proper nouns or specific named things (Setharkk, FastAPI, PostgreSQL, Samir)\n"
                "- NEVER extract generic words (code, file, error, function, variable, project, work, continuation)\n"
                "- NEVER extract tool names or parameter names (recall, search, query, timeout)\n"
                "- NEVER extract Python variable names or function names from code snippets\n"
                "- If a fact is about code (variables, functions, paths), extract the TECHNOLOGY name, not the code itself\n"
                f"- Maximum {self._max_entities} entities per session. Quality over quantity.\n"
                "- Only create relationships between entities you extracted. No orphan references.\n"
                "Call extract_knowledge with your findings. If nothing meaningful, call it with empty arrays."
            )},
            {"role": "user", "content": "Session facts:\n" + "\n".join(fact_lines)},
        ]

        result = await self._model.chat_with_tools(messages, [self._EXTRACT_TOOL], tool_choice="auto")
        if not result.get("tool_calls"):
            return 0

        tc = result["tool_calls"][0]
        args = tc["function"]["arguments"]
        entities = args.get("entities", [])
        relationships = args.get("relationships", [])

        # Stocker les entites (avec filtre anti-bruit)
        type_lookup: dict[str, str] = {}
        for ent in entities:
            name = ent.get("name", "").strip()
            etype = ent.get("type", "concept")
            desc = ent.get("description", "")
            if not name or len(name) < self._min_entity_name or len(name) > self._max_entity_name:
                continue
            # Rejeter les noms generiques / noms de tools / snake_case variables
            if "_" in name and name == name.lower():
                continue  # snake_case = variable Python, pas une entite
            if name.lower() in (
                "code", "file", "error", "function", "variable", "project",
                "work", "continuation", "recall", "search", "query", "timeout",
                "result", "response", "user", "assistant", "system", "tool",
                "memory", "session", "context", "data", "output", "input",
            ):
                continue
            # Concept sans description = faible valeur, probablement du bruit
            if not desc and etype == "concept":
                continue
            await self._archival.add_entity(name, etype, desc, source_session_id=session_id)
            type_lookup[name.lower()] = etype

        # Stocker les relations
        for rel in relationships:
            src = rel.get("source", "")
            tgt = rel.get("target", "")
            rtype = rel.get("type", "relates_to")
            rdesc = rel.get("description", "")
            if src and tgt:
                src_type = type_lookup.get(src.lower(), "concept")
                tgt_type = type_lookup.get(tgt.lower(), "concept")
                await self._archival.add_relationship(
                    src, src_type, tgt, tgt_type, rtype, rdesc, source_session_id=session_id,
                )

        return len(entities) + len(relationships)

    async def _is_duplicate(self, content: str) -> bool:
        """Verifie si un fait similaire existe deja en archival."""
        try:
            results = await self._archival.recall(content, k=1, min_similarity=self._dedup_threshold)
            if results and results[0].get("similarity", 0) >= self._dedup_threshold:
                return True
        except Exception as exc:
            self._logger.debug("Dedup check failed, storing anyway: %s", exc)
        return False

    @staticmethod
    def _build_session_summary(rows: list) -> str:
        """Resume deterministe d'une session (meme logique que ContextManager)."""
        user_msgs: list[str] = []
        tool_calls: list[str] = []
        assistant_actions: list[str] = []

        for r in rows:
            role = r["role"]
            content = (r["content"] or "")[:150]
            if role == "user" and content:
                user_msgs.append(content)
            elif role == "assistant" and content and r["tool_name"] != "tool_calling":
                assistant_actions.append(content)
            elif role == "tool":
                tool_calls.append(r["tool_name"] or "?")

        parts: list[str] = []
        if user_msgs:
            parts.append("OBJECTIF: " + " | ".join(user_msgs[-5:]))
        if tool_calls:
            parts.append("ACTIONS: " + ", ".join(tool_calls[-10:]))
        if assistant_actions:
            parts.append("PROGRES: " + " | ".join(assistant_actions[-3:]))

        return "\n".join(parts) if parts else ""


class Memory:
    """Facade unifiee : PostgreSQL (core + recall) + Neo4j (archival + KG).

    Garde une API backward-compatible pour le code existant.
    """

    def __init__(self, config: dict | None = None, model: object | None = None):
        cfg = config or _load_config()
        mem_cfg = cfg["memory"]
        self.top_k = mem_cfg.get("long_term_top_k", 15)
        self.pg_cfg = mem_cfg["postgres"]
        self._neo4j_cfg: dict = mem_cfg.get("neo4j", {})
        self._scoring_cfg: dict = mem_cfg.get("scoring", {})
        self._extractor_cfg: dict = mem_cfg.get("extractor", {})
        self._model = model
        self._pool: asyncpg.Pool | None = None
        self._neo4j_driver = None
        self._embedder = TextEmbedding(
            model_name=mem_cfg["embedding_model"],
            cache_dir=mem_cfg["embedding_cache_dir"],
        )

        # Tiers (initialises dans connect)
        self.core: CoreMemory | None = None
        self.recall: RecallMemory | None = None
        self.archival: ArchivalMemory | None = None
        self._extractor: SessionExtractor | None = None

    async def connect(self) -> None:
        # PostgreSQL (core + recall)
        self._pool = await asyncpg.create_pool(
            host=self.pg_cfg["host"],
            port=self.pg_cfg["port"],
            database=self.pg_cfg["database"],
            user=self.pg_cfg["user"],
            password=self.pg_cfg["password"],
            min_size=1,
            max_size=3,
        )
        self.core = CoreMemory(self._pool)
        max_per_session = self._extractor_cfg.get("max_per_session_search", 3)
        self.recall = RecallMemory(self._pool, max_per_session_search=max_per_session)
        self.recall._session_id = str(uuid.uuid4())[:8]

        # Neo4j (archival + KG)
        neo_cfg = self._neo4j_cfg
        self._neo4j_driver = AsyncGraphDatabase.driver(
            neo_cfg.get("uri", "bolt://127.0.0.1:7687"),
            auth=(neo_cfg.get("user", "neo4j"), neo_cfg.get("password", "")),
        )
        self.archival = ArchivalMemory(self._neo4j_driver, self._embedder, self.top_k, self._scoring_cfg)
        await self.archival.init_schema()
        self._extractor = SessionExtractor(self._pool, self.archival, self._model, self._extractor_cfg)

    async def init_table(self) -> None:
        """Cree les tables PostgreSQL (core + recall seulement)."""
        async with self._pool.acquire() as conn:
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

    # ── Backward-compatible API ──────────────────────────────────────

    async def add_message(self, role: str, content: str):
        """Ajoute un message (ecrit dans recall tier)."""
        await self.recall.add_message(role, content)

    async def get_messages(self) -> list[dict]:
        """Recupere les messages de session (lit recall tier)."""
        return await self.recall.get_session()

    async def add_tool_result(self, tool_call_id: str, tool_name: str, content: str) -> None:
        """Persiste un resultat de tool dans recall tier."""
        await self.recall.add_tool_result(tool_call_id, tool_name, content)

    async def add_assistant_tool_calls(self, tool_calls: list[dict]) -> None:
        """Persiste un message assistant avec tool_calls dans recall tier."""
        await self.recall.add_assistant_tool_calls(tool_calls)

    async def store(self, content: str, metadata: dict | None = None, source_session_id: str = "") -> str:
        """Stocke dans archival tier. Retourne l'elementId Neo4j."""
        return await self.archival.store(content, metadata, source_session_id)

    async def recall_archival(self, query: str, k: int | None = None) -> list[dict]:
        """Recherche dans archival tier."""
        return await self.archival.recall(query, k)

    async def delete_by_similarity(self, query: str, threshold: float = 0.85) -> int:
        """Supprime les souvenirs similaires dans archival tier."""
        return await self.archival.delete_by_similarity(query, threshold)

    # Ancien nom pour backward compat (agent.py l'utilise)
    async def recall_memories(self, query: str, k: int | None = None) -> list[dict]:
        return await self.recall_archival(query, k)

    async def get_core_prompt(self) -> str:
        """Retourne la core memory formatee pour le system prompt."""
        return await self.core.to_prompt_block()

    async def new_session(self) -> None:
        """Demarre une nouvelle session. Auto-save de la session courante avant reset."""
        if self._extractor and self.recall and self.recall._session_id:
            await self._extractor.extract_and_save(self.recall._session_id)
        self.recall._session_id = str(uuid.uuid4())[:8]

    async def recall_with_graph(self, query: str, k: int = 5, graph_depth: int = 1) -> dict:
        """Combine vector search (archival) + graph traversal (KG) en un seul appel."""
        memories = await self.archival.recall(query, k)
        subgraph = await self.archival.get_subgraph(query, k=3, depth=graph_depth)
        return {
            "memories": memories,
            "entities": subgraph.get("entities", []),
            "relationships": subgraph.get("relationships", []),
            "graph_memories": subgraph.get("memories", []),
        }

    async def close(self) -> None:
        """Ferme les connexions. Auto-save de la session courante avant fermeture."""
        if self._extractor and self.recall and self.recall._session_id:
            await self._extractor.extract_and_save(self.recall._session_id)
        if self._neo4j_driver:
            await self._neo4j_driver.close()
        if self._pool:
            await self._pool.close()

    @property
    def _session_id(self) -> str:
        return self.recall._session_id if self.recall else ""
