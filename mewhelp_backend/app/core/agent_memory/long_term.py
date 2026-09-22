from __future__ import annotations

import json
from typing import Any

from sqlalchemy import create_engine, text

from .repository import JsonMemoryRepository, MemoryRepository
from .schemas import MemoryOperation, MemoryRetrievalResult, utc_now_iso


class PostgresLongTermMemoryRepository:
    """PostgreSQL profile memory with repository fallback.

    Phase 3 keeps the existing long_term dict contract while persisting profile
    fields into a versionable memory_items table. Episode/task/skill rows use
    the same table in later phases.
    """

    PROFILE_KEYS = {
        "preferred_industries",
        "preferred_technologies",
        "preferred_stages",
        "preferred_cooperation",
        "preferred_regions",
        "history_queries",
    }

    def __init__(
        self,
        fallback: MemoryRepository,
        database_url: str,
        *,
        tenant_id: str,
    ) -> None:
        self.fallback = fallback
        self.database_url = database_url
        self.tenant_id = tenant_id
        self.available = False
        self.last_error = ""
        self.engine: Any = None
        self._connect()

    def _connect(self) -> None:
        try:
            self.engine = create_engine(self.database_url, pool_pre_ping=True)
            self.initialize()
            self.available = True
            self.last_error = ""
        except Exception as exc:
            self.engine = None
            self.available = False
            self.last_error = str(exc)

    def initialize(self) -> None:
        if self.engine is None:
            return
        with self.engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            connection.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS memory_items (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        tenant_id VARCHAR(128) NOT NULL,
                        user_id VARCHAR(128) NOT NULL,
                        session_id VARCHAR(128),
                        task_id VARCHAR(128),
                        memory_type VARCHAR(32) NOT NULL,
                        memory_key VARCHAR(128) NOT NULL,
                        content TEXT NOT NULL DEFAULT '',
                        structured_value JSONB NOT NULL DEFAULT '{}'::jsonb,
                        embedding vector(1536),
                        importance DOUBLE PRECISION NOT NULL DEFAULT 0.5,
                        confidence DOUBLE PRECISION NOT NULL DEFAULT 0.5,
                        status VARCHAR(32) NOT NULL DEFAULT 'active',
                        valid_from TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        valid_to TIMESTAMPTZ,
                        source_type VARCHAR(64),
                        source_id VARCHAR(128),
                        evidence TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        last_accessed_at TIMESTAMPTZ,
                        access_count INTEGER NOT NULL DEFAULT 0,
                        version INTEGER NOT NULL DEFAULT 1
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_memory_items_lookup
                    ON memory_items (tenant_id, user_id, memory_type, memory_key, status)
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_memory_items_task
                    ON memory_items (tenant_id, user_id, task_id, memory_type, status)
                    """
                )
            )

    @staticmethod
    def _normalize_long_term(user_id: str, long_term: dict[str, Any]) -> dict[str, Any]:
        return JsonMemoryRepository._normalize_long_term(user_id, long_term)

    @staticmethod
    def _content_for_value(memory_key: str, value: Any) -> str:
        if isinstance(value, list):
            return "、".join(str(item) for item in value if str(item).strip())
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return str(value or "")

    def _read_profile(self, user_id: str) -> dict[str, Any]:
        if self.engine is None:
            return {}
        rows = []
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT memory_key, structured_value
                    FROM memory_items
                    WHERE tenant_id = :tenant_id
                      AND user_id = :user_id
                      AND memory_type = 'profile'
                      AND status = 'active'
                      AND valid_to IS NULL
                    """
                ),
                {"tenant_id": self.tenant_id, "user_id": user_id},
            ).mappings().all()
            connection.execute(
                text(
                    """
                    UPDATE memory_items
                    SET last_accessed_at = NOW(), access_count = access_count + 1
                    WHERE tenant_id = :tenant_id
                      AND user_id = :user_id
                      AND memory_type = 'profile'
                      AND status = 'active'
                      AND valid_to IS NULL
                    """
                ),
                {"tenant_id": self.tenant_id, "user_id": user_id},
            )
        result: dict[str, Any] = {}
        for row in rows:
            value = row["structured_value"]
            if isinstance(value, str):
                value = json.loads(value)
            result[str(row["memory_key"])] = value.get("value") if isinstance(value, dict) and "value" in value else value
        return result

    def _upsert_profile_key(self, user_id: str, memory_key: str, value: Any) -> None:
        if self.engine is None or memory_key not in self.PROFILE_KEYS:
            return
        structured_value = {"value": value}
        content = self._content_for_value(memory_key, value)
        with self.engine.begin() as connection:
            existing = connection.execute(
                text(
                    """
                    SELECT id, structured_value, version
                    FROM memory_items
                    WHERE tenant_id = :tenant_id
                      AND user_id = :user_id
                      AND memory_type = 'profile'
                      AND memory_key = :memory_key
                      AND status = 'active'
                      AND valid_to IS NULL
                    ORDER BY version DESC
                    LIMIT 1
                    """
                ),
                {"tenant_id": self.tenant_id, "user_id": user_id, "memory_key": memory_key},
            ).mappings().first()
            if existing:
                old_value = existing["structured_value"]
                if isinstance(old_value, str):
                    old_value = json.loads(old_value)
                if old_value == structured_value:
                    return
                connection.execute(
                    text(
                        """
                        UPDATE memory_items
                        SET status = 'superseded', valid_to = NOW(), updated_at = NOW()
                        WHERE id = :id
                        """
                    ),
                    {"id": existing["id"]},
                )
                version = int(existing["version"]) + 1
            else:
                version = 1

            connection.execute(
                text(
                    """
                    INSERT INTO memory_items (
                        tenant_id, user_id, memory_type, memory_key, content,
                        structured_value, importance, confidence, source_type,
                        evidence, version
                    )
                    VALUES (
                        :tenant_id, :user_id, 'profile', :memory_key, :content,
                        CAST(:structured_value AS JSONB), :importance, :confidence,
                        'memory_agent', :evidence, :version
                    )
                    """
                ),
                {
                    "tenant_id": self.tenant_id,
                    "user_id": user_id,
                    "memory_key": memory_key,
                    "content": content,
                    "structured_value": json.dumps(structured_value, ensure_ascii=False),
                    "importance": 0.7 if memory_key != "history_queries" else 0.45,
                    "confidence": 0.75,
                    "evidence": f"profile field updated by MemoryAgent at {utc_now_iso()}",
                    "version": version,
                },
            )

    def get_session(self, session_id: str, *, user_id: str = "") -> dict[str, Any]:
        return self.fallback.get_session(session_id, user_id=user_id)

    def get_long_term(self, user_id: str) -> dict[str, Any]:
        fallback_long_term = self.fallback.get_long_term(user_id)
        if not self.available:
            return fallback_long_term
        try:
            profile = self._read_profile(user_id)
            merged = dict(fallback_long_term)
            merged.update(profile)
            return self._normalize_long_term(user_id, merged)
        except Exception as exc:
            self.available = False
            self.last_error = str(exc)
            return fallback_long_term

    def save(self, session: dict[str, Any], long_term: dict[str, Any]) -> None:
        self.fallback.save(session, long_term)
        if not self.available:
            return
        try:
            user_id = str(long_term["user_id"])
            normalized = self._normalize_long_term(user_id, long_term)
            for key in self.PROFILE_KEYS:
                self._upsert_profile_key(user_id, key, normalized.get(key, []))
        except Exception as exc:
            self.available = False
            self.last_error = str(exc)

    def apply_operations(
        self,
        *,
        user_id: str,
        session_id: str,
        operations: list[MemoryOperation],
    ) -> list[dict[str, Any]]:
        fallback_results = self.fallback.apply_operations(
            user_id=user_id,
            session_id=session_id,
            operations=operations,
        )
        if not self.available or self.engine is None:
            return fallback_results

        try:
            with self.engine.begin() as connection:
                for operation in operations:
                    if operation.memory_type.value == "profile":
                        continue
                    operation_name = operation.operation.value
                    active = connection.execute(
                        text(
                            """
                            SELECT id, version
                            FROM memory_items
                            WHERE tenant_id = :tenant_id
                              AND user_id = :user_id
                              AND memory_type = :memory_type
                              AND memory_key = :memory_key
                              AND status = 'active'
                              AND valid_to IS NULL
                            ORDER BY version DESC
                            LIMIT 1
                            """
                        ),
                        {
                            "tenant_id": self.tenant_id,
                            "user_id": user_id,
                            "memory_type": operation.memory_type.value,
                            "memory_key": operation.memory_key,
                        },
                    ).mappings().first()
                    if operation_name == "INVALIDATE":
                        if active:
                            connection.execute(
                                text(
                                    """
                                    UPDATE memory_items
                                    SET status = 'invalidated', valid_to = NOW(), updated_at = NOW()
                                    WHERE id = :id
                                    """
                                ),
                                {"id": active["id"]},
                            )
                        continue
                    if operation_name == "IGNORE":
                        continue
                    if active and operation_name in {"UPDATE", "MERGE"}:
                        connection.execute(
                            text(
                                """
                                UPDATE memory_items
                                SET status = 'superseded', valid_to = NOW(), updated_at = NOW()
                                WHERE id = :id
                                """
                            ),
                            {"id": active["id"]},
                        )
                        version = int(active["version"]) + 1
                    else:
                        version = 1
                    value = operation.value if isinstance(operation.value, dict) else {"value": operation.value}
                    connection.execute(
                        text(
                            """
                            INSERT INTO memory_items (
                                tenant_id, user_id, session_id, memory_type, memory_key,
                                content, structured_value, importance, confidence,
                                status, source_type, source_id, evidence, version
                            )
                            VALUES (
                                :tenant_id, :user_id, :session_id, :memory_type, :memory_key,
                                :content, CAST(:structured_value AS JSONB), :importance, :confidence,
                                'active', 'memory_writer', :source_id, :evidence, :version
                            )
                            """
                        ),
                        {
                            "tenant_id": self.tenant_id,
                            "user_id": user_id,
                            "session_id": session_id,
                            "memory_type": operation.memory_type.value,
                            "memory_key": operation.memory_key,
                            "content": operation.evidence or "",
                            "structured_value": json.dumps(value, ensure_ascii=False),
                            "importance": 0.6,
                            "confidence": operation.confidence,
                            "source_id": session_id,
                            "evidence": operation.evidence,
                            "version": version,
                        },
                    )
            return fallback_results
        except Exception as exc:
            self.available = False
            self.last_error = str(exc)
            return fallback_results

    def retrieve_items(
        self,
        *,
        user_id: str,
        memory_types: list[str],
        query: str = "",
        task_id: str = "",
        top_k: int = 5,
    ) -> list[MemoryRetrievalResult]:
        if not self.available or self.engine is None:
            return self.fallback.retrieve_items(
                user_id=user_id,
                memory_types=memory_types,
                query=query,
                task_id=task_id,
                top_k=top_k,
            )
        try:
            params: dict[str, Any] = {
                "tenant_id": self.tenant_id,
                "user_id": user_id,
                "limit": max(1, top_k),
            }
            type_clause = ""
            if memory_types:
                params["memory_types"] = memory_types
                type_clause = "AND memory_type = ANY(:memory_types)"
            task_clause = ""
            if task_id:
                params["task_id"] = task_id
                task_clause = "AND (task_id IS NULL OR task_id = :task_id)"
            rows = []
            with self.engine.begin() as connection:
                rows = connection.execute(
                    text(
                        f"""
                        SELECT id, memory_type, memory_key, content, structured_value,
                               importance, confidence
                        FROM memory_items
                        WHERE tenant_id = :tenant_id
                          AND user_id = :user_id
                          AND status = 'active'
                          AND valid_to IS NULL
                          {type_clause}
                          {task_clause}
                        ORDER BY importance DESC, confidence DESC, updated_at DESC
                        LIMIT :limit
                        """
                    ),
                    params,
                ).mappings().all()
            return [
                MemoryRetrievalResult(
                    memory_id=str(row["id"]),
                    memory_type=row["memory_type"],
                    memory_key=row["memory_key"],
                    content=row["content"] or "",
                    structured_value=row["structured_value"] or {},
                    retrieval_score=round(float(row["confidence"] or 0.0), 4),
                    rerank_score=round(float(row["importance"] or 0.0), 4),
                    source="postgres_memory_items",
                )
                for row in rows
            ]
        except Exception as exc:
            self.available = False
            self.last_error = str(exc)
            return self.fallback.retrieve_items(
                user_id=user_id,
                memory_types=memory_types,
                query=query,
                task_id=task_id,
                top_k=top_k,
            )
