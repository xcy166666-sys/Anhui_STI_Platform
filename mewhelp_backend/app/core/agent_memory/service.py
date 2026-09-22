from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .long_term import PostgresLongTermMemoryRepository
from .repository import JsonMemoryRepository, MemoryRepository
from .schemas import MemoryOperation, MemoryRetrievalResult
from .short_term import RedisSessionMemoryRepository


class AgentMemoryService:
    """Stable facade used by AgentHarness while repositories evolve by phase."""

    def __init__(self, repository: MemoryRepository) -> None:
        self.repository = repository
        self.backend = repository.__class__.__name__

    @classmethod
    def from_json_file(cls, path: Path) -> "AgentMemoryService":
        fallback = JsonMemoryRepository(path)
        enabled = os.getenv("MEMORY_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
        repository: MemoryRepository = fallback
        if enabled:
            database_url = os.getenv("MEMORY_DATABASE_URL", os.getenv("DATABASE_URL", "")).strip()
            if database_url:
                tenant_id = os.getenv("RAG_TENANT_ID", "anhui-sti")
                repository = PostgresLongTermMemoryRepository(repository, database_url, tenant_id=tenant_id)
        redis_url = os.getenv("REDIS_URL", "").strip()
        if enabled and redis_url:
            ttl_seconds = _env_int("MEMORY_TTL_SECONDS", 60 * 60 * 24 * 7)
            repository = RedisSessionMemoryRepository(repository, redis_url, ttl_seconds=ttl_seconds)
        return cls(repository)

    def get_session(self, session_id: str, *, user_id: str = "") -> dict[str, Any]:
        return self.repository.get_session(session_id, user_id=user_id)

    def get_long_term(self, user_id: str) -> dict[str, Any]:
        return self.repository.get_long_term(user_id)

    def save(self, session: dict[str, Any], long_term: dict[str, Any]) -> None:
        self.repository.save(session, long_term)

    def apply_operations(
        self,
        *,
        user_id: str,
        session_id: str,
        operations: list[MemoryOperation],
    ) -> list[dict[str, Any]]:
        return self.repository.apply_operations(
            user_id=user_id,
            session_id=session_id,
            operations=operations,
        )

    def retrieve_items(
        self,
        *,
        user_id: str,
        memory_types: list[str],
        query: str = "",
        task_id: str = "",
        top_k: int = 5,
    ) -> list[MemoryRetrievalResult]:
        return self.repository.retrieve_items(
            user_id=user_id,
            memory_types=memory_types,
            query=query,
            task_id=task_id,
            top_k=top_k,
        )

    def status(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "redis_available": bool(getattr(self.repository, "available", False)),
            "redis_error": str(getattr(self.repository, "last_error", ""))[:200],
            "repository_chain": _repository_chain(self.repository),
        }


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _repository_chain(repository: MemoryRepository) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    current: Any = repository
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(
            {
                "name": current.__class__.__name__,
                "available": getattr(current, "available", True),
                "error": str(getattr(current, "last_error", ""))[:200],
            }
        )
        current = getattr(current, "fallback", None)
    return chain
