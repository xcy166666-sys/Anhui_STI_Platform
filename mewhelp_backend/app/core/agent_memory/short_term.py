from __future__ import annotations

import json
from typing import Any

from .repository import MemoryRepository
from .schemas import MemoryOperation, MemoryRetrievalResult


class RedisSessionMemoryRepository:
    """Redis-backed session memory with JSON repository fallback.

    Long-term memory remains delegated to the fallback repository in Phase 2.
    Session writes are also persisted to fallback storage so Redis outages do not
    erase the current demo behavior.
    """

    def __init__(
        self,
        fallback: MemoryRepository,
        redis_url: str,
        *,
        ttl_seconds: int,
        key_prefix: str = "memory:session",
    ) -> None:
        self.fallback = fallback
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix.rstrip(":")
        self.available = False
        self.last_error = ""
        self._client: Any = None
        self._connect()

    def _connect(self) -> None:
        try:
            import redis

            client = redis.Redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=1.0,
                socket_timeout=1.0,
            )
            client.ping()
            self._client = client
            self.available = True
            self.last_error = ""
        except Exception as exc:
            self._client = None
            self.available = False
            self.last_error = str(exc)

    def _key(self, session_id: str) -> str:
        return f"{self.key_prefix}:{session_id}"

    def _read_session(self, session_id: str) -> dict[str, Any] | None:
        if not self.available or self._client is None:
            return None
        try:
            raw = self._client.get(self._key(session_id))
            if not raw:
                return None
            value = json.loads(raw)
            return value if isinstance(value, dict) else None
        except Exception as exc:
            self.available = False
            self.last_error = str(exc)
            return None

    def _write_session(self, session: dict[str, Any]) -> None:
        if not self.available or self._client is None:
            return
        try:
            payload = json.dumps(session, ensure_ascii=False)
            key = self._key(str(session["session_id"]))
            if self.ttl_seconds > 0:
                self._client.setex(key, self.ttl_seconds, payload)
            else:
                self._client.set(key, payload)
        except Exception as exc:
            self.available = False
            self.last_error = str(exc)

    def get_session(self, session_id: str, *, user_id: str = "") -> dict[str, Any]:
        session = self._read_session(session_id)
        if session is not None:
            if user_id and not session.get("user_id"):
                session["user_id"] = user_id
            return session

        session = self.fallback.get_session(session_id, user_id=user_id)
        self._write_session(session)
        return session

    def get_long_term(self, user_id: str) -> dict[str, Any]:
        return self.fallback.get_long_term(user_id)

    def save(self, session: dict[str, Any], long_term: dict[str, Any]) -> None:
        self.fallback.save(session, long_term)
        self._write_session(session)

    def apply_operations(
        self,
        *,
        user_id: str,
        session_id: str,
        operations: list[MemoryOperation],
    ) -> list[dict[str, Any]]:
        return self.fallback.apply_operations(
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
        return self.fallback.retrieve_items(
            user_id=user_id,
            memory_types=memory_types,
            query=query,
            task_id=task_id,
            top_k=top_k,
        )
