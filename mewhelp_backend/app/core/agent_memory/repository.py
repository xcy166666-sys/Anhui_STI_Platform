from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from .schemas import (
    MemoryOperation,
    MemoryRetrievalResult,
    SessionMemoryDocument,
    ShortTermMemoryState,
    UserLongTermState,
    utc_now_iso,
)


class MemoryRepository(Protocol):
    def get_session(self, session_id: str, *, user_id: str = "") -> dict[str, Any]:
        ...

    def get_long_term(self, user_id: str) -> dict[str, Any]:
        ...

    def save(self, session: dict[str, Any], long_term: dict[str, Any]) -> None:
        ...

    def apply_operations(
        self,
        *,
        user_id: str,
        session_id: str,
        operations: list[MemoryOperation],
    ) -> list[dict[str, Any]]:
        ...

    def retrieve_items(
        self,
        *,
        user_id: str,
        memory_types: list[str],
        query: str = "",
        task_id: str = "",
        top_k: int = 5,
    ) -> list[MemoryRetrievalResult]:
        ...


class JsonMemoryRepository:
    """Compatibility repository for the current chat_state.json memory file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._save({"sessions": {}, "long_term": {}, "memory_items": []})
        else:
            self._save(self._normalize_state(self._load()))

    def _load(self) -> dict[str, Any]:
        try:
            raw = self.path.read_text(encoding="utf-8")
            state = json.loads(raw) if raw.strip() else {}
        except (OSError, json.JSONDecodeError):
            state = {}
        return self._normalize_state(state)

    def _save(self, state: dict[str, Any]) -> None:
        state = self._normalize_state(state)
        self.path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _normalize_state(state: dict[str, Any]) -> dict[str, Any]:
        state.setdefault("sessions", {})
        state.setdefault("long_term", {})
        state.setdefault("memory_items", [])
        return state

    @staticmethod
    def _session_default(session_id: str, user_id: str = "") -> dict[str, Any]:
        return SessionMemoryDocument(session_id=session_id, user_id=user_id).model_dump()

    @staticmethod
    def _long_term_default(user_id: str) -> dict[str, Any]:
        return UserLongTermState(user_id=user_id).model_dump()

    @staticmethod
    def _normalize_short_term(short: dict[str, Any]) -> dict[str, Any]:
        baseline = ShortTermMemoryState().model_dump()
        baseline.update(short or {})
        return baseline

    def _normalize_session(self, session_id: str, session: dict[str, Any], *, user_id: str = "") -> dict[str, Any]:
        baseline = self._session_default(session_id, user_id=user_id)
        baseline.update(session or {})
        baseline["session_id"] = session_id
        if user_id and not baseline.get("user_id"):
            baseline["user_id"] = user_id
        baseline["short_term_memory"] = self._normalize_short_term(baseline.get("short_term_memory") or {})
        baseline.setdefault("turns", [])
        baseline.setdefault("created_at", utc_now_iso())
        baseline["updated_at"] = baseline.get("updated_at") or utc_now_iso()
        return baseline

    @staticmethod
    def _normalize_long_term(user_id: str, long_term: dict[str, Any]) -> dict[str, Any]:
        baseline = JsonMemoryRepository._long_term_default(user_id)
        baseline.update(long_term or {})
        baseline["user_id"] = user_id
        baseline["updated_at"] = baseline.get("updated_at") or utc_now_iso()
        return baseline

    def get_session(self, session_id: str, *, user_id: str = "") -> dict[str, Any]:
        state = self._load()
        session = self._normalize_session(session_id, state["sessions"].get(session_id, {}), user_id=user_id)
        state["sessions"][session_id] = session
        self._save(state)
        return session

    def get_long_term(self, user_id: str) -> dict[str, Any]:
        state = self._load()
        long_term = self._normalize_long_term(user_id, state["long_term"].get(user_id, {}))
        state["long_term"][user_id] = long_term
        self._save(state)
        return long_term

    def save(self, session: dict[str, Any], long_term: dict[str, Any]) -> None:
        state = self._load()
        session_id = str(session["session_id"])
        user_id = str(long_term["user_id"])
        state["sessions"][session_id] = self._normalize_session(session_id, session, user_id=user_id)
        state["long_term"][user_id] = self._normalize_long_term(user_id, long_term)
        self._save(state)

    def apply_operations(
        self,
        *,
        user_id: str,
        session_id: str,
        operations: list[MemoryOperation],
    ) -> list[dict[str, Any]]:
        state = self._load()
        results: list[dict[str, Any]] = []
        for operation in operations:
            operation_name = operation.operation.value
            if operation_name == "IGNORE":
                results.append({"operation": operation_name, "memory_key": operation.memory_key, "status": "ignored"})
                continue

            items = state["memory_items"]
            active = next(
                (
                    item
                    for item in reversed(items)
                    if item.get("user_id") == user_id
                    and item.get("memory_type") == operation.memory_type.value
                    and item.get("memory_key") == operation.memory_key
                    and item.get("status", "active") == "active"
                ),
                None,
            )
            if operation_name == "INVALIDATE":
                if active:
                    active["status"] = "invalidated"
                    active["valid_to"] = utc_now_iso()
                    active["updated_at"] = utc_now_iso()
                    results.append({"operation": operation_name, "memory_key": operation.memory_key, "status": "applied"})
                else:
                    results.append({"operation": operation_name, "memory_key": operation.memory_key, "status": "noop"})
                continue

            value = operation.value
            normalized_value = value if isinstance(value, dict) else {"value": value}
            if operation_name == "ADD" and active:
                if active.get("structured_value") == normalized_value:
                    results.append(
                        {
                            "operation": "IGNORE",
                            "memory_key": operation.memory_key,
                            "status": "duplicate",
                            "memory_id": active.get("id"),
                        }
                    )
                    continue
            if operation_name in {"UPDATE", "MERGE"} and active:
                if operation_name == "MERGE" and isinstance(active.get("structured_value"), dict) and isinstance(value, dict):
                    merged = dict(active["structured_value"])
                    merged.update(value)
                    value = merged
                active["status"] = "superseded"
                active["valid_to"] = utc_now_iso()
                active["updated_at"] = utc_now_iso()
                version = int(active.get("version", 1)) + 1
            else:
                version = 1

            persisted_value = value if isinstance(value, dict) else {"value": value}
            item = {
                "id": operation.target_memory_id or f"mem_{len(items) + 1}",
                "tenant_id": "anhui-sti",
                "user_id": user_id,
                "session_id": session_id,
                "task_id": "",
                "memory_type": operation.memory_type.value,
                "memory_key": operation.memory_key,
                "content": operation.evidence or "",
                "structured_value": persisted_value,
                "importance": 0.5,
                "confidence": operation.confidence,
                "status": "active",
                "valid_from": utc_now_iso(),
                "valid_to": None,
                "source_type": "memory_writer",
                "source_id": session_id,
                "evidence": operation.evidence,
                "created_at": utc_now_iso(),
                "updated_at": utc_now_iso(),
                "last_accessed_at": None,
                "access_count": 0,
                "version": version,
            }
            items.append(item)
            results.append(
                {
                    "operation": operation_name,
                    "memory_key": operation.memory_key,
                    "status": "applied",
                    "memory_id": item["id"],
                    "version": version,
                }
            )
        self._save(state)
        return results

    def retrieve_items(
        self,
        *,
        user_id: str,
        memory_types: list[str],
        query: str = "",
        task_id: str = "",
        top_k: int = 5,
    ) -> list[MemoryRetrievalResult]:
        state = self._load()
        query_terms = {item for item in str(query or "").lower().split() if item}
        rows: list[tuple[float, dict[str, Any]]] = []
        for item in state["memory_items"]:
            if item.get("user_id") != user_id or item.get("status", "active") != "active":
                continue
            if memory_types and item.get("memory_type") not in memory_types:
                continue
            if task_id and item.get("task_id") not in {"", task_id}:
                continue
            structured = item.get("structured_value") or {}
            haystack = " ".join(
                [
                    str(item.get("memory_key") or "").lower(),
                    str(item.get("content") or "").lower(),
                    json.dumps(structured, ensure_ascii=False).lower(),
                ]
            )
            overlap = sum(1 for term in query_terms if term and term in haystack)
            score = min(1.0, 0.2 + 0.15 * overlap + 0.1 * float(item.get("importance", 0.5)))
            rows.append((score, item))
        rows.sort(key=lambda pair: pair[0], reverse=True)
        return [
            MemoryRetrievalResult(
                memory_id=str(item.get("id") or ""),
                memory_type=item.get("memory_type", "episode"),
                memory_key=str(item.get("memory_key") or ""),
                content=str(item.get("content") or ""),
                structured_value=item.get("structured_value") or {},
                retrieval_score=round(score, 4),
                rerank_score=round(score, 4),
                source="json_memory_items",
            )
            for score, item in rows[: max(1, top_k)]
        ]
