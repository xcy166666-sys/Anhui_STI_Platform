from __future__ import annotations

from typing import Any


class MemoryContextBuilder:
    """Build bounded, typed memory blocks for downstream agents and prompts."""

    def build(
        self,
        state: dict[str, Any],
        *,
        memories: list[Any] | None = None,
        token_budget: int = 1200,
    ) -> dict[str, Any]:
        memory_rows = []
        for item in memories or []:
            memory_rows.append(
                {
                    "memory_id": getattr(item, "memory_id", ""),
                    "memory_type": getattr(item, "memory_type", ""),
                    "memory_key": getattr(item, "memory_key", ""),
                    "content": getattr(item, "content", ""),
                    "structured_value": getattr(item, "structured_value", {}),
                    "retrieval_score": getattr(item, "retrieval_score", 0.0),
                    "rerank_score": getattr(item, "rerank_score", 0.0),
                }
            )
        return {
            "token_budget": token_budget,
            "current_state": {
                "intent": state.get("intent", {}),
                "slots": state.get("merged_slots") or state.get("slots") or {},
                "task_id": (state.get("session", {}).get("short_term_memory") or {}).get("task_id", ""),
            },
            "memories": memory_rows,
        }
