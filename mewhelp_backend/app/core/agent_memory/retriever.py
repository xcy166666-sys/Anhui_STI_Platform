from __future__ import annotations

from typing import Any

from .schemas import MemoryRetrievalResult


class MemoryRetriever:
    """Route memory retrieval through the service/repository abstraction."""

    def __init__(self, memory_service: Any) -> None:
        self.memory_service = memory_service

    def retrieve(
        self,
        *,
        user_id: str,
        memory_types: list[str],
        query: str = "",
        task_id: str = "",
        top_k: int = 5,
    ) -> list[MemoryRetrievalResult]:
        return self.memory_service.retrieve_items(
            user_id=user_id,
            memory_types=memory_types,
            query=query,
            task_id=task_id,
            top_k=top_k,
        )
