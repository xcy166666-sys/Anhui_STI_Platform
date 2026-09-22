from __future__ import annotations

from typing import Any

from .schemas import MemoryOperation, MemoryOperationType, MemoryType, MemoryWriteCandidate


class MemoryWriter:
    """Build validated memory operations from an executed AgentHarness state.

    The writer is deterministic in Phase 5. LLM extraction can be added later,
    but persistence always receives Pydantic-validated operations.
    """

    def plan(self, candidates: list[MemoryWriteCandidate]) -> list[MemoryOperation]:
        return [
            MemoryOperation(
                operation=item.operation,
                memory_type=item.memory_type,
                memory_key=item.memory_key,
                value=item.value,
                confidence=item.confidence,
                evidence=item.evidence,
            )
            for item in candidates
        ]

    def plan_state(self, state: dict[str, Any]) -> list[MemoryOperation]:
        query = str(state.get("query") or "").strip()
        if not query:
            return []

        intent = state.get("intent") or {}
        slots = state.get("merged_slots") or state.get("slots") or {}
        trace_id = str(state.get("trace_id") or "")
        recommendation_ids = [
            str(item.get("project_id"))
            for item in state.get("recommendations", [])
            if item.get("project_id")
        ]
        evidence = f"query={query}"
        operations: list[MemoryOperation] = []

        profile_mapping = {
            "industries": "preferred_industries",
            "technologies": "preferred_technologies",
            "stages": "preferred_stages",
            "cooperation": "preferred_cooperation",
            "regions": "preferred_regions",
        }
        for slot_key, memory_key in profile_mapping.items():
            values = list(slots.get(slot_key) or [])
            if not values:
                continue
            operations.append(
                MemoryOperation(
                    operation=MemoryOperationType.UPDATE,
                    memory_type=MemoryType.PROFILE,
                    memory_key=memory_key,
                    value=values,
                    confidence=0.82,
                    evidence=evidence,
                )
            )

        operations.append(
            MemoryOperation(
                operation=MemoryOperationType.ADD,
                memory_type=MemoryType.EPISODE,
                memory_key=f"turn:{trace_id or query[:24]}",
                value={
                    "event": "recommendation_turn",
                    "query": query,
                    "intent": intent.get("name", ""),
                    "sub_intent": intent.get("sub_intent", ""),
                    "recommendation_ids": recommendation_ids,
                    "reference": intent.get("reference") or {},
                },
                confidence=float(intent.get("confidence", 0.5) or 0.5),
                evidence=evidence,
            )
        )

        if intent.get("name") in {
            "project_recommendation",
            "refine_recommendation",
            "result_followup",
            "project_detail",
            "explain_reason",
        }:
            task_id = str((state.get("session", {}).get("short_term_memory") or {}).get("task_id") or "")
            operations.append(
                MemoryOperation(
                    operation=MemoryOperationType.UPDATE,
                    memory_type=MemoryType.TASK,
                    memory_key=task_id or "current_task",
                    value={
                        "goal": query,
                        "intent": intent.get("name", ""),
                        "constraints": slots,
                        "recommended_projects": recommendation_ids,
                        "status": "finished" if recommendation_ids else "running",
                        "summary": state.get("answer", "")[:500],
                    },
                    confidence=float(intent.get("confidence", 0.5) or 0.5),
                    evidence=evidence,
                )
            )
        return operations
