from __future__ import annotations

from typing import Any

from .schemas import ShortTermMemoryState, UserLongTermState, utc_now_iso


def unique_values(values: list[str], limit: int = 12) -> list[str]:
    out: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in out:
            out.append(item)
        if len(out) >= limit:
            break
    return out


class MemoryTurnService:
    """Compatible turn-level memory logic extracted from MemoryAgent.

    This service keeps the old state contract intact while making the update
    logic reusable and testable before introducing MemoryWriter in later phases.
    """

    @staticmethod
    def ensure_short_term(short: dict[str, Any]) -> dict[str, Any]:
        baseline = ShortTermMemoryState().model_dump()
        baseline.update(short or {})
        for key in [
            "industries",
            "technologies",
            "stages",
            "cooperation",
            "regions",
            "last_recommendation_ids",
            "excluded_project_ids",
            "selected_project_ids",
            "recent_tool_results",
        ]:
            baseline[key] = list(baseline.get(key) or [])
        baseline["last_slots"] = dict(baseline.get("last_slots") or {})
        baseline["current_constraints"] = dict(baseline.get("current_constraints") or {})
        baseline["current_task"] = dict(baseline.get("current_task") or {})
        return baseline

    @staticmethod
    def ensure_long_term(long_term: dict[str, Any]) -> dict[str, Any]:
        user_id = str((long_term or {}).get("user_id") or "")
        baseline = UserLongTermState(user_id=user_id).model_dump()
        baseline.update(long_term or {})
        for key in [
            "preferred_industries",
            "preferred_technologies",
            "preferred_stages",
            "preferred_cooperation",
            "preferred_regions",
            "history_queries",
        ]:
            baseline[key] = list(baseline.get(key) or [])
        return baseline

    @staticmethod
    def merge_slots(previous: dict[str, Any], current: dict[str, Any], *, use_previous: bool) -> dict[str, Any]:
        if not use_previous:
            return dict(current)
        merged = dict(previous or {})
        for key, value in current.items():
            if isinstance(value, list):
                merged[key] = unique_values(list(merged.get(key, [])) + value)
            elif value not in ("", None, [], {}):
                merged[key] = value
        return merged

    @staticmethod
    def slot_terms(slots: dict[str, Any]) -> list[str]:
        terms: list[str] = []
        for key in ["industries", "technologies", "stages", "cooperation", "regions"]:
            terms.extend(slots.get(key) or [])
        for key in ["budget"]:
            if slots.get(key):
                terms.append(str(slots[key]))
        return unique_values(terms)

    def apply_turn(
        self,
        state: dict[str, Any],
        *,
        query: str,
        current_slots: dict[str, Any],
        base_memory_terms: list[str],
        limit: int | None,
        follow_up_markers: list[str],
        context_reuse_markers: list[str],
        exclude_previous_markers: list[str],
    ) -> dict[str, Any]:
        session = state["session"]
        short = self.ensure_short_term(session.get("short_term_memory") or {})
        session["short_term_memory"] = short
        long_term = self.ensure_long_term(state.get("long_term_memory") or {})
        state["long_term_memory"] = long_term

        previous_query = str(short.get("last_query") or "").strip()
        previous_slots = dict(short.get("last_slots") or {})

        for key in ["industries", "technologies", "stages", "cooperation", "regions"]:
            short[key] = unique_values(list(short.get(key) or []) + list(current_slots.get(key) or []))

        long_term["preferred_industries"] = unique_values(
            long_term["preferred_industries"] + list(current_slots.get("industries") or [])
        )
        long_term["preferred_technologies"] = unique_values(
            long_term["preferred_technologies"] + list(current_slots.get("technologies") or [])
        )
        long_term["preferred_stages"] = unique_values(
            long_term["preferred_stages"] + list(current_slots.get("stages") or [])
        )
        long_term["preferred_cooperation"] = unique_values(
            long_term["preferred_cooperation"] + list(current_slots.get("cooperation") or [])
        )
        long_term["preferred_regions"] = unique_values(
            long_term.get("preferred_regions", []) + list(current_slots.get("regions") or [])
        )
        long_term["history_queries"] = unique_values(long_term["history_queries"] + [query], limit=30)
        long_term["updated_at"] = utc_now_iso()

        retrieved_terms: list[str] = []
        for memory in (state.get("memory_context") or {}).get("memories", []):
            structured = memory.get("structured_value") or {}
            value = structured.get("value") if isinstance(structured, dict) else structured
            if isinstance(value, list):
                retrieved_terms.extend(str(item) for item in value)
            elif isinstance(value, str):
                retrieved_terms.append(value)

        memory_terms = unique_values(base_memory_terms + retrieved_terms)
        if any(word in query for word in context_reuse_markers):
            memory_terms = unique_values(
                memory_terms
                + short["industries"]
                + short["technologies"]
                + short["stages"]
                + short["cooperation"]
            )

        is_follow_up = (
            state.get("intent", {}).get("name") == "refine_recommendation"
            or (previous_query and any(word in query for word in follow_up_markers))
        )
        merged_slots = self.merge_slots(previous_slots, current_slots, use_previous=is_follow_up)
        if is_follow_up:
            memory_terms = unique_values(
                memory_terms
                + short["industries"]
                + short["technologies"]
                + short["stages"]
                + short["cooperation"]
            )
        memory_terms = unique_values(memory_terms + self.slot_terms(merged_slots))

        if is_follow_up and previous_query:
            slot_text = " ".join(self.slot_terms(merged_slots))
            expanded_query = f"{previous_query}\n补充要求：{query}\n结构化条件：{slot_text}".strip()
            exclude_project_ids = list(short.get("excluded_project_ids", []))
            if any(word in query for word in exclude_previous_markers):
                exclude_project_ids = list(
                    dict.fromkeys(exclude_project_ids + list(short.get("last_recommendation_ids", [])))
                )
        else:
            expanded_query = " ".join(unique_values([query] + memory_terms))
            exclude_project_ids = list(short.get("excluded_project_ids", []))

        short["last_query"] = expanded_query
        short["last_raw_query"] = query
        short["last_slots"] = merged_slots
        short["current_constraints"] = merged_slots
        short["updated_at"] = utc_now_iso()

        if limit:
            state["top_k"] = limit
        state["exclude_project_ids"] = exclude_project_ids
        state["slots"] = current_slots
        state["merged_slots"] = merged_slots
        state["memory_terms"] = memory_terms
        state["expanded_query"] = expanded_query
        state["rewritten_query"] = expanded_query
        state["is_follow_up"] = is_follow_up
        return state
