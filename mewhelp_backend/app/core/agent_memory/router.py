from __future__ import annotations


class MemoryRouter:
    """Lightweight router shell for selecting memory types in later phases."""

    def select_types(self, intent: str, reference: dict | None = None) -> list[str]:
        if intent in {"conversation_help", "out_of_scope"}:
            return []
        if reference:
            return ["task", "episode"]
        if intent in {"project_recommendation", "refine_recommendation"}:
            return ["profile", "episode"]
        return ["profile"]
