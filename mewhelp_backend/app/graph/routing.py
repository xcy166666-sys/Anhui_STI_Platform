from __future__ import annotations

from langchain_core.messages import AIMessage

from app.config import settings


INTENT_TO_ROUTE: dict[str, str] = {
    "conversation_help": "fallback_script",
    "clarification": "fallback_script",
    "out_of_scope": "fallback_script",
    "project_recommendation": "knowledge",
    "refine_recommendation": "knowledge",
    "result_followup": "business",
}


def route_by_intent(state) -> str:
    return INTENT_TO_ROUTE.get(state.get("intent", ""), "fallback_script")


def confidence_gate(state) -> str:
    return "strong" if state.get("evidence_strong") else "weak"


def should_continue(state) -> str:
    last = state["messages"][-1]
    has_tool_calls = isinstance(last, AIMessage) and bool(last.tool_calls)
    if not has_tool_calls:
        return "stop"
    if state.get("steps", 0) >= settings.max_agent_steps:
        return "stop"
    return "continue"
