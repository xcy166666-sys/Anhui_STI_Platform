from __future__ import annotations

import json
import os
from pathlib import Path

from app.core.agent_memory import (
    AgentMemoryService,
    JsonMemoryRepository,
    MemoryOperation,
    MemoryOperationType,
    MemoryContextBuilder,
    MemoryRetriever,
    MemoryRouter,
    MemoryTurnService,
    MemoryType,
    MemoryWriteCandidate,
    MemoryWriter,
)


def test_json_memory_repository_preserves_legacy_fields(tmp_path: Path) -> None:
    path = tmp_path / "chat_state.json"
    path.write_text(
        json.dumps(
            {
                "sessions": {
                    "s1": {
                        "session_id": "s1",
                        "turns": [],
                        "short_term_memory": {
                            "industries": ["人工智能"],
                            "last_slots": {"technologies": ["AI"]},
                            "last_recommendation_ids": ["p1", "p2"],
                            "excluded_project_ids": ["p0"],
                        },
                    }
                },
                "long_term": {
                    "u1": {
                        "user_id": "u1",
                        "preferred_industries": ["人工智能"],
                        "history_queries": ["找AI项目"],
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    repository = JsonMemoryRepository(path)
    session = repository.get_session("s1", user_id="u1")
    long_term = repository.get_long_term("u1")

    short = session["short_term_memory"]
    assert short["industries"] == ["人工智能"]
    assert short["last_slots"] == {"technologies": ["AI"]}
    assert short["last_recommendation_ids"] == ["p1", "p2"]
    assert short["excluded_project_ids"] == ["p0"]
    assert short["selected_project_ids"] == []
    assert short["conversation_summary"] == ""
    assert "task_id" in short

    assert long_term["preferred_industries"] == ["人工智能"]
    assert long_term["history_queries"] == ["找AI项目"]
    assert long_term["preferred_regions"] == []


def test_agent_memory_service_round_trip(tmp_path: Path) -> None:
    service = AgentMemoryService.from_json_file(tmp_path / "chat_state.json")

    session = service.get_session("s2", user_id="u2")
    long_term = service.get_long_term("u2")
    session["short_term_memory"]["last_query"] = "找新能源项目"
    session["short_term_memory"]["selected_project_ids"] = ["p9"]
    long_term["preferred_technologies"] = ["新能源"]
    service.save(session, long_term)

    restored_session = service.get_session("s2", user_id="u2")
    restored_long_term = service.get_long_term("u2")

    assert restored_session["user_id"] == "u2"
    assert restored_session["short_term_memory"]["last_query"] == "找新能源项目"
    assert restored_session["short_term_memory"]["selected_project_ids"] == ["p9"]
    assert restored_long_term["preferred_technologies"] == ["新能源"]


def test_agent_memory_service_falls_back_when_redis_unavailable(tmp_path: Path) -> None:
    previous_redis_url = os.environ.get("REDIS_URL")
    previous_enabled = os.environ.get("MEMORY_ENABLED")
    try:
        os.environ["REDIS_URL"] = "redis://127.0.0.1:1/0"
        os.environ["MEMORY_ENABLED"] = "true"
        service = AgentMemoryService.from_json_file(tmp_path / "chat_state.json")
        session = service.get_session("s3", user_id="u3")
        long_term = service.get_long_term("u3")
        session["short_term_memory"]["last_raw_query"] = "测试Redis故障"
        service.save(session, long_term)

        restored = service.get_session("s3", user_id="u3")
        assert restored["short_term_memory"]["last_raw_query"] == "测试Redis故障"
        assert service.status()["backend"] in {"RedisSessionMemoryRepository", "JsonMemoryRepository"}
    finally:
        if previous_redis_url is None:
            os.environ.pop("REDIS_URL", None)
        else:
            os.environ["REDIS_URL"] = previous_redis_url
        if previous_enabled is None:
            os.environ.pop("MEMORY_ENABLED", None)
        else:
            os.environ["MEMORY_ENABLED"] = previous_enabled


def test_agent_memory_service_falls_back_when_postgres_unavailable(tmp_path: Path) -> None:
    previous_database_url = os.environ.get("MEMORY_DATABASE_URL")
    previous_redis_url = os.environ.get("REDIS_URL")
    previous_enabled = os.environ.get("MEMORY_ENABLED")
    try:
        os.environ["MEMORY_DATABASE_URL"] = "postgresql+psycopg2://invalid:invalid@127.0.0.1:1/missing"
        os.environ.pop("REDIS_URL", None)
        os.environ["MEMORY_ENABLED"] = "true"
        service = AgentMemoryService.from_json_file(tmp_path / "chat_state.json")
        session = service.get_session("s4", user_id="u4")
        long_term = service.get_long_term("u4")
        long_term["preferred_regions"] = ["合肥"]
        service.save(session, long_term)

        restored = service.get_long_term("u4")
        assert restored["preferred_regions"] == ["合肥"]
        assert service.status()["repository_chain"][0]["name"] == "PostgresLongTermMemoryRepository"
        assert service.status()["repository_chain"][0]["available"] is False
    finally:
        if previous_database_url is None:
            os.environ.pop("MEMORY_DATABASE_URL", None)
        else:
            os.environ["MEMORY_DATABASE_URL"] = previous_database_url
        if previous_redis_url is None:
            os.environ.pop("REDIS_URL", None)
        else:
            os.environ["REDIS_URL"] = previous_redis_url
        if previous_enabled is None:
            os.environ.pop("MEMORY_ENABLED", None)
        else:
            os.environ["MEMORY_ENABLED"] = previous_enabled


def test_memory_turn_service_preserves_follow_up_behavior() -> None:
    state = {
        "query": "换一批，只看中试阶段",
        "intent": {"name": "refine_recommendation"},
        "top_k": 5,
        "session": {
            "short_term_memory": {
                "last_query": "推荐新能源项目",
                "last_slots": {"technologies": ["新能源"]},
                "last_recommendation_ids": ["p1", "p2"],
                "excluded_project_ids": [],
            }
        },
        "long_term_memory": {"user_id": "u5"},
    }

    MemoryTurnService().apply_turn(
        state,
        query=state["query"],
        current_slots={
            "industries": [],
            "technologies": [],
            "stages": ["中试阶段"],
            "cooperation": [],
            "regions": [],
            "exclude_terms": [],
        },
        base_memory_terms=[],
        limit=None,
        follow_up_markers=["换一批", "只看"],
        context_reuse_markers=["只看"],
        exclude_previous_markers=["换一批"],
    )

    assert state["is_follow_up"] is True
    assert state["merged_slots"]["technologies"] == ["新能源"]
    assert state["merged_slots"]["stages"] == ["中试阶段"]
    assert state["exclude_project_ids"] == ["p1", "p2"]
    assert "推荐新能源项目" in state["expanded_query"]
    assert state["long_term_memory"]["preferred_stages"] == ["中试阶段"]


def test_memory_writer_plans_profile_episode_and_task() -> None:
    state = {
        "trace_id": "trace_test",
        "query": "推荐三个新能源中试项目",
        "intent": {
            "name": "project_recommendation",
            "sub_intent": "field_search",
            "confidence": 0.86,
            "reference": {},
        },
        "merged_slots": {
            "technologies": ["新能源"],
            "stages": ["中试阶段"],
            "cooperation": [],
        },
        "session": {"short_term_memory": {"task_id": ""}},
        "recommendations": [{"project_id": "p1"}, {"project_id": "p2"}],
        "answer": "推荐结果",
    }

    operations = MemoryWriter().plan_state(state)
    keys = {(item.memory_type.value, item.memory_key) for item in operations}

    assert ("profile", "preferred_technologies") in keys
    assert ("profile", "preferred_stages") in keys
    assert ("episode", "turn:trace_test") in keys
    assert ("task", "current_task") in keys


def test_json_memory_writer_operations_are_versioned_and_invalidatable(tmp_path: Path) -> None:
    repository = JsonMemoryRepository(tmp_path / "chat_state.json")
    operation = MemoryOperation(
        operation=MemoryOperationType.ADD,
        memory_type=MemoryType.EPISODE,
        memory_key="turn:t1",
        value={"event": "recommendation_turn"},
        evidence="test",
    )
    first = repository.apply_operations(user_id="u6", session_id="s6", operations=[operation])
    second = repository.apply_operations(user_id="u6", session_id="s6", operations=[operation])
    assert first[0]["status"] == "applied"
    assert second[0]["operation"] == "IGNORE"
    assert second[0]["status"] == "duplicate"

    invalidate = MemoryOperation(
        operation=MemoryOperationType.INVALIDATE,
        memory_type=MemoryType.EPISODE,
        memory_key="turn:t1",
        evidence="invalidated by test",
    )
    result = repository.apply_operations(user_id="u6", session_id="s6", operations=[invalidate])
    assert result[0]["status"] == "applied"


def test_memory_router_retriever_and_context_builder(tmp_path: Path) -> None:
    service = AgentMemoryService.from_json_file(tmp_path / "chat_state.json")
    repository = JsonMemoryRepository(tmp_path / "chat_state.json")
    operations = MemoryWriter().plan(
        [
            # The operation is deliberately explicit to test repository retrieval.
            MemoryWriteCandidate(
                operation=MemoryOperationType.ADD,
                memory_type=MemoryType.PROFILE,
                memory_key="preferred_technologies",
                value=["新能源"],
                confidence=0.9,
                evidence="用户多次关注新能源",
            )
        ]
    )
    repository.apply_operations(user_id="u7", session_id="s7", operations=operations)
    retriever = MemoryRetriever(service)
    rows = retriever.retrieve(user_id="u7", memory_types=["profile"], query="新能源", top_k=3)
    assert rows
    assert rows[0].memory_type.value == "profile"

    router = MemoryRouter()
    assert router.select_types("conversation_help") == []
    assert "task" in router.select_types("result_followup", {"type": "rank", "rank": 2})
    context = MemoryContextBuilder().build(
        {"intent": {"name": "project_recommendation"}, "merged_slots": {"technologies": ["新能源"]}},
        memories=rows,
    )
    assert context["memories"][0]["memory_key"] == "preferred_technologies"
