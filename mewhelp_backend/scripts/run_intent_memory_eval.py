from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mewhelp_backend"))

os.environ["MODEL_API_KEY"] = ""

from app.anhui_main import BailianClient, MemoryAgent, PlannerExecutorAgent  # noqa: E402


CASES_FILE = ROOT / "mewhelp_backend" / "eval" / "intent_memory_cases.jsonl"


def _base_state(query: str, intent: str, previous_query: str) -> dict[str, Any]:
    return {
        "query": query,
        "intent": {"name": intent},
        "top_k": 5,
        "session": {
            "short_term_memory": {
                "industries": [],
                "technologies": [],
                "stages": [],
                "cooperation": [],
                "regions": [],
                "last_slots": {},
                "last_query": previous_query,
                "last_raw_query": previous_query,
                "last_recommendation_ids": ["p1", "p2"],
                "excluded_project_ids": [],
            }
        },
        "long_term_memory": {
            "preferred_industries": [],
            "preferred_technologies": [],
            "preferred_stages": [],
            "preferred_cooperation": [],
            "history_queries": [],
            "updated_at": "",
        },
        "agent_trace": [],
    }


def _contains_expected_slots(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, list):
            if not set(expected_value).issubset(set(actual_value or [])):
                return False
        elif actual_value != expected_value:
            return False
    return True


def main() -> int:
    cases = [json.loads(line) for line in CASES_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    client = BailianClient()
    memory_agent = MemoryAgent()
    rows: list[dict[str, Any]] = []
    passed = 0
    for case in cases:
        query = case["query"]
        context = case.get("context") or {}
        intent_result = client.classify_intent(query, context)
        previous_query = context.get("last_query", "")
        state = _base_state(query, intent_result["intent"], previous_query)
        if intent_result["should_retrieve"]:
            memory_agent.run(state)
            slots = state.get("merged_slots", {})
        else:
            slots = {}
        candidate_intents = {item.get("intent") for item in intent_result.get("candidate_intents", [])}
        candidate_ok = True
        if case.get("expected_candidate_contains"):
            candidate_ok = case["expected_candidate_contains"] in candidate_intents
        multitask_ok = intent_result.get("is_multi_task", False) is case.get("expected_is_multi_task", False)
        if case.get("expected_min_tasks") is not None:
            multitask_ok = multitask_ok and len(intent_result.get("tasks") or []) >= int(case["expected_min_tasks"])
        ok = (
            intent_result["intent"] == case["expected_intent"]
            and intent_result.get("sub_intent") == case["expected_sub_intent"]
            and intent_result["should_retrieve"] is case["expected_should_retrieve"]
            and _contains_expected_slots(slots, case.get("expected_slots") or {})
            and candidate_ok
            and multitask_ok
        )
        passed += int(ok)
        rows.append(
            {
                "case_id": case["case_id"],
                "ok": ok,
                "intent": intent_result,
                "slots": slots,
            }
        )
    planner_ok, planner_report = _run_planner_eval()
    report = {
        "cases": len(cases),
        "passed": passed,
        "accuracy": round(passed / max(len(cases), 1), 4),
        "planner_ok": planner_ok,
        "planner": planner_report,
        "failed": [row for row in rows if not row["ok"]],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed == len(cases) and planner_ok else 1


def _run_planner_eval() -> tuple[bool, dict[str, Any]]:
    state = {
        "query": "帮我找三个新能源项目，比较一下合作方式，再介绍第一个项目",
        "intent": {
            "tasks": [
                {"task_id": "task1", "intent": "project_recommendation", "sub_intent": "field_search", "depends_on": []},
                {"task_id": "task2", "intent": "result_followup", "sub_intent": "compare", "depends_on": ["task1"]},
                {"task_id": "task3", "intent": "result_followup", "sub_intent": "detail", "depends_on": ["task1", "task2"]},
            ]
        },
        "recommendations": [
            {
                "project_id": "p1",
                "name": "新能源储能示范项目",
                "track": "新能源与节能环保",
                "technology": "储能",
                "stage": "中试阶段",
                "cooperation": "技术合作",
                "summary": "面向新能源储能场景的示范项目。",
                "source_name": "测试来源",
                "source_url": "https://example.com/p1",
                "fused_score": 0.91,
                "reason": "测试理由",
            },
            {
                "project_id": "p2",
                "name": "光伏材料项目",
                "track": "新材料",
                "technology": "光伏材料",
                "stage": "产业化",
                "cooperation": "成果转化",
                "summary": "面向光伏材料应用。",
                "source_name": "测试来源",
                "source_url": "https://example.com/p2",
                "fused_score": 0.82,
                "reason": "测试理由",
            },
        ],
        "agent_trace": [],
    }
    PlannerExecutorAgent().run(state)
    outputs = state.get("planner_outputs") or {}
    ok = (
        len(outputs.get("searched_projects") or []) == 2
        and len(outputs.get("comparison_result") or []) == 2
        and (outputs.get("project_detail") or {}).get("project_id") == "p1"
        and bool(outputs.get("final_answer"))
    )
    return ok, {
        "outputs_keys": sorted(outputs.keys()),
        "trace": state.get("agent_trace"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
