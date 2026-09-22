import pytest

from app.core import intent


@pytest.mark.asyncio
async def test_project_recommendation_outputs_candidates_slots_and_tasks():
    out = await intent.classify("帮我找三个新能源项目，比较一下融资情况，再介绍第一个项目", "")
    assert out["intent"] == "project_recommendation"
    assert out["confidence"] >= 0.8
    assert out["slots"]["technologies"] == ["新能源"]
    assert out["slots"]["limit"] == 3
    assert out["is_multi_task"] is True
    assert [task["sub_intent"] for task in out["tasks"]] == ["field_search", "compare", "detail"]
    assert out["tasks"][1]["depends_on"] == ["task1"]
    assert out["tasks"][2]["reference"] == {"type": "rank", "rank": 1}
    assert any(item["intent"] == "project_recommendation" for item in out["candidate_intents"])


@pytest.mark.asyncio
async def test_refine_recommendation_uses_history():
    out = await intent.classify("只看中试阶段，不要高校", "用户: 帮我推荐合肥人工智能项目")
    assert out["intent"] == "refine_recommendation"
    assert out["sub_intent"] in {"add_filter", "exclude"}
    assert out["slots"]["stages"] == ["中试阶段"]
    assert out["slots"]["exclude_terms"] == ["高校"]


@pytest.mark.asyncio
async def test_result_followup_reference_is_structured():
    out = await intent.classify("介绍一下第二个项目", "助手: 1. 项目A\n2. 项目B")
    assert out["intent"] == "result_followup"
    assert out["sub_intent"] == "detail"
    assert out["reference"] == {"type": "rank", "rank": 2}


def test_semantic_candidates_are_anhui_project_intents():
    candidates = intent.semantic_candidates("合肥人工智能中试项目")
    assert candidates
    assert all(item["intent"] in intent.INTENTS for item in candidates)
    assert any(item["intent"] == "project_recommendation" for item in candidates)


def test_extract_slots_for_project_domain():
    slots = intent.extract_slots("帮我推荐三个合肥人工智能中试项目，合作方式最好是技术入股")
    assert slots["technologies"] == ["人工智能"]
    assert slots["stages"] == ["中试阶段"]
    assert slots["regions"] == ["合肥"]
    assert slots["cooperation"] == ["技术入股"]
    assert slots["limit"] == 3
