from __future__ import annotations

import re
import os
from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INTENTS = (
    "conversation_help",
    "clarification",
    "project_recommendation",
    "refine_recommendation",
    "result_followup",
    "out_of_scope",
)

INTENT_ALIASES = {
    "project_search": "project_recommendation",
    "search_project": "project_recommendation",
    "recommend_project": "project_recommendation",
    "project_detail": "result_followup",
    "project_compare": "result_followup",
    "explain_reason": "result_followup",
    "chat": "conversation_help",
    "help": "conversation_help",
}

INTENT_REGISTRY: list[dict[str, Any]] = [
    {
        "intent": "conversation_help",
        "route": "fallback_script",
        "description": "问候、感谢、能力说明、字段解释、平台使用帮助",
        "examples": ["你好", "你能做什么", "怎么使用", "谢谢", "合作方式是什么意思", "项目阶段有哪些"],
    },
    {
        "intent": "clarification",
        "route": "fallback_script",
        "description": "信息不足、筛选条件缺失、指代不清，需要追问用户补充",
        "examples": ["帮我看看", "这个怎么样", "有没有合适的", "再说一个", "我想找合适的"],
    },
    {
        "intent": "project_recommendation",
        "route": "knowledge",
        "description": "首次项目检索、技术项目推荐、按领域阶段合作方式匹配科创项目",
        "examples": ["推荐三个新能源项目", "找合肥人工智能中试项目", "匹配技术合作项目", "有没有大模型成果转化项目"],
    },
    {
        "intent": "refine_recommendation",
        "route": "knowledge",
        "description": "基于上一轮结果继续筛选、排除、放宽条件、换一批或查看更多结果",
        "examples": ["只看中试阶段", "不要高校项目", "再找几个", "换一批", "条件放宽一点", "排除刚才这些"],
    },
    {
        "intent": "result_followup",
        "route": "business",
        "description": "围绕上一轮推荐结果追问详情、比较、推荐理由、来源或指定第几个项目",
        "examples": ["介绍一下第二个", "为什么推荐第一个", "比较一下这三个", "刚才那个来源是什么", "它适合怎么合作"],
    },
    {
        "intent": "out_of_scope",
        "route": "fallback_script",
        "description": "与安徽科创项目推荐无关、违法、隐私或系统不支持的请求",
        "examples": ["写首诗", "查股票价格", "今天天气", "帮我破解密码", "查询身份证信息"],
    },
]


INTENT_CLASSIFY_WITH_CANDIDATES_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是安徽科创项目推荐助手的意图识别节点。"
            "必须从候选意图中判断用户目的，输出结构化结果，不要输出自然语言解释。"
            "优先参考规则初判和 Top-K 语义候选；如果用户一句话包含检索、比较、详情等多个动作，"
            "需要在 tasks 中拆出任务和 depends_on 依赖关系。",
        ),
        (
            "human",
            "最近对话:\n{history}\n\n"
            "当前用户问题:{query}\n\n"
            "规则初判:{rule_intent} confidence={rule_confidence}\n"
            "Top-K 候选意图:\n{candidate_text}",
        ),
    ]
)


class _Intent(BaseModel):
    intent: Literal[
        "conversation_help",
        "clarification",
        "project_recommendation",
        "refine_recommendation",
        "result_followup",
        "out_of_scope",
    ] = Field(description="安徽科创项目推荐助手的主意图")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    sub_intent: str = Field(default="")
    reason: str = Field(default="")


async def classify(query: str, history: str = "") -> dict[str, Any]:
    rule = rule_classify(query, history)
    candidates = semantic_candidates(query, top_k=3)
    if rule["confidence"] >= 0.92:
        return enrich_result(rule, query, candidates, source="rule")
    if not _can_use_legacy_llm():
        return enrich_result(fallback_without_llm(rule, candidates), query, candidates, source="fallback_without_llm")

    try:
        from app.core import llm

        model = llm.structured(_Intent, slot="intent")
        result: _Intent = await (INTENT_CLASSIFY_WITH_CANDIDATES_PROMPT | model).ainvoke(
            {
                "query": query,
                "history": history or "(无)",
                "rule_intent": rule["intent"],
                "rule_confidence": f"{rule['confidence']:.2f}",
                "candidate_text": format_candidates(candidates),
            }
        )
    except Exception:
        return enrich_result(fallback_without_llm(rule, candidates), query, candidates, source="fallback_without_llm")

    intent = normalize_intent(result.intent)
    confidence = float(result.confidence)
    top = candidates[0] if candidates else {}
    if confidence < 0.45 and top.get("intent") == intent and float(top.get("score", 0.0)) >= 0.18:
        confidence = 0.62
    if confidence < 0.45 and rule["intent"] == intent and rule["confidence"] >= 0.78:
        confidence = 0.62
    if confidence < 0.35:
        intent = "clarification"
    return enrich_result(
        {
            "intent": intent,
            "confidence": confidence,
            "sub_intent": result.sub_intent or infer_sub_intent(query, intent),
            "reason": result.reason,
        },
        query,
        candidates,
        source="llm",
        rule=rule,
    )


def _can_use_legacy_llm() -> bool:
    required = ["CHAT_MODEL", "CHAT_BASE_URL", "CHAT_API_KEY", "EMBED_API_KEY", "RERANK_API_KEY"]
    return all(os.getenv(name) for name in required)


def normalize_intent(value: Any) -> str:
    raw = str(value or "").strip()
    intent = INTENT_ALIASES.get(raw, raw)
    return intent if intent in INTENTS else "clarification"


def rule_classify(query: str, history: str = "") -> dict[str, Any]:
    current = query.strip()
    text = f"{history}\n{current}".lower()
    if any(word in current for word in ["密码", "身份证", "破解", "攻击", "绕过", "隐私"]):
        return {"intent": "out_of_scope", "confidence": 0.95, "sub_intent": "unsafe_or_private", "reason": "命中安全和隐私规则"}
    if any(word in current for word in ["写首诗", "天气", "股票", "翻译", "做饭"]) and not any(word in current for word in ["项目", "技术", "科创"]):
        return {"intent": "out_of_scope", "confidence": 0.9, "sub_intent": "unrelated", "reason": "超出项目推荐范围"}
    if any(word in current for word in ["你好", "您好", "在吗", "谢谢", "感谢", "你能做什么", "怎么用", "帮助", "功能", "什么意思", "字段"]):
        return {"intent": "conversation_help", "confidence": 0.93, "sub_intent": infer_sub_intent(current, "conversation_help"), "reason": "对话帮助类请求"}
    if history and any(word in current for word in ["只看", "不要", "排除", "过滤", "缩小", "再找", "更多", "换一批", "放宽", "不限", "继续"]):
        return {"intent": "refine_recommendation", "confidence": 0.86, "sub_intent": infer_sub_intent(current, "refine_recommendation"), "reason": "基于上下文继续筛选"}
    current_has_new_search = any(word in current for word in ["推荐", "匹配", "筛选", "找", "合作", "投资", "融资", "中试", "研发", "产业化", "成果转化", "技术入股", "人工智能", "新能源", "新材料", "生物医药"])
    if not current_has_new_search and _has_result_reference(current) and any(word in current for word in ["介绍", "详细", "为什么", "理由", "比较", "对比", "来源", "第一个", "第二个", "第三个", "刚才", "它"]):
        return {"intent": "result_followup", "confidence": 0.88, "sub_intent": infer_sub_intent(current, "result_followup"), "reason": "命中上一轮推荐结果追问"}
    if any(word in text for word in ["推荐", "匹配", "筛选", "找", "合作", "投资", "融资", "中试", "研发", "产业化", "成果转化", "技术入股", "人工智能", "新能源", "新材料", "生物医药"]):
        return {"intent": "project_recommendation", "confidence": 0.82, "sub_intent": infer_sub_intent(current, "project_recommendation"), "reason": "需要检索推荐科创项目"}
    if _has_result_reference(current) and any(word in current for word in ["介绍", "详细", "为什么", "理由", "比较", "对比", "来源", "第一个", "第二个", "第三个", "刚才", "它"]):
        return {"intent": "result_followup", "confidence": 0.88, "sub_intent": infer_sub_intent(current, "result_followup"), "reason": "命中上一轮推荐结果追问"}
    if len(current) < 8:
        return {"intent": "clarification", "confidence": 0.62, "sub_intent": "low_information", "reason": "信息过少"}
    return {"intent": "clarification", "confidence": 0.58, "sub_intent": "missing_condition", "reason": "条件不完整"}


def semantic_candidates(query: str, *, top_k: int = 3) -> list[dict[str, Any]]:
    corpus: list[str] = []
    labels: list[str] = []
    descriptions: dict[str, str] = {}
    routes: dict[str, str] = {}
    for item in INTENT_REGISTRY:
        intent = item["intent"]
        descriptions[intent] = item["description"]
        routes[intent] = item["route"]
        for example in item["examples"]:
            corpus.append(f"{item['description']} {example}")
            labels.append(intent)
    if not query.strip():
        return []
    try:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        matrix = vectorizer.fit_transform(corpus + [query])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    except Exception:
        return []
    best: dict[str, float] = {}
    for intent, score in zip(labels, scores):
        best[intent] = max(best.get(intent, 0.0), float(score))
    ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [
        {"intent": intent, "score": round(score, 4), "route": routes.get(intent, ""), "description": descriptions.get(intent, "")}
        for intent, score in ranked
    ]


def format_candidates(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return "(无)"
    return "\n".join(f"- {item['intent']} score={item['score']}: {item.get('description', '')}" for item in candidates)


def fallback_without_llm(rule: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    top = candidates[0] if candidates else {}
    top_score = float(top.get("score", 0.0))
    if rule.get("intent") == "clarification" and top.get("intent") in INTENTS and top_score >= 0.30:
        intent = top["intent"]
        return {
            "intent": intent,
            "confidence": max(0.62, min(0.85, top_score + 0.25)),
            "sub_intent": infer_sub_intent("", intent),
            "reason": "LLM 不可用时采用高分语义候选",
        }
    return rule


def infer_sub_intent(query: str, intent: str) -> str:
    text = query.lower()
    if intent == "conversation_help":
        if any(word in text for word in ["谢谢", "感谢", "多谢", "thanks"]):
            return "thanks"
        if any(word in text for word in ["能做什么", "怎么用", "帮助", "功能"]):
            return "capability_help"
        if any(word in text for word in ["阶段", "合作方式", "赛道", "字段", "什么意思"]):
            return "field_help"
        return "greeting"
    if intent == "clarification":
        if _has_result_reference(query):
            return "ambiguous_reference"
        return "low_information" if len(text.strip()) < 8 else "missing_condition"
    if intent == "project_recommendation":
        if any(word in text for word in ["需求类型", "需求描述", "预算范围", "表单提交", "需求单"]):
            return "from_demand_form"
        if any(word in text for word in ["阶段", "合作", "赛道", "中试", "研发", "产业化", "融资", "技术入股"]):
            return "field_search"
        return "keyword_search"
    if intent == "refine_recommendation":
        if any(word in text for word in ["排除", "不要", "去掉"]):
            return "exclude"
        if any(word in text for word in ["再找", "更多", "换一批"]):
            return "more_results"
        if any(word in text for word in ["放宽", "不限", "宽一点"]):
            return "relax_filter"
        return "add_filter"
    if intent == "result_followup":
        if any(word in text for word in ["比较", "对比", "差异", "优劣"]):
            return "compare"
        if any(word in text for word in ["为什么", "理由", "依据", "凭什么"]):
            return "explain_reason"
        if any(word in text for word in ["来源", "出处", "链接"]):
            return "source"
        return "detail"
    if intent == "out_of_scope":
        if any(word in text for word in ["隐私", "身份证", "密码", "攻击", "破解"]):
            return "unsafe_or_private"
        return "unrelated"
    return ""


def extract_slots(query: str) -> dict[str, Any]:
    slots: dict[str, Any] = {}
    technologies = _collect_terms(
        query,
        {
            "人工智能": ["人工智能", "AI", "ai", "大模型", "算法", "机器视觉", "智能感知"],
            "新能源": ["新能源", "储能", "光伏", "氢能", "电池"],
            "新材料": ["新材料", "材料", "复合材料", "半导体材料"],
            "生物医药": ["生物医药", "医疗器械", "药物", "诊断"],
            "高端装备": ["高端装备", "机器人", "智能制造", "装备"],
        },
    )
    stages = _collect_terms(
        query,
        {
            "研发阶段": ["研发"],
            "中试阶段": ["中试"],
            "产业化": ["产业化", "量产"],
            "成熟应用": ["成熟应用", "应用成熟"],
            "孵化": ["孵化"],
            "样品阶段": ["样品"],
        },
    )
    cooperation = _collect_terms(
        query,
        {
            "技术入股": ["技术入股", "入股"],
            "技术许可": ["技术许可", "许可"],
            "技术合作": ["技术合作"],
            "成果转化": ["成果转化", "转化"],
            "企业孵化": ["企业孵化", "孵化"],
        },
    )
    regions = [term for term in ["安徽", "合肥", "芜湖", "蚌埠", "淮南", "马鞍山", "淮北", "铜陵", "安庆", "黄山", "滁州", "阜阳", "宿州", "六安", "亳州", "池州", "宣城"] if term in query]
    exclude_terms = _extract_exclude_terms(query)
    limit = _extract_limit(query)
    budget = _extract_budget(query)
    if technologies:
        slots["technologies"] = technologies
    if stages:
        slots["stages"] = stages
    if cooperation:
        slots["cooperation"] = cooperation
    if regions:
        slots["regions"] = regions
    if exclude_terms:
        slots["exclude_terms"] = exclude_terms
    if limit:
        slots["limit"] = limit
    if budget:
        slots["budget"] = budget
    return slots


def parse_tasks(query: str, intent: str, slots: dict[str, Any]) -> list[dict[str, Any]]:
    if intent not in {"project_recommendation", "refine_recommendation", "result_followup"}:
        return []
    tasks: list[dict[str, Any]] = []
    needs_search = intent in {"project_recommendation", "refine_recommendation"} or not _has_result_reference(query)
    if needs_search:
        tasks.append(
            {
                "task_id": "task1",
                "intent": "project_recommendation" if intent == "result_followup" else intent,
                "sub_intent": infer_sub_intent(query, "project_recommendation" if intent == "result_followup" else intent),
                "tool": "project_search",
                "slots": slots,
                "depends_on": [],
                "output_key": "searched_projects",
            }
        )
    if any(word in query for word in ["比较", "对比", "差异", "优劣", "融资", "合作方式"]):
        depends_on = [tasks[0]["task_id"]] if tasks else []
        tasks.append(
            {
                "task_id": f"task{len(tasks) + 1}",
                "intent": "result_followup",
                "sub_intent": "compare",
                "tool": "project_compare",
                "depends_on": depends_on,
                "input_from": ["task1.searched_projects"] if depends_on else ["last_projects"],
                "output_key": "comparison_result",
            }
        )
    if any(word in query for word in ["介绍", "详细", "展开", "第一个", "第二个", "第三个", "刚才", "它", "这个"]):
        depends_on = [tasks[0]["task_id"]] if tasks else []
        if tasks and tasks[-1].get("sub_intent") == "compare":
            depends_on.append(tasks[-1]["task_id"])
        tasks.append(
            {
                "task_id": f"task{len(tasks) + 1}",
                "intent": "result_followup",
                "sub_intent": "detail",
                "tool": "project_detail",
                "depends_on": depends_on,
                "input_from": ["task1.searched_projects"] if depends_on else ["last_projects"],
                "reference": extract_reference(query),
                "output_key": "project_detail",
            }
        )
    return tasks if len(tasks) > 1 else []


def enrich_result(
    result: dict[str, Any],
    query: str,
    candidates: list[dict[str, Any]],
    *,
    source: str,
    rule: dict[str, Any] | None = None,
) -> dict[str, Any]:
    intent = normalize_intent(result.get("intent"))
    slots = extract_slots(query)
    tasks = parse_tasks(query, intent, slots)
    return {
        "intent": intent,
        "confidence": float(result.get("confidence", 0.0)),
        "sub_intent": result.get("sub_intent") or infer_sub_intent(query, intent),
        "reason": result.get("reason", ""),
        "candidate_intents": candidates,
        "rule_intent": normalize_intent((rule or result).get("intent", "")),
        "rule_confidence": float((rule or result).get("confidence", 0.0)),
        "intent_source": source,
        "slots": slots,
        "reference": extract_reference(query),
        "is_multi_task": len(tasks) > 1,
        "tasks": tasks,
    }


def extract_reference(query: str) -> dict[str, Any]:
    ordinal_map = {
        "第一": 1,
        "第1": 1,
        "1号": 1,
        "第二": 2,
        "第2": 2,
        "2号": 2,
        "第三": 3,
        "第3": 3,
        "3号": 3,
        "第四": 4,
        "第4": 4,
        "4号": 4,
        "第五": 5,
        "第5": 5,
        "5号": 5,
    }
    for marker, rank in ordinal_map.items():
        if marker in query:
            return {"type": "rank", "rank": rank}
    if any(word in query for word in ["刚才", "上一个", "这个", "那个", "它"]):
        return {"type": "last_project"}
    return {}


def _has_result_reference(query: str) -> bool:
    return bool(extract_reference(query))


def _collect_terms(query: str, mapping: dict[str, list[str]]) -> list[str]:
    values: list[str] = []
    for value, aliases in mapping.items():
        if any(alias in query for alias in aliases):
            values.append(value)
    return list(dict.fromkeys(values))


def _extract_limit(query: str) -> int | None:
    match = re.search(r"(?<!第)([一二两三四五六七八九十\d]{1,3})\s*(?:个|项|条|个项目|项项目)", query)
    if not match:
        return None
    limit = _cn_number(match.group(1))
    return max(1, min(limit, 10)) if limit is not None else None


def _cn_number(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    mapping = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    if value in mapping:
        return mapping[value]
    if value.startswith("十") and len(value) == 2:
        return 10 + mapping.get(value[1], 0)
    if value.endswith("十") and len(value) == 2:
        return mapping.get(value[0], 0) * 10
    if "十" in value and len(value) == 3:
        return mapping.get(value[0], 0) * 10 + mapping.get(value[2], 0)
    return None


def _extract_budget(query: str) -> str:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(万|万元|亿|亿元)", query)
    return match.group(0) if match else ""


def _extract_exclude_terms(query: str) -> list[str]:
    terms: list[str] = []
    for marker in ["不要", "排除", "去掉"]:
        if marker in query:
            tail = query.split(marker, 1)[1]
            tail = re.split(r"[，。；;、\s]", tail, maxsplit=1)[0]
            if tail and tail not in {"这些", "刚才", "上次", "推荐"}:
                terms.append(tail)
    return list(dict.fromkeys(terms))
