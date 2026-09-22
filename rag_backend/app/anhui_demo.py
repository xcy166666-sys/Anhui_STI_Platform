from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine, text


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "anhui_data" / "cleaned" / "project_vectors_source.jsonl"
STATE_FILE = ROOT / "anhui_data" / "chat_state.json"
FRONTEND_DIR = ROOT / "anhui_frontend"

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:15432/rag_center",
)
RAG_KB_ID = os.getenv("RAG_KB_ID", "2bb8255a-4817-50f4-ba0d-49688b7fe8b5")
RAG_TENANT_ID = os.getenv("RAG_TENANT_ID", "anhui-sti")
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "")
MODEL_BASE_URL = os.getenv(
    "MODEL_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen-plus")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, float) and pd.isna(value):
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def parse_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def token_set(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", text or "")
        if len(token) > 1
    }


def unique(values: list[str], limit: int = 12) -> list[str]:
    out: list[str] = []
    for value in values:
        value = value.strip()
        if value and value not in out:
            out.append(value)
        if len(out) >= limit:
            break
    return out


class ChatRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    session_id: str | None = None
    user_id: str = Field(default="demo-user", min_length=1, max_length=128)


class ProjectIndex:
    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"project source file not found: {path}")
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                records.append(record)
        self.df = pd.DataFrame(records).fillna("")
        if "project_id" not in self.df.columns:
            raise ValueError("project_vectors_source.jsonl missing project_id")
        self.df["search_text"] = self.df.apply(self._search_text, axis=1)
        self.id_to_index = {
            clean(value): index
            for index, value in self.df["project_id"].items()
            if clean(value)
        }
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)
        self.matrix = self.vectorizer.fit_transform(self.df["search_text"].tolist())

    @staticmethod
    def _search_text(row: pd.Series) -> str:
        metadata = parse_metadata(row.get("metadata"))
        parts = [
            clean(row.get("content")),
            clean(row.get("project_id")),
            clean(row.get("vector_ready")),
            clean(metadata),
        ]
        return "\n".join(part for part in parts if part)

    def retrieve_local(self, query: str, top_k: int) -> list[dict[str, Any]]:
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        candidates: list[dict[str, Any]] = []
        for idx, score in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[: max(top_k * 5, 20)]:
            candidates.append(
                {
                    "row_index": idx,
                    "rag_score": float(score),
                    "row": self.df.iloc[idx],
                }
            )
        return candidates

    def to_project(self, row: pd.Series) -> dict[str, Any]:
        metadata = parse_metadata(row.get("metadata"))
        return {
            "project_id": clean(row.get("project_id") or metadata.get("project_id")),
            "source_record_id": clean(metadata.get("source_record_id")),
            "name": clean(metadata.get("project_name") or metadata.get("椤圭洰鍚嶇О")),
            "category": clean(metadata.get("category") or metadata.get("椤圭洰澶х被")),
            "subcategory": clean(metadata.get("subcategory") or metadata.get("椤圭洰瀛愮被")),
            "track": clean(metadata.get("track") or metadata.get("浜т笟璧涢亾")),
            "technology": clean(metadata.get("technology") or metadata.get("鎶€鏈柟鍚?")),
            "original_industry": clean(metadata.get("original_industry")),
            "original_technology": clean(metadata.get("original_technology")),
            "stage": clean(metadata.get("stage")),
            "stage_basis": clean(metadata.get("stage_basis") or metadata.get("闃舵鍒ゅ畾渚濇嵁")),
            "summary": clean(metadata.get("summary") or metadata.get("project_summary") or row.get("content")),
            "core_technology": clean(metadata.get("core_technology") or metadata.get("鏍稿績鎶€鏈?")),
            "application": clean(metadata.get("application") or metadata.get("搴旂敤鍦烘櫙")),
            "cooperation": clean(metadata.get("cooperation") or metadata.get("鍚堜綔/杞寲鏂瑰紡")),
            "source_name": clean(metadata.get("source_name")),
            "source_id": clean(metadata.get("source_id")),
            "source_entry": clean(metadata.get("source_entry")),
            "source_url": clean(metadata.get("source_url")),
            "publish_date": clean(metadata.get("publish_date")),
            "collected_at": clean(metadata.get("collected_at")),
            "quality": clean(metadata.get("quality")),
            "technical_match_available": clean(metadata.get("technical_match_available")),
            "vc_data": clean(metadata.get("vc_data")),
            "has_summary": bool(metadata.get("has_summary")),
            "has_core_technology": bool(metadata.get("has_core_technology")),
            "has_application": bool(metadata.get("has_application")),
            "raw_record": metadata.get("raw_record") or {},
        }


class PgvectorClient:
    def __init__(self) -> None:
        self.engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        self.api_key = MODEL_API_KEY
        self.available = bool(self.api_key)

    def _embed(self, text_value: str) -> list[float]:
        if not self.api_key:
            raise RuntimeError("MODEL_API_KEY is not configured")
        payload = {
            "model": EMBEDDING_MODEL,
            "input": [text_value],
        }
        if EMBEDDING_DIMENSIONS:
            payload["dimensions"] = EMBEDDING_DIMENSIONS
        response = httpx.post(
            f"{MODEL_BASE_URL.rstrip('/')}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=60,
        )
        if response.status_code >= 400 and "dimensions" in payload:
            payload.pop("dimensions", None)
            response = httpx.post(
                f"{MODEL_BASE_URL.rstrip('/')}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=60,
            )
        response.raise_for_status()
        vector = response.json()["data"][0]["embedding"]
        if EMBEDDING_DIMENSIONS and len(vector) != EMBEDDING_DIMENSIONS:
            raise RuntimeError(f"unexpected embedding dimension: {len(vector)}")
        return vector

    def retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        vector = self._embed(query)
        vector_literal = "[" + ",".join(str(value) for value in vector) + "]"
        statement = text(
            """
            SELECT document_id, title, content, metadata,
                   1 - (embedding <=> CAST(:embedding AS vector)) AS score
            FROM chunks
            WHERE tenant_id = :tenant_id AND kb_id = :kb_id
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :top_k
            """
        )
        with self.engine.connect() as connection:
            rows = connection.execute(
                statement,
                {
                    "embedding": vector_literal,
                    "tenant_id": RAG_TENANT_ID,
                    "kb_id": RAG_KB_ID,
                    "top_k": top_k,
                },
            ).mappings()
            return [dict(row) for row in rows]


class BailianClient:
    ALLOWED_INTENTS = {
        "conversation_help",
        "clarification",
        "project_recommendation",
        "refine_recommendation",
        "out_of_scope",
    }
    INTENT_ALIASES = {
        "conversation": "conversation_help",
        "chat": "conversation_help",
        "help": "conversation_help",
        "capability_help": "conversation_help",
        "project_search": "project_recommendation",
        "search_project": "project_recommendation",
        "recommend_project": "project_recommendation",
    }
    INTENT_REGISTRY = [
        {
            "intent": "conversation_help",
            "description": "问候、感谢、能力说明、字段解释、使用帮助",
            "examples": ["你好", "你能做什么", "怎么使用", "谢谢", "项目阶段是什么意思", "合作方式有哪些"],
        },
        {
            "intent": "clarification",
            "description": "信息不足、指代不清、条件缺失，需要追问",
            "examples": ["帮我看看", "这个怎么样", "合适的吗", "再说一下", "我想找合适的", "第二个是哪一个"],
        },
        {
            "intent": "project_recommendation",
            "description": "首次项目检索、技术项目推荐、按行业阶段合作方式匹配项目",
            "examples": ["推荐三个新能源项目", "找合肥人工智能中试项目", "匹配技术合作项目", "有没有大模型成果转化项目"],
        },
        {
            "intent": "refine_recommendation",
            "description": "基于上一轮结果继续筛选、排除、放宽、换一批、更多结果",
            "examples": ["只看中试阶段", "不要高校项目", "再找几个", "换一批", "条件放宽一点", "排除刚才这些"],
        },
        {
            "intent": "out_of_scope",
            "description": "与安徽科创项目推荐无关、违法、隐私或系统不支持",
            "examples": ["写首诗", "查股票价格", "今天天气", "帮我破解密码", "查询身份证信息"],
        },
    ]

    def __init__(self) -> None:
        self.api_key = MODEL_API_KEY

    def _chat(self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int) -> str:
        if not self.api_key:
            raise RuntimeError("MODEL_API_KEY is not configured")
        response = httpx.post(
            f"{MODEL_BASE_URL.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": CHAT_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=90,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def classify_intent(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        candidate_intents = self._semantic_candidates(query)
        rule_result = self._heuristic_intent(query, context)
        if not self.api_key:
            return self._with_analysis(rule_result, query, candidate_intents)
        prompt = (
            "你是安徽科创项目推荐聊天机器人的意图识别器。"
            "只输出 JSON，不要输出其他内容。"
            '格式: {"intent":"conversation_help|clarification|project_recommendation|refine_recommendation|out_of_scope",'
            '"sub_intent":"greeting|thanks|capability_help|field_help|missing_condition|low_information|ambiguous_reference|from_demand_form|keyword_search|field_search|add_filter|exclude|more_results|relax_filter|unrelated|unsafe_or_private|unsupported_action",'
            '"should_retrieve":true/false,"confidence":0.0-1.0,"reason":"简短原因","is_multi_task":true/false,"tasks":[]}。'
            "判断规则："
            "你会收到 Top-K 候选意图，只能优先在候选意图中选择；除非候选明显错误，才允许选择其他合法意图。"
            "问候、感谢、能力咨询、字段解释 -> conversation_help 且 should_retrieve=false；"
            "信息不足、条件不全 -> clarification 且 should_retrieve=false；"
            "明确要找项目、筛选项目、匹配技术、找合作、找投资、找中试/研发/产业化项目 -> project_recommendation 且 should_retrieve=true；"
            "在上轮条件基础上继续缩小范围、排除、放宽、再找 -> refine_recommendation 且 should_retrieve=true；"
            "与安徽科创项目推荐无关、违法、隐私或系统不支持的请求 -> out_of_scope 且 should_retrieve=false。"
        )
        try:
            content = self._chat(
                [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"query": query, "context": context, "candidate_intents": candidate_intents},
                            ensure_ascii=False,
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=260,
            )
            parsed = json.loads(content)
            intent = self._normalize_intent(parsed.get("intent"))
            sub_intent = clean(parsed.get("sub_intent")) or self._infer_sub_intent(query, intent)
            confidence = float(parsed.get("confidence", 0.5))
            top_candidate = candidate_intents[0] if candidate_intents else {}
            rule_intent = self._normalize_intent(rule_result.get("intent"))
            if confidence < 0.45 and top_candidate.get("intent") == intent and float(top_candidate.get("score", 0.0)) >= 0.18:
                confidence = 0.62
            if confidence < 0.45 and rule_intent == intent and float(rule_result.get("confidence", 0.0)) >= 0.78:
                confidence = 0.62
            if confidence < 0.45:
                intent = "clarification"
                sub_intent = "low_information"
            return {
                "intent": intent,
                "sub_intent": sub_intent,
                "should_retrieve": self._should_retrieve(intent, parsed.get("should_retrieve")),
                "confidence": confidence,
                "reason": clean(parsed.get("reason")),
                "candidate_intents": candidate_intents,
                "is_multi_task": bool(parsed.get("is_multi_task", False)),
                "tasks": parsed.get("tasks")
                if isinstance(parsed.get("tasks"), list) and all(isinstance(item, dict) for item in parsed.get("tasks", []))
                else self._parse_tasks(query, intent),
            }
        except Exception:
            return self._with_analysis(rule_result, query, candidate_intents)

    @classmethod
    def _semantic_candidates(cls, query: str, *, top_k: int = 3) -> list[dict[str, Any]]:
        corpus: list[str] = []
        labels: list[str] = []
        descriptions: dict[str, str] = {}
        for item in cls.INTENT_REGISTRY:
            intent = item["intent"]
            descriptions[intent] = item["description"]
            for example in item["examples"]:
                corpus.append(f"{item['description']} {example}")
                labels.append(intent)
        if not clean(query):
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
            {
                "intent": intent,
                "score": round(score, 4),
                "description": descriptions.get(intent, ""),
            }
            for intent, score in ranked
        ]

    @classmethod
    def _parse_tasks(cls, query: str, intent: str) -> list[dict[str, Any]]:
        if intent not in {"project_recommendation", "refine_recommendation"}:
            return []
        tasks: list[dict[str, Any]] = [
            {
                "task_id": "task1",
                "intent": intent,
                "sub_intent": cls._infer_sub_intent(query, intent),
                "depends_on": [],
                "output_key": "searched_projects",
            }
        ]
        if any(word in query for word in ["比较", "对比", "比一下", "融资", "合作方式差异", "优劣"]):
            tasks.append(
                {
                    "task_id": "task2",
                    "intent": "result_followup",
                    "sub_intent": "compare",
                    "depends_on": ["task1"],
                    "input_from": ["task1.searched_projects"],
                    "output_key": "comparison_result",
                }
            )
        if any(word in query for word in ["介绍第", "详细介绍", "展开第", "第一个", "第二个"]):
            depends_on = ["task1"]
            if len(tasks) > 1:
                depends_on.append(tasks[-1]["task_id"])
            tasks.append(
                {
                    "task_id": f"task{len(tasks) + 1}",
                    "intent": "result_followup",
                    "sub_intent": "detail",
                    "depends_on": depends_on,
                    "input_from": ["task1.searched_projects"],
                    "output_key": "project_detail",
                }
            )
        return tasks if len(tasks) > 1 else []

    @classmethod
    def _with_analysis(cls, result: dict[str, Any], query: str, candidate_intents: list[dict[str, Any]]) -> dict[str, Any]:
        intent = cls._normalize_intent(result.get("intent"))
        tasks = cls._parse_tasks(query, intent)
        enriched = dict(result)
        enriched["intent"] = intent
        enriched.setdefault("sub_intent", cls._infer_sub_intent(query, intent))
        enriched["should_retrieve"] = cls._should_retrieve(intent, enriched.get("should_retrieve"))
        enriched["candidate_intents"] = candidate_intents
        enriched["is_multi_task"] = bool(tasks)
        enriched["tasks"] = tasks
        return enriched

    @classmethod
    def _normalize_intent(cls, value: Any) -> str:
        raw = clean(value).strip().lower()
        intent = cls.INTENT_ALIASES.get(raw, raw)
        return intent if intent in cls.ALLOWED_INTENTS else "clarification"

    @staticmethod
    def _should_retrieve(intent: str, raw_value: Any = None) -> bool:
        if intent in {"project_recommendation", "refine_recommendation"}:
            return True
        if intent in {"conversation_help", "clarification", "out_of_scope"}:
            return False
        return bool(raw_value)

    @staticmethod
    def _infer_sub_intent(query: str, intent: str) -> str:
        text_value = query.strip().lower()
        if intent == "conversation_help":
            if any(word in text_value for word in ["谢谢", "感谢", "多谢", "thanks"]):
                return "thanks"
            if any(word in text_value for word in ["能做什么", "怎么用", "帮助", "功能"]):
                return "capability_help"
            if any(word in text_value for word in ["阶段", "合作方式", "赛道", "字段", "什么意思"]):
                return "field_help"
            return "greeting"
        if intent == "clarification":
            if any(word in text_value for word in ["这个", "那个", "它", "第一个", "第二个", "刚才"]):
                return "ambiguous_reference"
            return "low_information" if len(text_value) < 8 else "missing_condition"
        if intent == "project_recommendation":
            if any(word in text_value for word in ["需求类型", "需求描述", "预算范围", "表单提交", "需求单"]):
                return "from_demand_form"
            if any(word in text_value for word in ["阶段", "合作", "赛道", "中试", "研发", "产业化"]):
                return "field_search"
            return "keyword_search"
        if intent == "refine_recommendation":
            if any(word in text_value for word in ["排除", "不要", "去掉"]):
                return "exclude"
            if any(word in text_value for word in ["再找", "更多", "换一批"]):
                return "more_results"
            if any(word in text_value for word in ["放宽", "不限", "宽一点"]):
                return "relax_filter"
            return "add_filter"
        if intent == "out_of_scope":
            if any(word in text_value for word in ["隐私", "身份证", "密码", "攻击", "破解"]):
                return "unsafe_or_private"
            return "unrelated"
        return ""

    @classmethod
    def _heuristic_intent(cls, query: str, context: dict[str, Any]) -> dict[str, Any]:
        text_value = query.strip()
        greeting = any(word in text_value for word in ["你好", "您好", "哈喽", "在吗", "hello", "hi", "谢谢", "感谢"])
        help_query = any(word in text_value for word in ["能做什么", "怎么用", "帮助", "功能", "什么意思", "字段"])
        unsafe = any(word in text_value for word in ["密码", "身份证", "破解", "攻击", "绕过", "隐私"])
        unrelated = any(word in text_value for word in ["写首诗", "股票价格", "天气", "翻译", "做饭"])
        refine = any(word in text_value for word in ["只看", "排除", "不要", "过滤", "缩小", "再加", "再找", "更多", "换一批", "放宽", "不限"])
        search = any(word in text_value for word in ["找", "推荐", "匹配", "筛选", "项目", "合作", "投资", "中试", "研发", "产业化"])
        too_short = len(text_value) < 6
        if unsafe or (unrelated and not search):
            intent = "out_of_scope"
            return {"intent": intent, "sub_intent": cls._infer_sub_intent(query, intent), "should_retrieve": False, "confidence": 0.86, "reason": "超出安徽科创项目推荐范围"}
        if (greeting or help_query) and not search:
            intent = "conversation_help"
            return {"intent": intent, "sub_intent": cls._infer_sub_intent(query, intent), "should_retrieve": False, "confidence": 0.92, "reason": "对话或帮助咨询"}
        if refine and context.get("last_query"):
            intent = "refine_recommendation"
            return {"intent": intent, "sub_intent": cls._infer_sub_intent(query, intent), "should_retrieve": True, "confidence": 0.82, "reason": "基于上轮条件继续筛选"}
        if search:
            intent = "project_recommendation"
            return {"intent": intent, "sub_intent": cls._infer_sub_intent(query, intent), "should_retrieve": True, "confidence": 0.78, "reason": "需要项目检索推荐"}
        if too_short:
            intent = "clarification"
            return {"intent": intent, "sub_intent": cls._infer_sub_intent(query, intent), "should_retrieve": False, "confidence": 0.55, "reason": "信息过少"}
        intent = "clarification"
        return {"intent": intent, "sub_intent": cls._infer_sub_intent(query, intent), "should_retrieve": False, "confidence": 0.6, "reason": "条件不完整"}

    def answer(self, query: str, recommendations: list[dict[str, Any]]) -> str:
        if not self.api_key:
            return self._template_answer(query, recommendations)
        payload = {
            "需求": query,
            "候选项目": [
                {
                    "name": item.get("name"),
                    "track": item.get("track"),
                    "technology": item.get("technology"),
                    "stage": item.get("stage"),
                    "summary": (item.get("summary") or "")[:600],
                    "fused_score": item.get("fused_score"),
                }
                for item in recommendations[:8]
            ],
        }
        prompt = (
            "你是安徽科创项目推荐助手。只能基于给定候选项目回答，不能编造项目。"
            "请输出中文，给出简短结论、推荐理由和最匹配的项目名单。"
        )
        try:
            return self._chat(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=0.2,
                max_tokens=700,
            )
        except Exception:
            return self._template_answer(query, recommendations)

    def answer_planned(
        self,
        query: str,
        recommendations: list[dict[str, Any]],
        planner_outputs: dict[str, Any],
    ) -> str:
        fallback = clean(planner_outputs.get("final_answer"))
        if not self.api_key:
            return fallback or self._template_answer(query, recommendations)
        payload = {
            "用户问题": query,
            "推荐项目": [
                {
                    "rank": index,
                    "name": item.get("name"),
                    "track": item.get("track"),
                    "technology": item.get("technology"),
                    "stage": item.get("stage"),
                    "cooperation": item.get("cooperation"),
                    "summary": (item.get("summary") or "")[:800],
                    "source_name": item.get("source_name"),
                    "source_url": item.get("source_url"),
                    "fused_score": item.get("fused_score"),
                    "reason": item.get("reason"),
                }
                for index, item in enumerate(recommendations[:8], start=1)
            ],
            "任务执行结果": planner_outputs,
        }
        prompt = (
            "你是安徽科创项目推荐顾问。请严格基于给定推荐项目和任务执行结果回答，不能编造项目、金额、主体或联系方式。"
            "如果任务包含比较，请用简洁表格或分点比较；如果包含详情，请介绍被指代的项目。"
            "保留项目排序和关键依据，中文回答。"
        )
        try:
            return self._chat(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=0.2,
                max_tokens=900,
            )
        except Exception:
            return fallback or self._template_answer(query, recommendations)

    def answer_dialogue(
        self,
        query: str,
        intent: dict[str, Any],
        session: dict[str, Any],
        long_term: dict[str, Any],
    ) -> str:
        """Generate an LLM response for every non-retrieval turn as well."""
        if not self.api_key:
            if intent.get("name") == "conversation_help":
                return "你好，我可以帮你从安徽科创项目库里找项目。你可以先说行业、技术方向、项目阶段或合作方式。"
            if intent.get("name") == "out_of_scope":
                return "这个问题超出了安徽科创项目推荐范围。我可以继续帮你找项目、筛选技术方向、解释项目阶段或补充合作方式。"
            return "可以，先告诉我行业方向、技术关键词、项目阶段或合作方式，我再帮你匹配。"

        history = []
        for turn in session.get("turns", [])[-6:]:
            history.append({"role": "user", "content": turn.get("query", "")})
        payload = {
            "当前问题": query,
            "当前意图": intent,
            "最近对话": history,
            "长期偏好": {
                key: value
                for key, value in long_term.items()
                if key.startswith("preferred_")
            },
        }
        prompt = (
            "你是安徽科创项目推荐助手，正在进行自然、多轮、像真人一样的中文对话。"
            "当前轮没有进入项目检索，因此不要编造项目名称、分数或数据库事实。"
            "如果是寒暄就自然回应；如果条件不足，就只追问最有价值的1到2个条件；"
            "如果用户的问题超出安徽科创项目推荐范围，要礼貌说明边界并引导回项目推荐；"
            "如果用户是在补充上一轮条件，要结合对话上下文确认下一步。回复简洁、友好、直接。"
        )
        try:
            return self._chat(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=0.35,
                max_tokens=320,
            )
        except Exception:
            return "我理解了。请再补充行业方向、技术关键词、项目阶段或合作方式中的一项，我就可以继续帮你筛选。"

    @staticmethod
    def _template_answer(query: str, recommendations: list[dict[str, Any]]) -> str:
        if not recommendations:
            return f"我暂时没有找到足够匹配的项目。你可以再补充行业方向、技术关键词、阶段或合作方式。"
        lines = ["我根据你的需求先给出这几项："]
        for idx, item in enumerate(recommendations[:5], start=1):
            lines.append(
                f"{idx}. {item.get('name','')}：{item.get('stage','')}，{item.get('technology','')}，"
                f"综合分 {item.get('fused_score', 0):.3f}"
            )
        lines.append("你可以继续补充“只看中试阶段”“排除某行业”“优先技术合作”这类条件，我再缩小范围。")
        return "\n".join(lines)


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._save({"sessions": {}, "long_term": {}})

    def _load(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, state: dict[str, Any]) -> None:
        self.path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_session(self, session_id: str) -> dict[str, Any]:
        state = self._load()
        return state["sessions"].setdefault(
            session_id,
            {
                "session_id": session_id,
                "created_at": now_iso(),
                "updated_at": now_iso(),
                "turns": [],
                "short_term_memory": {
                    "industries": [],
                    "technologies": [],
                    "stages": [],
                    "cooperation": [],
                    "regions": [],
                    "last_slots": {},
                    "last_query": "",
                    "last_raw_query": "",
                    "last_recommendation_ids": [],
                    "excluded_project_ids": [],
                },
            },
        )

    def get_long_term(self, user_id: str) -> dict[str, Any]:
        state = self._load()
        return state["long_term"].setdefault(
            user_id,
            {
                "user_id": user_id,
                "preferred_industries": [],
                "preferred_technologies": [],
                "preferred_stages": [],
                "preferred_cooperation": [],
                "history_queries": [],
                "updated_at": now_iso(),
            },
        )

    def save(self, session: dict[str, Any], long_term: dict[str, Any]) -> None:
        state = self._load()
        state["sessions"][session["session_id"]] = session
        state["long_term"][long_term["user_id"]] = long_term
        self._save(state)


class IntentAgent:
    def __init__(self, client: BailianClient) -> None:
        self.client = client

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        context = {
            "last_query": state["session"]["short_term_memory"].get("last_query", ""),
            "history_queries": state["long_term_memory"].get("history_queries", [])[-5:],
            "previous_intent": state["session"].get("turns", [])[-1]["intent"]["name"] if state["session"].get("turns") else "",
        }
        result = self.client.classify_intent(state["query"], context)
        state["intent"] = {
            "name": result.get("intent", "clarification"),
            "sub_intent": result.get("sub_intent", ""),
            "confidence": result.get("confidence", 0.5),
            "reason": result.get("reason", ""),
            "candidate_intents": result.get("candidate_intents", []),
            "is_multi_task": bool(result.get("is_multi_task", False)),
            "tasks": result.get("tasks", []),
        }
        state["should_retrieve"] = bool(result.get("should_retrieve", False))
        state["agent_trace"].append(
            {
                "agent": "IntentAgent",
                "intent": state["intent"]["name"],
                "sub_intent": state["intent"].get("sub_intent", ""),
                "confidence": state["intent"].get("confidence", 0.0),
                "candidate_intents": state["intent"].get("candidate_intents", []),
                "is_multi_task": state["intent"].get("is_multi_task", False),
                "tasks": state["intent"].get("tasks", []),
                "should_retrieve": state["should_retrieve"],
            }
        )
        return state


class MemoryAgent:
    INDUSTRY_HINTS = [
        "人工智能与软件",
        "生物医药与医疗健康",
        "高端装备与智能制造",
        "新材料",
        "新能源与节能环保",
        "集成电路与光电信息",
        "量子信息与聚变能源",
        "资源环境与生态治理",
        "空天信息与低空经济",
    ]
    STAGE_HINTS = [
        "研发阶段",
        "中试阶段",
        "产业化",
        "成熟应用",
        "孵化",
        "成长",
        "立项/规划",
        "样品阶段",
    ]
    COOP_HINTS = [
        "技术入股",
        "技术许可",
        "技术合作",
        "成果转化",
        "企业孵化",
        "拟成立公司",
    ]
    TECH_HINTS = [
        "算法",
        "软件",
        "大模型",
        "智能感知",
        "机器视觉",
        "机器人",
        "生物材料",
        "医疗器械",
        "新能源",
        "储能",
    ]
    REGION_HINTS = [
        "安徽",
        "合肥",
        "芜湖",
        "蚌埠",
        "淮南",
        "马鞍山",
        "淮北",
        "铜陵",
        "安庆",
        "黄山",
        "滁州",
        "阜阳",
        "宿州",
        "六安",
        "亳州",
        "池州",
        "宣城",
    ]
    TECH_ALIASES = {
        "AI": "人工智能",
        "ai": "人工智能",
        "人工智能": "人工智能",
        "大模型": "大模型",
        "机器人": "机器人",
        "储能": "储能",
        "新能源": "新能源",
    }
    STAGE_ALIASES = {
        "研发": "研发阶段",
        "中试": "中试阶段",
        "产业化": "产业化",
        "成熟应用": "成熟应用",
        "样品": "样品阶段",
        "孵化": "孵化",
    }
    COOP_ALIASES = {
        "入股": "技术入股",
        "技术入股": "技术入股",
        "许可": "技术许可",
        "技术许可": "技术许可",
        "合作": "技术合作",
        "技术合作": "技术合作",
        "成果转化": "成果转化",
        "孵化": "企业孵化",
    }

    @staticmethod
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

    @classmethod
    def _extract_limit(cls, query: str) -> int | None:
        match = re.search(r"([一二两三四五六七八九十\d]{1,3})\s*(?:个|项|条)", query)
        if not match:
            return None
        limit = cls._cn_number(match.group(1))
        if limit is None:
            return None
        return max(1, min(limit, 10))

    @staticmethod
    def _extract_budget(query: str) -> str:
        match = re.search(r"(\d+(?:\.\d+)?)\s*(万|亿|万元|亿元)", query)
        return match.group(0) if match else ""

    @staticmethod
    def _extract_exclude_terms(query: str) -> list[str]:
        terms: list[str] = []
        for marker in ["不要", "排除", "去掉"]:
            if marker in query:
                tail = query.split(marker, 1)[1]
                tail = re.split(r"[，,。；;、\s]", tail, maxsplit=1)[0]
                if tail and tail not in ["这些", "刚才", "上次", "推荐"]:
                    terms.append(tail)
        return unique(terms)

    @staticmethod
    def _merge_slots(previous: dict[str, Any], current: dict[str, Any], *, use_previous: bool) -> dict[str, Any]:
        if not use_previous:
            return dict(current)
        merged = dict(previous or {})
        for key, value in current.items():
            if isinstance(value, list):
                merged[key] = unique(list(merged.get(key, [])) + value)
            elif value not in ("", None, [], {}):
                merged[key] = value
        return merged

    @staticmethod
    def _slot_terms(slots: dict[str, Any]) -> list[str]:
        terms: list[str] = []
        for key in ["industries", "technologies", "stages", "cooperation", "regions"]:
            terms.extend(slots.get(key) or [])
        for key in ["budget"]:
            if slots.get(key):
                terms.append(str(slots[key]))
        return unique(terms)

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state["query"]
        short = state["session"]["short_term_memory"]
        long_term = state["long_term_memory"]
        for key in ["industries", "technologies", "stages", "cooperation", "regions", "last_recommendation_ids", "excluded_project_ids"]:
            short.setdefault(key, [])
        short.setdefault("last_slots", {})
        short.setdefault("last_query", "")
        short.setdefault("last_raw_query", "")
        previous_query = clean(short.get("last_query"))
        previous_slots = dict(short.get("last_slots") or {})

        industries = [x for x in self.INDUSTRY_HINTS if x in query]
        stages = [x for x in self.STAGE_HINTS if x in query]
        cooperation = [x for x in self.COOP_HINTS if x in query]
        technologies = [x for x in self.TECH_HINTS if x in query]
        technologies = unique(technologies + [value for key, value in self.TECH_ALIASES.items() if key in query])
        stages = unique(stages + [value for key, value in self.STAGE_ALIASES.items() if key in query])
        cooperation = unique(cooperation + [value for key, value in self.COOP_ALIASES.items() if key in query])
        regions = [x for x in self.REGION_HINTS if x in query]
        limit = self._extract_limit(query)
        budget = self._extract_budget(query)
        exclude_terms = self._extract_exclude_terms(query)

        current_slots: dict[str, Any] = {
            "industries": industries,
            "technologies": technologies,
            "stages": stages,
            "cooperation": cooperation,
            "regions": regions,
            "exclude_terms": exclude_terms,
        }
        if limit:
            current_slots["limit"] = limit
        if budget:
            current_slots["budget"] = budget

        short["industries"] = unique(short["industries"] + industries)
        short["technologies"] = unique(short["technologies"] + technologies)
        short["stages"] = unique(short["stages"] + stages)
        short["cooperation"] = unique(short["cooperation"] + cooperation)
        short["regions"] = unique(short.get("regions", []) + regions)

        long_term["preferred_industries"] = unique(long_term["preferred_industries"] + industries)
        long_term["preferred_technologies"] = unique(long_term["preferred_technologies"] + technologies)
        long_term["preferred_stages"] = unique(long_term["preferred_stages"] + stages)
        long_term["preferred_cooperation"] = unique(long_term["preferred_cooperation"] + cooperation)
        long_term["history_queries"] = unique(long_term["history_queries"] + [query], limit=30)
        long_term["updated_at"] = now_iso()

        memory_terms = unique(industries + technologies + stages + cooperation)
        if any(word in query for word in ["只看", "筛选", "继续", "再找", "不要", "排除"]):
            memory_terms = unique(memory_terms + short["industries"] + short["technologies"] + short["stages"] + short["cooperation"])
        follow_up_markers = ["其他", "还有", "换一批", "更多", "继续", "再找", "再来", "下一批", "只看", "排除", "不要", "过滤", "缩小"]
        is_follow_up = (
            state.get("intent", {}).get("name") == "refine_recommendation"
            or (previous_query and any(word in query for word in follow_up_markers))
        )
        merged_slots = self._merge_slots(previous_slots, current_slots, use_previous=is_follow_up)
        if is_follow_up:
            memory_terms = unique(memory_terms + short["industries"] + short["technologies"] + short["stages"] + short["cooperation"])
        memory_terms = unique(memory_terms + self._slot_terms(merged_slots))

        if is_follow_up and previous_query:
            slot_text = " ".join(self._slot_terms(merged_slots))
            expanded_query = f"{previous_query}\n补充要求：{query}\n结构化条件：{slot_text}".strip()
            exclude_previous = any(word in query for word in ["换一批", "更多", "其他", "不要这些", "不要刚才", "排除刚才", "排除上次"])
            state["exclude_project_ids"] = list(short.get("excluded_project_ids", []))
            if exclude_previous:
                state["exclude_project_ids"] = list(
                    dict.fromkeys(state["exclude_project_ids"] + short.get("last_recommendation_ids", []))
                )
        else:
            expanded_query = " ".join(unique([query] + memory_terms))
            state["exclude_project_ids"] = list(short.get("excluded_project_ids", []))

        short["last_query"] = expanded_query
        short["last_raw_query"] = query
        short["last_slots"] = merged_slots
        if limit:
            state["top_k"] = limit
        state["slots"] = current_slots
        state["merged_slots"] = merged_slots
        state["memory_terms"] = memory_terms
        state["expanded_query"] = expanded_query
        state["rewritten_query"] = expanded_query
        state["is_follow_up"] = is_follow_up
        state["agent_trace"].append(
            {
                "agent": "MemoryAgent",
                "slots": current_slots,
                "merged_slots": merged_slots,
                "rewritten_query": expanded_query,
                "terms": memory_terms,
                "is_follow_up": is_follow_up,
                "exclude_count": len(state["exclude_project_ids"]),
            }
        )
        return state


class RetrievalAgent:
    def __init__(self, project_index: ProjectIndex, rag_client: PgvectorClient) -> None:
        self.project_index = project_index
        self.rag_client = rag_client

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state["expanded_query"]
        exclude_project_ids = set(state.get("exclude_project_ids") or [])
        retrieve_top_k = max(state["top_k"] * 5, 20) + len(exclude_project_ids)
        try:
            rag_rows = self.rag_client.retrieve(query, retrieve_top_k)
            candidates: list[dict[str, Any]] = []
            for item in rag_rows:
                metadata = parse_metadata(item.get("metadata"))
                project_id = clean(metadata.get("project_id") or metadata.get("id") or item.get("document_id"))
                if project_id in exclude_project_ids:
                    continue
                row_index = self.project_index.id_to_index.get(project_id)
                if row_index is None:
                    continue
                candidates.append(
                    {
                        "row_index": row_index,
                        "rag_score": max(0.0, float(item.get("score", 0.0))),
                        "row": self.project_index.df.iloc[row_index],
                        "retrieval_source": "pgvector",
                    }
                )
            state["retrieval_mode"] = "pgvector"
            state["candidates"] = candidates
        except Exception as exc:
            state["retrieval_mode"] = "local_tfidf_fallback"
            state["retrieval_error"] = clean(str(exc))[:240]
            state["candidates"] = [
                candidate
                for candidate in self.project_index.retrieve_local(query, retrieve_top_k)
                if clean(parse_metadata(candidate["row"].get("metadata")).get("project_id")) not in exclude_project_ids
            ]
        state["agent_trace"].append(
            {
                "agent": "RetrievalAgent",
                "candidate_count": len(state["candidates"]),
                "mode": state["retrieval_mode"],
                "excluded": len(exclude_project_ids),
            }
        )
        return state


class MatchingAgent:
    FIELD_NAMES = [
        "track",
        "technology",
        "stage",
        "category",
        "subcategory",
        "cooperation",
        "original_industry",
        "original_technology",
    ]

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        query_tokens = token_set(state["expanded_query"])
        memory_terms = state.get("memory_terms") or []
        exclude_terms = set((state.get("merged_slots") or {}).get("exclude_terms") or [])
        scored: list[dict[str, Any]] = []
        for candidate in state["candidates"]:
            row = candidate["row"]
            metadata = parse_metadata(row.get("metadata"))
            field_text = " ".join(clean(metadata.get(field)) for field in self.FIELD_NAMES)
            all_text = f"{field_text} {clean(row.get('content'))}"
            keyword_score = len(query_tokens & token_set(all_text)) / max(len(query_tokens), 1)
            term_hits = sum(1 for term in memory_terms if term and term in all_text)
            term_score = term_hits / max(len(memory_terms), 1) if memory_terms else 0.0
            exclude_penalty = 0.35 if any(term and term in all_text for term in exclude_terms) else 0.0
            traditional_score = max(
                0.0,
                min(1.0, 0.55 * keyword_score + 0.3 * term_score + 0.15 * candidate["rag_score"] - exclude_penalty),
            )
            item = dict(candidate)
            item["traditional_score"] = traditional_score
            scored.append(item)
        state["scored_candidates"] = scored
        state["agent_trace"].append({"agent": "MatchingAgent", "scored_count": len(scored)})
        return state


class RecommendationAgent:
    def __init__(self, project_index: ProjectIndex) -> None:
        self.project_index = project_index

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for candidate in state["scored_candidates"]:
            project = self.project_index.to_project(candidate["row"])
            rag_score = float(candidate["rag_score"])
            traditional_score = float(candidate["traditional_score"])
            fused_score = 0.55 * traditional_score + 0.45 * rag_score
            project.update(
                {
                    "rag_score": round(rag_score, 4),
                    "traditional_score": round(traditional_score, 4),
                    "fused_score": round(fused_score, 4),
                    "reason": self._reason(project, traditional_score, rag_score),
                }
            )
            results.append(project)
        state["recommendations"] = sorted(results, key=lambda item: item["fused_score"], reverse=True)[: state["top_k"]]
        state["agent_trace"].append({"agent": "RecommendationAgent", "result_count": len(state["recommendations"])})
        return state

    @staticmethod
    def _reason(project: dict[str, Any], traditional: float, rag: float) -> str:
        basis = "、".join(x for x in [project.get("track"), project.get("technology"), project.get("stage")] if x)
        return f"传统匹配 {traditional:.0%}，语义检索 {rag:.0%}，依据：{basis or '项目文本'}。"


class EvidenceAgent:
    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        checked: list[dict[str, Any]] = []
        for item in state["recommendations"]:
            warnings: list[str] = []
            if not item.get("source_name"):
                warnings.append("缺少来源名称")
            if not item.get("source_url"):
                warnings.append("缺少来源链接")
            if not item.get("summary"):
                warnings.append("项目摘要较少")
            item["evidence_status"] = "pass" if not warnings else "partial"
            item["evidence_warnings"] = warnings
            checked.append(item)
        state["recommendations"] = checked
        state["agent_trace"].append({"agent": "EvidenceAgent", "checked_count": len(checked)})
        return state


class PlannerExecutorAgent:
    @staticmethod
    def _select_project(query: str, recommendations: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not recommendations:
            return None
        ordinal_map = [
            (["第一个", "第一项", "第1个", "第1项", "1号"], 0),
            (["第二个", "第二项", "第2个", "第2项", "2号"], 1),
            (["第三个", "第三项", "第3个", "第3项", "3号"], 2),
            (["第四个", "第四项", "第4个", "第4项", "4号"], 3),
            (["第五个", "第五项", "第5个", "第5项", "5号"], 4),
        ]
        for markers, index in ordinal_map:
            if any(marker in query for marker in markers) and index < len(recommendations):
                return recommendations[index]
        return recommendations[0]

    @staticmethod
    def _search_summary(recommendations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "rank": index,
                "project_id": item.get("project_id"),
                "name": item.get("name"),
                "track": item.get("track"),
                "technology": item.get("technology"),
                "stage": item.get("stage"),
                "cooperation": item.get("cooperation"),
                "fused_score": item.get("fused_score"),
                "reason": item.get("reason"),
            }
            for index, item in enumerate(recommendations, start=1)
        ]

    @staticmethod
    def _compare_projects(recommendations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "rank": index,
                "name": item.get("name"),
                "technology": item.get("technology") or "未标注",
                "stage": item.get("stage") or "未标注",
                "cooperation": item.get("cooperation") or "未标注",
                "score": item.get("fused_score"),
                "source": item.get("source_name") or "未标注",
            }
            for index, item in enumerate(recommendations[:5], start=1)
        ]

    @staticmethod
    def _project_detail(project: dict[str, Any] | None) -> dict[str, Any]:
        if not project:
            return {}
        return {
            "project_id": project.get("project_id"),
            "name": project.get("name"),
            "track": project.get("track"),
            "technology": project.get("technology"),
            "stage": project.get("stage"),
            "cooperation": project.get("cooperation"),
            "summary": project.get("summary"),
            "core_technology": project.get("core_technology"),
            "application": project.get("application"),
            "source_name": project.get("source_name"),
            "source_url": project.get("source_url"),
            "score": project.get("fused_score"),
            "reason": project.get("reason"),
        }

    @staticmethod
    def _format_final_answer(outputs: dict[str, Any]) -> str:
        lines: list[str] = []
        search_results = outputs.get("searched_projects") or []
        if search_results:
            lines.append("我先按需求检索并排序，推荐结果如下：")
            for item in search_results[:5]:
                lines.append(
                    f"{item['rank']}. {item.get('name') or ''}：{item.get('technology') or '未标注'}，"
                    f"{item.get('stage') or '未标注'}，合作方式：{item.get('cooperation') or '未标注'}，"
                    f"综合分 {item.get('fused_score', 0)}。"
                )
        comparison = outputs.get("comparison_result") or []
        if comparison:
            lines.append("\n对比结果：")
            for item in comparison:
                lines.append(
                    f"- {item['rank']}. {item.get('name') or ''}：技术方向 {item['technology']}；"
                    f"阶段 {item['stage']}；合作方式 {item['cooperation']}；综合分 {item.get('score')}。"
                )
        detail = outputs.get("project_detail") or {}
        if detail:
            lines.append("\n指定项目详情：")
            lines.append(
                f"{detail.get('name') or ''} 属于 {detail.get('track') or '未标注赛道'}，"
                f"技术方向为 {detail.get('technology') or '未标注'}，阶段为 {detail.get('stage') or '未标注'}，"
                f"合作方式为 {detail.get('cooperation') or '未标注'}。"
            )
            if detail.get("summary"):
                lines.append(f"项目简介：{clean(detail['summary'])[:360]}")
            if detail.get("source_name"):
                lines.append(f"来源：{detail.get('source_name')} {detail.get('source_url') or ''}".strip())
        return "\n".join(lines).strip()

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        tasks = state.get("intent", {}).get("tasks") or []
        if not tasks:
            state["agent_trace"].append({"agent": "PlannerExecutorAgent", "mode": "not_required"})
            return state

        recommendations = state.get("recommendations", [])
        outputs: dict[str, Any] = {
            "tasks": tasks,
            "searched_projects": self._search_summary(recommendations),
        }
        executed: list[str] = ["search"]
        for task in tasks:
            sub_intent = clean(task.get("sub_intent"))
            if sub_intent == "compare":
                outputs["comparison_result"] = self._compare_projects(recommendations)
                executed.append("compare")
            elif sub_intent in {"detail", "explain", "source", "summary"}:
                selected = self._select_project(state["query"], recommendations)
                outputs["project_detail"] = self._project_detail(selected)
                executed.append("detail")

        outputs["final_answer"] = self._format_final_answer(outputs)
        state["planner_outputs"] = outputs
        state["agent_trace"].append(
            {
                "agent": "PlannerExecutorAgent",
                "mode": "executed",
                "executed": executed,
                "task_count": len(tasks),
            }
        )
        return state


class ResponseAgent:
    def __init__(self, chat_client: BailianClient) -> None:
        self.chat_client = chat_client

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        if state["should_retrieve"]:
            if state.get("planner_outputs"):
                answer = self.chat_client.answer_planned(
                    state["query"],
                    state.get("recommendations", []),
                    state["planner_outputs"],
                )
                mode = "bailian_planner" if self.chat_client.api_key else "planner_template"
            else:
                answer = self.chat_client.answer(state["query"], state.get("recommendations", []))
                mode = "bailian" if self.chat_client.api_key else "template_fallback"
        else:
            answer = self.chat_client.answer_dialogue(
                state["query"],
                state["intent"],
                state["session"],
                state["long_term_memory"],
            )
            mode = "bailian_dialogue" if self.chat_client.api_key else "dialogue_fallback"
        state["answer"] = answer
        state["answer_mode"] = mode
        state["agent_trace"].append({"agent": "ResponseAgent", "mode": mode})
        return state


class AgentHarness:
    def __init__(
        self,
        project_index: ProjectIndex,
        memory_store: MemoryStore,
        rag_client: PgvectorClient,
        chat_client: BailianClient,
    ) -> None:
        self.memory_store = memory_store
        self.chat_client = chat_client
        self.agents = [
            IntentAgent(chat_client),
            MemoryAgent(),
            RetrievalAgent(project_index, rag_client),
            MatchingAgent(),
            RecommendationAgent(project_index),
            EvidenceAgent(),
            PlannerExecutorAgent(),
            ResponseAgent(chat_client),
        ]

    def run(self, request: ChatRequest) -> dict[str, Any]:
        session_id = request.session_id or f"chat_{uuid.uuid4().hex[:12]}"
        session = self.memory_store.get_session(session_id)
        long_term = self.memory_store.get_long_term(request.user_id)
        state: dict[str, Any] = {
            "trace_id": f"trace_{uuid.uuid4().hex[:12]}",
            "session": session,
            "long_term_memory": long_term,
            "query": request.query.strip(),
            "top_k": request.top_k,
            "agent_trace": [],
        }

        state = self.agents[0].run(state)
        if not state["should_retrieve"]:
            state = self.agents[-1].run(state)
            answer = state["answer"]
            session["updated_at"] = now_iso()
            session["turns"].append(
                {
                    "at": now_iso(),
                    "query": state["query"],
                    "intent": state["intent"],
                    "recommendation_ids": [],
                    "trace_id": state["trace_id"],
                }
            )
            self.memory_store.save(session, long_term)
            return {
                "session_id": session_id,
                "trace_id": state["trace_id"],
                "answer": answer,
                "answer_mode": state["answer_mode"],
                "intent": state["intent"],
                "short_term_memory": session["short_term_memory"],
                "long_term_memory": long_term,
                "retrieval_mode": "not_run",
                "agent_trace": state["agent_trace"],
                "results": [],
            }

        for agent in self.agents[1:-1]:
            state = agent.run(state)

        recommendations = state["recommendations"]
        state = self.agents[-1].run(state)
        answer = state["answer"]
        answer_mode = state["answer_mode"]
        session["updated_at"] = now_iso()
        session["short_term_memory"]["last_recommendation_ids"] = [item["project_id"] for item in recommendations]
        session["turns"].append(
            {
                "at": now_iso(),
                "query": state["query"],
                "intent": state["intent"],
                "recommendation_ids": session["short_term_memory"]["last_recommendation_ids"],
                "trace_id": state["trace_id"],
            }
        )
        self.memory_store.save(session, long_term)

        return {
            "session_id": session_id,
            "trace_id": state["trace_id"],
            "answer": answer,
            "intent": state["intent"],
            "short_term_memory": session["short_term_memory"],
            "long_term_memory": long_term,
            "retrieval_mode": state["retrieval_mode"],
            "answer_mode": answer_mode,
            "agent_trace": state["agent_trace"],
            "results": recommendations,
        }


project_index = ProjectIndex(DATA_FILE)
memory_store = MemoryStore(STATE_FILE)
rag_client = PgvectorClient()
chat_client = BailianClient()
harness = AgentHarness(project_index, memory_store, rag_client, chat_client)

app = FastAPI(title="安徽科创项目推荐 Demo")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def home() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "status": "ok",
        "projects": len(project_index.df),
        "mode": "anhui-project-recommendation-demo",
        "rag": {
            "database_url": DATABASE_URL.split("@")[-1],
            "knowledge_base_id": RAG_KB_ID,
            "tenant_id": RAG_TENANT_ID,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimensions": EMBEDDING_DIMENSIONS,
            "api_key_configured": rag_client.available,
            "chat_model": CHAT_MODEL,
        },
        "agents": [
            "IntentAgent",
            "MemoryAgent",
            "RetrievalAgent",
            "MatchingAgent",
            "RecommendationAgent",
            "EvidenceAgent",
            "ResponseAgent",
        ],
    }


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="请输入需求")
    return harness.run(request)


@app.post("/api/recommend")
def recommend(request: ChatRequest) -> dict[str, Any]:
    return harness.run(request)


@app.post("/api/v1/chat")
def chat_v1(request: ChatRequest) -> dict[str, Any]:
    return chat(request)
