from __future__ import annotations

import asyncio
import json
import os
import re
import hashlib
import hmac
import secrets
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
from sqlalchemy import create_engine, text

from app.anhui_observability import AnhuiTraceRecorder
from app.core.agent_memory import (
    AgentMemoryService,
    MemoryContextBuilder,
    MemoryRetriever,
    MemoryRouter,
    MemoryTurnService,
    MemoryWriter,
)
from app.core import intent as unified_intent


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


def char_ngrams(text: str, *, min_n: int = 2, max_n: int = 4, limit: int = 500) -> set[str]:
    normalized = re.sub(r"\s+", "", text.lower())
    if not normalized:
        return set()
    grams: set[str] = set()
    for n in range(min_n, max_n + 1):
        if len(normalized) < n:
            continue
        for index in range(len(normalized) - n + 1):
            grams.add(normalized[index : index + n])
            if len(grams) >= limit:
                return grams
    return grams


def unique(values: list[str], limit: int = 12) -> list[str]:
    out: list[str] = []
    for value in values:
        value = value.strip()
        if value and value not in out:
            out.append(value)
        if len(out) >= limit:
            break
    return out


def contains_any(text_value: str, keywords: list[str]) -> bool:
    return any(keyword and keyword in text_value for keyword in keywords)


def looks_like_project_search(query: str) -> bool:
    text_value = query.strip()
    if not text_value:
        return False
    action_terms = [
        "找",
        "查",
        "查找",
        "检索",
        "搜索",
        "推荐",
        "匹配",
        "筛选",
        "有没有",
        "介绍",
        "了解",
        "find",
        "search",
        "recommend",
        "lookup",
    ]
    domain_terms = [
        "项目",
        "科创",
        "技术",
        "成果",
        "芯片",
        "新能源",
        "人工智能",
        "低空",
        "新材料",
        "生物医药",
        "高端装备",
        "量子",
        "测试板",
        "project",
        "technology",
        "chip",
    ]
    has_project_code = bool(re.search(r"\b[A-Za-z]{1,8}\s*\d{2,}[A-Za-z0-9-]*\b", text_value))
    has_action = contains_any(text_value, action_terms)
    has_domain = contains_any(text_value, domain_terms)
    return has_project_code or (has_action and has_domain)


def looks_like_reference_followup(query: str) -> bool:
    reference_terms = [
        "第一个",
        "第二个",
        "第三个",
        "第四个",
        "第五个",
        "刚才",
        "上一个",
        "这个",
        "那个",
        "它",
        "1号",
        "2号",
        "3号",
    ]
    return contains_any(query, reference_terms)


class ChatRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    session_id: str | None = None
    user_id: str = Field(default="demo-user", min_length=1, max_length=128)


class UserRegisterRequest(BaseModel):
    name: str = Field(min_length=2, max_length=50)
    organization: str = Field(min_length=2, max_length=120)
    mobile: str = Field(pattern=r"^1\d{10}$")
    role: str = Field(min_length=2, max_length=30)
    password: str = Field(min_length=8, max_length=128)


class UserLoginRequest(BaseModel):
    mobile: str = Field(pattern=r"^1\d{10}$")
    password: str = Field(min_length=8, max_length=128)


class DemandCreateRequest(BaseModel):
    ticket_no: str = Field(min_length=6, max_length=40)
    user_id: str = Field(min_length=1, max_length=128)
    session_id: str | None = Field(default=None, max_length=128)
    payload: dict[str, Any]


class DemandMessageRequest(BaseModel):
    role: str = Field(pattern=r"^(user|assistant|system)$")
    content: str = Field(min_length=1, max_length=12000)
    trace_id: str | None = Field(default=None, max_length=128)
    metadata: dict[str, Any] = Field(default_factory=dict)


def password_hash(password: str, salt: str | None = None) -> str:
    """Return a portable PBKDF2 record without persisting plain passwords."""
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("ascii"), 210_000)
    return f"pbkdf2_sha256$210000${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iterations, salt, _ = stored.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        expected = password_hash_with_iterations(password, salt, int(iterations))
        return hmac.compare_digest(expected, stored)
    except (TypeError, ValueError):
        return False


def password_hash_with_iterations(password: str, salt: str, iterations: int) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("ascii"), iterations)
    return f"pbkdf2_sha256${iterations}${salt}${digest.hex()}"


class UserStore:
    def __init__(self) -> None:
        self.engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    def initialize(self) -> None:
        with self.engine.begin() as connection:
            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS anhui_users (
                    id UUID PRIMARY KEY,
                    name VARCHAR(50) NOT NULL,
                    organization VARCHAR(120) NOT NULL,
                    mobile VARCHAR(11) UNIQUE NOT NULL,
                    role VARCHAR(30) NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_login_at TIMESTAMPTZ
                )
            """))

    @staticmethod
    def public_user(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(row["id"]),
            "name": row["name"],
            "org": row["organization"],
            "mobile": row["mobile"],
            "role": row["role"],
        }

    def register(self, request: UserRegisterRequest) -> dict[str, Any]:
        user_id = uuid.uuid4()
        try:
            with self.engine.begin() as connection:
                row = connection.execute(
                    text("""
                        INSERT INTO anhui_users (id, name, organization, mobile, role, password_hash)
                        VALUES (:id, :name, :organization, :mobile, :role, :password_hash)
                        RETURNING id, name, organization, mobile, role
                    """),
                    {
                        "id": user_id,
                        "name": request.name.strip(),
                        "organization": request.organization.strip(),
                        "mobile": request.mobile,
                        "role": request.role.strip(),
                        "password_hash": password_hash(request.password),
                    },
                ).mappings().one()
        except Exception as exc:
            if "unique" in str(exc).lower() or "duplicate" in str(exc).lower():
                raise HTTPException(status_code=409, detail="该手机号已注册，请直接登录") from exc
            raise
        return self.public_user(dict(row))

    def login(self, request: UserLoginRequest) -> dict[str, Any]:
        with self.engine.begin() as connection:
            row = connection.execute(
                text("""
                    SELECT id, name, organization, mobile, role, password_hash
                    FROM anhui_users WHERE mobile = :mobile
                """),
                {"mobile": request.mobile},
            ).mappings().first()
            if not row or not verify_password(request.password, row["password_hash"]):
                raise HTTPException(status_code=401, detail="手机号或密码不正确")
            connection.execute(
                text("UPDATE anhui_users SET last_login_at = NOW() WHERE id = :id"),
                {"id": row["id"]},
            )
        return self.public_user(dict(row))


class DemandStore:
    def __init__(self) -> None:
        self.engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    def initialize(self) -> None:
        with self.engine.begin() as connection:
            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS anhui_demands (
                    ticket_no VARCHAR(40) PRIMARY KEY,
                    user_id VARCHAR(128) NOT NULL,
                    session_id VARCHAR(128),
                    status VARCHAR(32) NOT NULL DEFAULT 'AI_RECOMMENDING',
                    fingerprint VARCHAR(64) NOT NULL,
                    payload JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """))
            connection.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_anhui_demands_user_updated
                ON anhui_demands (user_id, updated_at DESC)
            """))
            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS anhui_demand_messages (
                    id UUID PRIMARY KEY,
                    ticket_no VARCHAR(40) NOT NULL REFERENCES anhui_demands(ticket_no) ON DELETE CASCADE,
                    role VARCHAR(20) NOT NULL,
                    content TEXT NOT NULL,
                    trace_id VARCHAR(128),
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """))
            connection.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_anhui_demand_messages_ticket_created
                ON anhui_demand_messages (ticket_no, created_at ASC)
            """))

    @staticmethod
    def _fingerprint(payload: dict[str, Any]) -> str:
        keys = ["demandType", "track", "technology", "stage", "cooperation", "region", "keywords", "description"]
        material = "|".join(clean(payload.get(key)).lower() for key in keys)
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _public(row: dict[str, Any], messages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        payload = row.get("payload") or {}
        if isinstance(payload, str):
            payload = json.loads(payload)
        return {
            "ticket_no": row["ticket_no"],
            "user_id": row["user_id"],
            "session_id": row.get("session_id"),
            "status": row["status"],
            "payload": payload,
            "created_at": row["created_at"].isoformat() if hasattr(row["created_at"], "isoformat") else row["created_at"],
            "updated_at": row["updated_at"].isoformat() if hasattr(row["updated_at"], "isoformat") else row["updated_at"],
            "messages": messages or [],
        }

    def create(self, request: DemandCreateRequest) -> dict[str, Any]:
        payload = dict(request.payload)
        fingerprint = self._fingerprint(payload)
        with self.engine.begin() as connection:
            existing = connection.execute(
                text("""
                    SELECT ticket_no, user_id, session_id, status, payload, created_at, updated_at
                    FROM anhui_demands
                    WHERE user_id = :user_id AND fingerprint = :fingerprint AND status <> 'CLOSED'
                    ORDER BY updated_at DESC
                    LIMIT 1
                """),
                {"user_id": request.user_id, "fingerprint": fingerprint},
            ).mappings().first()
            if existing:
                return {**self.get(str(existing["ticket_no"])), "reused": True}

            row = connection.execute(
                text("""
                    INSERT INTO anhui_demands (ticket_no, user_id, session_id, fingerprint, payload)
                    VALUES (:ticket_no, :user_id, :session_id, :fingerprint, CAST(:payload AS JSONB))
                    ON CONFLICT (ticket_no) DO UPDATE SET
                        user_id = EXCLUDED.user_id,
                        session_id = EXCLUDED.session_id,
                        fingerprint = EXCLUDED.fingerprint,
                        payload = EXCLUDED.payload,
                        updated_at = NOW()
                    RETURNING ticket_no, user_id, session_id, status, payload, created_at, updated_at
                """),
                {
                    "ticket_no": request.ticket_no,
                    "user_id": request.user_id,
                    "session_id": request.session_id,
                    "fingerprint": fingerprint,
                    "payload": json.dumps(payload, ensure_ascii=False),
                },
            ).mappings().one()
        return {**self._public(dict(row)), "reused": False}

    def get(self, ticket_no: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                text("""
                    SELECT ticket_no, user_id, session_id, status, payload, created_at, updated_at
                    FROM anhui_demands WHERE ticket_no = :ticket_no
                """),
                {"ticket_no": ticket_no},
            ).mappings().first()
            if not row:
                raise HTTPException(status_code=404, detail="需求单不存在")
            message_rows = connection.execute(
                text("""
                    SELECT id, role, content, trace_id, metadata, created_at
                    FROM anhui_demand_messages
                    WHERE ticket_no = :ticket_no
                    ORDER BY created_at ASC
                """),
                {"ticket_no": ticket_no},
            ).mappings().all()
        messages = []
        for item in message_rows:
            metadata = item.get("metadata") or {}
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            messages.append(
                {
                    "id": str(item["id"]),
                    "role": item["role"],
                    "content": item["content"],
                    "trace_id": item.get("trace_id"),
                    "metadata": metadata,
                    "created_at": item["created_at"].isoformat() if hasattr(item["created_at"], "isoformat") else item["created_at"],
                }
            )
        return self._public(dict(row), messages)

    def add_message(self, ticket_no: str, request: DemandMessageRequest) -> dict[str, Any]:
        trace_id = request.trace_id or clean(request.metadata.get("trace_id")) or None
        with self.engine.begin() as connection:
            exists = connection.execute(
                text("SELECT 1 FROM anhui_demands WHERE ticket_no = :ticket_no"),
                {"ticket_no": ticket_no},
            ).first()
            if not exists:
                raise HTTPException(status_code=404, detail="需求单不存在")
            row = connection.execute(
                text("""
                    INSERT INTO anhui_demand_messages (id, ticket_no, role, content, trace_id, metadata)
                    VALUES (:id, :ticket_no, :role, :content, :trace_id, CAST(:metadata AS JSONB))
                    RETURNING id, role, content, trace_id, metadata, created_at
                """),
                {
                    "id": uuid.uuid4(),
                    "ticket_no": ticket_no,
                    "role": request.role,
                    "content": request.content,
                    "trace_id": trace_id,
                    "metadata": json.dumps(request.metadata, ensure_ascii=False),
                },
            ).mappings().one()
            connection.execute(
                text("""
                    UPDATE anhui_demands
                    SET updated_at = NOW(),
                        session_id = COALESCE(:session_id, session_id)
                    WHERE ticket_no = :ticket_no
                """),
                {"ticket_no": ticket_no, "session_id": request.metadata.get("session_id")},
            )
        return {
            "id": str(row["id"]),
            "role": row["role"],
            "content": row["content"],
            "trace_id": row.get("trace_id"),
            "metadata": row.get("metadata") or {},
            "created_at": row["created_at"].isoformat() if hasattr(row["created_at"], "isoformat") else row["created_at"],
        }


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
        query_tokens = token_set(query)
        query_grams = char_ngrams(query)
        candidates: list[dict[str, Any]] = []
        for idx, row in self.df.iterrows():
            text_value = clean(row.get("search_text"))
            text_tokens = token_set(text_value)
            token_score = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
            text_grams = char_ngrams(text_value)
            gram_score = len(query_grams & text_grams) / max(len(query_grams), 1)
            exact_bonus = 0.1 if query and query in text_value else 0.0
            score = min(1.0, 0.65 * token_score + 0.35 * gram_score + exact_bonus)
            candidates.append(
                {
                    "row_index": idx,
                    "rag_score": float(score),
                    "row": self.df.iloc[idx],
                }
            )
        return sorted(candidates, key=lambda item: item["rag_score"], reverse=True)[: max(top_k * 5, 20)]

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

    def projects_by_ids(self, project_ids: list[str]) -> list[dict[str, Any]]:
        projects: list[dict[str, Any]] = []
        for project_id in project_ids:
            row_index = self.id_to_index.get(clean(project_id))
            if row_index is None:
                continue
            projects.append(self.to_project(self.df.iloc[row_index]))
        return projects


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
        "result_followup",
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
        history = self._format_intent_history(context)
        try:
            unified_result = asyncio.run(unified_intent.classify(query, history))
            if self.api_key:
                refined_result = self._refine_unified_intent(query, context, unified_result)
                if refined_result:
                    unified_result = refined_result
            intent = self._normalize_intent(unified_result.get("intent"))
            should_retrieve = self._should_retrieve(intent, unified_result.get("should_retrieve"))
            if intent == "result_followup":
                should_retrieve = False
            return {
                "intent": intent,
                "sub_intent": clean(unified_result.get("sub_intent")) or self._infer_sub_intent(query, intent),
                "should_retrieve": should_retrieve,
                "confidence": float(unified_result.get("confidence", 0.5)),
                "reason": clean(unified_result.get("reason")),
                "candidate_intents": unified_result.get("candidate_intents", []),
                "is_multi_task": bool(unified_result.get("is_multi_task", False)),
                "tasks": unified_result.get("tasks", []),
                "slots": unified_result.get("slots", {}),
                "reference": unified_result.get("reference", {}),
                "intent_source": unified_result.get("intent_source", "unified"),
                "rule_intent": unified_result.get("rule_intent", ""),
                "rule_confidence": unified_result.get("rule_confidence", 0.0),
            }
        except Exception:
            pass

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

    def _refine_unified_intent(
        self,
        query: str,
        context: dict[str, Any],
        base_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        prompt = (
            "你是安徽科创项目推荐助手的意图精判器。只能输出 JSON，不要输出解释文本。"
            "允许的 intent: conversation_help, clarification, project_recommendation, "
            "refine_recommendation, result_followup, out_of_scope。"
            "Intent 表示用户要做什么；slots 表示筛选参数；reference 表示第几个/刚才那个/它；"
            "如果一句话包含多个动作，需要保留或修正 tasks 和 depends_on。"
            "低置信度时输出 clarification，不要强行选择项目推荐。"
            'JSON 格式: {"intent":"","sub_intent":"","should_retrieve":false,'
            '"confidence":0.0,"reason":"","is_multi_task":false,"tasks":[]}'
        )
        payload = {
            "query": query,
            "context": context,
            "base_intent_analysis": base_result,
        }
        try:
            content = self._chat(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=0.0,
                max_tokens=320,
            )
            parsed = json.loads(content)
        except Exception:
            return None
        intent = self._normalize_intent(parsed.get("intent"))
        confidence = float(parsed.get("confidence", base_result.get("confidence", 0.5)))
        if confidence < 0.45:
            intent = "clarification"
        tasks = parsed.get("tasks")
        if not isinstance(tasks, list) or not all(isinstance(item, dict) for item in tasks):
            tasks = base_result.get("tasks", [])
        return {
            **base_result,
            "intent": intent,
            "sub_intent": clean(parsed.get("sub_intent")) or base_result.get("sub_intent", ""),
            "should_retrieve": self._should_retrieve(intent, parsed.get("should_retrieve")),
            "confidence": confidence,
            "reason": clean(parsed.get("reason")) or base_result.get("reason", ""),
            "is_multi_task": bool(parsed.get("is_multi_task", bool(tasks))),
            "tasks": tasks,
            "intent_source": "bailian_unified_refine",
        }

    @staticmethod
    def _format_intent_history(context: dict[str, Any]) -> str:
        lines: list[str] = []
        if context.get("previous_intent"):
            lines.append(f"previous_intent: {context['previous_intent']}")
        if context.get("last_query"):
            lines.append(f"last_query: {context['last_query']}")
        for item in context.get("history_queries", [])[-5:]:
            if item:
                lines.append(f"history_query: {item}")
        return "\n".join(lines)

    @classmethod
    def _semantic_candidates(cls, query: str, *, top_k: int = 3) -> list[dict[str, Any]]:
        descriptions: dict[str, str] = {}
        query_grams = char_ngrams(query, limit=300)
        query_tokens = token_set(query)
        for item in cls.INTENT_REGISTRY:
            intent = item["intent"]
            descriptions[intent] = item["description"]
        if not clean(query):
            return []
        best: dict[str, float] = {}
        for item in cls.INTENT_REGISTRY:
            intent = item["intent"]
            for example in item["examples"]:
                candidate_text = f"{item['description']} {example}"
                candidate_grams = char_ngrams(candidate_text, limit=300)
                candidate_tokens = token_set(candidate_text)
                gram_score = len(query_grams & candidate_grams) / max(len(query_grams), 1)
                token_score = len(query_tokens & candidate_tokens) / max(len(query_tokens), 1)
                score = 0.7 * gram_score + 0.3 * token_score
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
        if intent in {"conversation_help", "clarification", "result_followup", "out_of_scope"}:
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

    def answer_project_detail(self, query: str, projects: list[dict[str, Any]], intent_name: str) -> str:
        if not projects:
            return self.answer_dialogue(query, {"name": intent_name}, {"turns": []}, {})
        if not self.api_key:
            return self._template_answer(query, projects)
        payload = {
            "用户追问": query,
            "追问类型": intent_name,
            "关联项目": [
                {
                    "name": item.get("name"),
                    "track": item.get("track"),
                    "technology": item.get("technology"),
                    "stage": item.get("stage"),
                    "cooperation": item.get("cooperation"),
                    "summary": (item.get("summary") or "")[:1200],
                    "core_technology": (item.get("core_technology") or "")[:800],
                    "application": (item.get("application") or "")[:600],
                    "source_name": item.get("source_name"),
                    "source_url": item.get("source_url"),
                    "fused_score": item.get("fused_score"),
                    "rag_score": item.get("rag_score"),
                    "traditional_score": item.get("traditional_score"),
                    "reason": item.get("reason"),
                }
                for item in projects[:5]
            ],
        }
        prompt = (
            "你是安徽科创项目顾问。用户正在追问上一轮推荐结果，不是在要求重新推荐。"
            "如果追问类型是 project_detail，请介绍指定项目的名称、技术方向、阶段、核心内容、合作方式和适配场景。"
            "如果追问类型是 explain_reason，请解释为什么上一轮推荐这些项目，依据只能来自给定项目字段和分数。"
            "不要说“推荐第一个项目”作为标题，不要编造数据库里没有的主体、金额、联系方式。中文回答，简洁但信息完整。"
        )
        try:
            return self._chat(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=0.2,
                max_tokens=800,
            )
        except Exception:
            return self._template_answer(query, projects)

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
        self.service = AgentMemoryService.from_json_file(path)

    def get_session(self, session_id: str, *, user_id: str = "") -> dict[str, Any]:
        return self.service.get_session(session_id, user_id=user_id)

    def get_long_term(self, user_id: str) -> dict[str, Any]:
        return self.service.get_long_term(user_id)

    def save(self, session: dict[str, Any], long_term: dict[str, Any]) -> None:
        self.service.save(session, long_term)

    def apply_operations(
        self,
        *,
        user_id: str,
        session_id: str,
        operations: list[Any],
    ) -> list[dict[str, Any]]:
        return self.service.apply_operations(
            user_id=user_id,
            session_id=session_id,
            operations=operations,
        )

    def status(self) -> dict[str, Any]:
        return self.service.status()


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
        forced_project_search = looks_like_project_search(state["query"])
        state["intent"] = {
            "name": result.get("intent", "clarification"),
            "sub_intent": result.get("sub_intent", ""),
            "confidence": result.get("confidence", 0.5),
            "reason": result.get("reason", ""),
            "candidate_intents": result.get("candidate_intents", []),
            "is_multi_task": bool(result.get("is_multi_task", False)),
            "tasks": result.get("tasks", []),
            "slots": result.get("slots", {}),
            "reference": result.get("reference", {}),
            "intent_source": result.get("intent_source", ""),
            "rule_intent": result.get("rule_intent", ""),
            "rule_confidence": result.get("rule_confidence", 0.0),
        }
        state["should_retrieve"] = bool(result.get("should_retrieve", False))
        last_ids = state["session"]["short_term_memory"].get("last_recommendation_ids", [])
        query = state["query"]
        if forced_project_search and not (last_ids and looks_like_reference_followup(query)):
            state["intent"] = {
                **state["intent"],
                "name": "project_recommendation",
                "sub_intent": state["intent"].get("sub_intent") or "keyword_search",
                "confidence": max(float(state["intent"].get("confidence", 0.0)), 0.9),
                "reason": "命中项目检索规则，强制进入 RAG 检索",
                "intent_source": "project_search_rule",
            }
            state["should_retrieve"] = True
        ordinal_reference = any(word in query for word in ["第一个", "第二个", "第三个", "第四个", "第五个", "第1个", "第2个", "第3个", "1号", "2号", "这个", "那个", "它", "刚才"])
        detail_followup = any(word in query for word in ["详细", "介绍", "具体", "展开", "说说", "讲讲", "是什么", "怎么样", "如何", "情况"])
        if last_ids and any(word in query for word in ["为什么", "为啥", "推荐理由", "依据", "凭什么"]):
            state["intent"] = {"name": "explain_reason", "sub_intent": "result_followup", "confidence": 0.9, "reason": "用户追问上一轮推荐依据"}
            state["should_retrieve"] = False
        elif last_ids and (detail_followup or ordinal_reference):
            state["intent"] = {"name": "project_detail", "sub_intent": "result_followup", "confidence": 0.9, "reason": "用户追问上一轮推荐项目详情"}
            state["should_retrieve"] = False
        state["agent_trace"].append(
            {
                "agent": "IntentAgent",
                "intent": state["intent"]["name"],
                "sub_intent": state["intent"].get("sub_intent", ""),
                "confidence": state["intent"].get("confidence", 0.0),
                "candidate_intents": state["intent"].get("candidate_intents", []),
                "is_multi_task": state["intent"].get("is_multi_task", False),
                "tasks": state["intent"].get("tasks", []),
                "slots": state["intent"].get("slots", {}),
                "reference": state["intent"].get("reference", {}),
                "intent_source": state["intent"].get("intent_source", ""),
                "rule_intent": state["intent"].get("rule_intent", ""),
                "rule_confidence": state["intent"].get("rule_confidence", 0.0),
                "should_retrieve": state["should_retrieve"],
            }
        )
        return state


class MemoryAgent:
    def __init__(self) -> None:
        self.turn_service = MemoryTurnService()

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

        state = self.turn_service.apply_turn(
            state,
            query=query,
            current_slots=current_slots,
            base_memory_terms=unique(industries + technologies + stages + cooperation),
            limit=limit,
            follow_up_markers=["其他", "还有", "换一批", "更多", "继续", "再找", "再来", "下一批", "只看", "排除", "不要", "过滤", "缩小"],
            context_reuse_markers=["只看", "筛选", "继续", "再找", "不要", "排除"],
            exclude_previous_markers=["换一批", "更多", "其他", "不要这些", "不要刚才", "排除刚才", "排除上次"],
        )
        state["agent_trace"].append(
            {
                "agent": "MemoryAgent",
                "slots": current_slots,
                "merged_slots": state["merged_slots"],
                "rewritten_query": state["expanded_query"],
                "terms": state["memory_terms"],
                "is_follow_up": state["is_follow_up"],
                "exclude_count": len(state["exclude_project_ids"]),
                "memory_service": "MemoryTurnService",
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
        self.project_index = project_index
        self.memory_store = memory_store
        self.chat_client = chat_client
        self.memory_router = MemoryRouter()
        self.memory_retriever = MemoryRetriever(memory_store.service)
        self.memory_context_builder = MemoryContextBuilder()
        self.memory_writer = MemoryWriter()
        self.trace_recorder = AnhuiTraceRecorder()
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

    def _run_agent(self, agent: Any, state: dict[str, Any]) -> dict[str, Any]:
        stage_name = agent.__class__.__name__
        with self.trace_recorder.stage(state["trace_id"], stage_name, state):
            return agent.run(state)

    @staticmethod
    def _select_referenced_ids(query: str, project_ids: list[str], *, explain_all: bool) -> list[str]:
        if not project_ids:
            return []
        ordinal_map = [
            (["第一个", "第一项", "第1个", "第1项", "1号"], 0),
            (["第二个", "第二项", "第2个", "第2项", "2号"], 1),
            (["第三个", "第三项", "第3个", "第3项", "3号"], 2),
            (["第四个", "第四项", "第4个", "第4项", "4号"], 3),
            (["第五个", "第五项", "第5个", "第5项", "5号"], 4),
        ]
        for markers, index in ordinal_map:
            if any(marker in query for marker in markers) and index < len(project_ids):
                return [project_ids[index]]
        return project_ids[:5] if explain_all else project_ids[:1]

    def _answer_project_followup(self, state: dict[str, Any]) -> dict[str, Any]:
        intent_name = state["intent"]["name"]
        last_ids = state["session"]["short_term_memory"].get("last_recommendation_ids", [])
        selected_ids = self._select_referenced_ids(
            state["query"],
            last_ids,
            explain_all=intent_name == "explain_reason",
        )
        projects = self.project_index.projects_by_ids(selected_ids)
        answer = self.chat_client.answer_project_detail(state["query"], projects, intent_name) if projects else self.chat_client.answer_dialogue(
            state["query"],
            state["intent"],
            state["session"],
            state["long_term_memory"],
        )
        state["answer"] = answer
        state["answer_mode"] = "bailian_project_context" if self.chat_client.api_key else "template_fallback"
        state["recommendations"] = projects
        state["agent_trace"].append(
            {
                "agent": "ProjectContextAgent",
                "intent": intent_name,
                "project_count": len(projects),
                "project_ids": selected_ids,
            }
        )
        return state

    def _write_memory(self, state: dict[str, Any], *, user_id: str, session_id: str) -> dict[str, Any]:
        try:
            operations = self.memory_writer.plan_state(state)
            results = self.memory_store.apply_operations(
                user_id=user_id,
                session_id=session_id,
                operations=operations,
            )
            trace = {
                "agent": "MemoryWriter",
                "operation_count": len(operations),
                "operations": [
                    {
                        "operation": item.operation.value,
                        "memory_type": item.memory_type.value,
                        "memory_key": item.memory_key,
                    }
                    for item in operations
                ],
                "results": results,
                "status": "ok",
            }
        except Exception as exc:
            trace = {
                "agent": "MemoryWriter",
                "operation_count": 0,
                "status": "fallback",
                "error": clean(str(exc))[:240],
            }
        state["agent_trace"].append(trace)
        return state

    def _retrieve_memory_context(
        self,
        state: dict[str, Any],
        *,
        user_id: str,
    ) -> dict[str, Any]:
        intent = state.get("intent") or {}
        memory_types = self.memory_router.select_types(
            intent.get("name", ""),
            intent.get("reference") or {},
        )
        task_id = str((state.get("session", {}).get("short_term_memory") or {}).get("task_id") or "")
        try:
            memories = self.memory_retriever.retrieve(
                user_id=user_id,
                memory_types=memory_types,
                query=state.get("query", ""),
                task_id=task_id,
                top_k=5,
            )
            context = self.memory_context_builder.build(state, memories=memories)
            state["memory_context"] = context
            state["agent_trace"].append(
                {
                    "agent": "MemoryRetriever",
                    "memory_types": memory_types,
                    "candidate_count": len(memories),
                    "task_id": task_id,
                    "status": "ok",
                }
            )
        except Exception as exc:
            state["memory_context"] = self.memory_context_builder.build(state)
            state["agent_trace"].append(
                {
                    "agent": "MemoryRetriever",
                    "memory_types": memory_types,
                    "candidate_count": 0,
                    "status": "fallback",
                    "error": clean(str(exc))[:240],
                }
            )
        return state

    def run(self, request: ChatRequest) -> dict[str, Any]:
        session_id = request.session_id or f"chat_{uuid.uuid4().hex[:12]}"
        session = self.memory_store.get_session(session_id, user_id=request.user_id)
        long_term = self.memory_store.get_long_term(request.user_id)
        state: dict[str, Any] = {
            "trace_id": f"trace_{uuid.uuid4().hex[:12]}",
            "session": session,
            "long_term_memory": long_term,
            "query": request.query.strip(),
            "top_k": request.top_k,
            "agent_trace": [],
        }

        state = self._run_agent(self.agents[0], state)
        state = self._retrieve_memory_context(state, user_id=request.user_id)
        if not state["should_retrieve"]:
            if state["intent"]["name"] in {"project_detail", "explain_reason"}:
                state = self._answer_project_followup(state)
            else:
                state = self._run_agent(self.agents[-1], state)
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
            state = self._write_memory(state, user_id=request.user_id, session_id=session_id)
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
                "results": state.get("recommendations", []),
            }

        for agent in self.agents[1:-1]:
            state = self._run_agent(agent, state)

        recommendations = state["recommendations"]
        state = self._run_agent(self.agents[-1], state)
        answer = state["answer"]
        answer_mode = state["answer_mode"]
        session["updated_at"] = now_iso()
        shown_ids = [item["project_id"] for item in recommendations]
        short_memory = session["short_term_memory"]
        short_memory["excluded_project_ids"] = unique(
            list(dict.fromkeys(short_memory.get("excluded_project_ids", []) + shown_ids)),
            limit=200,
        )
        short_memory["last_recommendation_ids"] = shown_ids
        session["turns"].append(
            {
                "at": now_iso(),
                "query": state["query"],
                "intent": state["intent"],
                "recommendation_ids": shown_ids,
                "trace_id": state["trace_id"],
            }
        )
        state = self._write_memory(state, user_id=request.user_id, session_id=session_id)
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
user_store = UserStore()
demand_store = DemandStore()

app = FastAPI(title="安徽科创项目推荐 Demo")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.on_event("startup")
def initialize_application() -> None:
    user_store.initialize()
    demand_store.initialize()


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
        "observability": {
            "local_trace_jsonl": str((ROOT / "logs" / "traces").as_posix()),
            "langfuse_configured": harness.trace_recorder.langfuse_enabled,
        },
        "memory": memory_store.status(),
        "agents": [
            "IntentAgent",
            "MemoryRouter",
            "MemoryRetriever",
            "MemoryAgent",
            "MemoryWriter",
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


@app.post("/api/auth/register", status_code=201)
def register(request: UserRegisterRequest) -> dict[str, Any]:
    return {"user": user_store.register(request)}


@app.post("/api/auth/login")
def login(request: UserLoginRequest) -> dict[str, Any]:
    return {"user": user_store.login(request)}


@app.post("/api/demands", status_code=201)
def create_demand(request: DemandCreateRequest) -> dict[str, Any]:
    return {"demand": demand_store.create(request)}


@app.get("/api/demands/{ticket_no}")
def get_demand(ticket_no: str) -> dict[str, Any]:
    return {"demand": demand_store.get(ticket_no)}


@app.post("/api/demands/{ticket_no}/messages", status_code=201)
def add_demand_message(ticket_no: str, request: DemandMessageRequest) -> dict[str, Any]:
    return {"message": demand_store.add_message(ticket_no, request)}


@app.post("/api/recommend")
def recommend(request: ChatRequest) -> dict[str, Any]:
    return harness.run(request)


@app.post("/api/v1/chat")
def chat_v1(request: ChatRequest) -> dict[str, Any]:
    return chat(request)
