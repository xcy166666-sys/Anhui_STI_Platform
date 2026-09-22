from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


class MemoryType(str, Enum):
    PROFILE = "profile"
    EPISODE = "episode"
    TASK = "task"
    SKILL = "skill"


class MemoryOperationType(str, Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    MERGE = "MERGE"
    IGNORE = "IGNORE"
    INVALIDATE = "INVALIDATE"
    DELETE = "DELETE"


class ShortTermMemoryState(BaseModel):
    industries: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    stages: list[str] = Field(default_factory=list)
    cooperation: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    last_slots: dict[str, Any] = Field(default_factory=dict)
    last_query: str = ""
    last_raw_query: str = ""
    last_recommendation_ids: list[str] = Field(default_factory=list)
    excluded_project_ids: list[str] = Field(default_factory=list)

    task_id: str = ""
    current_task: dict[str, Any] = Field(default_factory=dict)
    current_constraints: dict[str, Any] = Field(default_factory=dict)
    selected_project_ids: list[str] = Field(default_factory=list)
    recent_tool_results: list[dict[str, Any]] = Field(default_factory=list)
    conversation_summary: str = ""
    updated_at: str = Field(default_factory=utc_now_iso)


class SessionMemoryDocument(BaseModel):
    session_id: str
    user_id: str = ""
    task_id: str = ""
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    turns: list[dict[str, Any]] = Field(default_factory=list)
    short_term_memory: ShortTermMemoryState = Field(default_factory=ShortTermMemoryState)


class UserLongTermState(BaseModel):
    user_id: str
    preferred_industries: list[str] = Field(default_factory=list)
    preferred_technologies: list[str] = Field(default_factory=list)
    preferred_stages: list[str] = Field(default_factory=list)
    preferred_cooperation: list[str] = Field(default_factory=list)
    preferred_regions: list[str] = Field(default_factory=list)
    history_queries: list[str] = Field(default_factory=list)
    updated_at: str = Field(default_factory=utc_now_iso)


class LongTermMemoryItem(BaseModel):
    id: str
    tenant_id: str
    user_id: str
    session_id: str = ""
    task_id: str = ""
    memory_type: MemoryType
    memory_key: str
    content: str = ""
    structured_value: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    status: str = "active"
    valid_from: str = Field(default_factory=utc_now_iso)
    valid_to: str | None = None
    source_type: str = ""
    source_id: str = ""
    evidence: str = ""
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    last_accessed_at: str | None = None
    access_count: int = 0
    version: int = 1


class MemoryWriteCandidate(BaseModel):
    operation: MemoryOperationType
    memory_type: MemoryType
    memory_key: str
    value: Any = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: str = ""
    source_type: str = ""
    source_id: str = ""


class MemoryOperation(BaseModel):
    operation: MemoryOperationType
    memory_type: MemoryType
    memory_key: str
    value: Any = None
    target_memory_id: str = ""
    reason: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: str = ""


class MemoryRetrievalResult(BaseModel):
    memory_id: str
    memory_type: MemoryType
    memory_key: str
    content: str = ""
    structured_value: dict[str, Any] = Field(default_factory=dict)
    retrieval_score: float = 0.0
    rerank_score: float = 0.0
    source: str = ""
