from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[2]
TRACE_DIR = ROOT / "logs" / "traces"


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_json(v) for v in value[:50]]
    if isinstance(value, tuple):
        return [_safe_json(v) for v in value[:50]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _state_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    intent = state.get("intent")
    if isinstance(intent, dict):
        intent_name = intent.get("name")
        intent_confidence = intent.get("confidence")
    else:
        intent_name = intent
        intent_confidence = None
    return {
        "trace_id": state.get("trace_id"),
        "query": state.get("query"),
        "intent": intent_name,
        "intent_confidence": intent_confidence,
        "should_retrieve": state.get("should_retrieve"),
        "retrieval_mode": state.get("retrieval_mode"),
        "candidates": len(state.get("candidates", []) or []),
        "scored_candidates": len(state.get("scored_candidates", []) or []),
        "recommendations": [
            {
                "project_id": item.get("project_id"),
                "name": item.get("name"),
                "fused_score": item.get("fused_score"),
                "rag_score": item.get("rag_score"),
                "traditional_score": item.get("traditional_score"),
            }
            for item in (state.get("recommendations", []) or [])[:5]
        ],
        "answer_mode": state.get("answer_mode"),
    }


class AnhuiTraceRecorder:
    """Records per-agent stage traces locally and optionally sends events to Langfuse."""

    def __init__(self) -> None:
        self.langfuse = None
        self.langfuse_enabled = bool(
            os.getenv("LANGFUSE_PUBLIC_KEY")
            and os.getenv("LANGFUSE_SECRET_KEY")
            and os.getenv("LANGFUSE_BASE_URL")
        )
        if self.langfuse_enabled:
            try:
                from langfuse import Langfuse

                self.langfuse = Langfuse(
                    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                    base_url=os.getenv("LANGFUSE_BASE_URL"),
                )
            except Exception:
                self.langfuse = None
                self.langfuse_enabled = False

    @contextmanager
    def stage(self, trace_id: str, stage_name: str, state: dict[str, Any]) -> Iterator[None]:
        started = time.perf_counter()
        record: dict[str, Any] = {
            "trace_id": trace_id,
            "stage": stage_name,
            "status": "ok",
            "started_at": time.time(),
            "input": _state_snapshot(state),
        }
        try:
            yield
        except Exception as exc:
            record["status"] = "error"
            record["error"] = repr(exc)
            raise
        finally:
            record["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
            record["output"] = _state_snapshot(state)
            self._write_local(record)
            self._write_langfuse(record)

    def _write_local(self, record: dict[str, Any]) -> None:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        path = TRACE_DIR / f"{time.strftime('%Y%m%d')}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_safe_json(record), ensure_ascii=False) + "\n")

    def _write_langfuse(self, record: dict[str, Any]) -> None:
        if not self.langfuse:
            return
        try:
            metadata = {
                "trace_id": record["trace_id"],
                "stage": record["stage"],
                "status": record["status"],
                "duration_ms": record["duration_ms"],
            }
            name = f"anhui.{record['stage']}"
            if hasattr(self.langfuse, "create_event"):
                try:
                    self.langfuse.create_event(
                        name=name,
                        input=record.get("input"),
                        output=record.get("output"),
                        metadata=metadata,
                    )
                except TypeError:
                    self.langfuse.create_event(name=name, metadata=metadata)
            elif hasattr(self.langfuse, "event"):
                self.langfuse.event(name=name, metadata=metadata)
            if hasattr(self.langfuse, "flush"):
                self.langfuse.flush()
        except Exception:
            return

