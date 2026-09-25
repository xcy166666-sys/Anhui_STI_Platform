"""Import cleaned Anhui project records into the relational project master table.

This script is idempotent by project_id. It imports every source field through
metadata.raw_record while also exposing commonly filtered columns as typed
relational columns.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text


ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else Path("/")
DEFAULT_SOURCE = ROOT / "anhui_data" / "cleaned" / "project_vectors_source.jsonl"


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def load_records(path: Path, offset: int, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if line_no < offset:
                continue
            if limit and len(records) >= limit:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("project_id"):
                records.append(record)
    return records


def ensure_schema(engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS anhui_projects (
                    project_id VARCHAR(128) PRIMARY KEY,
                    project_name TEXT,
                    track TEXT,
                    technology TEXT,
                    stage TEXT,
                    source_name TEXT,
                    source_id TEXT,
                    city TEXT,
                    district TEXT,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    raw_record JSONB NOT NULL DEFAULT '{}'::jsonb,
                    vector_ready BOOLEAN NOT NULL DEFAULT TRUE,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
        )
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_anhui_projects_track ON anhui_projects (track)"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_anhui_projects_technology ON anhui_projects (technology)"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_anhui_projects_source ON anhui_projects (source_name)"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_anhui_projects_metadata ON anhui_projects USING GIN (metadata)"))


def metadata_of(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata") or {}
    return metadata if isinstance(metadata, dict) else {}


def upsert_records(engine, records: list[dict[str, Any]], batch_size: int) -> int:
    imported = 0
    statement = text(
        """
        INSERT INTO anhui_projects (
            project_id, project_name, track, technology, stage, source_name,
            source_id, city, district, content, metadata, raw_record, vector_ready
        )
        VALUES (
            :project_id, :project_name, :track, :technology, :stage, :source_name,
            :source_id, :city, :district, :content, CAST(:metadata AS JSONB),
            CAST(:raw_record AS JSONB), :vector_ready
        )
        ON CONFLICT (project_id) DO UPDATE SET
            project_name = EXCLUDED.project_name,
            track = EXCLUDED.track,
            technology = EXCLUDED.technology,
            stage = EXCLUDED.stage,
            source_name = EXCLUDED.source_name,
            source_id = EXCLUDED.source_id,
            city = EXCLUDED.city,
            district = EXCLUDED.district,
            content = EXCLUDED.content,
            metadata = EXCLUDED.metadata,
            raw_record = EXCLUDED.raw_record,
            vector_ready = EXCLUDED.vector_ready,
            updated_at = NOW()
        """
    )
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        payloads: list[dict[str, Any]] = []
        for item in batch:
            metadata = metadata_of(item)
            raw_record = metadata.get("raw_record") or {}
            payloads.append(
                {
                    "project_id": clean(item.get("project_id")),
                    "project_name": clean(metadata.get("project_name") or raw_record.get("project_name")),
                    "track": clean(metadata.get("track") or raw_record.get("industry_field")),
                    "technology": clean(metadata.get("technology") or raw_record.get("technology_field")),
                    "stage": clean(metadata.get("stage") or raw_record.get("project_type")),
                    "source_name": clean(metadata.get("source_name") or raw_record.get("source_name")),
                    "source_id": clean(metadata.get("source_id") or raw_record.get("source_id")),
                    "city": clean(raw_record.get("city")),
                    "district": clean(raw_record.get("district")),
                    "content": clean(item.get("content")),
                    "metadata": json.dumps(metadata, ensure_ascii=False),
                    "raw_record": json.dumps(raw_record, ensure_ascii=False),
                    "vector_ready": bool(item.get("vector_ready", True)),
                }
            )
        with engine.begin() as connection:
            connection.execute(statement, payloads)
        imported += len(batch)
        print(f"imported {imported}/{len(records)}", flush=True)
    return imported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", ""))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    if not args.database_url:
        raise SystemExit("DATABASE_URL is required")
    if not args.source.exists():
        raise SystemExit(f"source not found: {args.source}")

    started = time.time()
    records = load_records(args.source, args.offset, args.limit)
    engine = create_engine(args.database_url, pool_pre_ping=True)
    ensure_schema(engine)
    imported = upsert_records(engine, records, args.batch_size)
    with engine.connect() as connection:
        count = connection.execute(text("SELECT COUNT(*) FROM anhui_projects")).scalar_one()
    print(
        json.dumps(
            {
                "source": str(args.source),
                "records": len(records),
                "imported": imported,
                "anhui_projects_count": count,
                "elapsed_seconds": round(time.time() - started, 1),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
