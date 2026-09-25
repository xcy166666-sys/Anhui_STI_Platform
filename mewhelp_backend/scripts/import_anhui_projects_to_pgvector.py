"""Create the Anhui RAG schema and import project vectors into PostgreSQL.

The operation is idempotent by project_id. It is intended to run inside the
anhui-sti-app container, where the mounted JSONL and deployment environment
are both available.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import create_engine, text


ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else Path("/")
DEFAULT_SOURCE = ROOT / "anhui_data" / "cleaned" / "project_vectors_source.jsonl"
NAMESPACE = uuid.UUID("5d8c1d54-8e96-47c8-ae9e-4baf68efde39")


def stable_id(prefix: str, project_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{prefix}:{project_id}"))


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
            if record.get("project_id") and record.get("content"):
                records.append(record)
    return records


def ensure_schema(engine) -> None:
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS knowledge_bases (
                    id VARCHAR(36) PRIMARY KEY,
                    tenant_id VARCHAR(128) NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id VARCHAR(36) PRIMARY KEY,
                    tenant_id VARCHAR(128) NOT NULL,
                    kb_id VARCHAR(36) NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
                    title VARCHAR(255) NOT NULL,
                    source_type VARCHAR(64) NOT NULL DEFAULT 'excel_project',
                    status VARCHAR(32) NOT NULL DEFAULT 'success',
                    error_message TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id VARCHAR(36) PRIMARY KEY,
                    tenant_id VARCHAR(128) NOT NULL,
                    kb_id VARCHAR(36) NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
                    document_id VARCHAR(36) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    title VARCHAR(255) NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    embedding vector(1536),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
        )
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_chunks_kb_tenant ON chunks (tenant_id, kb_id)"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_documents_kb ON documents (tenant_id, kb_id)"))


def ensure_kb(engine, tenant_id: str, kb_id: str) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO knowledge_bases (id, tenant_id, name, description)
                VALUES (:id, :tenant_id, :name, :description)
                ON CONFLICT (id) DO UPDATE SET updated_at = NOW()
                """
            ),
            {
                "id": kb_id,
                "tenant_id": tenant_id,
                "name": "安徽科创项目库",
                "description": "Anhui STI project recommendation knowledge base",
            },
        )


async def embed_batch(client: httpx.AsyncClient, texts: list[str], *, model: str, dimensions: int, retries: int) -> list[list[float]]:
    payload: dict[str, Any] = {"model": model, "input": texts}
    if dimensions:
        payload["dimensions"] = dimensions
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = await client.post("/embeddings", json=payload)
            if response.status_code >= 400 and "dimensions" in payload:
                payload.pop("dimensions", None)
                response = await client.post("/embeddings", json=payload)
            response.raise_for_status()
            data = response.json().get("data", [])
            embeddings = [item["embedding"] for item in sorted(data, key=lambda item: item.get("index", 0))]
            if len(embeddings) != len(texts):
                raise RuntimeError(f"embedding count mismatch: {len(embeddings)} != {len(texts)}")
            return embeddings
        except Exception as exc:  # retry transient provider/network failures
            last_error = exc
            if attempt < retries:
                await asyncio.sleep(min(30, 2**attempt))
    raise RuntimeError(f"embedding failed after retries: {last_error}") from last_error


async def run(args: argparse.Namespace) -> dict[str, Any]:
    records = load_records(args.source, args.offset, args.limit)
    if not records:
        raise SystemExit("No valid records found")
    engine = create_engine(args.database_url, pool_pre_ping=True)
    ensure_schema(engine)
    ensure_kb(engine, args.tenant_id, args.kb_id)

    headers = {"Authorization": f"Bearer {args.api_key}"}
    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/"), headers=headers, timeout=90) as client:
        imported = 0
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            embeddings = await embed_batch(
                client,
                [item["content"] for item in batch],
                model=args.embedding_model,
                dimensions=args.embedding_dimensions,
                retries=args.retries,
            )
            with engine.begin() as connection:
                for item, embedding in zip(batch, embeddings, strict=True):
                    project_id = str(item["project_id"])
                    document_id = stable_id("document", project_id)
                    chunk_id = stable_id("chunk", project_id)
                    metadata = json.dumps(item.get("metadata") or {}, ensure_ascii=False)
                    vector = "[" + ",".join(str(value) for value in embedding) + "]"
                    connection.execute(
                        text(
                            """
                            INSERT INTO documents (id, tenant_id, kb_id, title, source_type, status, error_message)
                            VALUES (:id, :tenant_id, :kb_id, :title, 'excel_project', 'success', NULL)
                            ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title,
                                status = 'success', error_message = NULL, updated_at = NOW()
                            """
                        ),
                        {"id": document_id, "tenant_id": args.tenant_id, "kb_id": args.kb_id,
                         "title": (item.get("metadata") or {}).get("project_name") or project_id},
                    )
                    connection.execute(
                        text(
                            """
                            INSERT INTO chunks (id, tenant_id, kb_id, document_id, title, content, metadata, embedding)
                            VALUES (:id, :tenant_id, :kb_id, :document_id, :title, :content,
                                    CAST(:metadata AS JSONB), CAST(:embedding AS vector))
                            ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title,
                                content = EXCLUDED.content, metadata = EXCLUDED.metadata,
                                embedding = EXCLUDED.embedding, updated_at = NOW()
                            """
                        ),
                        {"id": chunk_id, "tenant_id": args.tenant_id, "kb_id": args.kb_id,
                         "document_id": document_id,
                         "title": (item.get("metadata") or {}).get("project_name") or project_id,
                         "content": item["content"], "metadata": metadata, "embedding": vector},
                    )
            imported += len(batch)
            print(f"imported {imported}/{len(records)}", flush=True)
    with engine.connect() as connection:
        count = connection.execute(
            text("SELECT COUNT(*) FROM chunks WHERE tenant_id = :tenant_id AND kb_id = :kb_id"),
            {"tenant_id": args.tenant_id, "kb_id": args.kb_id},
        ).scalar_one()
    return {"records": len(records), "imported": imported, "pgvector_count": count,
            "knowledge_base_id": args.kb_id, "embedding_model": args.embedding_model,
            "embedding_dimensions": args.embedding_dimensions}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("MODEL_API_KEY", ""))
    parser.add_argument("--base-url", default=os.getenv("MODEL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"))
    parser.add_argument("--embedding-dimensions", type=int, default=int(os.getenv("EMBEDDING_DIMENSIONS", "1536")))
    parser.add_argument("--tenant-id", default=os.getenv("RAG_TENANT_ID", "anhui-sti"))
    parser.add_argument("--kb-id", default=os.getenv("RAG_KB_ID", "2bb8255a-4817-50f4-ba0d-49688b7fe8b5"))
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()
    if not args.database_url or not args.api_key:
        raise SystemExit("DATABASE_URL and MODEL_API_KEY are required")
    started = time.time()
    result = asyncio.run(run(args))
    result["elapsed_seconds"] = round(time.time() - started, 1)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
