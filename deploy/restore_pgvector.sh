#!/usr/bin/env bash
set -euo pipefail

DUMP_FILE="${1:-server-data/backups/anhui_pgvector.dump}"
DB="${POSTGRES_DB:-rag_center}"
USER="${POSTGRES_USER:-postgres}"

if [ ! -f "$DUMP_FILE" ]; then
  echo "dump file not found: $DUMP_FILE" >&2
  exit 1
fi

docker compose -f docker-compose.prod.yml up -d postgres
docker compose -f docker-compose.prod.yml exec -T postgres sh -c "until pg_isready -U '$USER' -d '$DB'; do sleep 2; done"

docker compose -f docker-compose.prod.yml cp "$DUMP_FILE" postgres:/tmp/anhui_pgvector.dump
docker compose -f docker-compose.prod.yml exec -T postgres pg_restore -U "$USER" -d "$DB" --clean --if-exists --no-owner /tmp/anhui_pgvector.dump

echo "restored pgvector database from $DUMP_FILE"
