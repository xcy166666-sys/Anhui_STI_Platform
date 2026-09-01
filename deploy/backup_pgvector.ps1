param(
    [string]$Container = "integrated-demo-postgres-1",
    [string]$Database = "rag_center",
    [string]$User = "postgres",
    [string]$Output = "deploy/backups/anhui_pgvector.dump"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Output) | Out-Null

docker exec $Container pg_dump -U $User -d $Database -Fc -f /tmp/anhui_pgvector.dump
docker cp "${Container}:/tmp/anhui_pgvector.dump" $Output

Write-Host "Created database backup: $Output"
