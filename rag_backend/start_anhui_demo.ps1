param(
    [int]$Port = 8010
)

Set-Location $PSScriptRoot
python -m uvicorn app.anhui_demo:app --app-dir . --host 127.0.0.1 --port $Port
