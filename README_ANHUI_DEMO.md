# 安徽科创项目推荐 Demo

这是从 `financial_rag` 复制出来的安徽科创项目推荐版入口，原仓库内容保留不动。

## 目录

- `rag_backend/app/anhui_demo.py`：安徽专用聊天、记忆、RAG、传统匹配、融合推荐入口
- `anhui_frontend/index.html`：聊天式前端
- `anhui_data/cleaned/project_vectors_source.jsonl`：安徽项目向量化源数据（仅本地保存，不提交公开仓库）

## 启动

先在同一个 PowerShell 窗口设置环境变量：

```powershell
$env:MODEL_API_KEY = "你的百炼 API Key"
$env:MODEL_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:CHAT_MODEL = "qwen-plus"
$env:EMBEDDING_MODEL = "text-embedding-v4"
$env:EMBEDDING_DIMENSIONS = "1536"
```

然后启动：

```powershell
cd "D:\Cording_V1.0\Anhui STI Investment Platform\integrated-demo\financial_rag_anhui\rag_backend"
python -m uvicorn app.anhui_demo:app --app-dir . --host 127.0.0.1 --port 8010
```

访问：

```text
http://127.0.0.1:8010
```

## 行为

- 每轮消息先由百炼大模型完成意图判断和自然语言回复
- 问候类消息由大模型正常对话，但不查项目库
- 需求明确时进入项目检索
- 支持多轮追问
- 支持短期记忆和长期偏好记忆
- 支持 pgvector 检索、传统匹配、融合排序

## 说明

如果 `MODEL_API_KEY` 没有注入当前终端，系统会退回到本地规则和模板回复；如果 pgvector 不可用，项目检索会退回本地 TF-IDF。降级模式仅用于调试。

完整项目数据包含负责人、联系人和邮箱等字段，因此被 `.gitignore` 排除。克隆仓库后需要从受控存储恢复到 `anhui_data/cleaned/project_vectors_source.jsonl`。
