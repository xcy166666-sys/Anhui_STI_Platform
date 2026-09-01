# Anhui STI Platform

安徽科创项目推荐聊天机器人 Demo。用户可以用自然语言描述产业方向、技术需求、项目阶段和合作方式，系统通过多 Agent 编排完成意图判断、会话记忆、RAG 检索、传统匹配、融合排序和证据检查。

## 核心能力

- 每轮消息先由百炼 `qwen-plus` 判断意图并生成自然语言回复。
- 7 个 Agent：Intent、Memory、Retrieval、Matching、Recommendation、Evidence、Response。
- 支持短期会话记忆和长期用户偏好。
- 支持 pgvector 向量检索，以及本地 TF-IDF 降级检索。
- 支持传统字段匹配与 RAG 分数融合推荐。
- 使用合肥天鹅湖全屏背景的三栏聊天工作台。

## 项目入口

- 后端：[rag_backend/app/anhui_demo.py](rag_backend/app/anhui_demo.py)
- 前端：[anhui_frontend/index.html](anhui_frontend/index.html)
- 启动脚本：[rag_backend/start_anhui_demo.ps1](rag_backend/start_anhui_demo.ps1)
- 详细说明：[README_ANHUI_DEMO.md](README_ANHUI_DEMO.md)
- 上游项目文档：[README_UPSTREAM_FINANCIAL_RAG.md](README_UPSTREAM_FINANCIAL_RAG.md)

## 数据准备

公开仓库不包含完整安徽项目数据和聊天状态，避免公开联系人及邮箱等字段。请在本机准备：

```text
anhui_data/cleaned/project_vectors_source.jsonl
```

该文件每行一条 JSON，至少包含 `project_id`、`content` 和 `metadata`。真实项目向量已导入 pgvector 时，还需要配置对应的知识库 ID、租户 ID 和数据库连接。

## 配置百炼

在 PowerShell 中设置环境变量，不要把真实 Key 写入代码或 `.env` 后提交：

```powershell
$env:MODEL_API_KEY = "你的百炼 API Key"
$env:MODEL_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:CHAT_MODEL = "qwen-plus"
$env:EMBEDDING_MODEL = "text-embedding-v4"
$env:EMBEDDING_DIMENSIONS = "1536"
```

## 启动

```powershell
cd rag_backend
.\start_anhui_demo.ps1
```

打开 `http://127.0.0.1:8010`。

未配置百炼或 pgvector 时，Demo 会回退到本地意图规则、模板回复和 TF-IDF 检索；该模式仅用于本地调试。

## 上游来源

本仓库基于 [Serein-81/financial_rag](https://github.com/Serein-81/financial_rag) 的完整历史构建，原项目文档和主体代码均予以保留。安徽推荐功能位于上述独立入口中。
