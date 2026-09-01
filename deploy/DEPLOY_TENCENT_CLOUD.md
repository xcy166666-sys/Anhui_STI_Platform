# 安徽科创项目推荐 Demo 腾讯云部署

## 服务器

当前服务器信息：

- 公网 IP：`124.221.136.14`
- 系统：Ubuntu 22.04 LTS
- 用户：`ubuntu`
- Docker：26.1.3
- Docker Compose：v2.27.1
- 内存：约 3.3 GiB
- 系统盘：40G，当前可用约 22G

## 部署原则

代码进入 GitHub，业务数据不进入 GitHub。

线上机器人依赖两类数据：

- `server-data/anhui_data/cleaned/project_vectors_source.jsonl`：本地检索和项目详情展示所需的原始结构化文本。
- PostgreSQL/pgvector 数据库：RAG 向量检索所需的 `chunks` 等表。

## 第一次上服务器

在腾讯云 Web 终端执行：

```bash
sudo mkdir -p /opt/anhui-sti-platform
sudo chown ubuntu:ubuntu /opt/anhui-sti-platform
cd /opt/anhui-sti-platform
git clone https://github.com/xcy166666-sys/Anhui_STI_Platform.git .
cp deploy/.env.example .env
nano .env
```

把 `.env` 里的占位符改成真实值。API Key 和数据库密码只放服务器 `.env`，不要提交 GitHub。

## 上传数据

在服务器创建目录：

```bash
mkdir -p /opt/anhui-sti-platform/server-data/anhui_data/cleaned
mkdir -p /opt/anhui-sti-platform/server-data/backups
```

从本地上传 JSONL：

```powershell
scp "D:\Cording_V1.0\Anhui STI Investment Platform\integrated-demo\financial_rag_anhui\anhui_data\cleaned\project_vectors_source.jsonl" ubuntu@124.221.136.14:/opt/anhui-sti-platform/server-data/anhui_data/cleaned/
```

从本地导出 pgvector 数据库：

```powershell
cd "D:\Cording_V1.0\Anhui STI Investment Platform\integrated-demo\financial_rag_anhui"
.\deploy\backup_pgvector.ps1 -Container "integrated-demo-postgres-1" -Database "rag_center" -User "postgres"
```

如果本地 PostgreSQL 容器名不同，先运行：

```powershell
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Ports}}"
```

然后把生成的 dump 上传服务器：

```powershell
scp ".\deploy\backups\anhui_pgvector.dump" ubuntu@124.221.136.14:/opt/anhui-sti-platform/server-data/backups/
```

在服务器恢复数据库：

```bash
cd /opt/anhui-sti-platform
bash deploy/restore_pgvector.sh server-data/backups/anhui_pgvector.dump
```

## 启动

```bash
cd /opt/anhui-sti-platform
docker compose -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.prod.yml ps
curl http://127.0.0.1:8010/healthz
```

浏览器访问：

```text
http://124.221.136.14
```

腾讯云轻量应用服务器防火墙需要放行：

- TCP 22
- TCP 80
- TCP 443

第一版先走 HTTP。域名备案和 HTTPS 后续再加。

## GitHub Actions CI/CD

需要在 GitHub 仓库设置 Secrets：

- `CVM_HOST`：`124.221.136.14`
- `CVM_USER`：`ubuntu`
- `CVM_PORT`：`22`
- `CVM_SSH_PRIVATE_KEY`：部署专用 SSH 私钥
- `DEPLOY_PATH`：`/opt/anhui-sti-platform`

服务器上需要把对应公钥加入：

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
nano ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

以后推送到 `main` 后，GitHub Actions 会自动登录服务器，拉取最新代码并执行：

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

数据文件和数据库 volume 保留在服务器，不会被 GitHub 覆盖。
