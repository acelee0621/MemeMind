
# MemeMind - 本地化 RAG 知识库演示系统（LangChain 架构）

🎯 **MemeMind** 是一个基于 FastAPI + LangChain 的本地化 RAG（检索增强生成）问答系统，结合 Gradio UI，可实现私有文档上传、智能分块、中文语义检索和精准问答，完全脱离互联网即可运行。

项目使用如下模型组合：

- **嵌入模型**：[BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5)
- **精排模型**：[BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- **问答模型**：[Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)

---

## ✨ 核心特性

✅ 支持多种格式的文档上传、自动解析与文本分块  
✅ 完整 RAG 链基于 LangChain 异步架构构建  
✅ 精排采用高精度中文交叉编码器  
✅ 支持类 ChatGPT 的本地 Qwen3 问答体验  
✅ 完全本地部署，保护隐私  
✅ Gradio 提供交互界面，开箱即用  
✅ 支持 Docker & TaskIQ 后台任务调度

---

## 🛠️ 技术栈

| 模块             | 技术/工具                                 |
|------------------|--------------------------------------------|
| 后端服务         | FastAPI + LangChain + SQLAlchemy           |
| 向量存储         | ChromaDB                                   |
| 嵌入模型         | BAAI/bge-large-zh-v1.5                      |
| 精排模型         | BAAI/bge-reranker-v2-m3                     |
| 大语言模型       | Qwen3-4B 本地运行                           |
| 文档解析         | Unstructured + LangChain loaders           |
| 任务队列         | TaskIQ + RabbitMQ + Redis                  |
| 用户界面         | Gradio                                     |
| 配置管理         | `.env` + Pydantic Settings                 |

---

## 🚀 快速开始

### 1️⃣ 克隆项目

```bash
git clone https://github.com/acelee0621/MemeMind.git
cd MemeMind
````

### 2️⃣ 安装依赖

推荐使用 Python 3.10+ 和 [`uv`](https://github.com/astral-sh/uv) 管理依赖：

```bash
uv venv
uv sync
```

### 3️⃣ 下载模型（可选但推荐）

```bash
# 嵌入模型
uv run huggingface-cli download BAAI/bge-large-zh-v1.5 --local-dir ./local_models/embedding/bge-large-zh-v1.5

# 精排模型
uv run huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir ./local_models/reranker/bge-reranker-v2-m3

# 问答模型
uv run huggingface-cli download Qwen/Qwen3-4B --local-dir ./local_models/llm/Qwen3-4B
```

### 4️⃣ 启动后端服务

```bash
uv run fastapi dev
```

打开浏览器访问 `http://localhost:8000/docs` 查看 API 文档。

### 5️⃣ 访问 Gradio UI（交互界面）

打开浏览器访问 `http://127.0.0.1:8000/gradio/` 即可打开 Gradio 界面。

---

## 📦 项目结构

```bash
.
├── app/
│   ├── chains/              # LangChain 模型与组件封装
│   ├── core/                # 配置、数据库初始化
│   ├── repository/          # 数据访问层
│   ├── services/            # 业务逻辑服务层
│   ├── api/                 # FastAPI 路由
│   ├── tasks/               # TaskIQ 后台任务
│   ├── ui/                  # Gradio 界面
│   └── main.py              # FastAPI 启动入口
├── local_models/            # 本地模型存储目录（需手动下载）
├── alembic/                 # 数据库迁移脚本
└── .env                     # 环境变量配置
```

---

## 🔄 RAG 流程

1. 上传并解析文档（支持 PDF、Word、TXT 等）
2. 文本自动分块，并存入 PostgreSQL
3. 使用 BGE 模型嵌入文本，保存到 ChromaDB 向量库
4. 用户提问：

   * 先用向量召回粗选文本块
   * 然后用交叉编码精排模型筛选最相关内容
   * 最终整理为 Qwen3 聊天格式 Prompt，调用本地 Qwen3-4B 生成回答

---

## 🧠 模型配置一览

| 模型类型      | 模型名称                    | 功能描述               |
| --------- | ----------------------- | ------------------ |
| Embedding | BAAI/bge-large-zh-v1.5  | 中文语义向量生成           |
| Reranker  | BAAI/bge-reranker-v2-m3 | 精准语义相似度排序          |
| LLM       | Qwen/Qwen3-4B           | 类 ChatGPT 的本地大模型问答 |

---

## 🔧 环境变量配置（`.env`）

```env
# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=mememind
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# Redis & RabbitMQ
REDIS_HOST=localhost:6379
RABBITMQ_HOST=localhost:5672
RABBITMQ_USER=user
RABBITMQ_PASSWORD=bitnami

# Chroma 向量数据库
CHROMA_HOST=localhost
CHROMA_PORT=5500
CHROMA_COLLECTION_NAME=mememind_rag_collection

# 模型路径
EMBEDDING_MODEL_PATH=local_models/embedding/bge-large-zh-v1.5
RERANKER_MODEL_PATH=local_models/reranker/bge-reranker-v2-m3
LLM_MODEL_PATH=local_models/llm/Qwen3-4B
```

---

## 🧪 健康检查接口

以下接口用于检查各服务是否运行正常：

* `GET /health/db`：PostgreSQL 状态
* `GET /health/redis`：Redis 状态
* `GET /health/rabbitmq`：RabbitMQ 状态
* `GET /health`：聚合检查接口

---

## 🤝 贡献 & 协议

欢迎通过 Issue 或 PR 参与贡献！

本项目使用 **MIT 开源协议**。

