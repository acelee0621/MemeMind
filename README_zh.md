
## MemeMind - 本地RAG知识库Demo

🎯 **MemeMind** 是一个基于 FastAPI 的 RAG（Retrieval-Augmented Generation）知识库Demo，集成了 Gradio 作为UI界面，适用于个人本地部署，便于用户快速体验大语言模型（LLM）在本地知识库检索与问答的能力。它集成了以下功能模块：

* **向量检索**：使用 [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) 进行文本嵌入。
* **精排重排**：使用 [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) 对检索结果进行精排。
* **生成式回答**：使用 [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) 模型生成最终答案。
* **文档存储**：使用 MinIO 作为对象存储服务。
* **文档解析**：使用 `unstructured` 进行文档解析和分块。

---

## ✨ 特性一览

✅ 支持多种文档上传与解析，灵活构建本地知识库
✅ 基于 Gradio 提供简洁直观的交互式 UI
✅ 完全本地化部署，支持离线运行
✅ 采用小参数量模型，适合本地硬件运行
✅ 方便的 Docker 部署和依赖管理

---

## 🛠️ 技术栈

| 模块   | 技术/工具                         |
| ---- | ----------------------------- |
| 后端   | FastAPI、SQLAlchemy、Alembic    |
| 向量检索 | ChromaDB、Qwen3-Embedding-0.6B |
| 精排   | Qwen3-Reranker-0.6B           |
| 生成模型 | Qwen2.5-1.5B-Instruct         |
| 文档存储 | MinIO                         |
| 文档解析 | unstructured                  |
| 任务队列 | Celery、RabbitMQ               |
| 依赖管理 | uv                           |
| 界面   | Gradio                        |

---

## 🚀 快速开始

### 1️⃣ 克隆仓库

```bash
git clone https://github.com/acelee0621/MemeMind.git
cd MemeMind
```

### 2️⃣ 安装依赖

推荐使用 Python 3.10+ 及 Poetry（或 uv）进行依赖管理。

```bash
uv venv
uv sync  # 如果使用uv
```

### 3️⃣ 启动服务

```bash
uv run fastapi dev
```

启动后访问 `http://localhost:8000/docs` 查看API文档，或访问 Gradio 界面。

### 4️⃣ 运行 Gradio UI

```bash
python app/ui/gradio_interface.py
```

---

## 📦 项目结构

```bash
.
├── app/
│   ├── core/                # 核心模块（模型加载、数据库、配置）
│   ├── query/               # 查询和RAG服务
│   ├── source_doc/          # 文档上传与解析
│   ├── text_chunk/          # 文本块管理
│   ├── ui/                  # Gradio UI
│   └── main.py              # FastAPI入口
├── alembic/                 # 数据库迁移
├── README_zh.md             # 中文README
└── README.md                # 英文README
```

---

## ⚙️ 主要功能介绍

### 📚 文档上传与解析

* 使用 MinIO 存储上传的文档
* 依赖 `unstructured` 对PDF、DOCX、TXT等多格式文件进行切块解析

### 🔍 RAG流程

1. 使用 **Qwen3-Embedding-0.6B** 生成向量
2. 通过 ChromaDB 进行向量检索
3. 使用 **Qwen3-Reranker-0.6B** 精排
4. 使用 **Qwen2.5-1.5B-Instruct** 生成回答

### 🖥️ 本地化模型

* 所有模型均支持本地加载，无需联网即可运行
* 适合个人PC部署，减少GPU占用（CPU和MPS也支持）

---

## 📝 配置说明

请在 `.env` 文件中配置以下关键参数：

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=mememind
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minio
MINIO_SECRET_KEY=miniosecret
MINIO_BUCKET=mememind

CHROMA_HTTP_ENDPOINT=http://localhost:5500
CHROMA_COLLECTION_NAME=mememind_rag_collection

RABBITMQ_HOST=localhost:5672
RABBITMQ_USER=user
RABBITMQ_PASSWORD=bitnami
```

---

## 🤝 贡献与许可

欢迎提出Issue和PR以改进本项目。
本项目基于 **MIT License** 开源。

