# MemeMind：你的个人 RAG 问答系统

MemeMind 是一个本地部署的检索增强生成（RAG）问答系统。您可以上传您的文档（目前主要支持 TXT 文件），并基于这些文档内容提出问题，系统将利用先进的 Embedding 模型、Reranker 模型以及本地运行的大语言模型来提供智能回答。

## 主要特性

* **文档上传与管理**：安全地上传和管理您的源文档（当前支持 TXT，架构已为 PDF/Markdown 等格式预留扩展空间）。 [cite: 163, 170, 174, 175, 176, 179, 185]
* **异步文档处理**：通过 Celery 实现高效的后台处理流程 [cite: 512]：
    * **内容解析**：从上传文件中提取文本。 [cite: 200]
    * **文本分块**：使用 LangChain 的 `RecursiveCharacterTextSplitter` 将文档分割成易于处理的文本块。 [cite: 223]
    * **文本向量化**：通过 Hugging Face Transformers 调用 `maidalun1020/bce-embedding-base_v1` 模型生成文本嵌入。 [cite: 40]
    * **向量存储**：将生成的向量及元数据存入 ChromaDB，以支持高效的相似性搜索。 [cite: 31, 251]
* **二阶段查询检索**：
    * **初步召回**：在 ChromaDB 中进行快速向量搜索，检索大量候选文本块。 [cite: 96, 105, 268]
    * **Reranker 精排**：使用 `maidalun1020/bce-reranker-base_v1` 模型对候选块进行重新评分和排序，以提高精度。 [cite: 60, 109]
* **LLM 智能问答**：集成 `tensorblock/Qwen_Qwen3-4B-GGUF` 模型（通过 `llama-cpp-python` 库加载），根据检索并精排后的上下文生成答案。 [cite: 14, 53]
* **API 接口**：
    * 文档管理：上传、列表、下载、删除、获取预签名URL。 [cite: 154, 155, 156, 157, 159, 161]
    * 问答：提交问题并获取答案。 [cite: 87]
    * 健康检查。
* **灵活的用户处理**：系统设计兼容未来的完整用户认证集成（例如 `fastapi-users`），当前支持在有或无 `current_user` 上下文的情况下操作。 [cite: 142, 547, 579]

## 系统架构概览

MemeMind 为其模块采用了经典的三层架构（路由层、服务层、仓库层），以确保职责分离和代码的可维护性。

**数据处理流程：**
1.  **文档注入**：用户通过 API 上传文档 -> `SourceDocumentService` 将文件存入 MinIO，元数据存入 PostgreSQL -> 触发 Celery 任务 `process_document_task`。 [cite: 171, 576]
2.  **异步处理 (Celery `process_document_task`)**：
    * 从 MinIO 下载文档。 [cite: 208, 209]
    * 解析文本内容。 [cite: 200, 213]
    * 文本分块。 [cite: 223, 224, 226]
    * 通过 `TextChunkService` 将文本块存入 PostgreSQL。 [cite: 232, 233]
    * 通过 `EmbeddingService`（实际是 `app.core.embedding` 中的函数）为文本块生成向量。 [cite: 240]
    * 将向量和元数据存入 ChromaDB。 [cite: 251, 253]
    * 更新 `SourceDocument` 状态。 [cite: 258]
3.  **查询与回答 (API 调用 `/query/ask`)**：
    * 用户提交查询。
    * `QueryService` 编排执行：
        * 向量化查询。 [cite: 92]
        * 从 ChromaDB 初步召回候选文本块 (`settings.INITIAL_RETRIEVAL_TOP_K` 数量)。 [cite: 105, 268]
        * 通过 `TextChunkService` 从 PostgreSQL 获取候选文本块内容。 [cite: 106, 107]
        * 使用 `RerankerService`（实际是 `app.core.reranker` 中的函数）对候选块进行精排。 [cite: 109]
        * 选取最终的上下文 (`settings.FINAL_CONTEXT_TOP_N` 数量的文本块)。 [cite: 112, 113]
        * 结合上下文和用户查询构建 Prompt。 [cite: 118]
        * 调用 `LLMService`（实际是 `app.core.llm_service` 中的函数）生成答案。 [cite: 119]
    * API 返回答案。 [cite: 89]

**关键组件：**
* **FastAPI 后端**：提供 API 服务。
* **PostgreSQL**：存储用户、文档、文本块、对话等元数据。 [cite: 39, 515]
* **MinIO**：对象存储，用于存放上传的原始文档。 [cite: 514]
* **ChromaDB**：向量数据库，存储和检索文本嵌入。 [cite: 31, 37]
* **Celery**：异步任务队列，用于文档处理。 [cite: 512]
* **RabbitMQ**：Celery 的消息代理。 [cite: 36]
* **Redis**：Celery 的结果后端。 [cite: 661]

## 技术栈

* **后端**：Python 3.13 [cite: 622]，FastAPI [cite: 356, 634]，SQLAlchemy (异步模式，配合 `asyncpg` [cite: 625])，Alembic (数据库迁移) [cite: 324, 623]
* **异步任务**：Celery [cite: 340, 629]，`asgiref`
* **存储**：PostgreSQL，MinIO，ChromaDB [cite: 347, 349]
* **机器学习/自然语言处理**：Hugging Face Transformers [cite: 463]，`llama-cpp-python` [cite: 383]，LangChain (用于文本分割) [cite: 380]，Loguru (日志) [cite: 40, 384]
    * Embedding 模型：`maidalun1020/bce-embedding-base_v1` [cite: 40]
    * Reranker 模型：`maidalun1020/bce-reranker-base_v1` [cite: 60]
    * 大语言模型 (LLM)：`tensorblock/Qwen_Qwen3-4B-GGUF` (或您配置的其他 GGUF 模型) [cite: 38]
* **容器化**：Docker, Docker Compose (用于 ChromaDB 及其他服务)
* **包管理**：`uv` [cite: 323]

## 开始使用

### 环境准备

* Python 3.13+
* Docker 和 Docker Compose
* `uv` Python 包管理器 (通过 `pip install uv` 安装)

### 安装与配置

1.  **克隆仓库：**
    ```bash
    git clone <your-repo-url>
    cd MemeMind # 或者你的项目目录
    ```

2.  **项目配置：**
    * 如果项目中有 `.env.example` 文件，请复制为 `.env` 和 `.env.local`。您的 `app/core/config.py` 会从这些文件加载配置。 [cite: 38]
    * 根据您的实际环境，更新 `.env` 文件中的以下配置：
        * PostgreSQL (`POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_DB`) [cite: 513]
        * RabbitMQ (`RABBITMQ_USER`, `RABBITMQ_PASSWORD`, `RABBITMQ_HOST`) [cite: 36]
        * Redis (`REDIS_HOST`) [cite: 36]
        * MinIO (`MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET`) [cite: 514]
        * ChromaDB (`CHROMA_HTTP_ENDPOINT`, `CHROMA_COLLECTION_NAME`) [cite: 37]
        * 模型路径 (`EMBEDDING_MODEL_PATH`[cite: 40], `RERANKER_MODEL_PATH`[cite: 60], `LLM_MODEL_PATH` [cite: 38])
        * 分块与检索参数 (`CHUNK_SIZE`, `CHUNK_OVERLAP`, `INITIAL_RETRIEVAL_TOP_K`, `FINAL_CONTEXT_TOP_N`, `EMBEDDING_INSTRUCTION_FOR_RETRIEVAL`) [cite: 38]

3.  **启动外部服务 (PostgreSQL, MinIO, RabbitMQ, Redis)：**
    * 确保这些服务正在运行，并且配置信息与 `.env` 文件一致。推荐使用 Docker 运行它们。

4.  **启动 ChromaDB：**
    * 在包含 ChromaDB 的 `docker-compose.yml` 文件的目录下运行：
        ```bash
        docker-compose up -d
        ```

5.  **下载机器学习模型：**
    * 下载您选择的 Embedding, Reranker, 和 LLM (GGUF格式) 模型文件。
    * 将它们放置到您在配置文件中指定的路径下 (例如 `app/embeddings/`, `app/llm_models/`)。`embedding.py` 文件中包含了下载模型的命令提示。 [cite: 40]

6.  **安装 Python 依赖：**
    * 建议使用虚拟环境。
    * 使用 `uv` 和 `uv.lock` 文件进行安装，以确保依赖版本的一致性：
        ```bash
        uv venv # 如果尚未创建虚拟环境
        uv sync
        ```
        或者，如果您只有 `pyproject.toml` 文件：
        ```bash
        uv pip install -r pyproject.toml # 或直接 uv pip install .
        ```

7.  **数据库迁移：**
    * 确保 `alembic.ini` 中的 PostgreSQL 连接 URL 配置正确（或如项目中通过 `env.py` 动态设置 [cite: 506, 508]）。
    * 运行 Alembic 迁移命令：
        ```bash
        uv run alembic upgrade head
        ```
        (项目中的 `app/main.py` 也会尝试在启动时通过 `run_migrations()` 执行迁移 [cite: 307, 303, 304, 305, 306]。)

8.  **确保 MinIO Bucket 存在：**
    * `app/main.py` 的 `lifespan` 事件中的 `ensure_minio_bucket_exists` 会在应用启动时尝试创建配置的 Bucket。 [cite: 307, 519, 520]

### 运行应用

1.  **启动 FastAPI 服务器：**
    ```bash
    uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ```

2.  **启动 Celery Worker：**
    * 打开一个新的终端。
    * 进入项目根目录。
    * 运行 Celery Worker (确保 RabbitMQ 和 Redis 正在运行)：
        ```bash
        # 此命令将监听 'document_queue' (用于文档处理和查询处理) 及默认的 'celery' 队列
        uv run celery -A app.core.celery_app worker --loglevel=info --pool=threads -Q document_queue,celery --autoscale=4,2
        ```
        (你的 `celery_app.py` 中将 `app.tasks.document_task.*` 的任务路由到了 `document_queue` [cite: 30]。如果 `process_query_task` 在不同的模块中，请相应调整路由或 Worker 监听的队列。)

## API 端点概览

* **文档管理 (`/documents`)**: [cite: 153]
    * `POST /`: 上传文档。 [cite: 154]
    * `GET /`: 列出已上传的文档 (支持分页、排序)。 [cite: 157, 158, 562]
    * `GET /{document_id}`: 获取文档详情或用于下载的预签名URL。 [cite: 159, 160, 161]
    * `GET /{document_id}/download`: 下载文档。 [cite: 155, 560]
    * `DELETE /{document_id}`: 删除文档及其关联数据。 [cite: 156, 561]
* **查询与问答 (`/query`)**: [cite: 84]
    * `POST /retrieve-chunks`: (测试用) 根据查询检索并返回相关的文本块（经过二阶段检索）。 [cite: 86]
    * `POST /ask`: 提交查询，获取由 LLM 生成的、基于 RAG 的回答。 [cite: 87]
* **健康检查**:
    * `GET /health`: 检查 API 服务健康状态。

## 使用示例 (简要流程)

1.  **上传文档**：向 `/documents` 端点发送 POST 请求，并附带一个 TXT 文件。API 会快速响应，后台 Celery 任务开始处理文档。
2.  **(等待处理)**：Celery 任务会执行文档解析、分块、向量化和存储。
3.  **提出问题**：向 `/query/ask` 端点发送 POST 请求，请求体中包含你的问题：
    ```json
    {
      "query": "你的问题是什么？"
    }
    ```
4.  **获取回答**：API 将返回一个包含 LLM 生成的答案以及可选的上下文信息的 JSON 响应。 [cite: 85, 89]

## 未来工作 (可选)

* 支持更多文档类型（如 PDF, Markdown）。
* 实现完整的用户认证和授权。
* 探索更高级的 RAG 技术（如查询重写、HyDE 等）。
* 开发一个专门的用户界面 (前端)。
* 将 Celery 处理流程重构为任务链，以获得更好的工作流管理和错误处理。
* 增强系统的监控和可观测性。