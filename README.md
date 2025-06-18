[中文文档](https://github.com/acelee0621/mememind/blob/main/README_zh.md)

# MemeMind - Local RAG Knowledge Base Demo (LangChain Version)

🎯 **MemeMind** is a fully local Retrieval-Augmented Generation (RAG) system, built with FastAPI and LangChain. It features a Gradio UI and enables document-based question answering on your own machine with Chinese-optimized models.

This project combines modern RAG techniques with high-efficiency local deployment and supports fully offline usage. It leverages:

- **Embedding**: [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5)
- **Reranking**: [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- **LLM for Answering**: [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)

---

## ✨ Key Features

✅ Multi-format document upload & automatic chunking  
✅ LangChain-based modular pipeline with full async support  
✅ High-precision reranking with Chinese-optimized CrossEncoder  
✅ Local generation using Qwen3-4B with chat-style prompt tuning  
✅ Gradio-powered interactive interface  
✅ Fully offline, privacy-preserving setup  
✅ Docker & TaskIQ support for production task processing

---

## 🛠️ Tech Stack

| Module             | Technology                               |
|-------------------|-------------------------------------------|
| Backend           | FastAPI, LangChain, SQLAlchemy            |
| Vector Store      | ChromaDB                                  |
| Embeddings        | BAAI/bge-large-zh-v1.5                    |
| Reranker          | BAAI/bge-reranker-v2-m3                   |
| LLM               | Qwen3-4B (local, via Transformers)        |
| Document Parsing  | Unstructured + LangChain loaders          |
| Task Queue        | TaskIQ, RabbitMQ, Redis                   |
| UI                | Gradio                                    |
| Config & Env Mgmt | `.env` + Pydantic Settings                |

---

## 🚀 Getting Started

### 1️⃣ Clone the Repo

```bash
git clone https://github.com/acelee0621/MemeMind.git
cd MemeMind
````

### 2️⃣ Install Dependencies

Requires Python 3.10+ and [`uv`](https://github.com/astral-sh/uv) or `poetry`.

```bash
uv venv
uv sync
```

### 3️⃣ Prepare Models (Optional)

Download the models for offline use:

```bash
# Embedding model
uv run huggingface-cli download BAAI/bge-large-zh-v1.5 --local-dir ./local_models/embedding/bge-large-zh-v1.5

# Reranker
uv run huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir ./local_models/reranker/bge-reranker-v2-m3

# Qwen3 LLM
uv run huggingface-cli download Qwen/Qwen3-4B --local-dir ./local_models/llm/Qwen3-4B
```

### 4️⃣ Start Backend Server

```bash
uv run fastapi dev
```

Browse to `http://localhost:8000/docs` for API or `http://127.0.0.1:8000/gradio/` for Gradio UI.

---

## 📦 Project Structure

```bash
.
├── app/
│   ├── chains/              # LangChain RAG components
│   ├── core/                # Config, DB, and settings
│   ├── repository/          # SQLAlchemy-based repositories
│   ├── services/            # Business logic layer
│   ├── api/                 # FastAPI routes
│   ├── tasks/               # TaskIQ background tasks
│   ├── ui/                  # Gradio interface
│   └── main.py              # FastAPI app entry
├── local_models/            # Pre-downloaded models (optional)
├── alembic/                 # DB migrations
└── .env                     # Environment variables
```

---

## 🔄 RAG Pipeline

1. Upload & parse documents using `unstructured`
2. Split text into chunks
3. Store chunks in PostgreSQL and embed via **BGE Embedding**
4. Save vectors to **ChromaDB**
5. During query:

   * Retrieve top-K chunks
   * Rerank with **BGE Reranker**
   * Format into chat prompt (Qwen3-style)
   * Generate answer using **Qwen3-4B**

---

## 🧠 Models Used

| Role         | Model                   | Notes                              |
| ------------ | ----------------------- | ---------------------------------- |
| Embedding    | BAAI/bge-large-zh-v1.5  | Strong for Chinese dense retrieval |
| Reranking    | BAAI/bge-reranker-v2-m3 | Cross-encoder precision reranking  |
| LLM (Answer) | Qwen/Qwen3-4B           | Local, chat-format capable         |

---

## 🔧 Configuration

Set your `.env` file with environment variables like:

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=mememind
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

REDIS_HOST=localhost:6379
RABBITMQ_HOST=localhost:5672
RABBITMQ_USER=user
RABBITMQ_PASSWORD=bitnami

CHROMA_HOST=localhost
CHROMA_PORT=5500
CHROMA_COLLECTION_NAME=mememind_rag_collection

EMBEDDING_MODEL_PATH=local_models/embedding/bge-large-zh-v1.5
RERANKER_MODEL_PATH=local_models/reranker/bge-reranker-v2-m3
LLM_MODEL_PATH=local_models/llm/Qwen3-4B
```

---

## 🧪 Health Check

You can verify system status with:

* `GET /health/db`
* `GET /health/redis`
* `GET /health/rabbitmq`
* `GET /health` (aggregate)

---

## 📬 Contribution & License

This project is licensed under the **MIT License**.
Feel free to open Issues or PRs to contribute to MemeMind!

