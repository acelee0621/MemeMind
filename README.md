[中文文档](https://github.com/acelee0621/mememind/blob/main/README_zh.md)


## MemeMind - Local RAG Knowledge Base Demo

🎯 **MemeMind** is a local RAG (Retrieval-Augmented Generation) knowledge base demo built with FastAPI, featuring Gradio as the user interface. It allows users to quickly experience the power of LLM-based question answering using a local knowledge base. This project integrates:

* **Vector Search**: Powered by [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).
* **Reranking**: Uses [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) for precise result reranking.
* **Answer Generation**: Leverages [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) for final answers.
* **Document Storage**: Utilizes MinIO as an object storage service.
* **Document Parsing**: Employs `unstructured` for file parsing and chunking.

---

## ✨ Key Features

✅ Supports multi-format document uploads and parsing to build a flexible local knowledge base
✅ User-friendly interactive UI powered by Gradio
✅ Fully local deployment, no internet required
✅ Lightweight model choices tailored for personal hardware
✅ Easy Docker deployment and dependency management

---

## 🛠️ Tech Stack

| Module           | Technology                     |
| ---------------- | ------------------------------ |
| Backend          | FastAPI, SQLAlchemy, Alembic   |
| Vector Search    | ChromaDB, Qwen3-Embedding-0.6B |
| Reranking        | Qwen3-Reranker-0.6B            |
| Generation Model | Qwen2.5-1.5B-Instruct          |
| Document Storage | MinIO                          |
| Document Parsing | unstructured                   |
| Task Queue       | Celery, RabbitMQ               |
| Dependency Mgmt  | uv                             |
| UI               | Gradio                         |

---

## 🚀 Getting Started

### 1️⃣ Clone the repository

```bash
git clone https://github.com/acelee0621/MemeMind.git
cd MemeMind
```

### 2️⃣ Install dependencies

It’s recommended to use Python 3.10+ and Poetry (or uv) for dependency management.

```bash
uv venv
uv sync
```

### 3️⃣ Start the FastAPI server

```bash
uv run fastapi dev
```

Once started, visit `http://localhost:8000/docs` for the interactive API docs or launch the Gradio UI.

### 4️⃣ Launch the Gradio UI

```bash
python app/ui/gradio_interface.py
```

---

## 📦 Project Structure

```bash
.
├── app/
│   ├── core/                # Core modules (model loading, DB, config)
│   ├── query/               # Query and RAG services
│   ├── source_doc/          # Document upload and parsing
│   ├── text_chunk/          # Text chunk management
│   ├── ui/                  # Gradio UI
│   └── main.py              # FastAPI entry point
├── alembic/                 # DB migrations
├── README_zh.md             # Chinese README
└── README.md                # English README
```

---

## ⚙️ Key Functionality

### 📚 Document Upload & Parsing

* MinIO object storage for uploaded documents
* `unstructured` to parse PDFs, DOCX, TXT, and more

### 🔍 RAG Workflow

1. Generate embeddings with **Qwen3-Embedding-0.6B**
2. Perform vector search using ChromaDB
3. Rerank results with **Qwen3-Reranker-0.6B**
4. Answer generation with **Qwen2.5-1.5B-Instruct**

### 🖥️ Local Models

* All models are loaded locally—no internet connection needed
* CPU, MPS, and GPU supported for various devices

---

## 📝 Configuration

Set the following environment variables in a `.env` file:

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

## 🤝 Contributing & License

Contributions via Issues and PRs are welcome!
This project is licensed under the **MIT License**.
