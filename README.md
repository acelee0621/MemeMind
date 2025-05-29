[中文文档](https://github.com/acelee0621/mememind/blob/main/README_zh.md)

# MemeMind: Your Personal RAG Q&A System

MemeMind is a local Retrieval Augmented Generation (RAG) question-answering system. Upload your documents (TXT files currently supported) and ask questions against your knowledge base, powered by advanced embedding, reranking, and large language models running locally.

## Features

* **Document Upload & Management**: Securely upload and manage your source documents (TXT supported, with a Gcs架构 for PDF/Markdown). [cite: 163, 170, 174, 175, 176, 179, 185]
* **Asynchronous Document Processing**: Utilizes Celery for efficient background processing[cite: 512]:
    * **Parsing**: Extracts text content from uploaded files. [cite: 200]
    * **Chunking**: Splits documents into manageable text chunks using LangChain's `RecursiveCharacterTextSplitter`. [cite: 223]
    * **Text Embedding**: Generates embeddings using `maidalun1020/bce-embedding-base_v1` via Hugging Face Transformers. [cite: 40]
    * **Vector Storage**: Stores embeddings and metadata in ChromaDB for efficient similarity search. [cite: 31, 251]
* **Two-Stage Retrieval for Queries**:
    * **Initial Recall**: Fast vector search in ChromaDB to retrieve a large set of candidate chunks. [cite: 96, 105, 268]
    * **Reranking**: Employs `maidalun1020/bce-reranker-base_v1` to re-score and rank the candidates for higher precision. [cite: 60, 109]
* **LLM-Powered Q&A**: Integrates `tensorblock/Qwen_Qwen3-4B-GGUF` (via `llama-cpp-python`) to generate answers based on the retrieved and reranked context. [cite: 14, 53]
* **API Endpoints**:
    * Document management: Upload, list, download, delete, get presigned URLs. [cite: 154, 155, 156, 157, 159, 161]
    * Q&A: Submit queries and receive answers. [cite: 87]
    * Health check.
* **Flexible User Handling**: Designed to easily integrate full user authentication (e.g., `fastapi-users`), currently supports operations with or without a `current_user` context. [cite: 142, 547, 579]

## Architecture Overview

MemeMind employs a three-layer architecture (Routes, Services, Repositories) for its modules, ensuring separation of concerns and maintainability.

**Data Flow:**
1.  **Document Ingestion**: User uploads a document via API -> `SourceDocumentService` stores the file to MinIO and metadata to PostgreSQL -> Celery task `process_document_task` is triggered. [cite: 171, 576]
2.  **Asynchronous Processing (Celery `process_document_task`)**:
    * Downloads document from MinIO. [cite: 208, 209]
    * Parses text content. [cite: 200, 213]
    * Chunks text. [cite: 223, 224, 226]
    * Stores chunks in PostgreSQL via `TextChunkService`. [cite: 232, 233]
    * Generates embeddings for chunks via `EmbeddingService`. [cite: 240]
    * Stores embeddings and metadata in ChromaDB. [cite: 251, 253]
    * Updates `SourceDocument` status. [cite: 258]
3.  **Query & Answer (API Call to `/query/ask`)**:
    * User submits a query.
    * `QueryService` orchestrates:
        * Embeds the query. [cite: 92]
        * Performs initial recall from ChromaDB (`settings.INITIAL_RETRIEVAL_TOP_K` candidates). [cite: 105, 268]
        * Fetches candidate chunk texts from PostgreSQL via `TextChunkService`. [cite: 106, 107]
        * Reranks candidates using `RerankerService`. [cite: 109]
        * Selects final context (`settings.FINAL_CONTEXT_TOP_N` chunks). [cite: 112, 113]
        * Constructs a prompt with context and query. [cite: 118]
        * Calls `LLMService` to generate an answer. [cite: 119]
    * API returns the answer. [cite: 89]

**Key Components:**
* **FastAPI Backend**: Serves the API.
* **PostgreSQL**: Stores metadata for users, documents, text chunks, and conversations. [cite: 39, 515]
* **MinIO**: Object storage for uploaded documents. [cite: 514]
* **ChromaDB**: Vector database for storing and searching text embeddings. [cite: 31, 37]
* **Celery**: Asynchronous task queue for document processing. [cite: 512]
* **RabbitMQ**: Message broker for Celery. [cite: 36]
* **Redis**: Result backend for Celery. [cite: 661]

## Tech Stack

* **Backend**: Python 3.13[cite: 622], FastAPI[cite: 356, 634], SQLAlchemy (async with `asyncpg` [cite: 625]), Alembic [cite: 324, 623]
* **Asynchronous Tasks**: Celery[cite: 340, 629], `asgiref`
* **Storage**: PostgreSQL, MinIO, ChromaDB [cite: 347, 349]
* **ML/NLP**: Hugging Face Transformers[cite: 463], `llama-cpp-python`[cite: 383], LangChain (for text splitting)[cite: 380], Loguru (for logging) [cite: 40, 384]
    * Embedding Model: `maidalun1020/bce-embedding-base_v1` [cite: 40]
    * Reranker Model: `maidalun1020/bce-reranker-base_v1` [cite: 60]
    * LLM: `tensorblock/Qwen_Qwen3-4B-GGUF` (or your configured GGUF model) [cite: 38]
* **Containerization**: Docker, Docker Compose (for ChromaDB and other services)
* **Package Management**: `uv` [cite: 323]

## Getting Started

### Prerequisites

* Python 3.13+
* Docker and Docker Compose
* `uv` Python package manager (`pip install uv`)

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd MemeMind # Or your project directory
    ```

2.  **Configuration:**
    * Copy `.env.example` (if you create one) to `.env` and `.env.local`. Your `app/core/config.py` loads settings from these files. [cite: 38]
    * Update `.env` with your actual credentials and paths for:
        * PostgreSQL (`POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_DB`) [cite: 513]
        * RabbitMQ (`RABBITMQ_USER`, `RABBITMQ_PASSWORD`, `RABBITMQ_HOST`) [cite: 36]
        * Redis (`REDIS_HOST`) [cite: 36]
        * MinIO (`MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET`) [cite: 514]
        * ChromaDB (`CHROMA_HTTP_ENDPOINT`, `CHROMA_COLLECTION_NAME`) [cite: 37]
        * Model Paths (`EMBEDDING_MODEL_PATH`[cite: 40], `RERANKER_MODEL_PATH`[cite: 60], `LLM_MODEL_PATH` [cite: 38])
        * Chunking & Retrieval parameters (`CHUNK_SIZE`, `CHUNK_OVERLAP`, `INITIAL_RETRIEVAL_TOP_K`, `FINAL_CONTEXT_TOP_N`, `EMBEDDING_INSTRUCTION_FOR_RETRIEVAL`) [cite: 38]

3.  **Setup External Services (PostgreSQL, MinIO, RabbitMQ, Redis):**
    * Ensure instances of PostgreSQL, MinIO, RabbitMQ, and Redis are running and accessible with the configured credentials. You can run them using Docker.

4.  **Setup ChromaDB:**
    * Navigate to the directory containing your `docker-compose.yml` for ChromaDB.
    * Run: `docker-compose up -d`

5.  **Download ML Models:**
    * Download the GGUF versions of `maidalun1020/bce-embedding-base_v1`, `maidalun1020/bce-reranker-base_v1`, and `tensorblock/Qwen_Qwen3-4B-GGUF` (or your chosen LLM).
    * Place them into the respective paths configured in your settings (e.g., `app/embeddings/`, `app/llm_models/`). The `embedding.py` file contains a hint for a download command. [cite: 40]

6.  **Install Python Dependencies:**
    * It's recommended to use a virtual environment.
    * Use `uv` with the `uv.lock` file for deterministic installs:
        ```bash
        uv venv # Create virtual environment (if not already done)
        uv sync
        ```
        Alternatively, if you only have `pyproject.toml`:
        ```bash
        uv pip install -r pyproject.toml # Or simply uv pip install .
        ```

7.  **Database Migrations:**
    * Ensure your PostgreSQL connection URL is correctly set in `alembic.ini` (or dynamically via `env.py` as you've done [cite: 506, 508]).
    * Run Alembic migrations:
        ```bash
        uv run alembic upgrade head
        ```
        (Your `app/main.py` also attempts to run migrations via `run_migrations()`[cite: 307], which uses `subprocess`[cite: 303, 304, 305, 306].)

8.  **Ensure MinIO Bucket Exists:**
    * Your `app/main.py`'s lifespan event `ensure_minio_bucket_exists` will attempt to create the bucket if it doesn't exist. [cite: 307, 519, 520]

### Running the Application

1.  **Start FastAPI Server:**
    ```bash
    uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ```

2.  **Start Celery Worker:**
    * Open a new terminal.
    * Navigate to your project root.
    * Run the Celery worker (ensure RabbitMQ and Redis are running):
        ```bash
        # This command listens to 'document_queue' (for document processing and query processing) 
        # and the default 'celery' queue.
        uv run celery -A app.core.celery_app worker --loglevel=info --pool=threads -Q document_queue,celery --autoscale=4,2
        ```
        (Your `celery_app.py` routes `app.tasks.document_task.*` to `document_queue`[cite: 30]. If `process_query_task` is in a different module, adjust routing or worker queues accordingly.)

## API Endpoints Overview

* **Documents (`/documents`)**: [cite: 153]
    * `POST /`: Upload a document. [cite: 154]
    * `GET /`: List uploaded documents (supports pagination, ordering). [cite: 157, 158, 562]
    * `GET /{document_id}`: Get document details or a presigned URL for download. [cite: 159, 160, 161]
    * `GET /{document_id}/download`: Download a document. [cite: 155, 560]
    * `DELETE /{document_id}`: Delete a document and its associated data. [cite: 156, 561]
* **Query & RAG (`/query`)**: [cite: 84]
    * `POST /retrieve-chunks`: (For testing) Retrieve relevant text chunks for a query after二阶段检索. [cite: 86]
    * `POST /ask`: Submit a query and get an LLM-generated answer based on RAG. [cite: 87]
* **Health Check**:
    * `GET /health`: Check API health.

## How to Use (Example Workflow)

1.  **Upload a Document**: Send a `POST` request with a TXT file to `/documents`. The system will respond quickly, and a Celery task will start processing the document in the background.
2.  **(Wait for Processing)**: The Celery task will parse, chunk, embed, and store the document.
3.  **Ask a Question**: Send a `POST` request to `/query/ask` with your question in the request body:
    ```json
    {
      "query": "你的问题是什么？"
    }
    ```
4.  **Receive Answer**: The API will return a JSON response with the LLM-generated answer and optionally the context used. [cite: 85, 89]

## Future Work (Optional)

* Support for more document types (PDF, Markdown).
* Full user authentication and authorization.
* More advanced RAG techniques (query rewriting, HyDE, etc.).
* A dedicated user interface (frontend).
* Refactor Celery processing into chained tasks for better workflow management and error handling.
* Enhanced monitoring and observability.