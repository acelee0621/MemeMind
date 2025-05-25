from asyncio import to_thread

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager


from app.core.config import settings
# from app.core.logging import setup_logging, get_logger
from app.core.s3_client import ensure_minio_bucket_exists
from app.utils.migrations import run_migrations
from app.source_doc.routes import router as source_doc_router
from app.query.routes import router as query_router


# Run migrations on startup
run_migrations()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await to_thread(ensure_minio_bucket_exists, bucket_name=settings.MINIO_BUCKET)
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(source_doc_router)
app.include_router(query_router)

@app.get("/health")
async def health_check(response: Response):
    response.status_code = 200
    return {"status": "ok 👍 "}
