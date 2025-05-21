import os

import redis

from app.core.celery_app import celery_app

redis_host = os.getenv("REDIS_HOST", "localhost:6379")
REDIS_URL = f"redis://{redis_host}/0"

redis_client = redis.from_url(
    REDIS_URL,
    health_check_interval=30,
)


@celery_app.task(name="app.tasks.document_task.process_document_task")
def process_document_task(document_id: int):
    pass
