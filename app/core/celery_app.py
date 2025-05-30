from celery import Celery
from app.core.config import settings


CELERY_BROKER_URL = f"amqp://{settings.RABBITMQ_USER}:{settings.RABBITMQ_PASSWORD}@{settings.RABBITMQ_HOST}//"


CELERY_RESULT_BACKEND = f"redis://{settings.REDIS_HOST}/2"


celery_app = Celery(
    "mememind",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["app.tasks.document_task"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    result_expires=3600,
    task_routes={"app.tasks.document_task.*": {"queue": "document_queue"}},
)


# uv run celery -A app.core.celery_app worker --loglevel=info --pool=threads -Q celery,document_queue --autoscale=4,2
