from loguru import logger
from app.chains.ingestion_pipeline import run_ingestion_pipeline
from app.core.taskiq_app import broker  # 你的 TaskIQ broker 实例
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db_for_taskiq


@broker.task
async def process_document_task(
    document_id: int, task_id: str, session: AsyncSession = get_db_for_taskiq
) -> dict:
    task_id_log_prefix = f"[TaskIQ Task ID: {task_id}]"
    logger.info(f"{task_id_log_prefix} 开始处理文档 ID: {document_id}")
    try:
        result = await run_ingestion_pipeline(document_id, task_id_log_prefix, session)
        logger.success(f"{task_id_log_prefix} 处理成功: {result}")
        return result
    except Exception as e:
        logger.exception(f"{task_id_log_prefix} 处理失败: {e}")
        raise
