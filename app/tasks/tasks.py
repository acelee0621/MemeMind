from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.chains.ingestion_pipeline import run_ingestion_pipeline
from app.core.taskiq_app import broker  # 你的 TaskIQ broker 实例
from app.core.database import get_db_for_taskiq


@broker.task
async def process_document_task(
    document_id: int, session: AsyncSession = get_db_for_taskiq
) -> dict:    
    logger.info(f"Taskiq 开始处理文档 ID: {document_id} ")
    try:
        result = await run_ingestion_pipeline(document_id, session)
        logger.success(f" 文档 {document_id} 已由 Taskiq 处理成功: {result}")
        return result
    except Exception as e:
        logger.exception(f"Taskiq 处理文档 {document_id} 失败: {e}")
        raise
