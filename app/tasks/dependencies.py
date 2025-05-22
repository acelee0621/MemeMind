from sqlalchemy.ext.asyncio import AsyncSession

from app.source_doc.service import SourceDocumentService
from app.source_doc.repository import SourceDocumentRepository


def get_document_service_for_task(db_session: AsyncSession) -> SourceDocumentService:
    """为 Celery 任务创建 SourceDocumentService 实例。"""
    repository = SourceDocumentRepository(db_session)
    return SourceDocumentService(repository)