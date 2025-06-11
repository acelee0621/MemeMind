import os
import uuid
import mimetypes
from datetime import datetime, timezone
from urllib.parse import quote

import aiofiles
from loguru import logger
from fastapi import HTTPException
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.celery_app import celery_app
from app.models.models import StorageType
from app.core.exceptions import NotFoundException
from app.schemas.schemas import (
    SourceDocumentCreate,
    SourceDocumentUpdate,
    SourceDocumentResponse,
)
from app.repository.doc_repository import SourceDocumentRepository
from app.services.chunk_service import TextChunkService
from app.chains.vector_store import get_chroma_vector_store


class SourceDocumentService:
    def __init__(
        self,
        doc_repository: SourceDocumentRepository,
        chunk_service: TextChunkService | None = None,
    ):
        self.repository = doc_repository
        self.chunk_service = chunk_service

    async def add_document(
        self, file_content: bytes, filename: str, content_type: str
    ) -> SourceDocumentResponse:
        """处理文档上传，将其保存到本地文件系统，并在数据库中创建记录。"""

        # ===== 1. 文件元数据处理 =====
        original_filename = filename or f"unnamed_{uuid.uuid4()}"
        client_provided_content_type = content_type
        guessed_type, _ = mimetypes.guess_type(original_filename)
        final_content_type = (
            guessed_type or client_provided_content_type or "application/octet-stream"
        )

        logger.info(
            f"收到文件上传: '{original_filename}', 客户端类型: {client_provided_content_type}, "
            f"最终类型: {final_content_type}"
        )

        size = len(file_content)

        # ===== 2. 保存文件到本地 =====
        storage_path = settings.LOCAL_STORAGE_PATH
        os.makedirs(storage_path, exist_ok=True)

        file_extension = os.path.splitext(original_filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        local_file_path = os.path.join(storage_path, unique_filename)

        try:
            async with aiofiles.open(local_file_path, "wb") as f:
                await f.write(file_content)
            logger.success(
                f"文件 '{original_filename}' 已成功保存到本地: '{local_file_path}'"
            )
        except Exception as e:
            logger.error(f"无法保存文件到本地 '{local_file_path}': {e}")
            raise HTTPException(status_code=500, detail=f"无法保存文件: {e}")

        # ===== 3. 在数据库中创建记录 =====
        document_data = SourceDocumentCreate(
            file_path=local_file_path,
            original_filename=original_filename,
            content_type=final_content_type,
            size=size,
        )
        try:
            new_document = await self.repository.create(document_data)

            # ===== 4. 触发异步处理任务 =====
            logger.info(f"发送文档处理任务到 Celery，文档 ID: {new_document.id}")
            celery_app.send_task(
                "app.tasks.document_task.process_document_task",
                args=[new_document.id],
                task_id=f"process_document_task_{new_document.id}",
            )
            return SourceDocumentResponse.model_validate(new_document)
        except Exception as e:
            # 数据库创建失败，需要清理已上传的本地文件，确保一致性
            logger.error(
                f"数据库记录创建失败: {e}。正在尝试清理物理文件 '{local_file_path}'..."
            )
            try:
                os.remove(local_file_path)
                logger.info(f"成功清理孤立文件: '{local_file_path}'")
            except OSError as cleanup_error:
                logger.critical(
                    f"清理孤立文件 '{local_file_path}' 失败: {cleanup_error}。需要手动介入！"
                )

            # 重新抛出异常
            raise HTTPException(status_code=500, detail=f"创建文档数据库记录失败: {e}")

    async def get_document(self, document_id: int) -> SourceDocumentResponse:
        """获取单个文档的详细信息。"""
        document = await self.repository.get_by_id(document_id)
        return SourceDocumentResponse.model_validate(document)

    async def get_documents(
        self, order_by: str | None, limit: int, offset: int
    ) -> list[SourceDocumentResponse]:
        """获取文档列表。"""
        documents = await self.repository.get_all(
            order_by=order_by, limit=limit, offset=offset
        )
        return [SourceDocumentResponse.model_validate(doc) for doc in documents]

    async def delete_document(self, document_id: int) -> None:
        """
        从数据库、向量库和本地文件系统完整地删除一个文档。
        """
        logger.info(f"开始执行删除文档的完整流程，ID: {document_id}")

        document = await self.repository.get_by_id(document_id)

        # 为了从向量库删除，我们还是需要先获取ID
        chunk_ids_to_delete = await self.chunk_service.get_chunk_ids_by_document_id(
            document_id
        )
        if chunk_ids_to_delete:
            logger.info(
                f"文档 {document_id} 关联了 {len(chunk_ids_to_delete)} 个向量，准备从 ChromaDB 中删除。"
            )
            try:
                vector_store = get_chroma_vector_store()
                vector_store.delete(ids=[str(cid) for cid in chunk_ids_to_delete])
                logger.success(
                    f"已成功从 ChromaDB 中删除 {len(chunk_ids_to_delete)} 个关联向量。"
                )
            except Exception as e:
                logger.error(f"从 ChromaDB 删除向量时发生错误: {e}。")

        # 删除物理文件
        if document.storage_type == StorageType.LOCAL and os.path.exists(
            document.file_path
        ):
            os.remove(document.file_path)
            logger.info(f"成功删除本地物理文件: '{document.file_path}'")

        # 从 PostgreSQL 中删除文档记录 (这将级联删除所有文本块)
        await self.repository.delete(document_id)
        logger.success(
            f"已成功从 PostgreSQL 中删除文档记录 ID: {document_id} 及其关联的文本块。"
        )

    async def download_document(self, document_id: int) -> FileResponse:
        """提供本地存储文档的直接下载。"""
        logger.info(f"请求下载文档 ID: {document_id}")
        document = await self.repository.get_by_id(document_id)

        if document.storage_type != StorageType.LOCAL:
            logger.warning(
                f"拒绝下载，文档 {document.id} 类型为 '{document.storage_type}'，不支持下载。"
            )
            raise HTTPException(
                status_code=400, detail="非本地存储的文档不支持直接下载。"
            )

        local_file_path = document.file_path
        if not os.path.exists(local_file_path):
            logger.error(
                f"文件在数据库中存在，但在物理位置上不存在: '{local_file_path}'"
            )
            raise NotFoundException("文件在存储中丢失，请联系管理员。")

        safe_filename = quote(document.original_filename)
        return FileResponse(
            path=local_file_path,
            filename=document.original_filename,
            media_type=document.content_type,
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{safe_filename}"
            },
        )

    async def update_document_processing_info(
        self,
        document_id: int,
        status: str | None = None,
        processed_at: datetime | None = None,
        number_of_chunks: int | None = None,
        error_message: str | None = None,
        set_processed_now: bool = False,
    ) -> SourceDocumentResponse:
        """更新文档的处理状态信息。"""
        actual_processed_at = processed_at
        if set_processed_now:
            actual_processed_at = datetime.now(timezone.utc)

        update_payload = SourceDocumentUpdate(
            status=status,
            processed_at=actual_processed_at,
            number_of_chunks=number_of_chunks,
            error_message=error_message,
        )

        try:
            updated_document = await self.repository.update(
                data=update_payload, document_id=document_id
            )
            return SourceDocumentResponse.model_validate(updated_document)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except NotFoundException:
            raise
