from typing import Optional
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

from fastapi import UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from botocore.exceptions import ClientError

from app.core.logging import get_logger
from app.core.config import settings
from app.core.s3_client import s3_client
from app.core.celery_app import celery_app
from app.core.exceptions import NotFoundException, ForbiddenException
from app.source_doc.repository import SourceDocumentRepository
from app.schemas.schemas import (
    SourceDocumentCreate,
    SourceDocumentUpdate,
    SourceDocumentResponse,
    PresignedUrlResponse,
)

logger = get_logger(__name__)


class SourceDocumentService:
    def __init__(self, repository: SourceDocumentRepository):
        """Service layer for document operations."""

        self.repository = repository

    async def add_document(
        self, file: UploadFile, current_user
    ) -> SourceDocumentResponse:
        # ===== 1. 文件元数据处理,从 UploadFile 获取文件元数据 =====
        original_filename = file.filename or f"unnamed_{uuid.uuid4()}"
        content_type = file.content_type or "application/octet-stream"

        try:
            file.file.seek(0, 2)  # 移动到文件末尾以获取大小
            size = file.file.tell()  # 获取文件大小
            file.file.seek(0)  # 重置文件指针到开头以供上传
        except (AttributeError, OSError) as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid file stream: {str(e)}"
            )

        # ===== 2. 生成唯一的 object_name 对象名称（使用 UUID + 文件扩展名） =====
        file_extension = (
            original_filename.rsplit(".", 1)[-1] if "." in original_filename else ""
        )
        object_name = f"documents/{uuid.uuid4()}.{file_extension}"

        # ===== 3. 使用 boto3 上传文件到 MinIO =====
        try:
            s3_client.upload_fileobj(
                Fileobj=file.file,
                Bucket=settings.MINIO_BUCKET,
                Key=object_name,
                ExtraArgs={"ContentType": content_type},
            )
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "UnknownError")
            logger.error(
                f"Failed to upload file {original_filename}: {error_code} - {str(e)}"
            )
            match error_code:
                case "404":
                    raise NotFoundException("Storage bucket does not exist")
                case "403":
                    raise ForbiddenException("Permission denied to upload file")
                case _:
                    raise HTTPException(
                        status_code=500, detail=f"S3 upload failed: {str(e)}"
                    )
        except Exception as e:
            logger.error(
                f"Unexpected error uploading file {original_filename}: {str(e)}"
            )
            raise HTTPException(status_code=500, detail=f"File upload error: {str(e)}")

        # ===== 4. 构造 SourceDocumentCreate 数据,并在数据库中创建记录 =====

        document_data = SourceDocumentCreate(
            object_name=object_name,
            bucket_name=settings.MINIO_BUCKET,
            original_filename=original_filename,
            content_type=content_type,
            size=size,
        )
        try:
            new_document = await self.repository.create(document_data, current_user)
            result = SourceDocumentResponse.model_validate(new_document)
            celery_app.send_task(
                "app.tasks.document_task.process_document_task",
                args=[new_document.id],
                task_id=f"process_document_task_{new_document.id}",
            )
            return result

        except Exception as e:
            # 数据库失败后尝试清理已上传的文件
            try:
                s3_client.delete_object(Bucket=settings.MINIO_BUCKET, Key=object_name)
                logger.info(
                    f"Cleaned up orphaned file {object_name} after database failure"
                )
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up {object_name}: {str(cleanup_error)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to save document: {str(e)}"
            )

    async def get_document(
        self, document_id: int, current_user
    ) -> SourceDocumentResponse:
        document = await self.repository.get_by_id(document_id, current_user)
        return SourceDocumentResponse.model_validate(document)

    async def get_documents(
        self,
        order_by: str | None,
        limit: int,
        offset: int,
        current_user,
    ) -> list[SourceDocumentResponse]:
        documents = await self.repository.get_all(
            order_by=order_by,
            limit=limit,
            offset=offset,
            current_user=current_user,
        )
        return [
            SourceDocumentResponse.model_validate(document) for document in documents
        ]

    async def delete_document(self, document_id: int, current_user) -> None:
        # 先删除数据库记录
        document = await self.get_document(
            document_id=document_id, current_user=current_user
        )
        await self.repository.delete(document.id, current_user)
        logger.info(f"Deleted document record {document_id} from database")

        # 再删除文件
        try:
            s3_client.delete_object(
                Bucket=document.bucket_name, Key=document.object_name
            )
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            logger.error(
                f"Failed to delete document {document_id}: {error_code} - {str(e)}"
            )
            # TODO: 可选择记录到日志或队列，异步清理，优化一致性
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error deleting document {document_id}: {str(e)}",
            )

    async def download_document(self, document_id: int, current_user):
        document = await self.get_document(
            document_id=document_id, current_user=current_user
        )

        try:
            s3_response = s3_client.get_object(
                Bucket=document.bucket_name, Key=document.object_name
            )
            file_stream = s3_response["Body"]
            safe_filename = quote(document.original_filename)
            return StreamingResponse(
                content=file_stream,
                media_type=document.content_type,
                headers={
                    "Content-Disposition": f"attachment; filename*=UTF-8''{safe_filename}"
                },
            )
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            logger.error(
                f"Failed to download document {document_id}: {error_code} - {str(e)}"
            )
            match error_code:
                case "404":
                    raise NotFoundException("File not found in storage")
                case "403":
                    raise ForbiddenException("Permission denied to access file")
                case _:
                    raise HTTPException(
                        status_code=500, detail="Failed to download file"
                    )
        except Exception as e:
            logger.error(
                f"Unexpected error downloading document {document_id}: {str(e)}"
            )
            raise HTTPException(status_code=500, detail="An unexpected error occurred")

    async def get_presigned_url(
        self, document_id: int, current_user
    ) -> PresignedUrlResponse:
        document = await self.get_document(
            document_id=document_id, current_user=current_user
        )
        expires_in = 60 * 60 * 24  # 24小时
        try:
            presigned_url = s3_client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": document.bucket_name,
                    "Key": document.object_name,
                },
                ExpiresIn=expires_in,
            )
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            logger.error(
                f"Failed to generate presigned URL for such document {document_id}: {error_code}"
            )
            match error_code:
                case "404":
                    raise NotFoundException("File not found in storage")
                case "403":
                    raise ForbiddenException("Permission denied to access file")
                case _:
                    raise HTTPException(
                        status_code=500, detail="Failed to generate presigned URL"
                    )

        expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        return PresignedUrlResponse(
            url=presigned_url,
            expires_at=expires_at,
            filename=document.original_filename,
            content_type=document.content_type,
            size=document.size,
            attachment_id=document_id,
        )

    async def update_document_processing_info(
        self,
        document_id: int,
        status: Optional[str] = None,
        processed_at: Optional[datetime] = None,
        number_of_chunks: Optional[int] = None,
        error_message: Optional[str] = None,  # 传入 None 来清除错误信息
        set_processed_now: bool = False,  # 便捷标志，用于将 processed_at 设置为当前时间
    ) -> SourceDocumentResponse:        

        # 确定 processed_at 的值
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
            logger.info(f"成功更新文档 ID: {document_id} 的处理信息")
            return SourceDocumentResponse.model_validate(updated_document)

        except ValueError as e: 
            logger.warning(f"为文档 {document_id} 调用更新，但无有效更改: {str(e)}")            
            raise HTTPException(
                status_code=400,
                detail=f"未提供有效字段进行更新或未检测到更改: {str(e)}",
            )
        except NotFoundException:            
            raise
        except Exception as e:            
            logger.error(f"更新文档 {document_id} 处理信息时发生意外错误: {str(e)}")
            raise HTTPException(
                status_code=500, detail="更新文档处理信息时发生意外错误。"
            )
