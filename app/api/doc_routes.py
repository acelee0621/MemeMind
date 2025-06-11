# app/source_doc/routes.py

from typing import Annotated, List
from loguru import logger
from fastapi import APIRouter, Depends, UploadFile, File, status, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.doc_service import SourceDocumentService
from app.repository.doc_repository import SourceDocumentRepository
from app.schemas.schemas import SourceDocumentResponse
from app.schemas.param_schemas import DocumentQueryParams
from app.core.exceptions import NotFoundException

router = APIRouter(prefix="/documents", tags=["Documents"])


def get_document_service(session: AsyncSession = Depends(get_db)) -> SourceDocumentService:
    repository = SourceDocumentRepository(session)
    return SourceDocumentService(repository)

@router.post(
    "",
    response_model=SourceDocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="上传文档",
)
async def upload_document_route(
    file: Annotated[UploadFile, File(description="要上传的源文件")],
    service: SourceDocumentService = Depends(get_document_service),
):
    """
    处理单个文件的上传。
    成功后，将触发后台的 Celery 任务进行文档处理。
    """
    logger.info(f"接收到文件上传请求: {file.filename}")
    try:
        created_document = await service.add_document(file=file)
        logger.success(f"文件 '{file.filename}' 已成功创建记录，ID: {created_document.id}。处理任务已派发。")
        return created_document
    except Exception as e:
        logger.error(f"文件上传失败: {file.filename}, 错误: {e}", exc_info=True)
        # 将内部错误重新包装为标准的 HTTP 异常
        raise HTTPException(status_code=500, detail=f"文件上传过程中发生内部错误: {e}")

@router.get(
    "/{document_id}/download",
    response_class=FileResponse, # 直接指定响应类为 FileResponse
    summary="下载文档",
)
async def download_document_route(
    document_id: int, service: SourceDocumentService = Depends(get_document_service)
):
    """根据文档ID下载对应的原始文件。"""
    logger.info(f"请求下载文档 ID: {document_id}")
    try:
        # Service 层现在直接返回一个 FileResponse 对象
        return await service.download_document(document_id=document_id)
    except NotFoundException as e:
        logger.warning(f"下载失败，未找到文档 ID: {document_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"下载文档 {document_id} 时发生未知错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"下载文件时发生错误: {e}")

@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除文档",
)
async def delete_document_route(
    document_id: int, service: SourceDocumentService = Depends(get_document_service)
):
    """根据ID删除一个文档及其所有相关数据（包括物理文件和向量）。"""
    logger.info(f"请求删除文档 ID: {document_id}")
    try:
        await service.delete_document(document_id=document_id)
        logger.success(f"文档 ID: {document_id} 已成功删除。")
        # 成功时没有响应体
    except NotFoundException as e:
        logger.warning(f"删除失败，未找到文档 ID: {document_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"删除文档 {document_id} 时发生未知错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除文档时发生错误: {e}")

@router.get(
    "",
    response_model=List[SourceDocumentResponse],
    summary="获取所有文档信息",
)
async def get_all_documents_route(
    params: DocumentQueryParams = Depends(),
    service: SourceDocumentService = Depends(get_document_service),
):
    """获取所有文档的元数据列表，支持分页和排序。"""
    try:
        all_documents = await service.get_documents(
            order_by=params.order_by, limit=params.limit, offset=params.offset
        )
        return all_documents
    except Exception as e:
        logger.error(f"获取文档列表时发生错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="获取文档列表时发生错误。")

@router.get(
    "/{document_id}",
    response_model=SourceDocumentResponse,
    summary="根据ID获取单个文档信息",
)
async def get_document_route(
    document_id: int,
    service: SourceDocumentService = Depends(get_document_service),
):
    """
    获取单个文档的元数据信息。
    (已移除预签名URL的逻辑)
    """
    logger.info(f"请求获取文档元数据 ID: {document_id}")
    try:
        document = await service.get_document(document_id=document_id)
        return document
    except NotFoundException as e:
        logger.warning(f"获取元数据失败，未找到文档 ID: {document_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"获取文档 {document_id} 时发生未知错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取文档信息时发生错误: {e}")