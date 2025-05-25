from typing import Annotated, Union

from loguru import logger
from fastapi import APIRouter, Depends, Query, UploadFile, File, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.source_doc.service import SourceDocumentService
from app.source_doc.repository import SourceDocumentRepository
from app.schemas.schemas import SourceDocumentResponse, PresignedUrlResponse
from app.schemas.param_schemas import DocumentQueryParams


router = APIRouter(prefix="/documents", tags=["Documents"])


def get_document_service(
    session: AsyncSession = Depends(get_db),
) -> SourceDocumentService:
    """Dependency for getting SourceDocumentService instance."""
    repository = SourceDocumentRepository(session)
    return SourceDocumentService(repository)


@router.post(
    "",
    response_model=SourceDocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload Document",
)
async def upload_document_route(
    file: Annotated[
        UploadFile, File(..., title="Source Document", description="Upload a file")
    ],
    service: SourceDocumentService = Depends(get_document_service),
    # current_user: UserResponse = Depends(get_current_user),
):
    try:
        created_document = await service.add_document(file=file, current_user=None)
        logger.info(f"Uploaded document {created_document.id}")
        return created_document  # 返回创建的附件信息
    except Exception as e:
        logger.error(f"Failed to upload document {created_document.id}: {str(e)}")
        raise


@router.get(
    "/{document_id}/download",
    response_class=StreamingResponse,
    summary="Download document",
)
async def download_attachment_route(
    document_id: int,
    service: SourceDocumentService = Depends(get_document_service),
    # current_user: UserResponse = Depends(get_current_user),
):
    try:
        response = await service.download_document(
            document_id=document_id, current_user=None
        )
        return response
    except Exception as e:
        logger.error(f"Failed to download document {document_id}: {str(e)}")
        raise


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an document",
)
async def delete_attachment_route(
    document_id: int,
    service: SourceDocumentService = Depends(get_document_service),
    # current_user: UserResponse = Depends(get_current_user),
):
    try:
        await service.delete_document(document_id=document_id, current_user=None)
        logger.info(f"Deleted document {document_id}")
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {str(e)}")
        raise


@router.get(
    "",
    response_model=list[SourceDocumentResponse],
    summary="Get all documents Info",
)
async def get_all_documents(
    params: DocumentQueryParams = Depends(),
    service: SourceDocumentService = Depends(get_document_service),
    # current_user: UserResponse = Depends(get_current_user),
) -> list[SourceDocumentResponse]:
    try:
        all_documents = await service.get_documents(
            order_by=params.order_by,
            limit=params.limit,
            offset=params.offset,
            current_user=None,
        )
        logger.info(f"Retrieved {len(all_documents)} documents")
        return all_documents
    except Exception as e:
        logger.error(f"Failed to fetch all documents: {str(e)}")
        raise


@router.get(
    "/{document_id}",
    response_model=Union[SourceDocumentResponse, PresignedUrlResponse],
    summary="Get document by id or pre-signed URL",
)
async def get_document(
    document_id: int,
    presigned: Annotated[
        bool, Query(description="If true, return pre-signed URL")
    ] = False,
    service: SourceDocumentService = Depends(get_document_service),
    # current_user: UserResponse = Depends(get_current_user),
) -> Union[SourceDocumentResponse, PresignedUrlResponse]:
    if presigned:
        try:
            response = await service.get_presigned_url(document_id, current_user=None)
            logger.info(f"Generated pre-signed URL for document {document_id}")
            return response
        except Exception as e:
            logger.error(
                f"Failed to get pre-signed URL for document {document_id}: {str(e)}"
            )
            raise
    else:
        try:
            document = await service.get_document(
                document_id=document_id,
                current_user=None,
            )
            logger.info(f"Retrieved document {document_id}")
            return document
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {str(e)}")
            raise
