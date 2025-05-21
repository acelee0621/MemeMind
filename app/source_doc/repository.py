from sqlalchemy import select, desc, asc
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AlreadyExistsException, NotFoundException
from app.models.models import SourceDocument
from app.schemas.schemas import SourceDocumentCreate,SourceDocumentUpdate


class SourceDocumentRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, data: SourceDocumentCreate, current_user) -> SourceDocument:
        new_document = SourceDocument(
            object_name=data.object_name,
            bucket_name=data.bucket_name,
            original_filename=data.original_filename,
            content_type=data.content_type,
            size=data.size,
            owner_id=current_user.id,
        )
        self.session.add(new_document)
        try:
            await self.session.commit()
            await self.session.refresh(new_document)
            return new_document
        except IntegrityError:
            await self.session.rollback()
            raise AlreadyExistsException(
                f"SourceDocument with content {data.original_filename} already exists"
            )

    async def get_by_id(self, document_id: int, current_user) -> SourceDocument:
        query = select(SourceDocument).where(
            SourceDocument.id == document_id,
            SourceDocument.owner_id == current_user.id,
        )
        result = await self.session.scalars(query)
        document = result.one_or_none()
        if not document:
            raise NotFoundException(f"SourceDocument with id {document_id} not found")
        return document

    async def get_by_id_internal(self, document_id: int) -> SourceDocument:
        query = select(SourceDocument).where(SourceDocument.id == document_id)
        result = await self.session.scalars(query)
        document = result.one_or_none()
        if not document:
            raise NotFoundException(f"SourceDocument with id {document_id} not found")
        return document

    async def get_all(
        self,
        limit: int,
        offset: int,
        order_by: str | None,
        current_user,
    ) -> list[SourceDocument]:
        query = select(SourceDocument).where(SourceDocument.owner_id == current_user.id)

        if order_by:
            if order_by == "created_at desc":
                query = query.order_by(desc(SourceDocument.created_at))
            elif order_by == "created_at asc":
                query = query.order_by(asc(SourceDocument.created_at))

        # 分页功能
        query = query.limit(limit).offset(offset)

        result = await self.session.scalars(query)
        return list(result.all())
    
    async def update(self, data: SourceDocumentUpdate, document_id: int) -> SourceDocument:
        
        query = select(SourceDocument).where(SourceDocument.id == document_id)
        result = await self.session.scalars(query)
        document = result.one_or_none()
        if not document:
            raise NotFoundException(
                f"Document with id {document_id} not found."
            )
        update_data = data.model_dump(exclude_unset=True)
        # 确保不修改 id 和 owner_id
        update_data.pop("id", None)
        update_data.pop("owner_id", None)
        if not update_data:
            raise ValueError("No fields to update")
        for key, value in update_data.items():
            setattr(document, key, value)
        await self.session.commit()
        await self.session.refresh(document)
        return document

    async def delete(self, document_id: int, current_user) -> None:
        document = await self.session.get(SourceDocument, document_id)

        if not document or document.owner_id != current_user.id:
            raise NotFoundException(f"Attachment with id {document_id} not found")

        await self.session.delete(document)
        await self.session.commit()
