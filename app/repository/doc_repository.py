from loguru import logger
from sqlalchemy import select, desc, asc
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AlreadyExistsException, NotFoundException
from app.models.models import SourceDocument
from app.schemas.schemas import SourceDocumentCreate, SourceDocumentUpdate


class SourceDocumentRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, data: SourceDocumentCreate) -> SourceDocument:
        """在数据库中创建一个新的文档记录。"""
        logger.info(f"正在创建文档记录，文件名: '{data.original_filename}'")
        
        # 这里的字段已与新模型对齐，无需修改
        new_document = SourceDocument(
            file_path=data.file_path,
            original_filename=data.original_filename,
            content_type=data.content_type,
            size=data.size,
        )
        self.session.add(new_document)
        try:
            await self.session.commit()
            await self.session.refresh(new_document)
            logger.success(f"成功创建文档记录 ID: {new_document.id}, 路径: '{new_document.file_path}'")
            return new_document
        except IntegrityError:
            await self.session.rollback()
            logger.error(
                f"创建文档失败，路径 '{data.file_path}' 已存在或违反唯一性约束。"
            )
            raise AlreadyExistsException(
                f"具有路径 '{data.file_path}' 的文档已存在。"
            )

    async def get_by_id(self, document_id: int) -> SourceDocument:
        """通过 ID 获取单个文档记录。"""
        logger.debug(f"正在查询文档 ID: {document_id}")
        query = select(SourceDocument).where(SourceDocument.id == document_id)
        result = await self.session.scalars(query)
        document = result.one_or_none()
        if not document:
            logger.warning(f"查询失败，未找到文档 ID: {document_id}")
            raise NotFoundException(f"未找到 ID 为 {document_id} 的文档")
        return document

    async def get_all(
        self, limit: int, offset: int, order_by: str | None
    ) -> list[SourceDocument]:
        """获取所有文档记录（支持分页和排序）。"""
        logger.debug(f"正在查询文档列表，limit={limit}, offset={offset}, order_by='{order_by}'")
        query = select(SourceDocument)

        if order_by:
            if order_by == "created_at desc":
                query = query.order_by(desc(SourceDocument.created_at))
            elif order_by == "created_at asc":
                query = query.order_by(asc(SourceDocument.created_at))

        query = query.limit(limit).offset(offset)
        result = await self.session.scalars(query)
        return list(result.all())

    async def update(
        self, data: SourceDocumentUpdate, document_id: int
    ) -> SourceDocument:
        """更新一个已存在的文档记录。"""
        logger.info(f"正在更新文档 ID: {document_id}，数据: {data.model_dump(exclude_unset=True)}")
        document = await self.get_by_id(document_id) # 复用get_by_id以包含其日志和错误处理
        
        update_data = data.model_dump(exclude_unset=True)
        update_data.pop("id", None)
        if not update_data:
            logger.warning(f"为文档 {document_id} 调用更新，但未提供任何有效字段。")
            raise ValueError("没有提供需要更新的字段")

        for key, value in update_data.items():
            setattr(document, key, value)
            
        await self.session.commit()
        await self.session.refresh(document)
        logger.success(f"成功更新文档 ID: {document_id}")
        return document

    async def delete(self, document_id: int) -> None:
        """从数据库中删除一个文档记录。"""
        logger.info(f"正在从数据库删除文档记录 ID: {document_id}")
        document = await self.get_by_id(document_id) # 复用get_by_id
        await self.session.delete(document)
        await self.session.commit()
        logger.success(f"成功从数据库删除文档记录 ID: {document_id}")