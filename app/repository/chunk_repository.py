from sqlalchemy import select, delete
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.models import TextChunk
from app.schemas.schemas import TextChunkCreate


class TextChunkRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, data: TextChunkCreate) -> TextChunk:
        """
        创建一个新的文本块记录。
        """
        new_chunk = TextChunk(
            source_document_id=data.source_document_id,
            chunk_text=data.chunk_text,
            sequence_in_document=data.sequence_in_document,
            metadata_json=data.metadata_json,
        )
        self.session.add(new_chunk)
        try:
            await self.session.commit()
            await self.session.refresh(new_chunk)
            return new_chunk
        except (
            IntegrityError
        ) as e:  # 主要可能是外键约束失败，例如 source_document_id 不存在
            await self.session.rollback()
            # 根据实际可能发生的 IntegrityError 类型调整错误信息
            # 例如，如果 source_document_id 无效，这通常表明一个逻辑错误
            raise ValueError(
                f"创建 TextChunk 失败，可能由于无效的 source_document_id: {str(e)}"
            )
        except Exception as e:
            await self.session.rollback()
            raise e  # 重新抛出其他未知异常

    async def create_bulk(self, chunks_data: list[TextChunkCreate]) -> list[TextChunk]:
        """
        批量创建文本块记录。
        这比逐个创建效率高得多。
        """
        if not chunks_data:
            return []

        new_chunks_orm = [TextChunk(**data.model_dump()) for data in chunks_data]

        self.session.add_all(new_chunks_orm)
        try:
            await self.session.commit()
            # 批量 refresh 可能比较麻烦或效率不高，通常在批量创建后，
            # 如果需要立即使用这些对象的数据库生成属性（如ID），
            # 可以选择不 refresh，或者如果数量不多，可以逐个 refresh。
            # 对于 RAG 流程，通常创建后会立即用于向量化，ID 是关键。
            # SQLAlchemy 2.0 的 add_all 配合 commit 后，对象会自动填充 ID。
            # 如果需要其他数据库默认值或触发器生成的值，才需要 refresh。
            # 我们这里假设 ID 会被自动填充。
            return new_chunks_orm
        except IntegrityError as e:
            await self.session.rollback()
            raise ValueError(
                f"批量创建 TextChunk 失败，可能由于无效的 source_document_id: {str(e)}"
            )
        except Exception as e:
            await self.session.rollback()
            raise e

    async def get_by_ids(self, chunk_ids: list[int]) -> list[TextChunk]:
        if not chunk_ids:
            return []
        query = select(TextChunk).where(TextChunk.id.in_(chunk_ids)).options(selectinload(TextChunk.source_document))
        result = await self.session.scalars(query)
        return list(result.all())

    async def get_by_document_id(
        self, document_id: int, limit: int = 1000, offset: int = 0
    ) -> list[TextChunk]:
        """
        根据源文档 ID 获取其所有文本块。
        添加了分页参数以防一个文档有过多文本块。
        """
        query = (
            select(TextChunk)
            .where(TextChunk.source_document_id == document_id)
            .order_by(TextChunk.sequence_in_document)  # 通常按顺序获取
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.scalars(query)
        return list(result.all())

    async def delete_chunks_by_document_id(self, document_id: int):
        """
        根据源文档 ID 删除其所有关联的文本块。
        返回被删除的行数。
        注意：如果 SourceDocument 和 TextChunk 之间设置了 ondelete="CASCADE" 并且
        你通过 ORM 删除了 SourceDocument 对象，那么数据库会自动处理 TextChunk 的删除。
        这个方法主要用于需要显式、独立地删除某个文档的所有文本块的场景（例如，重新处理文档前）。
        """
        stmt = delete(TextChunk).where(TextChunk.source_document_id == document_id)
        result = await self.session.execute(stmt)
        await self.session.commit()  # 删除操作需要 commit
        return result.rowcount  # 返回影响的行数
