from loguru import logger

from app.repository.chunk_repository import TextChunkRepository
from app.schemas.schemas import TextChunkCreate, TextChunkResponse


class TextChunkService:
    def __init__(self, repository: TextChunkRepository):
        """Service layer for TextChunk operations."""

        self.repository = repository

    async def add_text_chunk(self, data: TextChunkCreate) -> TextChunkResponse:
        new_chunk = await self.repository.create(data)
        return TextChunkResponse.model_validate(new_chunk)

    async def add_chunks_in_bulk(
        self, chunks_data: list[TextChunkCreate]
    ) -> list[TextChunkResponse]:
        new_chunks = await self.repository.create_bulk(chunks_data)
        return [TextChunkResponse.model_validate(chunk) for chunk in new_chunks]

    async def get_chunks_by_ids(self, chunk_ids: list[int]) -> list[TextChunkResponse]:
        if not chunk_ids:
            return []
        chunks = await self.repository.get_by_ids(chunk_ids)
        return [TextChunkResponse.model_validate(chunk) for chunk in chunks]

    async def get_document_chunks_for_display(
        self, document_id: int, limit: int = 1000, offset: int = 0
    ) -> list[TextChunkResponse]:
        chunks = await self.repository.get_by_document_id(document_id, limit, offset)
        return [TextChunkResponse.model_validate(chunk) for chunk in chunks]

    async def delete_all_chunks_for_document(self, document_id: int) -> int:
        deleted_count = await self.repository.delete_chunks_by_document_id(document_id)
        logger.info(f"为文档 ID {document_id} 删除了 {deleted_count} 个文本块。")
        return deleted_count
