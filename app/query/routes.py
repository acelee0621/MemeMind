from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.core.database import get_db
from app.core.logging import get_logger
from app.text_chunk.repository import TextChunkRepository # QueryService 依赖 TextChunkService
from app.text_chunk.service import TextChunkService     # QueryService 依赖 TextChunkService
from app.schemas.schemas import TextChunkResponse
from .service import QueryService # 导入你刚创建的 QueryService

logger = get_logger(__name__)

router = APIRouter(prefix="/query", tags=["Query & RAG"])

# 依赖注入 QueryService
def get_query_service(db: AsyncSession = Depends(get_db)) -> QueryService:
    text_chunk_repo = TextChunkRepository(db)
    text_chunk_service_instance = TextChunkService(text_chunk_repo)
    return QueryService(text_chunk_service=text_chunk_service_instance)

class QueryRequest(BaseModel): # 定义请求体
    query: str
    top_k: int = 5

@router.post("/retrieve-chunks", response_model=List[TextChunkResponse])
async def retrieve_chunks_for_query(
    request_data: QueryRequest, # 使用请求体
    query_service: QueryService = Depends(get_query_service)
):
    """
    根据用户查询，检索相关的文本块 (用于测试检索效果)。
    """
    try:
        relevant_chunks = await query_service.retrieve_relevant_chunks(
            query_text=request_data.query, 
            top_k=request_data.top_k
        )
        if not relevant_chunks:
            # 可以返回空列表，或者根据业务需求抛出 404
            # raise HTTPException(status_code=404, detail="No relevant chunks found.")
            pass
        return relevant_chunks
    except ValueError as ve: # 捕获服务层可能抛出的特定错误
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"检索文本块时发生错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving chunks.")