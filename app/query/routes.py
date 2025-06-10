from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.text_chunk.repository import TextChunkRepository
from app.text_chunk.service import TextChunkService
from app.schemas.schemas import TextChunkResponse
from .service import QueryService


router = APIRouter(prefix="/query", tags=["Query & RAG"])


# 依赖注入 QueryService
def get_query_service(db: AsyncSession = Depends(get_db)) -> QueryService:
    text_chunk_repo = TextChunkRepository(db)
    text_chunk_service_instance = TextChunkService(text_chunk_repo)
    return QueryService(text_chunk_service=text_chunk_service_instance)


class QueryRequest(BaseModel):  # 定义请求体
    query: str
    top_k: int = 5


class AskQueryRequest(BaseModel):  # 用于接收问答请求
    query: str
    # 可以添加 LLM 调用参数的可选字段，如果希望用户能控制
    # max_tokens: Optional[int] = 512
    # temperature: Optional[float] = 0.7


class AskQueryResponse(BaseModel):  # 用于返回问答结果
    query: str
    answer: str
    retrieved_context_texts: list[str] | None = None  # 可选，是否返回上下文给前端


@router.post("/retrieve-chunks", response_model=list[TextChunkResponse])
async def retrieve_chunks_for_query(
    request_data: QueryRequest,  # 使用请求体
    query_service: QueryService = Depends(get_query_service),
):
    """
    根据用户查询，检索相关的文本块 (用于测试检索效果)。
    """
    try:
        relevant_chunks = await query_service.retrieve_relevant_chunks(
            query_text=request_data.query, top_k_final_reranked=request_data.top_k
        )
        if not relevant_chunks:
            # 可以返回空列表，或者根据业务需求抛出 404
            # raise HTTPException(status_code=404, detail="No relevant chunks found.")
            pass
        return relevant_chunks
    except ValueError as ve:  # 捕获服务层可能抛出的特定错误
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"检索文本块时发生错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving chunks.")


@router.post("/ask", response_model=AskQueryResponse)
async def ask_llm_question(
    request_data: AskQueryRequest,
    query_service: QueryService = Depends(get_query_service),
):
    """
    接收用户查询，执行 RAG 流程（检索上下文 + LLM 生成答案），并返回结果。
    """
    try:
        # 调用 QueryService 中新的问答方法
        result_dict = await query_service.generate_answer_from_query(
            query_text=request_data.query
            # 如果 AskQueryRequest 中定义了llm参数，可以在这里传递
            # llm_max_tokens=request_data.max_tokens or 512,
            # llm_temperature=request_data.temperature or 0.7
        )

        # 如果发生错误，generate_answer_from_query 内部会记录日志，
        # 并可能返回一个包含错误信息的 answer。
        # 这里可以根据 result_dict 的内容进一步判断是否要抛出 HTTPException
        if "error" in result_dict.get("answer", "").lower():  # 简单错误检查
            # 可以不抛错，直接返回LLM服务返回的错误提示
            pass

        return AskQueryResponse(**result_dict)

    except ValueError as ve:  # 例如向量化失败等在 service 中抛出的 ValueError
        logger.error(f"处理问答请求时发生参数或逻辑错误: {ve}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:  # 例如模型加载失败等在 service 中抛出的 RuntimeError
        logger.error(f"处理问答请求时发生运行时错误: {re}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:  # 其他未知错误
        logger.error(f"处理问答请求 '/ask' 时发生未知错误: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="处理您的问题时发生内部错误，请稍后再试。"
        )
