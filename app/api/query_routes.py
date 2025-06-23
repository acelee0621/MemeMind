from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from starlette.responses import StreamingResponse

from app.services.query_service import QueryService
from app.chains.qa_chain import get_standalone_retriever

router = APIRouter(prefix="/query", tags=["Query & RAG"])


async def get_query_service() -> QueryService:
    """异步创建并缓存 QueryService 实例"""
    logger.info("正在异步创建 QueryService")
    query_service = await QueryService.create()
    return query_service


# --- 新的、简化的请求和响应模型 ---
class AskRequest(BaseModel):
    query: str


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = Field(5, gt=0, le=100, description="要返回的精排后文本块数量")


# --- API 端点 ---
@router.post("/ask/stream")  # 使用Ollama模型时，流式传输就不好用了，暂时留着
async def stream_ask_llm_question(
    request: AskRequest,
    query_service: QueryService = Depends(get_query_service),
):
    """
    接收用户查询，并以流式方式返回 RAG 链生成的答案。
    """
    try:
        # service.stream_answer 返回一个异步生成器
        # StreamingResponse 可以直接消费这个生成器，将数据块实时发送给客户端
        return StreamingResponse(
            query_service.stream_answer(request.query), media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"处理流式问答请求时出错: {e}", exc_info=True)
        # 在流式传输中处理错误比较复杂，通常在客户端处理连接断开
        # 这里可以在日志中记录，但很难再向客户端发送一个标准的HTTP错误
        # 实际生产中可能需要更复杂的错误信令机制
        return StreamingResponse(
            iter(["Error: An unexpected error occurred."]),
            media_type="text/event-stream",
            status_code=500,
        )


@router.post("/retrieve-chunks")
async def retrieve_chunks_for_query(request: RetrieveRequest):
    """
    用于调试的端点，仅执行向量检索，返回 top_k 个最相关的文本块。
    """
    logger.info(f"执行调试检索，查询: '{request.query}', top_k: {request.top_k}")
    try:
        # 直接调用我们简化的独立检索函数
        retrieved_docs = await get_standalone_retriever(
            query=request.query, top_k=request.top_k
        )
        return retrieved_docs
    except Exception as e:
        logger.error(f"调试检索时出错: {e}", exc_info=True)
        return {"error": str(e)}
    
    
@router.post("/ask")
async def ask_llm_question(
    request: AskRequest,
    query_service: QueryService = Depends(get_query_service),
):
    """
    接收用户查询，一次性返回 RAG 链生成的完整答案。
    """
    try:
        # 使用 .ainvoke() 而不是 .astream() 来获取完整答案
        full_response = await query_service.rag_chain.ainvoke(request.query)
        return {"response": full_response}
    except Exception as e:
        logger.error(f"处理非流式问答请求时出错: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
