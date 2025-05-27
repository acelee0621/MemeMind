import asyncio

from loguru import logger

# --- 核心和工具类导入 ---
from app.core.config import settings
from app.core.embedding import get_embeddings
from app.core.chromadb_client import get_chroma_collection
from app.core.reranker import rerank_documents # 假设你已创建 reranker.py 并定义了此函数
from app.core.database import SessionLocal # 用于创建数据库会话

# --- 服务和仓库层导入 ---
from app.text_chunk.service import TextChunkService
from app.text_chunk.repository import TextChunkRepository
# 注意：QueryService 通常不直接依赖 SourceDocumentService/Repository 来处理查询，
# 除非你需要获取源文档的某些特定信息，但核心检索流程主要依赖 TextChunkService。

# --- Pydantic Schemas ---
from app.schemas.schemas import TextChunkResponse


async def _embed_query_for_processing(query_text: str) -> list[float]:
    """
    (内部辅助函数) 向量化查询文本。
    logger_instance 是从 Celery 任务传递过来的 logger。
    """
    logger.debug(f"查询处理工具：开始为查询文本生成向量嵌入: '{query_text[:50]}...'")
    try:
        embeddings_list = await asyncio.to_thread(
            get_embeddings,
            [query_text],
            instruction=settings.EMBEDDING_INSTRUCTION_FOR_RETRIEVAL
        )
        if not embeddings_list:
            logger.error(f"查询处理工具：查询文本 '{query_text}' 的向量化结果为空。")
            raise ValueError("未能为查询生成向量嵌入。")
        logger.debug("查询处理工具：查询文本向量嵌入生成完毕。")
        return embeddings_list[0]
    except Exception as e:
        logger.error(f"查询处理工具：查询向量化失败: {e}", exc_info=True)
        raise ValueError(f"查询向量化失败: {e}")


async def _search_vector_db_for_processing(
    query_embedding: list[float], initial_top_k: int,
) -> list[int]:
    """
    (内部辅助函数) 在 ChromaDB 中进行初步向量召回。
    返回召回的 TextChunk 在 PostgreSQL 中的 ID 列表。
    """
    logger.debug(f"查询处理工具：开始在 ChromaDB 中搜索 top_k={initial_top_k} 个相关文本块。")
    
    def _query_chroma_sync_internal():
        chroma_collection = get_chroma_collection()
        results = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=initial_top_k,
            # include=["metadatas", "distances"]
        )
        return results

    try:
        results = await asyncio.to_thread(_query_chroma_sync_internal)
        retrieved_pg_ids: list[int] = []
        if results and results.get("ids") and results["ids"][0]:
            chroma_ids_str_list = results["ids"][0]
            for str_id in chroma_ids_str_list:
                try:
                    retrieved_pg_ids.append(int(str_id))
                except ValueError:
                    logger.warning(f"查询处理工具：无法将 ChromaDB ID '{str_id}' 转为整数。已跳过。")
        logger.info(f"查询处理工具：从 ChromaDB 初步召回 {len(retrieved_pg_ids)} 个文本块 ID: {retrieved_pg_ids}")
        return retrieved_pg_ids
    except Exception as e:
        logger.error(f"查询处理工具：向量数据库搜索失败: {e}", exc_info=True)
        raise ValueError(f"向量数据库搜索失败: {e}")

# --- 异步业务逻辑核心 (针对查询处理) ---
async def execute_query_processing_async(
    query_text: str,
    top_k_final_reranked: int, # 用户期望最终看到的精排后的数量
    task_id_for_log: str      # Celery 任务ID，用于日志追踪    
) -> list[TextChunkResponse]: # 返回精排后的文本块 Pydantic 模型列表
    """
    在 Celery 任务中被调用的核心异步处理函数，用于处理用户查询。
    包含数据库会话管理、服务实例化、查询向量化、向量召回、
    获取文本块、Rerank精排，并返回最终的文本块列表。
    """
    logger.info(
        f"{task_id_for_log} (Async Query Logic) 开始处理查询: '{query_text[:100]}...', 目标返回精排后 top {top_k_final_reranked} 条"
    )

    # text_chunk_service 变量将在 try 块内部的数据库会话上下文中被赋值
    text_chunk_service: TextChunkService | None = None 

    try:
        async with SessionLocal() as db: # 管理异步数据库会话
            logger.info(
                f"{task_id_for_log} (Async Query Logic) 数据库会话已为查询 '{query_text[:50]}...' 创建"
            )

            # 实例化 Repositories 和 Services (这里主要需要 TextChunkService)
            text_chunk_repo = TextChunkRepository(db)
            text_chunk_service = TextChunkService(text_chunk_repo)
            # 如果 QueryService 本身有一些不依赖特定请求的辅助方法，也可以在这里实例化，但当前核心逻辑是检索。

            # ==================================================================
            # 第1步：将查询文本向量化
            # ==================================================================
            query_embedding = await _embed_query_for_processing(query_text)

            # ==================================================================
            # 第2步：【召回阶段】在向量数据库中搜索大量候选文本块的ID
            # ==================================================================
            # 使用配置中定义的较大 top_k (INITIAL_RETRIEVAL_TOP_K) 进行初步召回
            candidate_chunk_pg_ids = await _search_vector_db_for_processing(
                query_embedding, settings.INITIAL_RETRIEVAL_TOP_K
            )

            if not candidate_chunk_pg_ids:
                logger.info(f"{task_id_for_log} (Async Query Logic) 向量数据库初步召回未找到相关的文本块ID。")
                return [] # 返回空列表

            # ==================================================================
            # 第3步：【召回阶段】使用这些ID从 PostgreSQL 中获取候选文本块的详细信息
            # ==================================================================
            candidate_chunks: list[TextChunkResponse]
            try:
                candidate_chunks = await text_chunk_service.get_chunks_by_ids(
                    chunk_ids=candidate_chunk_pg_ids
                )
                logger.info(f"{task_id_for_log} (Async Query Logic) 已从 PostgreSQL 获取 {len(candidate_chunks)} 个候选文本块。")
            except Exception as e:
                logger.error(f"{task_id_for_log} (Async Query Logic) 从 PostgreSQL 获取候选文本块详情时发生错误: {e}", exc_info=True)
                # 这种情况下，由于是核心数据获取失败，直接向上抛出，让顶层异常处理
                raise ValueError(f"无法从数据库获取候选文本块详情: {e}") 
            
            if not candidate_chunks:
                logger.info(f"{task_id_for_log} (Async Query Logic) 未能从数据库获取候选文本块内容，尽管已召回ID。")
                return []

            # ==================================================================
            # 第4步：【精排阶段】使用 Reranker 模型对候选文本块进行重排序
            # ==================================================================
            final_reranked_chunks: list[TextChunkResponse]
            try:
                logger.debug(f"{task_id_for_log} (Async Query Logic) 开始对 {len(candidate_chunks)} 个候选块进行 Rerank...")
                
                reranked_scored_results: list[tuple[TextChunkResponse, float]] = await asyncio.to_thread(
                    rerank_documents, # 这是你 app.core.reranker 中的函数
                    query_text,
                    candidate_chunks 
                )
                
                # 根据传入的 top_k_final_reranked 参数，提取最终的文本块
                final_reranked_chunks = [
                    doc_response for doc_response, score in reranked_scored_results[:top_k_final_reranked]
                ]
                logger.info(f"{task_id_for_log} (Async Query Logic) Rerank 完成，最终选取 {len(final_reranked_chunks)} 个文本块。")
                
            except Exception as e:
                logger.error(f"{task_id_for_log} (Async Query Logic) Reranking 过程中发生错误: {e}", exc_info=True)
                # Rerank 失败的降级策略：可以选择返回初步召回结果的前 N 个，或者直接报错/返回空
                # 这里我们选择向上抛出，让调用方决定如何处理（例如 Celery 任务标记失败）
                raise ValueError(f"Reranking失败: {e}")
                
            return final_reranked_chunks # 返回精排后的 Pydantic 模型列表

    except ValueError as ve: # 捕获由 _embed_query, _search_vector_db, rerank 等内部逻辑抛出的 ValueError
        logger.error(f"{task_id_for_log} (Async Query Logic) 处理查询时发生已知错误: {str(ve)}", exc_info=True)
        # 对于 ValueError，通常是可预期的错误，直接向上抛出
        raise
    except Exception as e:  # 捕获其他所有未预料的顶层错误
        logger.error(
            f"{task_id_for_log} (Async Query Logic) 处理查询 '{query_text[:100]}...' 时发生无法处理的错误: {str(e)}",
            exc_info=True,
        )        
        # 对于查询处理任务，通常没有像文档处理那样的“状态”可以更新到数据库来标记错误。
        # 主要依赖 Celery 将任务标记为失败，并记录异常信息。
        raise e # 将异常向上抛给 async_to_sync，再由同步任务的 except 块处理