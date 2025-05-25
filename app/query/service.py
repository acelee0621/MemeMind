import asyncio
# from typing import List, Dict, Any, Optional # 确保导入 Optional

from app.core.logging import get_logger
from app.core.config import settings
from app.core.embedding import get_embeddings
from app.core.chromadb_client import get_chroma_collection
from app.core.exceptions import NotFoundException # 如果需要抛出特定异常
from app.text_chunk.service import TextChunkService # 用于根据ID获取文本块
from app.schemas.schemas import TextChunkResponse # Service 方法的返回类型

logger = get_logger(__name__)

class QueryService:
    def __init__(self, text_chunk_service: TextChunkService):
        """
        查询服务层，负责处理 RAG 检索逻辑。
        :param text_chunk_service: 用于从 PostgreSQL 获取文本块的服务实例。
        """
        self.text_chunk_service = text_chunk_service
        # Embedding 模型和 ChromaDB 集合通过导入的辅助函数按需加载/获取

    async def _embed_query_async(self, query_text: str) -> list[float]:
        """
        异步包裹查询文本的向量化过程。
        get_embeddings 是一个同步的、计算密集型函数。
        """
        logger.debug(f"开始为查询文本生成向量嵌入: '{query_text[:50]}...'")
        try:
            # get_embeddings 期望一个文本列表和指令
            # 它内部处理模型的加载、设备选择、分词、推理和归一化
            embeddings_list = await asyncio.to_thread(
                get_embeddings, # 同步函数
                [query_text],   # 作为单元素列表传入
                # instruction=settings.EMBEDDING_INSTRUCTION_FOR_RETRIEVAL # 从配置中获取指令
                instruction=""
            )
            if not embeddings_list:
                logger.error(f"查询文本 '{query_text}' 的向量化结果为空。")
                raise ValueError("未能为查询生成向量嵌入。")
            logger.debug("查询文本向量嵌入生成完毕。")
            return embeddings_list[0] # 返回单个查询向量
        except Exception as e:
            logger.error(f"查询向量化失败: {e}", exc_info=True)
            # 可以选择重新抛出特定类型的异常或通用异常
            raise ValueError(f"查询向量化失败: {e}")


    async def _search_vector_db_async(
        self, query_embedding: list[float], top_k: int
    ) -> list[int]: # 返回 TextChunk 在 PostgreSQL 中的 ID 列表
        """
        异步包裹在 ChromaDB 中进行向量搜索的过程。
        返回检索到的文本块在 PostgreSQL 中的主键 ID 列表。
        """
        logger.debug(f"开始在 ChromaDB 中搜索 top_k={top_k} 个相关文本块。")
        def _query_chroma_sync():
            # get_chroma_collection 内部处理客户端和集合的获取/创建
            # 并且已经配置了使用 'cosine' 相似度
            chroma_collection = get_chroma_collection()
            
            # 执行查询
            # 我们期望 ChromaDB 中的 ID 就是 PostgreSQL 中 TextChunk 的 ID (字符串形式)
            results = chroma_collection.query(
                query_embeddings=[query_embedding], # query_embeddings 需要一个向量列表
                n_results=top_k,
                include=["metadatas", "distances"] # metadatas 可能包含 text_chunk_db_id，distances 用于调试或排序
            )
            return results

        try:
            results = await asyncio.to_thread(_query_chroma_sync)
            
            retrieved_pg_ids: list[int] = []
            if results and results.get("ids") and results["ids"][0]:
                # results["ids"] 是一个列表的列表, 例如: [['id1', 'id2'], ...]
                # 对于单个查询向量，我们关心 results["ids"][0]
                chroma_ids_str_list = results["ids"][0]
                for str_id in chroma_ids_str_list:
                    try:
                        retrieved_pg_ids.append(int(str_id))
                    except ValueError:
                        logger.warning(f"无法将 ChromaDB 中检索到的 ID '{str_id}' 转换为整数。已跳过。")
                
                # 你也可以选择从 metadatas 中提取 text_chunk_db_id (如果你存储了这个字段)
                # 例如:
                # if results.get("metadatas") and results["metadatas"][0]:
                #     for metadata_item in results["metadatas"][0]:
                #         db_id = metadata_item.get("text_chunk_db_id")
                #         if db_id is not None:
                #             retrieved_pg_ids.append(int(db_id))
                # else:
                #     logger.warning("ChromaDB 查询结果中未找到 metadatas 或 metadatas 为空。")

            logger.info(f"从 ChromaDB 检索到 {len(retrieved_pg_ids)} 个文本块 ID: {retrieved_pg_ids}")
            return retrieved_pg_ids
        except Exception as e:
            logger.error(f"向量数据库搜索失败: {e}", exc_info=True)
            raise ValueError(f"向量数据库搜索失败: {e}")


    async def retrieve_relevant_chunks(
        self, query_text: str, top_k: int = 5 # 默认检索最相关的5个块
    ) -> list[TextChunkResponse]:
        """
        为给定查询检索最相关的文本块。
        步骤:
        1. 将查询文本向量化。
        2. 在向量数据库 (ChromaDB) 中搜索相关文本块的ID。
        3. 使用这些ID从 PostgreSQL 中获取完整的文本块详情。
        """
        logger.info(f"开始为查询 '{query_text[:100]}...' 检索相关的文本块 (top_k={top_k})")

        # 1. 将查询文本向量化
        try:
            query_embedding = await self._embed_query_async(query_text)
        except ValueError as e: # 捕获向量化失败
            logger.error(f"无法为查询生成向量: {e}")
            return [] # 或者可以向上抛出 HTTPException

        # 2. 在向量数据库中搜索相关文本块的ID
        try:
            relevant_chunk_pg_ids = await self._search_vector_db_async(query_embedding, top_k)
        except ValueError as e: # 捕获向量搜索失败
            logger.error(f"向量数据库搜索时发生错误: {e}")
            return []


        if not relevant_chunk_pg_ids:
            logger.info("在向量数据库中没有找到相关的文本块ID。")
            return []

        # 3. 使用这些ID从 PostgreSQL 中获取完整的文本块详情
        try:
            # text_chunk_service.get_chunks_by_ids 返回 List[TextChunkResponse]
            chunk_responses = await self.text_chunk_service.get_chunks_by_ids(
                chunk_ids=relevant_chunk_pg_ids
            )
            logger.info(f"已从 PostgreSQL 获取 {len(chunk_responses)} 个文本块的详细信息。")

            # 可选：根据 ChromaDB 返回的 ID 顺序对结果进行重新排序
            # ChromaDB 返回的 ID 是按相似度（或距离）排序的。
            # text_chunk_service.get_chunks_by_ids 返回的列表顺序可能与输入 ID 列表顺序不同。
            if chunk_responses:
                id_to_chunk_map = {chunk.id: chunk for chunk in chunk_responses}
                ordered_responses = [
                    id_to_chunk_map[pg_id] for pg_id in relevant_chunk_pg_ids if pg_id in id_to_chunk_map
                ]
                if len(ordered_responses) != len(chunk_responses):
                    logger.warning("重新排序后，文本块数量与从数据库获取的数量不一致，可能存在ID丢失或重复。")
                return ordered_responses
            
            return chunk_responses
        except Exception as e:
            logger.error(f"从 PostgreSQL 获取文本块详情时发生错误: {e}", exc_info=True)
            return [] # 或者向上抛出 HTTPException