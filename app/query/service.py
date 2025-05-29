import asyncio
from typing import Any
from loguru import logger

from app.core.config import settings
from app.core.embedding import get_embeddings
from app.core.chromadb_client import get_chroma_collection
from app.core.reranker import rerank_documents
from app.core.llm_service import generate_text_from_llm
from app.text_chunk.service import TextChunkService
from app.schemas.schemas import TextChunkResponse


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
                get_embeddings,  # 同步函数
                [query_text],  # 作为单元素列表传入
                instruction=settings.EMBEDDING_INSTRUCTION_FOR_RETRIEVAL,  # 从配置中获取指令
            )
            if not embeddings_list:
                logger.error(f"查询文本 '{query_text}' 的向量化结果为空。")
                raise ValueError("未能为查询生成向量嵌入。")
            logger.debug("查询文本向量嵌入生成完毕。")
            return embeddings_list[0]  # 返回单个查询向量
        except Exception as e:
            logger.error(f"查询向量化失败: {e}", exc_info=True)
            # 可以选择重新抛出特定类型的异常或通用异常
            raise ValueError(f"查询向量化失败: {e}")

    async def _search_vector_db_async(
        self, query_embedding: list[float], top_k: int
    ) -> list[int]:  # 返回 TextChunk 在 PostgreSQL 中的 ID 列表
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
            results = chroma_collection.query(
                query_embeddings=[query_embedding],  # query_embeddings 需要一个向量列表
                n_results=top_k,
                include=[
                    "metadatas",
                    "distances",
                ],  # metadatas 可能包含 text_chunk_db_id，distances 用于调试或排序
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
                        logger.warning(
                            f"无法将 ChromaDB 中检索到的 ID '{str_id}' 转换为整数。已跳过。"
                        )

                # 你也可以选择从 metadatas 中提取 text_chunk_db_id (如果你存储了这个字段)
                # 例如:
                # if results.get("metadatas") and results["metadatas"][0]:
                #     for metadata_item in results["metadatas"][0]:
                #         db_id = metadata_item.get("text_chunk_db_id")
                #         if db_id is not None:
                #             retrieved_pg_ids.append(int(db_id))
                # else:
                #     logger.warning("ChromaDB 查询结果中未找到 metadatas 或 metadatas 为空。")

            logger.info(
                f"从 ChromaDB 初步召回 {len(retrieved_pg_ids)} 个文本块 ID: {retrieved_pg_ids}"
            )
            return retrieved_pg_ids
        except Exception as e:
            logger.error(f"向量数据库搜索失败: {e}", exc_info=True)
            raise ValueError(f"向量数据库搜索失败: {e}")

    async def retrieve_relevant_chunks(
        self, query_text: str, top_k_final_reranked: int
    ) -> list[TextChunkResponse]:  # 最终返回的仍然是 TextChunkResponse 列表
        """
        为给定查询检索最相关的文本块 (包含召回和精排)。
        """
        logger.info(f"开始为查询 '{query_text[:100]}...' 检索并精排文本块。")

        # 1. 将查询文本向量化
        try:
            query_embedding = await self._embed_query_async(query_text)
        except ValueError as e:
            logger.error(f"无法为查询生成向量: {e}")
            return []

        # 2. 【召回阶段】在向量数据库中搜索大量候选文本块的ID
        try:
            # 使用配置中定义的较大 top_k进行初步召回
            candidate_chunk_pg_ids = await self._search_vector_db_async(
                query_embedding, top_k=settings.INITIAL_RETRIEVAL_TOP_K
            )
        except ValueError as e:
            logger.error(f"向量数据库初步召回时发生错误: {e}")
            return []

        if not candidate_chunk_pg_ids:
            logger.info("向量数据库初步召回未找到相关的文本块ID。")
            return []

        # 3. 【召回阶段】使用这些ID从 PostgreSQL 中获取候选文本块的详细信息
        try:
            candidate_chunks: list[
                TextChunkResponse
            ] = await self.text_chunk_service.get_chunks_by_ids(
                chunk_ids=candidate_chunk_pg_ids
            )
            logger.info(
                f"已从 PostgreSQL 获取 {len(candidate_chunks)} 个候选文本块的详细信息。"
            )
        except Exception as e:
            logger.error(
                f"从 PostgreSQL 获取候选文本块详情时发生错误: {e}", exc_info=True
            )
            return []

        if not candidate_chunks:
            logger.info("未能从数据库获取候选文本块内容，尽管已召回ID。")
            return []

        # 4. 【精排阶段】使用 Reranker 模型对候选文本块进行重排序
        try:
            logger.debug(f"开始对 {len(candidate_chunks)} 个候选块进行 Rerank...")
            # rerank_documents 是同步的，CPU/GPU密集型，也需要放入线程
            reranked_results: list[
                tuple[TextChunkResponse, float]
            ] = await asyncio.to_thread(rerank_documents, query_text, candidate_chunks)

            # 提取最终的 top_n 个文档块
            final_top_n_chunks: list[TextChunkResponse] = [
                doc_response
                for doc_response, score in reranked_results[:top_k_final_reranked]
            ]
            logger.info(f"Rerank 完成，最终选取 {len(final_top_n_chunks)} 个文本块。")
            return final_top_n_chunks

        except Exception as e:
            logger.error(f"Reranking 过程中发生错误: {e}", exc_info=True)
            # 如果 Rerank 失败，是返回初步召回的结果还是空列表？通常可能是空列表或抛出异常
            # 这里我们选择返回空列表，或者你可以选择返回 initial_retrieved_chunks 的前N个作为降级方案
            return []

    async def get_context_for_llm(self, query_text: str) -> list[str]:
        """
        获取为 LLM 准备的最终上下文文本列表。
        内部会调用 retrieve_relevant_chunks 并使用 settings.FINAL_CONTEXT_TOP_N。
        """
        logger.info(f"正在为查询 '{query_text[:100]}...' 获取 LLM 上下文...")

        # 调用 retrieve_relevant_chunks，并传入系统配置的最终上下文数量
        # retrieve_relevant_chunks 方法内部已经包含了向量召回和 Reranker 精排的完整流程
        final_reranked_chunk_responses: list[
            TextChunkResponse
        ] = await self.retrieve_relevant_chunks(
            query_text=query_text,
            top_k_final_reranked=settings.FINAL_CONTEXT_TOP_N,  # 使用配置中为LLM准备的块数量
        )

        if not final_reranked_chunk_responses:
            logger.warning("未能检索到或精排出任何相关的文本块作为 LLM 上下文。")
            return []

        # 提取纯文本内容
        context_texts: list[str] = [
            chunk.chunk_text for chunk in final_reranked_chunk_responses
        ]

        logger.info(f"已为 LLM 准备了 {len(context_texts)} 段上下文文本。")
        return context_texts

    # --- 新增方法：用于生成最终答案 ---
    async def generate_answer_from_query(
        self,
        query_text: str,
        # 可以从 Pydantic 请求模型中获取这些参数，或者使用默认值
        llm_max_tokens: int = 512,
        llm_temperature: float = 0.7,
        llm_top_p: float = 0.9,
        llm_stop: list[str] | None = None,
    ) -> dict[str, Any]:  # 返回包含答案和可能上下文的字典
        """
        处理用户查询，检索上下文，并调用LLM生成答案。
        """
        logger.info(f"开始为查询生成答案: '{query_text[:100]}...'")

        # 1. 获取上下文
        context_strings = await self.get_context_for_llm(query_text)

        if not context_strings:
            logger.warning(
                f"未能为查询 '{query_text[:100]}...' 获取到上下文，将尝试无上下文回答或返回提示。"
            )
            # 策略1：直接返回无上下文提示
            # return {"answer": "抱歉，未能找到与您问题相关的具体信息来生成回答。", "context_used": []}
            # 策略2：尝试让LLM无上下文回答（可能效果不佳）
            # context_for_prompt = "没有可用的上下文信息。"
            # 或者直接将 context_strings 作为空列表处理，由 prompt 模板决定如何展示
            pass  # 继续往下走，让 prompt 模板处理空上下文

        # 2. 构建 Prompt
        # 你需要精心设计你的 Prompt 模板
        context_block = (
            "\n---\n".join(context_strings)
            if context_strings
            else "没有额外的上下文信息。"
        )

        prompt = f"""【指令】根据下面提供的上下文信息来回答用户提出的问题。如果上下文中没有足够的信息来回答问题，请明确说明你无法从已知信息中找到答案，不要编造。请使用中文回答。

【上下文信息】
{context_block}

【用户问题】
{query_text}

【回答】
"""
        logger.debug(f"构建的 Prompt (部分内容):\n{prompt[:500]}...")

        # 3. 调用 LLM 生成文本 (这是一个同步阻塞操作)
        try:
            answer_text = await asyncio.to_thread(
                generate_text_from_llm,  # 调用你 llm_service 中的函数
                prompt=prompt,
                max_tokens=llm_max_tokens,
                temperature=llm_temperature,
                top_p=llm_top_p,
                stop=llm_stop,
            )
            logger.info(f"LLM 成功为查询 '{query_text[:50]}...' 生成答案。")
            # 为了透明度，可以考虑同时返回使用的上下文（Pydantic模型，而非纯文本）
            # final_context_responses = await self.retrieve_relevant_chunks(query_text, settings.FINAL_CONTEXT_TOP_N)

            return {
                "query": query_text,
                "answer": answer_text,
                "retrieved_context_texts": context_strings,  # 返回实际用于生成答案的纯文本上下文
                # "retrieved_context_full": [chunk.model_dump() for chunk in final_context_responses] # 可选：返回更详细的上下文信息
            }
        except RuntimeError as llm_error:  # 捕获 generate_text_from_llm 可能抛出的错误
            logger.error(f"调用 LLM 生成答案时失败: {llm_error}", exc_info=True)
            # 可以返回一个特定的错误响应
            return {
                "query": query_text,
                "answer": f"抱歉，回答您的问题时发生内部错误: {llm_error}",
                "retrieved_context_texts": context_strings,
            }
        except Exception as e:
            logger.error(f"生成答案过程中发生未知错误: {e}", exc_info=True)
            return {
                "query": query_text,
                "answer": "抱歉，处理您的问题时发生未知错误。",
                "retrieved_context_texts": context_strings,
            }
