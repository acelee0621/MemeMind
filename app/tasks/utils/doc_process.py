import asyncio

from loguru import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.s3_client import s3_client
from app.core.database import create_engine_and_session_for_celery
from app.core.embedding_qwen import get_embeddings
from app.core.chromadb_client import get_chroma_collection
from app.source_doc.repository import SourceDocumentRepository
from app.source_doc.service import SourceDocumentService
from app.text_chunk.repository import TextChunkRepository
from app.text_chunk.service import TextChunkService
from app.models.models import TextChunk
from app.schemas.schemas import SourceDocumentResponse, TextChunkCreate
from .doc_parser import parse_and_clean_document


# --- 异步业务逻辑核心 ---
async def _execute_document_processing_async(document_id: int, task_id_for_log: str):
    # 1. 延迟初始化：在函数开始时，为本次任务创建专属的数据库引擎和会话工厂
    #    它们会自动绑定到由 Celery 任务创建的那个全新的事件循环上。
    db_engine, SessionLocal = create_engine_and_session_for_celery()
    logger.info(f"{task_id_for_log} (Async Logic) 开始处理文档 ID: {document_id}")

    source_doc_service: SourceDocumentService
    text_chunk_service: TextChunkService

    document_response: SourceDocumentResponse | None = None
    number_of_chunks_created = 0  # 初始化

    try:
        async with SessionLocal() as db:
            logger.info(
                f"{task_id_for_log} (Async Logic) 数据库会话已为文档 ID {document_id} 创建"
            )

            # 实例化 Repositories 和 Services
            source_doc_repo = SourceDocumentRepository(db)
            source_doc_service = SourceDocumentService(source_doc_repo)
            text_chunk_repo = TextChunkRepository(db)
            text_chunk_service = TextChunkService(text_chunk_repo)

            # ==================================================================
            # 第1步：获取文档元数据
            # ==================================================================
            document_response = await source_doc_service.get_document(
                document_id=document_id
            )
            if not document_response:
                logger.error(
                    f"{task_id_for_log} (Async Logic) 文档 ID: {document_id} 未找到。任务终止。"
                )
                # 通常 Service 会抛 NotFoundException，这里是双重保障
                return {"status": "error", "message": "Document not found in DB"}

            logger.info(
                f"{task_id_for_log} (Async Logic) 成功获取文档 {document_response.id} (原始文件名: {document_response.original_filename})"
            )

            # ==================================================================
            # 第2步：更新文档状态为 "processing"
            # ==================================================================
            await source_doc_service.update_document_processing_info(
                document_id=document_response.id, status="processing"
            )
            logger.info(
                f"{task_id_for_log} (Async Logic) 文档 {document_response.id} 状态更新为 'processing'"
            )

            # ==================================================================
            # 第3步：从 S3 (MinIO) 下载文档内容
            # ==================================================================
            file_content_bytes: bytes
            try:
                s3_object = await asyncio.to_thread(
                    s3_client.get_object,
                    Bucket=document_response.bucket_name,
                    Key=document_response.object_name,
                )
                file_content_bytes = s3_object["Body"].read()
                logger.info(
                    f"{task_id_for_log} (Async Logic) 文档 {document_response.object_name} ({len(file_content_bytes)} bytes) 已从 S3 下载。"
                )
            except Exception as s3_error:
                logger.error(
                    f"{task_id_for_log} (Async Logic) 从 S3 下载文档 {document_response.object_name} 失败: {s3_error}",
                    exc_info=True,
                )
                await source_doc_service.update_document_processing_info(
                    document_id=document_response.id,
                    status="error",
                    error_message=f"S3下载失败: {str(s3_error)[:255]}",
                )
                raise s3_error

            # ==================================================================
            # 第4步：根据文档类型解析文本内容
            # ==================================================================
            raw_text: str = ""
            content_type = document_response.content_type.lower()

            try:
                logger.info("开始解析和清洗文档内容...")
                # 将同步的、CPU密集型的解析和清洗任务放入后台线程执行
                raw_text = await asyncio.to_thread(
                    parse_and_clean_document,
                    file_content_bytes,
                    document_response.original_filename,
                    document_response.content_type,
                )

                if not raw_text.strip():
                    empty_content_msg = "解析后文本内容为空"
                    logger.warning(
                        f"{task_id_for_log} {empty_content_msg} (文档ID: {document_id})"
                    )
                    await source_doc_service.update_document_processing_info(
                        document_id=document_response.id,
                        status="error",
                        error_message=empty_content_msg,
                    )
                    return {"status": "error", "message": empty_content_msg}

                logger.info(f"文档内容解析和清洗成功，提取文本长度: {len(raw_text)}")

            except Exception as parse_error:
                logger.error(
                    f"{task_id_for_log} (Async Logic) 解析文档 {document_response.original_filename} 失败: {parse_error}",
                    exc_info=True,
                )
                await source_doc_service.update_document_processing_info(
                    document_id=document_response.id,
                    status="error",
                    error_message=f"文档解析失败: {str(parse_error)[:255]}",
                )
                raise parse_error  # 重新抛出，让 Celery 标记任务失败

            # ==================================================================
            # 第5步：将文本内容分割成小块 (Chunks)
            # ==================================================================

            # 示例：简单的按固定长度（大致）分块，带重叠
            chunk_size = settings.CHUNK_SIZE  # 每个块的目标字符数
            chunk_overlap = settings.CHUNK_OVERLAP  # 相邻块之间的重叠字符数

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""],
            )
            chunks_texts_list = text_splitter.split_text(raw_text)

            chunks_texts_list: list[str] = []
            if len(raw_text) > chunk_size:
                for i in range(0, len(raw_text), chunk_size - chunk_overlap):
                    chunk = raw_text[i : i + chunk_size]
                    chunks_texts_list.append(chunk)
            elif raw_text:  # 如果文本不为空但小于块大小，则本身作为一个块
                chunks_texts_list.append(raw_text)

            if (
                not chunks_texts_list and raw_text
            ):  # 如果有原始文本但未能分块（逻辑问题）
                logger.warning(
                    f"{task_id_for_log} (Async Logic) 文本分块结果为空，但原始文本不为空。文档ID: {document_id}"
                )
                # 也许也应该标记为错误

            logger.info(
                f"{task_id_for_log} (Async Logic) 文档 ID: {document_id} 被分割成 {len(chunks_texts_list)} 个文本块"
            )

            # ==================================================================
            # 第6步：将文本块存入数据库 (使用 TextChunkService)
            # ==================================================================
            created_chunk_db_objects: list[TextChunk] = []  # 用于存储返回的 ORM 对象
            if chunks_texts_list:
                chunks_to_create: list[TextChunkCreate] = []
                for i, chunk_text_content in enumerate(chunks_texts_list):
                    # 准备 Pydantic 模型数据
                    # 你可以在这里为 metadata_json 添加更多信息，例如如果解析时得到了页码
                    chunk_meta = {"parsed_by": "default_parser_v1"}
                    if (
                        "pdf" in content_type
                    ):  # 示例：如果是pdf，可以尝试从解析器获取页码信息
                        # chunk_meta["page_number"] = extracted_page_number_for_this_chunk
                        pass

                    chunks_to_create.append(
                        TextChunkCreate(
                            source_document_id=document_response.id,
                            chunk_text=chunk_text_content,
                            sequence_in_document=i,
                            metadata_json=chunk_meta,
                        )
                    )

                try:
                    created_chunk_db_objects = (
                        await text_chunk_service.add_chunks_for_document(
                            chunks_data=chunks_to_create
                        )
                    )
                    number_of_chunks_created = len(created_chunk_db_objects)
                    logger.info(
                        f"{task_id_for_log} (Async Logic) 文档 ID: {document_id} 的 {number_of_chunks_created} 个文本块已存入数据库"
                    )
                except Exception as db_chunk_error:
                    logger.error(
                        f"{task_id_for_log} (Async Logic) 存储文本块到数据库时失败 (文档ID: {document_id}): {db_chunk_error}",
                        exc_info=True,
                    )
                    await source_doc_service.update_document_processing_info(
                        document_id=document_response.id,
                        status="error",
                        error_message=f"存储文本块失败: {str(db_chunk_error)[:255]}",
                    )
                    raise db_chunk_error
            else:  # 如果没有文本块产生（例如原文件为空或解析后为空）
                number_of_chunks_created = 0
                logger.info(
                    f"{task_id_for_log} (Async Logic) 文档 ID: {document_id} 未产生文本块可存储。"
                )

            # ==================================================================
            # 第7步：为文本块生成向量嵌入
            # ==================================================================
            if created_chunk_db_objects:  # 仅当有成功创建的文本块时才进行向量化
                texts_to_embed = [
                    chunk.chunk_text for chunk in created_chunk_db_objects
                ]

                logger.info(
                    f"{task_id_for_log} (Async Logic) 开始为 {len(texts_to_embed)} 个文本块生成向量嵌入..."
                )
                try:
                    # 将同步的 embedding 操作放到线程中执行，避免阻塞事件循环
                    embeddings_list = await asyncio.to_thread(
                        get_embeddings,
                        texts_to_embed,                        
                        task_description=settings.EMBEDDING_INSTRUCTION_FOR_RETRIEVAL,
                        is_query=False,
                    )
                    logger.info(
                        f"{task_id_for_log} (Async Logic) 成功生成 {len(embeddings_list)} 组向量嵌入。"
                    )
                except Exception as embed_error:
                    logger.error(
                        f"{task_id_for_log} (Async Logic) 生成向量嵌入失败: {embed_error}",
                        exc_info=True,
                    )
                    await source_doc_service.update_document_processing_info(
                        document_id=document_response.id,
                        status="error",
                        error_message=f"向量化失败: {str(embed_error)[:255]}",
                        number_of_chunks=number_of_chunks_created,  # 已创建的块数量还是记录一下
                    )
                    raise embed_error

                # ==================================================================
                # 第8步：将向量和文本块的关联信息存入向量数据库 (ChromaDB)
                # ==================================================================
                if embeddings_list:
                    chroma_chunk_ids = [
                        str(chunk.id) for chunk in created_chunk_db_objects
                    ]  # ChromaDB 需要字符串ID
                    chroma_metadatas = [
                        {
                            "source_document_id": document_response.id,
                            "original_filename": document_response.original_filename,
                            "text_chunk_db_id": chunk.id,  # 对应 PostgreSQL TextChunk 表的ID
                            "sequence": chunk.sequence_in_document,
                        }
                        for chunk in created_chunk_db_objects
                    ]
                    # 有些向量数据库也允许存储文档本身，ChromaDB 也支持
                    # chroma_documents_text = texts_to_embed

                    try:
                        logger.info(
                            f"{task_id_for_log} (Async Logic) 准备将向量存入 ChromaDB 集合: {settings.CHROMA_COLLECTION_NAME}..."
                        )
                        chroma_collection = await asyncio.to_thread(
                            get_chroma_collection
                        )  # 获取集合也可能是阻塞的

                        # ChromaDB 的 add/upsert 是同步的，也需要用 to_thread
                        await asyncio.to_thread(
                            chroma_collection.add,  # 或者 .upsert 如果你想支持幂等更新
                            ids=chroma_chunk_ids,
                            embeddings=embeddings_list,
                            metadatas=chroma_metadatas,
                            # documents=chroma_documents_text # 可选
                        )
                        logger.info(
                            f"{task_id_for_log} (Async Logic) 成功将 {len(chroma_chunk_ids)} 个向量存入 ChromaDB。"
                        )
                    except Exception as chroma_error:
                        logger.error(
                            f"{task_id_for_log} (Async Logic) 存入 ChromaDB 失败: {chroma_error}",
                            exc_info=True,
                        )
                        await source_doc_service.update_document_processing_info(
                            document_id=document_response.id,
                            status="error",
                            error_message=f"向量存储失败: {str(chroma_error)[:255]}",
                            number_of_chunks=number_of_chunks_created,
                        )
                        raise chroma_error
            else:  # 如果没有文本块被创建和向量化
                logger.info(
                    f"{task_id_for_log} (Async Logic) 没有文本块进行向量化和存储。文档ID: {document_id}"
                )

            # ==================================================================
            # 第9步：更新最终文档状态为 "ready"
            # ==================================================================
            await source_doc_service.update_document_processing_info(
                document_id=document_response.id,
                status="ready",
                set_processed_now=True,
                number_of_chunks=number_of_chunks_created,
                error_message=None,  # 清除之前的错误信息
            )
            logger.info(
                f"{task_id_for_log} (Async Logic) 文档 {document_response.id} 所有处理步骤完成，状态更新为 'ready'"
            )
            return {
                "status": "success",
                "document_id": document_id,
                "chunks_created": number_of_chunks_created,
            }

    except Exception as e:  # 捕获 _execute_document_processing_async 内部的顶层错误
        logger.error(
            f"{task_id_for_log} (Async Logic) 处理文档 ID: {document_id} 时发生无法处理的错误: {str(e)}",
            exc_info=True,
        )
        if document_id:
            async with SessionLocal() as error_db:
                try:
                    error_repo = SourceDocumentRepository(error_db)
                    error_service = SourceDocumentService(
                        error_repo
                    )  # 重新实例化 service
                    await error_service.update_document_processing_info(
                        document_id=document_id,
                        status="error",
                        error_message=f"Celery任务异步逻辑最终错误: {str(e)[:250]}",  # 确保不超过数据库字段长度
                    )
                except Exception as final_update_error:
                    logger.error(
                        f"{task_id_for_log} (Async Logic) 更新最终错误状态时再次失败: {final_update_error}",
                        exc_info=True,
                    )
        raise e  # 将异常向上抛给 asyncio.run()，再由同步任务的 except 块处理

    finally:
        # 最终清理
        # 无论成功还是失败，最后都要关闭数据库引擎的连接池
        logger.info(f"{task_id_for_log} (Async Logic) 关闭数据库连接池...")
        await db_engine.dispose()
        logger.info(f"{task_id_for_log} (Async Logic) 异步资源清理完毕。")
