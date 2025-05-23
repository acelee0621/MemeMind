import os
import asyncio

import redis
from asgiref.sync import async_to_sync
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.logging import get_logger
from app.core.celery_app import celery_app
from app.core.s3_client import s3_client
from app.core.database import SessionLocal
from app.source_doc.repository import SourceDocumentRepository
from app.source_doc.service import SourceDocumentService
from app.text_chunk.repository import TextChunkRepository
from app.text_chunk.service import TextChunkService
from app.schemas.schemas import TextChunkCreate


logger = get_logger(__name__)


redis_host = os.getenv("REDIS_HOST", "localhost:6379")
REDIS_URL = f"redis://{redis_host}/0"

redis_client = redis.from_url(
    REDIS_URL,
    health_check_interval=30,
)


def parse_txt_bytes(file_bytes: bytes) -> str:
    """
    解析 TXT 文件字节流。
    """
    logger.info("开始解析 TXT 内容...")
    try:
        raw_text = file_bytes.decode(
            "utf-8"
        )  # 尝试 UTF-8，如果可能，考虑更灵活的编码检测
        logger.info(f"TXT 解析完成，提取文本长度: {len(raw_text)}")
        return raw_text
    except UnicodeDecodeError as e:
        logger.error(f"TXT 文件解码失败 (尝试UTF-8): {e}", exc_info=True)
        # 可以尝试其他常见编码或直接报错
        raise ValueError(f"无法解码 TXT 文件 (尝试UTF-8): {e}")
    except Exception as e:
        logger.error(f"解析 TXT 时发生错误: {e}", exc_info=True)
        raise ValueError(f"无法解析 TXT 文件: {e}")


# --- 异步业务逻辑核心 ---
async def _execute_document_processing_async(
    document_id: int, task_id_for_log: str, logger_instance
):
    logger_instance.info(
        f"{task_id_for_log} (Async Logic) 开始处理文档 ID: {document_id}"
    )

    source_doc_service: SourceDocumentService
    text_chunk_service: TextChunkService
    try:
        async with SessionLocal() as db:
            logger_instance.info(
                f"{task_id_for_log} (Async Logic) 数据库会话已为文档 ID {document_id} 创建"
            )

            # 实例化 Repositories 和 Services
            source_doc_repo = SourceDocumentRepository(db)
            source_doc_service = SourceDocumentService(source_doc_repo)
            text_chunk_repo = TextChunkRepository(db)
            text_chunk_service = TextChunkService(text_chunk_repo)

            # 1. 获取文档元数据
            document_response = await source_doc_service.get_document(
                document_id=document_id, current_user=None
            )
            if not document_response:
                logger_instance.error(
                    f"{task_id_for_log} (Async Logic) 文档 ID: {document_id} 未找到。任务终止。"
                )
                # 通常 Service 会抛 NotFoundException，这里是双重保障
                return {"status": "error", "message": "Document not found in DB"}

            logger_instance.info(
                f"{task_id_for_log} (Async Logic) 成功获取文档 {document_response.id} (原始文件名: {document_response.original_filename})"
            )

            # 2. 更新文档状态为 "processing"
            await source_doc_service.update_document_processing_info(
                document_id=document_response.id, current_user=None, status="processing"
            )
            logger_instance.info(
                f"{task_id_for_log} (Async Logic) 文档 {document_response.id} 状态更新为 'processing'"
            )

            # 3. 从 S3 (MinIO) 下载文档内容
            file_content_bytes: bytes
            try:
                s3_object = await asyncio.to_thread(
                    s3_client.get_object,
                    Bucket=document_response.bucket_name,
                    Key=document_response.object_name,
                )
                file_content_bytes = s3_object["Body"].read()
                logger_instance.info(
                    f"{task_id_for_log} (Async Logic) 文档 {document_response.object_name} ({len(file_content_bytes)} bytes) 已从 S3 下载。"
                )
            except Exception as s3_error:
                logger_instance.error(
                    f"{task_id_for_log} (Async Logic) 从 S3 下载文档 {document_response.object_name} 失败: {s3_error}",
                    exc_info=True,
                )
                await source_doc_service.update_document_processing_info(
                    document_id=document_response.id,
                    current_user=None,
                    status="error",
                    error_message=f"S3下载失败: {str(s3_error)[:255]}",
                )
                raise s3_error

            # ==================================================================
            # 第4步：根据文档类型解析文本内容
            # ==================================================================
            raw_text: str = ""
            content_type = document_response.content_type.lower()  # 转小写以便匹配
            original_filename = document_response.original_filename.lower()

            try:
                if "text/plain" in content_type or original_filename.endswith(".txt"):
                    raw_text = parse_txt_bytes(file_content_bytes)
                else:
                    unsupported_msg = f"不支持的文件类型: {document_response.content_type} (文件名: {document_response.original_filename})"
                    logger_instance.warning(f"{task_id_for_log} {unsupported_msg}")
                    await source_doc_service.update_document_processing_info(
                        document_id=document_response.id,
                        current_user=None,
                        status="error",
                        error_message=unsupported_msg,
                    )
                    return {"status": "error", "message": unsupported_msg}  # 任务结束

                if not raw_text.strip():  # 如果解析后文本为空
                    empty_content_msg = "解析后文本内容为空"
                    logger_instance.warning(
                        f"{task_id_for_log} {empty_content_msg} (文档ID: {document_id})"
                    )
                    await source_doc_service.update_document_processing_info(
                        document_id=document_response.id,
                        current_user=None,
                        status="error",
                        error_message=empty_content_msg,
                    )
                    return {"status": "error", "message": empty_content_msg}

                logger_instance.info(
                    f"{task_id_for_log} (Async Logic) 文档 ID: {document_id} 内容解析完成，提取文本长度: {len(raw_text)}"
                )

            except ValueError as parse_error:  # 捕获由解析函数抛出的 ValueError
                logger_instance.error(
                    f"{task_id_for_log} (Async Logic) 解析文档 {document_response.original_filename} 失败: {parse_error}",
                    exc_info=True,
                )
                await source_doc_service.update_document_processing_info(
                    document_id=document_response.id,
                    current_user=None,
                    status="error",
                    error_message=f"文档解析失败: {str(parse_error)[:255]}",
                )
                raise parse_error  # 重新抛出，让 Celery 标记任务失败或重试

            # ==================================================================
            # 第5步：将文本内容分割成小块 (Chunks)
            # ==================================================================
            # 你需要选择并实现一个分块策略。这里使用一个概念性的简单分块器。
            # 实际中你可能会用 LangChain 的 RecursiveCharacterTextSplitter 或类似工具。

            # 示例：简单的按固定长度（大致）分块，带重叠 (你需要一个更成熟的方案)
            chunk_size = 1000  # 每个块的目标字符数
            chunk_overlap = 100  # 相邻块之间的重叠字符数

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
                logger_instance.warning(
                    f"{task_id_for_log} (Async Logic) 文本分块结果为空，但原始文本不为空。文档ID: {document_id}"
                )
                # 也许也应该标记为错误

            logger_instance.info(
                f"{task_id_for_log} (Async Logic) 文档 ID: {document_id} 被分割成 {len(chunks_texts_list)} 个文本块"
            )

            # ==================================================================
            # 第6步：将文本块存入数据库 (使用 TextChunkService)
            # ==================================================================
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
                    created_chunk_responses = (
                        await text_chunk_service.add_chunks_for_document(
                            chunks_data=chunks_to_create
                        )
                    )
                    number_of_chunks_created = len(created_chunk_responses)
                    logger_instance.info(
                        f"{task_id_for_log} (Async Logic) 文档 ID: {document_id} 的 {number_of_chunks_created} 个文本块已存入数据库"
                    )
                except Exception as db_chunk_error:
                    logger_instance.error(
                        f"{task_id_for_log} (Async Logic) 存储文本块到数据库时失败 (文档ID: {document_id}): {db_chunk_error}",
                        exc_info=True,
                    )
                    await source_doc_service.update_document_processing_info(
                        document_id=document_response.id,
                        current_user=None,
                        status="error",
                        error_message=f"存储文本块失败: {str(db_chunk_error)[:255]}",
                    )
                    raise db_chunk_error
            else:  # 如果没有文本块产生（例如原文件为空或解析后为空）
                number_of_chunks_created = 0
                logger_instance.info(
                    f"{task_id_for_log} (Async Logic) 文档 ID: {document_id} 未产生文本块可存储。"
                )

            # ==================================================================
            # 第7、8步占位：向量化文本块并存入向量数据库
            # ==================================================================
            # TODO:
            # 1. 加载/获取你的文本嵌入模型 (Embedding Model)
            #    embedding_model = SentenceTransformer('your-chosen-model')
            # 2. 从数据库获取刚创建的文本块内容 (如果需要，或者直接用 chunks_texts_list)
            #    db_chunks_for_embedding = await text_chunk_service.get_document_chunks_for_display(document_id, limit=number_of_chunks_created)
            #    texts_to_embed = [chunk.chunk_text for chunk in db_chunks_for_embedding]
            #    chunk_ids_for_vector_db = [str(chunk.id) for chunk in db_chunks_for_embedding]
            # 3. 为文本块生成向量
            #    embeddings = embedding_model.encode(texts_to_embed)
            # 4. 将 (向量, TextChunk ID, 其他元数据) 存入向量数据库
            #    vector_db.add(ids=chunk_ids_for_vector_db, embeddings=embeddings, metadatas=...)
            logger_instance.info(
                f"{task_id_for_log} (Async Logic) TODO: 实现向量化和向量存储逻辑 for doc ID: {document_id}"
            )

            # ==================================================================
            # 第9步：更新最终文档状态为 "ready"
            # ==================================================================
            await source_doc_service.update_document_processing_info(
                document_id=document_response.id,
                current_user=None,
                status="ready",
                set_processed_now=True,
                number_of_chunks=number_of_chunks_created,
                error_message=None,  # 清除之前的错误信息
            )
            logger_instance.info(
                f"{task_id_for_log} (Async Logic) 文档 {document_response.id} 所有处理步骤完成，状态更新为 'ready'"
            )
            return {
                "status": "success",
                "document_id": document_id,
                "chunks_created": number_of_chunks_created,
            }

    except Exception as e:  # 捕获 _execute_document_processing_async 内部的顶层错误
        logger_instance.error(
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
                        current_user=None,
                        status="error",
                        error_message=f"Celery任务异步逻辑最终错误: {str(e)[:250]}",  # 确保不超过数据库字段长度
                    )
                except Exception as final_update_error:
                    logger_instance.error(
                        f"{task_id_for_log} (Async Logic) 更新最终错误状态时再次失败: {final_update_error}",
                        exc_info=True,
                    )
        raise e  # 将异常向上抛给 asyncio.run()，再由同步任务的 except 块处理


# --- Celery 同步任务入口点 ---
@celery_app.task(
    name="app.tasks.document_task.process_document_task",
    bind=True,  # bind=True 允许你通过 self 访问任务实例 (例如 self.request.id, self.retry)
    # autoretry_for=(Exception,), # 可以为特定异常配置自动重试
    # retry_kwargs={'max_retries': 3, 'countdown': 60} # 重试参数
)
def process_document_task(self, document_id: int):  # bind=True后，第一个参数是self
    task_id_log_prefix = f"[Celery Task ID: {self.request.id}]"  # 用于日志追踪
    logger.info(f"{task_id_log_prefix} 开始处理文档 ID: {document_id}")

    logger.info(
        f"{task_id_log_prefix} (Sync Entry with asgiref - Minimal Test) 接收到文档 ID: {document_id}"
    )
    try:
        result = async_to_sync(_execute_document_processing_async)(
            document_id, task_id_log_prefix, logger
        )
        logger.info(
            f"{task_id_log_prefix} (Sync Entry with asgiref - Minimal Test) 异步逻辑完成，结果: {result}"
        )
        logger.info(f"{task_id_log_prefix} 文档 ID: {document_id} 处理任务结束。")
        return result
    except Exception as e:
        logger.error(
            f"{task_id_log_prefix} (Sync Entry with asgiref - Minimal Test) 异步逻辑中发生错误: {str(e)}",
            exc_info=True,
        )
        raise
