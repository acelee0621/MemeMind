import asyncio
from loguru import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document

from app.core.config import settings
from app.core.database import create_engine_and_session_for_celery
from app.repository.doc_repository import SourceDocumentRepository
from app.services.doc_service import SourceDocumentService
from app.repository.chunk_repository import TextChunkRepository
from app.services.chunk_service import TextChunkService
from app.schemas.schemas import TextChunkCreate
from app.chains.vector_store import get_chroma_vector_store


async def run_ingestion_pipeline(document_id: int, task_id_for_log: str):
    """
    一个完整的、基于 LangChain 的文档注入流水线。
    """
    logger.info(f"{task_id_for_log} 开始执行 LangChain 文档注入流水线...")
    db_engine, SessionLocal = create_engine_and_session_for_celery()

    try:
        async with SessionLocal() as db:
            source_doc_repo = SourceDocumentRepository(db)
            source_doc_service = SourceDocumentService(source_doc_repo)
            text_chunk_repo = TextChunkRepository(db)
            text_chunk_service = TextChunkService(text_chunk_repo)

            await source_doc_service.update_document_processing_info(
                document_id, status="processing"
            )
            doc_record = await source_doc_repo.get_by_id(document_id)
            logger.info(
                f"{task_id_for_log} 状态更新为 'processing'，文件路径: '{doc_record.file_path}'"
            )

            # 1. Load (使用新的 UnstructuredLoader)
            logger.info(
                f"{task_id_for_log} [Load] 正在使用 UnstructuredLoader 加载文档..."
            )
            loader = UnstructuredLoader(doc_record.file_path)
            loaded_docs = await asyncio.to_thread(loader.load)

            for doc in loaded_docs:
                # 确保每个从文件中加载出来的 Document 对象都知道它的原始文件名
                doc.metadata["original_filename"] = doc_record.original_filename

            # 2. Split
            logger.info(f"{task_id_for_log} [Split] 正在分割文档...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                length_function=len,
            )
            split_docs = text_splitter.split_documents(loaded_docs)
            logger.info(f"{task_id_for_log} [Filter] 正在清理文档元数据...")
            split_docs = filter_complex_metadata(split_docs)
            for doc in split_docs:
                if "metadata" not in doc:
                    doc.metadata = {}
                doc.metadata["original_filename"] = doc_record.original_filename
                # 保留 source 字段，它指向内部存储路径，也很有用
                doc.metadata["source"] = doc_record.file_path
            number_of_chunks = len(split_docs)
            if number_of_chunks == 0:
                logger.warning(
                    f"{task_id_for_log} 文档解析后未产生任何文本块，任务终止。"
                )
                await source_doc_service.update_document_processing_info(
                    document_id, status="error", error_message="解析后文本内容为空"
                )
                return {"status": "warning", "message": "No content to process."}
            logger.success(
                f"{task_id_for_log} [Split] 分割完成，共产生 {number_of_chunks} 个文本块。"
            )

            logger.info(
                f"{task_id_for_log} 正在为 {number_of_chunks} 个文本块构建干净的元数据..."
            )
            final_docs_for_storage = []
            for i, doc in enumerate(split_docs):
                # 为每个块创建一个干净、可控的元数据字典
                final_metadata = {
                    "source": doc_record.file_path,
                    "original_filename": doc_record.original_filename,  # 明确注入原始文件名
                    "sequence": i,
                }
                # 从 unstructured 的结果中安全地提取页码（如果存在）
                if "page" in doc.metadata:
                    final_metadata["page"] = doc.metadata["page"]

                new_doc = Document(
                    page_content=doc.page_content, metadata=final_metadata
                )
                final_docs_for_storage.append(new_doc)

            # 3. Store to SQL
            logger.info(f"{task_id_for_log} [Store SQL] 正在将文本块存入 PostgreSQL...")
            chunks_to_create = [
                TextChunkCreate(
                    source_document_id=document_id,
                    chunk_text=doc.page_content,
                    sequence_in_document=i,
                    metadata_json=doc.metadata,
                )
                for i, doc in enumerate(split_docs)
            ]

            created_chunk_orm_objects = (
                await text_chunk_service.add_chunks_in_bulk(
                    chunks_data=chunks_to_create
                )
            )

            logger.success(
                f"{task_id_for_log} [Store SQL] {len(created_chunk_orm_objects)} 个文本块已存入 PostgreSQL。"
            )

            # 4. Embed & Store to Vector DB
            logger.info(
                f"{task_id_for_log} [Embed & Store Vector] 准备将文本块存入 ChromaDB..."
            )
            vector_store = get_chroma_vector_store()

            for i, doc in enumerate(split_docs):
                doc.metadata["text_chunk_pg_id"] = created_chunk_orm_objects[i].id

            await asyncio.to_thread(
                vector_store.add_documents,
                documents=split_docs,
                ids=[str(chunk.id) for chunk in created_chunk_orm_objects],
            )
            logger.success(
                f"{task_id_for_log} [Embed & Store Vector] {number_of_chunks} 个文本块已成功嵌入并存入 ChromaDB。"
            )
        
            # 5. 结束阶段
            await source_doc_service.update_document_processing_info(
                document_id,
                status="ready",
                number_of_chunks=number_of_chunks,
                set_processed_now=True,
                error_message=None,
            )
            logger.success(
                f"{task_id_for_log} 流水线处理成功，文档状态更新为 'ready'。"
            )
            return {"status": "success", "chunks_created": number_of_chunks}

    except Exception as e:
        logger.error(f"{task_id_for_log} 流水线处理失败: {e}", exc_info=True)
        async with SessionLocal() as error_db:
            error_repo = SourceDocumentRepository(error_db)
            error_service = SourceDocumentService(error_repo)
            await error_service.update_document_processing_info(
                document_id, status="error", error_message=str(e)[:500]
            )
        raise e
    finally:
        await db_engine.dispose()
        logger.info(f"{task_id_for_log} 数据库连接池已关闭。")
