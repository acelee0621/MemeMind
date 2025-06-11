import asyncio
from loguru import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader

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
    它取代了旧的 _execute_document_processing_async 函数。
    流程: Load -> Split -> Store (SQL) -> Embed & Store (Vector)
    """
    logger.info(f"{task_id_for_log} 开始执行 LangChain 文档注入流水线...")

    # 为 Celery 任务创建独立的数据库连接
    db_engine, SessionLocal = create_engine_and_session_for_celery()
    
    try:
        async with SessionLocal() as db:
            # --- 0. 准备工作：初始化服务并更新状态 ---
            source_doc_repo = SourceDocumentRepository(db)
            source_doc_service = SourceDocumentService(source_doc_repo)
            text_chunk_repo = TextChunkRepository(db)
            text_chunk_service = TextChunkService(text_chunk_repo)

            await source_doc_service.update_document_processing_info(document_id, status="processing")
            doc_record = await source_doc_repo.get_by_id(document_id)
            logger.info(f"{task_id_for_log} 状态更新为 'processing'，文件路径: '{doc_record.file_path}'")

            # --- 1. Load (加载) ---
            # 使用 LangChain 的 UnstructuredFileLoader 直接从本地文件加载。            
            logger.info(f"{task_id_for_log} [Load] 正在使用 UnstructuredFileLoader 加载文档...")
            loader = UnstructuredFileLoader(doc_record.file_path)
            loaded_docs = await asyncio.to_thread(loader.load)

            # --- 2. Split (分割) ---
            # 使用标准的 LangChain 文本分割器。
            logger.info(f"{task_id_for_log} [Split] 正在分割文档...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                length_function=len,
            )
            split_docs = text_splitter.split_documents(loaded_docs)
            number_of_chunks = len(split_docs)
            if number_of_chunks == 0:
                logger.warning(f"{task_id_for_log} 文档解析后未产生任何文本块，任务终止。")
                await source_doc_service.update_document_processing_info(
                    document_id, status="error", error_message="解析后文本内容为空"
                )
                return {"status": "warning", "message": "No content to process."}
            logger.success(f"{task_id_for_log} [Split] 分割完成，共产生 {number_of_chunks} 个文本块。")

            # --- 3. Store to SQL (存入关系型数据库) ---            
            logger.info(f"{task_id_for_log} [Store SQL] 正在将文本块存入 PostgreSQL...")
            chunks_to_create = [
                TextChunkCreate(
                    source_document_id=document_id,
                    chunk_text=doc.page_content,
                    sequence_in_document=i,
                    metadata_json=doc.metadata, # unstructured 的元数据可以直接存入
                ) for i, doc in enumerate(split_docs)
            ]
            # 我们需要批量创建后返回的 ORM 对象，以获取它们的ID
            created_chunk_orm_objects = await text_chunk_service.add_chunks_in_bulk(chunks_data=chunks_to_create)
            logger.success(f"{task_id_for_log} [Store SQL] {len(created_chunk_orm_objects)} 个文本块已存入 PostgreSQL。")
            
            # --- 4. Embed & Store to Vector DB (嵌入并存入向量数据库) ---
            # 这是最核心的 LangChain 魔法
            logger.info(f"{task_id_for_log} [Embed & Store Vector] 准备将文本块存入 ChromaDB...")
            vector_store = get_chroma_vector_store()

            # 为了能从向量库关联回SQL，我们将SQL的ID加入到元数据中
            for i, doc in enumerate(split_docs):
                doc.metadata["text_chunk_pg_id"] = created_chunk_orm_objects[i].id
            
            # LangChain 的 .add_documents 会自动处理嵌入和存储的所有细节
            await asyncio.to_thread(
                vector_store.add_documents,
                documents=split_docs,
                ids=[str(chunk.id) for chunk in created_chunk_orm_objects] # 使用PG的ID作为ChromaDB的ID，确保唯一性
            )
            logger.success(f"{task_id_for_log} [Embed & Store Vector] {number_of_chunks} 个文本块已成功嵌入并存入 ChromaDB。")
            
            # --- 5. 结束阶段：更新最终状态 ---
            await source_doc_service.update_document_processing_info(
                document_id, status="ready", number_of_chunks=number_of_chunks,
                set_processed_now=True, error_message=None
            )
            logger.success(f"{task_id_for_log} 流水线处理成功，文档状态更新为 'ready'。")
            return {"status": "success", "chunks_created": number_of_chunks}

    except Exception as e:
        logger.error(f"{task_id_for_log} 流水线处理失败: {e}", exc_info=True)
        # 异常处理：更新数据库状态为 "error"
        async with SessionLocal() as error_db:
            error_repo = SourceDocumentRepository(error_db)
            error_service = SourceDocumentService(error_repo)
            await error_service.update_document_processing_info(
                document_id, status="error", error_message=str(e)[:500]
            )
        raise e # 重新抛出异常，让Celery知道任务失败
    finally:
        await db_engine.dispose()
        logger.info(f"{task_id_for_log} 数据库连接池已关闭。")