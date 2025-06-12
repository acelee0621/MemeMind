import asyncio

from loguru import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

from app.core.config import settings
from app.core.database import create_engine_and_session_for_celery
from app.repository.doc_repository import SourceDocumentRepository
from app.services.doc_service import SourceDocumentService
from app.repository.chunk_repository import TextChunkRepository
from app.services.chunk_service import TextChunkService
from app.schemas.schemas import TextChunkCreate
from app.chains.vector_store import get_chroma_vector_store


def standardize_metadata(
    doc: Document, original_filename: str, file_path: str
) -> Document:
    """统一设置文档元数据，确保一致性"""
    if "metadata" not in doc:
        doc.metadata = {}
    doc.metadata.update({"original_filename": original_filename, "source": file_path})
    return doc


async def load_documents(
    file_path: str, original_filename: str, task_id_for_log: str
) -> list[Document]:
    """加载文档并设置初始元数据"""
    logger.info(f"{task_id_for_log} [Load] 正在使用 UnstructuredLoader 加载文档...")
    loader = UnstructuredLoader(file_path)
    loaded_docs = await asyncio.to_thread(loader.load)
    for doc in loaded_docs:
        standardize_metadata(doc, original_filename, file_path)
    return loaded_docs


async def split_documents(
    loaded_docs: list[Document], task_id_for_log: str
) -> list[Document]:
    """分割文档并清理元数据"""
    logger.info(f"{task_id_for_log} [Split] 正在分割文档...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
    )
    split_docs = text_splitter.split_documents(loaded_docs)

    logger.info(f"{task_id_for_log} [Filter] 正在清理文档元数据...")
    original_metadata = [doc.metadata.copy() for doc in split_docs]
    split_docs = filter_complex_metadata(split_docs)

    for i, doc in enumerate(split_docs):
        removed_keys = set(original_metadata[i].keys()) - set(doc.metadata.keys())
        if removed_keys:
            logger.debug(f"{task_id_for_log} [Filter] 移除的元数据字段: {removed_keys}")
        # 重新设置元数据，确保一致性
        standardize_metadata(
            doc,
            original_metadata[i]["original_filename"],
            original_metadata[i]["source"],
        )

    return split_docs


async def store_to_sql(
    split_docs: list[Document],
    document_id: int,
    text_chunk_service: TextChunkService,
    task_id_for_log: str,
) -> list:
    """将文本块存储到 PostgreSQL"""
    logger.info(f"{task_id_for_log} [Store to SQL] 正在将文本块存入 PostgreSQL...")
    chunks_to_create = [
        TextChunkCreate(
            source_document_id=document_id,
            chunk_text=doc.page_content,
            sequence_in_document=i,
            metadata_json=doc.metadata,
        )
        for i, doc in enumerate(split_docs)
    ]
    created_chunk_orm_objects = await text_chunk_service.add_chunks_in_bulk(
        chunks_data=chunks_to_create
    )
    logger.success(
        f"{task_id_for_log} [Store to SQL] {len(created_chunk_orm_objects)} 个文本块已存入 PostgreSQL。"
    )
    return created_chunk_orm_objects


async def store_to_vector_db(
    split_docs: list[Document], created_chunk_orm_objects: list, task_id_for_log: str
) -> None:
    """将文本块嵌入并存储到 ChromaDB"""
    logger.info(
        f"{task_id_for_log} [Embedding & Store to VectorDB] 准备将文本块存入 ChromaDB..."
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
        f"{task_id_for_log} [Embedding & Store to VectorDB] {len(split_docs)} 个文本块已成功嵌入并存入 ChromaDB。"
    )


async def run_ingestion_pipeline(
    document_id: int, task_id_for_log: str
) -> dict[str, any]:
    """
    一个完整的、基于 LangChain 的文档注入流水线。
    """
    task_logger = logger.bind(task_id=task_id_for_log)
    task_logger.info("开始执行 LangChain 文档注入 Pipeline...")

    db_engine, SessionLocal = create_engine_and_session_for_celery()

    try:
        async with SessionLocal() as db:
            source_doc_repo = SourceDocumentRepository(db)
            source_doc_service = SourceDocumentService(source_doc_repo)
            text_chunk_repo = TextChunkRepository(db)
            text_chunk_service = TextChunkService(text_chunk_repo)

            # 更新文档状态为处理中
            await source_doc_service.update_document_processing_info(
                document_id, status="processing"
            )
            doc_record = await source_doc_repo.get_by_id(document_id)
            task_logger.info(
                f"状态更新为 'processing'，文件路径: '{doc_record.file_path}'"
            )

            # 1. 加载文档
            loaded_docs = await load_documents(
                doc_record.file_path, doc_record.original_filename, task_id_for_log
            )

            # 2. 分割文档
            split_docs = await split_documents(loaded_docs, task_id_for_log)
            number_of_chunks = len(split_docs)

            if number_of_chunks == 0:
                task_logger.warning("文档解析后未产生任何文本块，任务终止。")
                await source_doc_service.update_document_processing_info(
                    document_id, status="error", error_message="解析后文本内容为空"
                )
                return {"status": "warning", "message": "No content to process."}
            task_logger.success(
                f"[Split] 分割完成，共产生 {number_of_chunks} 个文本块。"
            )

            # 3. 存储到 SQL
            created_chunk_orm_objects = await store_to_sql(
                split_docs, document_id, text_chunk_service, task_id_for_log
            )

            # 4. 嵌入并存储到向量数据库
            await store_to_vector_db(
                split_docs, created_chunk_orm_objects, task_id_for_log
            )

            # 5. 更新文档状态为完成
            await source_doc_service.update_document_processing_info(
                document_id,
                status="ready",
                number_of_chunks=number_of_chunks,
                set_processed_now=True,
                error_message=None,
            )
            task_logger.success("Pipeline 处理成功，文档状态更新为 'ready'。")
            return {"status": "success", "chunks_created": number_of_chunks}

    except Exception as e:
        task_logger.error(f"Pipeline 处理失败: {e}", exc_info=True)
        async with SessionLocal() as error_db:
            error_repo = SourceDocumentRepository(error_db)
            error_service = SourceDocumentService(error_repo)
            await error_service.update_document_processing_info(
                document_id, status="error", error_message=str(e)[:500]
            )
        raise e
    finally:
        await db_engine.dispose()
        task_logger.info("数据库连接池已关闭。")
