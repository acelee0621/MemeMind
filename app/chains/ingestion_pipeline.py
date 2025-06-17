import asyncio
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_unstructured import UnstructuredLoader

from app.chains.vector_store import get_chroma_vector_store
from app.core.config import settings
from app.schemas.schemas import TextChunkCreate


# --- 流水线的构建块 ---
async def _load_docs(input_dict: dict) -> list[Document]:
    doc_record = input_dict["doc_record"]
    logger.info(f"[1/5 Load] 使用 UnstructuredLoader 加载文档: {doc_record.file_path}")
    loader = UnstructuredLoader(doc_record.file_path)
    loaded_docs = await asyncio.to_thread(loader.load)
    for doc in loaded_docs:
        doc.metadata = {
            "original_filename": doc_record.original_filename,
            "source": doc_record.file_path,
        }
    return loaded_docs


def _split_docs(documents: list[Document]) -> list[Document]:
    logger.info(
        f"[2/5 Split] 使用 RecursiveCharacterTextSplitter 分割 {len(documents)} 个文档..."
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    logger.success(f"[2/5 Split] 分割完成，产生 {len(chunks)} 个文本块。")
    return chunks


async def _store_chunks_to_sql(input_dict: dict) -> list:
    chunks = input_dict["chunks"]
    document_id = input_dict["doc_record"].id
    text_chunk_service = input_dict["text_chunk_service"]

    logger.info(f"[3/5 SQL Store] 准备将 {len(chunks)} 个文本块存入 PostgreSQL...")
    if not chunks:
        return []

    chunks_to_create = [
        TextChunkCreate(
            source_document_id=document_id,
            chunk_text=doc.page_content,
            sequence_in_document=i,
            metadata_json=doc.metadata,
        )
        for i, doc in enumerate(chunks)
    ]
    created_pydantic_chunks = await text_chunk_service.add_chunks_in_bulk(
        chunks_data=chunks_to_create
    )
    logger.success(
        f"[3/5 SQL Store] {len(created_pydantic_chunks)} 个文本块已存入 PostgreSQL。"
    )
    return created_pydantic_chunks


async def _add_to_vector_store(input_dict: dict) -> int:
    split_docs = input_dict["chunks"]
    sql_chunks = input_dict["sql_chunks"]

    if not split_docs or not sql_chunks:
        logger.warning("[4/5 Vector Store] 没有文本块需要存入向量库。")
        return 0

    logger.info(
        f"[4/5 Vector Store] 准备将 {len(split_docs)} 个文本块嵌入并存入 ChromaDB..."
    )

    ids_for_vector_db = [str(chunk.id) for chunk in sql_chunks]
    for i, doc in enumerate(split_docs):
        doc.metadata["text_chunk_pg_id"] = sql_chunks[i].id

    vector_store = get_chroma_vector_store()

    await vector_store.aadd_documents(documents=split_docs, ids=ids_for_vector_db)

    logger.success(
        f"[4/5 Vector Store] {len(split_docs)} 个文本块已成功嵌入并存入 ChromaDB。"
    )
    return len(split_docs)


# --- 主流水线运行函数 ---
async def run_ingestion_pipeline(
    document_id: int,
    session: AsyncSession,
):
    """
    一个完整的、基于LCEL的文档注入流水线。
    """

    logger.info("开始执行基于LCEL的文档注入 Pipeline...")
    from app.repository.doc_repository import SourceDocumentRepository
    from app.services.doc_service import SourceDocumentService
    from app.repository.chunk_repository import TextChunkRepository
    from app.services.chunk_service import TextChunkService

    try:
        # 重新获取与当前会话绑定的服务和仓库
        doc_repo = SourceDocumentRepository(session)
        source_doc_service = SourceDocumentService(doc_repository=doc_repo)
        chunk_repo = TextChunkRepository(session)
        text_chunk_service = TextChunkService(chunk_repo)

        # --- 1. 准备工作：使用 service 层更新状态 ---
        await source_doc_service.update_document_processing_info(
            document_id, status="processing"
        )
        doc_record = await doc_repo.get_by_id(document_id)
        logger.info(f"状态更新为 'processing', 文件路径: '{doc_record.file_path}'")

        # --- 2. 定义LCEL流水线 ---
        ingestion_chain = (
            RunnablePassthrough.assign(
                chunks=RunnableLambda(_load_docs) | RunnableLambda(_split_docs)
            )
            | RunnablePassthrough.assign(
                sql_chunks=RunnableLambda(_store_chunks_to_sql)
            )
            | RunnableLambda(_add_to_vector_store)
        )

        # --- 3. 执行流水线 ---
        initial_input = {
            "doc_record": doc_record,
            "text_chunk_service": text_chunk_service,
        }
        number_of_chunks = await ingestion_chain.ainvoke(initial_input)

        # --- 4. 收尾工作：使用 service 层更新最终状态 ---
        if number_of_chunks > 0:
            await source_doc_service.update_document_processing_info(
                document_id,
                status="ready",
                number_of_chunks=number_of_chunks,
                set_processed_now=True,  # service层会自动处理时间
                error_message=None,
            )
            logger.success("[5/5 Finish] Pipeline 处理成功，文档状态更新为 'ready'。")
            return {"status": "success", "chunks_created": number_of_chunks}
        else:
            await source_doc_service.update_document_processing_info(
                document_id,
                status="error",
                error_message="文档解析后未产生任何文本块",
            )
            logger.warning("[5/5 Finish] 文档解析后未产生任何文本块，任务终止。")
            return {"status": "warning", "message": "No content to process."}

    except Exception as e:
        logger.error(f"Pipeline 处理失败: {e}", exc_info=True)
        # 在异常处理中，也使用 service 层来更新状态
        await source_doc_service.update_document_processing_info(
            document_id, status="error", error_message=str(e)[:500]
        )
        raise e
