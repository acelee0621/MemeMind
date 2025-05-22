import os
import tempfile  # 用于处理临时文件
# from pathlib import Path # 如果需要处理文件路径

import redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.celery_app import celery_app
from app.core.config import settings
from app.core.s3_client import s3_client
from app.core.database import SessionLocal

# --- 导入实例化 Service 的辅助函数 ---
from app.tasks.dependencies import get_document_service_for_task
# from app.schemas.schemas import SourceDocumentUpdate # 如果需要构造更新的 Pydantic 模型


# --- 文本解析库 (示例) ---
# from pypdf import PdfReader # 处理 PDF
# import markdown # 处理 Markdown
# from bs4 import BeautifulSoup # 从 Markdown (HTML) 中提取文本

# --- 文本分块库 (示例) ---
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 向量模型 (示例) ---
# from sentence_transformers import SentenceTransformer

# --- 向量数据库 (示例) ---
# import chromadb # 或者 FAISS, LanceDB 等


redis_host = os.getenv("REDIS_HOST", "localhost:6379")
REDIS_URL = f"redis://{redis_host}/0"

redis_client = redis.from_url(
    REDIS_URL,
    health_check_interval=30,
)


# --- 初始化 (通常在 Celery worker 启动时或任务内部按需进行) ---
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2') # 或者你选择的其他模型，确保它支持中文（如果需要）
# vector_db_client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH) # 示例
# vector_db_collection = vector_db_client.get_or_create_collection(name=settings.CHROMA_COLLECTION_NAME)


@celery_app.task(
    name="app.tasks.document_task.process_document_task",
    bind=True,  # bind=True 允许你通过 self 访问任务实例 (例如 self.request.id, self.retry)
    # autoretry_for=(Exception,), # 可以为特定异常配置自动重试
    # retry_kwargs={'max_retries': 3, 'countdown': 5} # 重试参数
)
async def process_document_task(
    self, document_id: int
):  # bind=True后，第一个参数是self
    """
    Celery 任务：处理已上传的文档。
    1. 从数据库获取文档元数据。
    2. 更新文档状态为 "processing"。
    3. 从 S3 (MinIO) 下载文档。
    4. 根据文档类型解析文本内容。
    5. 将文本内容分割成小块 (chunks)。
    6. 将文本块存入数据库 (TextChunk 表)。
    7. 为每个文本块生成向量嵌入 (embeddings)。
    8. 将向量和文本块的关联信息存入向量数据库。
    9. 更新文档状态为 "ready" 或 "error"。
    """
    logger = process_document_task.get_logger()  # 获取 Celery 任务专用的 logger
    task_id_log_prefix = f"[Celery Task ID: {self.request.id}]"  # 用于日志追踪
    logger.info(f"{task_id_log_prefix} 开始处理文档 ID: {document_id}")

    # !!! 0. 准备数据库会话和服务/仓库 (根据你的项目结构调整)
    # 例如 (需要你在项目中定义 SessionLocal 和相关服务/仓库的初始化方式):
    # from app.core.db import SessionLocal
    # from app.source_doc.service import SourceDocumentService # 或者直接用 Repository
    # from app.source_doc.repository import SourceDocumentRepository
    # from app.chunk.repository import TextChunkRepository

    db: AsyncSession | None = None  # 初始化 db 变量，确保在 finally 中可用
    try:
        # 3. 使用 async with SessionLocal() 来获取和管理异步数据库会话
        async with SessionLocal() as db:
            logger.info(
                f"{task_id_log_prefix} 数据库会话已为文档 ID {document_id} 创建"
            )

            # 4. 使用获取到的会话实例化 Service
            source_doc_service = get_document_service_for_task(db)
            # 如果你希望通过 Service 层来操作，可以这样：
            # from app.source_doc.service import SourceDocumentService
            # source_doc_service = SourceDocumentService(source_doc_repo)

            # ==================================================================
            # 第一步：获取文档元数据
            # ==================================================================            
            document = await source_doc_service.get_document(document_id)

            if not document:
                logger.error(
                    f"{task_id_log_prefix} 文档 ID: {document_id} 在数据库中未找到。任务终止。"
                )
                # 此处可以直接 return，因为 document 不存在，后续操作无法进行
                # Celery 任务会成功结束，但没有实际处理。
                # 如果认为这是一个不应发生的情况，也可以抛出异常：
                # raise ValueError(f"文档 ID: {document_id} 未找到，无法处理。")
                return

            logger.info(
                f"{task_id_log_prefix} 成功获取文档 {document.id} (原始文件名: {document.original_filename}, MinIO对象名: {document.object_name})"
            )

            # ==================================================================
            # 后续步骤将在这里添加 (目前暂时 pass)
            # ==================================================================
            logger.info(
                f"{task_id_log_prefix} 文档 ID: {document_id} 的核心处理逻辑将在此实现..."
            )

            # 示例：假设这是处理的最后一步，更新状态为 "ready"
            # (实际应该在所有解析、分块、向量化成功后调用)
            # 注意：你需要使用你之前创建的 update_document_processing_info 服务层方法
            # 或者直接使用 repository.update 方法。
            # 例如，如果你有一个服务层方法:
            # from app.source_doc.service import SourceDocumentService # 假设已实例化
            # from app.schemas.schemas import SourceDocumentUpdate
            # from datetime import datetime, timezone
            # source_doc_service = SourceDocumentService(source_doc_repo) # 实例化服务
            # await source_doc_service.update_document_processing_info(
            #     document_id=document.id,
            #     status="ready", # 假设这是处理成功的状态
            #     set_processed_now=True,
            #     number_of_chunks=0 # 暂时用0代替，后续会计算
            # )
            # logger.info(f"{task_id_log_prefix} 文档 ID: {document_id} 模拟处理完成，状态已更新。")

            # TODO: 在这里添加文档下载、解析、分块、向量化、存入向量数据库的逻辑
            pass  # 当前的 pass 语句

    except Exception as e:
        logger.error(
            f"{task_id_log_prefix} 处理文档 ID: {document_id} 时发生无法恢复的错误: {str(e)}",
            exc_info=True,
        )
        # 在这里，你可以尝试通过服务层将文档状态更新为 "error"
        # 这需要小心处理，因为如果错误是数据库连接本身的问题，再次尝试写入可能会失败
        # 一种更健壮的方式是，如果 update_document_processing_info 方法内部能处理自己的会话创建，
        # 或者你在这里创建一个新的短会话专门用于错误状态更新。
        # 例如:
        # async with SessionLocal() as error_db:
        #     try:
        #         error_repo = SourceDocumentRepository(error_db)
        #         error_service = SourceDocumentService(error_repo)
        #         await error_service.update_document_processing_info(
        #             document_id=document_id,
        #             status="error",
        #             error_message=str(e)[:1024] # 限制错误信息长度
        #         )
        #         logger.info(f"{task_id_log_prefix} 文档 ID: {document_id} 状态因错误已更新为 'error'")
        #     except Exception as update_error:
        #         logger.error(f"{task_id_log_prefix} 更新文档 ID: {document_id} 状态为 'error' 时再次失败: {update_error}", exc_info=True)

        # 重新抛出异常，Celery 会根据配置（如 autoretry_for）进行重试，
        # 或者在重试次数耗尽后将任务标记为 FAILURE。
        raise
    # `async with SessionLocal() as db:` 会自动处理会话的关闭，即使发生异常。
    logger.info(f"{task_id_log_prefix} 文档 ID: {document_id} 处理任务结束。")
