from loguru import logger
from asgiref.sync import async_to_sync

from app.core.celery_app import celery_app
from app.tasks.utils.doc_process import _execute_document_processing_async
from app.tasks.utils.query_process import execute_query_processing_async


# --- Celery 同步任务入口点 ---
@celery_app.task(
    name="app.tasks.document_task.process_document_task",
    bind=True,  # bind=True 允许你通过 self 访问任务实例 (例如 self.request.id, self.retry)
    # autoretry_for=(Exception,), # 可以为特定异常配置自动重试
    # retry_kwargs={'max_retries': 3, 'countdown': 60} # 重试参数
)
def process_document_task(self, document_id: int):  # bind=True后，第一个参数是self
    task_id_log_prefix = f"[Celery Task ID: {self.request.id}]"  # 用于日志追踪
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


@celery_app.task(name="app.tasks.document_task.process_query_task", bind=True)
def process_query_task(self, message: dict):    
    task_id_log_prefix = f"[Celery QueryTask ID: {self.request.id}]"
    
    query_text = message.get("query_text")
    top_k_final_reranked = message.get("top_k_final_reranked")

    logger.info(f"{task_id_log_prefix} (Sync Entry) 接收到查询请求: '{query_text}', top_k: {top_k_final_reranked}")

    if not query_text or top_k_final_reranked is None:
        # ... (参数校验和错误返回) ...
        error_msg = f"无效的查询任务参数: {message}"
        logger.error(f"{task_id_log_prefix} {error_msg}")
        return {"status": "error", "message": error_msg, "result": []}

    try:
        # 直接调用 query_process.py 中的核心异步函数
        # 它内部会处理数据库会话和依赖实例化
        result = async_to_sync(execute_query_processing_async)(
            query_text=query_text,
            top_k_final_reranked=top_k_final_reranked,
            task_id_for_log=task_id_log_prefix, # 可以将任务ID前缀也传进去            
        )
        logger.info(f"{task_id_log_prefix} (Sync Entry) 查询处理完成，结果数量: {len(result) if isinstance(result, list) else 'N/A'}")
        return result 
    except Exception as e:
        logger.error(f"{task_id_log_prefix} (Sync Entry) 查询处理过程中发生严重错误: {str(e)}", exc_info=True)
        raise 