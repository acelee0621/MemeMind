import asyncio

from loguru import logger
from asgiref.sync import async_to_sync

from app.core.celery_app import celery_app
from app.tasks.utils.doc_process import _execute_document_processing_async
from app.tasks.utils.query_process import execute_query_processing_async


# --- Celery 同步任务入口点 ---
@celery_app.task(
    name="app.tasks.document_task.process_document_task",
    bind=True,
    # autoretry_for=(Exception,), # 自动重试可以根据需要开启
    # retry_kwargs={'max_retries': 3, 'countdown': 60}
)
def process_document_task(self, document_id: int):
    """
    一个健壮的 Celery 任务，用于安全地执行异步代码。

    它遵循以下模式：
    1. 为每个任务创建一个全新的、隔离的事件循环。
    2. 使用 try...finally 结构确保事件循环总是被关闭，防止资源泄露。
    3. 在 try...except 中处理业务逻辑异常，并记录日志。
    """
    task_id_log_prefix = f"[Celery Task ID: {self.request.id}]"
    logger.info(
        f"{task_id_log_prefix} 接收到文档 ID: {document_id}。开始设置异步环境。"
    )

    # 1. 创建全新的事件循环，并设置为当前线程的循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # 2. 将您原来的业务逻辑和异常处理放在这个 try 块中
        #    loop.run_until_complete 会执行异步函数，如果函数内有异常，会在这里被重新抛出
        result = loop.run_until_complete(
            _execute_document_processing_async(document_id, task_id_log_prefix)
        )
        logger.info(f"{task_id_log_prefix} 异步逻辑成功完成，结果: {result}")
        return result
    except Exception as e:
        # 3. 如果异步函数中发生任何未捕获的异常，在这里记录
        logger.error(
            f"{task_id_log_prefix} 异步逻辑中发生致命错误: {e}",
            exc_info=True,  # exc_info=True 会记录完整的堆栈跟踪信息
        )
        # 4. 重新抛出异常，这对于 Celery 至关重要。
        #    Celery 会捕获这个异常，并将任务状态标记为 FAILED。
        #    如果不抛出，Celery 会认为任务成功了。
        raise
    finally:
        # 5. 关键步骤：这个 finally 块中的代码保证【总是】会执行。
        #    无论 try 块是成功返回、还是发生异常，都会执行这里。
        logger.info(f"{task_id_log_prefix} 开始清理异步环境，关闭事件循环。")
        loop.close()


@celery_app.task(name="app.tasks.document_task.process_query_task", bind=True)
def process_query_task(self, message: dict):
    task_id_log_prefix = f"[Celery QueryTask ID: {self.request.id}]"

    query_text = message.get("query_text")
    top_k_final_reranked = message.get("top_k_final_reranked")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    logger.info(
        f"{task_id_log_prefix} (Sync Entry) 接收到查询请求: '{query_text}', top_k: {top_k_final_reranked}"
    )

    if not query_text or top_k_final_reranked is None:
        # ... (参数校验和错误返回) ...
        error_msg = f"无效的查询任务参数: {message}"
        logger.error(f"{task_id_log_prefix} {error_msg}")
        return {"status": "error", "message": error_msg, "result": []}

    try:
        # 直接调用 query_process.py 中的核心异步函数
        # 它内部会处理数据库会话和依赖实例化
        result = loop.run_until_complete(
            execute_query_processing_async(
                query_text=query_text,
                top_k_final_reranked=top_k_final_reranked,
                task_id_for_log=task_id_log_prefix,  # 可以将任务ID前缀也传进去
            )
        )
        logger.info(
            f"{task_id_log_prefix} (Sync Entry) 查询处理完成，结果数量: {len(result) if isinstance(result, list) else 'N/A'}"
        )
        return result
    except Exception as e:
        logger.error(
            f"{task_id_log_prefix} (Sync Entry) 查询处理过程中发生严重错误: {str(e)}",
            exc_info=True,
        )
        raise

    finally:
        logger.info(f"{task_id_log_prefix} 开始清理异步环境，关闭事件循环。")
        loop.close()
