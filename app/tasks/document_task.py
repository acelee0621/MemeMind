import asyncio
from loguru import logger
from app.core.celery_app import celery_app
from app.chains.ingestion_pipeline import run_ingestion_pipeline


# --- 文档处理任务 ---
@celery_app.task(
    name="app.tasks.document_task.process_document_task",
    bind=True,
)
def process_document_task(self, document_id: int):
    """
    Celery 任务入口点，负责调用 LangChain 文档注入流水线。
    """
    task_id_log_prefix = f"[Celery Task ID: {self.request.id}]"
    logger.info(f"{task_id_log_prefix} 接收到文档 ID: {document_id}，准备执行注入流水线。")

    # 使用 asyncio.run() 是在同步函数中运行异步代码的现代、推荐方式
    try:
        # 核心逻辑只有一行：调用我们的流水线
        result = asyncio.run(run_ingestion_pipeline(document_id, task_id_log_prefix))
        logger.success(f"{task_id_log_prefix} 流水线执行成功，结果: {result}")
        return result
    except Exception as e:
        logger.error(
            f"{task_id_log_prefix} 流水线执行过程中发生未捕获的错误: {e}",
            exc_info=True,
        )
        raise # 重新抛出，让 Celery 将任务标记为 FAILED
    

# --- 查询处理任务（暂时保持不变，我们稍后会重构它） ---
@celery_app.task(name="app.tasks.document_task.process_query_task", bind=True)
def process_query_task(self, message: dict):
    # ... 你的旧查询逻辑暂时保留 ...
    # 我们将在下一步重构这个部分
    pass