from functools import lru_cache

import torch
from loguru import logger
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from app.core.config import settings

class BGEInstructionalEmbeddings(HuggingFaceEmbeddings):
    """
    为 BAAI/bge 系列 embedding 模型定制的嵌入类。
    
    它会自动为所有"查询"任务的文本添加BGE模型要求的特定指令。
    该类同时支持同步和异步操作。
    """
    
    # BGE中文模型进行检索任务时，官方推荐的指令
    query_instruction: str = "为这个句子生成表示以用于检索相关文章："

    def embed_query(self, text: str) -> list[float]:
        """
        对单个查询进行同步嵌入，并自动添加指令。
        """
        instructed_text = self.query_instruction + text
        return super().embed_query(instructed_text)

    async def aembed_query(self, text: str) -> list[float]:
        """
        对单个查询进行异步嵌入，并自动添加指令。
        """
        instructed_text = self.query_instruction + text
        return await super().aembed_query(instructed_text)

@lru_cache(maxsize=1)
def get_bge_embeddings() -> BGEInstructionalEmbeddings:
    """
    加载并缓存 BAAI BGE 嵌入模型。
    """
    logger.info("开始初始化 BAAI BGE 嵌入模型组件...")

    # 自动设备检测
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("检测到 CUDA，BGE Embedding 将使用 GPU。")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("检测到 MPS (Apple Silicon)，BGE Embedding 将使用 MPS。")
    else:
        device = "cpu"
        logger.info("未检测到 CUDA 或 MPS，BGE Embedding 将使用 CPU。")
    
    try:
        # 使用我们定制的 BGEInstructionalEmbeddings 类
        bge_embeddings = BGEInstructionalEmbeddings(            
            model_name=settings.EMBEDDING_MODEL_PATH,
            model_kwargs={"device": device},
            encode_kwargs={
                # BGE 模型推荐进行归一化
                "normalize_embeddings": True,
            },
        )
        logger.success(f"BAAI BGE 嵌入模型组件初始化成功，运行于设备: '{device}'")
        return bge_embeddings
    except Exception as e:
        logger.error(f"初始化 BAAI BGE 嵌入模型组件失败: {e}", exc_info=True)
        raise