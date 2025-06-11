from functools import lru_cache

import torch
from loguru import logger
from langchain.embeddings import HuggingFaceEmbeddings
from app.core.config import settings

# --- 1. 自定义 LangChain 嵌入类以支持 Qwen 的指令格式 ---

class QwenInstructionalEmbeddings(HuggingFaceEmbeddings):
    """
    一个自定义的嵌入类，继承自 HuggingFaceEmbeddings。
    它专门用于处理像 Qwen 这样需要在查询（Query）前添加特定指令（Instruction）的模型。
    """
    def __init__(self, query_instruction: str, **kwargs):
        """
        在初始化时，接收一个用于查询的指令字符串。
        
        Args:
            query_instruction (str): 添加在每个查询文本前的指令。
            **kwargs: 传递给父类 HuggingFaceEmbeddings 的其他所有参数。
        """
        super().__init__(**kwargs)
        self.query_instruction = query_instruction
        logger.info(f"自定义查询指令已设置: '{self.query_instruction}'")

    def embed_query(self, text: str) -> list[float]:
        """
        重写 embed_query 方法。
        这是 LangChain 中专门用于处理单个查询文本的方法。
        """
        # 按照 Qwen 的要求，格式化查询文本
        instructed_text = f"Instruct: {self.query_instruction}\nQuery: {text}"
        
        # 调用父类的 embed_query 方法，用格式化后的文本进行嵌入
        return super().embed_query(instructed_text)

    # embed_documents 方法无需重写，因为 Qwen 的文档侧不需要指令，
    # 父类的默认行为（直接嵌入文本列表）正好符合要求。


# --- 2. 创建并缓存嵌入模型实例的工厂函数 ---

@lru_cache(maxsize=1)
def get_qwen_embeddings() -> QwenInstructionalEmbeddings:
    """
    加载并缓存 Qwen 嵌入模型，返回一个配置好的自定义实例。
    使用 lru_cache 确保在应用生命周期内模型只被加载一次。
    """
    logger.info("开始初始化 Qwen 嵌入模型组件...")

    # --- 自动设备检测 ---
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info("检测到 CUDA，将使用 GPU。")
    elif torch.backends.mps.is_available():
        device = 'mps'
        logger.info("检测到 MPS (Apple Silicon)，将使用 MPS。")
    else:
        device = 'cpu'
        logger.info("未检测到 CUDA 或 MPS，将使用 CPU。")
        
    try:
        # 使用我们自定义的类来实例化
        qwen_embeddings = QwenInstructionalEmbeddings(
            # a. 传入自定义指令
            query_instruction=settings.EMBEDDING_INSTRUCTION_FOR_RETRIEVAL,
            
            # b. 传入 HuggingFaceEmbeddings 的标准参数
            model_name=settings.EMBEDDING_MODEL_PATH,
            model_kwargs={'device': device},
            encode_kwargs={
                'normalize_embeddings': True, # 推荐进行归一化
            }
        )
        logger.success(f"Qwen 嵌入模型组件初始化成功，运行于设备: '{device}'")
        return qwen_embeddings
        
    except Exception as e:
        logger.error(f"初始化 Qwen 嵌入模型组件失败: {e}", exc_info=True)
        raise