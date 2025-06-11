from functools import lru_cache

from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import HuggingFaceLLM

from app.core.config import settings


""" 下载新模型命令
    uv run huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct /
    --local-dir ./local_models/llm/Qwen2.5-1.5B-Instruct
"""


@lru_cache(maxsize=1)
def get_qwen_llm() -> HuggingFaceLLM:
    """
    加载并缓存 Qwen2.5 LLM，返回一个配置好的 LangChain LLM 实例。
    """
    logger.info("开始初始化 Qwen LLM 组件...")

    try:
        # --- 1. 加载模型和分词器 ---
        # 这里的参数完全遵循 Qwen 官方文档的推荐
        model = AutoModelForCausalLM.from_pretrained(
            settings.LLM_MODEL_PATH,
            torch_dtype="auto", # 自动选择最佳精度 (如 bfloat16)
            device_map="auto"   # 使用 accelerate 库自动分配设备 (CPU/GPU)
        )
        tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL_PATH)
        
        device = next(model.parameters()).device
        logger.success(f"LLM 模型和分词器加载成功，运行于设备: '{device}'")

        # --- 2. 使用 HuggingFaceLLM 封装 ---
        # 这是将本地模型接入 LangChain 生态系统的关键一步
        llm = HuggingFaceLLM(
            model=model,
            tokenizer=tokenizer,
            task="text-generation", # 指定任务类型
            model_kwargs={
                # 在这里可以定义默认的生成参数
                "max_new_tokens": 1024,
                "temperature": 0.7,
                "top_p": 0.9,
            },
        )
        logger.success("HuggingFaceLLM 组件初始化成功。")
        return llm

    except Exception as e:
        logger.error(f"初始化 Qwen LLM 组件失败: {e}", exc_info=True)
        raise