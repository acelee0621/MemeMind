from functools import lru_cache

from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

from app.core.config import settings


""" 下载新模型命令
    uv run huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct /
    --local-dir ./local_models/llm/Qwen2.5-1.5B-Instruct
"""


@lru_cache(maxsize=1)
def get_qwen_llm() -> HuggingFacePipeline:
    """
    加载并缓存 Qwen2.5 LLM，返回一个配置好的 LangChain Pipeline 实例。
    """
    logger.info("开始初始化 Qwen LLM 组件...")

    try:
        # --- 1. 加载模型和分词器 ---
        model = AutoModelForCausalLM.from_pretrained(
            settings.LLM_MODEL_PATH, torch_dtype="auto", device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL_PATH)

        device = next(model.parameters()).device
        logger.success(f"LLM 模型和分词器加载成功，运行于设备: '{device}'")

        # --- 2. 创建一个 transformers pipeline 对象 ---
        # 这是将模型和分词器组合起来的标准方式
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
        )
        logger.info("transformers 的 text-generation pipeline 创建成功。")

        # --- 3. 使用 HuggingFacePipeline 封装 ---
        llm = HuggingFacePipeline(pipeline=pipe)

        logger.success("HuggingFacePipeline 组件初始化成功。")
        return llm

    except Exception as e:
        logger.error(f"初始化 Qwen LLM 组件失败: {e}", exc_info=True)
        raise
