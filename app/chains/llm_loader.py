from functools import lru_cache

from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from app.core.config import settings

""" 下载新模型命令
    uv run huggingface-cli download Qwen/Qwen3-4B /
    --local-dir ./local_models/llm/Qwen3-4B
"""


@lru_cache(maxsize=1)
def get_qwen_llm() -> HuggingFacePipeline:
    logger.info(f"开始初始化 LLM 组件，模型路径: {settings.LLM_MODEL_PATH}...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            settings.LLM_MODEL_PATH, torch_dtype="auto", device_map="auto"
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            settings.LLM_MODEL_PATH, trust_remote_code=True
        )

        devices = {p.device for p in model.parameters()}
        logger.success(f"LLM 模型和分词器加载成功，运行于设备: {devices}")

        # # --- 精确的终止符配置 ---
        terminator_ids = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
        ]
        # 使用 set 和 filter 去除重复和 None 值
        terminator_ids = list(set(filter(None, terminator_ids)))

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.8,
            do_sample=True,
            eos_token_id=terminator_ids,
            generation_kwargs={"presence_penalty": 1.5},
        )
        logger.info(
            "transformers 的 text-generation pipeline 创建成功，并已配置正确的终止符。"
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        logger.success("HuggingFacePipeline 组件初始化成功，可用于LangChain。")
        return llm

    except Exception as e:
        logger.error(f"初始化 LLM 组件失败: {e}", exc_info=True)
        raise
