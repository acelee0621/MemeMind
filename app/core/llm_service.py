from pathlib import Path
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger

# --- 全局变量，用于存储加载后的模型和分词器 ---
llm_model: Optional[AutoModelForCausalLM] = None
llm_tokenizer: Optional[AutoTokenizer] = None

# --- 模型配置 ---
LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LLM_MODEL_PATH = "app/llm_models/Qwen2.5-1.5B-Instruct"


def _load_llm_model():
    """
    使用 transformers 库加载 Qwen2.5 模型。
    这个函数将在 FastAPI 启动时被调用一次。
    """
    global llm_model, llm_tokenizer

    if llm_model is None or llm_tokenizer is None:
        logger.info(f"首次加载 LLM 模型: {LLM_MODEL_NAME} ...")

        try:
            model_path = Path(LLM_MODEL_PATH)
            if not model_path.exists():
                raise FileNotFoundError(f"模型路径不存在: {model_path.absolute()}")
            # 2. 根据官方文档，使用 torch_dtype="auto" 进行更智能的类型选择
            llm_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",  # accelerate 会自动处理设备映射
            )

            llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)

            # 将模型设置为评估模式
            llm_model.eval()

            device = next(llm_model.parameters()).device
            logger.info(
                f"LLM 模型 {LLM_MODEL_NAME} 加载成功，运行于设备: {device}"
            )

        except Exception as e:
            logger.error(
                f"加载 LLM 模型 {LLM_MODEL_NAME} 失败: {e}", exc_info=True
            )
            raise RuntimeError(f"无法加载 LLM 模型: {LLM_MODEL_NAME}") from e


def generate_text_from_llm(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",  # 可选的系统提示词
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    使用加载的 Qwen2.5 模型根据给定的提示生成文本。
    """
    _load_llm_model()
    
    if llm_model is None or llm_tokenizer is None:
        raise RuntimeError("LLM 模型实例未成功加载，无法生成文本。")

    logger.debug(f"向 LLM 发送的 Prompt (部分内容):\n{prompt[:100]}")

    try:
        # 构建符合 Qwen2.5-Instruct 模型的聊天模板
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 将格式化后的文本 tokenize
        model_inputs = llm_tokenizer([text], return_tensors="pt").to(llm_model.device)

        # 使用模型生成文本
        with torch.no_grad():
            # 3. 根据官方文档，使用 **model_inputs 解包方式传递参数
            generated_ids = llm_model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )

        # 4. 根据官方文档，使用更健壮的方式来分离生成的部分
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # 解码生成的 token
        response = llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]

        logger.info(f"LLM 生成的文本 (前100字符): {response[:100]}...")
        return response

    except Exception as e:
        logger.error(f"LLM 生成文本时发生错误: {e}", exc_info=True)
        raise RuntimeError(f"LLM 生成文本失败: {e}") from e
