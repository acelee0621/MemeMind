from pathlib import Path
from loguru import logger
import torch
# Qwen Reranker 使用 AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.schemas.schemas import TextChunkResponse

# --- 1. 修改模型常量 ---
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
RERANKER_MODEL_PATH = "app/embeddings/Qwen3-Reranker-0.6B"
QWEN_RERANKER_MAX_LENGTH = 8192

""" 下载新模型命令
    uv run huggingface-cli download Qwen/Qwen3-Reranker-0.6B /
    --local-dir ./app/embeddings/Qwen3-Reranker-0.6B
"""

# --- 2. 增加 Qwen Reranker 所需的全局变量和提示词模板 ---
reranker_tokenizer = None
reranker_model_global = None
reranker_device = None

# Qwen Reranker 的特定提示词（Prompt）结构
# 这些是固定的，只需要定义一次
PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
    "<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

# 这些变量将在模型加载时被初始化
prefix_tokens = None
suffix_tokens = None
token_true_id = None
token_false_id = None


def _load_reranker_model():
    global reranker_tokenizer, reranker_model_global, reranker_device
    global prefix_tokens, suffix_tokens, token_true_id, token_false_id
    
    if reranker_tokenizer is None or reranker_model_global is None:
        logger.info(f"首次加载 Reranker 模型: {RERANKER_MODEL_NAME}...")
        try:
            model_path = Path(RERANKER_MODEL_PATH)
            if not model_path.exists():
                raise FileNotFoundError(f"模型路径不存在: {model_path.absolute()}")

            logger.info(f"从本地加载 Reranker 模型: {model_path}...")
            
            # --- 3. 修改 Tokenizer 和模型加载方式 ---
            reranker_tokenizer = AutoTokenizer.from_pretrained(
                RERANKER_MODEL_PATH, padding_side='left'
            )
            
            # 初始化提示词的 token
            # 这些只需要计算一次，所以放在加载函数里
            prefix_tokens = reranker_tokenizer.encode(PREFIX, add_special_tokens=False)
            suffix_tokens = reranker_tokenizer.encode(SUFFIX, add_special_tokens=False)
            token_true_id = reranker_tokenizer.convert_tokens_to_ids("yes")
            token_false_id = reranker_tokenizer.convert_tokens_to_ids("no")

            # 判断设备并加载模型
            if torch.cuda.is_available():
                reranker_device = torch.device("cuda")
                logger.info("检测到 CUDA，Reranker 模型将使用 GPU。")
                try:
                    reranker_model_global = AutoModelForCausalLM.from_pretrained(
                        RERANKER_MODEL_PATH,
                        torch_dtype=torch.float16,
                        attn_implementation="flash_attention_2"
                    ).to(reranker_device)
                    logger.info("已启用 Flash Attention 2 加速。")
                except Exception as e:
                    logger.warning(f"加载 Flash Attention 2 失败: {e}，将使用标准模式。")
                    reranker_model_global = AutoModelForCausalLM.from_pretrained(
                        RERANKER_MODEL_PATH, torch_dtype=torch.float16
                    ).to(reranker_device)
            elif torch.backends.mps.is_available():
                reranker_device = torch.device("mps")
                logger.info("检测到 MPS，Reranker 模型将使用 MPS。")
                reranker_model_global = AutoModelForCausalLM.from_pretrained(
                    RERANKER_MODEL_PATH, torch_dtype=torch.float16
                ).to(reranker_device)
            else:
                reranker_device = torch.device("cpu")
                logger.info("未检测到 CUDA 或 MPS，Reranker 模型将使用 CPU。")
                reranker_model_global = AutoModelForCausalLM.from_pretrained(
                    RERANKER_MODEL_PATH
                ).to(reranker_device)
            
            reranker_model_global.eval()
            logger.info(f"Reranker 模型 {RERANKER_MODEL_NAME} 加载完成并移至 {reranker_device}。")
        except Exception as e:
            logger.error(f"加载 Reranker 模型 {RERANKER_MODEL_NAME} 失败: {e}", exc_info=True)
            raise RuntimeError(f"无法加载 Reranker 模型: {RERANKER_MODEL_NAME}") from e


def rerank_documents(query: str, documents: list[TextChunkResponse], task_instruction: str = None) -> list[tuple[TextChunkResponse, float]]:
    """
    使用 Qwen3-Reranker-4B 模型对初步检索到的文档块列表进行重排序。
    """
    _load_reranker_model()
    
    if not query or not documents:
        return []

    # --- 4. 完全重写评分逻辑以适配 Qwen Reranker ---
    
    # 4.1 格式化输入
    # 如果没有提供任务指令，使用默认值
    if task_instruction is None:
        task_instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        
    pairs = [
        f"<Instruct>: {task_instruction}\n<Query>: {query}\n<Document>: {doc.chunk_text}"
        for doc in documents
    ]

    try:
        with torch.no_grad():
            # 4.2 自定义 Tokenization 过程
            # 先对核心文本进行 tokenize，不填充，不加特殊符号
            inputs = reranker_tokenizer(
                pairs, padding=False, truncation='longest_first',
                return_attention_mask=False,
                max_length=QWEN_RERANKER_MAX_LENGTH - len(prefix_tokens) - len(suffix_tokens)
            )
            
            # 手动为每个序列加上前后缀
            for i in range(len(inputs['input_ids'])):
                inputs['input_ids'][i] = prefix_tokens + inputs['input_ids'][i] + suffix_tokens
            
            # 对整个批次进行填充
            inputs = reranker_tokenizer.pad(inputs, padding=True, return_tensors="pt")
            inputs = {k: v.to(reranker_device) for k, v in inputs.items()}

            # 4.3 计算分数
            # 获取模型在最后一个 token 位置上的 logits
            last_token_logits = reranker_model_global(**inputs).logits[:, -1, :]
            
            # 提取 "yes" 和 "no" 两个词的 logits
            true_vector = last_token_logits[:, token_true_id]
            false_vector = last_token_logits[:, token_false_id]
            
            # 计算 LogSoftmax 并转换为概率
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            # 取出 "yes" 的概率作为最终得分
            scores = batch_scores[:, 1].exp().cpu().tolist()
            
        # 4.4 关联分数并排序
        scored_documents = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        logger.info(f"对 {len(documents)} 个文档块进行了 Rerank，查询: '{query[:50]}...'")
        return scored_documents
        
    except Exception as e:
        logger.error(f"使用 Reranker 对文档块进行重排序时发生错误: {e}", exc_info=True)
        raise