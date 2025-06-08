from pathlib import Path
import torch
import torch.nn.functional as F
from torch import Tensor
from loguru import logger
from transformers import AutoTokenizer, AutoModel

# --- 1. 修改模型名称和路径 ---
# 新模型的 Hugging Face 名称
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
# 建议为新模型创建一个新的本地路径
EMBEDDING_MODEL_PATH = "app/embeddings/Qwen3-Embedding-0.6B"
# 新模型的最大长度
QWEN_MAX_LENGTH = 8192

""" 下载新模型命令
    uv run huggingface-cli download Qwen/Qwen3-Embedding-0.6B /
    --local-dir ./app/embeddings/Qwen3-Embedding-0.6B
"""

tokenizer = None
embedding_model_global = None
device = None


# --- 2. 添加新模型所需的 last_token_pool 函数 ---
# 这个函数来自 Qwen 官方示例，用于从模型的输出中正确地提取句向量
def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    根据 attention_mask 从 last_hidden_states 中获取最后一个有效 token 的向量。
    """
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def _load_embedding_model():
    """延迟加载 Embedding 模型和 Tokenizer，并移至可用设备。"""
    global tokenizer, embedding_model_global, device
    if tokenizer is None or embedding_model_global is None:
        logger.info(f"首次加载 Embedding 模型: {EMBEDDING_MODEL_NAME}...")
        try:
            model_path = Path(EMBEDDING_MODEL_PATH)
            if not model_path.exists():
                raise FileNotFoundError(f"模型路径不存在: {model_path.absolute()}")

            logger.info(f"从本地加载 Embedding 模型: {model_path}...")

            # --- 3. 修改 Tokenizer 和模型加载方式 ---
            # Qwen 模型推荐使用 'left' 作为填充侧，这对于 last_token_pool至关重要
            tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

            # 判断是否有可用 GPU
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("检测到 CUDA，Embedding 模型将使用 GPU。")
                # 为获得最佳性能，Qwen 推荐在支持的 GPU 上使用 flash_attention_2 和 bfloat16
                try:
                    embedding_model_global = AutoModel.from_pretrained(
                        model_path,
                        attn_implementation="flash_attention_2",
                        torch_dtype=torch.bfloat16,
                    ).to(device)
                    logger.info("已启用 Flash Attention 2 和 bfloat16 加速。")
                except Exception as e:
                    logger.warning(f"加载 Flash Attention 2 失败: {e}，将使用标准模式。")
                    embedding_model_global = AutoModel.from_pretrained(
                        model_path, torch_dtype=torch.bfloat16
                    ).to(device)

            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("检测到 MPS (Apple Silicon GPU)，Embedding 模型将使用 MPS。")
                # Apple Silicon 不支持 Flash Attention，但可以使用 bfloat16
                embedding_model_global = AutoModel.from_pretrained(
                    model_path, torch_dtype=torch.bfloat16
                ).to(device)
            else:
                device = torch.device("cpu")
                logger.info("未检测到 CUDA 或 MPS，Embedding 模型将使用 CPU。")
                embedding_model_global = AutoModel.from_pretrained(model_path).to(
                    device
                )

            embedding_model_global.eval()
            logger.info(
                f"Embedding 模型 {EMBEDDING_MODEL_NAME} 加载完成并移至 {device}。"
            )
        except Exception as e:
            logger.error(
                f"加载 Embedding 模型 {EMBEDDING_MODEL_NAME} 失败: {e}", exc_info=True
            )
            raise RuntimeError(
                f"无法加载 Embedding 模型: {EMBEDDING_MODEL_NAME}"
            ) from e


def get_embeddings(
    texts: list[str], task_description: str, is_query: bool
) -> list[list[float]]:
    """
    使用 Qwen3-Embedding 模型为文本列表生成向量嵌入。

    Args:
        texts (list[str]): 需要编码的文本列表。
        task_description (str): 描述任务的指令，例如 "为这个句子生成表示以用于检索相关文章"。
        is_query (bool): 指示输入是查询（True）还是文档（False）。
                         查询文本前会添加指令，文档则不会。
    """
    _load_embedding_model()

    if not texts:
        return []

    # --- 4. 修改输入文本的格式化方式 ---
    # 根据 is_query 参数决定是否添加指令
    if is_query:
        # Qwen 的指令格式为 'Instruct: {task_description}\nQuery: {text}'
        instructed_texts = [
            f"Instruct: {task_description}\nQuery: {s}" for s in texts
        ]
    else:
        # 文档（documents）不需要指令
        instructed_texts = texts

    try:
        # Tokenize
        inputs = tokenizer(
            instructed_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=QWEN_MAX_LENGTH,  # 使用新模型的最大长度
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = embedding_model_global(**inputs, return_dict=True)
            # --- 5. 使用 last_token_pool 提取向量 ---
            embeddings = last_token_pool(
                outputs.last_hidden_state, inputs["attention_mask"]
            )
            # L2 归一化
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

        return normalized_embeddings.cpu().tolist()
    except Exception as e:
        logger.error(f"生成文本嵌入时发生错误: {e}", exc_info=True)
        raise