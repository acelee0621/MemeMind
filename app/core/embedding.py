from pathlib import Path

from transformers import AutoTokenizer, AutoModel
import torch  # PyTorch 是 Transformers 的依赖之一

from app.core.logging import get_logger

# 在线模型
EMBEDDING_MODEL_NAME = "maidalun1020/bce-embedding-base_v1"
# 本地模型路径
""" 下载模型命令
    uv run huggingface-cli download maidalun1020/bce-embedding-base_v1 /
    --local-dir ./app/embeddings/bce-embedding-base_v1
"""
EMBEDDING_MODEL_PATH = "app/embeddings/bce-embedding-base_v1"

tokenizer = None
embedding_model_global = None  # 使用 embedding_model_global 以区分局部变量
device = None

logger = get_logger(__name__)


def _load_embedding_model():
    """延迟加载 Embedding 模型和 Tokenizer，并移至可用设备。"""
    global tokenizer, embedding_model_global, device
    if tokenizer is None or embedding_model_global is None:
        logger.info(f"首次加载 Embedding 模型: {EMBEDDING_MODEL_NAME}...")
        try:
            # 如果在线下载模型，用下面这两条
            # tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
            # embedding_model_global = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)

            # 检查路径是否存在
            model_path = Path(EMBEDDING_MODEL_PATH)
            if not model_path.exists():
                raise FileNotFoundError(f"模型路径不存在: {model_path.absolute()}")

            logger.info(f"从本地加载 Embedding 模型: {model_path}...")

            # 从本地路径加载
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            embedding_model_global = AutoModel.from_pretrained(model_path)

            # 判断是否有可用 GPU，优先使用 GPU
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("检测到 CUDA，Embedding 模型将使用 GPU。")
            elif torch.backends.mps.is_available():  # 适用于 Apple Silicon (M1/M2/M3)
                device = torch.device("mps")
                logger.info(
                    "检测到 MPS (Apple Silicon GPU)，Embedding 模型将使用 MPS。"
                )
            else:
                device = torch.device("cpu")
                logger.info("未检测到 CUDA 或 MPS，Embedding 模型将使用 CPU。")

            embedding_model_global.to(device)
            embedding_model_global.eval()  # 设置为评估模式，关闭 dropout 等训练特有层
            logger.info(
                f"Embedding 模型 {EMBEDDING_MODEL_NAME} 加载完成并移至 {device}。"
            )
        except Exception as e:
            logger.error(
                f"加载 Embedding 模型 {EMBEDDING_MODEL_NAME} 失败: {e}", exc_info=True
            )
            # 如果模型加载失败，后续任务可能无法执行，可以考虑抛出致命错误或设置一个标志
            raise RuntimeError(
                f"无法加载 Embedding 模型: {EMBEDDING_MODEL_NAME}"
            ) from e


# 在 Worker 启动时或首次调用任务时，可以预先调用 _load_embedding_model()
# 或者在 get_embeddings 函数中按需加载。为简单起见，我们可以在 get_embeddings 中检查和加载。


def get_embeddings(texts: list[str], instruction: str = "") -> list[list[float]]:
    """
    使用 BCE 模型为文本列表生成向量嵌入。
    BCE 模型通常需要在输入前添加指令。
    """
    _load_embedding_model()  # 确保模型已加载

    if not texts:
        return []

    # 为每个句子添加指令 (BCE 模型可以为空指令)
    # instruction 可以根据你的具体任务调整，例如用于检索、分类等
    # instruction: str = "为这个句子生成表示以用于检索相关文章："
    # 如果你的文本主要是英文，可以使用英文指令
    # instruction_en = "Encode this sentence for retrieving relevant articles:"
    instructed_texts = [instruction + s for s in texts]

    try:
        # Tokenize sentences
        inputs = tokenizer(
            instructed_texts,
            padding=True,  # 填充到批次中最长句子的长度
            truncation=True,  # 截断超过模型最大长度的句子
            return_tensors="pt",  # 返回 PyTorch 张量
            max_length=tokenizer.model_max_length,  # 使用模型自身的max_length，bce-base 通常是512
        )
        inputs = {
            k: v.to(device) for k, v in inputs.items()
        }  # 将输入数据移至与模型相同的设备

        # Get embeddings
        with torch.no_grad():  # 在推理时不需要计算梯度
            outputs = embedding_model_global(**inputs, return_dict=True)
            # 通常取 [CLS] token的输出或对最后一层hidden states进行平均池化
            # 对于 BCE 模型，官方推荐使用 last_hidden_state 的第一个 token ([CLS]) 的表示
            embeddings = outputs.last_hidden_state[:, 0]
            # L2 归一化
            normalized_embeddings = torch.nn.functional.normalize(
                embeddings, p=2, dim=1
            )

        return normalized_embeddings.cpu().tolist()  # 移回 CPU 并转为 Python 列表
    except Exception as e:
        logger.error(f"生成文本嵌入时发生错误: {e}", exc_info=True)
        raise  # 重新抛出异常，让 Celery 任务捕获
