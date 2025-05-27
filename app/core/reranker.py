from pathlib import Path

from loguru import logger
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification # Reranker 模型通常用这个

from app.schemas.schemas import TextChunkResponse # 用于类型提示



# 模型名称或本地路径
RERANKER_MODEL_NAME = "maidalun1020/bce-reranker-base_v1"
RERANKER_MODEL_PATH = "app/embeddings/bce-reranker-base_v1"

reranker_tokenizer = None
reranker_model_global = None
reranker_device = None

def _load_reranker_model():
    global reranker_tokenizer, reranker_model_global, reranker_device
    if reranker_tokenizer is None or reranker_model_global is None:
        logger.info(f"首次加载 Reranker 模型: {RERANKER_MODEL_NAME}...")
        try:
            # 检查路径是否存在
            model_path = Path(RERANKER_MODEL_PATH)
            if not model_path.exists():
                raise FileNotFoundError(f"模型路径不存在: {model_path.absolute()}")

            logger.info(f"从本地加载 Embedding 模型: {model_path}...")
            
            reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_PATH)
            reranker_model_global = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_PATH)

            if torch.cuda.is_available():
                reranker_device = torch.device("cuda")
                logger.info("检测到 CUDA，Reranker 模型将使用 GPU。")
            elif torch.backends.mps.is_available():
                reranker_device = torch.device("mps")
                logger.info("检测到 MPS，Reranker 模型将使用 MPS。")
            else:
                reranker_device = torch.device("cpu")
                logger.info("未检测到 CUDA 或 MPS，Reranker 模型将使用 CPU。")
            
            reranker_model_global.to(reranker_device)
            reranker_model_global.eval()
            logger.info(f"Reranker 模型 {RERANKER_MODEL_PATH} 加载完成并移至 {reranker_device}。")
        except Exception as e:
            logger.error(f"加载 Reranker 模型 {RERANKER_MODEL_PATH} 失败: {e}", exc_info=True)
            raise RuntimeError(f"无法加载 Reranker 模型: {RERANKER_MODEL_PATH}") from e

def rerank_documents(query: str, documents: list[TextChunkResponse]) -> list[tuple[TextChunkResponse, float]]:
    """
    使用 Reranker 模型对初步检索到的文档块列表进行重排序。
    返回 (文档块, 相关性得分) 的元组列表，按得分降序排列。
    """
    _load_reranker_model() # 确保模型已加载
    
    if not query or not documents:
        return []

    # 构造 Reranker 输入：查询和每个文档块文本的配对
    pairs: list[list[str]] = []
    for doc_response in documents:
        pairs.append([query, doc_response.chunk_text])

    if not pairs:
        return []
        
    try:
        with torch.no_grad():
            inputs = reranker_tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=512 # 查阅 bce-reranker-base_v1 的最大长度限制
            ).to(reranker_device)
            
            # Reranker 模型通常输出 logits，对于 BCE Reranker，可能是一个表示相关性的分数
            # 你需要查阅 bce-reranker-base_v1 的具体用法来获取正确的得分
            # 通常，得分越高越好。如果模型输出多个类别的 logits，你可能需要取特定类别的 logit 或进行 sigmoid/softmax.
            # 假设模型输出一个直接的分数或可以转换为分数的 logits
            scores_tensor = reranker_model_global(**inputs).logits.squeeze(-1) # 假设 squeeze 后是每个pair的分数
            # 如果需要，可以对分数进行 sigmoid 转换为概率 scores = torch.sigmoid(scores_tensor)
            scores = scores_tensor.cpu().tolist()
        
        # 将分数与原始文档块关联
        scored_documents = []
        for i, doc_response in enumerate(documents):
            scored_documents.append((doc_response, scores[i]))
            
        # 按相关性得分降序排列
        scored_documents.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"对 {len(documents)} 个文档块进行了 Rerank，查询: '{query[:50]}...'")
        return scored_documents
    except Exception as e:
        logger.error(f"使用 Reranker 对文档块进行重排序时发生错误: {e}", exc_info=True)
        raise # 重新抛出异常