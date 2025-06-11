from __future__ import annotations

from functools import lru_cache
from typing import List, Sequence

import torch
from loguru import logger
from langchain_core.documents import Document
from langchain_core.callbacks import Callbacks
from langchain_core.pydantic_v1 import Field
from langchain_core.documents.transformers import BaseDocumentTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

from app.core.config import settings


""" 下载新模型命令
    uv run huggingface-cli download Qwen/Qwen3-Reranker-0.6B /
    --local-dir ./local_models/reranker/Qwen3-Reranker-0.6B
"""

# --- 1. 自定义 LangChain 文档转换器以封装 Qwen Reranker ---

class QwenReranker(BaseDocumentTransformer):
    """
    一个自定义的 LangChain 文档转换器，用于封装 Qwen Reranker 的独特逻辑。
    它接收一个查询和一组文档，并根据模型计算的相关性分数对文档进行重新排序。
    """
    model: AutoModelForCausalLM = Field(exclude=True)
    tokenizer: AutoTokenizer = Field(exclude=True)
    device: torch.device = Field(exclude=True)
    
    # 预先计算好的、模型特定的 Token ID
    prefix_tokens: List[int] = Field(exclude=True)
    suffix_tokens: List[int] = Field(exclude=True)
    token_true_id: int = Field(exclude=True)
    token_false_id: int = Field(exclude=True)
    
    # Reranker 的指令和最大长度
    task_instruction: str
    max_length: int = 8192
    top_n: int = 5 # 默认返回重排后前 5 个结果

    class Config:
        arbitrary_types_allowed = True

    def transform_documents(
        self, documents: Sequence[Document], callbacks: Callbacks = None, **kwargs: any
    ) -> Sequence[Document]:
        """
        对文档进行重排序的核心方法。
        
        Args:
            documents: 从向量数据库初步检索到的文档列表。
            **kwargs: 必须包含一个 'query' 字符串。
        """
        query = kwargs.get("query")
        if query is None:
            raise ValueError("QwenReranker 调用时必须在参数中提供 'query'。")
        if not documents:
            return []

        logger.info(f"开始使用 Qwen Reranker 对 {len(documents)} 个文档进行重排，查询: '{query[:50]}...'")

        # --- a. 格式化输入 ---
        # 格式：<Instruct>: ...\n<Query>: ...\n<Document>: ...
        pairs = [
            f"<Instruct>: {self.task_instruction}\n<Query>: {query}\n<Document>: {doc.page_content}"
            for doc in documents
        ]

        # --- b. Tokenize 和 Pad ---
        # 这个过程完全复刻了官方文档的推荐做法
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
            )
            for i in range(len(inputs["input_ids"])):
                inputs["input_ids"][i] = self.prefix_tokens + inputs["input_ids"][i] + self.suffix_tokens
            
            inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # --- c. 计算分数 ---
            last_token_logits = self.model(**inputs).logits[:, -1, :]
            true_vector = last_token_logits[:, self.token_true_id]
            false_vector = last_token_logits[:, self.token_false_id]
            
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().cpu().tolist()

        # --- d. 关联分数并排序 ---
        for doc, score in zip(documents, scores):
            doc.metadata["relevance_score"] = score # 将分数存入每个文档的元数据中
        
        sorted_documents = sorted(documents, key=lambda d: d.metadata["relevance_score"], reverse=True)
        
        logger.success(f"重排完成，返回 Top {self.top_n} 个结果。")
        return sorted_documents[:self.top_n]


# --- 2. 创建并缓存 Reranker 实例的工厂函数 ---

@lru_cache(maxsize=1)
def get_qwen_reranker(top_n: int = 5) -> QwenReranker:
    """
    加载并缓存 Qwen Reranker 模型，返回一个配置好的自定义实例。
    """
    logger.info("开始初始化 Qwen Reranker 组件...")
    
    # --- 自动设备检测 ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("检测到 CUDA，Reranker 将使用 GPU。")
        model_kwargs = {"torch_dtype": torch.float16, "attn_implementation": "flash_attention_2"}
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("检测到 MPS (Apple Silicon)，Reranker 将使用 MPS。")
        model_kwargs = {"torch_dtype": torch.float16}
    else:
        device = torch.device("cpu")
        logger.info("未检测到 CUDA 或 MPS，Reranker 将使用 CPU。")
        model_kwargs = {}

    try:
        # --- 加载模型和 Tokenizer ---
        tokenizer = AutoTokenizer.from_pretrained(settings.RERANKER_MODEL_PATH, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(settings.RERANKER_MODEL_PATH, **model_kwargs).to(device).eval()

        # --- 预计算特殊的 Token ---
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        token_true_id = tokenizer.convert_tokens_to_ids("yes")
        token_false_id = tokenizer.convert_tokens_to_ids("no")

        # --- 实例化我们的自定义类 ---
        reranker = QwenReranker(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prefix_tokens=prefix_tokens,
            suffix_tokens=suffix_tokens,
            token_true_id=token_true_id,
            token_false_id=token_false_id,
            task_instruction=settings.RERANKER_INSTRUCTION,
            top_n=top_n
        )
        logger.success(f"Qwen Reranker 组件初始化成功，运行于设备: '{device}'")
        return reranker

    except Exception as e:
        logger.error(f"初始化 Qwen Reranker 组件失败: {e}", exc_info=True)
        raise