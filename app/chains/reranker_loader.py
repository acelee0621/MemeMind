from functools import lru_cache
from typing import List, Dict, Any
import torch
from loguru import logger
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.core.config import settings


@lru_cache(maxsize=1)
def _get_reranker_model_and_tokenizer() -> Dict[str, Any]:
    logger.info("开始初始化 Qwen Reranker 模型及相关组件...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"torch_dtype": torch.bfloat16} if device == "cuda" else {}

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            settings.RERANKER_MODEL_PATH, padding_side="left"
        )
        model = (
            AutoModelForCausalLM.from_pretrained(
                settings.RERANKER_MODEL_PATH, **model_kwargs
            )
            .to(device)
            .eval()
        )
        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return {
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
            "prefix_tokens": tokenizer.encode(prefix, add_special_tokens=False),
            "suffix_tokens": tokenizer.encode(suffix, add_special_tokens=False),
            "token_true_id": tokenizer.convert_tokens_to_ids("yes"),
            "token_false_id": tokenizer.convert_tokens_to_ids("no"),
        }
    except Exception as e:
        logger.error(f"初始化 Qwen Reranker 组件失败: {e}", exc_info=True)
        raise


def rerank_qwen_documents(inputs: dict) -> List[Document]:
    query: str = inputs["query"]
    documents: List[Document] = inputs["documents"]
    top_n = inputs.get("top_n", settings.FINAL_CONTEXT_TOP_N)
    if not query or not documents:
        return []

    components = _get_reranker_model_and_tokenizer()
    model, tokenizer = components["model"], components["tokenizer"]
    pairs = [
        f"<Instruct>: {settings.RERANKER_INSTRUCTION}\n<Query>: {query}\n<Document>: {doc.page_content}"
        for doc in documents
    ]

    with torch.no_grad():
        tokenized_inputs = tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=8192
            - len(components["prefix_tokens"])
            - len(components["suffix_tokens"]),
        )
        for i in range(len(tokenized_inputs["input_ids"])):
            tokenized_inputs["input_ids"][i] = (
                components["prefix_tokens"]
                + tokenized_inputs["input_ids"][i]
                + components["suffix_tokens"]
            )
        padded_inputs = tokenizer.pad(
            tokenized_inputs, padding=True, return_tensors="pt"
        )
        padded_inputs = {
            k: v.to(components["device"]) for k, v in padded_inputs.items()
        }
        last_token_logits = model(**padded_inputs).logits[:, -1, :]
        true_vector = last_token_logits[:, components["token_true_id"]]
        false_vector = last_token_logits[:, components["token_false_id"]]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        scores = (
            torch.nn.functional.log_softmax(batch_scores, dim=1)[:, 1]
            .exp()
            .cpu()
            .tolist()
        )

    for doc, score in zip(documents, scores):
        doc.metadata["relevance_score"] = score

    sorted_documents = sorted(
        documents, key=lambda d: d.metadata["relevance_score"], reverse=True
    )

    final_docs = sorted_documents[:top_n]

    logger.success(f"重排完成，返回 Top {len(final_docs)} 个结果。")
    return final_docs
