import asyncio
from typing import List
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from langchain_core.documents import Document
# 导入模型加载器和认证依赖
from app.chains.embedding_loader import get_bge_embeddings
from app.chains.reranker_loader import get_bge_reranker
from app.chains.llm_loader import get_qwen_llm
from app.core.auth import get_api_key

# 创建一个带有认证依赖的路由器
# 所有在此路由器下注册的接口，都必须通过 API 密钥认证
router = APIRouter(
    prefix="/dify", 
    tags=["Dify Integration Services"], 
    dependencies=[Depends(get_api_key)]
)

# --- 1. 嵌入服务 (Embedding Service) ---

class EmbeddingRequest(BaseModel):
    """Dify 发送的嵌入请求模型"""
    input: List[str]

class EmbeddingData(BaseModel):
    """单个嵌入向量的数据结构"""
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingResponse(BaseModel):
    """嵌入服务的响应模型"""
    object: str = "list"
    data: List[EmbeddingData]
    model: str = "bge-large-zh-v1.5" # 您可以自定义模型名称

@router.post("/embeddings", response_model=EmbeddingResponse, summary="Dify Embedding Service")
async def embed_texts_for_dify(request: EmbeddingRequest):
    """
    为 Dify 提供文本嵌入服务。
    注意：Dify 将文档和查询都发送到此端点。
    您的 BGEInstructionalEmbeddings 类仅对 aembed_query 添加指令。
    对于文档（aembed_documents），它不会添加指令，这符合预期。
    对于查询，这意味着它也不会被添加指令，性能可能略有下降，但能保证接口兼容性。
    """
    embedding_model = get_bge_embeddings()
    vectors = await embedding_model.aembed_documents(request.input)
    
    response_data = [
        EmbeddingData(embedding=vec, index=i) for i, vec in enumerate(vectors)
    ]
    
    return EmbeddingResponse(data=response_data)


# --- 2. 精排服务 (Reranking Service) ---

class RerankRequest(BaseModel):
    """Dify 发送的精排请求模型"""
    query: str
    documents: List[str]

class RerankedDocument(BaseModel):
    """单个重排后文档的数据结构"""
    document: str
    index: int
    relevance_score: float = Field(..., alias="score")

class RerankResponse(BaseModel):
    """精排服务的响应模型"""
    results: List[RerankedDocument]


@router.post("/rerank", response_model=RerankResponse, summary="Dify Rerank Service")
async def rerank_documents_for_dify(request: RerankRequest):
    """为 Dify 提供文档精排服务。"""
    reranker = get_bge_reranker()
    
    docs_to_rerank = [Document(page_content=text) for text in request.documents]
    
    # CrossEncoderReranker.compress_documents 是同步方法，需在线程中运行以避免阻塞
    reranked_docs = await asyncio.to_thread(
        reranker.compress_documents,
        documents=docs_to_rerank,
        query=request.query
    )
    
    response_results = [
        RerankedDocument(
            document=doc.page_content, 
            index=i, 
            score=doc.metadata.get("relevance_score", 0.0)
        )
        for i, doc in enumerate(reranked_docs)
    ]
    
    return RerankResponse(results=response_results)


# --- 3. 语言模型生成服务 (LLM Service) ---

class LLMRequest(BaseModel):
    """Dify 发送的 LLM 请求模型"""
    prompt: str
    stream: bool = False # Dify 通过此字段告知是否需要流式输出

# 非流式响应模型
class LLMResponse(BaseModel):
    text: str

@router.post("/llm/generate", summary="Dify LLM Service (Streaming & Non-Streaming)")
async def generate_answer_for_dify(request: LLMRequest):
    """
    为 Dify 提供大语言模型文本生成服务。
    支持流式 (Server-Sent Events) 和非流式两种模式。
    """
    llm = get_qwen_llm()
    
    if not request.stream:
        # 非流式调用
        result = await llm.ainvoke(request.prompt)
        return LLMResponse(text=result)
    else:
        # 流式调用
        async def event_generator():
            async for chunk in llm.astream(request.prompt):
                yield { "data": chunk }

        return EventSourceResponse(event_generator())

