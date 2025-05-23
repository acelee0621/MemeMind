from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime


# 配置基类，启用 ORM 模式
class BaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class UserRead(BaseSchema):
    id: int
    username: str = Field(..., min_length=3, max_length=100)
    full_name: str | None = Field(None, max_length=100)
    created_at: datetime
    updated_at: datetime


class UserResponse(UserRead):
    pass


class UserCreate(BaseSchema):
    username: str = Field(..., min_length=3, max_length=100)
    full_name: str | None = Field(None, max_length=100)
    email: str | None = Field(None, max_length=320)


class UserUpdate(BaseSchema):
    username: str = Field(..., min_length=3, max_length=100)
    full_name: str | None = Field(None, max_length=100)


class SourceDocumentBase(BaseSchema):    
    object_name: str = Field(..., max_length=512, description="MinIO对象存储路径")
    bucket_name: str = Field(..., max_length=100, description="MinIO存储桶名称")
    original_filename: str = Field(..., max_length=255, description="原始文件名")
    content_type: str = Field(..., max_length=100, description="文件MIME类型")
    size: int = Field(..., description="文件大小(字节)")
    


# 创建附件时的请求模型
class SourceDocumentCreate(SourceDocumentBase):
    pass


class SourceDocumentUpdate(BaseSchema):
    status: str | None = Field(..., description="文件状态")
    processed_at: datetime | None = Field(..., description="处理时间")
    error_message: str | None = Field(..., description="错误信息")
    number_of_chunks: int | None = Field(..., description="分块数量")
    
    
# 附件响应模型
class SourceDocumentResponse(SourceDocumentBase):
    id: int = Field(..., description="附件ID")
    owner_id: int | None = Field(..., description="所属用户ID")
    status: str = Field(..., description="文件状态")
    processed_at: datetime | None = Field(..., description="处理时间")
    error_message: str | None = Field(..., description="错误信息")
    number_of_chunks: int | None = Field(..., description="分块数量")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")


class PresignedUrlResponse(BaseModel):
    url: str = Field(..., description="Pre-signed URL for the attachment")
    expires_at: datetime = Field(
        ..., description="Expiration time of the pre-signed URL"
    )
    filename: str = Field(..., description="Original filename of the attachment")
    content_type: str = Field(..., description="MIME type of the attachment")
    size: int = Field(..., description="Size of the attachment in bytes")
    attachment_id: int = Field(..., description="ID of the attachment")
    
    
# --- TextChunk Pydantic Models ---

class TextChunkBase(BaseSchema):
    """
    TextChunk 基础模型，包含通用字段。
    """
    chunk_text: str = Field(..., description="文本块的实际内容")
    sequence_in_document: int = Field(..., ge=0, description="文本块在原文档中的顺序编号，从0开始")
    metadata_json: dict[str, Any] | None = Field(None, description="与文本块相关的其他元数据，例如页码、章节等")


class TextChunkCreate(TextChunkBase):
    """
    用于创建新 TextChunk 记录的模型。
    在 Celery 任务中，当你从文档中分割出文本块后，会用这个模型 (或类似的数据结构)
    来准备将要存入数据库的数据。
    """
    source_document_id: int = Field(..., description="关联的源文档ID")
    # chunk_text, sequence_in_document, metadata_json 继承自 TextChunkBase


class TextChunkUpdate(BaseSchema):
    """
    用于更新现有 TextChunk 记录的模型 (可选)。
    RAG 流程中通常不直接更新已生成的块，更多是删除旧块并创建新块。
    但如果需要，可以定义此模型。
    """
    chunk_text: str | None = Field(None, description="更新后的文本块内容")
    metadata_json: dict[str, Any] | None = Field(None, description="更新后的元数据")
    


class TextChunkResponse(TextChunkBase):
    """
    用于 API 响应或内部数据表示的 TextChunk 模型。
    包含从数据库读取的完整信息，包括ID和时间戳。
    """
    id: int = Field(..., description="文本块的唯一ID")
    source_document_id: int = Field(..., description="关联的源文档ID")
    created_at: datetime = Field(..., description="记录创建时间")
    updated_at: datetime = Field(..., description="记录最后更新时间")
    # 如果需要，可以在这里添加关联的 SourceDocument 的摘要信息 (需要嵌套 Pydantic 模型)
    # source_document: Optional[SourceDocumentInfo] = None # 例如

