from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime
import enum

# ===================================================================
# 通用配置和基础类
# ===================================================================

class BaseSchema(BaseModel):
    # 配置Pydantic模型以兼容ORM对象
    model_config = ConfigDict(from_attributes=True)

# ===================================================================
# SourceDocument 相关模型
# ===================================================================

class SourceDocumentBase(BaseSchema):
    object_name: str = Field(..., max_length=512, description="MinIO对象存储路径")
    bucket_name: str = Field(..., max_length=100, description="MinIO存储桶名称")
    original_filename: str = Field(..., max_length=255, description="原始文件名")
    content_type: str = Field(..., max_length=100, description="文件MIME类型")
    size: int = Field(..., description="文件大小(字节)")

class SourceDocumentCreate(SourceDocumentBase):
    pass

class SourceDocumentUpdate(BaseSchema):
    status: str | None = Field(None, description="文件状态")
    processed_at: datetime | None = Field(None, description="处理时间")
    error_message: str | None = Field(None, description="错误信息")
    number_of_chunks: int | None = Field(None, description="分块数量")

class SourceDocumentResponse(SourceDocumentBase):
    id: int = Field(..., description="附件ID")
    status: str = Field(..., description="文件状态")
    processed_at: datetime | None = Field(None, description="处理时间")
    error_message: str | None = Field(None, description="错误信息")
    number_of_chunks: int | None = Field(None, description="分块数量")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")

class PresignedUrlResponse(BaseModel):
    url: str
    expires_at: datetime
    filename: str
    content_type: str
    size: int
    attachment_id: int

# ===================================================================
# TextChunk 相关模型 
# ===================================================================

class TextChunkBase(BaseSchema):
    chunk_text: str = Field(..., description="文本块的实际内容")
    sequence_in_document: int = Field(..., ge=0, description="文本块在原文档中的顺序编号")
    metadata_json: dict[str, Any] | None = Field(None, description="与文本块相关的元数据")

class TextChunkCreate(TextChunkBase):
    source_document_id: int = Field(..., description="关联的源文档ID")

class TextChunkUpdate(BaseSchema):
    chunk_text: str | None = Field(None, description="更新后的文本块内容")
    metadata_json: dict[str, Any] | None = Field(None, description="更新后的元数据")

class TextChunkResponse(TextChunkBase):
    id: int = Field(..., description="文本块的唯一ID")
    source_document_id: int = Field(..., description="关联的源文档ID")
    created_at: datetime = Field(..., description="记录创建时间")
    updated_at: datetime = Field(..., description="记录最后更新时间")

# ===================================================================
#     Message 相关模型 (全新补充)  
# ===================================================================

# 与 SQLAlchemy 模型中的 Enum 保持一致，用于数据验证
class MessageAuthor(str, enum.Enum):
    USER = "user"
    BOT = "bot"

class MessageBase(BaseSchema):
    """
    Message 基础模型，包含通用字段。
    """
    author: MessageAuthor = Field(..., description="消息作者 (user 或 bot)")
    content: str = Field(..., description="消息内容 (用户的问题或模型的回答)")

class MessageCreate(MessageBase):
    """
    用于在数据库中创建新 Message 记录的模型。
    """
    response_to_id: int | None = Field(None, description="当作者是bot时，此字段指向对应用户消息的ID")
    retrieved_chunk_ids: list[int] | None = Field(None, description="当作者是bot时，此字段存储用于生成回答的文本块ID列表")

class MessageUpdate(BaseSchema):
    """
    用于更新现有 Message 记录的模型 (非常规操作)。
    通常聊天记录不应被修改。
    """
    content: str | None = Field(None, description="更新后的消息内容")
    
class MessageResponse(MessageBase):
    """
    用于 API 响应或内部数据表示的 Message 模型。
    """
    id: int = Field(..., description="消息的唯一ID")
    response_to_id: int | None = Field(None, description="如果此消息是回答，则为对应问题的ID")
    retrieved_chunk_ids: list[int] | None = Field(None, description="用于生成此回答的上下文文本块ID")
    created_at: datetime = Field(..., description="消息创建时间")