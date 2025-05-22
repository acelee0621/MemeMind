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


