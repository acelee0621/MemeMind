from datetime import datetime, timezone
from typing import Optional, List
import enum

from sqlalchemy import ForeignKey, Integer, String, DateTime, Text, JSON
from sqlalchemy import Enum as SQLAlchemyEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship, DeclarativeBase

# --- 基础类和混入 (Mixin) ---
class Base(DeclarativeBase):
    pass

class DateTimeMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

# --- 源文档和文本块模型 ---

class SourceDocument(Base, DateTimeMixin):
    __tablename__ = "source_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)    
    object_name: Mapped[str] = mapped_column(String(512), nullable=False, unique=True, index=True)
    bucket_name: Mapped[str] = mapped_column(String(100), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    size: Mapped[int] = mapped_column(Integer, nullable=False)    
    status: Mapped[str] = mapped_column(String(50), default="uploaded", index=True)
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    number_of_chunks: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    text_chunks: Mapped[List["TextChunk"]] = relationship(
        "TextChunk", back_populates="source_document", cascade="all, delete-orphan"
    )

class TextChunk(Base, DateTimeMixin):
    __tablename__ = "text_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_document_id: Mapped[int] = mapped_column(
        ForeignKey("source_documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_document: Mapped["SourceDocument"] = relationship(
        "SourceDocument", back_populates="text_chunks"
    )
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    sequence_in_document: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

# --- 全新设计的 Message 模型 ---

class MessageAuthor(str, enum.Enum):
    USER = "user"
    BOT = "bot"

class Message(Base, DateTimeMixin):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # 自引用的外键，用于连接回答和问题
    response_to_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("messages.id"), nullable=True, index=True
    )
    
    # 建立关系，一个'user'消息可以有多个'bot'回答（尽管通常只有一个）
    # 一个'bot'消息只对应一个'user'提问
    user_query: Mapped[Optional["Message"]] = relationship(
        "Message", remote_side=[id], back_populates="bot_responses"
    )
    bot_responses: Mapped[List["Message"]] = relationship(
        "Message", back_populates="user_query"
    )

    author: Mapped[MessageAuthor] = mapped_column(SQLAlchemyEnum(MessageAuthor), nullable=False)    
    
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # 存储用于生成答案的上下文信息 (仅对 bot 消息有意义)
    retrieved_chunk_ids: Mapped[Optional[List[int]]] = mapped_column(JSON, nullable=True)