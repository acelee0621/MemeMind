from datetime import datetime, timezone
from typing import Optional
import enum

from sqlalchemy import ForeignKey, Integer, String, DateTime, Boolean, Text, JSON
from sqlalchemy import Enum as SQLAlchemyEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship, DeclarativeBase


# 基类
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


# 用户模型 (不依赖 fastapi-users，但为未来集成做准备)
class User(DateTimeMixin, Base):
    __tablename__ = "user_account"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 你已有的字段
    username: Mapped[str] = mapped_column(
        String(100), index=True, unique=True, nullable=False
    )
    full_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    # 为 fastapi-users 预留的关键字段
    email: Mapped[Optional[str]] = mapped_column(
        String(320), unique=True, index=True, nullable=True
    )
    hashed_password: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # 关系映射
    source_documents: Mapped[list["SourceDocument"]] = relationship(
        "SourceDocument", back_populates="owner"
    )
    conversations: Mapped[list["Conversation"]] = relationship(
        "Conversation", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<User(id={self.id}, username={self.username})>"


# 源文档模型
class SourceDocument(Base, DateTimeMixin):
    __tablename__ = "source_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("user_account.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    object_name: Mapped[str] = mapped_column(
        String(512), nullable=False, unique=True, index=True
    )
    bucket_name: Mapped[str] = mapped_column(String(100), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    size: Mapped[int] = mapped_column(Integer, nullable=False)

    # RAG 相关字段
    status: Mapped[str] = mapped_column(
        String(50), default="uploaded", index=True
    )  # 例如: uploaded, processing, ready, error
    processed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error_message: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    number_of_chunks: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    owner: Mapped[Optional["User"]] = relationship(
        "User", back_populates="source_documents"
    )
    text_chunks: Mapped[list["TextChunk"]] = relationship(
        "TextChunk", back_populates="source_document", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<SourceDocument(id={self.id}, filename='{self.original_filename}', status='{self.status}')>"


class TextChunk(Base, DateTimeMixin):
    __tablename__ = "text_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # 关联到源文档
    source_document_id: Mapped[int] = mapped_column(
        ForeignKey("source_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_document: Mapped["SourceDocument"] = relationship(
        "SourceDocument", back_populates="text_chunks"
    )

    # 文本块内容
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    sequence_in_document: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    # （重要）用于关联向量数据库中的向量
    # 向量数据库通常会返回一个或多个ID，或者你可以用这个 TextChunk 的 id 作为在向量库中存储的元数据。
    # 向量数据库是独立的，通常会用这个 TextChunk.id 作为元数据与向量一起存储。

    # 其他元数据 (可选)
    metadata_json: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True
    )  # 例如：{'page': 5, 'section': 'Introduction'}

    def __repr__(self):
        return f"<TextChunk(id={self.id}, source_document_id={self.source_document_id}, len_text={len(self.chunk_text)})>"


class Conversation(Base, DateTimeMixin):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("user_account.id", ondelete="CASCADE"),
        index=True,
        nullable=True,
    )
    user: Mapped[Optional["User"]] = relationship(
        "User", back_populates="conversations"
    )
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # 反向关系
    messages: Mapped[list["Message"]] = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Conversation(id={self.id}, user_id={self.user_id}, title='{self.title}')>"


class MessageAuthor(str, enum.Enum):
    USER = "user"
    BOT = "bot"


class Message(Base, DateTimeMixin):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    conversation: Mapped["Conversation"] = relationship(
        "Conversation", back_populates="messages"
    )

    author: Mapped[MessageAuthor] = mapped_column(
        SQLAlchemyEnum(MessageAuthor), nullable=False
    )  # 区分是用户还是AI

    query_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # 用户的问题
    answer_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # AI的回答

    # 存储用于生成答案的上下文信息 (非常有用)
    # 可以是检索到的 TextChunk ID 列表
    retrieved_chunk_ids: Mapped[Optional[list[int]]] = mapped_column(
        JSON, nullable=True
    )  # 存储 TextChunk.id 列表

    def __repr__(self):
        if self.author == MessageAuthor.USER:
            return f"<Message(id={self.id}, conversation_id={self.conversation_id}, author='user', query_len={len(self.query_text or '')})>"
        else:
            return f"<Message(id={self.id}, conversation_id={self.conversation_id}, author='bot', answer_len={len(self.answer_text or '')})>"
