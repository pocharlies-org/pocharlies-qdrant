"""SQLAlchemy models for tasks and task_logs."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Task(Base):
    __tablename__ = "tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id = Column(String, nullable=False)
    trigger = Column(String, nullable=False)
    trigger_ref = Column(String, nullable=True)
    status = Column(String, default="pending")
    prompt = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    tools_used = Column(JSONB, server_default=text("'[]'::jsonb"))
    started_at = Column(DateTime(timezone=True), nullable=True)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    error = Column(Text, nullable=True)
    metadata_ = Column("metadata", JSONB, server_default=text("'{}'::jsonb"))

    logs = relationship("TaskLog", back_populates="task", cascade="all, delete-orphan")


class TaskLog(Base):
    __tablename__ = "task_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=False)
    level = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    data = Column(JSONB, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    task = relationship("Task", back_populates="logs")
