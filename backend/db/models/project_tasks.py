from sqlalchemy import Table, Column, Integer, String, Text, Boolean, TIMESTAMP, ForeignKey, func, Enum
from db.metadata import metadata  # Import metadata từ metadata.py
import enum

class TaskStatus(enum.Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"

class TaskPriority(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

project_tasks = Table(
    "project_tasks",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("title", String(255), nullable=False),  # Tiêu đề task
    Column("description", Text, nullable=True),  # Mô tả chi tiết
    Column("assignee", String(100), nullable=False),  # GitHub username của người được giao
    Column("status", Enum(TaskStatus), default=TaskStatus.TODO, nullable=False),  # Trạng thái task
    Column("priority", Enum(TaskPriority), default=TaskPriority.MEDIUM, nullable=False),  # Độ ưu tiên
    Column("due_date", String(10), nullable=True),  # Hạn hoàn thành (YYYY-MM-DD)
    Column("repo_owner", String(100), nullable=False),  # Owner của repository
    Column("repo_name", String(100), nullable=False),  # Tên repository
    Column("github_repo_id", Integer, nullable=True),  # ID repository từ GitHub API
    Column("is_completed", Boolean, default=False),  # Trạng thái hoàn thành
    Column("created_at", TIMESTAMP, server_default=func.now()),  # Thời gian tạo
    Column("updated_at", TIMESTAMP, server_default=func.now(), onupdate=func.now()),  # Thời gian cập nhật
    Column("created_by", String(100), nullable=True),  # Người tạo task
    # Indexes để tối ưu query
    extend_existing=True
)
