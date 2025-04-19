from sqlalchemy import Table, Column, Integer, String, TIMESTAMP, ForeignKey, func
from backend.db.metadata import metadata  # Import metadata từ metadata.py

pull_requests = Table(
    "pull_requests",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("title", String(255), nullable=False),  # Tiêu đề pull request
    Column("description", String(255), nullable=True),  # Mô tả pull request
    Column("repo_id", Integer, ForeignKey("repositories.id"), nullable=False),  # Liên kết với bảng repositories
    Column("created_at", TIMESTAMP, server_default=func.now()),  # Thời gian tạo
    Column("updated_at", TIMESTAMP, server_default=func.now(), onupdate=func.now()),  # Thời gian cập nhật
)
