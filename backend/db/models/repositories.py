from sqlalchemy import Table, Column, Integer, String, Text, TIMESTAMP, ForeignKey, func
from db.metadata import metadata  # Import metadata từ metadata.py

repositories = Table(
    "repositories",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("github_id", Integer, unique=True, nullable=True),  # ID GitHub của repository
    Column("owner", String(255), nullable=False),  # Chủ sở hữu repo
    Column("name", String(255), nullable=False),  # Tên repo
    Column("description", Text, nullable=True),  # Mô tả repo
    Column("user_id", Integer, ForeignKey("users.id"), nullable=True),  # Liên kết với bảng users
    Column("created_at", TIMESTAMP, server_default=func.now()),  # Thời gian tạo
    Column("updated_at", TIMESTAMP, server_default=func.now(), onupdate=func.now()),  # Thời gian cập nhật
)
