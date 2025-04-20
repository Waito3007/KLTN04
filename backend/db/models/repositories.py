from sqlalchemy import Table, Column, Integer, String, Text, TIMESTAMP, func
from backend.db.metadata import metadata  # Import metadata từ metadata.py

repositories = Table(
    "repositories",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("owner", String(255), nullable=False),  # Chủ sở hữu repo
    Column("name", String(255), nullable=False),  # Tên repo
    Column("description", Text, nullable=True),  # Mô tả repo
    Column("created_at", TIMESTAMP, server_default=func.now()),  # Thời gian tạo
    Column("updated_at", TIMESTAMP, server_default=func.now(), onupdate=func.now()),  # Thời gian cập nhật
)
