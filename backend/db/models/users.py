from sqlalchemy import Table, Column, Integer, String, TIMESTAMP, func
from backend.db.metadata import metadata  # Import metadata từ metadata.py

users = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("github_username", String(255), nullable=False, unique=True),  # Tên GitHub
    Column("email", String(255), nullable=False, unique=True),  # Email
    Column("created_at", TIMESTAMP, server_default=func.now()),  # Thời gian tạo
    Column("updated_at", TIMESTAMP, server_default=func.now(), onupdate=func.now()),  # Thời gian cập nhật
    extend_existing=True  # Thêm dòng này
)
