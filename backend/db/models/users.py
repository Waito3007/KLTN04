from sqlalchemy import Table, Column, Integer, String, TIMESTAMP, func
from db.metadata import metadata  # Import metadata từ metadata.py

users = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("github_id", Integer, unique=True, nullable=True),  # ID GitHub
    Column("github_username", String(255), nullable=False, unique=True),  # Tên GitHub
    Column("email", String(255), nullable=False, unique=True),  # Email
    Column("avatar_url", String(255), nullable=True),  # URL ảnh đại diện
    Column("created_at", TIMESTAMP, server_default=func.now()),  # Thời gian tạo
    Column("updated_at", TIMESTAMP, server_default=func.now(), onupdate=func.now()),  # Thời gian cập nhật
)
