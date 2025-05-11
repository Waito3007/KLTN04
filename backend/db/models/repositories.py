from sqlalchemy import Table, Column, Integer, String, Text, TIMESTAMP, ForeignKey, func
from db.metadata import metadata  # Import metadata từ metadata.py

repositories = Table(
    "repositories",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("github_id", Integer, unique=True, nullable=True),
    Column("owner", String(255), nullable=False),
    Column("name", String(255), nullable=False),
    Column("description", Text, nullable=True),
    Column("stars", Integer, nullable=True),  # Số lượng sao
    Column("forks", Integer, nullable=True),  # Số lần fork
    Column("language", String(255), nullable=True),  # Ngôn ngữ chính
    Column("open_issues", Integer, nullable=True),  # Số lượng issue mở
    Column("url", String(255), nullable=True),  # URL của repository
    Column("user_id", Integer, ForeignKey("users.id"), nullable=True),
    Column("created_at", TIMESTAMP, server_default=func.now()),
    Column("updated_at", TIMESTAMP, server_default=func.now(), onupdate=func.now()),
)
