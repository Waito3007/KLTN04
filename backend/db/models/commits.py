# backend\db\models\commits.py
from sqlalchemy import Table, Column, Integer, String, Text, TIMESTAMP, ForeignKey, func
from db.metadata import metadata  # Import metadata từ metadata.py

commits = Table(
    "commits",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("sha", String(40), nullable=False, unique=True),  # SHA của commit
    Column("message", Text, nullable=False),  # Nội dung commit
    Column("author_name", String(255), nullable=False),  # Tên tác giả
    Column("author_email", String(255), nullable=False),  # Email tác giả
    Column("date", TIMESTAMP, nullable=False),  # Ngày commit
    Column("insertions", Integer, default=0),  # Số dòng thêm
    Column("deletions", Integer, default=0),  # Số dòng xóa
    Column("files_changed", Integer, default=0),  # Số file thay đổi
    Column("repo_id", Integer, ForeignKey("repositories.id"), nullable=False),  # Liên kết với bảng repositories
    Column("created_at", TIMESTAMP, server_default=func.now()),  # Thời gian tạo
)
