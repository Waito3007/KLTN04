from sqlalchemy import Table, Column, Integer, String, Boolean, TIMESTAMP, ForeignKey, func
from db.metadata import metadata  # Import metadata từ metadata.py

assignments = Table(
    "assignments",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("task_name", String(255), nullable=False),  # Tên nhiệm vụ
    Column("description", String(255), nullable=True),  # Mô tả nhiệm vụ
    Column("is_completed", Boolean, default=False),  # Trạng thái hoàn thành
    Column("user_id", Integer, ForeignKey("users.id"), nullable=False),  # Liên kết với bảng users
    Column("created_at", TIMESTAMP, server_default=func.now()),  # Thời gian tạo
    Column("updated_at", TIMESTAMP, server_default=func.now(), onupdate=func.now()),  # Thời gian cập nhật
)
