from sqlalchemy import Table, Column, Integer, String, Boolean, TIMESTAMP, ForeignKey, Text
from backend.db.database import metadata

assignments = Table(
    "assignments",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id")),
    Column("commit_id", Integer, ForeignKey("commits.id")),
    Column("task_type", String(50)),
    Column("recommended_by_ai", Boolean, default=True),
    Column("assigned_by", Text),
    Column("created_at", TIMESTAMP, default="now()")
)
