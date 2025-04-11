from sqlalchemy import Table, Column, Integer, String, Text, TIMESTAMP, ForeignKey
from backend.db.database import metadata

commits = Table(
    "commits",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("sha", String(100), unique=True, nullable=False),
    Column("message", Text),
    Column("author_id", Integer, ForeignKey("users.id", ondelete="SET NULL")),
    Column("repo_id", Integer, ForeignKey("repositories.id", ondelete="CASCADE")),
    Column("committed_at", TIMESTAMP),
    Column("insertions", Integer),
    Column("deletions", Integer),
    Column("files_changed", Integer),
    Column("ai_label", String(100)),
    Column("created_at", TIMESTAMP, default="now()")
)
