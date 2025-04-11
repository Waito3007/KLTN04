from sqlalchemy import Table, Column, Integer, Text, String, TIMESTAMP, ForeignKey
from backend.db.database import metadata

pull_requests = Table(
    "pull_requests",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("pr_number", Integer),
    Column("title", Text),
    Column("body", Text),
    Column("state", String(20)),
    Column("author_id", Integer, ForeignKey("users.id")),
    Column("repo_id", Integer, ForeignKey("repositories.id")),
    Column("created_at", TIMESTAMP),
    Column("merged_at", TIMESTAMP),
    Column("closed_at", TIMESTAMP)
)
