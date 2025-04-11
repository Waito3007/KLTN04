from sqlalchemy import Table, Column, Integer, String, Text, TIMESTAMP, CheckConstraint, BigInteger
from backend.db.database import metadata

repositories = Table(
    "repositories",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(255), nullable=False),
    Column("provider", String(20), CheckConstraint("provider IN ('github', 'gitlab')")),
    Column("repo_id", BigInteger, nullable=False),
    Column("url", Text),
    Column("default_branch", String(100), default="main"),
    Column("created_at", TIMESTAMP, default="now()")
)
