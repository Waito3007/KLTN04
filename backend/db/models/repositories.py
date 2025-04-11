from sqlalchemy import Table, Column, Integer, String, Text, TIMESTAMP, CheckConstraint, BigInteger, func
from db.database import metadata

repositories = Table(
    "repositories",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(255), nullable=False),
    Column("provider", String(20), CheckConstraint("provider IN ('github', 'gitlab')", name="check_provider")),
    Column("repo_id", BigInteger, nullable=False),
    Column("url", Text),
    Column("default_branch", String(100), server_default="main"),
    Column("created_at", TIMESTAMP, server_default=func.now())
)
