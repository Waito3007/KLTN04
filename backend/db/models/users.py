from sqlalchemy import Table, Column, Integer, String, Boolean, BigInteger, TIMESTAMP, Text, func
from db.database import metadata

users = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("username", String(50), nullable=False),
    Column("email", String(100), nullable=False, unique=True),
    Column("github_id", BigInteger, unique=True),
    Column("gitlab_id", BigInteger, unique=True),
    Column("avatar_url", Text),
    Column("is_active", Boolean, server_default="true"),
    Column("created_at", TIMESTAMP, server_default=func.now())
)
