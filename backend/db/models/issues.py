# D:\Project\KLTN04\backend\db\models\issues.py
from sqlalchemy import Table, Column, Integer, String, Text, TIMESTAMP, ForeignKey, func
from db.metadata import metadata

issues = Table(
    "issues",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("github_id", Integer, unique=True, nullable=True),
    Column("title", String(255), nullable=False),
    Column("body", Text, nullable=True),
    Column("state", String(50), nullable=False),  # open, closed
    Column("created_at", TIMESTAMP, nullable=False),
    Column("updated_at", TIMESTAMP, nullable=True),
    Column("repo_id", Integer, ForeignKey("repositories.id"), nullable=False),
)