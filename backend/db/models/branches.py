from sqlalchemy import Table, Column, Integer, String, ForeignKey
from db.metadata import metadata

branches = Table(
    "branches",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(255), nullable=False),
    Column("repo_id", Integer, ForeignKey("repositories.id"), nullable=False)
)
