from sqlalchemy import Table, Column, Integer, String, DateTime, func, ForeignKey, BigInteger
from db.metadata import metadata

pull_requests = Table(
    'pull_requests',
    metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('github_id', BigInteger, nullable=True),  # Use BigInteger for GitHub IDs
    Column('title', String(255), nullable=False),
    Column('description', String(255), nullable=True),
    Column('state', String(50), nullable=True),
    Column('repo_id', Integer, ForeignKey('repositories.id'), nullable=False),
    Column('created_at', DateTime, nullable=True, server_default=func.now()),
    Column('updated_at', DateTime, nullable=True, server_default=func.now()),
)
