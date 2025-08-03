from sqlalchemy import Table, Column, Integer, String, Text, DateTime, ForeignKey, BigInteger
from db.metadata import metadata

issues = Table(
    'issues',
    metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('github_id', BigInteger, nullable=True),  # Use BigInteger for GitHub IDs
    Column('number', Integer, nullable=False),  # Issue number in the repository
    Column('title', String(255), nullable=False),
    Column('body', Text, nullable=True),
    Column('state', String(50), nullable=False),
    Column('author', String(255), nullable=True),  # Issue author username
    Column('created_at', DateTime, nullable=False),
    Column('updated_at', DateTime, nullable=True),
    Column('url', String(500), nullable=True),  # Issue HTML URL
    Column('repo_id', Integer, ForeignKey('repositories.id'), nullable=False),
)
