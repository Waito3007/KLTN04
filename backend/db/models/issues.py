from sqlalchemy import Table, Column, Integer, String, Text, DateTime, ForeignKey
from db.metadata import metadata

issues = Table(
    'issues',
    metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('github_id', Integer, nullable=True),
    Column('title', String(255), nullable=False),
    Column('body', Text, nullable=True),
    Column('state', String(50), nullable=False),
    Column('created_at', DateTime, nullable=False),
    Column('updated_at', DateTime, nullable=True),
    Column('repo_id', Integer, ForeignKey('repositories.id'), nullable=False),
)
