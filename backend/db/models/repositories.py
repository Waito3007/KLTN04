from sqlalchemy import Table, Column, Integer, String, Boolean, Text, DateTime, func, ForeignKey
from db.metadata import metadata

repositories = Table(
    'repositories',
    metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('github_id', Integer, nullable=False),
    Column('owner', String(255), nullable=False),
    Column('name', String(255), nullable=False),
    Column('full_name', String(500), nullable=True),
    Column('description', Text, nullable=True),
    Column('stars', Integer, nullable=True),
    Column('forks', Integer, nullable=True),
    Column('language', String(100), nullable=True),
    Column('open_issues', Integer, nullable=True),
    Column('url', String(500), nullable=True),
    Column('clone_url', String(500), nullable=True),
    Column('is_private', Boolean, nullable=True),
    Column('is_fork', Boolean, nullable=True),
    Column('default_branch', String(100), nullable=True),
    Column('last_synced', DateTime, nullable=True),
    Column('sync_status', String(20), nullable=True),
    Column('user_id', Integer, ForeignKey('users.id'), nullable=True),
    Column('created_at', DateTime, nullable=True, server_default=func.now()),
    Column('updated_at', DateTime, nullable=True, server_default=func.now()),
)
