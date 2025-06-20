from sqlalchemy import Table, Column, Integer, String, Boolean, DateTime, func, ForeignKey
from db.metadata import metadata

branches = Table(
    'branches',
    metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('name', String(255), nullable=False),
    Column('repo_id', Integer, ForeignKey('repositories.id'), nullable=False),
    Column('creator_user_id', Integer, ForeignKey('users.id'), nullable=True),
    Column('creator_name', String(255), nullable=True),
    Column('last_committer_user_id', Integer, ForeignKey('users.id'), nullable=True),
    Column('last_committer_name', String(255), nullable=True),
    Column('sha', String(40), nullable=True),
    Column('is_default', Boolean, nullable=True),
    Column('is_protected', Boolean, nullable=True),
    Column('created_at', DateTime, nullable=True),
    Column('last_commit_date', DateTime, nullable=True),
    Column('last_synced', DateTime, nullable=True, server_default=func.now()),
    Column('commits_count', Integer, nullable=True),
    Column('contributors_count', Integer, nullable=True),
)
