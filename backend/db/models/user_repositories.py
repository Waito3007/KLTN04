from sqlalchemy import Table, Column, Integer, String, Boolean, DateTime, func, ForeignKey
from db.metadata import metadata

user_repositories = Table(
    'user_repositories',
    metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('user_id', Integer, ForeignKey('users.id'), nullable=False),
    Column('repository_id', Integer, ForeignKey('repositories.id'), nullable=False),
    Column('role', String(12), nullable=False),
    Column('permissions', String(5), nullable=False),
    Column('is_primary_owner', Boolean, nullable=True),
    Column('joined_at', DateTime, nullable=True, server_default=func.now()),
    Column('last_accessed', DateTime, nullable=True),
    Column('created_at', DateTime, nullable=True, server_default=func.now()),
    Column('updated_at', DateTime, nullable=True, server_default=func.now()),
)
