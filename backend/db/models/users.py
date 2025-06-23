from sqlalchemy import Table, Column, Integer, String, Boolean, Text, DateTime, func
from db.metadata import metadata

users = Table(
    'users',
    metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('github_id', Integer, nullable=True),
    Column('github_username', String(255), nullable=False),
    Column('email', String(255), nullable=True),
    Column('display_name', String(255), nullable=True),
    Column('full_name', String(255), nullable=True),
    Column('avatar_url', String(500), nullable=True),
    Column('bio', Text, nullable=True),
    Column('location', String(255), nullable=True),
    Column('company', String(255), nullable=True),
    Column('blog', String(500), nullable=True),
    Column('twitter_username', String(255), nullable=True),
    Column('github_profile_url', String(500), nullable=True),
    Column('repos_url', String(500), nullable=True),
    Column('is_active', Boolean, nullable=True),
    Column('is_verified', Boolean, nullable=True),
    Column('github_created_at', DateTime, nullable=True),
    Column('last_synced', DateTime, nullable=True),
    Column('created_at', DateTime, nullable=True, server_default=func.now()),
    Column('updated_at', DateTime, nullable=True, server_default=func.now()),
)
