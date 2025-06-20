from sqlalchemy import Table, Column, Integer, String, Boolean, DateTime, func, ForeignKey
from db.metadata import metadata

repository_collaborators = Table(
    'repository_collaborators',
    metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('repository_id', Integer, ForeignKey('repositories.id'), nullable=False),
    Column('user_id', Integer, ForeignKey('users.id'), nullable=False),
    Column('role', String(8), nullable=False),
    Column('permissions', String(100), nullable=True),
    Column('is_owner', Boolean, nullable=True),
    Column('joined_at', DateTime, nullable=True),
    Column('invited_by', String(255), nullable=True),
    Column('invitation_status', String(20), nullable=True),
    Column('commits_count', Integer, nullable=True),
    Column('issues_count', Integer, nullable=True),
    Column('prs_count', Integer, nullable=True),
    Column('last_activity', DateTime, nullable=True),
    Column('last_synced', DateTime, nullable=True, server_default=func.now()),
)
