from sqlalchemy import Table, Column, Integer, String, DateTime, func, ForeignKey
from db.metadata import metadata

collaborators = Table(
    'collaborators',
    metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('user_id', Integer, ForeignKey('users.id'), nullable=False),
    Column('created_at', DateTime, nullable=True, server_default=func.now()),
    Column('github_user_id', Integer, nullable=True),
    Column('github_username', String(255), nullable=True),
)
