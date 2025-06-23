from sqlalchemy import Table, Column, Integer, String, Boolean, DateTime, func, ForeignKey
from db.metadata import metadata

assignments = Table(
    'assignments',
    metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('task_name', String(255), nullable=False),
    Column('description', String(255), nullable=True),
    Column('is_completed', Boolean, nullable=True),
    Column('user_id', Integer, ForeignKey('users.id'), nullable=False),
    Column('created_at', DateTime, nullable=True, server_default=func.now()),
    Column('updated_at', DateTime, nullable=True, server_default=func.now()),
)
