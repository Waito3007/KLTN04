from sqlalchemy import Table, Column, Integer, String, Boolean, Text, DateTime, func, ForeignKey
from enum import Enum
from db.metadata import metadata

# Định nghĩa các enum cho Task Status và Priority
class TaskStatus(str, Enum):
    TODO = "TODO"
    IN_PROGRESS = "IN_PROGRESS"
    DONE = "DONE"
    CANCELLED = "CANCELLED"  # Keep for compatibility

class TaskPriority(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    URGENT = "URGENT"  # Keep for compatibility

project_tasks = Table(
    'project_tasks',
    metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('title', String(255), nullable=False),
    Column('description', Text, nullable=True),
    Column('assignee_user_id', Integer, ForeignKey('users.id'), nullable=True),
    Column('assignee_github_username', String(100), nullable=True),
    Column('status', String(11), nullable=False),
    Column('priority', String(6), nullable=False),
    Column('due_date', String(10), nullable=True),
    Column('repository_id', Integer, ForeignKey('repositories.id'), nullable=True),
    Column('repo_owner', String(100), nullable=True),
    Column('repo_name', String(100), nullable=True),
    Column('is_completed', Boolean, nullable=True),
    Column('created_at', DateTime, nullable=True, server_default=func.now()),
    Column('updated_at', DateTime, nullable=True, server_default=func.now()),
    Column('created_by_user_id', Integer, ForeignKey('users.id'), nullable=True),
    Column('created_by', String(100), nullable=True),
)
