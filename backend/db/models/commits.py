from sqlalchemy import Table, Column, Integer, String, Boolean, Text, DateTime, func, ForeignKey, JSON
from sqlalchemy.dialects import postgresql, sqlite
from db.metadata import metadata

# Define JSON type that works with both PostgreSQL and SQLite
def JSONType():
    """Return appropriate JSON type based on database dialect"""
    return JSON().with_variant(Text(), "sqlite")

commits = Table(
    'commits',
    metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('sha', String(40), nullable=False),
    Column('message', Text, nullable=False),
    Column('author_user_id', Integer, ForeignKey('users.id'), nullable=True),
    Column('author_name', String(255), nullable=False),
    Column('author_email', String(255), nullable=False),
    Column('committer_user_id', Integer, ForeignKey('users.id'), nullable=True),
    Column('committer_name', String(255), nullable=True),
    Column('committer_email', String(255), nullable=True),
    Column('repo_id', Integer, ForeignKey('repositories.id'), nullable=False),
    Column('branch_id', Integer, ForeignKey('branches.id'), nullable=True),
    Column('branch_name', String(255), nullable=True),
    Column('author_role_at_commit', String(20), nullable=True),
    Column('author_permissions_at_commit', String(100), nullable=True),
    Column('date', DateTime, nullable=False),
    Column('committer_date', DateTime, nullable=True),
    Column('insertions', Integer, nullable=True),
    Column('deletions', Integer, nullable=True),
    Column('files_changed', Integer, nullable=True),
    Column('parent_sha', String(40), nullable=True),
    Column('is_merge', Boolean, nullable=True),
    Column('merge_from_branch', String(255), nullable=True),
    # Enhanced fields for file tracking (compatible with both PostgreSQL and SQLite)
    Column('modified_files', JSONType(), nullable=True, comment='List of modified file paths'),
    Column('file_types', JSONType(), nullable=True, comment='Dictionary of file extensions and their counts'),
    Column('modified_directories', JSONType(), nullable=True, comment='Dictionary of directories and their change counts'),
    Column('total_changes', Integer, nullable=True, comment='Total number of changes (additions + deletions)'),
    Column('change_type', String(50), nullable=True, comment='Type of change: feature, bugfix, refactor, etc.'),
    Column('commit_size', String(20), nullable=True, comment='Size category: small, medium, large'),
    Column('created_at', DateTime, nullable=True, server_default=func.now()),
    Column('last_synced', DateTime, nullable=True, server_default=func.now()),
    Column('diff_content', Text, nullable=True, comment='Raw diff content for the commit'),
)
