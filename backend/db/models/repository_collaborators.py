from sqlalchemy import Table, Column, Integer, String, Boolean, DateTime, func, ForeignKey, Index
from db.metadata import metadata

# Repository collaborators mapping table - links repositories to collaborators
# This table maps repositories to collaborators with their specific permissions and roles
repository_collaborators = Table(
    'repository_collaborators',
    metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('repository_id', Integer, ForeignKey('repositories.id'), nullable=False),
    Column('collaborator_id', Integer, ForeignKey('collaborators.id'), nullable=False),  # Changed to reference collaborators.id
    Column('role', String(50), nullable=False),  # admin, maintain, push, triage, pull
    Column('permissions', String(100), nullable=True),  # Additional permissions if needed
    Column('is_owner', Boolean, nullable=False, default=False),  # True if this collaborator is the repo owner
    Column('joined_at', DateTime, nullable=True),  # When they became a collaborator
    Column('invited_by', String(255), nullable=True),  # Who invited them
    Column('invitation_status', String(20), nullable=True),  # pending, accepted, declined
    Column('commits_count', Integer, nullable=True, default=0),  # Number of commits made
    Column('issues_count', Integer, nullable=True, default=0),  # Number of issues created
    Column('prs_count', Integer, nullable=True, default=0),  # Number of PRs created
    Column('last_activity', DateTime, nullable=True),  # Last time they did something in the repo
    Column('created_at', DateTime, nullable=False, server_default=func.now()),
    Column('updated_at', DateTime, nullable=False, server_default=func.now(), onupdate=func.now()),
    Column('last_synced', DateTime, nullable=False, server_default=func.now()),
    Index('idx_repo_collaborators_repo', 'repository_id'),
    Index('idx_repo_collaborators_collaborator', 'collaborator_id'),
    Index('idx_repo_collaborators_unique', 'repository_id', 'collaborator_id', unique=True),
)
