from sqlalchemy import Table, Column, Integer, String, DateTime, Text, Boolean, func, ForeignKey, Index
from db.metadata import metadata

# Collaborators table - canonical source for all GitHub users who can be collaborators
# This table stores all GitHub users regardless of which repos they're collaborators in
collaborators = Table(
    'collaborators',
    metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('github_user_id', Integer, unique=True, nullable=False),  # GitHub user ID (unique)
    Column('github_username', String(255), nullable=False),  # GitHub username
    Column('display_name', String(255), nullable=True),  # Display name from GitHub
    Column('email', String(255), nullable=True),  # Email from GitHub (if available)
    Column('avatar_url', String(500), nullable=True),  # Avatar URL from GitHub
    Column('bio', Text, nullable=True),  # Bio from GitHub
    Column('company', String(255), nullable=True),  # Company from GitHub
    Column('location', String(255), nullable=True),  # Location from GitHub
    Column('blog', String(500), nullable=True),  # Blog URL from GitHub
    Column('is_site_admin', Boolean, nullable=True, default=False),  # Site admin flag from GitHub
    Column('node_id', String(255), nullable=True),  # GitHub node ID
    Column('gravatar_id', String(255), nullable=True),  # Gravatar ID
    Column('type', String(50), nullable=True, default='User'),  # User or Organization
    Column('user_id', Integer, ForeignKey('users.id'), nullable=True),  # Link to users table if they've logged in
    Column('created_at', DateTime, nullable=False, server_default=func.now()),
    Column('updated_at', DateTime, nullable=False, server_default=func.now(), onupdate=func.now()),
    Index('idx_collaborators_github_user_id', 'github_user_id'),
    Index('idx_collaborators_github_username', 'github_username'),
)
