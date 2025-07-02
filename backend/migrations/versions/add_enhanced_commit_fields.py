"""Add enhanced commit tracking fields

Revision ID: enhanced_commit_fields_001
Revises: a989fa2a380c
Create Date: 2025-07-02 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'enhanced_commit_fields_001'
down_revision = 'a989fa2a380c'  # Link to the initial migration
branch_labels = None
depends_on = None


def upgrade():
    """Add enhanced fields to commits table"""
    
    # Check if we're using PostgreSQL or SQLite
    connection = op.get_bind()
    dialect_name = connection.dialect.name
    
    # For PostgreSQL, use JSON type; for SQLite, use TEXT
    if dialect_name == 'postgresql':
        json_type = sa.JSON()
    else:
        # For SQLite and other databases, use TEXT and handle JSON serialization in app
        json_type = sa.TEXT()
    
    # Add JSON columns for enhanced tracking
    op.add_column('commits', sa.Column('modified_files', json_type, nullable=True))
    op.add_column('commits', sa.Column('file_types', json_type, nullable=True))
    op.add_column('commits', sa.Column('modified_directories', json_type, nullable=True))
    
    # Add new tracking fields
    op.add_column('commits', sa.Column('total_changes', sa.Integer(), nullable=True))
    op.add_column('commits', sa.Column('change_type', sa.String(length=50), nullable=True))
    op.add_column('commits', sa.Column('commit_size', sa.String(length=20), nullable=True))


def downgrade():
    """Remove enhanced fields from commits table"""
    
    # Remove the added columns
    op.drop_column('commits', 'commit_size')
    op.drop_column('commits', 'change_type')
    op.drop_column('commits', 'total_changes')
    op.drop_column('commits', 'modified_directories')
    op.drop_column('commits', 'file_types')
    op.drop_column('commits', 'modified_files')
