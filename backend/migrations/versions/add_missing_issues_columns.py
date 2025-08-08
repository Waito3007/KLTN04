"""add missing issues columns

Revision ID: add_missing_issues_cols
Revises: 
Create Date: 2025-08-03 22:06:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_missing_issues_cols'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """Add missing columns to issues table"""
    try:
        # Add number column
        op.add_column('issues', sa.Column('number', sa.Integer(), nullable=True))
    except Exception as e:
        print(f"Column 'number' might already exist: {e}")
    
    try:
        # Add author column
        op.add_column('issues', sa.Column('author', sa.String(255), nullable=True))
    except Exception as e:
        print(f"Column 'author' might already exist: {e}")
    
    try:
        # Add url column
        op.add_column('issues', sa.Column('url', sa.String(500), nullable=True))
    except Exception as e:
        print(f"Column 'url' might already exist: {e}")


def downgrade():
    """Remove added columns"""
    op.drop_column('issues', 'url')
    op.drop_column('issues', 'author')
    op.drop_column('issues', 'number')
