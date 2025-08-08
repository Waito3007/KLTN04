"""Create sync_events table

Revision ID: create_sync_events_table
Revises: 
Create Date: 2025-08-03 18:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'create_sync_events_table'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Create sync_events table
    op.create_table(
        'sync_events',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('repo_key', sa.String(length=255), nullable=False),
        sa.Column('event_type', sa.String(length=50), nullable=False),
        sa.Column('event_data', sa.Text(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for better performance
    op.create_index('ix_sync_events_id', 'sync_events', ['id'])
    op.create_index('ix_sync_events_repo_key', 'sync_events', ['repo_key'])
    op.create_index('ix_sync_events_timestamp', 'sync_events', ['timestamp'])
    op.create_index('ix_sync_events_event_type', 'sync_events', ['event_type'])
    
    # Composite index for common queries
    op.create_index('ix_sync_events_repo_timestamp', 'sync_events', ['repo_key', 'timestamp'])

def downgrade():
    # Drop indexes
    op.drop_index('ix_sync_events_repo_timestamp', 'sync_events')
    op.drop_index('ix_sync_events_event_type', 'sync_events')
    op.drop_index('ix_sync_events_timestamp', 'sync_events')
    op.drop_index('ix_sync_events_repo_key', 'sync_events')
    op.drop_index('ix_sync_events_id', 'sync_events')
    
    # Drop table
    op.drop_table('sync_events')
