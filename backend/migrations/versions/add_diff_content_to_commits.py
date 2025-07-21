"""
Add diff_content field to commits table
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_diff_content_to_commits'
down_revision = 'a1234567890b'
branch_labels = None
depends_on = None

def upgrade():
    op.add_column('commits', sa.Column('diff_content', sa.Text(), nullable=True, comment='Raw diff content for the commit'))

def downgrade():
    op.drop_column('commits', 'diff_content')
