"""Initial migration with all models

Revision ID: a989fa2a380c
Revises: 
Create Date: 2025-06-20 21:03:16.707542

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'a989fa2a380c'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('repository_collaborators')
    op.drop_table('assignments')
    op.drop_table('collaborators')
    op.drop_table('project_tasks')
    op.drop_table('issues')
    op.drop_table('user_repositories')
    op.drop_table('users')
    op.drop_table('repositories')
    op.drop_table('pull_requests')
    op.drop_table('commits')
    op.drop_table('branches')
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('branches',
    sa.Column('id', sa.INTEGER(), server_default=sa.text("nextval('branches_id_seq'::regclass)"), autoincrement=True, nullable=False),
    sa.Column('name', sa.VARCHAR(length=255), autoincrement=False, nullable=False),
    sa.Column('repo_id', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('creator_user_id', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('creator_name', sa.VARCHAR(length=255), autoincrement=False, nullable=True),
    sa.Column('last_committer_user_id', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('last_committer_name', sa.VARCHAR(length=255), autoincrement=False, nullable=True),
    sa.Column('sha', sa.VARCHAR(length=40), autoincrement=False, nullable=True),
    sa.Column('is_default', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.Column('is_protected', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.Column('created_at', postgresql.TIMESTAMP(), autoincrement=False, nullable=True),
    sa.Column('last_commit_date', postgresql.TIMESTAMP(), autoincrement=False, nullable=True),
    sa.Column('last_synced', postgresql.TIMESTAMP(), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('commits_count', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('contributors_count', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['creator_user_id'], ['users.id'], name='branches_creator_user_id_fkey'),
    sa.ForeignKeyConstraint(['last_committer_user_id'], ['users.id'], name='branches_last_committer_user_id_fkey'),
    sa.ForeignKeyConstraint(['repo_id'], ['repositories.id'], name='branches_repo_id_fkey'),
    sa.PrimaryKeyConstraint('id', name='branches_pkey'),
    postgresql_ignore_search_path=False
    )
    op.create_table('commits',
    sa.Column('id', sa.INTEGER(), autoincrement=True, nullable=False),
    sa.Column('sha', sa.VARCHAR(length=40), autoincrement=False, nullable=False),
    sa.Column('message', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('author_user_id', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('author_name', sa.VARCHAR(length=255), autoincrement=False, nullable=False),
    sa.Column('author_email', sa.VARCHAR(length=255), autoincrement=False, nullable=False),
    sa.Column('committer_user_id', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('committer_name', sa.VARCHAR(length=255), autoincrement=False, nullable=True),
    sa.Column('committer_email', sa.VARCHAR(length=255), autoincrement=False, nullable=True),
    sa.Column('repo_id', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('branch_id', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('branch_name', sa.VARCHAR(length=255), autoincrement=False, nullable=True),
    sa.Column('author_role_at_commit', sa.VARCHAR(length=20), autoincrement=False, nullable=True),
    sa.Column('author_permissions_at_commit', sa.VARCHAR(length=100), autoincrement=False, nullable=True),
    sa.Column('date', postgresql.TIMESTAMP(), autoincrement=False, nullable=False),
    sa.Column('committer_date', postgresql.TIMESTAMP(), autoincrement=False, nullable=True),
    sa.Column('insertions', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('deletions', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('files_changed', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('parent_sha', sa.VARCHAR(length=40), autoincrement=False, nullable=True),
    sa.Column('is_merge', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.Column('merge_from_branch', sa.VARCHAR(length=255), autoincrement=False, nullable=True),
    sa.Column('created_at', postgresql.TIMESTAMP(), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('last_synced', postgresql.TIMESTAMP(), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['author_user_id'], ['users.id'], name=op.f('commits_author_user_id_fkey')),
    sa.ForeignKeyConstraint(['branch_id'], ['branches.id'], name=op.f('commits_branch_id_fkey')),
    sa.ForeignKeyConstraint(['committer_user_id'], ['users.id'], name=op.f('commits_committer_user_id_fkey')),
    sa.ForeignKeyConstraint(['repo_id'], ['repositories.id'], name=op.f('commits_repo_id_fkey')),
    sa.PrimaryKeyConstraint('id', name=op.f('commits_pkey'))
    )
    op.create_table('pull_requests',
    sa.Column('id', sa.INTEGER(), autoincrement=True, nullable=False),
    sa.Column('github_id', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('title', sa.VARCHAR(length=255), autoincrement=False, nullable=False),
    sa.Column('description', sa.VARCHAR(length=255), autoincrement=False, nullable=True),
    sa.Column('state', sa.VARCHAR(length=50), autoincrement=False, nullable=True),
    sa.Column('repo_id', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('created_at', postgresql.TIMESTAMP(), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('updated_at', postgresql.TIMESTAMP(), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['repo_id'], ['repositories.id'], name=op.f('pull_requests_repo_id_fkey')),
    sa.PrimaryKeyConstraint('id', name=op.f('pull_requests_pkey'))
    )
    op.create_table('repositories',
    sa.Column('id', sa.INTEGER(), server_default=sa.text("nextval('repositories_id_seq'::regclass)"), autoincrement=True, nullable=False),
    sa.Column('github_id', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('owner', sa.VARCHAR(length=255), autoincrement=False, nullable=False),
    sa.Column('name', sa.VARCHAR(length=255), autoincrement=False, nullable=False),
    sa.Column('full_name', sa.VARCHAR(length=500), autoincrement=False, nullable=True),
    sa.Column('description', sa.TEXT(), autoincrement=False, nullable=True),
    sa.Column('stars', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('forks', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('language', sa.VARCHAR(length=100), autoincrement=False, nullable=True),
    sa.Column('open_issues', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('url', sa.VARCHAR(length=500), autoincrement=False, nullable=True),
    sa.Column('clone_url', sa.VARCHAR(length=500), autoincrement=False, nullable=True),
    sa.Column('is_private', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.Column('is_fork', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.Column('default_branch', sa.VARCHAR(length=100), autoincrement=False, nullable=True),
    sa.Column('last_synced', postgresql.TIMESTAMP(), autoincrement=False, nullable=True),
    sa.Column('sync_status', sa.VARCHAR(length=20), autoincrement=False, nullable=True),
    sa.Column('user_id', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('created_at', postgresql.TIMESTAMP(), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('updated_at', postgresql.TIMESTAMP(), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name='repositories_user_id_fkey'),
    sa.PrimaryKeyConstraint('id', name='repositories_pkey'),
    postgresql_ignore_search_path=False
    )
    op.create_table('users',
    sa.Column('id', sa.INTEGER(), server_default=sa.text("nextval('users_id_seq'::regclass)"), autoincrement=True, nullable=False),
    sa.Column('github_id', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('github_username', sa.VARCHAR(length=255), autoincrement=False, nullable=False),
    sa.Column('email', sa.VARCHAR(length=255), autoincrement=False, nullable=True),
    sa.Column('display_name', sa.VARCHAR(length=255), autoincrement=False, nullable=True),
    sa.Column('full_name', sa.VARCHAR(length=255), autoincrement=False, nullable=True),
    sa.Column('avatar_url', sa.VARCHAR(length=500), autoincrement=False, nullable=True),
    sa.Column('bio', sa.TEXT(), autoincrement=False, nullable=True),
    sa.Column('location', sa.VARCHAR(length=255), autoincrement=False, nullable=True),
    sa.Column('company', sa.VARCHAR(length=255), autoincrement=False, nullable=True),
    sa.Column('blog', sa.VARCHAR(length=500), autoincrement=False, nullable=True),
    sa.Column('twitter_username', sa.VARCHAR(length=255), autoincrement=False, nullable=True),
    sa.Column('github_profile_url', sa.VARCHAR(length=500), autoincrement=False, nullable=True),
    sa.Column('repos_url', sa.VARCHAR(length=500), autoincrement=False, nullable=True),
    sa.Column('is_active', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.Column('is_verified', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.Column('github_created_at', postgresql.TIMESTAMP(), autoincrement=False, nullable=True),
    sa.Column('last_synced', postgresql.TIMESTAMP(), autoincrement=False, nullable=True),
    sa.Column('created_at', postgresql.TIMESTAMP(), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('updated_at', postgresql.TIMESTAMP(), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.PrimaryKeyConstraint('id', name='users_pkey'),
    postgresql_ignore_search_path=False
    )
    op.create_table('user_repositories',
    sa.Column('id', sa.INTEGER(), autoincrement=True, nullable=False),
    sa.Column('user_id', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('repository_id', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('role', sa.VARCHAR(length=12), autoincrement=False, nullable=False),
    sa.Column('permissions', sa.VARCHAR(length=5), autoincrement=False, nullable=False),
    sa.Column('is_primary_owner', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.Column('joined_at', postgresql.TIMESTAMP(), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('last_accessed', postgresql.TIMESTAMP(), autoincrement=False, nullable=True),
    sa.Column('created_at', postgresql.TIMESTAMP(), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('updated_at', postgresql.TIMESTAMP(), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['repository_id'], ['repositories.id'], name=op.f('user_repositories_repository_id_fkey')),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('user_repositories_user_id_fkey')),
    sa.PrimaryKeyConstraint('id', name=op.f('user_repositories_pkey'))
    )
    op.create_table('issues',
    sa.Column('id', sa.INTEGER(), autoincrement=True, nullable=False),
    sa.Column('github_id', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('title', sa.VARCHAR(length=255), autoincrement=False, nullable=False),
    sa.Column('body', sa.TEXT(), autoincrement=False, nullable=True),
    sa.Column('state', sa.VARCHAR(length=50), autoincrement=False, nullable=False),
    sa.Column('created_at', postgresql.TIMESTAMP(), autoincrement=False, nullable=False),
    sa.Column('updated_at', postgresql.TIMESTAMP(), autoincrement=False, nullable=True),
    sa.Column('repo_id', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.ForeignKeyConstraint(['repo_id'], ['repositories.id'], name=op.f('issues_repo_id_fkey')),
    sa.PrimaryKeyConstraint('id', name=op.f('issues_pkey'))
    )
    op.create_table('project_tasks',
    sa.Column('id', sa.INTEGER(), autoincrement=True, nullable=False),
    sa.Column('title', sa.VARCHAR(length=255), autoincrement=False, nullable=False),
    sa.Column('description', sa.TEXT(), autoincrement=False, nullable=True),
    sa.Column('assignee_user_id', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('assignee_github_username', sa.VARCHAR(length=100), autoincrement=False, nullable=True),
    sa.Column('status', sa.VARCHAR(length=11), autoincrement=False, nullable=False),
    sa.Column('priority', sa.VARCHAR(length=6), autoincrement=False, nullable=False),
    sa.Column('due_date', sa.VARCHAR(length=10), autoincrement=False, nullable=True),
    sa.Column('repository_id', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('repo_owner', sa.VARCHAR(length=100), autoincrement=False, nullable=True),
    sa.Column('repo_name', sa.VARCHAR(length=100), autoincrement=False, nullable=True),
    sa.Column('is_completed', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.Column('created_at', postgresql.TIMESTAMP(), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('updated_at', postgresql.TIMESTAMP(), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('created_by_user_id', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('created_by', sa.VARCHAR(length=100), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['assignee_user_id'], ['users.id'], name=op.f('project_tasks_assignee_user_id_fkey')),
    sa.ForeignKeyConstraint(['created_by_user_id'], ['users.id'], name=op.f('project_tasks_created_by_user_id_fkey')),
    sa.ForeignKeyConstraint(['repository_id'], ['repositories.id'], name=op.f('project_tasks_repository_id_fkey')),
    sa.PrimaryKeyConstraint('id', name=op.f('project_tasks_pkey'))
    )
    op.create_table('collaborators',
    sa.Column('id', sa.INTEGER(), autoincrement=True, nullable=False),
    sa.Column('user_id', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('created_at', postgresql.TIMESTAMP(), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('github_user_id', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('github_username', sa.VARCHAR(length=255), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('collaborators_user_id_fkey')),
    sa.PrimaryKeyConstraint('id', name=op.f('collaborators_pkey'))
    )
    op.create_table('assignments',
    sa.Column('id', sa.INTEGER(), autoincrement=True, nullable=False),
    sa.Column('task_name', sa.VARCHAR(length=255), autoincrement=False, nullable=False),
    sa.Column('description', sa.VARCHAR(length=255), autoincrement=False, nullable=True),
    sa.Column('is_completed', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.Column('user_id', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('created_at', postgresql.TIMESTAMP(), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('updated_at', postgresql.TIMESTAMP(), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('assignments_user_id_fkey')),
    sa.PrimaryKeyConstraint('id', name=op.f('assignments_pkey'))
    )
    op.create_table('repository_collaborators',
    sa.Column('id', sa.INTEGER(), autoincrement=True, nullable=False),
    sa.Column('repository_id', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('user_id', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('role', sa.VARCHAR(length=8), autoincrement=False, nullable=False),
    sa.Column('permissions', sa.VARCHAR(length=100), autoincrement=False, nullable=True),
    sa.Column('is_owner', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.Column('joined_at', postgresql.TIMESTAMP(), autoincrement=False, nullable=True),
    sa.Column('invited_by', sa.VARCHAR(length=255), autoincrement=False, nullable=True),
    sa.Column('invitation_status', sa.VARCHAR(length=20), autoincrement=False, nullable=True),
    sa.Column('commits_count', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('issues_count', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('prs_count', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('last_activity', postgresql.TIMESTAMP(), autoincrement=False, nullable=True),
    sa.Column('last_synced', postgresql.TIMESTAMP(), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['repository_id'], ['repositories.id'], name=op.f('repository_collaborators_repository_id_fkey')),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('repository_collaborators_user_id_fkey')),
    sa.PrimaryKeyConstraint('id', name=op.f('repository_collaborators_pkey'))
    )
    # ### end Alembic commands ###
