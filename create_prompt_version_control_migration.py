"""
Create database migration for prompt version control system.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import uuid

def upgrade():
    """Add version control tables."""
    
    # Create prompt_branches table
    op.create_table(
        'prompt_branches',
        sa.Column('id', sa.String(), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('prompt_id', sa.String(), sa.ForeignKey('advanced_prompt_templates.id'), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.Text()),
        sa.Column('parent_branch_id', sa.String(), sa.ForeignKey('prompt_branches.id')),
        sa.Column('head_version_id', sa.String(), sa.ForeignKey('advanced_prompt_versions.id')),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('is_protected', sa.Boolean(), default=False),
        sa.Column('created_by', sa.String(255), nullable=False),
        sa.Column('created_at', sa.DateTime(), default=sa.func.now()),
    )
    
    # Create prompt_commits table
    op.create_table(
        'prompt_commits',
        sa.Column('id', sa.String(), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('branch_id', sa.String(), sa.ForeignKey('prompt_branches.id'), nullable=False),
        sa.Column('version_id', sa.String(), sa.ForeignKey('advanced_prompt_versions.id'), nullable=False),
        sa.Column('parent_commit_id', sa.String(), sa.ForeignKey('prompt_commits.id')),
        sa.Column('commit_hash', sa.String(40), nullable=False, unique=True),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('author', sa.String(255), nullable=False),
        sa.Column('committer', sa.String(255), nullable=False),
        sa.Column('committed_at', sa.DateTime(), default=sa.func.now()),
    )
    
    # Create prompt_merge_requests table
    op.create_table(
        'prompt_merge_requests',
        sa.Column('id', sa.String(), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('source_branch_id', sa.String(), sa.ForeignKey('prompt_branches.id'), nullable=False),
        sa.Column('target_branch_id', sa.String(), sa.ForeignKey('prompt_branches.id'), nullable=False),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('description', sa.Text()),
        sa.Column('status', sa.String(20), default='open'),
        sa.Column('author', sa.String(255), nullable=False),
        sa.Column('assignee', sa.String(255)),
        sa.Column('reviewers', sa.JSON(), default=list),
        sa.Column('conflicts', sa.JSON(), default=list),
        sa.Column('auto_merge_enabled', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(), default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('merged_at', sa.DateTime()),
        sa.Column('merged_by', sa.String(255)),
    )
    
    # Create prompt_version_tags table
    op.create_table(
        'prompt_version_tags',
        sa.Column('id', sa.String(), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('version_id', sa.String(), sa.ForeignKey('advanced_prompt_versions.id'), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.Text()),
        sa.Column('tag_type', sa.String(20), default='release'),
        sa.Column('created_by', sa.String(255), nullable=False),
        sa.Column('created_at', sa.DateTime(), default=sa.func.now()),
    )
    
    # Create indexes for better performance
    op.create_index('idx_prompt_branches_prompt', 'prompt_branches', ['prompt_id'])
    op.create_index('idx_prompt_branches_name', 'prompt_branches', ['name'])
    op.create_index('idx_prompt_branches_active', 'prompt_branches', ['is_active'])
    
    op.create_index('idx_prompt_commits_branch', 'prompt_commits', ['branch_id'])
    op.create_index('idx_prompt_commits_version', 'prompt_commits', ['version_id'])
    op.create_index('idx_prompt_commits_hash', 'prompt_commits', ['commit_hash'])
    op.create_index('idx_prompt_commits_committed', 'prompt_commits', ['committed_at'])
    
    op.create_index('idx_prompt_merge_requests_source', 'prompt_merge_requests', ['source_branch_id'])
    op.create_index('idx_prompt_merge_requests_target', 'prompt_merge_requests', ['target_branch_id'])
    op.create_index('idx_prompt_merge_requests_status', 'prompt_merge_requests', ['status'])
    op.create_index('idx_prompt_merge_requests_author', 'prompt_merge_requests', ['author'])
    
    op.create_index('idx_prompt_version_tags_version', 'prompt_version_tags', ['version_id'])
    op.create_index('idx_prompt_version_tags_name', 'prompt_version_tags', ['name'])
    op.create_index('idx_prompt_version_tags_type', 'prompt_version_tags', ['tag_type'])


def downgrade():
    """Remove version control tables."""
    
    # Drop indexes
    op.drop_index('idx_prompt_version_tags_type')
    op.drop_index('idx_prompt_version_tags_name')
    op.drop_index('idx_prompt_version_tags_version')
    
    op.drop_index('idx_prompt_merge_requests_author')
    op.drop_index('idx_prompt_merge_requests_status')
    op.drop_index('idx_prompt_merge_requests_target')
    op.drop_index('idx_prompt_merge_requests_source')
    
    op.drop_index('idx_prompt_commits_committed')
    op.drop_index('idx_prompt_commits_hash')
    op.drop_index('idx_prompt_commits_version')
    op.drop_index('idx_prompt_commits_branch')
    
    op.drop_index('idx_prompt_branches_active')
    op.drop_index('idx_prompt_branches_name')
    op.drop_index('idx_prompt_branches_prompt')
    
    # Drop tables
    op.drop_table('prompt_version_tags')
    op.drop_table('prompt_merge_requests')
    op.drop_table('prompt_commits')
    op.drop_table('prompt_branches')


if __name__ == "__main__":
    print("Creating prompt version control migration...")
    
    # Create a simple test to verify the migration structure
    print("Migration structure:")
    print("- prompt_branches: Git-like branches for prompt development")
    print("- prompt_commits: Commit history for prompt changes")
    print("- prompt_merge_requests: Pull/merge requests for collaboration")
    print("- prompt_version_tags: Tags for marking specific versions")
    print("Migration ready for Alembic integration.")