"""
Create database migration for ScrollIntel support system.
"""

import os
import sys
from datetime import datetime
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.sql import func

def create_support_migration():
    """Create Alembic migration for support system tables."""
    
    # Get database URL from environment
    database_url = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/scrollintel")
    
    # Create migration content
    migration_content = f'''"""Create support system tables

Revision ID: support_system_{datetime.now().strftime("%Y%m%d_%H%M%S")}
Revises: 
Create Date: {datetime.now().isoformat()}

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'support_system_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    """Create support system tables."""
    
    # Support tickets table
    op.create_table('support_tickets',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('ticket_number', sa.String(length=20), nullable=False),
        sa.Column('user_id', sa.String(length=50), nullable=True),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('subject', sa.String(length=500), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.Column('priority', sa.String(length=20), nullable=True),
        sa.Column('category', sa.String(length=100), nullable=True),
        sa.Column('tags', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('assigned_to', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('resolved_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_support_tickets_id'), 'support_tickets', ['id'], unique=False)
    op.create_index(op.f('ix_support_tickets_ticket_number'), 'support_tickets', ['ticket_number'], unique=True)
    op.create_index(op.f('ix_support_tickets_user_id'), 'support_tickets', ['user_id'], unique=False)
    
    # Support messages table
    op.create_table('support_messages',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('ticket_id', sa.Integer(), nullable=True),
        sa.Column('sender_type', sa.String(length=20), nullable=True),
        sa.Column('sender_name', sa.String(length=255), nullable=True),
        sa.Column('sender_email', sa.String(length=255), nullable=True),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('is_internal', sa.Boolean(), nullable=True),
        sa.Column('attachments', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['ticket_id'], ['support_tickets.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Knowledge base articles table
    op.create_table('kb_articles',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(length=500), nullable=False),
        sa.Column('slug', sa.String(length=200), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('category', sa.String(length=100), nullable=True),
        sa.Column('tags', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('author', sa.String(length=255), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('view_count', sa.Integer(), nullable=True),
        sa.Column('helpful_votes', sa.Integer(), nullable=True),
        sa.Column('unhelpful_votes', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_kb_articles_id'), 'kb_articles', ['id'], unique=False)
    op.create_index(op.f('ix_kb_articles_slug'), 'kb_articles', ['slug'], unique=True)
    
    # FAQ table
    op.create_table('faqs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('question', sa.String(length=1000), nullable=False),
        sa.Column('answer', sa.Text(), nullable=False),
        sa.Column('category', sa.String(length=100), nullable=True),
        sa.Column('order_index', sa.Integer(), nullable=True),
        sa.Column('is_featured', sa.Boolean(), nullable=True),
        sa.Column('view_count', sa.Integer(), nullable=True),
        sa.Column('helpful_votes', sa.Integer(), nullable=True),
        sa.Column('unhelpful_votes', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # User feedback table
    op.create_table('user_feedback',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.String(length=50), nullable=True),
        sa.Column('email', sa.String(length=255), nullable=True),
        sa.Column('feedback_type', sa.String(length=50), nullable=True),
        sa.Column('title', sa.String(length=500), nullable=True),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('rating', sa.Integer(), nullable=True),
        sa.Column('page_url', sa.String(length=1000), nullable=True),
        sa.Column('user_agent', sa.String(length=500), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.Column('priority', sa.String(length=20), nullable=True),
        sa.Column('tags', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_user_feedback_user_id'), 'user_feedback', ['user_id'], unique=False)

def downgrade():
    """Drop support system tables."""
    op.drop_index(op.f('ix_user_feedback_user_id'), table_name='user_feedback')
    op.drop_table('user_feedback')
    op.drop_table('faqs')
    op.drop_index(op.f('ix_kb_articles_slug'), table_name='kb_articles')
    op.drop_index(op.f('ix_kb_articles_id'), table_name='kb_articles')
    op.drop_table('kb_articles')
    op.drop_table('support_messages')
    op.drop_index(op.f('ix_support_tickets_user_id'), table_name='support_tickets')
    op.drop_index(op.f('ix_support_tickets_ticket_number'), table_name='support_tickets')
    op.drop_index(op.f('ix_support_tickets_id'), table_name='support_tickets')
    op.drop_table('support_tickets')
'''

    # Write migration file
    migration_dir = "alembic/versions"
    os.makedirs(migration_dir, exist_ok=True)
    
    migration_filename = f"{migration_dir}/support_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    
    with open(migration_filename, 'w') as f:
        f.write(migration_content)
    
    print(f"Created migration file: {migration_filename}")
    return migration_filename

def run_migration():
    """Run the migration to create support system tables."""
    try:
        # Create migration file
        migration_file = create_support_migration()
        
        # Run migration using Alembic
        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")
        
        print("Support system migration completed successfully!")
        
    except Exception as e:
        print(f"Migration failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_migration()