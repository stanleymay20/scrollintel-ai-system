"""Legal compliance tables

Revision ID: legal_compliance_001
Revises: 
Create Date: 2025-08-14 08:18:35.066216

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = 'legal_compliance_001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Create legal_documents table
    op.create_table('legal_documents',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('document_type', sa.String(length=50), nullable=False),
        sa.Column('version', sa.String(length=20), nullable=False),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('effective_date', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('document_metadata', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_legal_documents_id'), 'legal_documents', ['id'], unique=False)
    op.create_index('ix_legal_documents_type_active', 'legal_documents', ['document_type', 'is_active'], unique=False)

    # Create user_consents table
    op.create_table('user_consents',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.String(length=100), nullable=False),
        sa.Column('consent_type', sa.String(length=50), nullable=False),
        sa.Column('consent_given', sa.Boolean(), nullable=False),
        sa.Column('consent_date', sa.DateTime(), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.String(length=500), nullable=True),
        sa.Column('document_version', sa.String(length=20), nullable=True),
        sa.Column('withdrawal_date', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_user_consents_id'), 'user_consents', ['id'], unique=False)
    op.create_index(op.f('ix_user_consents_user_id'), 'user_consents', ['user_id'], unique=False)
    op.create_index('ix_user_consents_user_type', 'user_consents', ['user_id', 'consent_type'], unique=False)

    # Create data_export_requests table
    op.create_table('data_export_requests',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.String(length=100), nullable=False),
        sa.Column('request_type', sa.String(length=20), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('requested_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('export_file_path', sa.String(length=500), nullable=True),
        sa.Column('verification_token', sa.String(length=100), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_data_export_requests_id'), 'data_export_requests', ['id'], unique=False)
    op.create_index(op.f('ix_data_export_requests_user_id'), 'data_export_requests', ['user_id'], unique=False)
    op.create_index('ix_data_export_requests_status', 'data_export_requests', ['status'], unique=False)

    # Create compliance_audits table
    op.create_table('compliance_audits',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('audit_type', sa.String(length=50), nullable=False),
        sa.Column('user_id', sa.String(length=100), nullable=True),
        sa.Column('action', sa.String(length=100), nullable=False),
        sa.Column('details', sa.JSON(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.String(length=500), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_compliance_audits_id'), 'compliance_audits', ['id'], unique=False)
    op.create_index(op.f('ix_compliance_audits_user_id'), 'compliance_audits', ['user_id'], unique=False)
    op.create_index('ix_compliance_audits_type_timestamp', 'compliance_audits', ['audit_type', 'timestamp'], unique=False)

def downgrade():
    op.drop_index('ix_compliance_audits_type_timestamp', table_name='compliance_audits')
    op.drop_index(op.f('ix_compliance_audits_user_id'), table_name='compliance_audits')
    op.drop_index(op.f('ix_compliance_audits_id'), table_name='compliance_audits')
    op.drop_table('compliance_audits')
    
    op.drop_index('ix_data_export_requests_status', table_name='data_export_requests')
    op.drop_index(op.f('ix_data_export_requests_user_id'), table_name='data_export_requests')
    op.drop_index(op.f('ix_data_export_requests_id'), table_name='data_export_requests')
    op.drop_table('data_export_requests')
    
    op.drop_index('ix_user_consents_user_type', table_name='user_consents')
    op.drop_index(op.f('ix_user_consents_user_id'), table_name='user_consents')
    op.drop_index(op.f('ix_user_consents_id'), table_name='user_consents')
    op.drop_table('user_consents')
    
    op.drop_index('ix_legal_documents_type_active', table_name='legal_documents')
    op.drop_index(op.f('ix_legal_documents_id'), table_name='legal_documents')
    op.drop_table('legal_documents')
