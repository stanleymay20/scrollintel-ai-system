"""Initial database schema migration for AI Data Readiness Platform."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import uuid


def upgrade():
    """Create initial database schema."""
    
    # Create datasets table
    op.create_table(
        'datasets',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('schema_definition', postgresql.JSON),
        sa.Column('dataset_metadata', postgresql.JSON),
        sa.Column('quality_score', sa.Float, default=0.0),
        sa.Column('ai_readiness_score', sa.Float, default=0.0),
        sa.Column('status', sa.Enum('pending', 'processing', 'ready', 'error', 'archived', name='dataset_status'), default='pending'),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('version', sa.String(50), default='1.0'),
        sa.Column('lineage', postgresql.JSON),
        sa.Column('owner', sa.String(255)),
        sa.Column('tags', postgresql.JSON)
    )
    
    # Create quality_reports table
    op.create_table(
        'quality_reports',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('dataset_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('overall_score', sa.Float, nullable=False),
        sa.Column('completeness_score', sa.Float, nullable=False),
        sa.Column('accuracy_score', sa.Float, nullable=False),
        sa.Column('consistency_score', sa.Float, nullable=False),
        sa.Column('validity_score', sa.Float, nullable=False),
        sa.Column('uniqueness_score', sa.Float, default=0.0),
        sa.Column('timeliness_score', sa.Float, default=0.0),
        sa.Column('issues', postgresql.JSON),
        sa.Column('recommendations', postgresql.JSON),
        sa.Column('generated_at', sa.DateTime, default=sa.func.now()),
        sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='CASCADE')
    )
    
    # Create bias_reports table
    op.create_table(
        'bias_reports',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('dataset_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('protected_attributes', postgresql.JSON),
        sa.Column('bias_metrics', postgresql.JSON),
        sa.Column('fairness_violations', postgresql.JSON),
        sa.Column('mitigation_strategies', postgresql.JSON),
        sa.Column('generated_at', sa.DateTime, default=sa.func.now()),
        sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='CASCADE')
    )
    
    # Create ai_readiness_scores table
    op.create_table(
        'ai_readiness_scores',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('dataset_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('overall_score', sa.Float, nullable=False),
        sa.Column('data_quality_score', sa.Float, nullable=False),
        sa.Column('feature_quality_score', sa.Float, nullable=False),
        sa.Column('bias_score', sa.Float, nullable=False),
        sa.Column('compliance_score', sa.Float, nullable=False),
        sa.Column('scalability_score', sa.Float, nullable=False),
        sa.Column('dimensions', postgresql.JSON),
        sa.Column('improvement_areas', postgresql.JSON),
        sa.Column('generated_at', sa.DateTime, default=sa.func.now()),
        sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='CASCADE')
    )
    
    # Create drift_reports table
    op.create_table(
        'drift_reports',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('dataset_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('reference_dataset_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('drift_score', sa.Float, nullable=False),
        sa.Column('feature_drift_scores', postgresql.JSON),
        sa.Column('statistical_tests', postgresql.JSON),
        sa.Column('alerts', postgresql.JSON),
        sa.Column('recommendations', postgresql.JSON),
        sa.Column('generated_at', sa.DateTime, default=sa.func.now()),
        sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['reference_dataset_id'], ['datasets.id'], ondelete='CASCADE')
    )
    
    # Create processing_jobs table
    op.create_table(
        'processing_jobs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('dataset_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('job_type', sa.String(100), nullable=False),
        sa.Column('status', sa.String(50), default='pending'),
        sa.Column('progress', sa.Float, default=0.0),
        sa.Column('error_message', sa.Text),
        sa.Column('started_at', sa.DateTime),
        sa.Column('completed_at', sa.DateTime),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('config', postgresql.JSON),
        sa.Column('results', postgresql.JSON),
        sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='CASCADE')
    )
    
    # Create indexes for better performance
    op.create_index('idx_datasets_status', 'datasets', ['status'])
    op.create_index('idx_datasets_owner', 'datasets', ['owner'])
    op.create_index('idx_datasets_created_at', 'datasets', ['created_at'])
    op.create_index('idx_quality_reports_dataset_id', 'quality_reports', ['dataset_id'])
    op.create_index('idx_bias_reports_dataset_id', 'bias_reports', ['dataset_id'])
    op.create_index('idx_ai_readiness_scores_dataset_id', 'ai_readiness_scores', ['dataset_id'])
    op.create_index('idx_drift_reports_dataset_id', 'drift_reports', ['dataset_id'])
    op.create_index('idx_processing_jobs_dataset_id', 'processing_jobs', ['dataset_id'])
    op.create_index('idx_processing_jobs_status', 'processing_jobs', ['status'])


def downgrade():
    """Drop all tables."""
    op.drop_table('processing_jobs')
    op.drop_table('drift_reports')
    op.drop_table('ai_readiness_scores')
    op.drop_table('bias_reports')
    op.drop_table('quality_reports')
    op.drop_table('datasets')
    
    # Drop enum type
    op.execute("DROP TYPE IF EXISTS dataset_status")