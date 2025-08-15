"""
Create legal compliance database migration for ScrollIntel.
"""

import os
import sys
from datetime import datetime
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_legal_compliance_migration():
    """Create database migration for legal compliance tables."""
    
    migration_content = '''"""Legal compliance tables

Revision ID: legal_compliance_001
Revises: 
Create Date: {create_date}

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
'''.format(create_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))

    # Create migration file
    migration_dir = "alembic/versions"
    os.makedirs(migration_dir, exist_ok=True)
    
    migration_filename = f"{migration_dir}/legal_compliance_001.py"
    
    with open(migration_filename, 'w') as f:
        f.write(migration_content)
    
    print(f"Created migration file: {migration_filename}")
    return migration_filename

def seed_legal_documents():
    """Seed initial legal documents."""
    
    from scrollintel.models.legal_models import LegalDocument
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    # Create database connection
    engine = create_engine('sqlite:///legal_compliance.db')
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Sample legal documents
    documents = [
        {
            "document_type": "terms_of_service",
            "version": "1.0",
            "title": "ScrollIntel Terms of Service",
            "content": """
<h2>1. Acceptance of Terms</h2>
<p>By accessing and using ScrollIntel ("the Service"), you accept and agree to be bound by the terms and provision of this agreement.</p>

<h2>2. Description of Service</h2>
<p>ScrollIntel is an AI-powered platform that provides CTO replacement capabilities, data analysis, and business intelligence services.</p>

<h2>3. User Accounts</h2>
<p>You are responsible for maintaining the confidentiality of your account and password and for restricting access to your computer.</p>

<h2>4. Acceptable Use</h2>
<p>You agree not to use the Service for any unlawful purpose or in any way that could damage, disable, overburden, or impair the Service.</p>

<h2>5. Privacy Policy</h2>
<p>Your privacy is important to us. Please review our Privacy Policy, which also governs your use of the Service.</p>

<h2>6. Intellectual Property</h2>
<p>The Service and its original content, features, and functionality are owned by ScrollIntel and are protected by international copyright, trademark, patent, trade secret, and other intellectual property laws.</p>

<h2>7. Termination</h2>
<p>We may terminate or suspend your account and bar access to the Service immediately, without prior notice or liability, under our sole discretion.</p>

<h2>8. Limitation of Liability</h2>
<p>In no event shall ScrollIntel be liable for any indirect, incidental, special, consequential, or punitive damages.</p>

<h2>9. Governing Law</h2>
<p>These Terms shall be interpreted and governed by the laws of the jurisdiction in which ScrollIntel operates.</p>

<h2>10. Contact Information</h2>
<p>If you have any questions about these Terms, please contact us at legal@scrollintel.com</p>
            """,
            "effective_date": datetime.utcnow(),
            "document_metadata": {"language": "en", "jurisdiction": "US"}
        },
        {
            "document_type": "privacy_policy",
            "version": "1.0",
            "title": "ScrollIntel Privacy Policy",
            "content": """
<h2>1. Information We Collect</h2>
<p>We collect information you provide directly to us, such as when you create an account, use our services, or contact us for support.</p>

<h2>2. How We Use Your Information</h2>
<p>We use the information we collect to provide, maintain, and improve our services, process transactions, and communicate with you.</p>

<h2>3. Information Sharing</h2>
<p>We do not sell, trade, or otherwise transfer your personal information to third parties without your consent, except as described in this policy.</p>

<h2>4. Data Security</h2>
<p>We implement appropriate security measures to protect your personal information against unauthorized access, alteration, disclosure, or destruction.</p>

<h2>5. Your Rights</h2>
<p>Under GDPR and other privacy laws, you have rights regarding your personal data, including the right to access, correct, delete, or port your data.</p>

<h2>6. Cookies and Tracking</h2>
<p>We use cookies and similar technologies to enhance your experience and analyze usage patterns. You can control cookie preferences in your browser settings.</p>

<h2>7. Data Retention</h2>
<p>We retain your personal information for as long as necessary to provide our services and comply with legal obligations.</p>

<h2>8. International Transfers</h2>
<p>Your information may be transferred to and processed in countries other than your own, with appropriate safeguards in place.</p>

<h2>9. Children's Privacy</h2>
<p>Our services are not intended for children under 13, and we do not knowingly collect personal information from children under 13.</p>

<h2>10. Changes to This Policy</h2>
<p>We may update this Privacy Policy from time to time. We will notify you of any changes by posting the new policy on this page.</p>

<h2>11. Contact Us</h2>
<p>If you have questions about this Privacy Policy, please contact us at privacy@scrollintel.com</p>
            """,
            "effective_date": datetime.utcnow(),
            "document_metadata": {"language": "en", "gdpr_compliant": True}
        },
        {
            "document_type": "cookie_policy",
            "version": "1.0",
            "title": "ScrollIntel Cookie Policy",
            "content": """
<h2>1. What Are Cookies</h2>
<p>Cookies are small text files that are placed on your computer or mobile device when you visit our website.</p>

<h2>2. Types of Cookies We Use</h2>

<h3>Necessary Cookies</h3>
<p>These cookies are essential for the website to function properly. They enable basic functions like page navigation and access to secure areas.</p>

<h3>Analytics Cookies</h3>
<p>These cookies help us understand how visitors interact with our website by collecting and reporting information anonymously.</p>

<h3>Marketing Cookies</h3>
<p>These cookies are used to deliver advertisements more relevant to you and your interests.</p>

<h3>Preference Cookies</h3>
<p>These cookies remember your preferences and settings to provide a more personalized experience.</p>

<h2>3. Managing Cookies</h2>
<p>You can control and manage cookies in various ways. Please note that removing or blocking cookies can impact your user experience.</p>

<h2>4. Third-Party Cookies</h2>
<p>We may use third-party services that place cookies on your device. These services have their own privacy policies.</p>

<h2>5. Cookie Consent</h2>
<p>By using our website, you consent to our use of cookies in accordance with this policy. You can withdraw consent at any time.</p>

<h2>6. Updates to This Policy</h2>
<p>We may update this Cookie Policy from time to time to reflect changes in our practices or for other operational, legal, or regulatory reasons.</p>

<h2>7. Contact Us</h2>
<p>If you have any questions about our use of cookies, please contact us at privacy@scrollintel.com</p>
            """,
            "effective_date": datetime.utcnow(),
            "document_metadata": {"language": "en", "cookie_types": ["necessary", "analytics", "marketing", "preferences"]}
        }
    ]
    
    try:
        db = SessionLocal()
        try:
            for doc_data in documents:
                # Check if document already exists
                existing = db.query(LegalDocument).filter(
                    LegalDocument.document_type == doc_data["document_type"],
                    LegalDocument.version == doc_data["version"]
                ).first()
                
                if not existing:
                    document = LegalDocument(**doc_data)
                    db.add(document)
            
            db.commit()
            print("Legal documents seeded successfully")
        finally:
            db.close()
    
    except Exception as e:
        print(f"Error seeding legal documents: {e}")

if __name__ == "__main__":
    print("Creating legal compliance migration...")
    migration_file = create_legal_compliance_migration()
    
    print("Seeding legal documents...")
    seed_legal_documents()
    
    print("Legal compliance setup completed!")
    print(f"Migration file created: {migration_file}")
    print("Run 'alembic upgrade head' to apply the migration to your database.")