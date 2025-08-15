#!/usr/bin/env python3
"""
Create database migration for ScrollAutoDev prompt engineering tables.
Adds PromptTemplate, PromptHistory, and PromptTest tables.
"""

import os
import sys
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scrollintel.models.database import Base, PromptTemplate, PromptHistory, PromptTest
from scrollintel.core.config import get_database_url


def create_migration():
    """Create migration for ScrollAutoDev tables."""
    print("🚀 Creating ScrollAutoDev Database Migration")
    print("=" * 50)
    
    try:
        # Get database URL
        database_url = get_database_url()
        print(f"Database URL: {database_url.split('@')[1] if '@' in database_url else 'local'}")
        
        # Create engine
        engine = create_engine(database_url)
        
        # Create session
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        print("\n📋 Creating ScrollAutoDev tables...")
        
        # Create tables
        Base.metadata.create_all(bind=engine, tables=[
            PromptTemplate.__table__,
            PromptHistory.__table__,
            PromptTest.__table__
        ])
        
        print("✅ Tables created successfully:")
        print("  • prompt_templates")
        print("  • prompt_history") 
        print("  • prompt_tests")
        
        # Verify tables exist
        print("\n🔍 Verifying table creation...")
        session = SessionLocal()
        
        try:
            # Check if tables exist by querying them
            session.execute(text("SELECT COUNT(*) FROM prompt_templates"))
            session.execute(text("SELECT COUNT(*) FROM prompt_history"))
            session.execute(text("SELECT COUNT(*) FROM prompt_tests"))
            print("✅ All tables verified successfully")
            
        except Exception as e:
            print(f"⚠️ Table verification warning: {str(e)}")
        
        finally:
            session.close()
        
        # Create sample data
        print("\n📝 Creating sample prompt templates...")
        create_sample_data(SessionLocal)
        
        print("\n🎉 ScrollAutoDev migration completed successfully!")
        print("\nNext steps:")
        print("1. Run the demo: python demo_scroll_autodev.py")
        print("2. Test the API endpoints")
        print("3. Start using prompt optimization features")
        
    except Exception as e:
        print(f"❌ Migration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def create_sample_data(SessionLocal):
    """Create sample prompt templates and data."""
    session = SessionLocal()
    
    try:
        # Sample prompt templates
        templates = [
            PromptTemplate(
                name="Data Analysis Template",
                description="General purpose data analysis template",
                category="data_analysis",
                industry="general",
                use_case="data_analysis",
                template_content="Analyze the following dataset {{data}} and provide insights on {{focus_area}}. Include statistical analysis, trends, and actionable recommendations.",
                variables=["data", "focus_area"],
                tags=["analysis", "statistics", "insights"],
                is_public=True,
                creator_id="00000000-0000-0000-0000-000000000000"  # System user
            ),
            PromptTemplate(
                name="Code Generation Template",
                description="Template for generating code with best practices",
                category="code_generation",
                industry="technology",
                use_case="software_development",
                template_content="Generate {{language}} code that {{requirement}}. Include error handling, documentation, and follow best practices for {{framework}}.",
                variables=["language", "requirement", "framework"],
                tags=["code", "programming", "development"],
                is_public=True,
                creator_id="00000000-0000-0000-0000-000000000000"
            ),
            PromptTemplate(
                name="Business Intelligence Template",
                description="Template for business intelligence and KPI analysis",
                category="business_intelligence",
                industry="business",
                use_case="kpi_analysis",
                template_content="Analyze the business metrics {{metrics}} for {{time_period}}. Calculate KPIs, identify trends, and provide strategic recommendations for {{business_area}}.",
                variables=["metrics", "time_period", "business_area"],
                tags=["business", "kpi", "metrics", "strategy"],
                is_public=True,
                creator_id="00000000-0000-0000-0000-000000000000"
            ),
            PromptTemplate(
                name="Healthcare Analysis Template",
                description="HIPAA-compliant healthcare data analysis template",
                category="data_analysis",
                industry="healthcare",
                use_case="patient_analysis",
                template_content="Analyze the healthcare data {{patient_data}} while maintaining HIPAA compliance. Focus on {{clinical_focus}} and provide medical insights for {{treatment_area}}. Ensure all analysis follows medical best practices.",
                variables=["patient_data", "clinical_focus", "treatment_area"],
                tags=["healthcare", "HIPAA", "medical", "clinical"],
                is_public=True,
                creator_id="00000000-0000-0000-0000-000000000000"
            ),
            PromptTemplate(
                name="Financial Risk Assessment Template",
                description="Template for financial risk analysis and assessment",
                category="business_intelligence",
                industry="finance",
                use_case="risk_assessment",
                template_content="Conduct risk assessment for {{financial_data}} focusing on {{risk_type}}. Analyze exposure levels, calculate risk metrics, and provide mitigation strategies for {{portfolio_type}}.",
                variables=["financial_data", "risk_type", "portfolio_type"],
                tags=["finance", "risk", "assessment", "compliance"],
                is_public=True,
                creator_id="00000000-0000-0000-0000-000000000000"
            ),
            PromptTemplate(
                name="Customer Behavior Analysis Template",
                description="Template for analyzing customer behavior and patterns",
                category="data_analysis",
                industry="retail",
                use_case="customer_analysis",
                template_content="Analyze customer behavior data {{customer_data}} for {{analysis_period}}. Identify patterns, segments, and trends in {{behavior_type}}. Provide recommendations for {{optimization_goal}}.",
                variables=["customer_data", "analysis_period", "behavior_type", "optimization_goal"],
                tags=["customer", "behavior", "retail", "segmentation"],
                is_public=True,
                creator_id="00000000-0000-0000-0000-000000000000"
            ),
            PromptTemplate(
                name="Strategic Planning Template",
                description="Template for strategic planning and decision making",
                category="strategic_planning",
                industry="general",
                use_case="strategic_planning",
                template_content="Develop a strategic plan for {{objective}} considering {{constraints}} and available {{resources}}. Include timeline, milestones, risk assessment, and success metrics for {{planning_horizon}}.",
                variables=["objective", "constraints", "resources", "planning_horizon"],
                tags=["strategy", "planning", "decision", "roadmap"],
                is_public=True,
                creator_id="00000000-0000-0000-0000-000000000000"
            ),
            PromptTemplate(
                name="Technical Documentation Template",
                description="Template for creating comprehensive technical documentation",
                category="technical_documentation",
                industry="technology",
                use_case="documentation",
                template_content="Create technical documentation for {{system_component}} including {{documentation_type}}. Cover architecture, implementation details, usage examples, and {{target_audience}} guidelines.",
                variables=["system_component", "documentation_type", "target_audience"],
                tags=["documentation", "technical", "architecture", "guide"],
                is_public=True,
                creator_id="00000000-0000-0000-0000-000000000000"
            )
        ]
        
        # Add templates to session
        for template in templates:
            session.add(template)
        
        # Commit changes
        session.commit()
        
        print(f"✅ Created {len(templates)} sample prompt templates")
        
        # Display created templates
        print("\nSample templates created:")
        for template in templates:
            print(f"  • {template.name} ({template.category})")
        
    except Exception as e:
        session.rollback()
        print(f"⚠️ Error creating sample data: {str(e)}")
    
    finally:
        session.close()


def verify_migration():
    """Verify the migration was successful."""
    print("\n🔍 Verifying migration...")
    
    try:
        database_url = get_database_url()
        engine = create_engine(database_url)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = SessionLocal()
        
        # Check table counts
        template_count = session.query(PromptTemplate).count()
        history_count = session.query(PromptHistory).count()
        test_count = session.query(PromptTest).count()
        
        print(f"📊 Table statistics:")
        print(f"  • prompt_templates: {template_count} records")
        print(f"  • prompt_history: {history_count} records")
        print(f"  • prompt_tests: {test_count} records")
        
        # Test basic operations
        if template_count > 0:
            sample_template = session.query(PromptTemplate).first()
            print(f"  • Sample template: '{sample_template.name}'")
        
        session.close()
        print("✅ Migration verification completed")
        
    except Exception as e:
        print(f"❌ Verification failed: {str(e)}")


if __name__ == "__main__":
    print("ScrollAutoDev Database Migration")
    print("This will create tables for prompt engineering features")
    
    # Confirm before proceeding
    if len(sys.argv) > 1 and sys.argv[1] == "--force":
        proceed = True
    else:
        proceed = input("\nProceed with migration? (y/N): ").lower().startswith('y')
    
    if proceed:
        create_migration()
        verify_migration()
    else:
        print("Migration cancelled")