#!/usr/bin/env python3
"""
Create database migration for Advanced Prompt Management System tables.
"""

import os
import sys
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scrollintel.models.database import Base
from scrollintel.models.prompt_models import AdvancedPromptTemplate, AdvancedPromptVersion, AdvancedPromptCategory, AdvancedPromptTag
from scrollintel.core.config import get_config


def create_migration():
    """Create migration for Advanced Prompt Management System tables."""
    print("🚀 Creating Advanced Prompt Management System Migration")
    print("=" * 60)
    
    try:
        # Get database URL
        config = get_config()
        database_url = config.database_url
        print(f"Database URL: {database_url.split('@')[1] if '@' in database_url else 'local'}")
        
        # Create engine
        engine = create_engine(database_url)
        
        # Create session
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        print("\n📋 Creating Advanced Prompt Management tables...")
        
        # Create tables
        Base.metadata.create_all(bind=engine, tables=[
            AdvancedPromptTemplate.__table__,
            AdvancedPromptVersion.__table__,
            AdvancedPromptCategory.__table__,
            AdvancedPromptTag.__table__
        ])
        
        print("✅ Tables created successfully:")
        print("  • advanced_prompt_templates")
        print("  • advanced_prompt_versions")
        print("  • advanced_prompt_categories")
        print("  • advanced_prompt_tags")
        
        # Verify tables exist
        print("\n🔍 Verifying table creation...")
        session = SessionLocal()
        
        try:
            # Check if tables exist by querying them
            session.execute(text("SELECT COUNT(*) FROM advanced_prompt_templates"))
            session.execute(text("SELECT COUNT(*) FROM advanced_prompt_versions"))
            session.execute(text("SELECT COUNT(*) FROM advanced_prompt_categories"))
            session.execute(text("SELECT COUNT(*) FROM advanced_prompt_tags"))
            print("✅ All tables verified successfully")
            
        except Exception as e:
            print(f"⚠️ Table verification warning: {str(e)}")
        
        finally:
            session.close()
        
        # Create sample data
        print("\n📝 Creating sample data...")
        create_sample_data(SessionLocal)
        
        print("\n🎉 Advanced Prompt Management migration completed successfully!")
        print("\nNext steps:")
        print("1. Run the demo: python demo_prompt_management.py")
        print("2. Test the API endpoints")
        print("3. Start using advanced prompt management features")
        
    except Exception as e:
        print(f"❌ Migration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def create_sample_data(SessionLocal):
    """Create sample prompt templates and data."""
    session = SessionLocal()
    
    try:
        # Sample categories
        categories = [
            AdvancedPromptCategory(
                name="Business Communication",
                description="Templates for business and professional communication"
            ),
            AdvancedPromptCategory(
                name="Data Analysis",
                description="Templates for data analysis and reporting"
            ),
            AdvancedPromptCategory(
                name="Creative Writing",
                description="Templates for creative and content writing"
            ),
            AdvancedPromptCategory(
                name="Technical Documentation",
                description="Templates for technical documentation and guides"
            )
        ]
        
        for category in categories:
            session.add(category)
        
        # Sample tags
        tags = [
            AdvancedPromptTag(name="professional", description="Professional tone", color="#0066CC"),
            AdvancedPromptTag(name="casual", description="Casual tone", color="#00AA44"),
            AdvancedPromptTag(name="technical", description="Technical content", color="#FF6600"),
            AdvancedPromptTag(name="creative", description="Creative content", color="#AA00FF"),
            AdvancedPromptTag(name="analysis", description="Data analysis", color="#FF0066"),
            AdvancedPromptTag(name="reporting", description="Report generation", color="#666666")
        ]
        
        for tag in tags:
            session.add(tag)
        
        session.flush()  # Get IDs for categories
        
        # Sample prompt templates
        templates = [
            AdvancedPromptTemplate(
                name="Business Email Template",
                content="Subject: {{subject}}\n\nDear {{recipient_name}},\n\n{{message_body}}\n\nBest regards,\n{{sender_name}}",
                category="Business Communication",
                tags=["professional", "email"],
                variables=[
                    {"name": "subject", "type": "string", "required": True, "description": "Email subject line"},
                    {"name": "recipient_name", "type": "string", "required": True, "description": "Recipient's name"},
                    {"name": "message_body", "type": "string", "required": True, "description": "Main message content"},
                    {"name": "sender_name", "type": "string", "required": True, "description": "Sender's name"}
                ],
                description="Professional business email template",
                created_by="system"
            ),
            AdvancedPromptTemplate(
                name="Data Analysis Report",
                content="# {{report_title}}\n\n## Executive Summary\n{{executive_summary}}\n\n## Key Findings\n{{key_findings}}\n\n## Methodology\n{{methodology}}\n\n## Recommendations\n{{recommendations}}",
                category="Data Analysis",
                tags=["analysis", "reporting", "technical"],
                variables=[
                    {"name": "report_title", "type": "string", "required": True, "description": "Report title"},
                    {"name": "executive_summary", "type": "string", "required": True, "description": "Executive summary"},
                    {"name": "key_findings", "type": "string", "required": True, "description": "Key findings"},
                    {"name": "methodology", "type": "string", "required": True, "description": "Analysis methodology"},
                    {"name": "recommendations", "type": "string", "required": True, "description": "Recommendations"}
                ],
                description="Comprehensive data analysis report template",
                created_by="system"
            ),
            AdvancedPromptTemplate(
                name="Creative Story Starter",
                content="In the {{setting}} of {{location}}, {{protagonist_name}} discovered something that would change everything. {{opening_scene}}\n\nThe {{tone}} adventure that followed would test {{protagonist_name}}'s {{character_trait}} like never before.",
                category="Creative Writing",
                tags=["creative", "storytelling"],
                variables=[
                    {"name": "setting", "type": "string", "required": True, "description": "Time period or setting"},
                    {"name": "location", "type": "string", "required": True, "description": "Physical location"},
                    {"name": "protagonist_name", "type": "string", "required": True, "description": "Main character's name"},
                    {"name": "opening_scene", "type": "string", "required": True, "description": "Opening scene description"},
                    {"name": "tone", "type": "string", "required": False, "default": "thrilling", "description": "Story tone"},
                    {"name": "character_trait", "type": "string", "required": True, "description": "Key character trait"}
                ],
                description="Creative story starter template for fiction writing",
                created_by="system"
            )
        ]
        
        for template in templates:
            session.add(template)
            session.flush()  # Get the template ID
            
            # Create initial version for each template
            initial_version = AdvancedPromptVersion(
                prompt_id=template.id,
                version="1.0.0",
                content=template.content,
                variables=template.variables,
                tags=template.tags,
                changes="Initial version",
                created_by="system"
            )
            session.add(initial_version)
        
        session.commit()
        
        # Print statistics
        template_count = session.query(AdvancedPromptTemplate).count()
        version_count = session.query(AdvancedPromptVersion).count()
        category_count = session.query(AdvancedPromptCategory).count()
        tag_count = session.query(AdvancedPromptTag).count()
        
        print(f"📊 Sample data created:")
        print(f"  • Templates: {template_count}")
        print(f"  • Versions: {version_count}")
        print(f"  • Categories: {category_count}")
        print(f"  • Tags: {tag_count}")
        
    except Exception as e:
        session.rollback()
        print(f"❌ Error creating sample data: {str(e)}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    create_migration()