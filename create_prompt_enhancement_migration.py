#!/usr/bin/env python3
"""
Migration script for prompt enhancement database schema.
Creates tables for storing successful prompt patterns, templates, and A/B testing data.
"""

import os
import sys
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scrollintel.models.database import Base
from scrollintel.models.prompt_enhancement_models import (
    VisualPromptTemplate, VisualPromptPattern, VisualPromptVariation, 
    VisualABTestExperiment, VisualABTestResult, VisualPromptCategory,
    VisualPromptUsageLog, VisualPromptOptimizationSuggestion
)

def create_prompt_enhancement_tables():
    """Create all prompt enhancement related tables."""
    print("Creating prompt enhancement database tables...")
    
    try:
        # Use SQLite for simplicity
        database_url = os.getenv("DATABASE_URL", "sqlite:///./scrollintel.db")
        engine = create_engine(database_url)
        
        # Create all tables defined in the models
        Base.metadata.create_all(engine, tables=[
            VisualPromptTemplate.__table__,
            VisualPromptPattern.__table__,
            VisualPromptVariation.__table__,
            VisualABTestExperiment.__table__,
            VisualABTestResult.__table__,
            VisualPromptCategory.__table__,
            VisualPromptUsageLog.__table__,
            VisualPromptOptimizationSuggestion.__table__
        ])
        
        print("✓ Successfully created prompt enhancement tables")
        return True, engine
        
    except Exception as e:
        print(f"✗ Error creating tables: {str(e)}")
        return False, None

def seed_prompt_templates(engine):
    """Seed the database with common prompt templates."""
    print("Seeding prompt enhancement database with common templates...")
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Create default categories
        categories = [
            VisualPromptCategory(
                name="Photorealistic",
                description="Templates for photorealistic image generation",
                is_active=True
            ),
            VisualPromptCategory(
                name="Artistic",
                description="Templates for artistic and stylized content",
                is_active=True
            ),
            VisualPromptCategory(
                name="Portrait",
                description="Templates for portrait generation",
                is_active=True
            ),
            VisualPromptCategory(
                name="Landscape",
                description="Templates for landscape and scenery",
                is_active=True
            ),
            VisualPromptCategory(
                name="Abstract",
                description="Templates for abstract and conceptual art",
                is_active=True
            )
        ]
        
        for category in categories:
            existing = session.query(VisualPromptCategory).filter_by(name=category.name).first()
            if not existing:
                session.add(category)
        
        session.commit()
        
        # Get category IDs for templates
        photo_cat = session.query(VisualPromptCategory).filter_by(name="Photorealistic").first()
        artistic_cat = session.query(VisualPromptCategory).filter_by(name="Artistic").first()
        portrait_cat = session.query(VisualPromptCategory).filter_by(name="Portrait").first()
        landscape_cat = session.query(VisualPromptCategory).filter_by(name="Landscape").first()
        
        # Create common prompt templates
        templates = [
            VisualPromptTemplate(
                name="Professional Portrait",
                template="professional portrait of {subject}, {lighting} lighting, {background} background, high resolution, detailed, photorealistic",
                description="Template for professional portrait photography",
                category_id=portrait_cat.id,
                parameters=["subject", "lighting", "background"],
                success_rate=0.85,
                usage_count=0,
                is_active=True,
                created_by="system"
            ),
            VisualPromptTemplate(
                name="Cinematic Landscape",
                template="cinematic {time_of_day} landscape of {location}, {weather} weather, {style} photography, ultra-wide shot, dramatic lighting, 4K resolution",
                description="Template for cinematic landscape photography",
                category_id=landscape_cat.id,
                parameters=["time_of_day", "location", "weather", "style"],
                success_rate=0.78,
                usage_count=0,
                is_active=True,
                created_by="system"
            ),
            VisualPromptTemplate(
                name="Artistic Style Transfer",
                template="{subject} in the style of {artist}, {medium} medium, {color_palette} colors, {composition} composition, masterpiece quality",
                description="Template for artistic style transfer",
                category_id=artistic_cat.id,
                parameters=["subject", "artist", "medium", "color_palette", "composition"],
                success_rate=0.82,
                usage_count=0,
                is_active=True,
                created_by="system"
            ),
            VisualPromptTemplate(
                name="Product Photography",
                template="professional product photography of {product}, {background} background, {lighting} lighting, commercial quality, high detail, sharp focus",
                description="Template for product photography",
                category_id=photo_cat.id,
                parameters=["product", "background", "lighting"],
                success_rate=0.88,
                usage_count=0,
                is_active=True,
                created_by="system"
            )
        ]
        
        for template in templates:
            existing = session.query(VisualPromptTemplate).filter_by(name=template.name).first()
            if not existing:
                session.add(template)
        
        session.commit()
        
        # Create successful prompt patterns
        patterns = [
            VisualPromptPattern(
                pattern_text="high resolution, detailed, photorealistic",
                pattern_type="quality_enhancer",
                success_rate=0.92,
                usage_count=0,
                context="photorealistic images",
                effectiveness_score=9.2
            ),
            VisualPromptPattern(
                pattern_text="cinematic lighting, dramatic shadows",
                pattern_type="lighting_enhancer",
                success_rate=0.87,
                usage_count=0,
                context="dramatic scenes",
                effectiveness_score=8.7
            ),
            VisualPromptPattern(
                pattern_text="ultra-wide shot, panoramic view",
                pattern_type="composition_enhancer",
                success_rate=0.83,
                usage_count=0,
                context="landscape photography",
                effectiveness_score=8.3
            ),
            VisualPromptPattern(
                pattern_text="masterpiece quality, award-winning",
                pattern_type="quality_enhancer",
                success_rate=0.89,
                usage_count=0,
                context="artistic content",
                effectiveness_score=8.9
            )
        ]
        
        for pattern in patterns:
            existing = session.query(VisualPromptPattern).filter_by(pattern_text=pattern.pattern_text).first()
            if not existing:
                session.add(pattern)
        
        session.commit()
        print("✓ Successfully seeded prompt enhancement database")
        return True
        
    except Exception as e:
        session.rollback()
        print(f"✗ Error seeding database: {str(e)}")
        return False
    finally:
        session.close()

def main():
    """Main migration function."""
    print("Starting prompt enhancement database migration...")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Create tables
    success, engine = create_prompt_enhancement_tables()
    if not success:
        sys.exit(1)
    
    # Seed with common templates
    if not seed_prompt_templates(engine):
        sys.exit(1)
    
    print("✓ Prompt enhancement database migration completed successfully!")

if __name__ == "__main__":
    main()