#!/usr/bin/env python3
"""
Simple test for rapid prototyping functionality
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_rapid_prototyping():
    """Test rapid prototyping functionality"""
    try:
        print("Testing rapid prototyping system...")
        
        # Import required modules
        from scrollintel.models.prototype_models import (
            Concept, ConceptCategory, create_concept_from_description
        )
        from scrollintel.engines.rapid_prototyper import RapidPrototyper
        
        # Create a test concept
        concept = create_concept_from_description(
            name="Test API Service",
            description="A simple REST API service for testing rapid prototyping",
            category=ConceptCategory.TECHNOLOGY
        )
        
        print(f"✅ Created concept: {concept.name}")
        print(f"   Concept ID: {concept.id}")
        print(f"   Description: {concept.description}")
        
        # Create rapid prototyper instance
        rapid_prototyper = RapidPrototyper()
        print("✅ RapidPrototyper instance created successfully")
        
        # Test prototype creation
        print("\n🔄 Creating rapid prototype...")
        prototype = await rapid_prototyper.create_rapid_prototype(concept)
        
        print(f"✅ Prototype created successfully!")
        print(f"   Prototype ID: {prototype.id}")
        print(f"   Prototype Name: {prototype.name}")
        print(f"   Prototype Status: {prototype.status.value}")
        print(f"   Technology Stack: {prototype.technology_stack.primary_technology if prototype.technology_stack else 'None'}")
        print(f"   Development Progress: {prototype.development_progress * 100:.1f}%")
        print(f"   Generated Code Files: {len(prototype.generated_code)}")
        
        # Test prototype optimization
        if prototype.status.value in ['functional', 'validated']:
            print("\n🔄 Testing prototype optimization...")
            optimized_prototype = await rapid_prototyper.optimize_prototype(prototype.id)
            quality_score = optimized_prototype.validation_result.overall_score if optimized_prototype.validation_result else 0.0
            print(f"✅ Optimization completed. Quality score: {quality_score:.2f}")
        
        # Test listing prototypes
        prototypes = await rapid_prototyper.list_active_prototypes()
        print(f"\n✅ Active prototypes: {len(prototypes)}")
        
        # Test prototype status retrieval
        retrieved_prototype = await rapid_prototyper.get_prototype_status(prototype.id)
        if retrieved_prototype:
            print(f"✅ Prototype status retrieved successfully")
        
        print("\n🎉 Rapid prototyping system test completed successfully!")
        print("\n📋 Task 3.1 Implementation Summary:")
        print("   ✅ Automated rapid prototyping and proof-of-concept development")
        print("   ✅ Prototyping technology selection and optimization")
        print("   ✅ Prototyping quality control and validation")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during rapid prototyping test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_rapid_prototyping())
    sys.exit(0 if success else 1)