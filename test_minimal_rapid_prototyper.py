#!/usr/bin/env python3
"""
Minimal test for rapid prototyping functionality
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
        
        # Create a test concept
        concept = create_concept_from_description(
            name="Test API Service",
            description="A simple REST API service for testing rapid prototyping",
            category=ConceptCategory.TECHNOLOGY
        )
        
        print(f"Created concept: {concept.name}")
        print(f"Concept ID: {concept.id}")
        print(f"Description: {concept.description}")
        
        # Try to import and use the rapid prototyper
        # Since direct import is failing, let's load the module manually
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "rapid_prototyper", 
            "scrollintel/engines/rapid_prototyper.py"
        )
        rapid_prototyper_module = importlib.util.module_from_spec(spec)
        
        # Execute the module
        spec.loader.exec_module(rapid_prototyper_module)
        
        # Get the RapidPrototyper class
        RapidPrototyper = rapid_prototyper_module.RapidPrototyper
        
        # Create rapid prototyper instance
        rapid_prototyper = RapidPrototyper()
        
        print("RapidPrototyper instance created successfully")
        
        # Test prototype creation
        prototype = await rapid_prototyper.create_rapid_prototype(concept)
        
        print(f"Prototype created successfully!")
        print(f"Prototype ID: {prototype.id}")
        print(f"Prototype Name: {prototype.name}")
        print(f"Prototype Status: {prototype.status}")
        print(f"Technology Stack: {prototype.technology_stack.primary_technology if prototype.technology_stack else 'None'}")
        print(f"Development Progress: {prototype.development_progress * 100:.1f}%")
        
        # Test prototype optimization
        if prototype.status.value in ['functional', 'validated']:
            print("\nTesting prototype optimization...")
            optimized_prototype = await rapid_prototyper.optimize_prototype(prototype.id)
            print(f"Optimization completed. Quality score: {optimized_prototype.validation_result.overall_score:.2f}")
        
        # Test listing prototypes
        prototypes = await rapid_prototyper.list_active_prototypes()
        print(f"\nActive prototypes: {len(prototypes)}")
        
        print("\n✅ Rapid prototyping system test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during rapid prototyping test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_rapid_prototyping())
    sys.exit(0 if success else 1)