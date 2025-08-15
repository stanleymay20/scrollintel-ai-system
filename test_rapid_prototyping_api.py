#!/usr/bin/env python3
"""
Test for rapid prototyping API routes
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_rapid_prototyping_api():
    """Test rapid prototyping API functionality"""
    try:
        print("Testing rapid prototyping API routes...")
        
        # Import required modules
        from scrollintel.models.prototype_models import (
            ConceptCategory, create_concept_from_description
        )
        from scrollintel.api.routes.rapid_prototyping_routes import (
            ConceptCreateRequest, PrototypeCreateRequest, rapid_prototyper
        )
        
        # Test concept creation request model
        concept_request = ConceptCreateRequest(
            name="API Test Concept",
            description="A test concept for API validation",
            category=ConceptCategory.TECHNOLOGY,
            requirements=["Fast response", "Scalable"],
            technical_complexity=0.7,
            innovation_potential=0.8
        )
        
        print(f"✅ Created concept request: {concept_request.name}")
        
        # Test prototype creation request model
        prototype_request = PrototypeCreateRequest(
            concept_id="test-concept-id",
            priority="high",
            custom_requirements=["Performance optimization"]
        )
        
        print(f"✅ Created prototype request for concept: {prototype_request.concept_id}")
        
        # Test the global rapid prototyper instance
        prototypes = await rapid_prototyper.list_active_prototypes()
        print(f"✅ API rapid prototyper instance working. Active prototypes: {len(prototypes)}")
        
        print("\n🎉 Rapid prototyping API test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during API test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_rapid_prototyping_api())
    sys.exit(0 if success else 1)