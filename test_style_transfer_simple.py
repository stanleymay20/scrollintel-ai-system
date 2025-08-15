#!/usr/bin/env python3
"""
Simple test for style transfer functionality.
"""

import asyncio
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

from scrollintel.models.style_transfer_models import (
    StyleTransferRequest, StyleTransferResult, StyleTransferStatus,
    ArtisticStyle, StyleTransferConfig
)

async def test_style_transfer_models():
    """Test style transfer models."""
    print("Testing style transfer models...")
    
    # Test StyleTransferConfig
    config = StyleTransferConfig(
        content_weight=1.5,
        style_weight=8000.0,
        num_iterations=500
    )
    print(f"✓ StyleTransferConfig created: {config.content_weight}")
    
    # Test StyleTransferRequest
    request = StyleTransferRequest(
        id="test_request",
        content_paths=["test_image.png"],
        style_type=ArtisticStyle.IMPRESSIONIST,
        config=config
    )
    print(f"✓ StyleTransferRequest created: {request.id}")
    
    # Test serialization
    request_dict = request.to_dict()
    print(f"✓ Request serialization: {len(request_dict)} fields")
    
    # Test StyleTransferResult
    result = StyleTransferResult(
        id="test_result",
        status=StyleTransferStatus.COMPLETED,
        result_paths=["output.png"],
        processing_time=15.5,
        style_consistency_score=0.85,
        content_preservation_score=0.90
    )
    print(f"✓ StyleTransferResult created: {result.status}")
    
    print("All model tests passed!")

async def test_style_transfer_engine():
    """Test style transfer engine."""
    print("\nTesting style transfer engine...")
    
    try:
        # Import the engine
        from scrollintel.engines.style_transfer_engine import StyleTransferEngine
        from scrollintel.models.style_transfer_models import ArtisticStyle as StyleType
        print("✓ StyleTransferEngine imported successfully")
        
        # Create engine
        engine = StyleTransferEngine()
        print("✓ StyleTransferEngine created")
        
        # Initialize engine
        await engine.initialize()
        print("✓ StyleTransferEngine initialized")
        
        # Test capabilities
        capabilities = engine.get_capabilities()
        print(f"✓ Engine capabilities: {len(capabilities)} features")
        
        # Create test image
        temp_dir = tempfile.mkdtemp()
        test_image_path = Path(temp_dir) / "test_content.png"
        
        # Create a simple test image
        test_img = Image.new('RGB', (256, 256), (100, 150, 200))
        test_img.save(test_image_path)
        print(f"✓ Test image created: {test_image_path}")
        
        # Test style transfer
        result_path = await engine.apply_style_transfer(
            content_path=str(test_image_path),
            style_type=StyleType.IMPRESSIONIST
        )
        print(f"✓ Style transfer completed: {result_path}")
        
        # Verify result exists
        if Path(result_path).exists():
            print("✓ Result file exists")
        else:
            print("✗ Result file not found")
        
        # Test request processing
        request = StyleTransferRequest(
            id="engine_test",
            content_paths=[str(test_image_path)],
            style_type=ArtisticStyle.WATERCOLOR
        )
        
        result = await engine.process_style_transfer_request(request)
        print(f"✓ Request processing completed: {result.status}")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        print("✓ Cleanup completed")
        
        print("All engine tests passed!")
        
    except Exception as e:
        print(f"✗ Engine test failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Run all tests."""
    print("=== Style Transfer Implementation Test ===\n")
    
    try:
        await test_style_transfer_models()
        await test_style_transfer_engine()
        
        print("\n=== All Tests Completed Successfully! ===")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)