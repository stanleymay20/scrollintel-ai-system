#!/usr/bin/env python3
"""Test engine instantiation."""

import asyncio

async def test_instantiation():
    try:
        from scrollintel.engines.style_transfer_engine import StyleTransferEngine
        print("✓ StyleTransferEngine imported")
        
        engine = StyleTransferEngine()
        print("✓ StyleTransferEngine instantiated")
        
        await engine.initialize()
        print("✓ StyleTransferEngine initialized")
        
        capabilities = engine.get_capabilities()
        print(f"✓ Capabilities: {len(capabilities)} features")
        
        status = engine.get_status()
        print(f"✓ Status: {status['healthy']}")
        
        await engine.cleanup()
        print("✓ Cleanup completed")
        
        print("All tests passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_instantiation())