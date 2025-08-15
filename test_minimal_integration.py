"""Minimal test of the integration system."""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test basic imports first
try:
    from scrollintel.core.failure_prevention import FailureEvent, FailureType
    print("✓ FailureEvent and FailureType imported")
except Exception as e:
    print(f"✗ Error importing from failure_prevention: {e}")
    exit(1)

try:
    from scrollintel.core.user_experience_protection import UserExperienceLevel
    print("✓ UserExperienceLevel imported")
except Exception as e:
    print(f"✗ Error importing from user_experience_protection: {e}")
    exit(1)

# Now try to create a minimal integration class
class MinimalFailureUXIntegrator:
    """Minimal version for testing."""
    
    def __init__(self):
        print("Initializing minimal integrator...")
        self.test_data = {"initialized": True}
    
    async def handle_unified_failure(self, failure_event: FailureEvent) -> Dict[str, Any]:
        """Handle failure with minimal processing."""
        print(f"Handling failure: {failure_event.failure_type.value}")
        return {
            "handled": True,
            "failure_type": failure_event.failure_type.value,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get status."""
        return self.test_data

# Test the minimal class
try:
    print("Creating minimal integrator...")
    integrator = MinimalFailureUXIntegrator()
    print("✓ Minimal integrator created")
    
    # Test handling a failure
    failure_event = FailureEvent(
        failure_type=FailureType.NETWORK_ERROR,
        timestamp=datetime.utcnow(),
        error_message="Test error",
        stack_trace="",
        context={"test": True}
    )
    
    async def test_failure_handling():
        result = await integrator.handle_unified_failure(failure_event)
        print(f"✓ Failure handled: {result}")
    
    asyncio.run(test_failure_handling())
    
    print("✓ All minimal tests passed")
    
except Exception as e:
    print(f"✗ Error with minimal integrator: {e}")
    import traceback
    traceback.print_exc()