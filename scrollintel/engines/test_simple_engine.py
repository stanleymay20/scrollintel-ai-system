"""Simple test engine."""

from .base_engine import BaseEngine, EngineCapability

class SimpleEngine(BaseEngine):
    """Simple test engine."""
    
    def __init__(self):
        super().__init__("simple", "Simple Engine", [])
    
    async def process(self, input_data, parameters=None):
        return "processed"