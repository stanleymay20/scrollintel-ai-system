"""
API Documentation and Integration Examples for Visual Generation

Provides comprehensive documentation with:
- OpenAPI schema customization
- Interactive examples
- Code samples in multiple languages
- Integration guides
- Best practices
"""

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from typing import Dict, Any, List
import json


class VisualGenerationDocs:
    """Documentation generator for Visual Generation API."""
    
    def __init__(self, app: FastAPI):
        self.app = app
    
    def customize_openapi_schema(self) -> Dict[str, Any]:
        """Customize OpenAPI schema with enhanced documentation."""
        if self.app.openapi_schema:
            return self.app.openapi_schema
        
        openapi_schema = get_openapi(
            title="ScrollIntel Visual Generation API",
            version="1.0.0",
            description=s