"""
ScrollIntel API Module
Contains FastAPI routes, middleware, and API-related components.
"""

from ..core.interfaces import AgentRequest, AgentResponse
from .gateway import ScrollIntelGateway, create_app, get_application
from .main import app

__all__ = [
    "AgentRequest",
    "AgentResponse",
    "ScrollIntelGateway",
    "create_app",
    "get_application",
    "app"
]