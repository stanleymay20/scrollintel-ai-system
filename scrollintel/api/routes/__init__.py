"""
API Routes module for ScrollIntel.
Contains all route handlers organized by functionality.
"""

from . import agent_routes, auth_routes, health_routes, admin_routes, file_routes

__all__ = [
    "agent_routes",
    "auth_routes", 
    "health_routes",
    "admin_routes",
    "file_routes"
]