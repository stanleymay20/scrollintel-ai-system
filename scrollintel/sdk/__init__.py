"""
ScrollIntel Python SDK - Programmatic access to the Advanced Prompt Management System.
"""

from .prompt_client import PromptClient
from .exceptions import (
    ScrollIntelSDKError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    NotFoundError
)
from .models import (
    PromptTemplate,
    PromptVersion,
    PromptVariable,
    SearchQuery,
    APIResponse
)

__version__ = "1.0.0"
__all__ = [
    "PromptClient",
    "ScrollIntelSDKError",
    "AuthenticationError", 
    "RateLimitError",
    "ValidationError",
    "NotFoundError",
    "PromptTemplate",
    "PromptVersion",
    "PromptVariable",
    "SearchQuery",
    "APIResponse"
]