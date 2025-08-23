"""
Comprehensive tests for API Integration Framework
Tests REST, GraphQL, SOAP connectors, webhooks, and rate limiting.
"""

import asyncio
import json
import pytest
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from scrollintel.connectors.api_connector import (
    APIConnector, APIType, AuthType, APICredentials,
    RateLimitConfig, RetryConfig, APIEndpoint as ConnectorEndpoint,
    RESTConnector, GraphQLConnector, SOAPConnector
)
from scrollintel.connectors.webhook_manager import (
    WebhookManager, WebhookEvent, WebhookConfig, WebhookPayload
)


class TestAPIConnector:
    """Test API connector factory and base functionality"""
    
    def test_create_rest_connector(self):
        """Test creating REST connector"""
        credentials = APICredentials(auth_type=AuthType.NONE)
        connector = APIConnector.create_connecto