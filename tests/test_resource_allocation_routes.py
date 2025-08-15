"""
Tests for resource allocation API routes.
"""
import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import json

from scrollintel.api.routes.resource_allocation_routes import router
fr