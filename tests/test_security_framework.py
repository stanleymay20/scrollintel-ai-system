"""
Comprehensive tests for the Security and Compliance Framework
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import json
import os
import tempfile

from scrollintel.security.security_framework import 