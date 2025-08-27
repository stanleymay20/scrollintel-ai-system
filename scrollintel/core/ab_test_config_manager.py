"""
A/B Test Configuration Management System.
Provides comprehensive configuration management for A/B testing experiments
including templates, validation, and automated configuration generation.
"""

import json
import yaml
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Exception raised for configuration validation errors."""
    pass


class ScheduleType(Enum):
    """Types of experiment schedules."""
    MANUAL = "manual"
    DAIL