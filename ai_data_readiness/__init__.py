"""
AI Data Readiness Platform

A comprehensive system for transforming raw data into AI-ready datasets through
automated assessment, preparation, and continuous monitoring.
"""

__version__ = "1.0.0"
__author__ = "AI Data Readiness Team"

from .core.config import Config
from .models.database import Database

__all__ = ["Config", "Database"]