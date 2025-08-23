"""
Compliance Reporting Engine
Automated compliance reporting and governance for various frameworks
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
from scrollintel.models.security_audit_models import (
    AuditLog, ComplianceReport, SecurityAlert, SecurityMetrics
)
from scrollintel.core.config import get_database_session
import logging

logger = logging.getLogger(__name__)

class ComplianceReportingEngine:
    """Engine for automated compliance reporting and governance"""
    
    def __init__(self):
        self.compliance_frameworks = {
            'SOX': 