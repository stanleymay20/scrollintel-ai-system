"""
Industry-Tailored Compliance Modules
"""

from .industry_compliance_framework import (
    IndustryComplianceOrchestrator,
    BaseComplianceModule,
    ComplianceRule,
    ComplianceAssessment,
    ComplianceStatus,
    IndustryType
)

from .banking_compliance import BankingComplianceModule
from .healthcare_compliance import HealthcareComplianceModule
from .manufacturing_compliance import ManufacturingComplianceModule
from .government_compliance import GovernmentComplianceModule
from .financial_services_compliance import FinancialServicesComplianceModule
from .pharmaceutical_compliance import PharmaceuticalComplianceModule

__all__ = [
    'IndustryComplianceOrchestrator',
    'BaseComplianceModule',
    'ComplianceRule',
    'ComplianceAssessment',
    'ComplianceStatus',
    'IndustryType',
    'BankingComplianceModule',
    'HealthcareComplianceModule',
    'ManufacturingComplianceModule',
    'GovernmentComplianceModule',
    'FinancialServicesComplianceModule',
    'PharmaceuticalComplianceModule'
]