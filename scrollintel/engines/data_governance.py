"""
Data Governance and Privacy Controls Engine

This module provides comprehensive data governance and privacy controls
for ensuring regulatory compliance and data protection.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
import hashlib
import re
import logging
from enum import Enum

from scrollintel.models.lineage_models import (
    DataGovernancePolicy, ComplianceRule, ComplianceRuleType
)
from scrollintel.models.database_utils import get_sync_db

logger = logging.getLogger(__name__)


class DataClassification(str, Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class PrivacyLevel(str, Enum):
    """Privacy protection levels"""
    NONE = "none"
    BASIC = "basic"
    ENHANCED = "enhanced"
    MAXIMUM = "maximum"


class DataGovernanceEngine:
    """Comprehensive data governance and privacy controls system"""
    
    def __init__(self):
        self.session = next(get_sync_db())
        self.classification_rules = {}
        self.privacy_policies = {}
        self.retention_policies = {}
        self._load_governance_policies()
    
    def classify_data(
        self,
        dataset_id: str,
        data_sample: Dict[str, Any],
        schema_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Automatically classify data based on content and schema
        
        Args:
            dataset_id: Dataset identifier
            data_sample: Sample of data for analysis
            schema_info: Schema information
            
        Returns:
            Dict: Classification results
        """
        try:
            classification_result = {
                "dataset_id": dataset_id,
                "classification": DataClassification.INTERNAL,
                "privacy_level": PrivacyLevel.BASIC,
                "sensitive_fields": [],
                "compliance_requirements": [],
                "recommended_controls": [],
                "confidence_score": 0.0
            }
            
            # Analyze data content
            sensitive_patterns = self._detect_sensitive_patterns(data_sample)
            classification_result["sensitive_fields"] = sensitive_patterns
            
            # Determine classification level
            classification_level = self._determine_classification_level(
                sensitive_patterns, schema_info
            )
            classification_result["classification"] = classification_level
            
            # Determine privacy level
            privacy_level = self._determine_privacy_level(
                classification_level, sensitive_patterns
            )
            classification_result["privacy_level"] = privacy_level
            
            # Identify compliance requirements
            compliance_reqs = self._identify_compliance_requirements(
                sensitive_patterns, classification_level
            )
            classification_result["compliance_requirements"] = compliance_reqs
            
            # Generate recommended controls
            controls = self._generate_recommended_controls(
                classification_level, privacy_level, compliance_reqs
            )
            classification_result["recommended_controls"] = controls
            
            # Calculate confidence score
            classification_result["confidence_score"] = self._calculate_confidence_score(
                sensitive_patterns, schema_info
            )
            
            return classification_result
            
        except Exception as e:
            logger.error(f"Error classifying data: {str(e)}")
            raise
    
    def enforce_data_governance_policy(
        self,
        policy_id: str,
        operation: str,
        data_context: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enforce data governance policy for an operation
        
        Args:
            policy_id: Policy to enforce
            operation: Operation being performed
            data_context: Data-related context
            user_context: User and session context
            
        Returns:
            Dict: Enforcement result
        """
        try:
            policy = self.session.query(DataGovernancePolicy).filter(
                DataGovernancePolicy.id == policy_id
            ).first()
            
            if not policy:
                raise ValueError(f"Policy not found: {policy_id}")
            
            enforcement_result = {
                "policy_id": policy_id,
                "allowed": True,
                "restrictions": [],
                "requirements": [],
                "violations": [],
                "remediation_actions": []
            }
            
            # Check policy applicability
            if not self._is_policy_applicable(policy, data_context, user_context):
                return enforcement_result
            
            # Evaluate policy rules
            for rule_name, rule_config in policy.rules.items():
                rule_result = self._evaluate_governance_rule(
                    rule_name, rule_config, operation, data_context, user_context
                )
                
                if not rule_result["allowed"]:
                    enforcement_result["allowed"] = False
                    enforcement_result["violations"].append(rule_result["violation"])
                
                enforcement_result["restrictions"].extend(rule_result.get("restrictions", []))
                enforcement_result["requirements"].extend(rule_result.get("requirements", []))
            
            # Generate remediation actions if needed
            if not enforcement_result["allowed"]:
                enforcement_result["remediation_actions"] = self._generate_remediation_actions(
                    policy, enforcement_result["violations"]
                )
            
            return enforcement_result
            
        except Exception as e:
            logger.error(f"Error enforcing governance policy: {str(e)}")
            raise
    
    def apply_privacy_controls(
        self,
        dataset_id: str,
        data: Dict[str, Any],
        privacy_level: PrivacyLevel,
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply privacy controls to data based on privacy level
        
        Args:
            dataset_id: Dataset identifier
            data: Data to apply controls to
            privacy_level: Required privacy level
            user_context: User context for access control
            
        Returns:
            Dict: Data with privacy controls applied
        """
        try:
            protected_data = data.copy()
            applied_controls = []
            
            # Get data classification
            classification = self._get_data_classification(dataset_id)
            
            # Apply controls based on privacy level
            if privacy_level in [PrivacyLevel.ENHANCED, PrivacyLevel.MAXIMUM]:
                # Apply field-level controls
                for field_name, field_value in data.items():
                    if self._is_sensitive_field(field_name, field_value):
                        control_result = self._apply_field_privacy_control(
                            field_name, field_value, privacy_level, user_context
                        )
                        protected_data[field_name] = control_result["protected_value"]
                        applied_controls.extend(control_result["controls"])
            
            # Apply record-level controls
            if privacy_level == PrivacyLevel.MAXIMUM:
                record_controls = self._apply_record_privacy_controls(
                    protected_data, user_context
                )
                protected_data = record_controls["protected_data"]
                applied_controls.extend(record_controls["controls"])
            
            return {
                "protected_data": protected_data,
                "applied_controls": applied_controls,
                "privacy_level": privacy_level.value,
                "access_granted": True
            }
            
        except Exception as e:
            logger.error(f"Error applying privacy controls: {str(e)}")
            raise
    
    def check_data_retention_compliance(
        self,
        dataset_id: str,
        data_age: timedelta,
        data_classification: DataClassification
    ) -> Dict[str, Any]:
        """
        Check data retention compliance
        
        Args:
            dataset_id: Dataset identifier
            data_age: Age of the data
            data_classification: Classification level
            
        Returns:
            Dict: Retention compliance status
        """
        try:
            # Get applicable retention policies
            retention_policies = self._get_retention_policies(
                dataset_id, data_classification
            )
            
            compliance_result = {
                "dataset_id": dataset_id,
                "data_age_days": data_age.days,
                "classification": data_classification.value,
                "compliant": True,
                "violations": [],
                "actions_required": []
            }
            
            for policy in retention_policies:
                policy_result = self._check_retention_policy(
                    policy, data_age, data_classification
                )
                
                if not policy_result["compliant"]:
                    compliance_result["compliant"] = False
                    compliance_result["violations"].append(policy_result["violation"])
                    compliance_result["actions_required"].extend(
                        policy_result["actions_required"]
                    )
            
            return compliance_result
            
        except Exception as e:
            logger.error(f"Error checking retention compliance: {str(e)}")
            raise
    
    def anonymize_data(
        self,
        data: Dict[str, Any],
        anonymization_level: str = "standard",
        preserve_utility: bool = True
    ) -> Dict[str, Any]:
        """
        Anonymize data for privacy protection
        
        Args:
            data: Data to anonymize
            anonymization_level: Level of anonymization (basic, standard, strong)
            preserve_utility: Whether to preserve data utility
            
        Returns:
            Dict: Anonymized data and metadata
        """
        try:
            anonymized_data = {}
            anonymization_metadata = {
                "techniques_applied": [],
                "fields_modified": [],
                "utility_preserved": preserve_utility,
                "anonymization_level": anonymization_level
            }
            
            for field_name, field_value in data.items():
                if self._is_personally_identifiable(field_name, field_value):
                    # Apply anonymization technique
                    anonymized_value, technique = self._anonymize_field(
                        field_name, field_value, anonymization_level, preserve_utility
                    )
                    anonymized_data[field_name] = anonymized_value
                    anonymization_metadata["techniques_applied"].append(technique)
                    anonymization_metadata["fields_modified"].append(field_name)
                else:
                    # Keep non-PII fields as-is
                    anonymized_data[field_name] = field_value
            
            return {
                "anonymized_data": anonymized_data,
                "metadata": anonymization_metadata
            }
            
        except Exception as e:
            logger.error(f"Error anonymizing data: {str(e)}")
            raise
    
    def _detect_sensitive_patterns(self, data_sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect sensitive data patterns in sample"""
        sensitive_patterns = []
        
        # Define sensitive data patterns
        patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-?\d{2}-?\d{4}\b',
            "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        }
        
        for field_name, field_value in data_sample.items():
            if isinstance(field_value, str):
                for pattern_name, pattern_regex in patterns.items():
                    if re.search(pattern_regex, field_value):
                        sensitive_patterns.append({
                            "field": field_name,
                            "pattern": pattern_name,
                            "confidence": 0.9
                        })
            
            # Check field names for sensitive indicators
            sensitive_field_names = [
                'password', 'ssn', 'social_security', 'credit_card',
                'phone', 'email', 'address', 'salary', 'income'
            ]
            
            if any(sensitive_name in field_name.lower() for sensitive_name in sensitive_field_names):
                sensitive_patterns.append({
                    "field": field_name,
                    "pattern": "sensitive_field_name",
                    "confidence": 0.8
                })
        
        return sensitive_patterns
    
    def _determine_classification_level(
        self,
        sensitive_patterns: List[Dict[str, Any]],
        schema_info: Optional[Dict[str, Any]]
    ) -> DataClassification:
        """Determine data classification level"""
        if not sensitive_patterns:
            return DataClassification.INTERNAL
        
        # Check for high-sensitivity patterns
        high_sensitivity_patterns = ['ssn', 'credit_card', 'password']
        if any(p['pattern'] in high_sensitivity_patterns for p in sensitive_patterns):
            return DataClassification.RESTRICTED
        
        # Check for medium-sensitivity patterns
        medium_sensitivity_patterns = ['email', 'phone', 'address']
        if any(p['pattern'] in medium_sensitivity_patterns for p in sensitive_patterns):
            return DataClassification.CONFIDENTIAL
        
        return DataClassification.INTERNAL
    
    def _determine_privacy_level(
        self,
        classification: DataClassification,
        sensitive_patterns: List[Dict[str, Any]]
    ) -> PrivacyLevel:
        """Determine required privacy level"""
        if classification == DataClassification.RESTRICTED:
            return PrivacyLevel.MAXIMUM
        elif classification == DataClassification.CONFIDENTIAL:
            return PrivacyLevel.ENHANCED
        elif sensitive_patterns:
            return PrivacyLevel.BASIC
        else:
            return PrivacyLevel.NONE
    
    def _identify_compliance_requirements(
        self,
        sensitive_patterns: List[Dict[str, Any]],
        classification: DataClassification
    ) -> List[str]:
        """Identify applicable compliance requirements"""
        requirements = []
        
        # Check for PII patterns (GDPR)
        pii_patterns = ['email', 'phone', 'address', 'ssn']
        if any(p['pattern'] in pii_patterns for p in sensitive_patterns):
            requirements.append("GDPR")
        
        # Check for health data patterns (HIPAA)
        health_patterns = ['medical_record', 'patient_id', 'diagnosis']
        if any(p['pattern'] in health_patterns for p in sensitive_patterns):
            requirements.append("HIPAA")
        
        # Check for financial data patterns (SOX, PCI-DSS)
        financial_patterns = ['credit_card', 'bank_account', 'financial_record']
        if any(p['pattern'] in financial_patterns for p in sensitive_patterns):
            requirements.extend(["SOX", "PCI-DSS"])
        
        return requirements
    
    def _generate_recommended_controls(
        self,
        classification: DataClassification,
        privacy_level: PrivacyLevel,
        compliance_requirements: List[str]
    ) -> List[str]:
        """Generate recommended security controls"""
        controls = []
        
        # Base controls for all classified data
        if classification != DataClassification.PUBLIC:
            controls.extend([
                "access_control",
                "audit_logging",
                "encryption_at_rest"
            ])
        
        # Additional controls based on classification
        if classification in [DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED]:
            controls.extend([
                "encryption_in_transit",
                "data_masking",
                "role_based_access"
            ])
        
        if classification == DataClassification.RESTRICTED:
            controls.extend([
                "multi_factor_authentication",
                "data_loss_prevention",
                "privileged_access_management"
            ])
        
        # Privacy-specific controls
        if privacy_level in [PrivacyLevel.ENHANCED, PrivacyLevel.MAXIMUM]:
            controls.extend([
                "data_anonymization",
                "consent_management",
                "right_to_erasure"
            ])
        
        # Compliance-specific controls
        if "GDPR" in compliance_requirements:
            controls.extend([
                "data_subject_rights",
                "privacy_impact_assessment",
                "data_protection_officer"
            ])
        
        if "HIPAA" in compliance_requirements:
            controls.extend([
                "minimum_necessary_standard",
                "business_associate_agreements",
                "breach_notification"
            ])
        
        return list(set(controls))  # Remove duplicates
    
    def _calculate_confidence_score(
        self,
        sensitive_patterns: List[Dict[str, Any]],
        schema_info: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for classification"""
        if not sensitive_patterns:
            return 0.5  # Medium confidence for no patterns
        
        # Average confidence of detected patterns
        total_confidence = sum(p['confidence'] for p in sensitive_patterns)
        avg_confidence = total_confidence / len(sensitive_patterns)
        
        # Boost confidence if schema info supports classification
        if schema_info:
            avg_confidence = min(1.0, avg_confidence + 0.1)
        
        return round(avg_confidence, 2)
    
    def _load_governance_policies(self):
        """Load governance policies into cache"""
        try:
            policies = self.session.query(DataGovernancePolicy).filter(
                DataGovernancePolicy.is_active == True
            ).all()
            
            for policy in policies:
                if policy.policy_type == "classification":
                    self.classification_rules[policy.id] = policy
                elif policy.policy_type == "privacy":
                    self.privacy_policies[policy.id] = policy
                elif policy.policy_type == "retention":
                    self.retention_policies[policy.id] = policy
                    
        except Exception as e:
            logger.error(f"Error loading governance policies: {str(e)}")
    
    def _is_policy_applicable(
        self,
        policy: DataGovernancePolicy,
        data_context: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> bool:
        """Check if policy is applicable to current context"""
        # Check data classification applicability
        if policy.data_classification:
            data_classification = data_context.get("classification")
            if data_classification not in policy.data_classification:
                return False
        
        # Check geographic applicability
        if policy.applicable_regions:
            user_region = user_context.get("region")
            if user_region not in policy.applicable_regions:
                return False
        
        return True
    
    def _evaluate_governance_rule(
        self,
        rule_name: str,
        rule_config: Dict[str, Any],
        operation: str,
        data_context: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a specific governance rule"""
        # Implementation would contain specific rule evaluation logic
        return {
            "allowed": True,
            "restrictions": [],
            "requirements": []
        }
    
    def _generate_remediation_actions(
        self,
        policy: DataGovernancePolicy,
        violations: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate remediation actions for policy violations"""
        actions = []
        
        for violation in violations:
            if violation.get("type") == "access_denied":
                actions.append("Request appropriate access permissions")
            elif violation.get("type") == "classification_mismatch":
                actions.append("Reclassify data or adjust access level")
            elif violation.get("type") == "retention_violation":
                actions.append("Archive or delete data per retention policy")
        
        return actions
    
    def _get_data_classification(self, dataset_id: str) -> DataClassification:
        """Get data classification for dataset"""
        # Implementation would query dataset registry
        return DataClassification.INTERNAL
    
    def _is_sensitive_field(self, field_name: str, field_value: Any) -> bool:
        """Check if field contains sensitive data"""
        sensitive_indicators = [
            'password', 'ssn', 'social_security', 'credit_card',
            'phone', 'email', 'address', 'salary'
        ]
        return any(indicator in field_name.lower() for indicator in sensitive_indicators)
    
    def _apply_field_privacy_control(
        self,
        field_name: str,
        field_value: Any,
        privacy_level: PrivacyLevel,
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply privacy control to a specific field"""
        if privacy_level == PrivacyLevel.MAXIMUM:
            # Full redaction
            return {
                "protected_value": "[REDACTED]",
                "controls": ["full_redaction"]
            }
        elif privacy_level == PrivacyLevel.ENHANCED:
            # Partial masking
            if isinstance(field_value, str) and len(field_value) > 4:
                masked_value = field_value[:2] + "*" * (len(field_value) - 4) + field_value[-2:]
                return {
                    "protected_value": masked_value,
                    "controls": ["partial_masking"]
                }
        
        return {
            "protected_value": field_value,
            "controls": []
        }
    
    def _apply_record_privacy_controls(
        self,
        data: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply record-level privacy controls"""
        # Implementation would apply record-level controls like k-anonymity
        return {
            "protected_data": data,
            "controls": []
        }
    
    def _get_retention_policies(
        self,
        dataset_id: str,
        classification: DataClassification
    ) -> List[DataGovernancePolicy]:
        """Get applicable retention policies"""
        return list(self.retention_policies.values())
    
    def _check_retention_policy(
        self,
        policy: DataGovernancePolicy,
        data_age: timedelta,
        classification: DataClassification
    ) -> Dict[str, Any]:
        """Check compliance with retention policy"""
        # Implementation would check specific retention rules
        return {
            "compliant": True,
            "violation": None,
            "actions_required": []
        }
    
    def _is_personally_identifiable(self, field_name: str, field_value: Any) -> bool:
        """Check if field contains personally identifiable information"""
        pii_indicators = [
            'name', 'email', 'phone', 'address', 'ssn', 'id',
            'birth', 'age', 'gender', 'race', 'religion'
        ]
        return any(indicator in field_name.lower() for indicator in pii_indicators)
    
    def _anonymize_field(
        self,
        field_name: str,
        field_value: Any,
        level: str,
        preserve_utility: bool
    ) -> Tuple[Any, str]:
        """Anonymize a specific field"""
        if level == "strong":
            # Strong anonymization - hash the value
            if isinstance(field_value, str):
                hashed_value = hashlib.sha256(field_value.encode()).hexdigest()[:8]
                return f"ANON_{hashed_value}", "hashing"
        elif level == "standard":
            # Standard anonymization - generalization
            if "age" in field_name.lower() and isinstance(field_value, int):
                # Age ranges
                age_range = f"{(field_value // 10) * 10}-{(field_value // 10) * 10 + 9}"
                return age_range, "generalization"
        
        # Basic anonymization - simple masking
        if isinstance(field_value, str):
            return "*" * len(field_value), "masking"
        
        return field_value, "none"