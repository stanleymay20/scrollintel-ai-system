"""
Audit Trail Generation and Reporting Engine

This module provides comprehensive audit trail generation and reporting
for compliance and regulatory requirements.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
import json
import csv
import io
import logging
from dataclasses import dataclass

from scrollintel.models.lineage_models import (
    AuditTrail, DataLineage, ComplianceViolation, ComplianceRule
)
from scrollintel.models.database_utils import get_sync_db

logger = logging.getLogger(__name__)


@dataclass
class AuditReportConfig:
    """Configuration for audit reports"""
    report_type: str
    start_date: datetime
    end_date: datetime
    include_lineage: bool = True
    include_violations: bool = True
    include_user_activity: bool = True
    format: str = "json"  # json, csv, pdf
    filters: Optional[Dict[str, Any]] = None


class AuditReporter:
    """Comprehensive audit trail generation and reporting system"""
    
    def __init__(self):
        self.session = next(get_sync_db())
    
    def generate_audit_report(
        self,
        config: AuditReportConfig
    ) -> Dict[str, Any]:
        """
        Generate comprehensive audit report
        
        Args:
            config: Report configuration
            
        Returns:
            Dict: Audit report data
        """
        try:
            report = {
                "report_id": str(uuid.uuid4()),
                "generated_at": datetime.utcnow().isoformat(),
                "config": {
                    "report_type": config.report_type,
                    "start_date": config.start_date.isoformat(),
                    "end_date": config.end_date.isoformat(),
                    "format": config.format
                },
                "summary": {},
                "sections": {}
            }
            
            # Generate summary
            report["summary"] = self._generate_audit_summary(config)
            
            # Generate detailed sections
            if config.include_user_activity:
                report["sections"]["user_activity"] = self._generate_user_activity_report(config)
            
            if config.include_lineage:
                report["sections"]["data_lineage"] = self._generate_lineage_report(config)
            
            if config.include_violations:
                report["sections"]["compliance_violations"] = self._generate_violations_report(config)
            
            # Add system changes section
            report["sections"]["system_changes"] = self._generate_system_changes_report(config)
            
            # Add data access section
            report["sections"]["data_access"] = self._generate_data_access_report(config)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating audit report: {str(e)}")
            raise
    
    def generate_compliance_audit_report(
        self,
        regulation_type: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Generate compliance-specific audit report
        
        Args:
            regulation_type: Type of regulation (GDPR, HIPAA, SOX, etc.)
            start_date: Report start date
            end_date: Report end date
            
        Returns:
            Dict: Compliance audit report
        """
        try:
            report = {
                "report_id": str(uuid.uuid4()),
                "regulation_type": regulation_type,
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "generated_at": datetime.utcnow().isoformat(),
                "compliance_status": "compliant",  # Will be updated based on findings
                "sections": {}
            }
            
            # Generate regulation-specific sections
            if regulation_type.upper() == "GDPR":
                report["sections"] = self._generate_gdpr_audit_report(start_date, end_date)
            elif regulation_type.upper() == "HIPAA":
                report["sections"] = self._generate_hipaa_audit_report(start_date, end_date)
            elif regulation_type.upper() == "SOX":
                report["sections"] = self._generate_sox_audit_report(start_date, end_date)
            else:
                # Generic compliance report
                report["sections"] = self._generate_generic_compliance_report(
                    regulation_type, start_date, end_date
                )
            
            # Determine overall compliance status
            report["compliance_status"] = self._determine_compliance_status(report["sections"])
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating compliance audit report: {str(e)}")
            raise
    
    def export_audit_trail(
        self,
        start_date: datetime,
        end_date: datetime,
        format: str = "csv",
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Export audit trail data
        
        Args:
            start_date: Export start date
            end_date: Export end date
            format: Export format (csv, json)
            filters: Additional filters
            
        Returns:
            str: Exported data as string
        """
        try:
            # Get audit trail data
            query = self.session.query(AuditTrail).filter(
                and_(
                    AuditTrail.timestamp >= start_date,
                    AuditTrail.timestamp <= end_date
                )
            )
            
            # Apply filters
            if filters:
                if "entity_type" in filters:
                    query = query.filter(AuditTrail.entity_type == filters["entity_type"])
                
                if "user_id" in filters:
                    query = query.filter(AuditTrail.user_id == filters["user_id"])
                
                if "action" in filters:
                    query = query.filter(AuditTrail.action == filters["action"])
            
            audit_entries = query.order_by(AuditTrail.timestamp).all()
            
            if format.lower() == "csv":
                return self._export_to_csv(audit_entries)
            elif format.lower() == "json":
                return self._export_to_json(audit_entries)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting audit trail: {str(e)}")
            raise
    
    def get_data_processing_history(
        self,
        dataset_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get complete processing history for a dataset
        
        Args:
            dataset_id: Dataset to get history for
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Dict: Processing history
        """
        try:
            # Get lineage events
            lineage_query = self.session.query(DataLineage).filter(
                or_(
                    DataLineage.source_dataset_id == dataset_id,
                    DataLineage.target_dataset_id == dataset_id
                )
            )
            
            if start_date:
                lineage_query = lineage_query.filter(
                    DataLineage.event_timestamp >= start_date
                )
            
            if end_date:
                lineage_query = lineage_query.filter(
                    DataLineage.event_timestamp <= end_date
                )
            
            lineage_events = lineage_query.order_by(DataLineage.event_timestamp).all()
            
            # Get audit trail entries
            audit_query = self.session.query(AuditTrail).filter(
                and_(
                    AuditTrail.entity_type == "dataset",
                    AuditTrail.entity_id == dataset_id
                )
            )
            
            if start_date:
                audit_query = audit_query.filter(AuditTrail.timestamp >= start_date)
            
            if end_date:
                audit_query = audit_query.filter(AuditTrail.timestamp <= end_date)
            
            audit_entries = audit_query.order_by(AuditTrail.timestamp).all()
            
            # Combine and format results
            history = {
                "dataset_id": dataset_id,
                "period": {
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None
                },
                "processing_events": [
                    {
                        "timestamp": event.event_timestamp.isoformat(),
                        "event_type": event.event_type,
                        "pipeline_id": event.pipeline_id,
                        "transformation_id": event.transformation_id,
                        "source_dataset": event.source_dataset_id,
                        "target_dataset": event.target_dataset_id,
                        "data_volume": event.data_volume,
                        "processing_duration": event.processing_duration,
                        "user_id": event.user_id
                    }
                    for event in lineage_events
                ],
                "system_changes": [
                    {
                        "timestamp": entry.timestamp.isoformat(),
                        "action": entry.action,
                        "user_id": entry.user_id,
                        "old_values": entry.old_values,
                        "new_values": entry.new_values,
                        "change_summary": entry.change_summary
                    }
                    for entry in audit_entries
                ]
            }
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting data processing history: {str(e)}")
            raise
    
    def _generate_audit_summary(self, config: AuditReportConfig) -> Dict[str, Any]:
        """Generate audit report summary"""
        try:
            # Count audit entries by type
            audit_counts = self.session.query(
                AuditTrail.entity_type,
                func.count(AuditTrail.id)
            ).filter(
                and_(
                    AuditTrail.timestamp >= config.start_date,
                    AuditTrail.timestamp <= config.end_date
                )
            ).group_by(AuditTrail.entity_type).all()
            
            # Count unique users
            unique_users = self.session.query(
                func.count(func.distinct(AuditTrail.user_id))
            ).filter(
                and_(
                    AuditTrail.timestamp >= config.start_date,
                    AuditTrail.timestamp <= config.end_date
                )
            ).scalar()
            
            # Count compliance violations
            violation_count = self.session.query(
                func.count(ComplianceViolation.id)
            ).filter(
                and_(
                    ComplianceViolation.detected_at >= config.start_date,
                    ComplianceViolation.detected_at <= config.end_date
                )
            ).scalar()
            
            return {
                "total_audit_entries": sum(count for _, count in audit_counts),
                "audit_entries_by_type": dict(audit_counts),
                "unique_users": unique_users,
                "compliance_violations": violation_count,
                "period_days": (config.end_date - config.start_date).days
            }
            
        except Exception as e:
            logger.error(f"Error generating audit summary: {str(e)}")
            return {}
    
    def _generate_user_activity_report(self, config: AuditReportConfig) -> Dict[str, Any]:
        """Generate user activity section"""
        try:
            # Get user activity statistics
            user_activity = self.session.query(
                AuditTrail.user_id,
                func.count(AuditTrail.id).label('activity_count'),
                func.min(AuditTrail.timestamp).label('first_activity'),
                func.max(AuditTrail.timestamp).label('last_activity')
            ).filter(
                and_(
                    AuditTrail.timestamp >= config.start_date,
                    AuditTrail.timestamp <= config.end_date
                )
            ).group_by(AuditTrail.user_id).all()
            
            # Get activity by action type
            action_stats = self.session.query(
                AuditTrail.action,
                func.count(AuditTrail.id)
            ).filter(
                and_(
                    AuditTrail.timestamp >= config.start_date,
                    AuditTrail.timestamp <= config.end_date
                )
            ).group_by(AuditTrail.action).all()
            
            return {
                "user_statistics": [
                    {
                        "user_id": user_id,
                        "activity_count": activity_count,
                        "first_activity": first_activity.isoformat(),
                        "last_activity": last_activity.isoformat()
                    }
                    for user_id, activity_count, first_activity, last_activity in user_activity
                ],
                "action_statistics": dict(action_stats)
            }
            
        except Exception as e:
            logger.error(f"Error generating user activity report: {str(e)}")
            return {}
    
    def _generate_lineage_report(self, config: AuditReportConfig) -> Dict[str, Any]:
        """Generate data lineage section"""
        try:
            # Get lineage statistics
            lineage_stats = self.session.query(
                DataLineage.event_type,
                func.count(DataLineage.id)
            ).filter(
                and_(
                    DataLineage.event_timestamp >= config.start_date,
                    DataLineage.event_timestamp <= config.end_date
                )
            ).group_by(DataLineage.event_type).all()
            
            # Get pipeline activity
            pipeline_stats = self.session.query(
                DataLineage.pipeline_id,
                func.count(DataLineage.id)
            ).filter(
                and_(
                    DataLineage.event_timestamp >= config.start_date,
                    DataLineage.event_timestamp <= config.end_date
                )
            ).group_by(DataLineage.pipeline_id).all()
            
            return {
                "lineage_events_by_type": dict(lineage_stats),
                "pipeline_activity": dict(pipeline_stats),
                "total_lineage_events": sum(count for _, count in lineage_stats)
            }
            
        except Exception as e:
            logger.error(f"Error generating lineage report: {str(e)}")
            return {}
    
    def _generate_violations_report(self, config: AuditReportConfig) -> Dict[str, Any]:
        """Generate compliance violations section"""
        try:
            violations = self.session.query(ComplianceViolation).filter(
                and_(
                    ComplianceViolation.detected_at >= config.start_date,
                    ComplianceViolation.detected_at <= config.end_date
                )
            ).all()
            
            # Group by severity
            severity_counts = {}
            status_counts = {}
            
            for violation in violations:
                severity_counts[violation.severity] = severity_counts.get(violation.severity, 0) + 1
                status_counts[violation.status] = status_counts.get(violation.status, 0) + 1
            
            return {
                "total_violations": len(violations),
                "violations_by_severity": severity_counts,
                "violations_by_status": status_counts,
                "violation_details": [
                    {
                        "id": v.id,
                        "rule_id": v.rule_id,
                        "pipeline_id": v.pipeline_id,
                        "violation_type": v.violation_type,
                        "severity": v.severity,
                        "status": v.status,
                        "detected_at": v.detected_at.isoformat(),
                        "resolved_at": v.resolved_at.isoformat() if v.resolved_at else None
                    }
                    for v in violations
                ]
            }
            
        except Exception as e:
            logger.error(f"Error generating violations report: {str(e)}")
            return {}
    
    def _generate_system_changes_report(self, config: AuditReportConfig) -> Dict[str, Any]:
        """Generate system changes section"""
        try:
            # Get system configuration changes
            system_changes = self.session.query(AuditTrail).filter(
                and_(
                    AuditTrail.timestamp >= config.start_date,
                    AuditTrail.timestamp <= config.end_date,
                    AuditTrail.entity_type.in_(['pipeline', 'rule', 'policy'])
                )
            ).order_by(desc(AuditTrail.timestamp)).all()
            
            return {
                "total_changes": len(system_changes),
                "changes": [
                    {
                        "timestamp": change.timestamp.isoformat(),
                        "entity_type": change.entity_type,
                        "entity_id": change.entity_id,
                        "action": change.action,
                        "user_id": change.user_id,
                        "change_summary": change.change_summary
                    }
                    for change in system_changes
                ]
            }
            
        except Exception as e:
            logger.error(f"Error generating system changes report: {str(e)}")
            return {}
    
    def _generate_data_access_report(self, config: AuditReportConfig) -> Dict[str, Any]:
        """Generate data access section"""
        try:
            # Get data access events
            access_events = self.session.query(AuditTrail).filter(
                and_(
                    AuditTrail.timestamp >= config.start_date,
                    AuditTrail.timestamp <= config.end_date,
                    AuditTrail.action.in_(['read', 'access', 'query'])
                )
            ).all()
            
            # Group by user and dataset
            user_access = {}
            for event in access_events:
                user_id = event.user_id
                if user_id not in user_access:
                    user_access[user_id] = {
                        "access_count": 0,
                        "datasets_accessed": set(),
                        "first_access": event.timestamp,
                        "last_access": event.timestamp
                    }
                
                user_access[user_id]["access_count"] += 1
                user_access[user_id]["datasets_accessed"].add(event.entity_id)
                
                if event.timestamp < user_access[user_id]["first_access"]:
                    user_access[user_id]["first_access"] = event.timestamp
                if event.timestamp > user_access[user_id]["last_access"]:
                    user_access[user_id]["last_access"] = event.timestamp
            
            # Convert to serializable format
            access_summary = []
            for user_id, stats in user_access.items():
                access_summary.append({
                    "user_id": user_id,
                    "access_count": stats["access_count"],
                    "unique_datasets": len(stats["datasets_accessed"]),
                    "first_access": stats["first_access"].isoformat(),
                    "last_access": stats["last_access"].isoformat()
                })
            
            return {
                "total_access_events": len(access_events),
                "user_access_summary": access_summary
            }
            
        except Exception as e:
            logger.error(f"Error generating data access report: {str(e)}")
            return {}
    
    def _generate_gdpr_audit_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate GDPR-specific audit report"""
        # Implementation for GDPR compliance reporting
        return {
            "data_subject_rights": {},
            "consent_management": {},
            "data_processing_lawfulness": {},
            "data_protection_impact_assessments": {}
        }
    
    def _generate_hipaa_audit_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate HIPAA-specific audit report"""
        # Implementation for HIPAA compliance reporting
        return {
            "phi_access_controls": {},
            "audit_controls": {},
            "integrity_controls": {},
            "transmission_security": {}
        }
    
    def _generate_sox_audit_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate SOX-specific audit report"""
        # Implementation for SOX compliance reporting
        return {
            "financial_data_controls": {},
            "change_management": {},
            "access_controls": {},
            "data_integrity": {}
        }
    
    def _generate_generic_compliance_report(
        self,
        regulation_type: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate generic compliance report"""
        return {
            "regulation_type": regulation_type,
            "compliance_controls": {},
            "audit_findings": {},
            "remediation_actions": {}
        }
    
    def _determine_compliance_status(self, sections: Dict[str, Any]) -> str:
        """Determine overall compliance status from report sections"""
        # Simple logic - would be more sophisticated in real implementation
        violations = sections.get("compliance_violations", {})
        total_violations = violations.get("total_violations", 0)
        
        if total_violations == 0:
            return "compliant"
        elif total_violations < 5:
            return "minor_issues"
        else:
            return "non_compliant"
    
    def _export_to_csv(self, audit_entries: List[AuditTrail]) -> str:
        """Export audit entries to CSV format"""
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'ID', 'Entity Type', 'Entity ID', 'Action', 'User ID',
            'Timestamp', 'IP Address', 'Change Summary'
        ])
        
        # Write data
        for entry in audit_entries:
            writer.writerow([
                entry.id,
                entry.entity_type,
                entry.entity_id,
                entry.action,
                entry.user_id,
                entry.timestamp.isoformat(),
                entry.ip_address,
                entry.change_summary
            ])
        
        return output.getvalue()
    
    def _export_to_json(self, audit_entries: List[AuditTrail]) -> str:
        """Export audit entries to JSON format"""
        data = []
        for entry in audit_entries:
            data.append({
                "id": entry.id,
                "entity_type": entry.entity_type,
                "entity_id": entry.entity_id,
                "action": entry.action,
                "user_id": entry.user_id,
                "timestamp": entry.timestamp.isoformat(),
                "ip_address": entry.ip_address,
                "user_agent": entry.user_agent,
                "old_values": entry.old_values,
                "new_values": entry.new_values,
                "change_summary": entry.change_summary,
                "audit_metadata": entry.audit_metadata
            })
        
        return json.dumps(data, indent=2)