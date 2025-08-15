"""Compliance reporting system for governance requirements."""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from collections import defaultdict
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from ..models.governance_models import (
    ComplianceReport, AuditEvent, AuditEventType, DataClassification,
    GovernancePolicy, PolicyType, PolicyStatus
)
from ..models.governance_database import (
    ComplianceReportModel, AuditEventModel, GovernancePolicyModel,
    DataCatalogEntryModel, AccessControlEntryModel, UserModel
)
from ..models.database import get_db_session
from ..core.exceptions import AIDataReadinessError


logger = logging.getLogger(__name__)


class ComplianceReporterError(AIDataReadinessError):
    """Exception raised for compliance reporter errors."""
    pass


class ComplianceReporter:
    """Compliance reporting system for governance requirements."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Define compliance frameworks and their requirements
        self.compliance_frameworks = {
            'GDPR': {
                'name': 'General Data Protection Regulation',
                'requirements': [
                    'data_minimization',
                    'purpose_limitation',
                    'consent_management',
                    'right_to_erasure',
                    'data_portability',
                    'privacy_by_design',
                    'breach_notification'
                ]
            },
            'CCPA': {
                'name': 'California Consumer Privacy Act',
                'requirements': [
                    'consumer_rights',
                    'data_disclosure',
                    'opt_out_rights',
                    'non_discrimination',
                    'data_security'
                ]
            },
            'SOX': {
                'name': 'Sarbanes-Oxley Act',
                'requirements': [
                    'data_integrity',
                    'audit_trails',
                    'access_controls',
                    'change_management'
                ]
            },
            'HIPAA': {
                'name': 'Health Insurance Portability and Accountability Act',
                'requirements': [
                    'phi_protection',
                    'access_controls',
                    'audit_logs',
                    'encryption',
                    'breach_notification'
                ]
            }
        }
    
    def generate_compliance_report(
        self,
        framework: str,
        scope: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        generated_by: str = "system"
    ) -> ComplianceReport:
        """Generate comprehensive compliance report."""
        try:
            if framework not in self.compliance_frameworks:
                raise ComplianceReporterError(f"Unsupported compliance framework: {framework}")
            
            # Set default date range if not provided
            if not end_date:
                end_date = datetime.utcnow()
            if not start_date:
                start_date = end_date - timedelta(days=90)  # Default 90-day period
            
            with get_db_session() as session:
                # Assess compliance based on framework
                assessment_result = self._assess_compliance(
                    session, framework, scope, start_date, end_date
                )
                
                # Create compliance report
                report = ComplianceReport(
                    report_type=framework,
                    scope=scope or [],
                    compliance_score=assessment_result['overall_score'],
                    violations=assessment_result['violations'],
                    recommendations=assessment_result['recommendations'],
                    assessment_criteria=assessment_result['criteria'],
                    generated_by=generated_by,
                    period_start=start_date,
                    period_end=end_date
                )
                
                # Save report to database
                report_model = ComplianceReportModel(
                    report_type=report.report_type,
                    scope=report.scope,
                    compliance_score=report.compliance_score,
                    violations=report.violations,
                    recommendations=report.recommendations,
                    assessment_criteria=report.assessment_criteria,
                    generated_by=report.generated_by,
                    period_start=report.period_start,
                    period_end=report.period_end
                )
                
                session.add(report_model)
                session.commit()
                session.refresh(report_model)
                
                report.id = str(report_model.id)
                
                self.logger.info(f"Generated {framework} compliance report with score: {report.compliance_score}")
                
                return report
                
        except Exception as e:
            self.logger.error(f"Error generating compliance report: {str(e)}")
            raise ComplianceReporterError(f"Failed to generate compliance report: {str(e)}")
    
    def get_compliance_dashboard(
        self,
        frameworks: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get compliance dashboard with key metrics."""
        try:
            if not frameworks:
                frameworks = list(self.compliance_frameworks.keys())
            
            if not end_date:
                end_date = datetime.utcnow()
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            dashboard = {
                'period': {
                    'start_date': start_date,
                    'end_date': end_date
                },
                'frameworks': {},
                'overall_metrics': {},
                'trending': {},
                'alerts': []
            }
            
            with get_db_session() as session:
                for framework in frameworks:
                    framework_metrics = self._get_framework_metrics(
                        session, framework, start_date, end_date
                    )
                    dashboard['frameworks'][framework] = framework_metrics
                
                # Calculate overall metrics
                dashboard['overall_metrics'] = self._calculate_overall_compliance_metrics(
                    session, start_date, end_date
                )
                
                # Get compliance trending
                dashboard['trending'] = self._get_compliance_trending(
                    session, frameworks, start_date, end_date
                )
                
                # Get compliance alerts
                dashboard['alerts'] = self._get_compliance_alerts(
                    session, frameworks
                )
            
            return dashboard
            
        except Exception as e:
            self.logger.error(f"Error getting compliance dashboard: {str(e)}")
            raise ComplianceReporterError(f"Failed to get compliance dashboard: {str(e)}")
    
    def validate_data_classification_compliance(
        self,
        dataset_id: str
    ) -> Dict[str, Any]:
        """Validate data classification compliance for a dataset."""
        try:
            with get_db_session() as session:
                # Get dataset information
                dataset = session.query(DataCatalogEntryModel).filter(
                    DataCatalogEntryModel.dataset_id == dataset_id
                ).first()
                
                if not dataset:
                    raise ComplianceReporterError(f"Dataset {dataset_id} not found")
                
                validation_result = {
                    'dataset_id': dataset_id,
                    'classification': dataset.classification.value if dataset.classification else 'unclassified',
                    'compliance_status': 'compliant',
                    'issues': [],
                    'recommendations': []
                }
                
                # Check if dataset is classified
                if not dataset.classification or dataset.classification == DataClassification.PUBLIC:
                    validation_result['issues'].append({
                        'type': 'missing_classification',
                        'severity': 'high',
                        'description': 'Dataset lacks proper data classification'
                    })
                    validation_result['recommendations'].append({
                        'type': 'classify_data',
                        'description': 'Classify dataset according to data sensitivity'
                    })
                    validation_result['compliance_status'] = 'non_compliant'
                
                # Check if sensitive data has proper access controls
                if dataset.classification in [DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED]:
                    access_controls = session.query(AccessControlEntryModel).filter(
                        AccessControlEntryModel.resource_id == dataset_id,
                        AccessControlEntryModel.is_active == True
                    ).count()
                    
                    if access_controls == 0:
                        validation_result['issues'].append({
                            'type': 'missing_access_controls',
                            'severity': 'critical',
                            'description': 'Sensitive dataset lacks access controls'
                        })
                        validation_result['recommendations'].append({
                            'type': 'implement_access_controls',
                            'description': 'Implement proper access controls for sensitive data'
                        })
                        validation_result['compliance_status'] = 'non_compliant'
                
                # Check for data retention policy
                if not dataset.retention_policy:
                    validation_result['issues'].append({
                        'type': 'missing_retention_policy',
                        'severity': 'medium',
                        'description': 'Dataset lacks data retention policy'
                    })
                    validation_result['recommendations'].append({
                        'type': 'define_retention_policy',
                        'description': 'Define and implement data retention policy'
                    })
                
                return validation_result
                
        except Exception as e:
            self.logger.error(f"Error validating data classification compliance: {str(e)}")
            raise ComplianceReporterError(f"Failed to validate data classification compliance: {str(e)}")
    
    def audit_access_compliance(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Audit access compliance across the system."""
        try:
            if not end_date:
                end_date = datetime.utcnow()
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            with get_db_session() as session:
                # Get all access events in the period
                access_events = session.query(AuditEventModel).filter(
                    AuditEventModel.event_type == AuditEventType.DATA_ACCESS.value,
                    AuditEventModel.timestamp >= start_date,
                    AuditEventModel.timestamp <= end_date
                ).all()
                
                audit_result = {
                    'period': {
                        'start_date': start_date,
                        'end_date': end_date
                    },
                    'total_access_events': len(access_events),
                    'compliance_violations': [],
                    'access_patterns': {},
                    'risk_indicators': {},
                    'recommendations': []
                }
                
                # Analyze access patterns
                audit_result['access_patterns'] = self._analyze_access_patterns(access_events)
                
                # Identify compliance violations
                audit_result['compliance_violations'] = self._identify_access_violations(
                    session, access_events
                )
                
                # Calculate risk indicators
                audit_result['risk_indicators'] = self._calculate_access_risk_indicators(
                    access_events
                )
                
                # Generate recommendations
                audit_result['recommendations'] = self._generate_access_recommendations(
                    audit_result
                )
                
                return audit_result
                
        except Exception as e:
            self.logger.error(f"Error auditing access compliance: {str(e)}")
            raise ComplianceReporterError(f"Failed to audit access compliance: {str(e)}")
    
    def generate_audit_trail_report(
        self,
        resource_id: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive audit trail report."""
        try:
            if not end_date:
                end_date = datetime.utcnow()
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            with get_db_session() as session:
                # Build query
                query = session.query(AuditEventModel).filter(
                    AuditEventModel.timestamp >= start_date,
                    AuditEventModel.timestamp <= end_date
                )
                
                if resource_id:
                    query = query.filter(AuditEventModel.resource_id == resource_id)
                
                if user_id:
                    query = query.filter(AuditEventModel.user_id == user_id)
                
                if event_types:
                    query = query.filter(AuditEventModel.event_type.in_(event_types))
                
                events = query.order_by(AuditEventModel.timestamp.desc()).all()
                
                report = {
                    'filters': {
                        'resource_id': resource_id,
                        'user_id': user_id,
                        'start_date': start_date,
                        'end_date': end_date,
                        'event_types': event_types
                    },
                    'summary': {
                        'total_events': len(events),
                        'unique_users': len(set(event.user_id for event in events)),
                        'unique_resources': len(set(f"{event.resource_type}:{event.resource_id}" 
                                                 for event in events if event.resource_id)),
                        'event_type_breakdown': self._get_event_type_breakdown(events),
                        'success_rate': sum(1 for event in events if event.success) / len(events) if events else 0
                    },
                    'timeline': self._create_audit_timeline(events),
                    'detailed_events': [
                        {
                            'id': str(event.id),
                            'timestamp': event.timestamp,
                            'event_type': event.event_type,
                            'user_id': event.user_id,
                            'resource_id': event.resource_id,
                            'resource_type': event.resource_type,
                            'action': event.action,
                            'success': event.success,
                            'ip_address': event.ip_address,
                            'details': event.details
                        }
                        for event in events[:1000]  # Limit to 1000 events
                    ],
                    'compliance_notes': self._generate_compliance_notes(events)
                }
                
                return report
                
        except Exception as e:
            self.logger.error(f"Error generating audit trail report: {str(e)}")
            raise ComplianceReporterError(f"Failed to generate audit trail report: {str(e)}")
    
    def _assess_compliance(
        self,
        session: Session,
        framework: str,
        scope: Optional[List[str]],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Assess compliance for a specific framework."""
        framework_config = self.compliance_frameworks[framework]
        requirements = framework_config['requirements']
        
        assessment = {
            'overall_score': 0.0,
            'violations': [],
            'recommendations': [],
            'criteria': framework_config
        }
        
        requirement_scores = {}
        
        for requirement in requirements:
            score, violations, recommendations = self._assess_requirement(
                session, framework, requirement, scope, start_date, end_date
            )
            
            requirement_scores[requirement] = score
            assessment['violations'].extend(violations)
            assessment['recommendations'].extend(recommendations)
        
        # Calculate overall score
        assessment['overall_score'] = sum(requirement_scores.values()) / len(requirement_scores)
        assessment['requirement_scores'] = requirement_scores
        
        return assessment
    
    def _assess_requirement(
        self,
        session: Session,
        framework: str,
        requirement: str,
        scope: Optional[List[str]],
        start_date: datetime,
        end_date: datetime
    ) -> tuple:
        """Assess a specific compliance requirement."""
        score = 100.0
        violations = []
        recommendations = []
        
        # Framework-specific requirement assessment
        if framework == 'GDPR':
            score, violations, recommendations = self._assess_gdpr_requirement(
                session, requirement, scope, start_date, end_date
            )
        elif framework == 'CCPA':
            score, violations, recommendations = self._assess_ccpa_requirement(
                session, requirement, scope, start_date, end_date
            )
        elif framework == 'SOX':
            score, violations, recommendations = self._assess_sox_requirement(
                session, requirement, scope, start_date, end_date
            )
        elif framework == 'HIPAA':
            score, violations, recommendations = self._assess_hipaa_requirement(
                session, requirement, scope, start_date, end_date
            )
        
        return score, violations, recommendations
    
    def _assess_gdpr_requirement(
        self,
        session: Session,
        requirement: str,
        scope: Optional[List[str]],
        start_date: datetime,
        end_date: datetime
    ) -> tuple:
        """Assess GDPR-specific requirement."""
        score = 100.0
        violations = []
        recommendations = []
        
        if requirement == 'data_minimization':
            # Check if datasets collect only necessary data
            datasets = session.query(DataCatalogEntryModel).all()
            for dataset in datasets:
                if not dataset.business_glossary_terms:
                    violations.append({
                        'type': 'data_minimization',
                        'resource_id': dataset.dataset_id,
                        'description': 'Dataset lacks business purpose documentation',
                        'severity': 'medium'
                    })
                    score -= 10
        
        elif requirement == 'consent_management':
            # Check for consent tracking mechanisms
            consent_events = session.query(AuditEventModel).filter(
                AuditEventModel.action.like('%consent%'),
                AuditEventModel.timestamp >= start_date,
                AuditEventModel.timestamp <= end_date
            ).count()
            
            if consent_events == 0:
                violations.append({
                    'type': 'consent_management',
                    'description': 'No consent management events found',
                    'severity': 'high'
                })
                score -= 30
        
        elif requirement == 'right_to_erasure':
            # Check for data deletion capabilities
            deletion_events = session.query(AuditEventModel).filter(
                AuditEventModel.action.in_(['delete', 'erase', 'remove']),
                AuditEventModel.timestamp >= start_date,
                AuditEventModel.timestamp <= end_date
            ).count()
            
            if deletion_events == 0:
                recommendations.append({
                    'type': 'implement_deletion',
                    'description': 'Implement data deletion capabilities for right to erasure'
                })
        
        return max(0, score), violations, recommendations
    
    def _assess_ccpa_requirement(
        self,
        session: Session,
        requirement: str,
        scope: Optional[List[str]],
        start_date: datetime,
        end_date: datetime
    ) -> tuple:
        """Assess CCPA-specific requirement."""
        score = 100.0
        violations = []
        recommendations = []
        
        if requirement == 'consumer_rights':
            # Check for consumer rights implementation
            rights_events = session.query(AuditEventModel).filter(
                AuditEventModel.action.like('%consumer_right%'),
                AuditEventModel.timestamp >= start_date,
                AuditEventModel.timestamp <= end_date
            ).count()
            
            if rights_events == 0:
                violations.append({
                    'type': 'consumer_rights',
                    'description': 'No consumer rights events found',
                    'severity': 'high'
                })
                score -= 25
        
        elif requirement == 'data_disclosure':
            # Check for data disclosure documentation
            datasets = session.query(DataCatalogEntryModel).all()
            undisclosed_count = sum(1 for dataset in datasets if not dataset.description)
            
            if undisclosed_count > 0:
                violations.append({
                    'type': 'data_disclosure',
                    'description': f'{undisclosed_count} datasets lack disclosure documentation',
                    'severity': 'medium'
                })
                score -= (undisclosed_count * 5)
        
        return max(0, score), violations, recommendations
    
    def _assess_sox_requirement(
        self,
        session: Session,
        requirement: str,
        scope: Optional[List[str]],
        start_date: datetime,
        end_date: datetime
    ) -> tuple:
        """Assess SOX-specific requirement."""
        score = 100.0
        violations = []
        recommendations = []
        
        if requirement == 'audit_trails':
            # Check audit trail completeness
            total_events = session.query(AuditEventModel).filter(
                AuditEventModel.timestamp >= start_date,
                AuditEventModel.timestamp <= end_date
            ).count()
            
            if total_events == 0:
                violations.append({
                    'type': 'audit_trails',
                    'description': 'No audit events found in the period',
                    'severity': 'critical'
                })
                score -= 50
        
        elif requirement == 'access_controls':
            # Check access control implementation
            access_controls = session.query(AccessControlEntryModel).filter(
                AccessControlEntryModel.is_active == True
            ).count()
            
            total_resources = session.query(DataCatalogEntryModel).count()
            
            if access_controls < total_resources * 0.5:  # Less than 50% coverage
                violations.append({
                    'type': 'access_controls',
                    'description': 'Insufficient access control coverage',
                    'severity': 'high'
                })
                score -= 30
        
        return max(0, score), violations, recommendations
    
    def _assess_hipaa_requirement(
        self,
        session: Session,
        requirement: str,
        scope: Optional[List[str]],
        start_date: datetime,
        end_date: datetime
    ) -> tuple:
        """Assess HIPAA-specific requirement."""
        score = 100.0
        violations = []
        recommendations = []
        
        if requirement == 'phi_protection':
            # Check for PHI data classification
            phi_datasets = session.query(DataCatalogEntryModel).filter(
                DataCatalogEntryModel.tags.contains(['PHI'])
            ).all()
            
            for dataset in phi_datasets:
                if dataset.classification != DataClassification.RESTRICTED:
                    violations.append({
                        'type': 'phi_protection',
                        'resource_id': dataset.dataset_id,
                        'description': 'PHI dataset not classified as restricted',
                        'severity': 'critical'
                    })
                    score -= 20
        
        elif requirement == 'encryption':
            # Check for encryption implementation (would need additional metadata)
            recommendations.append({
                'type': 'verify_encryption',
                'description': 'Verify encryption implementation for PHI data'
            })
        
        return max(0, score), violations, recommendations
    
    def _get_framework_metrics(
        self,
        session: Session,
        framework: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get metrics for a specific compliance framework."""
        # Get latest compliance report for the framework
        latest_report = session.query(ComplianceReportModel).filter(
            ComplianceReportModel.report_type == framework
        ).order_by(ComplianceReportModel.generated_at.desc()).first()
        
        metrics = {
            'framework': framework,
            'name': self.compliance_frameworks[framework]['name'],
            'latest_score': latest_report.compliance_score if latest_report else 0,
            'last_assessment': latest_report.generated_at if latest_report else None,
            'violations_count': len(latest_report.violations) if latest_report else 0,
            'status': 'compliant' if (latest_report and latest_report.compliance_score >= 80) else 'non_compliant'
        }
        
        return metrics
    
    def _calculate_overall_compliance_metrics(
        self,
        session: Session,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Calculate overall compliance metrics."""
        # Get all recent compliance reports
        reports = session.query(ComplianceReportModel).filter(
            ComplianceReportModel.generated_at >= start_date
        ).all()
        
        if not reports:
            return {
                'average_score': 0,
                'total_violations': 0,
                'compliant_frameworks': 0,
                'total_frameworks': len(self.compliance_frameworks)
            }
        
        total_score = sum(report.compliance_score for report in reports)
        average_score = total_score / len(reports)
        total_violations = sum(len(report.violations) for report in reports)
        compliant_frameworks = sum(1 for report in reports if report.compliance_score >= 80)
        
        return {
            'average_score': average_score,
            'total_violations': total_violations,
            'compliant_frameworks': compliant_frameworks,
            'total_frameworks': len(self.compliance_frameworks)
        }
    
    def _get_compliance_trending(
        self,
        session: Session,
        frameworks: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get compliance trending data."""
        trending = {}
        
        for framework in frameworks:
            reports = session.query(ComplianceReportModel).filter(
                ComplianceReportModel.report_type == framework,
                ComplianceReportModel.generated_at >= start_date,
                ComplianceReportModel.generated_at <= end_date
            ).order_by(ComplianceReportModel.generated_at).all()
            
            trend_data = [
                {
                    'date': report.generated_at.strftime('%Y-%m-%d'),
                    'score': report.compliance_score
                }
                for report in reports
            ]
            
            trending[framework] = {
                'data': trend_data,
                'trend': self._calculate_trend_direction([d['score'] for d in trend_data])
            }
        
        return trending
    
    def _get_compliance_alerts(
        self,
        session: Session,
        frameworks: List[str]
    ) -> List[Dict[str, Any]]:
        """Get compliance alerts."""
        alerts = []
        
        for framework in frameworks:
            latest_report = session.query(ComplianceReportModel).filter(
                ComplianceReportModel.report_type == framework
            ).order_by(ComplianceReportModel.generated_at.desc()).first()
            
            if latest_report:
                if latest_report.compliance_score < 70:
                    alerts.append({
                        'type': 'low_compliance_score',
                        'framework': framework,
                        'severity': 'high',
                        'message': f'{framework} compliance score is below 70%',
                        'score': latest_report.compliance_score
                    })
                
                critical_violations = [
                    v for v in latest_report.violations 
                    if v.get('severity') == 'critical'
                ]
                
                if critical_violations:
                    alerts.append({
                        'type': 'critical_violations',
                        'framework': framework,
                        'severity': 'critical',
                        'message': f'{len(critical_violations)} critical violations found',
                        'violations': critical_violations
                    })
        
        return alerts
    
    def _analyze_access_patterns(self, events: List[AuditEventModel]) -> Dict[str, Any]:
        """Analyze access patterns for compliance."""
        patterns = {
            'hourly_distribution': defaultdict(int),
            'user_access_frequency': defaultdict(int),
            'resource_access_frequency': defaultdict(int),
            'unusual_access_times': []
        }
        
        for event in events:
            hour = event.timestamp.hour
            patterns['hourly_distribution'][hour] += 1
            patterns['user_access_frequency'][event.user_id] += 1
            
            if event.resource_id:
                resource_key = f"{event.resource_type}:{event.resource_id}"
                patterns['resource_access_frequency'][resource_key] += 1
            
            # Flag unusual access times (outside business hours)
            if hour < 6 or hour > 22:
                patterns['unusual_access_times'].append({
                    'timestamp': event.timestamp,
                    'user_id': event.user_id,
                    'resource_id': event.resource_id,
                    'action': event.action
                })
        
        return {
            'hourly_distribution': dict(patterns['hourly_distribution']),
            'top_users': sorted(patterns['user_access_frequency'].items(), 
                              key=lambda x: x[1], reverse=True)[:10],
            'top_resources': sorted(patterns['resource_access_frequency'].items(), 
                                  key=lambda x: x[1], reverse=True)[:10],
            'unusual_access_count': len(patterns['unusual_access_times']),
            'unusual_access_details': patterns['unusual_access_times'][:50]  # Limit to 50
        }
    
    def _identify_access_violations(
        self,
        session: Session,
        events: List[AuditEventModel]
    ) -> List[Dict[str, Any]]:
        """Identify access compliance violations."""
        violations = []
        
        # Check for failed access attempts
        failed_events = [event for event in events if not event.success]
        if len(failed_events) > len(events) * 0.1:  # More than 10% failure rate
            violations.append({
                'type': 'high_failure_rate',
                'severity': 'medium',
                'description': f'High access failure rate: {len(failed_events)}/{len(events)}',
                'count': len(failed_events)
            })
        
        # Check for excessive access by single users
        user_counts = defaultdict(int)
        for event in events:
            user_counts[event.user_id] += 1
        
        avg_access = sum(user_counts.values()) / len(user_counts) if user_counts else 0
        for user_id, count in user_counts.items():
            if count > avg_access * 5:  # 5x above average
                violations.append({
                    'type': 'excessive_access',
                    'severity': 'medium',
                    'description': f'User {user_id} has excessive access count: {count}',
                    'user_id': user_id,
                    'access_count': count
                })
        
        return violations
    
    def _calculate_access_risk_indicators(self, events: List[AuditEventModel]) -> Dict[str, Any]:
        """Calculate access risk indicators."""
        total_events = len(events)
        failed_events = [event for event in events if not event.success]
        unusual_hours = [event for event in events if event.timestamp.hour < 6 or event.timestamp.hour > 22]
        
        return {
            'failure_rate': len(failed_events) / total_events if total_events > 0 else 0,
            'unusual_hour_rate': len(unusual_hours) / total_events if total_events > 0 else 0,
            'risk_score': min(100, (len(failed_events) * 2) + (len(unusual_hours) * 1)),
            'total_failed_attempts': len(failed_events),
            'unusual_hour_accesses': len(unusual_hours)
        }
    
    def _generate_access_recommendations(self, audit_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate access compliance recommendations."""
        recommendations = []
        
        if audit_result['risk_indicators']['failure_rate'] > 0.1:
            recommendations.append({
                'type': 'reduce_failure_rate',
                'priority': 'high',
                'description': 'Investigate and reduce high access failure rate'
            })
        
        if audit_result['risk_indicators']['unusual_hour_rate'] > 0.05:
            recommendations.append({
                'type': 'monitor_unusual_hours',
                'priority': 'medium',
                'description': 'Monitor and review unusual hour access patterns'
            })
        
        if len(audit_result['compliance_violations']) > 0:
            recommendations.append({
                'type': 'address_violations',
                'priority': 'high',
                'description': 'Address identified compliance violations'
            })
        
        return recommendations
    
    def _get_event_type_breakdown(self, events: List[AuditEventModel]) -> Dict[str, int]:
        """Get breakdown of events by type."""
        breakdown = defaultdict(int)
        for event in events:
            breakdown[event.event_type] += 1
        return dict(breakdown)
    
    def _create_audit_timeline(self, events: List[AuditEventModel]) -> List[Dict[str, Any]]:
        """Create audit timeline for visualization."""
        # Group events by day
        daily_counts = defaultdict(int)
        for event in events:
            day = event.timestamp.strftime('%Y-%m-%d')
            daily_counts[day] += 1
        
        timeline = [
            {'date': day, 'event_count': count}
            for day, count in sorted(daily_counts.items())
        ]
        
        return timeline
    
    def _generate_compliance_notes(self, events: List[AuditEventModel]) -> List[str]:
        """Generate compliance notes for the audit trail."""
        notes = []
        
        if events:
            notes.append(f"Audit trail contains {len(events)} events")
            
            unique_users = len(set(event.user_id for event in events))
            notes.append(f"Activity from {unique_users} unique users")
            
            failed_events = [event for event in events if not event.success]
            if failed_events:
                notes.append(f"{len(failed_events)} failed events require investigation")
        else:
            notes.append("No audit events found for the specified criteria")
        
        return notes
    
    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values."""
        if len(values) < 2:
            return "stable"
        
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        if second_avg > first_avg * 1.1:
            return "increasing"
        elif second_avg < first_avg * 0.9:
            return "decreasing"
        else:
            return "stable"