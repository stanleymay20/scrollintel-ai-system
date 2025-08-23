"""
Security Monitoring Engine
Real-time security event monitoring, threat detection, and response automation
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
from scrollintel.models.security_audit_models import (
    AuditLog, SecurityAlert, ThreatIntelligence, SecurityMetrics,
    SecurityEventType, SeverityLevel
)
from scrollintel.core.config import get_database_session
from scrollintel.engines.siem_integration_engine import SIEMIntegrationEngine
import logging
import re
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class SecurityMonitoringEngine:
    """Engine for real-time security monitoring and threat detection"""
    
    def __init__(self):
        self.siem_engine = SIEMIntegrationEngine()
        self.threat_rules = self._load_threat_rules()
        self.baseline_metrics = {}
        self.active_alerts = {}
        self._initialize_monitoring()
    
    def _initialize_monitoring(self):
        """Initialize security monitoring components"""
        try:
            self._load_threat_intelligence()
            self._calculate_baselines()
            logger.info("Security monitoring engine initialized")
        except Exception as e:
            logger.error(f"Error initializing security monitoring: {str(e)}")
    
    def _load_threat_rules(self) -> Dict[str, Any]:
        """Load threat detection rules"""
        return {
            'brute_force': {
                'threshold': 5,
                'window': 300,  # 5 minutes
                'severity': 'high'
            },
            'privilege_escalation': {
                'patterns': ['sudo', 'admin', 'root', 'elevated'],
                'severity': 'critical'
            },
            'data_exfiltration': {
                'threshold_mb': 100,
                'window': 3600,  # 1 hour
                'severity': 'high'
            },
            'anomalous_access': {
                'deviation_threshold': 3.0,  # Standard deviations
                'severity': 'medium'
            },
            'suspicious_ip': {
                'check_threat_intel': True,
                'severity': 'high'
            }
        }
    
    def _load_threat_intelligence(self):
        """Load active threat intelligence indicators"""
        try:
            with get_database_session() as db:
                active_iocs = db.query(ThreatIntelligence).filter(
                    ThreatIntelligence.is_active == True
                ).all()
                
                self.threat_iocs = {
                    'ips': [],
                    'domains': [],
                    'hashes': [],
                    'urls': []
                }
                
                for ioc in active_iocs:
                    if ioc.ioc_type in self.threat_iocs:
                        self.threat_iocs[ioc.ioc_type].append({
                            'value': ioc.ioc_value,
                            'confidence': ioc.confidence_level,
                            'threat_type': ioc.threat_type
                        })
                        
        except Exception as e:
            logger.error(f"Error loading threat intelligence: {str(e)}")
            self.threat_iocs = {'ips': [], 'domains': [], 'hashes': [], 'urls': []}
    
    def _calculate_baselines(self):
        """Calculate baseline metrics for anomaly detection"""
        try:
            with get_database_session() as db:
                # Calculate baseline for the last 30 days
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=30)
                
                # Login frequency baseline
                login_counts = db.query(
                    func.date(AuditLog.timestamp).label('date'),
                    func.count(AuditLog.id).label('count')
                ).filter(
                    and_(
                        AuditLog.timestamp >= start_date,
                        AuditLog.event_type == 'authentication',
                        AuditLog.result_status == 'success'
                    )
                ).group_by(func.date(AuditLog.timestamp)).all()
                
                if login_counts:
                    counts = [row.count for row in login_counts]
                    self.baseline_metrics['login_mean'] = sum(counts) / len(counts)
                    self.baseline_metrics['login_std'] = (
                        sum((x - self.baseline_metrics['login_mean']) ** 2 for x in counts) / len(counts)
                    ) ** 0.5
                
                # Data access baseline
                access_counts = db.query(
                    func.date(AuditLog.timestamp).label('date'),
                    func.count(AuditLog.id).label('count')
                ).filter(
                    and_(
                        AuditLog.timestamp >= start_date,
                        AuditLog.event_type == 'data_access'
                    )
                ).group_by(func.date(AuditLog.timestamp)).all()
                
                if access_counts:
                    counts = [row.count for row in access_counts]
                    self.baseline_metrics['access_mean'] = sum(counts) / len(counts)
                    self.baseline_metrics['access_std'] = (
                        sum((x - self.baseline_metrics['access_mean']) ** 2 for x in counts) / len(counts)
                    ) ** 0.5
                        
        except Exception as e:
            logger.error(f"Error calculating baselines: {str(e)}")
    
    async def process_security_event(self, audit_log: AuditLog) -> List[SecurityAlert]:
        """Process security event and detect threats"""
        alerts = []
        
        try:
            # Run threat detection rules
            brute_force_alert = await self._detect_brute_force(audit_log)
            if brute_force_alert:
                alerts.append(brute_force_alert)
            
            privilege_alert = await self._detect_privilege_escalation(audit_log)
            if privilege_alert:
                alerts.append(privilege_alert)
            
            exfiltration_alert = await self._detect_data_exfiltration(audit_log)
            if exfiltration_alert:
                alerts.append(exfiltration_alert)
            
            anomaly_alert = await self._detect_anomalous_access(audit_log)
            if anomaly_alert:
                alerts.append(anomaly_alert)
            
            threat_intel_alert = await self._check_threat_intelligence(audit_log)
            if threat_intel_alert:
                alerts.append(threat_intel_alert)
            
            # Send to SIEM platforms
            await self.siem_engine.send_security_event(audit_log)
            
            # Store alerts
            for alert in alerts:
                await self._store_alert(alert)
            
            # Update security metrics
            await self._update_security_metrics(audit_log, alerts)
            
        except Exception as e:
            logger.error(f"Error processing security event: {str(e)}")
        
        return alerts
    
    async def _detect_brute_force(self, audit_log: AuditLog) -> Optional[SecurityAlert]:
        """Detect brute force attacks"""
        if audit_log.event_type != 'authentication' or audit_log.result_status != 'failure':
            return None
        
        try:
            with get_database_session() as db:
                # Count failed logins in the last 5 minutes
                threshold_time = datetime.utcnow() - timedelta(
                    seconds=self.threat_rules['brute_force']['window']
                )
                
                failed_attempts = db.query(func.count(AuditLog.id)).filter(
                    and_(
                        AuditLog.timestamp >= threshold_time,
                        AuditLog.event_type == 'authentication',
                        AuditLog.result_status == 'failure',
                        or_(
                            AuditLog.user_id == audit_log.user_id,
                            AuditLog.ip_address == audit_log.ip_address
                        )
                    )
                ).scalar()
                
                if failed_attempts >= self.threat_rules['brute_force']['threshold']:
                    return SecurityAlert(
                        title="Brute Force Attack Detected",
                        description=f"Multiple failed login attempts detected from {audit_log.ip_address}",
                        severity=self.threat_rules['brute_force']['severity'],
                        alert_type="brute_force",
                        source_system="security_monitoring",
                        affected_resources=[audit_log.user_id, audit_log.ip_address],
                        indicators={
                            'failed_attempts': failed_attempts,
                            'ip_address': audit_log.ip_address,
                            'user_id': audit_log.user_id,
                            'time_window': self.threat_rules['brute_force']['window']
                        }
                    )
        except Exception as e:
            logger.error(f"Error detecting brute force: {str(e)}")
        
        return None
    
    async def _detect_privilege_escalation(self, audit_log: AuditLog) -> Optional[SecurityAlert]:
        """Detect privilege escalation attempts"""
        if not audit_log.details:
            return None
        
        try:
            details_str = json.dumps(audit_log.details).lower()
            patterns = self.threat_rules['privilege_escalation']['patterns']
            
            for pattern in patterns:
                if pattern in details_str:
                    return SecurityAlert(
                        title="Privilege Escalation Detected",
                        description=f"Potential privilege escalation attempt by {audit_log.user_id}",
                        severity=self.threat_rules['privilege_escalation']['severity'],
                        alert_type="privilege_escalation",
                        source_system="security_monitoring",
                        affected_resources=[audit_log.user_id],
                        indicators={
                            'pattern_matched': pattern,
                            'user_id': audit_log.user_id,
                            'action': audit_log.action_performed,
                            'details': audit_log.details
                        }
                    )
        except Exception as e:
            logger.error(f"Error detecting privilege escalation: {str(e)}")
        
        return None
    
    async def _detect_data_exfiltration(self, audit_log: AuditLog) -> Optional[SecurityAlert]:
        """Detect potential data exfiltration"""
        if audit_log.event_type != 'data_access':
            return None
        
        try:
            with get_database_session() as db:
                # Check data volume accessed in the last hour
                threshold_time = datetime.utcnow() - timedelta(
                    seconds=self.threat_rules['data_exfiltration']['window']
                )
                
                # This would need to be enhanced based on actual data volume tracking
                recent_accesses = db.query(func.count(AuditLog.id)).filter(
                    and_(
                        AuditLog.timestamp >= threshold_time,
                        AuditLog.event_type == 'data_access',
                        AuditLog.user_id == audit_log.user_id
                    )
                ).scalar()
                
                # Simple heuristic - too many data access events
                if recent_accesses > 50:  # Configurable threshold
                    return SecurityAlert(
                        title="Potential Data Exfiltration",
                        description=f"Unusual data access pattern detected for {audit_log.user_id}",
                        severity=self.threat_rules['data_exfiltration']['severity'],
                        alert_type="data_exfiltration",
                        source_system="security_monitoring",
                        affected_resources=[audit_log.user_id],
                        indicators={
                            'access_count': recent_accesses,
                            'user_id': audit_log.user_id,
                            'time_window': self.threat_rules['data_exfiltration']['window']
                        }
                    )
        except Exception as e:
            logger.error(f"Error detecting data exfiltration: {str(e)}")
        
        return None
    
    async def _detect_anomalous_access(self, audit_log: AuditLog) -> Optional[SecurityAlert]:
        """Detect anomalous access patterns"""
        if not self.baseline_metrics:
            return None
        
        try:
            # Check for unusual access times, locations, etc.
            current_hour = audit_log.timestamp.hour
            
            # Simple anomaly detection based on time
            if current_hour < 6 or current_hour > 22:  # Outside business hours
                return SecurityAlert(
                    title="Off-Hours Access Detected",
                    description=f"Access detected outside normal business hours by {audit_log.user_id}",
                    severity="medium",
                    alert_type="anomalous_access",
                    source_system="security_monitoring",
                    affected_resources=[audit_log.user_id],
                    indicators={
                        'access_time': audit_log.timestamp.isoformat(),
                        'user_id': audit_log.user_id,
                        'hour': current_hour
                    }
                )
        except Exception as e:
            logger.error(f"Error detecting anomalous access: {str(e)}")
        
        return None
    
    async def _check_threat_intelligence(self, audit_log: AuditLog) -> Optional[SecurityAlert]:
        """Check against threat intelligence indicators"""
        if not audit_log.ip_address:
            return None
        
        try:
            # Check IP against threat intelligence
            for ioc in self.threat_iocs.get('ips', []):
                if ioc['value'] == audit_log.ip_address:
                    return SecurityAlert(
                        title="Threat Intelligence Match",
                        description=f"Activity from known malicious IP: {audit_log.ip_address}",
                        severity="critical",
                        alert_type="threat_intelligence",
                        source_system="security_monitoring",
                        affected_resources=[audit_log.ip_address, audit_log.user_id],
                        indicators={
                            'ioc_type': 'ip',
                            'ioc_value': ioc['value'],
                            'confidence': ioc['confidence'],
                            'threat_type': ioc['threat_type'],
                            'user_id': audit_log.user_id
                        }
                    )
        except Exception as e:
            logger.error(f"Error checking threat intelligence: {str(e)}")
        
        return None
    
    async def _store_alert(self, alert: SecurityAlert):
        """Store security alert in database"""
        try:
            with get_database_session() as db:
                db.add(alert)
                db.commit()
                
                # Add to active alerts for correlation
                self.active_alerts[alert.alert_id] = alert
                
        except Exception as e:
            logger.error(f"Error storing alert: {str(e)}")
    
    async def _update_security_metrics(self, audit_log: AuditLog, alerts: List[SecurityAlert]):
        """Update security metrics"""
        try:
            with get_database_session() as db:
                # Event count metric
                event_metric = SecurityMetrics(
                    metric_name="security_events_total",
                    metric_value="1",
                    metric_type="counter",
                    tags={
                        'event_type': audit_log.event_type,
                        'severity': audit_log.severity,
                        'result': audit_log.result_status
                    }
                )
                db.add(event_metric)
                
                # Alert count metric
                if alerts:
                    alert_metric = SecurityMetrics(
                        metric_name="security_alerts_total",
                        metric_value=str(len(alerts)),
                        metric_type="counter",
                        tags={
                            'severity': alerts[0].severity if alerts else 'unknown'
                        }
                    )
                    db.add(alert_metric)
                
                db.commit()
                
        except Exception as e:
            logger.error(f"Error updating security metrics: {str(e)}")
    
    async def correlate_alerts(self) -> List[SecurityAlert]:
        """Correlate related security alerts"""
        correlated_alerts = []
        
        try:
            # Simple correlation based on time and resources
            time_window = timedelta(minutes=15)
            current_time = datetime.utcnow()
            
            # Group alerts by affected resources
            resource_groups = defaultdict(list)
            for alert_id, alert in self.active_alerts.items():
                if current_time - alert.created_at <= time_window:
                    for resource in alert.affected_resources or []:
                        resource_groups[resource].append(alert)
            
            # Create correlation alerts for resources with multiple alerts
            for resource, alerts in resource_groups.items():
                if len(alerts) > 1:
                    correlation_alert = SecurityAlert(
                        title="Correlated Security Incident",
                        description=f"Multiple security alerts detected for {resource}",
                        severity="high",
                        alert_type="correlation",
                        source_system="security_monitoring",
                        affected_resources=[resource],
                        indicators={
                            'correlated_alerts': [alert.alert_id for alert in alerts],
                            'alert_count': len(alerts),
                            'resource': resource
                        }
                    )
                    correlated_alerts.append(correlation_alert)
            
        except Exception as e:
            logger.error(f"Error correlating alerts: {str(e)}")
        
        return correlated_alerts
    
    async def generate_security_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        try:
            with get_database_session() as db:
                # Event statistics
                total_events = db.query(func.count(AuditLog.id)).filter(
                    and_(
                        AuditLog.timestamp >= start_date,
                        AuditLog.timestamp <= end_date
                    )
                ).scalar()
                
                # Alert statistics
                total_alerts = db.query(func.count(SecurityAlert.id)).filter(
                    and_(
                        SecurityAlert.created_at >= start_date,
                        SecurityAlert.created_at <= end_date
                    )
                ).scalar()
                
                # Top threat types
                threat_types = db.query(
                    SecurityAlert.alert_type,
                    func.count(SecurityAlert.id).label('count')
                ).filter(
                    and_(
                        SecurityAlert.created_at >= start_date,
                        SecurityAlert.created_at <= end_date
                    )
                ).group_by(SecurityAlert.alert_type).all()
                
                # Risk score distribution
                risk_distribution = db.query(
                    AuditLog.risk_score,
                    func.count(AuditLog.id).label('count')
                ).filter(
                    and_(
                        AuditLog.timestamp >= start_date,
                        AuditLog.timestamp <= end_date
                    )
                ).group_by(AuditLog.risk_score).all()
                
                return {
                    'period': {
                        'start': start_date.isoformat(),
                        'end': end_date.isoformat()
                    },
                    'summary': {
                        'total_events': total_events,
                        'total_alerts': total_alerts,
                        'alert_rate': (total_alerts / total_events * 100) if total_events > 0 else 0
                    },
                    'threat_types': [
                        {'type': row.alert_type, 'count': row.count}
                        for row in threat_types
                    ],
                    'risk_distribution': [
                        {'score': row.risk_score, 'count': row.count}
                        for row in risk_distribution
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error generating security report: {str(e)}")
            return {}
    
    async def update_threat_intelligence(self, ioc_data: Dict[str, Any]) -> bool:
        """Update threat intelligence with new IOCs"""
        try:
            with get_database_session() as db:
                threat_intel = ThreatIntelligence(
                    ioc_type=ioc_data['type'],
                    ioc_value=ioc_data['value'],
                    threat_type=ioc_data.get('threat_type'),
                    confidence_level=ioc_data.get('confidence', 50),
                    source=ioc_data.get('source', 'manual'),
                    description=ioc_data.get('description'),
                    tags=ioc_data.get('tags', {})
                )
                
                db.add(threat_intel)
                db.commit()
                
                # Reload threat intelligence
                self._load_threat_intelligence()
                
                return True
                
        except Exception as e:
            logger.error(f"Error updating threat intelligence: {str(e)}")
            return False