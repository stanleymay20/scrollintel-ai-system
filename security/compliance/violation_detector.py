"""
Automated Compliance Violation Detection with Remediation Workflow Triggers
Real-time monitoring and automated response to compliance violations
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from pathlib import Path
import threading
import queue
import uuid
from concurrent.futures import ThreadPoolExecutor
# Email imports removed for demo simplicity


class ViolationSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ViolationStatus(Enum):
    DETECTED = "detected"
    ACKNOWLEDGED = "acknowledged"
    IN_REMEDIATION = "in_remediation"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"


class RemediationStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ComplianceViolation:
    """Compliance violation record"""
    violation_id: str
    control_id: str
    framework: str
    violation_type: str
    severity: ViolationSeverity
    status: ViolationStatus
    title: str
    description: str
    detected_at: datetime
    detection_method: str
    affected_resources: List[str]
    evidence: Dict[str, Any]
    remediation_steps: List[str]
    assigned_to: Optional[str]
    due_date: Optional[datetime]
    resolved_at: Optional[datetime]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['severity'] = self.severity.value
        data['status'] = self.status.value
        data['detected_at'] = self.detected_at.isoformat()
        data['due_date'] = self.due_date.isoformat() if self.due_date else None
        data['resolved_at'] = self.resolved_at.isoformat() if self.resolved_at else None
        return data


@dataclass
class RemediationWorkflow:
    """Remediation workflow definition"""
    workflow_id: str
    violation_type: str
    severity: ViolationSeverity
    automated_steps: List[Dict[str, Any]]
    manual_steps: List[Dict[str, Any]]
    escalation_rules: Dict[str, Any]
    sla_hours: int
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['severity'] = self.severity.value
        return data


class ComplianceViolationDetector:
    """
    Automated compliance violation detection and remediation system
    """
    
    def __init__(self, db_path: str = "security/compliance.db"):
        self.db_path = db_path
        self.violation_queue = queue.Queue()
        self.remediation_queue = queue.Queue()
        self.running = False
        self.detection_thread = None
        self.remediation_thread = None
        
        self._init_database()
        self._init_detection_rules()
        self._init_remediation_workflows()
        self._init_notification_system()
    
    def _init_database(self):
        """Initialize violation tracking database"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance_violations (
                    violation_id TEXT PRIMARY KEY,
                    control_id TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    violation_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    detected_at DATETIME NOT NULL,
                    detection_method TEXT NOT NULL,
                    affected_resources_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    remediation_steps_json TEXT NOT NULL,
                    assigned_to TEXT,
                    due_date DATETIME,
                    resolved_at DATETIME,
                    metadata_json TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def _init_detection_rules(self):
        """Initialize compliance violation detection rules"""
        self.detection_rules = {
            'failed_login_threshold': {
                'description': 'Detect excessive failed login attempts',
                'threshold': 5,
                'time_window_minutes': 15,
                'severity': ViolationSeverity.HIGH,
                'control_ids': ['SOC2-CC2.1', 'ISO27001-A.9.1.1']
            }
        }
    
    def _init_remediation_workflows(self):
        """Initialize remediation workflows"""
        pass  # Simplified for demo
    
    def _init_notification_system(self):
        """Initialize notification system"""
        self.notification_config = {
            'smtp_server': 'localhost',
            'smtp_port': 587,
            'sender_email': 'security@company.com'
        }
    
    def start_monitoring(self):
        """Start violation detection monitoring"""
        self.running = True
        print("Violation detection monitoring started")
    
    def stop_monitoring(self):
        """Stop violation detection monitoring"""
        self.running = False
        print("Violation detection monitoring stopped")
    
    def get_violations_dashboard(self) -> Dict[str, Any]:
        """Get violations dashboard data"""
        return {
            'metrics': {
                'total_violations': 0,
                'open_violations': 0,
                'resolved_violations': 0,
                'critical_open': 0
            },
            'by_severity': {},
            'recent_violations': []
        }
        self.remediation_queue = queue.Queue()
        self.running = False
        self.detection_thread = None
        self.remediation_thread = None
        
        self._init_database()
        self._init_detection_rules()
        self._init_remediation_workflows()
        self._init_notification_system()
    
    def _init_database(self):
        """Initialize violation tracking database"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance_violations (
                    violation_id TEXT PRIMARY KEY,
                    control_id TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    violation_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    detected_at DATETIME NOT NULL,
                    detection_method TEXT NOT NULL,
                    affected_resources_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    remediation_steps_json TEXT NOT NULL,
                    assigned_to TEXT,
                    due_date DATETIME,
                    resolved_at DATETIME,
                    metadata_json TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS remediation_workflows (
                    workflow_id TEXT PRIMARY KEY,
                    violation_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    automated_steps_json TEXT NOT NULL,
                    manual_steps_json TEXT NOT NULL,
                    escalation_rules_json TEXT NOT NULL,
                    sla_hours INTEGER NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS violation_remediation_log (
                    log_id TEXT PRIMARY KEY,
                    violation_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    action_description TEXT NOT NULL,
                    performed_by TEXT NOT NULL,
                    performed_at DATETIME NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    FOREIGN KEY (violation_id) REFERENCES compliance_violations (violation_id)
                )
            """)
    
    def _init_detection_rules(self):
        """Initialize compliance violation detection rules"""
        self.detection_rules = {
            'failed_login_threshold': {
                'description': 'Detect excessive failed login attempts',
                'threshold': 5,
                'time_window_minutes': 15,
                'severity': ViolationSeverity.HIGH,
                'control_ids': ['SOC2-CC2.1', 'ISO27001-A.9.1.1']
            },
            'privileged_access_without_mfa': {
                'description': 'Detect privileged access without MFA',
                'severity': ViolationSeverity.CRITICAL,
                'control_ids': ['SOC2-CC3.1', 'HIPAA-164.312']
            },
            'unencrypted_data_transmission': {
                'description': 'Detect unencrypted data transmission',
                'severity': ViolationSeverity.HIGH,
                'control_ids': ['GDPR-ART32', 'HIPAA-164.312']
            },
            'unauthorized_data_access': {
                'description': 'Detect unauthorized data access attempts',
                'severity': ViolationSeverity.CRITICAL,
                'control_ids': ['GDPR-ART5', 'HIPAA-164.308']
            },
            'missing_security_patches': {
                'description': 'Detect systems with missing security patches',
                'severity': ViolationSeverity.MEDIUM,
                'control_ids': ['SOC2-CC3.1', 'ISO27001-A.12.6.1']
            },
            'data_retention_violation': {
                'description': 'Detect data retention policy violations',
                'severity': ViolationSeverity.MEDIUM,
                'control_ids': ['GDPR-ART5', 'HIPAA-164.308']
            }
        }
    
    def _init_remediation_workflows(self):
        """Initialize remediation workflows"""
        workflows = [
            RemediationWorkflow(
                workflow_id="failed_login_remediation",
                violation_type="failed_login_threshold",
                severity=ViolationSeverity.HIGH,
                automated_steps=[
                    {
                        'step': 'block_ip',
                        'description': 'Temporarily block source IP',
                        'timeout_minutes': 60
                    },
                    {
                        'step': 'notify_user',
                        'description': 'Send security alert to user',
                        'template': 'failed_login_alert'
                    }
                ],
                manual_steps=[
                    {
                        'step': 'investigate_source',
                        'description': 'Investigate the source of failed attempts',
                        'assigned_role': 'security_analyst'
                    },
                    {
                        'step': 'review_account',
                        'description': 'Review target account for compromise',
                        'assigned_role': 'security_analyst'
                    }
                ],
                escalation_rules={
                    'escalate_after_hours': 4,
                    'escalate_to': 'security_manager'
                },
                sla_hours=2
            ),
            RemediationWorkflow(
                workflow_id="privileged_access_mfa_remediation",
                violation_type="privileged_access_without_mfa",
                severity=ViolationSeverity.CRITICAL,
                automated_steps=[
                    {
                        'step': 'revoke_session',
                        'description': 'Immediately revoke privileged session',
                        'immediate': True
                    },
                    {
                        'step': 'require_mfa',
                        'description': 'Force MFA enrollment for account',
                        'immediate': True
                    }
                ],
                manual_steps=[
                    {
                        'step': 'verify_user',
                        'description': 'Verify user identity through alternate channel',
                        'assigned_role': 'security_analyst'
                    }
                ],
                escalation_rules={
                    'escalate_after_hours': 1,
                    'escalate_to': 'ciso'
                },
                sla_hours=1
            )
        ]
        
        # Store workflows in database
        for workflow in workflows:
            self._store_remediation_workflow(workflow)
    
    def _init_notification_system(self):
        """Initialize notification system"""
        self.notification_config = {
            'smtp_server': 'localhost',
            'smtp_port': 587,
            'sender_email': 'security@company.com',
            'security_team_email': 'security-team@company.com',
            'ciso_email': 'ciso@company.com'
        }
    
    def start_monitoring(self):
        """Start violation detection and remediation monitoring"""
        if self.running:
            return
        
        self.running = True
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self._detection_worker)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        # Start remediation thread
        self.remediation_thread = threading.Thread(target=self._remediation_worker)
        self.remediation_thread.daemon = True
        self.remediation_thread.start()
    
    def stop_monitoring(self):
        """Stop violation detection and remediation monitoring"""
        self.running = False
        
        if self.detection_thread:
            self.detection_thread.join(timeout=5)
        
        if self.remediation_thread:
            self.remediation_thread.join(timeout=5)
    
    def _detection_worker(self):
        """Background worker for violation detection"""
        while self.running:
            try:
                # Check for violations using various detection methods
                self._check_audit_log_violations()
                self._check_access_pattern_violations()
                self._check_configuration_violations()
                self._check_data_handling_violations()
                
                # Sleep before next check
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"Error in violation detection: {str(e)}")
                time.sleep(60)
    
    def _remediation_worker(self):
        """Background worker for remediation processing"""
        while self.running:
            try:
                violation = self.remediation_queue.get(timeout=5)
                self._process_remediation(violation)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in remediation processing: {str(e)}")
    
    def _check_audit_log_violations(self):
        """Check audit logs for compliance violations"""
        try:
            # Check for failed login threshold violations
            with sqlite3.connect("security/audit_blockchain.db") as conn:
                cursor = conn.execute("""
                    SELECT source_ip, COUNT(*) as failed_attempts
                    FROM audit_events 
                    WHERE event_type = 'login_failed' 
                    AND timestamp > ? 
                    GROUP BY source_ip
                    HAVING COUNT(*) >= ?
                """, (
                    (datetime.utcnow() - timedelta(minutes=15)).timestamp(),
                    self.detection_rules['failed_login_threshold']['threshold']
                ))
                
                for row in cursor.fetchall():
                    source_ip, failed_attempts = row
                    
                    violation = ComplianceViolation(
                        violation_id=str(uuid.uuid4()),
                        control_id='SOC2-CC2.1',
                        framework='SOC2_TYPE_II',
                        violation_type='failed_login_threshold',
                        severity=ViolationSeverity.HIGH,
                        status=ViolationStatus.DETECTED,
                        title=f"Excessive Failed Login Attempts from {source_ip}",
                        description=f"Detected {failed_attempts} failed login attempts from IP {source_ip} within 15 minutes",
                        detected_at=datetime.utcnow(),
                        detection_method='audit_log_analysis',
                        affected_resources=[source_ip],
                        evidence={
                            'source_ip': source_ip,
                            'failed_attempts': failed_attempts,
                            'time_window': '15 minutes'
                        },
                        remediation_steps=[
                            'Block source IP temporarily',
                            'Investigate source of attacks',
                            'Review affected user accounts'
                        ],
                        assigned_to=None,
                        due_date=datetime.utcnow() + timedelta(hours=2),
                        resolved_at=None,
                        metadata={'detection_rule': 'failed_login_threshold'}
                    )
                    
                    self._create_violation(violation)
        
        except Exception as e:
            print(f"Error checking audit log violations: {str(e)}")
    
    def _check_access_pattern_violations(self):
        """Check for access pattern violations"""
        try:
            # Check for privileged access without MFA
            with sqlite3.connect("security/audit_blockchain.db") as conn:
                cursor = conn.execute("""
                    SELECT user_id, resource, COUNT(*) as access_count
                    FROM audit_events 
                    WHERE event_type = 'privileged_access' 
                    AND timestamp > ? 
                    AND details_json NOT LIKE '%"mfa_verified": true%'
                    GROUP BY user_id, resource
                """, ((datetime.utcnow() - timedelta(hours=1)).timestamp(),))
                
                for row in cursor.fetchall():
                    user_id, resource, access_count = row
                    
                    violation = ComplianceViolation(
                        violation_id=str(uuid.uuid4()),
                        control_id='SOC2-CC3.1',
                        framework='SOC2_TYPE_II',
                        violation_type='privileged_access_without_mfa',
                        severity=ViolationSeverity.CRITICAL,
                        status=ViolationStatus.DETECTED,
                        title=f"Privileged Access Without MFA - User {user_id}",
                        description=f"User {user_id} accessed privileged resource {resource} without MFA verification",
                        detected_at=datetime.utcnow(),
                        detection_method='access_pattern_analysis',
                        affected_resources=[resource],
                        evidence={
                            'user_id': user_id,
                            'resource': resource,
                            'access_count': access_count,
                            'mfa_verified': False
                        },
                        remediation_steps=[
                            'Revoke current session',
                            'Require MFA enrollment',
                            'Verify user identity'
                        ],
                        assigned_to=None,
                        due_date=datetime.utcnow() + timedelta(hours=1),
                        resolved_at=None,
                        metadata={'detection_rule': 'privileged_access_without_mfa'}
                    )
                    
                    self._create_violation(violation)
        
        except Exception as e:
            print(f"Error checking access pattern violations: {str(e)}")
    
    def _check_configuration_violations(self):
        """Check for configuration-based violations"""
        # This would check system configurations for compliance violations
        # Implementation would depend on specific configuration management system
        pass
    
    def _check_data_handling_violations(self):
        """Check for data handling violations"""
        # This would check for data retention, encryption, and privacy violations
        # Implementation would integrate with data classification and protection systems
        pass
    
    def _create_violation(self, violation: ComplianceViolation):
        """Create a new compliance violation"""
        # Check if similar violation already exists
        if self._violation_exists(violation):
            return
        
        # Store violation in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO compliance_violations 
                (violation_id, control_id, framework, violation_type, severity, status,
                 title, description, detected_at, detection_method, affected_resources_json,
                 evidence_json, remediation_steps_json, due_date, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                violation.violation_id,
                violation.control_id,
                violation.framework,
                violation.violation_type,
                violation.severity.value,
                violation.status.value,
                violation.title,
                violation.description,
                violation.detected_at,
                violation.detection_method,
                json.dumps(violation.affected_resources),
                json.dumps(violation.evidence),
                json.dumps(violation.remediation_steps),
                violation.due_date,
                json.dumps(violation.metadata)
            ))
        
        # Add to remediation queue
        self.remediation_queue.put(violation)
        
        # Send notification
        self._send_violation_notification(violation)
    
    def _violation_exists(self, violation: ComplianceViolation) -> bool:
        """Check if similar violation already exists"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM compliance_violations 
                WHERE violation_type = ? 
                AND affected_resources_json = ? 
                AND status NOT IN ('resolved', 'false_positive')
                AND detected_at > ?
            """, (
                violation.violation_type,
                json.dumps(violation.affected_resources),
                (datetime.utcnow() - timedelta(hours=24)).isoformat()
            ))
            
            return cursor.fetchone()[0] > 0
    
    def _process_remediation(self, violation: ComplianceViolation):
        """Process remediation for a violation"""
        try:
            # Get remediation workflow
            workflow = self._get_remediation_workflow(violation.violation_type, violation.severity)
            
            if not workflow:
                print(f"No remediation workflow found for {violation.violation_type}")
                return
            
            # Execute automated steps
            for step in workflow.automated_steps:
                try:
                    result = self._execute_automated_step(violation, step)
                    self._log_remediation_action(
                        violation.violation_id,
                        'automated',
                        step['description'],
                        'system',
                        'completed' if result else 'failed',
                        {'step': step, 'result': result}
                    )
                except Exception as e:
                    self._log_remediation_action(
                        violation.violation_id,
                        'automated',
                        step['description'],
                        'system',
                        'failed',
                        {'step': step, 'error': str(e)}
                    )
            
            # Create manual tasks
            for step in workflow.manual_steps:
                self._create_manual_task(violation, step)
            
            # Update violation status
            self._update_violation_status(violation.violation_id, ViolationStatus.IN_REMEDIATION)
            
        except Exception as e:
            print(f"Error processing remediation for {violation.violation_id}: {str(e)}")
    
    def _execute_automated_step(self, violation: ComplianceViolation, step: Dict[str, Any]) -> bool:
        """Execute an automated remediation step"""
        step_type = step.get('step')
        
        if step_type == 'block_ip':
            return self._block_ip_address(violation.affected_resources[0])
        elif step_type == 'revoke_session':
            return self._revoke_user_session(violation.evidence.get('user_id'))
        elif step_type == 'notify_user':
            return self._notify_user(violation)
        elif step_type == 'require_mfa':
            return self._require_mfa_enrollment(violation.evidence.get('user_id'))
        
        return False
    
    def _block_ip_address(self, ip_address: str) -> bool:
        """Block an IP address (mock implementation)"""
        # In real implementation, this would integrate with firewall/WAF
        print(f"Blocking IP address: {ip_address}")
        return True
    
    def _revoke_user_session(self, user_id: str) -> bool:
        """Revoke user session (mock implementation)"""
        # In real implementation, this would integrate with session management
        print(f"Revoking session for user: {user_id}")
        return True
    
    def _notify_user(self, violation: ComplianceViolation) -> bool:
        """Send notification to user (mock implementation)"""
        print(f"Sending notification for violation: {violation.title}")
        return True
    
    def _require_mfa_enrollment(self, user_id: str) -> bool:
        """Require MFA enrollment for user (mock implementation)"""
        print(f"Requiring MFA enrollment for user: {user_id}")
        return True
    
    def _create_manual_task(self, violation: ComplianceViolation, step: Dict[str, Any]):
        """Create manual remediation task"""
        # In real implementation, this would integrate with task management system
        print(f"Creating manual task: {step['description']} for {step.get('assigned_role', 'security_team')}")
    
    def _get_remediation_workflow(self, violation_type: str, severity: ViolationSeverity) -> Optional[RemediationWorkflow]:
        """Get remediation workflow for violation type and severity"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT workflow_id, violation_type, severity, automated_steps_json,
                       manual_steps_json, escalation_rules_json, sla_hours
                FROM remediation_workflows 
                WHERE violation_type = ? AND severity = ?
            """, (violation_type, severity.value))
            
            row = cursor.fetchone()
            if row:
                return RemediationWorkflow(
                    workflow_id=row[0],
                    violation_type=row[1],
                    severity=ViolationSeverity(row[2]),
                    automated_steps=json.loads(row[3]),
                    manual_steps=json.loads(row[4]),
                    escalation_rules=json.loads(row[5]),
                    sla_hours=row[6]
                )
        
        return None
    
    def _store_remediation_workflow(self, workflow: RemediationWorkflow):
        """Store remediation workflow in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO remediation_workflows 
                (workflow_id, violation_type, severity, automated_steps_json,
                 manual_steps_json, escalation_rules_json, sla_hours)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                workflow.workflow_id,
                workflow.violation_type,
                workflow.severity.value,
                json.dumps(workflow.automated_steps),
                json.dumps(workflow.manual_steps),
                json.dumps(workflow.escalation_rules),
                workflow.sla_hours
            ))
    
    def _log_remediation_action(self, 
                              violation_id: str, 
                              action_type: str, 
                              description: str, 
                              performed_by: str, 
                              status: str, 
                              result: Optional[Dict[str, Any]] = None):
        """Log remediation action"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO violation_remediation_log 
                (log_id, violation_id, action_type, action_description, 
                 performed_by, performed_at, status, result_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()),
                violation_id,
                action_type,
                description,
                performed_by,
                datetime.utcnow(),
                status,
                json.dumps(result) if result else None
            ))
    
    def _update_violation_status(self, violation_id: str, status: ViolationStatus):
        """Update violation status"""
        with sqlite3.connect(self.db_path) as conn:
            resolved_at = datetime.utcnow() if status == ViolationStatus.RESOLVED else None
            
            conn.execute("""
                UPDATE compliance_violations 
                SET status = ?, resolved_at = ?, updated_at = ?
                WHERE violation_id = ?
            """, (status.value, resolved_at, datetime.utcnow(), violation_id))
    
    def _send_violation_notification(self, violation: ComplianceViolation):
        """Send violation notification"""
        try:
            subject = f"Compliance Violation Detected: {violation.title}"
            body = f"""
            A compliance violation has been detected:
            
            Violation ID: {violation.violation_id}
            Control: {violation.control_id}
            Framework: {violation.framework}
            Severity: {violation.severity.value.upper()}
            
            Description: {violation.description}
            
            Affected Resources: {', '.join(violation.affected_resources)}
            
            Remediation Steps:
            {chr(10).join(f"- {step}" for step in violation.remediation_steps)}
            
            Due Date: {violation.due_date.strftime('%Y-%m-%d %H:%M:%S') if violation.due_date else 'N/A'}
            
            Please review and take appropriate action.
            """
            
            # In real implementation, this would send actual email
            print(f"Notification sent: {subject}")
            
        except Exception as e:
            print(f"Error sending violation notification: {str(e)}")
    
    def get_violations_dashboard(self) -> Dict[str, Any]:
        """Get violations dashboard data"""
        with sqlite3.connect(self.db_path) as conn:
            # Overall metrics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_violations,
                    SUM(CASE WHEN status = 'detected' THEN 1 ELSE 0 END) as open_violations,
                    SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) as resolved_violations,
                    SUM(CASE WHEN severity = 'critical' AND status != 'resolved' THEN 1 ELSE 0 END) as critical_open
                FROM compliance_violations
                WHERE detected_at > ?
            """, ((datetime.utcnow() - timedelta(days=30)).isoformat(),))
            
            metrics = cursor.fetchone()
            
            # Violations by severity
            cursor = conn.execute("""
                SELECT severity, COUNT(*) as count
                FROM compliance_violations
                WHERE detected_at > ? AND status != 'resolved'
                GROUP BY severity
            """, ((datetime.utcnow() - timedelta(days=30)).isoformat(),))
            
            by_severity = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Recent violations
            cursor = conn.execute("""
                SELECT violation_id, title, severity, status, detected_at
                FROM compliance_violations
                ORDER BY detected_at DESC
                LIMIT 10
            """)
            
            recent_violations = [
                {
                    'violation_id': row[0],
                    'title': row[1],
                    'severity': row[2],
                    'status': row[3],
                    'detected_at': row[4]
                }
                for row in cursor.fetchall()
            ]
            
            return {
                'metrics': {
                    'total_violations': metrics[0],
                    'open_violations': metrics[1],
                    'resolved_violations': metrics[2],
                    'critical_open': metrics[3]
                },
                'by_severity': by_severity,
                'recent_violations': recent_violations
            }