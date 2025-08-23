"""
Vendor Security Incident Tracking and Management Workflows
Implements comprehensive incident tracking and response for vendor-related security events
"""

import asyncio
import json
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum

class IncidentSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class IncidentStatus(Enum):
    NEW = "new"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    CLOSED = "closed"
    ESCALATED = "escalated"

class IncidentCategory(Enum):
    DATA_BREACH = "data_breach"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    MALWARE_DETECTION = "malware_detection"
    VULNERABILITY_EXPLOIT = "vulnerability_exploit"
    SERVICE_DISRUPTION = "service_disruption"
    COMPLIANCE_VIOLATION = "compliance_violation"
    SUPPLY_CHAIN_COMPROMISE = "supply_chain_compromise"
    INSIDER_THREAT = "insider_threat"

class EscalationLevel(Enum):
    LEVEL_1 = "level_1"  # Security Analyst
    LEVEL_2 = "level_2"  # Senior Security Engineer
    LEVEL_3 = "level_3"  # Security Manager
    LEVEL_4 = "level_4"  # CISO/Executive

@dataclass
class IncidentEvidence:
    evidence_id: str
    incident_id: str
    evidence_type: str  # log, file, screenshot, network_capture, etc.
    description: str
    file_path: Optional[str]
    hash_sha256: Optional[str]
    collected_by: str
    collected_at: datetime
    chain_of_custody: List[Dict[str, Any]]

@dataclass
class IncidentAction:
    action_id: str
    incident_id: str
    action_type: str  # investigation, containment, eradication, recovery
    description: str
    assigned_to: str
    created_at: datetime
    due_date: Optional[datetime]
    completed_at: Optional[datetime]
    status: str
    results: Optional[str]

@dataclass
class VendorIncident:
    incident_id: str
    vendor_id: str
    vendor_name: str
    title: str
    description: str
    category: IncidentCategory
    severity: IncidentSeverity
    status: IncidentStatus
    reporter: str
    assigned_to: Optional[str]
    escalation_level: EscalationLevel
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime]
    closed_at: Optional[datetime]
    affected_systems: List[str]
    affected_data_types: List[str]
    business_impact: str
    technical_impact: str
    root_cause: Optional[str]
    lessons_learned: Optional[str]
    evidence: List[IncidentEvidence]
    actions: List[IncidentAction]
    timeline: List[Dict[str, Any]]
    notifications_sent: List[Dict[str, Any]]
    compliance_implications: List[str]
    estimated_cost: Optional[float]
    actual_cost: Optional[float]

class VendorIncidentTracker:
    def __init__(self, config_path: str = "security/config/incident_config.yaml"):
        self.config = self._load_config(config_path)
        self.incidents = {}  # incident_id -> VendorIncident
        self.incident_workflows = self._initialize_workflows()
        self.notification_templates = self._load_notification_templates()
        self.escalation_rules = self._load_escalation_rules()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load incident tracking configuration"""
        default_config = {
            "sla_response_times": {
                "critical": 15,  # minutes
                "high": 60,
                "medium": 240,
                "low": 1440
            },
            "sla_resolution_times": {
                "critical": 4,  # hours
                "high": 24,
                "medium": 72,
                "low": 168
            },
            "auto_escalation": {
                "enabled": True,
                "escalation_intervals": {
                    "critical": 30,  # minutes
                    "high": 120,
                    "medium": 480,
                    "low": 1440
                }
            },
            "notification_settings": {
                "immediate_notification": ["critical", "high"],
                "daily_digest": ["medium", "low"],
                "stakeholder_notification": {
                    "critical": ["ciso", "legal", "compliance", "vendor_manager"],
                    "high": ["security_manager", "vendor_manager"],
                    "medium": ["security_team"],
                    "low": ["assigned_analyst"]
                }
            },
            "compliance_frameworks": {
                "gdpr": {
                    "breach_notification_hours": 72,
                    "authority_notification_required": True
                },
                "sox": {
                    "financial_impact_threshold": 10000,
                    "audit_committee_notification": True
                },
                "hipaa": {
                    "phi_breach_notification_days": 60,
                    "hhs_notification_required": True
                }
            }
        }
        
        try:
            # In production, load from actual config file
            return default_config
        except Exception:
            return default_config
    
    def _initialize_workflows(self) -> Dict[str, Any]:
        """Initialize incident response workflows"""
        return {
            "data_breach": {
                "immediate_actions": [
                    "Contain the breach",
                    "Assess scope of data exposure",
                    "Notify legal and compliance teams",
                    "Preserve evidence"
                ],
                "investigation_steps": [
                    "Analyze attack vectors",
                    "Identify compromised systems",
                    "Determine data types affected",
                    "Interview relevant personnel"
                ],
                "recovery_actions": [
                    "Implement security patches",
                    "Reset compromised credentials",
                    "Monitor for continued threats",
                    "Validate system integrity"
                ]
            },
            "unauthorized_access": {
                "immediate_actions": [
                    "Revoke unauthorized access",
                    "Change affected credentials",
                    "Review access logs",
                    "Notify vendor security team"
                ],
                "investigation_steps": [
                    "Trace access patterns",
                    "Identify entry points",
                    "Assess privilege escalation",
                    "Review vendor security controls"
                ],
                "recovery_actions": [
                    "Implement additional access controls",
                    "Enhance monitoring",
                    "Update vendor agreements",
                    "Conduct security assessment"
                ]
            },
            "malware_detection": {
                "immediate_actions": [
                    "Isolate infected systems",
                    "Run malware analysis",
                    "Check for lateral movement",
                    "Preserve malware samples"
                ],
                "investigation_steps": [
                    "Analyze malware capabilities",
                    "Identify infection vectors",
                    "Assess data exfiltration",
                    "Review vendor software integrity"
                ],
                "recovery_actions": [
                    "Clean infected systems",
                    "Update security signatures",
                    "Implement additional scanning",
                    "Review vendor software supply chain"
                ]
            }
        }
    
    def _load_notification_templates(self) -> Dict[str, str]:
        """Load notification templates"""
        return {
            "incident_created": """
Security Incident Alert - {severity} Severity

Incident ID: {incident_id}
Vendor: {vendor_name}
Title: {title}
Category: {category}
Severity: {severity}
Reporter: {reporter}
Created: {created_at}

Description:
{description}

Immediate Actions Required:
{immediate_actions}

This incident requires immediate attention according to our SLA.
            """,
            "incident_escalated": """
Security Incident Escalated - {severity} Severity

Incident ID: {incident_id}
Vendor: {vendor_name}
Title: {title}
Escalated to: {escalation_level}
Previous Assignee: {previous_assignee}
Escalation Reason: {escalation_reason}

The incident has been escalated due to SLA breach or severity increase.
            """,
            "incident_resolved": """
Security Incident Resolved

Incident ID: {incident_id}
Vendor: {vendor_name}
Title: {title}
Resolution Time: {resolution_time}
Root Cause: {root_cause}

Summary of Actions Taken:
{actions_summary}

Lessons Learned:
{lessons_learned}
            """
        }
    
    def _load_escalation_rules(self) -> Dict[str, Any]:
        """Load escalation rules"""
        return {
            "time_based": {
                "critical": [
                    {"after_minutes": 15, "escalate_to": EscalationLevel.LEVEL_2},
                    {"after_minutes": 30, "escalate_to": EscalationLevel.LEVEL_3},
                    {"after_minutes": 60, "escalate_to": EscalationLevel.LEVEL_4}
                ],
                "high": [
                    {"after_minutes": 60, "escalate_to": EscalationLevel.LEVEL_2},
                    {"after_minutes": 240, "escalate_to": EscalationLevel.LEVEL_3}
                ],
                "medium": [
                    {"after_minutes": 480, "escalate_to": EscalationLevel.LEVEL_2}
                ]
            },
            "impact_based": {
                "financial_threshold": 50000,  # Escalate if estimated cost > $50k
                "customer_impact_threshold": 1000,  # Escalate if > 1000 customers affected
                "regulatory_impact": True  # Always escalate regulatory incidents
            }
        }
    
    async def create_incident(self, vendor_id: str, vendor_name: str, title: str,
                            description: str, category: IncidentCategory,
                            severity: IncidentSeverity, reporter: str,
                            affected_systems: Optional[List[str]] = None,
                            affected_data_types: Optional[List[str]] = None) -> VendorIncident:
        """Create new vendor security incident"""
        incident_id = self._generate_incident_id(vendor_id, category)
        
        # Determine initial escalation level
        escalation_level = self._determine_initial_escalation_level(severity, category)
        
        incident = VendorIncident(
            incident_id=incident_id,
            vendor_id=vendor_id,
            vendor_name=vendor_name,
            title=title,
            description=description,
            category=category,
            severity=severity,
            status=IncidentStatus.NEW,
            reporter=reporter,
            assigned_to=None,
            escalation_level=escalation_level,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            resolved_at=None,
            closed_at=None,
            affected_systems=affected_systems or [],
            affected_data_types=affected_data_types or [],
            business_impact="",
            technical_impact="",
            root_cause=None,
            lessons_learned=None,
            evidence=[],
            actions=[],
            timeline=[],
            notifications_sent=[],
            compliance_implications=[],
            estimated_cost=None,
            actual_cost=None
        )
        
        # Add initial timeline entry
        incident.timeline.append({
            "timestamp": datetime.now().isoformat(),
            "event": "incident_created",
            "description": f"Incident created by {reporter}",
            "user": reporter
        })
        
        # Store incident
        self.incidents[incident_id] = incident
        
        # Generate initial actions based on category
        await self._generate_initial_actions(incident)
        
        # Send notifications
        await self._send_incident_notifications(incident, "incident_created")
        
        # Start auto-escalation monitoring
        if self.config["auto_escalation"]["enabled"]:
            asyncio.create_task(self._monitor_escalation(incident_id))
        
        return incident
    
    def _determine_initial_escalation_level(self, severity: IncidentSeverity, 
                                          category: IncidentCategory) -> EscalationLevel:
        """Determine initial escalation level"""
        if severity == IncidentSeverity.CRITICAL:
            return EscalationLevel.LEVEL_3
        elif severity == IncidentSeverity.HIGH:
            return EscalationLevel.LEVEL_2
        elif category in [IncidentCategory.DATA_BREACH, IncidentCategory.SUPPLY_CHAIN_COMPROMISE]:
            return EscalationLevel.LEVEL_2
        else:
            return EscalationLevel.LEVEL_1
    
    async def _generate_initial_actions(self, incident: VendorIncident):
        """Generate initial response actions based on incident category"""
        workflow = self.incident_workflows.get(incident.category.value, {})
        immediate_actions = workflow.get("immediate_actions", [])
        
        for i, action_desc in enumerate(immediate_actions):
            action = IncidentAction(
                action_id=self._generate_action_id(incident.incident_id, f"initial_{i}"),
                incident_id=incident.incident_id,
                action_type="immediate_response",
                description=action_desc,
                assigned_to="security_team",
                created_at=datetime.now(),
                due_date=datetime.now() + timedelta(hours=1),  # 1 hour for immediate actions
                completed_at=None,
                status="pending",
                results=None
            )
            incident.actions.append(action)
    
    async def assign_incident(self, incident_id: str, assignee: str, assigner: str) -> bool:
        """Assign incident to analyst"""
        if incident_id not in self.incidents:
            return False
        
        incident = self.incidents[incident_id]
        incident.assigned_to = assignee
        incident.status = IncidentStatus.ASSIGNED
        incident.updated_at = datetime.now()
        
        # Add timeline entry
        incident.timeline.append({
            "timestamp": datetime.now().isoformat(),
            "event": "incident_assigned",
            "description": f"Incident assigned to {assignee} by {assigner}",
            "user": assigner
        })
        
        return True
    
    async def update_incident_status(self, incident_id: str, new_status: IncidentStatus,
                                   updater: str, notes: Optional[str] = None) -> bool:
        """Update incident status"""
        if incident_id not in self.incidents:
            return False
        
        incident = self.incidents[incident_id]
        old_status = incident.status
        incident.status = new_status
        incident.updated_at = datetime.now()
        
        # Set resolution/closure timestamps
        if new_status == IncidentStatus.RESOLVED:
            incident.resolved_at = datetime.now()
        elif new_status == IncidentStatus.CLOSED:
            incident.closed_at = datetime.now()
        
        # Add timeline entry
        timeline_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "status_changed",
            "description": f"Status changed from {old_status.value} to {new_status.value}",
            "user": updater
        }
        if notes:
            timeline_entry["notes"] = notes
        
        incident.timeline.append(timeline_entry)
        
        # Send notifications for significant status changes
        if new_status in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]:
            await self._send_incident_notifications(incident, "incident_resolved")
        
        return True
    
    async def add_evidence(self, incident_id: str, evidence_type: str, description: str,
                         file_path: Optional[str], collected_by: str) -> Optional[IncidentEvidence]:
        """Add evidence to incident"""
        if incident_id not in self.incidents:
            return None
        
        incident = self.incidents[incident_id]
        evidence_id = self._generate_evidence_id(incident_id, evidence_type)
        
        # Calculate file hash if file provided
        hash_sha256 = None
        if file_path:
            hash_sha256 = await self._calculate_file_hash(file_path)
        
        evidence = IncidentEvidence(
            evidence_id=evidence_id,
            incident_id=incident_id,
            evidence_type=evidence_type,
            description=description,
            file_path=file_path,
            hash_sha256=hash_sha256,
            collected_by=collected_by,
            collected_at=datetime.now(),
            chain_of_custody=[{
                "handler": collected_by,
                "action": "collected",
                "timestamp": datetime.now().isoformat()
            }]
        )
        
        incident.evidence.append(evidence)
        incident.updated_at = datetime.now()
        
        # Add timeline entry
        incident.timeline.append({
            "timestamp": datetime.now().isoformat(),
            "event": "evidence_added",
            "description": f"Evidence added: {evidence_type} - {description}",
            "user": collected_by
        })
        
        return evidence
    
    async def complete_action(self, incident_id: str, action_id: str, 
                            results: str, completed_by: str) -> bool:
        """Mark action as completed"""
        if incident_id not in self.incidents:
            return False
        
        incident = self.incidents[incident_id]
        
        # Find and update action
        for action in incident.actions:
            if action.action_id == action_id:
                action.completed_at = datetime.now()
                action.status = "completed"
                action.results = results
                
                # Add timeline entry
                incident.timeline.append({
                    "timestamp": datetime.now().isoformat(),
                    "event": "action_completed",
                    "description": f"Action completed: {action.description}",
                    "user": completed_by,
                    "results": results
                })
                
                incident.updated_at = datetime.now()
                return True
        
        return False
    
    async def escalate_incident(self, incident_id: str, escalation_reason: str,
                              escalated_by: str, target_level: Optional[EscalationLevel] = None) -> bool:
        """Escalate incident to higher level"""
        if incident_id not in self.incidents:
            return False
        
        incident = self.incidents[incident_id]
        previous_level = incident.escalation_level
        previous_assignee = incident.assigned_to
        
        # Determine target escalation level
        if target_level is None:
            target_level = self._get_next_escalation_level(incident.escalation_level)
        
        incident.escalation_level = target_level
        incident.status = IncidentStatus.ESCALATED
        incident.assigned_to = None  # Will be reassigned at new level
        incident.updated_at = datetime.now()
        
        # Add timeline entry
        incident.timeline.append({
            "timestamp": datetime.now().isoformat(),
            "event": "incident_escalated",
            "description": f"Escalated from {previous_level.value} to {target_level.value}",
            "user": escalated_by,
            "reason": escalation_reason
        })
        
        # Send escalation notifications
        await self._send_escalation_notifications(incident, escalation_reason, previous_assignee)
        
        return True
    
    def _get_next_escalation_level(self, current_level: EscalationLevel) -> EscalationLevel:
        """Get next escalation level"""
        level_order = [
            EscalationLevel.LEVEL_1,
            EscalationLevel.LEVEL_2,
            EscalationLevel.LEVEL_3,
            EscalationLevel.LEVEL_4
        ]
        
        try:
            current_index = level_order.index(current_level)
            if current_index < len(level_order) - 1:
                return level_order[current_index + 1]
        except ValueError:
            pass
        
        return EscalationLevel.LEVEL_4  # Maximum escalation
    
    async def _monitor_escalation(self, incident_id: str):
        """Monitor incident for auto-escalation"""
        while incident_id in self.incidents:
            incident = self.incidents[incident_id]
            
            # Skip if incident is resolved or closed
            if incident.status in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]:
                break
            
            # Check time-based escalation
            time_since_creation = datetime.now() - incident.created_at
            escalation_rules = self.escalation_rules["time_based"].get(incident.severity.value, [])
            
            for rule in escalation_rules:
                if (time_since_creation.total_seconds() / 60 >= rule["after_minutes"] and
                    incident.escalation_level.value < rule["escalate_to"].value):
                    
                    await self.escalate_incident(
                        incident_id,
                        f"Auto-escalation due to SLA breach ({rule['after_minutes']} minutes)",
                        "system_auto_escalation",
                        rule["escalate_to"]
                    )
                    break
            
            await asyncio.sleep(300)  # Check every 5 minutes
    
    async def _send_incident_notifications(self, incident: VendorIncident, notification_type: str):
        """Send incident notifications"""
        template = self.notification_templates.get(notification_type, "")
        
        # Get stakeholders to notify
        stakeholders = self._get_notification_stakeholders(incident)
        
        # Format notification
        notification_content = template.format(
            incident_id=incident.incident_id,
            vendor_name=incident.vendor_name,
            title=incident.title,
            category=incident.category.value,
            severity=incident.severity.value,
            reporter=incident.reporter,
            created_at=incident.created_at.isoformat(),
            description=incident.description,
            immediate_actions=self._format_immediate_actions(incident)
        )
        
        # Send notifications (in production, integrate with email/Slack/etc.)
        for stakeholder in stakeholders:
            notification_record = {
                "recipient": stakeholder,
                "type": notification_type,
                "sent_at": datetime.now().isoformat(),
                "content": notification_content
            }
            incident.notifications_sent.append(notification_record)
            
            # Simulate sending notification
            print(f"NOTIFICATION to {stakeholder}: {notification_type}")
    
    def _get_notification_stakeholders(self, incident: VendorIncident) -> List[str]:
        """Get list of stakeholders to notify"""
        severity_stakeholders = self.config["notification_settings"]["stakeholder_notification"]
        return severity_stakeholders.get(incident.severity.value, ["security_team"])
    
    def _format_immediate_actions(self, incident: VendorIncident) -> str:
        """Format immediate actions for notification"""
        immediate_actions = [a for a in incident.actions if a.action_type == "immediate_response"]
        return "\n".join([f"- {action.description}" for action in immediate_actions])
    
    async def _send_escalation_notifications(self, incident: VendorIncident, 
                                           escalation_reason: str, previous_assignee: Optional[str]):
        """Send escalation notifications"""
        template = self.notification_templates["incident_escalated"]
        
        notification_content = template.format(
            incident_id=incident.incident_id,
            vendor_name=incident.vendor_name,
            title=incident.title,
            severity=incident.severity.value,
            escalation_level=incident.escalation_level.value,
            previous_assignee=previous_assignee or "Unassigned",
            escalation_reason=escalation_reason
        )
        
        # Notify escalation level stakeholders
        escalation_stakeholders = self._get_escalation_stakeholders(incident.escalation_level)
        
        for stakeholder in escalation_stakeholders:
            notification_record = {
                "recipient": stakeholder,
                "type": "incident_escalated",
                "sent_at": datetime.now().isoformat(),
                "content": notification_content
            }
            incident.notifications_sent.append(notification_record)
    
    def _get_escalation_stakeholders(self, escalation_level: EscalationLevel) -> List[str]:
        """Get stakeholders for escalation level"""
        stakeholder_mapping = {
            EscalationLevel.LEVEL_1: ["security_analyst"],
            EscalationLevel.LEVEL_2: ["senior_security_engineer", "security_manager"],
            EscalationLevel.LEVEL_3: ["security_manager", "ciso"],
            EscalationLevel.LEVEL_4: ["ciso", "ceo", "legal", "compliance"]
        }
        return stakeholder_mapping.get(escalation_level, ["security_team"])
    
    async def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            return hashlib.sha256(content).hexdigest()
        except Exception:
            return "hash_calculation_failed"
    
    def get_incidents_by_vendor(self, vendor_id: str) -> List[VendorIncident]:
        """Get all incidents for a vendor"""
        return [incident for incident in self.incidents.values() 
                if incident.vendor_id == vendor_id]
    
    def get_open_incidents(self) -> List[VendorIncident]:
        """Get all open incidents"""
        open_statuses = [IncidentStatus.NEW, IncidentStatus.ASSIGNED, 
                        IncidentStatus.IN_PROGRESS, IncidentStatus.INVESTIGATING,
                        IncidentStatus.ESCALATED]
        return [incident for incident in self.incidents.values() 
                if incident.status in open_statuses]
    
    def get_incidents_by_severity(self, severity: IncidentSeverity) -> List[VendorIncident]:
        """Get incidents by severity level"""
        return [incident for incident in self.incidents.values() 
                if incident.severity == severity]
    
    async def generate_incident_report(self, incident_id: str) -> Dict[str, Any]:
        """Generate comprehensive incident report"""
        if incident_id not in self.incidents:
            return {"error": "Incident not found"}
        
        incident = self.incidents[incident_id]
        
        # Calculate metrics
        response_time = None
        resolution_time = None
        
        if incident.assigned_to and len(incident.timeline) > 1:
            # Find assignment time
            for entry in incident.timeline:
                if entry["event"] == "incident_assigned":
                    assignment_time = datetime.fromisoformat(entry["timestamp"])
                    response_time = (assignment_time - incident.created_at).total_seconds() / 60
                    break
        
        if incident.resolved_at:
            resolution_time = (incident.resolved_at - incident.created_at).total_seconds() / 3600
        
        # Compliance analysis
        compliance_status = await self._analyze_compliance_requirements(incident)
        
        return {
            "incident_summary": {
                "incident_id": incident.incident_id,
                "vendor_id": incident.vendor_id,
                "vendor_name": incident.vendor_name,
                "title": incident.title,
                "category": incident.category.value,
                "severity": incident.severity.value,
                "status": incident.status.value,
                "created_at": incident.created_at.isoformat(),
                "resolved_at": incident.resolved_at.isoformat() if incident.resolved_at else None,
                "closed_at": incident.closed_at.isoformat() if incident.closed_at else None
            },
            "metrics": {
                "response_time_minutes": response_time,
                "resolution_time_hours": resolution_time,
                "escalation_level": incident.escalation_level.value,
                "evidence_count": len(incident.evidence),
                "actions_count": len(incident.actions),
                "completed_actions": len([a for a in incident.actions if a.status == "completed"])
            },
            "impact_assessment": {
                "affected_systems": incident.affected_systems,
                "affected_data_types": incident.affected_data_types,
                "business_impact": incident.business_impact,
                "technical_impact": incident.technical_impact,
                "estimated_cost": incident.estimated_cost,
                "actual_cost": incident.actual_cost
            },
            "response_details": {
                "root_cause": incident.root_cause,
                "lessons_learned": incident.lessons_learned,
                "actions_taken": [
                    {
                        "description": action.description,
                        "status": action.status,
                        "completed_at": action.completed_at.isoformat() if action.completed_at else None,
                        "results": action.results
                    }
                    for action in incident.actions
                ],
                "evidence_collected": [
                    {
                        "type": evidence.evidence_type,
                        "description": evidence.description,
                        "collected_by": evidence.collected_by,
                        "collected_at": evidence.collected_at.isoformat()
                    }
                    for evidence in incident.evidence
                ]
            },
            "compliance_analysis": compliance_status,
            "timeline": incident.timeline,
            "generated_at": datetime.now().isoformat()
        }
    
    async def _analyze_compliance_requirements(self, incident: VendorIncident) -> Dict[str, Any]:
        """Analyze compliance requirements for incident"""
        compliance_status = {}
        
        # Check GDPR requirements
        if "personal_data" in incident.affected_data_types:
            gdpr_config = self.config["compliance_frameworks"]["gdpr"]
            notification_deadline = incident.created_at + timedelta(hours=gdpr_config["breach_notification_hours"])
            
            compliance_status["gdpr"] = {
                "applicable": True,
                "notification_deadline": notification_deadline.isoformat(),
                "deadline_met": datetime.now() <= notification_deadline,
                "authority_notification_required": gdpr_config["authority_notification_required"]
            }
        
        # Check SOX requirements
        if incident.estimated_cost and incident.estimated_cost > self.config["compliance_frameworks"]["sox"]["financial_impact_threshold"]:
            compliance_status["sox"] = {
                "applicable": True,
                "financial_threshold_exceeded": True,
                "audit_committee_notification": True
            }
        
        # Check HIPAA requirements
        if "phi" in incident.affected_data_types or "health_data" in incident.affected_data_types:
            hipaa_config = self.config["compliance_frameworks"]["hipaa"]
            notification_deadline = incident.created_at + timedelta(days=hipaa_config["phi_breach_notification_days"])
            
            compliance_status["hipaa"] = {
                "applicable": True,
                "notification_deadline": notification_deadline.isoformat(),
                "deadline_met": datetime.now() <= notification_deadline,
                "hhs_notification_required": hipaa_config["hhs_notification_required"]
            }
        
        return compliance_status
    
    async def generate_vendor_incident_summary(self, vendor_id: str, 
                                             period_days: int = 90) -> Dict[str, Any]:
        """Generate incident summary for vendor"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        vendor_incidents = [
            incident for incident in self.incidents.values()
            if incident.vendor_id == vendor_id and incident.created_at >= start_date
        ]
        
        if not vendor_incidents:
            return {
                "vendor_id": vendor_id,
                "period_days": period_days,
                "total_incidents": 0,
                "message": "No incidents recorded for this vendor in the specified period"
            }
        
        # Calculate statistics
        total_incidents = len(vendor_incidents)
        resolved_incidents = len([i for i in vendor_incidents if i.status == IncidentStatus.RESOLVED])
        open_incidents = len([i for i in vendor_incidents if i.status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]])
        
        # Severity distribution
        severity_distribution = {}
        for incident in vendor_incidents:
            severity = incident.severity.value
            severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
        
        # Category distribution
        category_distribution = {}
        for incident in vendor_incidents:
            category = incident.category.value
            category_distribution[category] = category_distribution.get(category, 0) + 1
        
        # Average resolution time
        resolved_with_time = [i for i in vendor_incidents if i.resolved_at]
        avg_resolution_hours = 0
        if resolved_with_time:
            total_resolution_time = sum(
                (i.resolved_at - i.created_at).total_seconds() / 3600
                for i in resolved_with_time
            )
            avg_resolution_hours = total_resolution_time / len(resolved_with_time)
        
        return {
            "vendor_id": vendor_id,
            "analysis_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": period_days
            },
            "incident_summary": {
                "total_incidents": total_incidents,
                "resolved_incidents": resolved_incidents,
                "open_incidents": open_incidents,
                "resolution_rate": (resolved_incidents / total_incidents * 100) if total_incidents > 0 else 0
            },
            "severity_distribution": severity_distribution,
            "category_distribution": category_distribution,
            "performance_metrics": {
                "average_resolution_hours": round(avg_resolution_hours, 2),
                "incidents_per_month": round(total_incidents / (period_days / 30), 2)
            },
            "risk_assessment": self._assess_vendor_risk(vendor_incidents),
            "recommendations": self._generate_vendor_recommendations(vendor_incidents),
            "generated_at": datetime.now().isoformat()
        }
    
    def _assess_vendor_risk(self, incidents: List[VendorIncident]) -> str:
        """Assess vendor risk based on incident history"""
        if not incidents:
            return "low"
        
        # Count critical and high severity incidents
        critical_count = len([i for i in incidents if i.severity == IncidentSeverity.CRITICAL])
        high_count = len([i for i in incidents if i.severity == IncidentSeverity.HIGH])
        
        # Check for data breaches or supply chain compromises
        serious_categories = [IncidentCategory.DATA_BREACH, IncidentCategory.SUPPLY_CHAIN_COMPROMISE]
        serious_incidents = len([i for i in incidents if i.category in serious_categories])
        
        if critical_count > 0 or serious_incidents > 0:
            return "high"
        elif high_count > 2 or len(incidents) > 10:
            return "medium"
        else:
            return "low"
    
    def _generate_vendor_recommendations(self, incidents: List[VendorIncident]) -> List[str]:
        """Generate recommendations based on incident patterns"""
        recommendations = []
        
        if not incidents:
            return ["Continue regular security monitoring"]
        
        # Check for recurring categories
        category_counts = {}
        for incident in incidents:
            category = incident.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        for category, count in category_counts.items():
            if count > 2:
                recommendations.append(f"Address recurring {category} incidents with vendor")
        
        # Check resolution times
        resolved_incidents = [i for i in incidents if i.resolved_at]
        if resolved_incidents:
            avg_resolution = sum(
                (i.resolved_at - i.created_at).total_seconds() / 3600
                for i in resolved_incidents
            ) / len(resolved_incidents)
            
            if avg_resolution > 48:  # More than 48 hours average
                recommendations.append("Work with vendor to improve incident response times")
        
        # Check for high-severity incidents
        critical_incidents = [i for i in incidents if i.severity == IncidentSeverity.CRITICAL]
        if critical_incidents:
            recommendations.append("Conduct security assessment due to critical incidents")
        
        if not recommendations:
            recommendations.append("Maintain current security monitoring and controls")
        
        return recommendations
    
    def _generate_incident_id(self, vendor_id: str, category: IncidentCategory) -> str:
        """Generate unique incident ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        category_code = category.value[:4].upper()
        vendor_code = vendor_id[:4].upper()
        return f"INC-{vendor_code}-{category_code}-{timestamp}"
    
    def _generate_action_id(self, incident_id: str, action_suffix: str) -> str:
        """Generate unique action ID"""
        return f"{incident_id}-ACT-{action_suffix}-{datetime.now().strftime('%H%M%S')}"
    
    def _generate_evidence_id(self, incident_id: str, evidence_type: str) -> str:
        """Generate unique evidence ID"""
        return f"{incident_id}-EVD-{evidence_type[:4].upper()}-{datetime.now().strftime('%H%M%S')}"