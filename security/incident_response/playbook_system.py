"""
Incident Response Playbook System
Provides automated incident response playbooks with workflow integration
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class IncidentSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class IncidentStatus(Enum):
    NEW = "new"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"

class PlaybookStatus(Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"

class ActionType(Enum):
    MANUAL = "manual"
    AUTOMATED = "automated"
    NOTIFICATION = "notification"
    ESCALATION = "escalation"

@dataclass
class PlaybookAction:
    """Individual action within a playbook"""
    id: str
    name: str
    description: str
    action_type: ActionType
    required: bool
    timeout_minutes: int
    dependencies: List[str]
    automation_script: Optional[str]
    notification_targets: List[str]
    escalation_conditions: Dict[str, Any]
    validation_criteria: Dict[str, Any]
    order: int

@dataclass
class IncidentPlaybook:
    """Incident response playbook definition"""
    id: str
    name: str
    description: str
    incident_types: List[str]
    severity_levels: List[IncidentSeverity]
    trigger_conditions: Dict[str, Any]
    actions: List[PlaybookAction]
    estimated_duration_minutes: int
    required_roles: List[str]
    escalation_path: List[str]
    success_criteria: List[str]
    status: PlaybookStatus
    version: str
    created_date: datetime
    last_updated: datetime
    created_by: str

@dataclass
class IncidentRecord:
    """Security incident record"""
    id: str
    title: str
    description: str
    incident_type: str
    severity: IncidentSeverity
    status: IncidentStatus
    assigned_to: Optional[str]
    reporter: str
    created_date: datetime
    updated_date: datetime
    resolved_date: Optional[datetime]
    playbook_id: Optional[str]
    affected_systems: List[str]
    indicators: List[str]
    timeline: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    lessons_learned: Optional[str]

@dataclass
class PlaybookExecution:
    """Playbook execution tracking"""
    id: str
    incident_id: str
    playbook_id: str
    started_by: str
    start_date: datetime
    end_date: Optional[datetime]
    status: str  # running, completed, failed, cancelled
    current_action: Optional[str]
    completed_actions: List[str]
    failed_actions: List[str]
    execution_log: List[Dict[str, Any]]
    metrics: Dict[str, Any]

class IncidentResponsePlaybookSystem:
    """Comprehensive incident response playbook management system"""
    
    def __init__(self, playbook_path: str = "security/incident_response"):
        self.playbook_path = Path(playbook_path)
        self.playbook_path.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.playbooks: Dict[str, IncidentPlaybook] = {}
        self.incidents: Dict[str, IncidentRecord] = {}
        self.executions: Dict[str, PlaybookExecution] = {}
        
        # Automation handlers
        self.automation_handlers: Dict[str, Callable] = {}
        self.notification_handlers: Dict[str, Callable] = {}
        
        self._load_playbook_data()
        self._initialize_default_playbooks()
        self._register_default_handlers()
    
    def _load_playbook_data(self):
        """Load playbook data from storage"""
        # Load playbooks
        playbooks_file = self.playbook_path / "playbooks.json"
        if playbooks_file.exists():
            with open(playbooks_file, 'r') as f:
                data = json.load(f)
                for playbook_id, playbook_data in data.items():
                    # Convert datetime strings
                    for date_field in ['created_date', 'last_updated']:
                        if playbook_data.get(date_field):
                            playbook_data[date_field] = datetime.fromisoformat(playbook_data[date_field])
                    
                    # Convert enums
                    playbook_data['status'] = PlaybookStatus(playbook_data['status'])
                    
                    # Convert actions
                    actions = []
                    for action_data in playbook_data.get('actions', []):
                        action_data['action_type'] = ActionType(action_data['action_type'])
                        actions.append(PlaybookAction(**action_data))
                    playbook_data['actions'] = actions
                    
                    self.playbooks[playbook_id] = IncidentPlaybook(**playbook_data)
        
        # Load incidents
        incidents_file = self.playbook_path / "incidents.json"
        if incidents_file.exists():
            with open(incidents_file, 'r') as f:
                data = json.load(f)
                for incident_id, incident_data in data.items():
                    # Convert datetime strings
                    for date_field in ['created_date', 'updated_date', 'resolved_date']:
                        if incident_data.get(date_field):
                            incident_data[date_field] = datetime.fromisoformat(incident_data[date_field])
                    
                    # Convert enums
                    incident_data['severity'] = IncidentSeverity(incident_data['severity'])
                    incident_data['status'] = IncidentStatus(incident_data['status'])
                    
                    self.incidents[incident_id] = IncidentRecord(**incident_data)
        
        # Load executions
        executions_file = self.playbook_path / "executions.json"
        if executions_file.exists():
            with open(executions_file, 'r') as f:
                data = json.load(f)
                for execution_id, execution_data in data.items():
                    # Convert datetime strings
                    for date_field in ['start_date', 'end_date']:
                        if execution_data.get(date_field):
                            execution_data[date_field] = datetime.fromisoformat(execution_data[date_field])
                    
                    self.executions[execution_id] = PlaybookExecution(**execution_data)
    
    def _save_playbook_data(self):
        """Save playbook data to storage"""
        # Save playbooks
        playbooks_data = {}
        for playbook_id, playbook in self.playbooks.items():
            playbook_data = asdict(playbook)
            
            # Convert datetime objects to strings
            for date_field in ['created_date', 'last_updated']:
                if playbook_data.get(date_field):
                    playbook_data[date_field] = playbook_data[date_field].isoformat()
            
            # Convert enums
            playbook_data['status'] = playbook_data['status'].value
            
            # Convert actions
            actions_data = []
            for action in playbook_data['actions']:
                action['action_type'] = action['action_type'].value
                actions_data.append(action)
            playbook_data['actions'] = actions_data
            
            playbooks_data[playbook_id] = playbook_data
        
        with open(self.playbook_path / "playbooks.json", 'w') as f:
            json.dump(playbooks_data, f, indent=2)
        
        # Save incidents
        incidents_data = {}
        for incident_id, incident in self.incidents.items():
            incident_data = asdict(incident)
            
            # Convert datetime objects to strings
            for date_field in ['created_date', 'updated_date', 'resolved_date']:
                if incident_data.get(date_field) and incident_data[date_field] is not None:
                    incident_data[date_field] = incident_data[date_field].isoformat()
            
            # Convert enums
            incident_data['severity'] = incident_data['severity'].value
            incident_data['status'] = incident_data['status'].value
            
            incidents_data[incident_id] = incident_data
        
        with open(self.playbook_path / "incidents.json", 'w') as f:
            json.dump(incidents_data, f, indent=2)
        
        # Save executions
        executions_data = {}
        for execution_id, execution in self.executions.items():
            execution_data = asdict(execution)
            
            # Convert datetime objects to strings
            for date_field in ['start_date', 'end_date']:
                if execution_data.get(date_field) and execution_data[date_field] is not None:
                    execution_data[date_field] = execution_data[date_field].isoformat()
            
            executions_data[execution_id] = execution_data
        
        with open(self.playbook_path / "executions.json", 'w') as f:
            json.dump(executions_data, f, indent=2)
    
    def _initialize_default_playbooks(self):
        """Initialize default incident response playbooks"""
        if not self.playbooks:
            default_playbooks = [
                {
                    "id": "pb-malware-001",
                    "name": "Malware Incident Response",
                    "description": "Response procedures for malware infections",
                    "incident_types": ["malware", "virus", "ransomware"],
                    "severity_levels": [IncidentSeverity.MEDIUM, IncidentSeverity.HIGH, IncidentSeverity.CRITICAL],
                    "trigger_conditions": {
                        "indicators": ["malware_detected", "suspicious_process", "file_encryption"]
                    },
                    "actions": [
                        {
                            "id": "action-001",
                            "name": "Isolate Affected System",
                            "description": "Immediately isolate the infected system from the network",
                            "action_type": ActionType.AUTOMATED,
                            "required": True,
                            "timeout_minutes": 5,
                            "dependencies": [],
                            "automation_script": "isolate_system.py",
                            "notification_targets": ["security-team", "it-ops"],
                            "escalation_conditions": {"timeout": True},
                            "validation_criteria": {"network_isolated": True},
                            "order": 1
                        },
                        {
                            "id": "action-002",
                            "name": "Notify Security Team",
                            "description": "Alert security team of malware incident",
                            "action_type": ActionType.NOTIFICATION,
                            "required": True,
                            "timeout_minutes": 2,
                            "dependencies": [],
                            "automation_script": None,
                            "notification_targets": ["security-team", "incident-commander"],
                            "escalation_conditions": {},
                            "validation_criteria": {"notification_sent": True},
                            "order": 2
                        },
                        {
                            "id": "action-003",
                            "name": "Collect Forensic Evidence",
                            "description": "Capture memory dump and disk image for analysis",
                            "action_type": ActionType.MANUAL,
                            "required": True,
                            "timeout_minutes": 30,
                            "dependencies": ["action-001"],
                            "automation_script": None,
                            "notification_targets": [],
                            "escalation_conditions": {"timeout": True},
                            "validation_criteria": {"evidence_collected": True},
                            "order": 3
                        },
                        {
                            "id": "action-004",
                            "name": "Analyze Malware Sample",
                            "description": "Submit malware sample for analysis",
                            "action_type": ActionType.AUTOMATED,
                            "required": True,
                            "timeout_minutes": 60,
                            "dependencies": ["action-003"],
                            "automation_script": "analyze_malware.py",
                            "notification_targets": ["security-analyst"],
                            "escalation_conditions": {"timeout": True},
                            "validation_criteria": {"analysis_complete": True},
                            "order": 4
                        },
                        {
                            "id": "action-005",
                            "name": "Remediate and Restore",
                            "description": "Clean infected system and restore from backup",
                            "action_type": ActionType.MANUAL,
                            "required": True,
                            "timeout_minutes": 120,
                            "dependencies": ["action-004"],
                            "automation_script": None,
                            "notification_targets": ["it-ops"],
                            "escalation_conditions": {"timeout": True},
                            "validation_criteria": {"system_clean": True, "services_restored": True},
                            "order": 5
                        }
                    ],
                    "estimated_duration_minutes": 240,
                    "required_roles": ["security-analyst", "incident-commander", "it-ops"],
                    "escalation_path": ["security-manager", "ciso", "cto"],
                    "success_criteria": [
                        "Malware contained and removed",
                        "System restored to normal operation",
                        "No lateral movement detected",
                        "Lessons learned documented"
                    ],
                    "status": PlaybookStatus.ACTIVE,
                    "version": "1.0",
                    "created_by": "security-team"
                },
                {
                    "id": "pb-data-breach-001",
                    "name": "Data Breach Response",
                    "description": "Response procedures for data breach incidents",
                    "incident_types": ["data_breach", "data_exfiltration", "unauthorized_access"],
                    "severity_levels": [IncidentSeverity.HIGH, IncidentSeverity.CRITICAL],
                    "trigger_conditions": {
                        "indicators": ["data_exfiltration", "unauthorized_database_access", "credential_compromise"]
                    },
                    "actions": [
                        {
                            "id": "action-101",
                            "name": "Assess Breach Scope",
                            "description": "Determine what data was accessed or stolen",
                            "action_type": ActionType.MANUAL,
                            "required": True,
                            "timeout_minutes": 30,
                            "dependencies": [],
                            "automation_script": None,
                            "notification_targets": ["security-team", "legal-team"],
                            "escalation_conditions": {"timeout": True},
                            "validation_criteria": {"scope_assessed": True},
                            "order": 1
                        },
                        {
                            "id": "action-102",
                            "name": "Contain the Breach",
                            "description": "Stop ongoing data exfiltration",
                            "action_type": ActionType.AUTOMATED,
                            "required": True,
                            "timeout_minutes": 15,
                            "dependencies": [],
                            "automation_script": "contain_breach.py",
                            "notification_targets": ["security-team"],
                            "escalation_conditions": {"timeout": True},
                            "validation_criteria": {"breach_contained": True},
                            "order": 2
                        },
                        {
                            "id": "action-103",
                            "name": "Notify Legal and Compliance",
                            "description": "Alert legal team for regulatory notification requirements",
                            "action_type": ActionType.NOTIFICATION,
                            "required": True,
                            "timeout_minutes": 10,
                            "dependencies": ["action-101"],
                            "automation_script": None,
                            "notification_targets": ["legal-team", "compliance-team", "privacy-officer"],
                            "escalation_conditions": {},
                            "validation_criteria": {"legal_notified": True},
                            "order": 3
                        },
                        {
                            "id": "action-104",
                            "name": "Preserve Evidence",
                            "description": "Secure forensic evidence for investigation",
                            "action_type": ActionType.MANUAL,
                            "required": True,
                            "timeout_minutes": 60,
                            "dependencies": ["action-102"],
                            "automation_script": None,
                            "notification_targets": ["forensics-team"],
                            "escalation_conditions": {"timeout": True},
                            "validation_criteria": {"evidence_preserved": True},
                            "order": 4
                        },
                        {
                            "id": "action-105",
                            "name": "Notify Affected Parties",
                            "description": "Notify customers and regulatory bodies as required",
                            "action_type": ActionType.MANUAL,
                            "required": True,
                            "timeout_minutes": 1440,  # 24 hours
                            "dependencies": ["action-103"],
                            "automation_script": None,
                            "notification_targets": ["communications-team"],
                            "escalation_conditions": {"timeout": True},
                            "validation_criteria": {"notifications_sent": True},
                            "order": 5
                        }
                    ],
                    "estimated_duration_minutes": 2880,  # 48 hours
                    "required_roles": ["security-analyst", "incident-commander", "legal-counsel", "privacy-officer"],
                    "escalation_path": ["security-manager", "ciso", "ceo", "board"],
                    "success_criteria": [
                        "Breach contained and stopped",
                        "Full scope of breach determined",
                        "All required notifications sent",
                        "Remediation plan implemented",
                        "Regulatory compliance maintained"
                    ],
                    "status": PlaybookStatus.ACTIVE,
                    "version": "1.0",
                    "created_by": "security-team"
                }
            ]
            
            for playbook_data in default_playbooks:
                playbook_data['created_date'] = datetime.now()
                playbook_data['last_updated'] = datetime.now()
                
                # Convert action data to PlaybookAction objects
                actions = []
                for action_data in playbook_data['actions']:
                    actions.append(PlaybookAction(**action_data))
                playbook_data['actions'] = actions
                
                playbook = IncidentPlaybook(**playbook_data)
                self.playbooks[playbook.id] = playbook
            
            self._save_playbook_data()
    
    def _register_default_handlers(self):
        """Register default automation and notification handlers"""
        # Automation handlers
        self.automation_handlers['isolate_system.py'] = self._isolate_system_handler
        self.automation_handlers['analyze_malware.py'] = self._analyze_malware_handler
        self.automation_handlers['contain_breach.py'] = self._contain_breach_handler
        
        # Notification handlers
        self.notification_handlers['email'] = self._email_notification_handler
        self.notification_handlers['slack'] = self._slack_notification_handler
        self.notification_handlers['sms'] = self._sms_notification_handler
    
    def create_incident(self, incident_data: Dict[str, Any]) -> str:
        """Create a new security incident"""
        try:
            incident_id = incident_data.get('id', f"INC-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8]}")
            incident_data['id'] = incident_id
            incident_data['created_date'] = datetime.now()
            incident_data['updated_date'] = datetime.now()
            incident_data['resolved_date'] = None
            incident_data['timeline'] = []
            incident_data['evidence'] = []
            incident_data['lessons_learned'] = None
            
            # Convert string enums to enum objects
            incident_data['severity'] = IncidentSeverity(incident_data['severity'])
            incident_data['status'] = IncidentStatus(incident_data.get('status', 'new'))
            
            incident = IncidentRecord(**incident_data)
            self.incidents[incident_id] = incident
            
            # Add initial timeline entry
            self._add_timeline_entry(incident_id, "incident_created", f"Incident created by {incident.reporter}")
            
            # Auto-select playbook if possible
            playbook_id = self._select_playbook(incident)
            if playbook_id:
                incident.playbook_id = playbook_id
                self._add_timeline_entry(incident_id, "playbook_selected", f"Playbook {playbook_id} selected")
            
            self._save_playbook_data()
            logger.info(f"Created incident: {incident_id}")
            return incident_id
            
        except Exception as e:
            logger.error(f"Failed to create incident: {str(e)}")
            raise
    
    def _select_playbook(self, incident: IncidentRecord) -> Optional[str]:
        """Automatically select appropriate playbook for incident"""
        for playbook_id, playbook in self.playbooks.items():
            if playbook.status != PlaybookStatus.ACTIVE:
                continue
            
            # Check incident type match
            if incident.incident_type in playbook.incident_types:
                # Check severity match
                if incident.severity in playbook.severity_levels:
                    # Check trigger conditions
                    if self._check_trigger_conditions(incident, playbook.trigger_conditions):
                        return playbook_id
        
        return None
    
    def _check_trigger_conditions(self, incident: IncidentRecord, conditions: Dict[str, Any]) -> bool:
        """Check if incident matches playbook trigger conditions"""
        # Simple indicator matching for now
        if 'indicators' in conditions:
            required_indicators = conditions['indicators']
            incident_indicators = incident.indicators
            
            # Check if any required indicator is present
            return any(indicator in incident_indicators for indicator in required_indicators)
        
        return True  # Default to match if no specific conditions
    
    def execute_playbook(self, incident_id: str, executed_by: str) -> str:
        """Execute incident response playbook for an incident"""
        try:
            if incident_id not in self.incidents:
                raise ValueError(f"Incident {incident_id} not found")
            
            incident = self.incidents[incident_id]
            if not incident.playbook_id:
                raise ValueError(f"No playbook assigned to incident {incident_id}")
            
            if incident.playbook_id not in self.playbooks:
                raise ValueError(f"Playbook {incident.playbook_id} not found")
            
            playbook = self.playbooks[incident.playbook_id]
            
            # Create execution record
            execution_id = str(uuid.uuid4())
            execution = PlaybookExecution(
                id=execution_id,
                incident_id=incident_id,
                playbook_id=incident.playbook_id,
                started_by=executed_by,
                start_date=datetime.now(),
                end_date=None,
                status="running",
                current_action=None,
                completed_actions=[],
                failed_actions=[],
                execution_log=[],
                metrics={}
            )
            
            self.executions[execution_id] = execution
            
            # Update incident status
            incident.status = IncidentStatus.IN_PROGRESS
            incident.updated_date = datetime.now()
            self._add_timeline_entry(incident_id, "playbook_execution_started", 
                                   f"Playbook execution started by {executed_by}")
            
            # Execute actions in order
            self._execute_playbook_actions(execution_id)
            
            self._save_playbook_data()
            logger.info(f"Started playbook execution: {execution_id}")
            return execution_id
            
        except Exception as e:
            logger.error(f"Failed to execute playbook: {str(e)}")
            raise
    
    def _execute_playbook_actions(self, execution_id: str):
        """Execute playbook actions in sequence"""
        execution = self.executions[execution_id]
        playbook = self.playbooks[execution.playbook_id]
        
        # Sort actions by order
        sorted_actions = sorted(playbook.actions, key=lambda x: x.order)
        
        for action in sorted_actions:
            # Check dependencies
            if not self._check_action_dependencies(action, execution.completed_actions):
                self._log_execution(execution_id, "dependency_check_failed", 
                                  f"Action {action.id} dependencies not met")
                continue
            
            execution.current_action = action.id
            self._log_execution(execution_id, "action_started", f"Starting action: {action.name}")
            
            # Execute action based on type
            success = False
            if action.action_type == ActionType.AUTOMATED:
                success = self._execute_automated_action(execution_id, action)
            elif action.action_type == ActionType.NOTIFICATION:
                success = self._execute_notification_action(execution_id, action)
            elif action.action_type == ActionType.MANUAL:
                success = self._execute_manual_action(execution_id, action)
            elif action.action_type == ActionType.ESCALATION:
                success = self._execute_escalation_action(execution_id, action)
            
            if success:
                execution.completed_actions.append(action.id)
                self._log_execution(execution_id, "action_completed", f"Completed action: {action.name}")
            else:
                execution.failed_actions.append(action.id)
                self._log_execution(execution_id, "action_failed", f"Failed action: {action.name}")
                
                if action.required:
                    execution.status = "failed"
                    self._log_execution(execution_id, "execution_failed", 
                                      f"Execution failed due to required action failure: {action.name}")
                    break
        
        # Check if execution completed successfully
        if execution.status == "running":
            execution.status = "completed"
            execution.end_date = datetime.now()
            self._log_execution(execution_id, "execution_completed", "Playbook execution completed successfully")
            
            # Update incident status
            incident = self.incidents[execution.incident_id]
            incident.status = IncidentStatus.RESOLVED
            incident.resolved_date = datetime.now()
            incident.updated_date = datetime.now()
            self._add_timeline_entry(execution.incident_id, "incident_resolved", 
                                   "Incident resolved through playbook execution")
    
    def _check_action_dependencies(self, action: PlaybookAction, completed_actions: List[str]) -> bool:
        """Check if action dependencies are satisfied"""
        return all(dep in completed_actions for dep in action.dependencies)
    
    def _execute_automated_action(self, execution_id: str, action: PlaybookAction) -> bool:
        """Execute automated action"""
        try:
            if action.automation_script and action.automation_script in self.automation_handlers:
                handler = self.automation_handlers[action.automation_script]
                result = handler(execution_id, action)
                return result.get('success', False)
            else:
                self._log_execution(execution_id, "automation_handler_missing", 
                                  f"No handler found for script: {action.automation_script}")
                return False
        except Exception as e:
            self._log_execution(execution_id, "automation_error", f"Automation error: {str(e)}")
            return False
    
    def _execute_notification_action(self, execution_id: str, action: PlaybookAction) -> bool:
        """Execute notification action"""
        try:
            success_count = 0
            for target in action.notification_targets:
                # Use email as default notification method
                if 'email' in self.notification_handlers:
                    handler = self.notification_handlers['email']
                    result = handler(execution_id, action, target)
                    if result.get('success', False):
                        success_count += 1
            
            return success_count > 0
        except Exception as e:
            self._log_execution(execution_id, "notification_error", f"Notification error: {str(e)}")
            return False
    
    def _execute_manual_action(self, execution_id: str, action: PlaybookAction) -> bool:
        """Execute manual action (requires human intervention)"""
        # For manual actions, we mark them as pending and return True
        # In a real system, this would create a task for a human operator
        self._log_execution(execution_id, "manual_action_pending", 
                          f"Manual action pending: {action.name}")
        return True
    
    def _execute_escalation_action(self, execution_id: str, action: PlaybookAction) -> bool:
        """Execute escalation action"""
        try:
            execution = self.executions[execution_id]
            incident = self.incidents[execution.incident_id]
            
            # Escalate to next level in escalation path
            playbook = self.playbooks[execution.playbook_id]
            if playbook.escalation_path:
                next_escalation = playbook.escalation_path[0]  # Simplified escalation
                
                # Send escalation notification
                if 'email' in self.notification_handlers:
                    handler = self.notification_handlers['email']
                    result = handler(execution_id, action, next_escalation)
                    return result.get('success', False)
            
            return True
        except Exception as e:
            self._log_execution(execution_id, "escalation_error", f"Escalation error: {str(e)}")
            return False
    
    def _log_execution(self, execution_id: str, event_type: str, message: str):
        """Log execution event"""
        execution = self.executions[execution_id]
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "message": message
        }
        execution.execution_log.append(log_entry)
    
    def _add_timeline_entry(self, incident_id: str, event_type: str, description: str):
        """Add entry to incident timeline"""
        incident = self.incidents[incident_id]
        timeline_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "description": description
        }
        incident.timeline.append(timeline_entry)
    
    # Default automation handlers
    def _isolate_system_handler(self, execution_id: str, action: PlaybookAction) -> Dict[str, Any]:
        """Handler for system isolation automation"""
        # Simulate system isolation
        self._log_execution(execution_id, "system_isolation", "System isolated from network")
        return {"success": True, "message": "System successfully isolated"}
    
    def _analyze_malware_handler(self, execution_id: str, action: PlaybookAction) -> Dict[str, Any]:
        """Handler for malware analysis automation"""
        # Simulate malware analysis
        self._log_execution(execution_id, "malware_analysis", "Malware sample submitted for analysis")
        return {"success": True, "message": "Malware analysis initiated"}
    
    def _contain_breach_handler(self, execution_id: str, action: PlaybookAction) -> Dict[str, Any]:
        """Handler for breach containment automation"""
        # Simulate breach containment
        self._log_execution(execution_id, "breach_containment", "Data exfiltration stopped")
        return {"success": True, "message": "Breach successfully contained"}
    
    # Default notification handlers
    def _email_notification_handler(self, execution_id: str, action: PlaybookAction, target: str) -> Dict[str, Any]:
        """Handler for email notifications"""
        # Simulate email notification
        self._log_execution(execution_id, "email_sent", f"Email notification sent to {target}")
        return {"success": True, "message": f"Email sent to {target}"}
    
    def _slack_notification_handler(self, execution_id: str, action: PlaybookAction, target: str) -> Dict[str, Any]:
        """Handler for Slack notifications"""
        # Simulate Slack notification
        self._log_execution(execution_id, "slack_sent", f"Slack notification sent to {target}")
        return {"success": True, "message": f"Slack message sent to {target}"}
    
    def _sms_notification_handler(self, execution_id: str, action: PlaybookAction, target: str) -> Dict[str, Any]:
        """Handler for SMS notifications"""
        # Simulate SMS notification
        self._log_execution(execution_id, "sms_sent", f"SMS notification sent to {target}")
        return {"success": True, "message": f"SMS sent to {target}"}
    
    def get_incident_status(self, incident_id: str) -> Dict[str, Any]:
        """Get comprehensive incident status"""
        if incident_id not in self.incidents:
            raise ValueError(f"Incident {incident_id} not found")
        
        incident = self.incidents[incident_id]
        
        # Get execution status if playbook is running
        execution_status = None
        for execution in self.executions.values():
            if execution.incident_id == incident_id:
                execution_status = {
                    "execution_id": execution.id,
                    "status": execution.status,
                    "current_action": execution.current_action,
                    "completed_actions": len(execution.completed_actions),
                    "failed_actions": len(execution.failed_actions),
                    "start_date": execution.start_date.isoformat(),
                    "end_date": execution.end_date.isoformat() if execution.end_date else None
                }
                break
        
        return {
            "incident_id": incident_id,
            "title": incident.title,
            "severity": incident.severity.value,
            "status": incident.status.value,
            "assigned_to": incident.assigned_to,
            "created_date": incident.created_date.isoformat(),
            "updated_date": incident.updated_date.isoformat(),
            "resolved_date": incident.resolved_date.isoformat() if incident.resolved_date else None,
            "playbook_id": incident.playbook_id,
            "execution_status": execution_status,
            "timeline_entries": len(incident.timeline),
            "evidence_items": len(incident.evidence)
        }
    
    def generate_incident_report(self) -> Dict[str, Any]:
        """Generate comprehensive incident response report"""
        total_incidents = len(self.incidents)
        open_incidents = sum(1 for i in self.incidents.values() if i.status != IncidentStatus.CLOSED)
        resolved_incidents = sum(1 for i in self.incidents.values() if i.status == IncidentStatus.RESOLVED)
        
        # Severity breakdown
        severity_breakdown = {}
        for severity in IncidentSeverity:
            severity_breakdown[severity.value] = sum(1 for i in self.incidents.values() if i.severity == severity)
        
        # Playbook usage
        playbook_usage = {}
        for incident in self.incidents.values():
            if incident.playbook_id:
                playbook_name = self.playbooks[incident.playbook_id].name
                playbook_usage[playbook_name] = playbook_usage.get(playbook_name, 0) + 1
        
        # Average resolution time
        resolved_with_times = [i for i in self.incidents.values() 
                             if i.resolved_date and i.created_date]
        avg_resolution_hours = 0
        if resolved_with_times:
            total_hours = sum((i.resolved_date - i.created_date).total_seconds() / 3600 
                            for i in resolved_with_times)
            avg_resolution_hours = total_hours / len(resolved_with_times)
        
        return {
            "total_incidents": total_incidents,
            "open_incidents": open_incidents,
            "resolved_incidents": resolved_incidents,
            "severity_breakdown": severity_breakdown,
            "playbook_usage": playbook_usage,
            "average_resolution_hours": round(avg_resolution_hours, 2),
            "total_playbooks": len(self.playbooks),
            "active_playbooks": sum(1 for p in self.playbooks.values() if p.status == PlaybookStatus.ACTIVE),
            "report_generated": datetime.now().isoformat()
        }