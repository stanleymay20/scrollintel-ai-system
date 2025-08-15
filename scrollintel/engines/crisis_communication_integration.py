"""Crisis Communication Integration Engine

Enhanced integration with all ScrollIntel communication systems to provide
seamless crisis-aware communication across all channels and interactions.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import logging
import json

logger = logging.getLogger(__name__)

class CommunicationChannel(Enum):
    EMAIL = "email"
    SLACK = "slack"
    SMS = "sms"
    CHAT = "chat"

class CrisisLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class CrisisContext:
    crisis_id: str
    crisis_type: str
    severity_level: CrisisLevel
    start_time: datetime
    affected_systems: List[str] = field(default_factory=list)
    stakeholders: List[str] = field(default_factory=list)
    status: str = "active"

@dataclass
class CommunicationMessage:
    id: str
    channel: CommunicationChannel
    recipient: str
    content: str
    crisis_context: Optional[CrisisContext] = None
    priority: str = "normal"
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class CommunicationResponse:
    success: bool
    message_id: str
    channel: CommunicationChannel
    delivery_status: str
    timestamp: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None@dataclass

class CommunicationSystemIntegration:
    """Integration configuration for external communication systems"""
    system_name: str
    system_type: str  # "chat", "email", "notification", "collaboration"
    api_endpoint: Optional[str] = None
    webhook_url: Optional[str] = None
    authentication: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    is_active: bool = True
    crisis_context_support: bool = False

@dataclass
class CrisisAwareResponse:
    """Crisis-aware response with context"""
    original_query: str
    base_response: str
    crisis_enhanced_response: str
    crisis_context: Optional[CrisisContext] = None
    confidence_score: float = 0.0
    response_type: str = "standard"
    escalation_recommended: bool = False
    additional_actions: List[str] = field(default_factory=list)cl
ass CrisisCommunicationIntegration:
    def __init__(self):
        self.active_crises: Dict[str, CrisisContext] = {}
        self.communication_history: List[CommunicationMessage] = []
        self.integrated_systems: Dict[str, CommunicationSystemIntegration] = {}
        self.response_enhancers: Dict[str, Callable] = {}
        self.context_propagation_rules: Dict[str, Dict[str, Any]] = {}
        
        # Enhanced message templates with context awareness
        self.message_templates = {
            "crisis_alert": "🚨 CRISIS ALERT: {crisis_type} detected. Severity: {severity}. Impact: {impact_summary}",
            "status_update": "📊 Crisis Update: {crisis_type} - Status: {status}. Progress: {progress_summary}",
            "resolution": "✅ Crisis Resolved: {crisis_type} has been resolved. Duration: {duration}",
            "escalation": "⚠️ Crisis Escalation: {crisis_type} severity increased to {new_severity}",
            "stakeholder_update": "📢 Stakeholder Update: {crisis_type} - {stakeholder_specific_message}",
            "media_response": "📰 Media Statement: {official_statement} - Contact: {media_contact}",
            "customer_notification": "Dear Customer, we are experiencing {customer_friendly_description}. ETA: {eta}",
            "employee_directive": "Team Alert: {crisis_type} - Action Required: {action_required}",
            "regulatory_report": "Regulatory Notification: {formal_description} - Compliance Status: {compliance_status}"
        }
        
        self.escalation_rules = {
            "low": ["team_lead"],
            "medium": ["team_lead", "manager"],
            "high": ["team_lead", "manager", "director"],
            "critical": ["team_lead", "manager", "director", "ceo", "board_chair"]
        }
        
        # Initialize integrated systems
        self._initialize_system_integrations()
        self._setup_response_enhancers()
        self._configure_context_propagation()    def _i
nitialize_system_integrations(self):
        """Initialize integrations with all ScrollIntel communication systems"""
        # Core ScrollIntel systems
        self.integrated_systems.update({
            "message_coordination": CommunicationSystemIntegration(
                system_name="Message Coordination Engine",
                system_type="coordination",
                capabilities=["message_approval", "version_control", "multi_channel"],
                crisis_context_support=True
            ),
            "stakeholder_notification": CommunicationSystemIntegration(
                system_name="Stakeholder Notification Engine",
                system_type="notification",
                capabilities=["stakeholder_prioritization", "template_customization", "delivery_tracking"],
                crisis_context_support=True
            ),
            "media_management": CommunicationSystemIntegration(
                system_name="Media Management Engine",
                system_type="media",
                capabilities=["media_relations", "press_releases", "sentiment_monitoring"],
                crisis_context_support=True
            ),
            "executive_communication": CommunicationSystemIntegration(
                system_name="Executive Communication Engine",
                system_type="executive",
                capabilities=["board_reporting", "investor_relations", "strategic_messaging"],
                crisis_context_support=True
            ),
            "chat_interface": CommunicationSystemIntegration(
                system_name="Chat Interface",
                system_type="chat",
                capabilities=["real_time_chat", "context_awareness", "escalation"],
                crisis_context_support=True
            ),
            "email_system": CommunicationSystemIntegration(
                system_name="Email System",
                system_type="email",
                capabilities=["mass_email", "personalization", "tracking"],
                crisis_context_support=True
            ),
            "collaboration_tools": CommunicationSystemIntegration(
                system_name="Collaboration Tools",
                system_type="collaboration",
                capabilities=["team_coordination", "document_sharing", "real_time_updates"],
                crisis_context_support=True
            )
        })
    
    def _setup_response_enhancers(self):
        """Setup response enhancement functions for different communication types"""
        self.response_enhancers = {
            "chat": self._enhance_chat_response,
            "email": self._enhance_email_response,
            "notification": self._enhance_notification_response,
            "media": self._enhance_media_response,
            "executive": self._enhance_executive_response,
            "customer": self._enhance_customer_response,
            "employee": self._enhance_employee_response,
            "regulatory": self._enhance_regulatory_response
        }
    
    def _configure_context_propagation(self):
        """Configure how crisis context propagates across different systems"""
        self.context_propagation_rules = {
            "all_systems": {
                "include_crisis_id": True,
                "include_severity": True,
                "include_status": True,
                "include_timestamp": True
            },
            "customer_facing": {
                "sanitize_technical_details": True,
                "use_customer_friendly_language": True,
                "include_eta_if_available": True,
                "include_support_contact": True
            },
            "internal_systems": {
                "include_full_context": True,
                "include_technical_details": True,
                "include_action_items": True,
                "include_escalation_path": True
            },
            "media_systems": {
                "use_official_language": True,
                "include_media_contact": True,
                "sanitize_internal_details": True,
                "include_company_statement": True
            },
            "regulatory_systems": {
                "use_formal_language": True,
                "include_compliance_status": True,
                "include_corrective_actions": True,
                "include_timeline": True
            }
        }    a
sync def register_crisis(self, crisis_context: CrisisContext) -> bool:
        """Register crisis and propagate context to all integrated systems"""
        try:
            self.active_crises[crisis_context.crisis_id] = crisis_context
            
            # Propagate crisis context to all integrated systems
            await self._propagate_crisis_context_to_systems(crisis_context)
            
            # Trigger initial communications
            await self._trigger_initial_communications(crisis_context)
            
            # Enable crisis-aware responses across all systems
            await self._enable_crisis_aware_responses(crisis_context)
            
            logger.info(f"Crisis {crisis_context.crisis_id} registered and propagated to all systems")
            return True
        except Exception as e:
            logger.error(f"Failed to register crisis: {str(e)}")
            return False

    async def _propagate_crisis_context_to_systems(self, crisis_context: CrisisContext):
        """Propagate crisis context to all integrated communication systems"""
        propagation_tasks = []
        
        for system_name, system_config in self.integrated_systems.items():
            if system_config.is_active and system_config.crisis_context_support:
                task = asyncio.create_task(
                    self._notify_system_of_crisis(system_name, system_config, crisis_context)
                )
                propagation_tasks.append(task)
        
        # Wait for all systems to be notified
        await asyncio.gather(*propagation_tasks, return_exceptions=True)
    
    async def _notify_system_of_crisis(
        self, 
        system_name: str, 
        system_config: CommunicationSystemIntegration, 
        crisis_context: CrisisContext
    ):
        """Notify individual system of crisis context"""
        try:
            # Prepare context based on system type
            context_data = self._prepare_context_for_system(system_config.system_type, crisis_context)
            
            # In a real implementation, this would make API calls or use message queues
            # For now, we'll simulate the integration
            logger.info(f"Notified {system_name} of crisis {crisis_context.crisis_id}")
            
            # Store the context for the system
            if not hasattr(self, 'system_contexts'):
                self.system_contexts = {}
            self.system_contexts[system_name] = context_data
            
        except Exception as e:
            logger.error(f"Failed to notify {system_name} of crisis: {str(e)}")
    
    def _prepare_context_for_system(self, system_type: str, crisis_context: CrisisContext) -> Dict[str, Any]:
        """Prepare crisis context data for specific system type"""
        base_context = {
            "crisis_id": crisis_context.crisis_id,
            "crisis_type": crisis_context.crisis_type,
            "severity_level": crisis_context.severity_level.value,
            "start_time": crisis_context.start_time.isoformat(),
            "status": crisis_context.status,
            "affected_systems": crisis_context.affected_systems,
            "stakeholders": crisis_context.stakeholders
        }
        
        # Apply system-specific context rules
        if system_type in ["chat", "collaboration"]:
            # Internal systems get full context
            rules = self.context_propagation_rules.get("internal_systems", {})
            if rules.get("include_full_context"):
                base_context.update({
                    "technical_details": self._get_technical_details(crisis_context),
                    "action_items": self._get_action_items(crisis_context),
                    "escalation_path": self._get_escalation_path(crisis_context)
                })
        
        elif system_type in ["notification", "email"]:
            # Customer-facing systems get sanitized context
            rules = self.context_propagation_rules.get("customer_facing", {})
            if rules.get("sanitize_technical_details"):
                base_context["customer_friendly_description"] = self._get_customer_friendly_description(crisis_context)
                base_context["eta"] = self._get_estimated_resolution_time(crisis_context)
                base_context["support_contact"] = "support@company.com"
        
        elif system_type == "media":
            # Media systems get official context
            rules = self.context_propagation_rules.get("media_systems", {})
            base_context.update({
                "official_statement": self._get_official_statement(crisis_context),
                "media_contact": "press@company.com",
                "company_position": self._get_company_position(crisis_context)
            })
        
        return base_context    de
f generate_crisis_aware_response(
        self,
        query: str,
        response_type: str = "chat",
        user_context: Optional[Dict[str, Any]] = None,
        channel: CommunicationChannel = CommunicationChannel.CHAT
    ) -> CrisisAwareResponse:
        """Generate crisis-aware response with full context and enhancement"""
        try:
            # Get active crisis context
            active_crisis = self._get_most_relevant_crisis(query, user_context)
            
            # Generate base response
            base_response = self._generate_base_response(query)
            
            # Enhance response based on crisis context and type
            if active_crisis and response_type in self.response_enhancers:
                enhanced_response = self.response_enhancers[response_type](
                    query, base_response, active_crisis, user_context
                )
            else:
                enhanced_response = base_response
            
            # Determine if escalation is needed
            escalation_needed = self._should_escalate_query(query, active_crisis, user_context)
            
            # Generate additional actions
            additional_actions = self._get_additional_actions(query, active_crisis, user_context)
            
            return CrisisAwareResponse(
                original_query=query,
                base_response=base_response,
                crisis_enhanced_response=enhanced_response,
                crisis_context=active_crisis,
                confidence_score=self._calculate_response_confidence(query, active_crisis),
                response_type=response_type,
                escalation_recommended=escalation_needed,
                additional_actions=additional_actions
            )
            
        except Exception as e:
            logger.error(f"Failed to generate crisis-aware response: {str(e)}")
            return CrisisAwareResponse(
                original_query=query,
                base_response="I'm currently experiencing technical difficulties. Please contact support.",
                crisis_enhanced_response="I'm currently experiencing technical difficulties. Please contact support.",
                confidence_score=0.0
            )
    
    def _get_most_relevant_crisis(
        self, 
        query: str, 
        user_context: Optional[Dict[str, Any]] = None
    ) -> Optional[CrisisContext]:
        """Get the most relevant active crisis for the query"""
        if not self.active_crises:
            return None
        
        # For now, return the highest severity crisis
        # In a real implementation, this would use NLP to match query to crisis
        highest_severity_crisis = None
        highest_severity_level = 0
        
        severity_levels = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        
        for crisis in self.active_crises.values():
            severity_level = severity_levels.get(crisis.severity_level.value, 0)
            if severity_level > highest_severity_level:
                highest_severity_level = severity_level
                highest_severity_crisis = crisis
        
        return highest_severity_crisis
    
    def _generate_base_response(self, query: str) -> str:
        query_lower = query.lower()
        if "status" in query_lower:
            return "Current system status is being monitored."
        elif "help" in query_lower:
            return "I'm here to help. What information do you need?"
        else:
            return "Thank you for your message. How can I assist you?"    def
 _enhance_chat_response(
        self, 
        query: str, 
        base_response: str, 
        crisis_context: CrisisContext, 
        user_context: Optional[Dict[str, Any]]
    ) -> str:
        """Enhance chat response with crisis context"""
        crisis_info = f"\n\n🚨 **Crisis Alert**: We are currently managing a {crisis_context.crisis_type.replace('_', ' ')} "
        crisis_info += f"(Severity: {crisis_context.severity_level.value.upper()})\n"
        crisis_info += f"Status: {crisis_context.status.replace('_', ' ').title()}\n"
        
        if "status" in query.lower() or "issue" in query.lower():
            crisis_info += f"Started: {crisis_context.start_time.strftime('%H:%M UTC')}\n"
            crisis_info += f"Affected: {', '.join(crisis_context.affected_systems[:3])}\n"
        
        crisis_info += "\nOur team is actively working on resolution. I'll prioritize your request."
        
        return base_response + crisis_info
    
    def _enhance_email_response(
        self, 
        query: str, 
        base_response: str, 
        crisis_context: CrisisContext, 
        user_context: Optional[Dict[str, Any]]
    ) -> str:
        """Enhance email response with crisis context"""
        crisis_header = f"[CRISIS-{crisis_context.severity_level.value.upper()}] "
        
        crisis_footer = f"\n\n---\nCRISIS UPDATE:\n"
        crisis_footer += f"We are currently addressing a {crisis_context.crisis_type.replace('_', ' ')} situation. "
        crisis_footer += f"Our team is working to resolve this as quickly as possible.\n"
        crisis_footer += f"For urgent matters, please contact our crisis hotline.\n"
        crisis_footer += f"Crisis ID: {crisis_context.crisis_id}"
        
        return crisis_header + base_response + crisis_footer
    
    def _enhance_notification_response(
        self, 
        query: str, 
        base_response: str, 
        crisis_context: CrisisContext, 
        user_context: Optional[Dict[str, Any]]
    ) -> str:
        """Enhance notification with crisis context"""
        return f"🚨 {base_response} [Crisis: {crisis_context.crisis_type.replace('_', ' ').title()}]"
    
    def _enhance_media_response(
        self, 
        query: str, 
        base_response: str, 
        crisis_context: CrisisContext, 
        user_context: Optional[Dict[str, Any]]
    ) -> str:
        """Enhance media response with official crisis context"""
        official_statement = f"We are currently addressing a {crisis_context.crisis_type.replace('_', ' ')} situation. "
        official_statement += f"Our team is working diligently to resolve this matter. "
        official_statement += f"We will provide updates as more information becomes available. "
        official_statement += f"For media inquiries, please contact our press relations team."
        
        return official_statement
    
    def _enhance_executive_response(
        self, 
        query: str, 
        base_response: str, 
        crisis_context: CrisisContext, 
        user_context: Optional[Dict[str, Any]]
    ) -> str:
        """Enhance executive response with strategic crisis context"""
        executive_context = f"\n\nEXECUTIVE BRIEFING:\n"
        executive_context += f"Crisis: {crisis_context.crisis_type.replace('_', ' ').title()}\n"
        executive_context += f"Severity: {crisis_context.severity_level.value.upper()}\n"
        executive_context += f"Duration: {self._get_crisis_duration(crisis_context)}\n"
        executive_context += f"Business Impact: {self._get_business_impact(crisis_context)}\n"
        executive_context += f"Next Actions: {self._get_executive_actions(crisis_context)}"
        
        return base_response + executive_context
    
    def _enhance_customer_response(
        self, 
        query: str, 
        base_response: str, 
        crisis_context: CrisisContext, 
        user_context: Optional[Dict[str, Any]]
    ) -> str:
        """Enhance customer response with customer-friendly crisis context"""
        customer_message = f"\n\nWe want to keep you informed: "
        customer_message += f"We're currently experiencing {self._get_customer_friendly_description(crisis_context)}. "
        customer_message += f"Our team is working to resolve this quickly. "
        
        eta = self._get_estimated_resolution_time(crisis_context)
        if eta:
            customer_message += f"Expected resolution: {eta}. "
        
        customer_message += f"Thank you for your patience."
        
        return base_response + customer_message
    
    def _enhance_employee_response(
        self, 
        query: str, 
        base_response: str, 
        crisis_context: CrisisContext, 
        user_context: Optional[Dict[str, Any]]
    ) -> str:
        """Enhance employee response with internal crisis context"""
        employee_context = f"\n\nTEAM UPDATE:\n"
        employee_context += f"Active Crisis: {crisis_context.crisis_type.replace('_', ' ').title()}\n"
        employee_context += f"Your Role: {self._get_employee_role_in_crisis(user_context, crisis_context)}\n"
        employee_context += f"Action Required: {self._get_employee_actions(user_context, crisis_context)}\n"
        employee_context += f"Escalation: Contact crisis team lead if urgent"
        
        return base_response + employee_context
    
    def _enhance_regulatory_response(
        self, 
        query: str, 
        base_response: str, 
        crisis_context: CrisisContext, 
        user_context: Optional[Dict[str, Any]]
    ) -> str:
        """Enhance regulatory response with compliance-focused crisis context"""
        regulatory_context = f"\n\nREGULATORY COMPLIANCE UPDATE:\n"
        regulatory_context += f"Incident Type: {crisis_context.crisis_type.replace('_', ' ').title()}\n"
        regulatory_context += f"Compliance Status: {self._get_compliance_status(crisis_context)}\n"
        regulatory_context += f"Corrective Actions: {self._get_corrective_actions(crisis_context)}\n"
        regulatory_context += f"Reporting Timeline: {self._get_reporting_timeline(crisis_context)}"
        
        return base_response + regulatory_context    
def _should_escalate_query(
        self, 
        query: str, 
        crisis_context: Optional[CrisisContext], 
        user_context: Optional[Dict[str, Any]]
    ) -> bool:
        """Determine if query should be escalated during crisis"""
        if not crisis_context:
            return False
        
        escalation_keywords = [
            "urgent", "emergency", "critical", "immediate", "help", 
            "broken", "down", "failed", "error", "issue", "problem"
        ]
        
        query_lower = query.lower()
        has_escalation_keywords = any(keyword in query_lower for keyword in escalation_keywords)
        
        # Escalate if high/critical severity and user mentions urgent issues
        if crisis_context.severity_level.value in ["high", "critical"] and has_escalation_keywords:
            return True
        
        # Escalate if user is a VIP stakeholder
        if user_context and user_context.get("stakeholder_type") in ["executive", "board_member"]:
            return True
        
        return False
    
    def _get_additional_actions(
        self, 
        query: str, 
        crisis_context: Optional[CrisisContext], 
        user_context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Get additional actions to recommend based on crisis context"""
        actions = []
        
        if not crisis_context:
            return actions
        
        # Standard crisis actions
        actions.append("Monitor crisis status updates")
        
        if "status" in query.lower():
            actions.append("Subscribe to crisis notifications")
            actions.append("Check crisis dashboard for real-time updates")
        
        if crisis_context.severity_level.value in ["high", "critical"]:
            actions.append("Contact crisis hotline for urgent matters")
            actions.append("Review crisis response procedures")
        
        # User-specific actions
        if user_context:
            user_type = user_context.get("stakeholder_type", "employee")
            if user_type == "customer":
                actions.append("Check service status page")
                actions.append("Contact customer support if needed")
            elif user_type == "employee":
                actions.append("Follow internal crisis protocols")
                actions.append("Report to designated crisis coordinator")
            elif user_type in ["executive", "board_member"]:
                actions.append("Review executive crisis briefing")
                actions.append("Prepare for stakeholder communications")
        
        return actions
    
    def _calculate_response_confidence(
        self, 
        query: str, 
        crisis_context: Optional[CrisisContext]
    ) -> float:
        """Calculate confidence score for the response"""
        base_confidence = 0.8
        
        if not crisis_context:
            return base_confidence
        
        # Higher confidence for well-defined crisis types
        if crisis_context.crisis_type in ["system_outage", "security_breach"]:
            base_confidence += 0.1
        
        # Lower confidence for complex queries during crisis
        complex_keywords = ["why", "how", "when", "complex", "detailed"]
        if any(keyword in query.lower() for keyword in complex_keywords):
            base_confidence -= 0.2
        
        return max(0.0, min(1.0, base_confidence))    #
 Helper methods for context enhancement
    def _get_technical_details(self, crisis_context: CrisisContext) -> Dict[str, Any]:
        """Get technical details for internal systems"""
        return {
            "affected_systems": crisis_context.affected_systems,
            "error_codes": ["SYS-001", "NET-404"],  # Simulated
            "performance_impact": "30% degradation",  # Simulated
            "root_cause_analysis": "Under investigation"
        }
    
    def _get_action_items(self, crisis_context: CrisisContext) -> List[str]:
        """Get action items for internal teams"""
        return [
            "Monitor system performance",
            "Prepare customer communications",
            "Coordinate with external vendors",
            "Update stakeholders every 30 minutes"
        ]
    
    def _get_escalation_path(self, crisis_context: CrisisContext) -> List[str]:
        """Get escalation path for the crisis"""
        return self.escalation_rules.get(crisis_context.severity_level.value, ["team_lead"])
    
    def _get_customer_friendly_description(self, crisis_context: CrisisContext) -> str:
        """Get customer-friendly description of the crisis"""
        descriptions = {
            "system_outage": "temporary service interruptions",
            "security_breach": "enhanced security measures",
            "performance_degradation": "slower than usual response times",
            "network_issue": "connectivity challenges",
            "database_failure": "data access delays"
        }
        return descriptions.get(crisis_context.crisis_type, "technical difficulties")
    
    def _get_estimated_resolution_time(self, crisis_context: CrisisContext) -> Optional[str]:
        """Get estimated resolution time"""
        # This would be calculated based on crisis type and historical data
        eta_map = {
            "system_outage": "2-4 hours",
            "security_breach": "24-48 hours",
            "performance_degradation": "1-2 hours",
            "network_issue": "30-60 minutes"
        }
        return eta_map.get(crisis_context.crisis_type)
    
    def _get_official_statement(self, crisis_context: CrisisContext) -> str:
        """Get official company statement for media"""
        return f"We are aware of the {crisis_context.crisis_type.replace('_', ' ')} and are working to resolve it promptly. We will provide updates as they become available."
    
    def _get_company_position(self, crisis_context: CrisisContext) -> str:
        """Get company's official position on the crisis"""
        return "We take this matter seriously and are committed to transparency and swift resolution."
    
    def _get_crisis_duration(self, crisis_context: CrisisContext) -> str:
        """Get crisis duration for executive briefing"""
        duration = datetime.now() - crisis_context.start_time
        hours = int(duration.total_seconds() // 3600)
        minutes = int((duration.total_seconds() % 3600) // 60)
        return f"{hours}h {minutes}m"
    
    def _get_business_impact(self, crisis_context: CrisisContext) -> str:
        """Get business impact assessment"""
        impact_map = {
            "low": "Minimal business impact",
            "medium": "Moderate impact on operations",
            "high": "Significant business disruption",
            "critical": "Severe impact on all operations"
        }
        return impact_map.get(crisis_context.severity_level.value, "Impact assessment in progress")
    
    def _get_executive_actions(self, crisis_context: CrisisContext) -> str:
        """Get recommended executive actions"""
        if crisis_context.severity_level.value == "critical":
            return "Consider activating business continuity plan"
        elif crisis_context.severity_level.value == "high":
            return "Prepare stakeholder communications"
        else:
            return "Monitor situation and prepare for escalation if needed"
    
    def _get_employee_role_in_crisis(
        self, 
        user_context: Optional[Dict[str, Any]], 
        crisis_context: CrisisContext
    ) -> str:
        """Get employee's role in crisis response"""
        if not user_context:
            return "Follow standard crisis procedures"
        
        department = user_context.get("department", "general")
        role_map = {
            "IT": "Technical response and system recovery",
            "Customer Service": "Customer communication and support",
            "Marketing": "External communications and PR",
            "Legal": "Compliance and regulatory response",
            "HR": "Employee communications and support"
        }
        return role_map.get(department, "Support crisis response as directed")
    
    def _get_employee_actions(
        self, 
        user_context: Optional[Dict[str, Any]], 
        crisis_context: CrisisContext
    ) -> str:
        """Get specific actions for employee"""
        if not user_context:
            return "Await further instructions"
        
        department = user_context.get("department", "general")
        if department == "IT":
            return "Monitor systems and report issues immediately"
        elif department == "Customer Service":
            return "Use approved crisis communication scripts"
        else:
            return "Continue normal duties unless otherwise directed"
    
    def _get_compliance_status(self, crisis_context: CrisisContext) -> str:
        """Get regulatory compliance status"""
        return "All required notifications have been submitted"
    
    def _get_corrective_actions(self, crisis_context: CrisisContext) -> str:
        """Get corrective actions for regulatory response"""
        return "Immediate containment measures implemented, full investigation underway"
    
    def _get_reporting_timeline(self, crisis_context: CrisisContext) -> str:
        """Get regulatory reporting timeline"""
        return "Initial report submitted within 24 hours, final report due within 30 days"  
  async def _enable_crisis_aware_responses(self, crisis_context: CrisisContext):
        """Enable crisis-aware response generation across all systems"""
        # This would integrate with chat systems, help desks, etc.
        # to ensure all responses are crisis-aware
        logger.info(f"Enabled crisis-aware responses for crisis {crisis_context.crisis_id}")
    
    async def _trigger_initial_communications(self, crisis_context: CrisisContext):
        """Trigger initial communications across all integrated systems"""
        stakeholders = self._get_crisis_stakeholders(crisis_context)
        
        # Use message coordination engine for coordinated messaging
        coordinated_tasks = []
        
        for stakeholder in stakeholders:
            # Generate crisis-aware message
            message = self._generate_crisis_message("crisis_alert", crisis_context, stakeholder)
            
            # Send through appropriate channels
            task = asyncio.create_task(
                self.send_crisis_aware_message(
                    CommunicationChannel.EMAIL, stakeholder, message, crisis_context, "high"
                )
            )
            coordinated_tasks.append(task)
        
        # Execute all communications
        await asyncio.gather(*coordinated_tasks, return_exceptions=True)
    
    def _get_crisis_stakeholders(self, crisis_context: CrisisContext) -> List[str]:
        base_stakeholders = crisis_context.stakeholders.copy()
        escalation_roles = self.escalation_rules.get(crisis_context.severity_level.value, [])
        base_stakeholders.extend(escalation_roles)
        return list(set(base_stakeholders))
    
    def _generate_crisis_message(self, template_key: str, crisis_context: CrisisContext, recipient: str) -> str:
        template = self.message_templates.get(template_key, "Crisis update: {crisis_type}")
        return template.format(
            crisis_type=crisis_context.crisis_type,
            severity=crisis_context.severity_level.value,
            status=crisis_context.status
        )
    
    async def send_crisis_aware_message(
        self,
        channel: CommunicationChannel,
        recipient: str,
        content: str,
        crisis_context: Optional[CrisisContext] = None,
        priority: str = "normal"
    ) -> CommunicationResponse:
        try:
            message = CommunicationMessage(
                id=f"msg_{datetime.now().timestamp()}",
                channel=channel,
                recipient=recipient,
                content=content,
                crisis_context=crisis_context,
                priority=priority
            )
            
            if crisis_context:
                enhanced_content = self._enhance_message_with_crisis_context(content, crisis_context)
                message.content = enhanced_content
            
            response = await self._send_through_channel(message)
            self.communication_history.append(message)
            
            return response
            
        except Exception as e:
            return CommunicationResponse(
                success=False,
                message_id="",
                channel=channel,
                delivery_status="failed",
                error_message=str(e)
            )
    
    def _enhance_message_with_crisis_context(self, content: str, crisis_context: CrisisContext) -> str:
        crisis_header = f"[CRISIS-{crisis_context.severity_level.value.upper()}] "
        crisis_footer = f"\n\nCrisis ID: {crisis_context.crisis_id} | Status: {crisis_context.status}"
        return crisis_header + content + crisis_footer
    
    async def _send_through_channel(self, message: CommunicationMessage) -> CommunicationResponse:
        await asyncio.sleep(0.1)
        return CommunicationResponse(
            success=True,
            message_id=message.id,
            channel=message.channel,
            delivery_status="delivered"
        )

    async def integrate_with_all_communication_channels(self) -> Dict[str, Any]:
        """Integrate crisis context with all communication channels"""
        try:
            integration_results = {}
            
            for system_name, system_config in self.integrated_systems.items():
                if system_config.is_active:
                    result = await self._integrate_with_system(system_name, system_config)
                    integration_results[system_name] = result
            
            logger.info("Completed integration with all communication channels")
            return {
                "success": True,
                "integrated_systems": len(integration_results),
                "results": integration_results
            }
            
        except Exception as e:
            logger.error(f"Failed to integrate with communication channels: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _integrate_with_system(
        self, 
        system_name: str, 
        system_config: CommunicationSystemIntegration
    ) -> Dict[str, Any]:
        """Integrate with individual communication system"""
        try:
            # Simulate system integration
            await asyncio.sleep(0.1)
            
            # Configure crisis context propagation
            if system_config.crisis_context_support:
                # Enable crisis-aware responses
                logger.info(f"Enabled crisis-aware responses for {system_name}")
                
                # Set up context propagation
                logger.info(f"Configured context propagation for {system_name}")
                
                return {
                    "status": "integrated",
                    "crisis_context_enabled": True,
                    "capabilities": system_config.capabilities
                }
            else:
                return {
                    "status": "basic_integration",
                    "crisis_context_enabled": False,
                    "note": "System does not support crisis context"
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of all system integrations"""
        status = {
            "total_systems": len(self.integrated_systems),
            "active_systems": len([s for s in self.integrated_systems.values() if s.is_active]),
            "crisis_aware_systems": len([s for s in self.integrated_systems.values() if s.crisis_context_support]),
            "active_crises": len(self.active_crises),
            "systems": {}
        }
        
        for name, config in self.integrated_systems.items():
            status["systems"][name] = {
                "type": config.system_type,
                "active": config.is_active,
                "crisis_context_support": config.crisis_context_support,
                "capabilities": config.capabilities
            }
        
        return status

    async def broadcast_crisis_update(
        self,
        crisis_id: str,
        update_message: str,
        channels: List[CommunicationChannel] = None
    ) -> List[CommunicationResponse]:
        try:
            crisis_context = self.active_crises.get(crisis_id)
            if not crisis_context:
                raise ValueError(f"Crisis {crisis_id} not found")
            
            if channels is None:
                channels = [CommunicationChannel.EMAIL, CommunicationChannel.SLACK]
            
            responses = []
            stakeholders = self._get_crisis_stakeholders(crisis_context)
            
            for channel in channels:
                for stakeholder in stakeholders:
                    response = await self.send_crisis_aware_message(
                        channel, stakeholder, update_message, crisis_context, "high"
                    )
                    responses.append(response)
            
            return responses
            
        except Exception as e:
            logger.error(f"Failed to broadcast crisis update: {str(e)}")
            return []
    
    def get_crisis_communication_history(self, crisis_id: str) -> List[CommunicationMessage]:
        return [
            msg for msg in self.communication_history
            if msg.crisis_context and msg.crisis_context.crisis_id == crisis_id
        ]
    
    def get_active_crises(self) -> Dict[str, CrisisContext]:
        return self.active_crises.copy()
    
    async def resolve_crisis(self, crisis_id: str) -> bool:
        try:
            crisis_context = self.active_crises.get(crisis_id)
            if not crisis_context:
                return False
            
            crisis_context.status = "resolved"
            resolution_message = self._generate_crisis_message("resolution", crisis_context, "all")
            
            await self.broadcast_crisis_update(crisis_id, resolution_message)
            del self.active_crises[crisis_id]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to resolve crisis {crisis_id}: {str(e)}")
            return False
    
    def get_communication_metrics(self) -> Dict[str, Any]:
        total_messages = len(self.communication_history)
        active_crises_count = len(self.active_crises)
        
        channel_distribution = {}
        for msg in self.communication_history:
            channel = msg.channel.value
            channel_distribution[channel] = channel_distribution.get(channel, 0) + 1
        
        return {
            "total_messages": total_messages,
            "active_crises": active_crises_count,
            "channel_distribution": channel_distribution,
            "last_message_time": self.communication_history[-1].timestamp if self.communication_history else None
        }

# Global instance
crisis_communication_integration = CrisisCommunicationIntegration()