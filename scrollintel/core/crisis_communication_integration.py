"""
Crisis Communication Integration System

Provides seamless integration with all communication channels during crisis situations,
ensuring crisis-aware response generation and contextual messaging.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import logging

from ..models.crisis_models_simple import Crisis, CrisisStatus, CrisisType
from ..models.crisis_communication_models import NotificationChannel, NotificationPriority

logger = logging.getLogger(__name__)


class CommunicationChannelType(Enum):
    """Types of communication channels"""
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    SMS = "sms"
    PHONE = "phone"
    SOCIAL_MEDIA = "social_media"
    INTERNAL_CHAT = "internal_chat"
    EXTERNAL_API = "external_api"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"


@dataclass
class CrisisContext:
    """Crisis context for communication integration"""
    crisis_id: str
    crisis_type: CrisisType
    severity_level: int
    status: CrisisStatus
    affected_systems: List[str]
    stakeholders: List[str]
    communication_protocols: Dict[str, Any]
    escalation_level: int
    start_time: datetime
    last_update: datetime


@dataclass
class CommunicationIntegrationConfig:
    """Configuration for communication integration"""
    enabled_channels: List[CommunicationChannelType]
    crisis_aware_filtering: bool = True
    auto_context_injection: bool = True
    priority_routing: bool = True
    escalation_triggers: Dict[str, Any] = field(default_factory=dict)
    message_templates: Dict[str, str] = field(default_factory=dict)


class CrisisCommunicationIntegrator:
    """
    Integrates crisis leadership capabilities with all communication systems
    """
    
    def __init__(self, config: CommunicationIntegrationConfig):
        self.config = config
        self.active_crises: Dict[str, CrisisContext] = {}
        self.channel_handlers: Dict[CommunicationChannelType, Any] = {}
        self.message_queue: List[Dict[str, Any]] = []
        self.context_cache: Dict[str, Any] = {}
        
    async def register_crisis(self, crisis: Crisis) -> bool:
        """Register an active crisis for communication integration"""
        try:
            # Convert severity level to integer if it's an enum
            if hasattr(crisis.severity_level, 'value'):
                severity_map = {
                    'low': 1,
                    'medium': 2, 
                    'high': 3,
                    'critical': 4
                }
                severity_int = severity_map.get(crisis.severity_level.value, 2)
            else:
                severity_int = crisis.severity_level
                
            crisis_context = CrisisContext(
                crisis_id=crisis.id,
                crisis_type=crisis.crisis_type,
                severity_level=severity_int,
                status=crisis.current_status,
                affected_systems=crisis.affected_areas,
                stakeholders=crisis.stakeholders_impacted,
                communication_protocols=self._get_communication_protocols(crisis),
                escalation_level=self._calculate_escalation_level(crisis),
                start_time=crisis.start_time,
                last_update=datetime.now()
            )
            
            self.active_crises[crisis.id] = crisis_context
            
            # Initialize crisis-aware communication for all channels
            await self._initialize_crisis_communication(crisis_context)
            
            logger.info(f"Crisis {crisis.id} registered for communication integration")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register crisis {crisis.id}: {str(e)}")
            return False
    
    async def process_communication(
        self, 
        channel: CommunicationChannelType,
        message: str,
        sender: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process communication with crisis awareness"""
        try:
            # Inject crisis context if active crises exist
            enhanced_context = await self._inject_crisis_context(context or {})
            
            # Generate crisis-aware response
            response = await self._generate_crisis_aware_response(
                channel, message, sender, enhanced_context
            )
            
            # Apply crisis communication protocols
            processed_response = await self._apply_crisis_protocols(
                response, channel, enhanced_context
            )
            
            # Route through appropriate channels
            routing_result = await self._route_crisis_communication(
                processed_response, channel, enhanced_context
            )
            
            return {
                "success": True,
                "response": processed_response,
                "routing": routing_result,
                "crisis_context": enhanced_context.get("crisis_info", {}),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Communication processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def broadcast_crisis_update(
        self,
        crisis_id: str,
        update_message: str,
        target_channels: Optional[List[CommunicationChannelType]] = None
    ) -> Dict[str, Any]:
        """Broadcast crisis updates across all relevant communication channels"""
        try:
            if crisis_id not in self.active_crises:
                raise ValueError(f"Crisis {crisis_id} not found in active crises")
            
            crisis_context = self.active_crises[crisis_id]
            channels = target_channels or self.config.enabled_channels
            
            broadcast_results = {}
            
            for channel in channels:
                try:
                    # Customize message for channel
                    customized_message = await self._customize_message_for_channel(
                        update_message, channel, crisis_context
                    )
                    
                    # Send through channel
                    result = await self._send_through_channel(
                        channel, customized_message, crisis_context
                    )
                    
                    broadcast_results[channel.value] = result
                    
                except Exception as e:
                    broadcast_results[channel.value] = {
                        "success": False,
                        "error": str(e)
                    }
            
            return {
                "success": True,
                "crisis_id": crisis_id,
                "broadcast_results": broadcast_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Crisis broadcast failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _inject_crisis_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Inject crisis context into communication context"""
        enhanced_context = context.copy()
        
        if self.active_crises and self.config.auto_context_injection:
            # Find most relevant crisis
            relevant_crisis = await self._find_relevant_crisis(context)
            
            if relevant_crisis:
                enhanced_context["crisis_info"] = {
                    "active_crisis": True,
                    "crisis_id": relevant_crisis.crisis_id,
                    "crisis_type": relevant_crisis.crisis_type.value,
                    "severity": relevant_crisis.severity_level,
                    "status": relevant_crisis.status.value,
                    "escalation_level": relevant_crisis.escalation_level,
                    "affected_systems": relevant_crisis.affected_systems,
                    "communication_protocols": relevant_crisis.communication_protocols
                }
        
        return enhanced_context
    
    async def _generate_crisis_aware_response(
        self,
        channel: CommunicationChannelType,
        message: str,
        sender: str,
        context: Dict[str, Any]
    ) -> str:
        """Generate response with crisis awareness"""
        crisis_info = context.get("crisis_info", {})
        
        if not crisis_info.get("active_crisis"):
            # Normal response generation
            return await self._generate_normal_response(message, sender, context)
        
        # Crisis-aware response generation
        crisis_type = crisis_info.get("crisis_type")
        severity = crisis_info.get("severity", 1)
        status = crisis_info.get("status")
        
        # Apply crisis communication principles
        response_template = self._get_crisis_response_template(
            crisis_type, severity, channel
        )
        
        # Generate contextual response
        response = await self._apply_crisis_response_template(
            response_template, message, sender, crisis_info
        )
        
        return response
    
    async def _apply_crisis_protocols(
        self,
        response: str,
        channel: CommunicationChannelType,
        context: Dict[str, Any]
    ) -> str:
        """Apply crisis communication protocols to response"""
        crisis_info = context.get("crisis_info", {})
        
        if not crisis_info.get("active_crisis"):
            return response
        
        protocols = crisis_info.get("communication_protocols", {})
        
        # Apply protocol modifications
        if protocols.get("require_approval") and crisis_info.get("severity", 1) >= 3:
            response = f"[PENDING APPROVAL] {response}"
        
        if protocols.get("add_crisis_header"):
            crisis_header = f"[CRISIS ALERT - {crisis_info.get('crisis_type', 'UNKNOWN')}] "
            response = f"{crisis_header}{response}"
        
        if protocols.get("include_escalation_info"):
            escalation_info = f"\n\nEscalation Level: {crisis_info.get('escalation_level', 1)}"
            response = f"{response}{escalation_info}"
        
        return response
    
    async def _route_crisis_communication(
        self,
        response: str,
        channel: CommunicationChannelType,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Route communication based on crisis protocols"""
        crisis_info = context.get("crisis_info", {})
        
        routing_result = {
            "primary_channel": channel.value,
            "additional_channels": [],
            "escalated": False,
            "approval_required": False
        }
        
        if not crisis_info.get("active_crisis"):
            return routing_result
        
        severity = crisis_info.get("severity", 1)
        escalation_level = crisis_info.get("escalation_level", 1)
        
        # Determine if escalation is needed
        if severity >= 4 or escalation_level >= 3:
            routing_result["escalated"] = True
            routing_result["additional_channels"] = [
                CommunicationChannelType.EMAIL.value,
                CommunicationChannelType.SMS.value
            ]
        
        # Check if approval is required
        protocols = crisis_info.get("communication_protocols", {})
        if protocols.get("require_approval") and severity >= 3:
            routing_result["approval_required"] = True
        
        return routing_result
    
    def _get_communication_protocols(self, crisis: Crisis) -> Dict[str, Any]:
        """Get communication protocols for crisis type"""
        protocols = {
            "require_approval": False,
            "add_crisis_header": True,
            "include_escalation_info": True,
            "auto_escalate": True,
            "broadcast_updates": True
        }
        
        # Convert severity level to integer if it's an enum
        if hasattr(crisis.severity_level, 'value'):
            severity_map = {
                'low': 1,
                'medium': 2, 
                'high': 3,
                'critical': 4
            }
            severity_int = severity_map.get(crisis.severity_level.value, 2)
        else:
            severity_int = crisis.severity_level
        
        # Adjust based on crisis type and severity
        if severity_int >= 4:
            protocols["require_approval"] = True
            protocols["auto_escalate"] = True
        
        if crisis.crisis_type in [CrisisType.SECURITY_BREACH, CrisisType.DATA_LOSS]:
            protocols["require_approval"] = True
            protocols["restricted_channels"] = ["social_media"]
        
        return protocols
    
    def _calculate_escalation_level(self, crisis: Crisis) -> int:
        """Calculate escalation level based on crisis characteristics"""
        # Convert severity level to integer if it's an enum
        if hasattr(crisis.severity_level, 'value'):
            # Map severity enum to integer
            severity_map = {
                'low': 1,
                'medium': 2, 
                'high': 3,
                'critical': 4
            }
            base_level = severity_map.get(crisis.severity_level.value, 2)
        else:
            base_level = crisis.severity_level
        
        # Adjust based on crisis type
        if crisis.crisis_type in [CrisisType.SECURITY_BREACH, CrisisType.DATA_LOSS]:
            base_level += 1
        
        # Adjust based on affected areas
        if len(crisis.affected_areas) > 3:
            base_level += 1
        
        # Adjust based on stakeholder impact
        if len(crisis.stakeholders_impacted) > 5:
            base_level += 1
        
        return min(base_level, 5)  # Cap at level 5
    
    async def _initialize_crisis_communication(self, crisis_context: CrisisContext):
        """Initialize crisis-aware communication for all channels"""
        for channel in self.config.enabled_channels:
            try:
                await self._setup_channel_crisis_mode(channel, crisis_context)
            except Exception as e:
                logger.error(f"Failed to setup crisis mode for {channel}: {str(e)}")
    
    async def _setup_channel_crisis_mode(
        self, 
        channel: CommunicationChannelType, 
        crisis_context: CrisisContext
    ):
        """Setup crisis mode for specific communication channel"""
        # Channel-specific crisis mode setup
        if channel == CommunicationChannelType.SLACK:
            await self._setup_slack_crisis_mode(crisis_context)
        elif channel == CommunicationChannelType.EMAIL:
            await self._setup_email_crisis_mode(crisis_context)
        elif channel == CommunicationChannelType.DASHBOARD:
            await self._setup_dashboard_crisis_mode(crisis_context)
        # Add more channel-specific setups as needed
    
    async def _find_relevant_crisis(self, context: Dict[str, Any]) -> Optional[CrisisContext]:
        """Find the most relevant active crisis for the current context"""
        if not self.active_crises:
            return None
        
        # Simple relevance scoring - can be enhanced
        best_crisis = None
        best_score = 0
        
        for crisis in self.active_crises.values():
            score = self._calculate_crisis_relevance(crisis, context)
            if score > best_score:
                best_score = score
                best_crisis = crisis
        
        return best_crisis if best_score > 0 else list(self.active_crises.values())[0]
    
    def _calculate_crisis_relevance(
        self, 
        crisis: CrisisContext, 
        context: Dict[str, Any]
    ) -> float:
        """Calculate relevance score for crisis to current context"""
        score = 0.0
        
        # Base score from severity
        score += crisis.severity_level * 0.2
        
        # Score from affected systems
        context_systems = context.get("affected_systems", [])
        if context_systems:
            overlap = set(crisis.affected_systems) & set(context_systems)
            score += len(overlap) * 0.3
        
        # Score from stakeholders
        context_stakeholders = context.get("stakeholders", [])
        if context_stakeholders:
            overlap = set(crisis.stakeholders) & set(context_stakeholders)
            score += len(overlap) * 0.2
        
        # Recency score
        time_diff = (datetime.now() - crisis.last_update).total_seconds()
        recency_score = max(0, 1 - (time_diff / 3600))  # Decay over 1 hour
        score += recency_score * 0.3
        
        return score
    
    def _get_crisis_response_template(
        self,
        crisis_type: str,
        severity: int,
        channel: CommunicationChannelType
    ) -> str:
        """Get response template for crisis type and channel"""
        template_key = f"{crisis_type}_{severity}_{channel.value}"
        
        if template_key in self.config.message_templates:
            return self.config.message_templates[template_key]
        
        # Default templates
        if severity >= 4:
            return "URGENT: Crisis situation detected. Immediate attention required. {message}"
        elif severity >= 3:
            return "ALERT: Crisis situation in progress. {message}"
        else:
            return "NOTICE: Potential crisis situation. {message}"
    
    async def _apply_crisis_response_template(
        self,
        template: str,
        original_message: str,
        sender: str,
        crisis_info: Dict[str, Any]
    ) -> str:
        """Apply crisis response template with context"""
        return template.format(
            message=original_message,
            sender=sender,
            crisis_type=crisis_info.get("crisis_type", "UNKNOWN"),
            severity=crisis_info.get("severity", 1),
            status=crisis_info.get("status", "UNKNOWN")
        )
    
    async def _generate_normal_response(
        self,
        message: str,
        sender: str,
        context: Dict[str, Any]
    ) -> str:
        """Generate normal response when no crisis is active"""
        # Placeholder for normal response generation
        return f"Acknowledged: {message}"
    
    async def _customize_message_for_channel(
        self,
        message: str,
        channel: CommunicationChannelType,
        crisis_context: CrisisContext
    ) -> str:
        """Customize message for specific communication channel"""
        if channel == CommunicationChannelType.SMS:
            # Truncate for SMS
            return message[:160] + "..." if len(message) > 160 else message
        elif channel == CommunicationChannelType.EMAIL:
            # Add email formatting
            return f"Subject: Crisis Update - {crisis_context.crisis_type.value}\n\n{message}"
        elif channel == CommunicationChannelType.SLACK:
            # Add Slack formatting
            return f"🚨 *Crisis Update* 🚨\n{message}"
        else:
            return message
    
    async def _send_through_channel(
        self,
        channel: CommunicationChannelType,
        message: str,
        crisis_context: CrisisContext
    ) -> Dict[str, Any]:
        """Send message through specific communication channel"""
        # Placeholder for actual channel integration
        return {
            "success": True,
            "channel": channel.value,
            "message_id": f"msg_{datetime.now().timestamp()}",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _setup_slack_crisis_mode(self, crisis_context: CrisisContext):
        """Setup Slack for crisis mode"""
        # Placeholder for Slack-specific crisis setup
        pass
    
    async def _setup_email_crisis_mode(self, crisis_context: CrisisContext):
        """Setup email for crisis mode"""
        # Placeholder for email-specific crisis setup
        pass
    
    async def _setup_dashboard_crisis_mode(self, crisis_context: CrisisContext):
        """Setup dashboard for crisis mode"""
        # Placeholder for dashboard-specific crisis setup
        pass
    
    async def update_crisis_status(self, crisis_id: str, new_status: CrisisStatus):
        """Update crisis status and adjust communication accordingly"""
        if crisis_id in self.active_crises:
            self.active_crises[crisis_id].status = new_status
            self.active_crises[crisis_id].last_update = datetime.now()
            
            # Broadcast status update
            await self.broadcast_crisis_update(
                crisis_id,
                f"Crisis status updated to: {new_status.value}"
            )
    
    async def resolve_crisis(self, crisis_id: str):
        """Mark crisis as resolved and cleanup communication integration"""
        if crisis_id in self.active_crises:
            # Broadcast resolution
            await self.broadcast_crisis_update(
                crisis_id,
                "Crisis has been resolved. Normal operations resuming."
            )
            
            # Remove from active crises
            del self.active_crises[crisis_id]
            
            logger.info(f"Crisis {crisis_id} resolved and removed from active crises")
    
    def get_active_crises(self) -> List[Dict[str, Any]]:
        """Get list of active crises"""
        return [
            {
                "crisis_id": crisis.crisis_id,
                "crisis_type": crisis.crisis_type.value,
                "severity": crisis.severity_level,
                "status": crisis.status.value,
                "start_time": crisis.start_time.isoformat(),
                "last_update": crisis.last_update.isoformat()
            }
            for crisis in self.active_crises.values()
        ]