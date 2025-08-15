"""
External Resource Coordination Engine for Crisis Leadership Excellence

This engine provides coordination with external partners and vendors,
external resource request and management, and partnership activation
and coordination protocols during crisis situations.
"""

from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import json

from ..models.resource_mobilization_models import (
    ExternalPartner, ExternalResourceRequest, CoordinationProtocol,
    ResourceType, ResourcePriority, ResourceRequirement
)
from ..models.crisis_models_simple import Crisis

logger = logging.getLogger(__name__)


class PartnerType(Enum):
    """Types of external partners"""
    VENDOR = "vendor"
    CONTRACTOR = "contractor"
    PARTNER_ORGANIZATION = "partner_organization"
    GOVERNMENT_AGENCY = "government_agency"
    EMERGENCY_SERVICES = "emergency_services"
    CONSULTING_FIRM = "consulting_firm"
    CLOUD_PROVIDER = "cloud_provider"
    EQUIPMENT_SUPPLIER = "equipment_supplier"
    STAFFING_AGENCY = "staffing_agency"


class RequestStatus(Enum):
    """Status of external resource requests"""
    DRAFT = "draft"
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    REJECTED = "rejected"
    FULFILLED = "fulfilled"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ActivationLevel(Enum):
    """Partnership activation levels"""
    STANDBY = "standby"
    ALERT = "alert"
    ACTIVATED = "activated"
    FULL_DEPLOYMENT = "full_deployment"
    EMERGENCY_RESPONSE = "emergency_response"


@dataclass
class PartnerCapability:
    """Capability offered by external partner"""
    id: str = field(default_factory=lambda: str(datetime.now().timestamp()))
    name: str = ""
    description: str = ""
    resource_type: ResourceType = ResourceType.EXTERNAL_SERVICES
    capacity: float = 0.0
    availability: str = "24/7"
    response_time: timedelta = field(default_factory=lambda: timedelta(hours=4))
    cost_structure: Dict[str, float] = field(default_factory=dict)
    quality_rating: float = 0.0
    last_used: Optional[datetime] = None
    certification_level: str = ""
    geographic_coverage: List[str] = field(default_factory=list)


@dataclass
class PartnerPerformance:
    """Performance tracking for external partners"""
    partner_id: str = ""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: timedelta = field(default_factory=lambda: timedelta(hours=0))
    average_fulfillment_time: timedelta = field(default_factory=lambda: timedelta(hours=0))
    reliability_score: float = 0.0
    quality_score: float = 0.0
    cost_efficiency: float = 0.0
    last_engagement: Optional[datetime] = None
    performance_trends: Dict[str, List[float]] = field(default_factory=dict)
    feedback_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoordinationEvent:
    """Event in partner coordination process"""
    id: str = field(default_factory=lambda: str(datetime.now().timestamp()))
    partner_id: str = ""
    event_type: str = ""
    event_description: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    initiated_by: str = ""
    status: str = "completed"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExternalResourceCoordinator:
    """
    Comprehensive external resource coordination system.
    
    Provides coordination with external partners, resource request management,
    and partnership activation protocols for crisis response.
    """
    
    def __init__(self):
        self.partner_registry = PartnerRegistry()
        self.request_manager = RequestManager()
        self.protocol_engine = ProtocolEngine()
        self.performance_tracker = PerformanceTracker()
        self.communication_hub = CommunicationHub()
        self.active_requests = {}
        self.coordination_history = []
    
    async def coordinate_with_partners(
        self,
        crisis: Crisis,
        resource_requirements: List[ResourceRequirement]
    ) -> Dict[str, Any]:
        """
        Coordinate with external partners for crisis response
        
        Args:
            crisis: Crisis requiring external resources
            resource_requirements: List of resource requirements
            
        Returns:
            Dict containing coordination results and partner responses
        """
        logger.info(f"Starting external resource coordination for crisis {crisis.id}")
        
        try:
            # Identify suitable partners for requirements
            suitable_partners = await self._identify_suitable_partners(resource_requirements)
            
            # Create coordination plan
            coordination_plan = await self._create_coordination_plan(
                crisis, resource_requirements, suitable_partners
            )
            
            # Activate partnerships based on crisis severity
            activation_results = await self._activate_partnerships(
                crisis, coordination_plan['partner_activations']
            )
            
            # Submit resource requests
            request_results = await self._submit_resource_requests(
                coordination_plan['resource_requests']
            )
            
            # Monitor partner responses
            response_monitoring = await self._initiate_response_monitoring(
                request_results['submitted_requests']
            )
            
            # Generate coordination summary
            coordination_summary = {
                'crisis_id': crisis.id,
                'coordination_plan': coordination_plan,
                'activation_results': activation_results,
                'request_results': request_results,
                'response_monitoring': response_monitoring,
                'total_partners_contacted': len(suitable_partners),
                'total_requests_submitted': len(request_results['submitted_requests']),
                'estimated_response_time': self._calculate_estimated_response_time(suitable_partners),
                'coordination_status': 'active',
                'coordination_timestamp': datetime.utcnow()
            }
            
            # Record coordination event
            await self._record_coordination_event(
                crisis.id, 'coordination_initiated', coordination_summary
            )
            
            logger.info(f"External resource coordination initiated for crisis {crisis.id}")
            return coordination_summary
            
        except Exception as e:
            logger.error(f"Error in external resource coordination: {str(e)}")
            raise
    
    async def manage_resource_requests(
        self,
        requests: List[ExternalResourceRequest]
    ) -> Dict[str, Any]:
        """
        Manage external resource requests throughout their lifecycle
        
        Args:
            requests: List of external resource requests to manage
            
        Returns:
            Dict containing request management results
        """
        logger.info(f"Managing {len(requests)} external resource requests")
        
        management_results = {
            'total_requests': len(requests),
            'request_status_summary': {},
            'successful_requests': [],
            'failed_requests': [],
            'pending_requests': [],
            'management_actions': [],
            'next_steps': []
        }
        
        for request in requests:
            try:
                # Update request status
                current_status = await self.request_manager.get_request_status(request.id)
                
                # Take appropriate management action based on status
                action_result = await self._take_management_action(request, current_status)
                management_results['management_actions'].append(action_result)
                
                # Categorize request based on current status
                if current_status in [RequestStatus.FULFILLED.value]:
                    management_results['successful_requests'].append(request.id)
                elif current_status in [RequestStatus.REJECTED.value, RequestStatus.FAILED.value]:
                    management_results['failed_requests'].append(request.id)
                else:
                    management_results['pending_requests'].append(request.id)
                
                # Update status summary
                status_key = current_status
                management_results['request_status_summary'][status_key] = \
                    management_results['request_status_summary'].get(status_key, 0) + 1
                
            except Exception as e:
                logger.error(f"Error managing request {request.id}: {str(e)}")
                management_results['failed_requests'].append(request.id)
        
        # Generate next steps
        management_results['next_steps'] = await self._generate_management_next_steps(
            management_results
        )
        
        logger.info(f"Resource request management completed")
        return management_results
    
    async def activate_partnership_protocols(
        self,
        partner_id: str,
        activation_level: ActivationLevel,
        crisis_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Activate partnership coordination protocols
        
        Args:
            partner_id: ID of partner to activate
            activation_level: Level of activation required
            crisis_context: Context information about the crisis
            
        Returns:
            Dict containing activation results
        """
        logger.info(f"Activating partnership protocols for partner {partner_id}")
        
        try:
            # Get partner information
            partner = await self.partner_registry.get_partner(partner_id)
            if not partner:
                raise ValueError(f"Partner {partner_id} not found")
            
            # Get coordination protocol
            protocol = await self.protocol_engine.get_protocol(partner_id)
            if not protocol:
                # Create default protocol
                protocol = await self.protocol_engine.create_default_protocol(partner)
            
            # Execute activation steps
            activation_steps = await self._execute_activation_steps(
                partner, protocol, activation_level, crisis_context
            )
            
            # Establish communication channels
            communication_setup = await self.communication_hub.establish_channels(
                partner, protocol.communication_channels
            )
            
            # Verify activation
            activation_verification = await self._verify_activation(
                partner, activation_level
            )
            
            # Record activation event
            activation_event = CoordinationEvent(
                partner_id=partner_id,
                event_type='partnership_activated',
                event_description=f'Partnership activated at {activation_level.value} level',
                initiated_by='external_resource_coordinator',
                metadata={
                    'activation_level': activation_level.value,
                    'crisis_context': crisis_context,
                    'activation_steps': len(activation_steps),
                    'communication_channels': len(communication_setup['established_channels'])
                }
            )
            
            self.coordination_history.append(activation_event)
            
            activation_result = {
                'partner_id': partner_id,
                'partner_name': partner.name,
                'activation_level': activation_level.value,
                'activation_successful': activation_verification['success'],
                'activation_steps': activation_steps,
                'communication_setup': communication_setup,
                'verification_result': activation_verification,
                'estimated_response_time': partner.response_time_sla,
                'activation_timestamp': datetime.utcnow(),
                'next_actions': await self._get_post_activation_actions(partner, activation_level)
            }
            
            logger.info(f"Partnership activation completed for partner {partner_id}")
            return activation_result
            
        except Exception as e:
            logger.error(f"Error activating partnership for {partner_id}: {str(e)}")
            raise
    
    async def monitor_partner_performance(
        self,
        partner_ids: Optional[List[str]] = None,
        time_window: timedelta = timedelta(days=30)
    ) -> Dict[str, PartnerPerformance]:
        """
        Monitor performance of external partners
        
        Args:
            partner_ids: List of partner IDs to monitor (None for all)
            time_window: Time window for performance analysis
            
        Returns:
            Dict mapping partner IDs to performance data
        """
        logger.info(f"Monitoring partner performance for time window: {time_window}")
        
        if partner_ids is None:
            partners = await self.partner_registry.get_all_partners()
            partner_ids = [p.id for p in partners]
        
        performance_data = {}
        
        for partner_id in partner_ids:
            try:
                performance = await self.performance_tracker.analyze_performance(
                    partner_id, time_window
                )
                performance_data[partner_id] = performance
                
            except Exception as e:
                logger.error(f"Error monitoring performance for partner {partner_id}: {str(e)}")
        
        logger.info(f"Performance monitoring completed for {len(performance_data)} partners")
        return performance_data
    
    async def _identify_suitable_partners(
        self,
        requirements: List[ResourceRequirement]
    ) -> List[ExternalPartner]:
        """Identify partners suitable for resource requirements"""
        suitable_partners = []
        all_partners = await self.partner_registry.get_all_partners()
        
        for requirement in requirements:
            for partner in all_partners:
                # Check if partner can provide required resource type
                if requirement.resource_type in partner.available_resources:
                    # Check capabilities match
                    partner_capabilities = set(partner.service_capabilities)
                    required_capabilities = set(requirement.required_capabilities)
                    
                    if required_capabilities.issubset(partner_capabilities):
                        # Check if partner is not already in list
                        if partner not in suitable_partners:
                            suitable_partners.append(partner)
        
        # Sort by reliability score
        suitable_partners.sort(key=lambda p: p.reliability_score, reverse=True)
        
        return suitable_partners
    
    async def _create_coordination_plan(
        self,
        crisis: Crisis,
        requirements: List[ResourceRequirement],
        partners: List[ExternalPartner]
    ) -> Dict[str, Any]:
        """Create comprehensive coordination plan"""
        plan = {
            'crisis_id': crisis.id,
            'partner_activations': [],
            'resource_requests': [],
            'communication_plan': {},
            'timeline': {},
            'risk_mitigation': []
        }
        
        # Plan partner activations
        for partner in partners:
            activation_level = self._determine_activation_level(crisis, partner)
            plan['partner_activations'].append({
                'partner_id': partner.id,
                'activation_level': activation_level.value,
                'priority': self._calculate_partner_priority(partner, requirements),
                'estimated_activation_time': partner.response_time_sla
            })
        
        # Plan resource requests
        for requirement in requirements:
            matching_partners = [
                p for p in partners
                if requirement.resource_type in p.available_resources
            ]
            
            for partner in matching_partners[:3]:  # Top 3 partners per requirement
                request = ExternalResourceRequest(
                    crisis_id=crisis.id,
                    partner_id=partner.id,
                    resource_type=requirement.resource_type,
                    requested_capabilities=requirement.required_capabilities,
                    quantity_requested=requirement.quantity_needed,
                    urgency_level=requirement.priority,
                    duration_needed=requirement.duration_needed,
                    budget_approved=requirement.budget_limit,
                    requested_by='external_resource_coordinator'
                )
                plan['resource_requests'].append(request)
        
        # Create communication plan
        plan['communication_plan'] = {
            'primary_contacts': [p.emergency_contact for p in partners],
            'communication_frequency': 'hourly' if crisis.severity_level.value >= 4 else 'every_4_hours',
            'escalation_triggers': ['no_response_2_hours', 'request_rejection', 'delivery_delay'],
            'status_reporting': 'real_time'
        }
        
        # Create timeline
        plan['timeline'] = {
            'activation_start': datetime.utcnow(),
            'first_responses_expected': datetime.utcnow() + timedelta(hours=2),
            'resource_deployment_target': datetime.utcnow() + timedelta(hours=8),
            'full_coordination_established': datetime.utcnow() + timedelta(hours=4)
        }
        
        return plan
    
    async def _activate_partnerships(
        self,
        crisis: Crisis,
        activations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Activate partnerships according to plan"""
        results = {
            'successful_activations': [],
            'failed_activations': [],
            'pending_activations': [],
            'total_activated': 0
        }
        
        for activation in activations:
            try:
                partner_id = activation['partner_id']
                activation_level = ActivationLevel(activation['activation_level'])
                
                result = await self.activate_partnership_protocols(
                    partner_id, activation_level, {'crisis_id': crisis.id}
                )
                
                if result['activation_successful']:
                    results['successful_activations'].append(result)
                    results['total_activated'] += 1
                else:
                    results['failed_activations'].append(result)
                    
            except Exception as e:
                logger.error(f"Error activating partner {activation['partner_id']}: {str(e)}")
                results['failed_activations'].append({
                    'partner_id': activation['partner_id'],
                    'error': str(e)
                })
        
        return results
    
    async def _submit_resource_requests(
        self,
        requests: List[ExternalResourceRequest]
    ) -> Dict[str, Any]:
        """Submit resource requests to partners"""
        results = {
            'submitted_requests': [],
            'failed_submissions': [],
            'total_submitted': 0
        }
        
        for request in requests:
            try:
                submission_result = await self.request_manager.submit_request(request)
                
                if submission_result['success']:
                    results['submitted_requests'].append(request)
                    results['total_submitted'] += 1
                    self.active_requests[request.id] = request
                else:
                    results['failed_submissions'].append({
                        'request_id': request.id,
                        'error': submission_result['error']
                    })
                    
            except Exception as e:
                logger.error(f"Error submitting request {request.id}: {str(e)}")
                results['failed_submissions'].append({
                    'request_id': request.id,
                    'error': str(e)
                })
        
        return results
    
    async def _initiate_response_monitoring(
        self,
        requests: List[ExternalResourceRequest]
    ) -> Dict[str, Any]:
        """Initiate monitoring of partner responses"""
        monitoring_setup = {
            'monitored_requests': len(requests),
            'monitoring_frequency': 'every_30_minutes',
            'escalation_thresholds': {
                'no_acknowledgment': timedelta(hours=1),
                'no_response': timedelta(hours=4),
                'delivery_delay': timedelta(hours=12)
            },
            'monitoring_channels': ['email', 'api', 'phone'],
            'monitoring_started': datetime.utcnow()
        }
        
        # Start monitoring tasks (would be implemented with actual monitoring system)
        for request in requests:
            logger.info(f"Monitoring initiated for request {request.id}")
        
        return monitoring_setup
    
    async def _take_management_action(
        self,
        request: ExternalResourceRequest,
        current_status: str
    ) -> Dict[str, Any]:
        """Take appropriate management action based on request status"""
        action_result = {
            'request_id': request.id,
            'current_status': current_status,
            'action_taken': 'none',
            'action_successful': True,
            'next_check': datetime.utcnow() + timedelta(hours=1)
        }
        
        if current_status == RequestStatus.PENDING.value:
            # Follow up on pending request
            action_result['action_taken'] = 'follow_up_sent'
            # Would send actual follow-up communication
            
        elif current_status == RequestStatus.REJECTED.value:
            # Try alternative partners or negotiate
            action_result['action_taken'] = 'alternative_partner_contacted'
            # Would contact alternative partners
            
        elif current_status == RequestStatus.IN_PROGRESS.value:
            # Monitor progress and provide support
            action_result['action_taken'] = 'progress_monitored'
            # Would check progress with partner
            
        elif current_status == RequestStatus.FAILED.value:
            # Escalate and find alternatives
            action_result['action_taken'] = 'escalation_initiated'
            # Would escalate to management and find alternatives
        
        return action_result
    
    async def _generate_management_next_steps(
        self,
        management_results: Dict[str, Any]
    ) -> List[str]:
        """Generate next steps for request management"""
        next_steps = []
        
        if management_results['failed_requests']:
            next_steps.append("Review failed requests and identify alternative solutions")
            next_steps.append("Escalate critical failed requests to senior management")
        
        if management_results['pending_requests']:
            next_steps.append("Follow up on pending requests with partners")
            next_steps.append("Set escalation timers for overdue responses")
        
        if management_results['successful_requests']:
            next_steps.append("Coordinate delivery and integration of fulfilled requests")
            next_steps.append("Update resource allocation plans with confirmed resources")
        
        next_steps.append("Continue monitoring all active requests")
        next_steps.append("Update crisis leadership on resource coordination status")
        
        return next_steps
    
    def _determine_activation_level(
        self,
        crisis: Crisis,
        partner: ExternalPartner
    ) -> ActivationLevel:
        """Determine appropriate activation level for partner"""
        if crisis.severity_level.value >= 4:  # Critical or higher
            return ActivationLevel.EMERGENCY_RESPONSE
        elif crisis.severity_level.value >= 3:  # High
            return ActivationLevel.FULL_DEPLOYMENT
        elif crisis.severity_level.value >= 2:  # Medium
            return ActivationLevel.ACTIVATED
        else:
            return ActivationLevel.ALERT
    
    def _calculate_partner_priority(
        self,
        partner: ExternalPartner,
        requirements: List[ResourceRequirement]
    ) -> int:
        """Calculate priority score for partner"""
        priority_score = 0
        
        # Base score from reliability
        priority_score += int(partner.reliability_score * 50)
        
        # Bonus for response time
        if partner.response_time_sla <= timedelta(hours=2):
            priority_score += 20
        elif partner.response_time_sla <= timedelta(hours=4):
            priority_score += 10
        
        # Bonus for capability match
        partner_capabilities = set(partner.service_capabilities)
        for req in requirements:
            required_capabilities = set(req.required_capabilities)
            if required_capabilities.issubset(partner_capabilities):
                priority_score += 15
        
        return min(100, priority_score)
    
    def _calculate_estimated_response_time(
        self,
        partners: List[ExternalPartner]
    ) -> timedelta:
        """Calculate estimated response time from partners"""
        if not partners:
            return timedelta(hours=24)
        
        # Use fastest partner's response time
        fastest_response = min(p.response_time_sla for p in partners)
        return fastest_response
    
    async def _execute_activation_steps(
        self,
        partner: ExternalPartner,
        protocol: CoordinationProtocol,
        activation_level: ActivationLevel,
        crisis_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute partnership activation steps"""
        steps = []
        
        # Step 1: Initial contact
        steps.append({
            'step': 'initial_contact',
            'description': f'Contact {partner.name} emergency contact',
            'status': 'completed',
            'timestamp': datetime.utcnow()
        })
        
        # Step 2: Activation notification
        steps.append({
            'step': 'activation_notification',
            'description': f'Send activation notification at {activation_level.value} level',
            'status': 'completed',
            'timestamp': datetime.utcnow()
        })
        
        # Step 3: Protocol execution
        for trigger in protocol.activation_triggers:
            steps.append({
                'step': 'protocol_execution',
                'description': f'Execute protocol trigger: {trigger}',
                'status': 'completed',
                'timestamp': datetime.utcnow()
            })
        
        return steps
    
    async def _verify_activation(
        self,
        partner: ExternalPartner,
        activation_level: ActivationLevel
    ) -> Dict[str, Any]:
        """Verify partnership activation"""
        # Mock verification - would implement actual verification
        return {
            'success': True,
            'verification_method': 'api_confirmation',
            'partner_response_time': timedelta(minutes=15),
            'activation_confirmed': True,
            'partner_status': 'active',
            'verification_timestamp': datetime.utcnow()
        }
    
    async def _get_post_activation_actions(
        self,
        partner: ExternalPartner,
        activation_level: ActivationLevel
    ) -> List[str]:
        """Get actions to take after activation"""
        actions = [
            f"Monitor {partner.name} response and readiness",
            "Establish regular communication schedule",
            "Prepare resource request submissions"
        ]
        
        if activation_level in [ActivationLevel.EMERGENCY_RESPONSE, ActivationLevel.FULL_DEPLOYMENT]:
            actions.extend([
                "Verify emergency contact availability",
                "Confirm expedited service level agreements",
                "Prepare for immediate resource deployment"
            ])
        
        return actions
    
    async def _record_coordination_event(
        self,
        crisis_id: str,
        event_type: str,
        event_data: Dict[str, Any]
    ):
        """Record coordination event for audit and analysis"""
        event = CoordinationEvent(
            partner_id=event_data.get('partner_id', 'multiple'),
            event_type=event_type,
            event_description=f"Crisis {crisis_id}: {event_type}",
            initiated_by='external_resource_coordinator',
            metadata=event_data
        )
        
        self.coordination_history.append(event)
        logger.info(f"Coordination event recorded: {event_type} for crisis {crisis_id}")


class PartnerRegistry:
    """Registry for managing external partner information"""
    
    def __init__(self):
        self.partners = {}
        self._initialize_sample_partners()
    
    async def get_partner(self, partner_id: str) -> Optional[ExternalPartner]:
        """Get partner by ID"""
        return self.partners.get(partner_id)
    
    async def get_all_partners(self) -> List[ExternalPartner]:
        """Get all registered partners"""
        return list(self.partners.values())
    
    async def register_partner(self, partner: ExternalPartner) -> bool:
        """Register new partner"""
        try:
            self.partners[partner.id] = partner
            logger.info(f"Registered partner {partner.id}: {partner.name}")
            return True
        except Exception as e:
            logger.error(f"Error registering partner: {str(e)}")
            return False
    
    def _initialize_sample_partners(self):
        """Initialize sample partners for demonstration"""
        partners = [
            ExternalPartner(
                name="Emergency IT Services Inc.",
                partner_type="vendor",
                contact_info={
                    "primary_phone": "+1-800-EMERGENCY",
                    "primary_email": "emergency@itservices.com",
                    "website": "https://emergencyit.com"
                },
                available_resources=[ResourceType.HUMAN_RESOURCES, ResourceType.TECHNICAL_INFRASTRUCTURE],
                service_capabilities=[
                    "24/7_support", "emergency_response", "system_recovery",
                    "incident_response", "technical_consulting"
                ],
                response_time_sla=timedelta(hours=2),
                cost_structure={
                    "hourly_rate": 200.0,
                    "emergency_surcharge": 1.5,
                    "minimum_engagement": 4.0
                },
                reliability_score=0.92,
                emergency_contact={
                    "name": "Emergency Operations Center",
                    "phone": "+1-800-EMERGENCY",
                    "email": "eoc@itservices.com"
                },
                activation_protocol=[
                    "Call emergency hotline",
                    "Provide crisis details and requirements",
                    "Receive confirmation and ETA",
                    "Coordinate resource deployment"
                ]
            ),
            ExternalPartner(
                name="CloudScale Solutions",
                partner_type="cloud_provider",
                contact_info={
                    "primary_phone": "+1-888-CLOUDSCALE",
                    "primary_email": "support@cloudscale.com",
                    "website": "https://cloudscale.com"
                },
                available_resources=[ResourceType.CLOUD_COMPUTE, ResourceType.DATA_STORAGE],
                service_capabilities=[
                    "elastic_scaling", "disaster_recovery", "global_deployment",
                    "managed_services", "security_services"
                ],
                response_time_sla=timedelta(minutes=30),
                cost_structure={
                    "compute_per_hour": 0.50,
                    "storage_per_gb": 0.10,
                    "data_transfer_per_gb": 0.05
                },
                reliability_score=0.98,
                emergency_contact={
                    "name": "24/7 Support Team",
                    "phone": "+1-888-CLOUDSCALE",
                    "email": "emergency@cloudscale.com"
                },
                activation_protocol=[
                    "Submit emergency scaling request via API",
                    "Provide resource specifications",
                    "Receive auto-provisioning confirmation",
                    "Monitor deployment status"
                ]
            ),
            ExternalPartner(
                name="Crisis Management Consultants",
                partner_type="consulting_firm",
                contact_info={
                    "primary_phone": "+1-555-CRISIS-HELP",
                    "primary_email": "help@crisismanagement.com",
                    "website": "https://crisismanagement.com"
                },
                available_resources=[ResourceType.HUMAN_RESOURCES, ResourceType.EXTERNAL_SERVICES],
                service_capabilities=[
                    "crisis_leadership", "strategic_planning", "stakeholder_communication",
                    "media_relations", "business_continuity"
                ],
                response_time_sla=timedelta(hours=1),
                cost_structure={
                    "senior_consultant_hourly": 350.0,
                    "consultant_hourly": 250.0,
                    "emergency_response_fee": 5000.0
                },
                reliability_score=0.95,
                emergency_contact={
                    "name": "Crisis Response Team",
                    "phone": "+1-555-CRISIS-HELP",
                    "email": "emergency@crisismanagement.com"
                },
                activation_protocol=[
                    "Contact crisis response hotline",
                    "Brief team on crisis situation",
                    "Deploy senior consultants",
                    "Establish command center liaison"
                ]
            )
        ]
        
        for partner in partners:
            self.partners[partner.id] = partner


class RequestManager:
    """Manager for external resource requests"""
    
    def __init__(self):
        self.requests = {}
        self.request_history = []
    
    async def submit_request(self, request: ExternalResourceRequest) -> Dict[str, Any]:
        """Submit resource request to partner"""
        try:
            # Mock submission - would integrate with partner APIs
            self.requests[request.id] = request
            request.request_status = RequestStatus.SUBMITTED.value
            
            # Record submission
            self.request_history.append({
                'request_id': request.id,
                'action': 'submitted',
                'timestamp': datetime.utcnow(),
                'partner_id': request.partner_id
            })
            
            return {
                'success': True,
                'request_id': request.id,
                'submission_timestamp': datetime.utcnow(),
                'expected_response_time': timedelta(hours=2)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'request_id': request.id
            }
    
    async def get_request_status(self, request_id: str) -> str:
        """Get current status of request"""
        request = self.requests.get(request_id)
        if request:
            # Mock status progression
            return RequestStatus.IN_PROGRESS.value
        return RequestStatus.FAILED.value


class ProtocolEngine:
    """Engine for managing coordination protocols"""
    
    def __init__(self):
        self.protocols = {}
    
    async def get_protocol(self, partner_id: str) -> Optional[CoordinationProtocol]:
        """Get coordination protocol for partner"""
        return self.protocols.get(partner_id)
    
    async def create_default_protocol(self, partner: ExternalPartner) -> CoordinationProtocol:
        """Create default coordination protocol for partner"""
        protocol = CoordinationProtocol(
            partner_id=partner.id,
            protocol_name=f"Default Protocol - {partner.name}",
            activation_triggers=[
                "crisis_severity_high",
                "resource_shortage_critical",
                "manual_activation"
            ],
            communication_channels=[
                "email", "phone", "api"
            ],
            escalation_procedures=[
                "primary_contact_no_response_1h",
                "secondary_contact_no_response_2h",
                "management_escalation_4h"
            ],
            resource_request_process=[
                "submit_formal_request",
                "await_acknowledgment",
                "negotiate_terms",
                "confirm_deployment"
            ],
            quality_standards={
                "response_time_sla": partner.response_time_sla.total_seconds(),
                "availability_requirement": "99.9%",
                "performance_threshold": 0.95
            },
            performance_metrics=[
                "response_time", "fulfillment_rate", "quality_score"
            ],
            review_schedule="quarterly"
        )
        
        self.protocols[partner.id] = protocol
        return protocol


class PerformanceTracker:
    """Tracker for partner performance metrics"""
    
    async def analyze_performance(
        self,
        partner_id: str,
        time_window: timedelta
    ) -> PartnerPerformance:
        """Analyze partner performance over time window"""
        # Mock performance analysis - would integrate with real data
        performance = PartnerPerformance(
            partner_id=partner_id,
            total_requests=25,
            successful_requests=23,
            failed_requests=2,
            average_response_time=timedelta(hours=1, minutes=30),
            average_fulfillment_time=timedelta(hours=6),
            reliability_score=0.92,
            quality_score=0.88,
            cost_efficiency=0.85,
            last_engagement=datetime.utcnow() - timedelta(days=5),
            performance_trends={
                "response_time": [1.5, 1.3, 1.4, 1.2, 1.1],
                "quality_score": [0.85, 0.87, 0.88, 0.89, 0.88],
                "cost_efficiency": [0.82, 0.84, 0.85, 0.86, 0.85]
            },
            feedback_summary={
                "positive_feedback": 18,
                "neutral_feedback": 5,
                "negative_feedback": 2,
                "common_praise": ["fast response", "quality service"],
                "common_complaints": ["high cost", "communication delays"]
            }
        )
        
        return performance


class CommunicationHub:
    """Hub for managing partner communications"""
    
    async def establish_channels(
        self,
        partner: ExternalPartner,
        channels: List[str]
    ) -> Dict[str, Any]:
        """Establish communication channels with partner"""
        established_channels = []
        failed_channels = []
        
        for channel in channels:
            try:
                # Mock channel establishment
                if channel in ["email", "phone", "api"]:
                    established_channels.append({
                        'channel': channel,
                        'status': 'active',
                        'contact_info': partner.contact_info.get(f'primary_{channel}', 'unknown'),
                        'established_at': datetime.utcnow()
                    })
                else:
                    failed_channels.append({
                        'channel': channel,
                        'error': 'unsupported_channel'
                    })
                    
            except Exception as e:
                failed_channels.append({
                    'channel': channel,
                    'error': str(e)
                })
        
        return {
            'partner_id': partner.id,
            'established_channels': established_channels,
            'failed_channels': failed_channels,
            'total_channels': len(established_channels),
            'establishment_timestamp': datetime.utcnow()
        }