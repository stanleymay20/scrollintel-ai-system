"""
Demo for Enhanced Crisis Communication Integration

This demonstrates the seamless integration with all communication channels,
crisis communication context in all interactions, and crisis-aware
response generation and messaging.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

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
    additional_actions: List[str] = field(default_factory=list)

class EnhancedCrisisCommunicationIntegration:
    def __init__(self):
        self.active_crises: Dict[str, CrisisContext] = {}
        self.integrated_systems: Dict[str, CommunicationSystemIntegration] = {}
        self.response_enhancers: Dict[str, Callable] = {}
        
        # Initialize integrated systems
        self._initialize_system_integrations()
        self._setup_response_enhancers()
    
    def _initialize_system_integrations(self):
        """Initialize integrations with all ScrollIntel communication systems"""
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
            )
        })
    
    def _setup_response_enhancers(self):
        """Setup response enhancement functions for different communication types"""
        self.response_enhancers = {
            "chat": self._enhance_chat_response,
            "email": self._enhance_email_response,
            "customer": self._enhance_customer_response,
            "executive": self._enhance_executive_response
        }
    
    async def register_crisis(self, crisis_context: CrisisContext) -> bool:
        """Register crisis and propagate context to all integrated systems"""
        try:
            self.active_crises[crisis_context.crisis_id] = crisis_context
            
            # Propagate crisis context to all integrated systems
            await self._propagate_crisis_context_to_systems(crisis_context)
            
            print(f"✅ Crisis {crisis_context.crisis_id} registered and propagated to all systems")
            return True
        except Exception as e:
            print(f"❌ Failed to register crisis: {str(e)}")
            return False

    async def _propagate_crisis_context_to_systems(self, crisis_context: CrisisContext):
        """Propagate crisis context to all integrated communication systems"""
        print(f"📡 Propagating crisis context to {len(self.integrated_systems)} systems...")
        
        for system_name, system_config in self.integrated_systems.items():
            if system_config.is_active and system_config.crisis_context_support:
                print(f"  ✓ Notified {system_name} of crisis {crisis_context.crisis_id}")
    
    def generate_crisis_aware_response(
        self,
        query: str,
        response_type: str = "chat",
        user_context: Optional[Dict[str, Any]] = None
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
                confidence_score=0.85,
                response_type=response_type,
                escalation_recommended=escalation_needed,
                additional_actions=additional_actions
            )
            
        except Exception as e:
            print(f"❌ Failed to generate crisis-aware response: {str(e)}")
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
        
        # Return the highest severity crisis
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
            return "Thank you for your message. How can I assist you?"
    
    def _enhance_chat_response(
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
        crisis_footer += f"Crisis ID: {crisis_context.crisis_id}"
        
        return crisis_header + base_response + crisis_footer
    
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
        customer_message += f"Our team is working to resolve this quickly. Thank you for your patience."
        
        return base_response + customer_message
    
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
        executive_context += f"Business Impact: Significant disruption to operations\n"
        executive_context += f"Next Actions: Prepare stakeholder communications"
        
        return base_response + executive_context
    
    def _should_escalate_query(
        self, 
        query: str, 
        crisis_context: Optional[CrisisContext], 
        user_context: Optional[Dict[str, Any]]
    ) -> bool:
        """Determine if query should be escalated during crisis"""
        if not crisis_context:
            return False
        
        escalation_keywords = ["urgent", "emergency", "critical", "immediate", "help", "broken", "down", "failed"]
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
        
        return actions
    
    def _get_customer_friendly_description(self, crisis_context: CrisisContext) -> str:
        """Get customer-friendly description of the crisis"""
        descriptions = {
            "system_outage": "temporary service interruptions",
            "security_breach": "enhanced security measures",
            "performance_degradation": "slower than usual response times",
            "network_issue": "connectivity challenges"
        }
        return descriptions.get(crisis_context.crisis_type, "technical difficulties")
    
    async def integrate_with_all_communication_channels(self) -> Dict[str, Any]:
        """Integrate crisis context with all communication channels"""
        try:
            integration_results = {}
            
            for system_name, system_config in self.integrated_systems.items():
                if system_config.is_active:
                    result = await self._integrate_with_system(system_name, system_config)
                    integration_results[system_name] = result
            
            print("✅ Completed integration with all communication channels")
            return {
                "success": True,
                "integrated_systems": len(integration_results),
                "results": integration_results
            }
            
        except Exception as e:
            print(f"❌ Failed to integrate with communication channels: {str(e)}")
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
                print(f"  ✓ Enabled crisis-aware responses for {system_name}")
                
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


async def main():
    """Demonstrate the enhanced crisis communication integration"""
    print("🚀 Starting Enhanced Crisis Communication Integration Demo")
    print("=" * 60)
    
    # Initialize the integration engine
    integration = EnhancedCrisisCommunicationIntegration()
    
    # Show initial integration status
    print("\n📊 Initial Integration Status:")
    status = integration.get_integration_status()
    print(f"  Total Systems: {status['total_systems']}")
    print(f"  Crisis-Aware Systems: {status['crisis_aware_systems']}")
    print(f"  Active Crises: {status['active_crises']}")
    
    # Integrate with all communication channels
    print("\n🔗 Integrating with all communication channels...")
    integration_result = await integration.integrate_with_all_communication_channels()
    print(f"  Integration Success: {integration_result['success']}")
    print(f"  Systems Integrated: {integration_result['integrated_systems']}")
    
    # Create a sample crisis
    print("\n🚨 Registering a sample crisis...")
    crisis = CrisisContext(
        crisis_id="crisis_001",
        crisis_type="system_outage",
        severity_level=CrisisLevel.HIGH,
        start_time=datetime.now(),
        affected_systems=["web_app", "api", "database"],
        stakeholders=["customers", "employees", "executives"],
        status="active"
    )
    
    # Register the crisis
    await integration.register_crisis(crisis)
    
    # Test crisis-aware response generation for different scenarios
    print("\n💬 Testing Crisis-Aware Response Generation:")
    print("-" * 50)
    
    # Test chat response
    print("\n1. Chat Response:")
    chat_response = integration.generate_crisis_aware_response(
        query="What's the current system status?",
        response_type="chat"
    )
    print(f"Query: {chat_response.original_query}")
    print(f"Enhanced Response: {chat_response.crisis_enhanced_response}")
    print(f"Escalation Recommended: {chat_response.escalation_recommended}")
    print(f"Additional Actions: {chat_response.additional_actions}")
    
    # Test email response
    print("\n2. Email Response:")
    email_response = integration.generate_crisis_aware_response(
        query="Need help with the service",
        response_type="email"
    )
    print(f"Query: {email_response.original_query}")
    print(f"Enhanced Response: {email_response.crisis_enhanced_response}")
    
    # Test customer response
    print("\n3. Customer Response:")
    customer_response = integration.generate_crisis_aware_response(
        query="Is the service working?",
        response_type="customer",
        user_context={"stakeholder_type": "customer"}
    )
    print(f"Query: {customer_response.original_query}")
    print(f"Enhanced Response: {customer_response.crisis_enhanced_response}")
    
    # Test executive response
    print("\n4. Executive Response:")
    executive_response = integration.generate_crisis_aware_response(
        query="What's the business impact?",
        response_type="executive",
        user_context={"stakeholder_type": "executive"}
    )
    print(f"Query: {executive_response.original_query}")
    print(f"Enhanced Response: {executive_response.crisis_enhanced_response}")
    print(f"Escalation Recommended: {executive_response.escalation_recommended}")
    
    # Test escalation detection
    print("\n5. Escalation Test (Urgent Query):")
    urgent_response = integration.generate_crisis_aware_response(
        query="URGENT: System is completely down!",
        response_type="chat"
    )
    print(f"Query: {urgent_response.original_query}")
    print(f"Escalation Recommended: {urgent_response.escalation_recommended}")
    print(f"Additional Actions: {urgent_response.additional_actions}")
    
    # Show final status
    print("\n📊 Final Integration Status:")
    final_status = integration.get_integration_status()
    print(f"  Total Systems: {final_status['total_systems']}")
    print(f"  Crisis-Aware Systems: {final_status['crisis_aware_systems']}")
    print(f"  Active Crises: {final_status['active_crises']}")
    
    print("\n✅ Enhanced Crisis Communication Integration Demo Complete!")
    print("=" * 60)
    
    # Summary of capabilities demonstrated
    print("\n🎯 Capabilities Demonstrated:")
    print("  ✓ Seamless integration with all communication channels")
    print("  ✓ Crisis communication context in all interactions")
    print("  ✓ Crisis-aware response generation and messaging")
    print("  ✓ Context propagation across different system types")
    print("  ✓ Response enhancement based on user type and crisis severity")
    print("  ✓ Automatic escalation detection and recommendations")
    print("  ✓ Additional action suggestions based on crisis context")
    print("  ✓ Integration status monitoring and reporting")


if __name__ == "__main__":
    asyncio.run(main())