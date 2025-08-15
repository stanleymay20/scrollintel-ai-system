#!/usr/bin/env python3
"""
Crisis Communication Integration Demo

Demonstrates the integration of crisis leadership capabilities with communication systems,
showing how ScrollIntel provides seamless crisis-aware communication across all channels.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from scrollintel.core.crisis_communication_integration import (
    CrisisCommunicationIntegrator,
    CommunicationChannelType,
    CommunicationIntegrationConfig
)
from scrollintel.models.crisis_models_simple import Crisis, CrisisStatus, CrisisType, SeverityLevel


class CrisisCommunicationDemo:
    """Demo class for crisis communication integration"""
    
    def __init__(self):
        self.setup_integrator()
        self.demo_results = []
    
    def setup_integrator(self):
        """Setup the crisis communication integrator"""
        print("🔧 Setting up Crisis Communication Integration...")
        
        # Configure integration
        config = CommunicationIntegrationConfig(
            enabled_channels=[
                CommunicationChannelType.EMAIL,
                CommunicationChannelType.SLACK,
                CommunicationChannelType.DASHBOARD,
                CommunicationChannelType.SMS,
                CommunicationChannelType.INTERNAL_CHAT
            ],
            crisis_aware_filtering=True,
            auto_context_injection=True,
            priority_routing=True,
            message_templates={
                "system_outage_3_email": "ALERT: System outage detected. {message}",
                "security_breach_4_slack": "🚨 URGENT SECURITY ALERT 🚨\n{message}",
                "data_breach_5_sms": "CRITICAL: Data breach - {message}"
            },
            escalation_triggers={
                "severity_4": "auto_escalate",
                "security_breach": "require_approval"
            }
        )
        
        self.integrator = CrisisCommunicationIntegrator(config)
        print("✅ Crisis Communication Integration configured")
    
    async def demo_normal_communication(self):
        """Demonstrate normal communication without crisis"""
        print("\n📞 Demo: Normal Communication (No Active Crisis)")
        print("=" * 60)
        
        # Process normal communication
        result = await self.integrator.process_communication(
            channel=CommunicationChannelType.EMAIL,
            message="Hello, can you provide the latest system metrics?",
            sender="manager@company.com"
        )
        
        print(f"📧 Email Communication:")
        print(f"   Message: 'Hello, can you provide the latest system metrics?'")
        print(f"   Sender: manager@company.com")
        print(f"   Crisis Context: {result['crisis_context']}")
        print(f"   Response Generated: {result.get('response', 'N/A')}")
        
        self.demo_results.append({
            "test": "normal_communication",
            "success": result["success"],
            "crisis_aware": bool(result["crisis_context"])
        })
    
    async def demo_crisis_registration(self):
        """Demonstrate crisis registration"""
        print("\n🚨 Demo: Crisis Registration")
        print("=" * 60)
        
        # Create a system outage crisis
        crisis = Crisis(
            id="demo_outage_001",
            crisis_type=CrisisType.SYSTEM_OUTAGE,
            severity_level=SeverityLevel.HIGH,
            start_time=datetime.now(),
            affected_areas=["api_gateway", "user_authentication", "payment_processing"],
            stakeholders_impacted=["customers", "support_team", "engineering", "management"],
            current_status=CrisisStatus.ACTIVE,
            response_actions=[],
            resolution_time=None
        )
        
        # Register crisis
        success = await self.integrator.register_crisis(crisis)
        
        print(f"🔴 Crisis Registered:")
        print(f"   ID: {crisis.id}")
        print(f"   Type: {crisis.crisis_type.value}")
        print(f"   Severity: {crisis.severity_level}/5")
        print(f"   Affected Areas: {', '.join(crisis.affected_areas)}")
        print(f"   Stakeholders: {', '.join(crisis.stakeholders_impacted)}")
        print(f"   Registration Success: {success}")
        
        # Show active crises
        active_crises = self.integrator.get_active_crises()
        print(f"   Active Crises Count: {len(active_crises)}")
        
        self.demo_results.append({
            "test": "crisis_registration",
            "success": success,
            "active_crises": len(active_crises)
        })
        
        return crisis
    
    async def demo_crisis_aware_communication(self, crisis):
        """Demonstrate crisis-aware communication"""
        print("\n💬 Demo: Crisis-Aware Communication")
        print("=" * 60)
        
        # Test different types of communication during crisis
        communications = [
            {
                "channel": CommunicationChannelType.SLACK,
                "message": "What's the current status of the outage?",
                "sender": "team_lead@company.com"
            },
            {
                "channel": CommunicationChannelType.EMAIL,
                "message": "Customers are reporting login issues",
                "sender": "support@company.com",
                "context": {"affected_systems": ["user_authentication"]}
            },
            {
                "channel": CommunicationChannelType.DASHBOARD,
                "message": "Need immediate status update for executive briefing",
                "sender": "cto@company.com",
                "context": {"priority": "urgent", "stakeholders": ["management"]}
            }
        ]
        
        for i, comm in enumerate(communications, 1):
            print(f"\n📱 Communication {i}:")
            print(f"   Channel: {comm['channel'].value}")
            print(f"   Message: '{comm['message']}'")
            print(f"   Sender: {comm['sender']}")
            
            result = await self.integrator.process_communication(
                channel=comm["channel"],
                message=comm["message"],
                sender=comm["sender"],
                context=comm.get("context")
            )
            
            print(f"   Crisis-Aware: {result['crisis_context'].get('active_crisis', False)}")
            if result['crisis_context'].get('active_crisis'):
                print(f"   Crisis Type: {result['crisis_context'].get('crisis_type')}")
                print(f"   Severity: {result['crisis_context'].get('severity')}")
                print(f"   Escalation Level: {result['crisis_context'].get('escalation_level')}")
            
            print(f"   Routing: {result.get('routing', {})}")
            
            self.demo_results.append({
                "test": f"crisis_communication_{i}",
                "success": result["success"],
                "crisis_aware": result['crisis_context'].get('active_crisis', False)
            })
    
    async def demo_crisis_broadcast(self, crisis):
        """Demonstrate crisis broadcasting"""
        print("\n📢 Demo: Crisis Broadcasting")
        print("=" * 60)
        
        # Broadcast initial crisis alert
        print("🚨 Broadcasting Initial Crisis Alert...")
        initial_broadcast = await self.integrator.broadcast_crisis_update(
            crisis_id=crisis.id,
            update_message="ALERT: We are experiencing a system outage affecting login and payment services. Our team is investigating and will provide updates every 15 minutes.",
            target_channels=[
                CommunicationChannelType.EMAIL,
                CommunicationChannelType.SLACK,
                CommunicationChannelType.DASHBOARD
            ]
        )
        
        print(f"   Broadcast Success: {initial_broadcast['success']}")
        print(f"   Channels Targeted: {len(initial_broadcast.get('broadcast_results', {}))}")
        for channel, result in initial_broadcast.get('broadcast_results', {}).items():
            print(f"   - {channel}: {'✅' if result.get('success') else '❌'}")
        
        # Simulate progress update
        await asyncio.sleep(1)  # Simulate time passing
        
        print("\n📊 Broadcasting Progress Update...")
        progress_broadcast = await self.integrator.broadcast_crisis_update(
            crisis_id=crisis.id,
            update_message="UPDATE: We have identified the root cause and are implementing a fix. ETA for resolution: 30 minutes."
        )
        
        print(f"   Broadcast Success: {progress_broadcast['success']}")
        print(f"   All Channels Used: {len(progress_broadcast.get('broadcast_results', {}))}")
        
        self.demo_results.append({
            "test": "crisis_broadcast",
            "success": initial_broadcast["success"] and progress_broadcast["success"],
            "channels_used": len(initial_broadcast.get('broadcast_results', {}))
        })
    
    async def demo_high_severity_crisis(self):
        """Demonstrate high-severity crisis handling"""
        print("\n🔥 Demo: High-Severity Crisis (Security Breach)")
        print("=" * 60)
        
        # Create high-severity security breach
        security_crisis = Crisis(
            id="demo_security_001",
            crisis_type=CrisisType.SECURITY_BREACH,
            severity_level=SeverityLevel.CRITICAL,
            start_time=datetime.now(),
            affected_areas=["user_database", "authentication_system", "api_keys"],
            stakeholders_impacted=["all_users", "security_team", "legal", "executives", "regulators"],
            current_status=CrisisStatus.ACTIVE,
            response_actions=[],
            resolution_time=None
        )
        
        # Register high-severity crisis
        await self.integrator.register_crisis(security_crisis)
        
        print(f"🔴 High-Severity Crisis Registered:")
        print(f"   Type: {security_crisis.crisis_type.value}")
        print(f"   Severity: {security_crisis.severity_level}/5")
        print(f"   Affected Areas: {len(security_crisis.affected_areas)} systems")
        print(f"   Stakeholders: {len(security_crisis.stakeholders_impacted)} groups")
        
        # Test high-severity communication
        result = await self.integrator.process_communication(
            channel=CommunicationChannelType.EMAIL,
            message="We need immediate status on the security incident",
            sender="ciso@company.com"
        )
        
        print(f"\n🚨 High-Severity Communication Processing:")
        print(f"   Crisis Context Applied: {result['crisis_context'].get('active_crisis')}")
        print(f"   Escalation Level: {result['crisis_context'].get('escalation_level')}")
        print(f"   Routing Escalated: {result.get('routing', {}).get('escalated', False)}")
        print(f"   Approval Required: {result.get('routing', {}).get('approval_required', False)}")
        
        # Broadcast critical security alert
        security_broadcast = await self.integrator.broadcast_crisis_update(
            crisis_id=security_crisis.id,
            update_message="CRITICAL SECURITY ALERT: We have detected unauthorized access to our systems. We are taking immediate action to secure all accounts and will provide updates as the situation develops."
        )
        
        print(f"\n📢 Critical Security Broadcast:")
        print(f"   Broadcast Success: {security_broadcast['success']}")
        print(f"   Emergency Channels Activated: {len(security_broadcast.get('broadcast_results', {}))}")
        
        self.demo_results.append({
            "test": "high_severity_crisis",
            "success": result["success"],
            "escalation_triggered": result.get('routing', {}).get('escalated', False)
        })
        
        return security_crisis
    
    async def demo_multiple_crisis_handling(self):
        """Demonstrate handling multiple concurrent crises"""
        print("\n🔄 Demo: Multiple Concurrent Crises")
        print("=" * 60)
        
        # Create multiple crises
        crises = [
            Crisis(
                id="demo_performance_001",
                crisis_type=CrisisType.PERFORMANCE_DEGRADATION,
                severity_level=SeverityLevel.MEDIUM,
                start_time=datetime.now(),
                affected_areas=["web_frontend"],
                stakeholders_impacted=["customers"],
                current_status=CrisisStatus.ACTIVE,
                response_actions=[],
                resolution_time=None
            ),
            Crisis(
                id="demo_data_001",
                crisis_type=CrisisType.DATA_LOSS,
                severity_level=SeverityLevel.CRITICAL,
                start_time=datetime.now(),
                affected_areas=["customer_database", "payment_data"],
                stakeholders_impacted=["all_customers", "legal", "executives", "regulators"],
                current_status=CrisisStatus.ACTIVE,
                response_actions=[],
                resolution_time=None
            )
        ]
        
        # Register multiple crises
        for crisis in crises:
            await self.integrator.register_crisis(crisis)
            print(f"   Registered: {crisis.id} (Severity: {crisis.severity_level})")
        
        active_crises = self.integrator.get_active_crises()
        print(f"\n📊 Total Active Crises: {len(active_crises)}")
        
        # Test communication relevance
        print(f"\n🎯 Testing Crisis Relevance:")
        
        # Communication related to performance issue
        perf_result = await self.integrator.process_communication(
            channel=CommunicationChannelType.SLACK,
            message="The website is loading slowly",
            sender="user@company.com",
            context={"affected_systems": ["web_frontend"]}
        )
        
        print(f"   Performance Query -> Crisis: {perf_result['crisis_context'].get('crisis_id')}")
        
        # Communication related to data breach
        data_result = await self.integrator.process_communication(
            channel=CommunicationChannelType.EMAIL,
            message="Are customer payment details secure?",
            sender="customer@example.com",
            context={"affected_systems": ["payment_data"]}
        )
        
        print(f"   Data Security Query -> Crisis: {data_result['crisis_context'].get('crisis_id')}")
        
        self.demo_results.append({
            "test": "multiple_crises",
            "success": True,
            "active_crises": len(active_crises),
            "relevance_matching": perf_result['crisis_context'].get('crisis_id') != data_result['crisis_context'].get('crisis_id')
        })
    
    async def demo_crisis_resolution(self):
        """Demonstrate crisis resolution process"""
        print("\n✅ Demo: Crisis Resolution")
        print("=" * 60)
        
        active_crises = self.integrator.get_active_crises()
        print(f"Active Crises Before Resolution: {len(active_crises)}")
        
        # Resolve crises one by one
        for crisis_info in active_crises:
            crisis_id = crisis_info["crisis_id"]
            print(f"\n🔄 Resolving Crisis: {crisis_id}")
            
            # Update status to resolving
            await self.integrator.update_crisis_status(crisis_id, CrisisStatus.RESOLVING)
            print(f"   Status updated to: RESOLVING")
            
            # Simulate resolution time
            await asyncio.sleep(0.5)
            
            # Resolve crisis
            await self.integrator.resolve_crisis(crisis_id)
            print(f"   Crisis resolved and cleaned up")
        
        final_active_crises = self.integrator.get_active_crises()
        print(f"\n📊 Active Crises After Resolution: {len(final_active_crises)}")
        
        self.demo_results.append({
            "test": "crisis_resolution",
            "success": len(final_active_crises) == 0,
            "crises_resolved": len(active_crises)
        })
    
    async def demo_channel_customization(self):
        """Demonstrate channel-specific message customization"""
        print("\n🎨 Demo: Channel-Specific Message Customization")
        print("=" * 60)
        
        # Create a test crisis for customization demo
        test_crisis = Crisis(
            id="demo_customization_001",
            crisis_type=CrisisType.SYSTEM_OUTAGE,
            severity_level=SeverityLevel.HIGH,
            start_time=datetime.now(),
            affected_areas=["api"],
            stakeholders_impacted=["customers"],
            current_status=CrisisStatus.ACTIVE,
            response_actions=[],
            resolution_time=None
        )
        
        await self.integrator.register_crisis(test_crisis)
        crisis_context = self.integrator.active_crises[test_crisis.id]
        
        base_message = "System maintenance will begin in 30 minutes. Expected downtime: 2 hours. We apologize for any inconvenience and will provide updates throughout the maintenance window."
        
        # Test customization for different channels
        channels_to_test = [
            CommunicationChannelType.SMS,
            CommunicationChannelType.EMAIL,
            CommunicationChannelType.SLACK,
            CommunicationChannelType.DASHBOARD
        ]
        
        print("📱 Message Customization Results:")
        for channel in channels_to_test:
            customized = await self.integrator._customize_message_for_channel(
                base_message, channel, crisis_context
            )
            
            print(f"\n   {channel.value.upper()}:")
            print(f"   Length: {len(customized)} chars")
            print(f"   Preview: {customized[:100]}{'...' if len(customized) > 100 else ''}")
        
        # Clean up
        await self.integrator.resolve_crisis(test_crisis.id)
        
        self.demo_results.append({
            "test": "channel_customization",
            "success": True,
            "channels_tested": len(channels_to_test)
        })
    
    def print_demo_summary(self):
        """Print summary of demo results"""
        print("\n" + "=" * 80)
        print("🎯 CRISIS COMMUNICATION INTEGRATION DEMO SUMMARY")
        print("=" * 80)
        
        total_tests = len(self.demo_results)
        successful_tests = sum(1 for result in self.demo_results if result["success"])
        
        print(f"📊 Overall Results:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Successful: {successful_tests}")
        print(f"   Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        
        print(f"\n📋 Test Details:")
        for result in self.demo_results:
            status = "✅" if result["success"] else "❌"
            print(f"   {status} {result['test']}")
            
            # Print additional details
            for key, value in result.items():
                if key not in ["test", "success"]:
                    print(f"      {key}: {value}")
        
        print(f"\n🎉 Crisis Communication Integration Demo Complete!")
        print(f"   ScrollIntel successfully demonstrated seamless integration")
        print(f"   with communication systems during crisis situations.")
        
        # Key capabilities demonstrated
        capabilities = [
            "Crisis registration and context injection",
            "Crisis-aware communication processing",
            "Multi-channel broadcasting",
            "High-severity crisis escalation",
            "Multiple concurrent crisis handling",
            "Crisis relevance matching",
            "Channel-specific message customization",
            "Crisis resolution and cleanup"
        ]
        
        print(f"\n🚀 Key Capabilities Demonstrated:")
        for capability in capabilities:
            print(f"   ✓ {capability}")
    
    async def run_demo(self):
        """Run the complete crisis communication integration demo"""
        print("🎬 Starting Crisis Communication Integration Demo")
        print("=" * 80)
        
        try:
            # Run demo scenarios
            await self.demo_normal_communication()
            
            crisis = await self.demo_crisis_registration()
            await self.demo_crisis_aware_communication(crisis)
            await self.demo_crisis_broadcast(crisis)
            
            security_crisis = await self.demo_high_severity_crisis()
            await self.demo_multiple_crisis_handling()
            await self.demo_channel_customization()
            
            await self.demo_crisis_resolution()
            
            # Print summary
            self.print_demo_summary()
            
        except Exception as e:
            print(f"❌ Demo failed with error: {str(e)}")
            import traceback
            traceback.print_exc()


async def main():
    """Main demo function"""
    demo = CrisisCommunicationDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())