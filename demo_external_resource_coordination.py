"""
Demo script for External Resource Coordination System

Demonstrates comprehensive external partner coordination, resource request management,
and partnership activation protocols for crisis response.
"""

import asyncio
import json
from datetime import datetime, timedelta
from scrollintel.engines.external_resource_coordinator import (
    ExternalResourceCoordinator, ActivationLevel
)
from scrollintel.models.resource_mobilization_models import (
    ResourceRequirement, ResourceType, ResourcePriority
)
from scrollintel.models.crisis_models_simple import Crisis, CrisisType, SeverityLevel


async def demo_external_resource_coordination():
    """Demonstrate comprehensive external resource coordination capabilities"""
    print("🌐 ScrollIntel External Resource Coordination System Demo")
    print("=" * 70)
    
    # Initialize the external resource coordinator
    coordinator = ExternalResourceCoordinator()
    
    # Create a sample crisis scenario
    crisis = Crisis(
        id="demo_crisis_ext_001",
        crisis_type=CrisisType.CYBER_ATTACK,
        severity_level=SeverityLevel.CRITICAL,
        description="Critical cyber attack requiring immediate external expertise and resources",
        affected_areas=["network_infrastructure", "customer_data", "payment_systems", "internal_systems"],
        stakeholders_impacted=["customers", "employees", "partners", "regulators", "media"]
    )
    
    print(f"🚨 Crisis Scenario: {crisis.description}")
    print(f"   Type: {crisis.crisis_type.value}")
    print(f"   Severity: {crisis.severity_level.value}")
    print(f"   Affected Areas: {', '.join(crisis.affected_areas)}")
    print(f"   Stakeholders: {', '.join(crisis.stakeholders_impacted)}")
    print()
    
    # Step 1: Define external resource requirements
    print("📋 Step 1: Defining External Resource Requirements")
    print("-" * 50)
    
    requirements = [
        ResourceRequirement(
            crisis_id=crisis.id,
            resource_type=ResourceType.EXTERNAL_SERVICES,
            required_capabilities=["forensic_analysis", "malware_analysis", "incident_response", "threat_intelligence"],
            quantity_needed=8.0,  # 8 service units
            priority=ResourcePriority.EMERGENCY,
            duration_needed=timedelta(hours=72),
            budget_limit=150000.0,
            justification="Critical forensic analysis and threat intelligence for cyber attack response",
            requested_by="crisis_management_system"
        ),
        ResourceRequirement(
            crisis_id=crisis.id,
            resource_type=ResourceType.HUMAN_RESOURCES,
            required_capabilities=["cybersecurity_expert", "incident_commander", "legal_counsel", "pr_specialist"],
            quantity_needed=12.0,  # 12 person-hours
            priority=ResourcePriority.CRITICAL,
            duration_needed=timedelta(hours=48),
            budget_limit=75000.0,
            justification="Expert cybersecurity consultants and crisis management specialists",
            requested_by="crisis_management_system"
        ),
        ResourceRequirement(
            crisis_id=crisis.id,
            resource_type=ResourceType.CLOUD_COMPUTE,
            required_capabilities=["isolated_environment", "enhanced_security", "forensic_tools", "monitoring_systems"],
            quantity_needed=5000.0,  # 5000 compute units
            priority=ResourcePriority.HIGH,
            duration_needed=timedelta(hours=168),  # 1 week
            budget_limit=25000.0,
            justification="Isolated cloud environment for forensic analysis and system recovery",
            requested_by="crisis_management_system"
        ),
        ResourceRequirement(
            crisis_id=crisis.id,
            resource_type=ResourceType.EQUIPMENT_HARDWARE,
            required_capabilities=["forensic_workstations", "network_analyzers", "secure_storage", "communication_equipment"],
            quantity_needed=6.0,  # 6 equipment units
            priority=ResourcePriority.HIGH,
            duration_needed=timedelta(hours=120),  # 5 days
            budget_limit=40000.0,
            justification="Specialized forensic equipment for on-site investigation",
            requested_by="crisis_management_system"
        )
    ]
    
    print("📝 External Resource Requirements:")
    for i, req in enumerate(requirements, 1):
        print(f"   {i}. {req.resource_type.value}")
        print(f"      Quantity: {req.quantity_needed}")
        print(f"      Priority: {req.priority.value}")
        print(f"      Duration: {req.duration_needed}")
        print(f"      Budget: ${req.budget_limit:,.2f}")
        print(f"      Capabilities: {', '.join(req.required_capabilities)}")
        print(f"      Justification: {req.justification}")
        print()
    
    # Step 2: Coordinate with external partners
    print("🤝 Step 2: Coordinating with External Partners")
    print("-" * 50)
    
    coordination_result = await coordinator.coordinate_with_partners(crisis, requirements)
    
    print(f"✅ Coordination initiated for crisis {crisis.id}")
    print(f"📊 Coordination Summary:")
    print(f"   Partners Contacted: {coordination_result['total_partners_contacted']}")
    print(f"   Requests Submitted: {coordination_result['total_requests_submitted']}")
    print(f"   Estimated Response Time: {coordination_result['estimated_response_time']}")
    print(f"   Coordination Status: {coordination_result['coordination_status']}")
    print()
    
    # Display partner activation results
    activation_results = coordination_result['activation_results']
    print("🔄 Partner Activation Results:")
    print(f"   Successful Activations: {len(activation_results['successful_activations'])}")
    print(f"   Failed Activations: {len(activation_results['failed_activations'])}")
    print(f"   Total Partners Activated: {activation_results['total_activated']}")
    print()
    
    if activation_results['successful_activations']:
        print("✅ Successfully Activated Partners:")
        for activation in activation_results['successful_activations']:
            print(f"   • {activation['partner_name']}")
            print(f"     Activation Level: {activation['activation_level']}")
            print(f"     Response Time: {activation['estimated_response_time']}")
            print(f"     Communication Channels: {len(activation['communication_setup']['established_channels'])}")
            print()
    
    # Display resource request results
    request_results = coordination_result['request_results']
    print("📤 Resource Request Results:")
    print(f"   Requests Submitted: {request_results['total_submitted']}")
    print(f"   Failed Submissions: {len(request_results['failed_submissions'])}")
    print()
    
    if request_results['submitted_requests']:
        print("📋 Submitted Requests:")
        for request in request_results['submitted_requests']:
            print(f"   • Request {request.id}")
            print(f"     Partner: {request.partner_id}")
            print(f"     Resource Type: {request.resource_type.value}")
            print(f"     Quantity: {request.quantity_requested}")
            print(f"     Urgency: {request.urgency_level.value}")
            print(f"     Status: {request.request_status}")
            print()
    
    # Step 3: Demonstrate individual partner activation
    print("🎯 Step 3: Individual Partner Activation Demo")
    print("-" * 50)
    
    # Get available partners
    partners = await coordinator.partner_registry.get_all_partners()
    
    if partners:
        demo_partner = partners[0]  # Use first partner for demo
        
        print(f"🔧 Activating Partnership: {demo_partner.name}")
        print(f"   Partner Type: {demo_partner.partner_type}")
        print(f"   Available Resources: {[rt.value for rt in demo_partner.available_resources]}")
        print(f"   Service Capabilities: {', '.join(demo_partner.service_capabilities[:3])}...")
        print(f"   Reliability Score: {demo_partner.reliability_score:.1%}")
        print()
        
        # Activate at emergency response level
        activation_result = await coordinator.activate_partnership_protocols(
            demo_partner.id,
            ActivationLevel.EMERGENCY_RESPONSE,
            {
                "crisis_id": crisis.id,
                "activation_reason": "cyber_attack_response",
                "urgency": "critical",
                "expected_duration": "72_hours"
            }
        )
        
        print("⚡ Emergency Activation Results:")
        print(f"   Activation Successful: {'✅' if activation_result['activation_successful'] else '❌'}")
        print(f"   Activation Level: {activation_result['activation_level']}")
        print(f"   Response Time: {activation_result['estimated_response_time']}")
        print()
        
        print("📋 Activation Steps Completed:")
        for step in activation_result['activation_steps']:
            status_icon = "✅" if step['status'] == 'completed' else "⏳"
            print(f"   {status_icon} {step['step']}: {step['description']}")
        print()
        
        print("📞 Communication Channels Established:")
        comm_setup = activation_result['communication_setup']
        for channel in comm_setup['established_channels']:
            print(f"   • {channel['channel'].upper()}: {channel['status']} ({channel['contact_info']})")
        print()
        
        print("🎯 Next Actions:")
        for action in activation_result['next_actions']:
            print(f"   • {action}")
        print()
    
    # Step 4: Resource request management
    print("📊 Step 4: Resource Request Management")
    print("-" * 50)
    
    submitted_requests = request_results['submitted_requests']
    
    if submitted_requests:
        print(f"🔍 Managing {len(submitted_requests)} active requests...")
        
        management_result = await coordinator.manage_resource_requests(submitted_requests)
        
        print("📈 Management Summary:")
        print(f"   Total Requests Managed: {management_result['total_requests']}")
        print(f"   Successful Requests: {len(management_result['successful_requests'])}")
        print(f"   Failed Requests: {len(management_result['failed_requests'])}")
        print(f"   Pending Requests: {len(management_result['pending_requests'])}")
        print()
        
        print("📊 Status Breakdown:")
        for status, count in management_result['request_status_summary'].items():
            print(f"   {status}: {count} requests")
        print()
        
        print("🔧 Management Actions Taken:")
        for action in management_result['management_actions'][:3]:  # Show first 3
            print(f"   • Request {action['request_id']}")
            print(f"     Status: {action['current_status']}")
            print(f"     Action: {action['action_taken']}")
            print(f"     Success: {'✅' if action['action_successful'] else '❌'}")
        print()
        
        print("📋 Next Steps:")
        for step in management_result['next_steps']:
            print(f"   • {step}")
        print()
    
    # Step 5: Partner performance monitoring
    print("📊 Step 5: Partner Performance Monitoring")
    print("-" * 50)
    
    # Monitor performance for all partners
    partner_ids = [p.id for p in partners]
    performance_data = await coordinator.monitor_partner_performance(
        partner_ids, timedelta(days=30)
    )
    
    print(f"📈 Performance Analysis (30-day window):")
    print(f"   Partners Monitored: {len(performance_data)}")
    print()
    
    for partner_id, performance in performance_data.items():
        partner = next((p for p in partners if p.id == partner_id), None)
        partner_name = partner.name if partner else f"Partner {partner_id}"
        
        print(f"🏢 {partner_name}")
        print(f"   Total Requests: {performance.total_requests}")
        print(f"   Success Rate: {(performance.successful_requests / performance.total_requests * 100) if performance.total_requests > 0 else 0:.1f}%")
        print(f"   Reliability Score: {performance.reliability_score:.1%}")
        print(f"   Quality Score: {performance.quality_score:.1%}")
        print(f"   Cost Efficiency: {performance.cost_efficiency:.1%}")
        print(f"   Avg Response Time: {performance.average_response_time}")
        print(f"   Last Engagement: {performance.last_engagement.strftime('%Y-%m-%d %H:%M') if performance.last_engagement else 'Never'}")
        
        # Performance rating
        avg_score = (performance.reliability_score + performance.quality_score + performance.cost_efficiency) / 3
        if avg_score >= 0.9:
            rating = "🌟 Excellent"
        elif avg_score >= 0.8:
            rating = "✅ Good"
        elif avg_score >= 0.7:
            rating = "⚠️ Satisfactory"
        else:
            rating = "❌ Needs Improvement"
        
        print(f"   Overall Rating: {rating}")
        print()
    
    # Step 6: Coordination history and audit trail
    print("📚 Step 6: Coordination History & Audit Trail")
    print("-" * 50)
    
    print(f"📋 Coordination Events: {len(coordinator.coordination_history)}")
    print()
    
    # Show recent events
    recent_events = sorted(coordinator.coordination_history, key=lambda x: x.timestamp, reverse=True)[:5]
    
    print("🕒 Recent Coordination Events:")
    for event in recent_events:
        print(f"   • {event.timestamp.strftime('%H:%M:%S')} - {event.event_type}")
        print(f"     Partner: {event.partner_id}")
        print(f"     Description: {event.event_description}")
        print(f"     Initiated by: {event.initiated_by}")
        print()
    
    # Step 7: Generate comprehensive coordination report
    print("📊 Step 7: Comprehensive Coordination Report")
    print("-" * 50)
    
    # Calculate overall coordination metrics
    total_partners = len(partners)
    activated_partners = activation_results['total_activated']
    total_requests = len(submitted_requests) if submitted_requests else 0
    successful_requests = len(management_result.get('successful_requests', [])) if submitted_requests else 0
    
    coordination_efficiency = (activated_partners / total_partners * 100) if total_partners > 0 else 0
    request_success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
    
    avg_partner_reliability = sum(p.reliability_score for p in partners) / len(partners) if partners else 0
    avg_response_time = sum(p.response_time_sla.total_seconds() / 3600 for p in partners) / len(partners) if partners else 0
    
    print("🎯 Crisis Response Coordination Assessment:")
    print(f"   Total External Partners: {total_partners}")
    print(f"   Partners Successfully Activated: {activated_partners}")
    print(f"   Coordination Efficiency: {coordination_efficiency:.1f}%")
    print(f"   Resource Requests Submitted: {total_requests}")
    print(f"   Request Success Rate: {request_success_rate:.1f}%")
    print(f"   Average Partner Reliability: {avg_partner_reliability:.1%}")
    print(f"   Average Response Time: {avg_response_time:.1f} hours")
    print()
    
    # Calculate overall readiness score
    readiness_factors = [
        coordination_efficiency / 100,
        request_success_rate / 100,
        avg_partner_reliability,
        min(1.0, 24 / avg_response_time) if avg_response_time > 0 else 0  # Prefer faster response
    ]
    
    overall_readiness = sum(readiness_factors) / len(readiness_factors) * 100
    
    print(f"🏆 External Resource Coordination Readiness: {overall_readiness:.1f}/100")
    
    if overall_readiness >= 90:
        print("   Status: 🌟 EXCELLENT - Outstanding external resource coordination capability")
    elif overall_readiness >= 80:
        print("   Status: ✅ VERY GOOD - Strong external resource coordination")
    elif overall_readiness >= 70:
        print("   Status: ✅ GOOD - Adequate external resource coordination")
    elif overall_readiness >= 60:
        print("   Status: ⚠️ FAIR - External resource coordination needs improvement")
    else:
        print("   Status: ❌ POOR - Critical external resource coordination gaps")
    
    print()
    
    # Step 8: Strategic recommendations
    print("💡 Step 8: Strategic Recommendations")
    print("-" * 50)
    
    recommendations = []
    
    if coordination_efficiency < 80:
        recommendations.append("🔄 Expand external partner network for better coverage")
        recommendations.append("📞 Improve partner activation protocols and response times")
    
    if request_success_rate < 90:
        recommendations.append("📋 Review and optimize resource request processes")
        recommendations.append("🤝 Strengthen partner relationships and service agreements")
    
    if avg_partner_reliability < 0.85:
        recommendations.append("⭐ Focus on higher-reliability partners for critical situations")
        recommendations.append("📊 Implement partner performance improvement programs")
    
    if avg_response_time > 4:
        recommendations.append("⚡ Negotiate faster response time SLAs with key partners")
        recommendations.append("🚨 Establish emergency response protocols for critical situations")
    
    if not recommendations:
        recommendations.extend([
            "✅ Maintain current high-performance external coordination capabilities",
            "🔄 Continue regular partner relationship management and performance monitoring",
            "📈 Explore opportunities for strategic partnership expansion",
            "🎯 Conduct regular crisis simulation exercises with external partners"
        ])
    
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    print()
    print("🎉 External Resource Coordination Demo Completed Successfully!")
    print("=" * 70)


async def demo_partner_activation_scenarios():
    """Demonstrate different partner activation scenarios"""
    print("\n🎯 Partner Activation Scenarios Demo")
    print("-" * 45)
    
    coordinator = ExternalResourceCoordinator()
    partners = await coordinator.partner_registry.get_all_partners()
    
    if not partners:
        print("No partners available for activation demo")
        return
    
    # Test different activation levels
    activation_scenarios = [
        {
            "level": ActivationLevel.ALERT,
            "scenario": "Potential threat detected - partners on alert",
            "context": {"threat_level": "medium", "estimated_impact": "low"}
        },
        {
            "level": ActivationLevel.ACTIVATED,
            "scenario": "Active incident - partners activated for response",
            "context": {"incident_type": "security_breach", "estimated_impact": "medium"}
        },
        {
            "level": ActivationLevel.FULL_DEPLOYMENT,
            "scenario": "Major crisis - full partner deployment required",
            "context": {"crisis_type": "system_outage", "estimated_impact": "high"}
        },
        {
            "level": ActivationLevel.EMERGENCY_RESPONSE,
            "scenario": "Critical emergency - immediate partner response needed",
            "context": {"emergency_type": "cyber_attack", "estimated_impact": "critical"}
        }
    ]
    
    for i, scenario in enumerate(activation_scenarios, 1):
        print(f"🔧 Scenario {i}: {scenario['scenario']}")
        print(f"   Activation Level: {scenario['level'].value}")
        
        # Use different partner for each scenario
        partner = partners[i % len(partners)]
        
        try:
            result = await coordinator.activate_partnership_protocols(
                partner.id, scenario['level'], scenario['context']
            )
            
            success_icon = "✅" if result['activation_successful'] else "❌"
            print(f"   Result: {success_icon} {partner.name}")
            print(f"   Response Time: {result['estimated_response_time']}")
            print(f"   Steps Completed: {len(result['activation_steps'])}")
            
        except Exception as e:
            print(f"   Result: ❌ Activation failed - {str(e)}")
        
        print()


if __name__ == "__main__":
    print("🚀 Starting ScrollIntel External Resource Coordination Demo")
    print()
    
    # Run the main demo
    asyncio.run(demo_external_resource_coordination())
    
    # Run partner activation scenarios demo
    asyncio.run(demo_partner_activation_scenarios())
    
    print("\n✨ All external resource coordination demos completed successfully!")