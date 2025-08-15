"""
Demo script for Crisis Detection and Assessment Engine

This script demonstrates the comprehensive crisis leadership capabilities
of ScrollIntel's Crisis Detection Engine.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

from scrollintel.engines.crisis_detection_engine import (
    CrisisDetectionEngine,
    Crisis,
    CrisisType,
    SeverityLevel,
    CrisisStatus,
    Signal
)


async def demo_crisis_detection_engine():
    """Demonstrate the Crisis Detection and Assessment Engine"""
    
    print("🚨 ScrollIntel Crisis Detection and Assessment Engine Demo")
    print("=" * 60)
    
    # Initialize the crisis detection engine
    engine = CrisisDetectionEngine()
    
    print("\n1. 🔍 EARLY WARNING SYSTEM DEMONSTRATION")
    print("-" * 40)
    
    # Demonstrate signal monitoring
    print("Monitoring system signals...")
    signals = await engine.early_warning_system.monitor_signals()
    print(f"✅ Detected {len(signals)} signals from monitoring systems")
    
    for signal in signals[:3]:  # Show first 3 signals
        print(f"   📊 {signal.source}: {signal.signal_type} = {signal.value} (confidence: {signal.confidence:.2f})")
    
    # Demonstrate potential crisis detection
    print("\nAnalyzing signals for potential crises...")
    potential_crises = await engine.early_warning_system.detect_potential_crises(signals)
    print(f"✅ Identified {len(potential_crises)} potential crisis situations")
    
    for potential in potential_crises:
        print(f"   ⚠️  {potential.crisis_type.value}: {potential.probability:.1%} probability")
        print(f"      Impact: {potential.predicted_impact}")
        if potential.time_to_crisis:
            print(f"      Time to crisis: {potential.time_to_crisis}")
    
    print("\n2. 🎯 CRISIS CLASSIFICATION DEMONSTRATION")
    print("-" * 40)
    
    # Create a test crisis for demonstration
    test_crisis = Crisis(
        id="demo_crisis_system_outage",
        crisis_type=CrisisType.SYSTEM_OUTAGE,
        severity_level=SeverityLevel.HIGH,
        start_time=datetime.now(),
        affected_areas=["production_systems", "api_endpoints", "customer_portal"],
        stakeholders_impacted=["customers", "engineering_team", "support_team", "executives"],
        current_status=CrisisStatus.DETECTED,
        signals=[
            Signal(
                source="system_monitor",
                signal_type="high_cpu_usage",
                value=95.0,
                timestamp=datetime.now(),
                confidence=0.9,
                metadata={"threshold": 90, "duration": "5min"}
            ),
            Signal(
                source="application_monitor",
                signal_type="high_error_rate",
                value=12.5,
                timestamp=datetime.now(),
                confidence=0.85,
                metadata={"threshold": 5, "window": "1hour"}
            )
        ]
    )
    
    # Classify the crisis
    classification = engine.crisis_classifier.classify_crisis(test_crisis)
    print(f"Crisis Classification Results:")
    print(f"   🏷️  Type: {classification.crisis_type.value}")
    print(f"   📈 Severity: {classification.severity_level.name} (Level {classification.severity_level.value})")
    print(f"   🎯 Confidence: {classification.confidence:.1%}")
    print(f"   📋 Sub-categories: {', '.join(classification.sub_categories)}")
    print(f"   🔗 Related crises: {', '.join(classification.related_crises)}")
    print(f"   💭 Rationale: {classification.classification_rationale}")
    
    print("\n3. 📊 IMPACT ASSESSMENT DEMONSTRATION")
    print("-" * 40)
    
    # Assess crisis impact
    impact = engine.impact_assessor.assess_impact(test_crisis)
    print("Impact Assessment Results:")
    
    print("\n   💰 Financial Impact:")
    for key, value in impact.financial_impact.items():
        if isinstance(value, (int, float)):
            print(f"      {key.replace('_', ' ').title()}: ${value:,.2f}")
    
    print("\n   ⚙️  Operational Impact:")
    for key, value in impact.operational_impact.items():
        print(f"      {key.replace('_', ' ').title()}: {value}")
    
    print("\n   🏢 Reputation Impact:")
    for key, value in impact.reputation_impact.items():
        print(f"      {key.replace('_', ' ').title()}: {value:.1f}/10")
    
    print("\n   👥 Stakeholder Impact:")
    for group, impacts in impact.stakeholder_impact.items():
        if impacts:
            print(f"      {group.title()}: {', '.join(impacts)}")
    
    print(f"\n   ⏱️  Recovery Estimate: {impact.recovery_estimate}")
    print(f"   🚨 Mitigation Urgency: {impact.mitigation_urgency.name}")
    
    print("\n   ⛓️  Cascading Risks:")
    for risk in impact.cascading_risks:
        print(f"      • {risk.replace('_', ' ').title()}")
    
    print("\n4. 📢 ESCALATION MANAGEMENT DEMONSTRATION")
    print("-" * 40)
    
    # Update crisis with classification and impact
    test_crisis.classification = classification
    test_crisis.impact_assessment = impact
    test_crisis.severity_level = classification.severity_level
    
    # Check escalation
    should_escalate = engine.escalation_manager.should_escalate(test_crisis)
    print(f"Escalation Required: {'✅ YES' if should_escalate else '❌ NO'}")
    
    if should_escalate:
        escalation_level = engine.escalation_manager._determine_escalation_level(test_crisis)
        print(f"Escalation Level: {escalation_level}")
        
        # Perform escalation
        escalation_result = await engine.escalation_manager.escalate_crisis(test_crisis)
        
        print("\nEscalation Results:")
        print(f"   📤 Escalated: {escalation_result['escalated']}")
        print(f"   📊 Level: {escalation_result['escalation_level']}")
        print(f"   📝 Reason: {escalation_result['escalation_reason']}")
        print(f"   📬 Notifications Sent: {len(escalation_result['notifications_sent'])}")
        
        for notification in escalation_result['notifications_sent']:
            print(f"      • {notification['recipient']} via {notification['channel']}")
    
    print("\n5. 🔄 COMPLETE CRISIS PROCESSING DEMONSTRATION")
    print("-" * 40)
    
    # Process the complete crisis workflow
    processed_crisis = await engine._process_crisis(test_crisis)
    engine.active_crises[processed_crisis.id] = processed_crisis
    
    print("Complete Crisis Processing Results:")
    print(f"   🆔 Crisis ID: {processed_crisis.id}")
    print(f"   📊 Status: {processed_crisis.current_status.value}")
    print(f"   🏷️  Final Type: {processed_crisis.crisis_type.value}")
    print(f"   📈 Final Severity: {processed_crisis.severity_level.name}")
    print(f"   🎯 Classification Confidence: {processed_crisis.classification.confidence:.1%}")
    print(f"   💰 Total Estimated Cost: ${processed_crisis.impact_assessment.financial_impact['total_estimated_cost']:,.2f}")
    
    print("\n6. 🎭 CRISIS SIMULATION DEMONSTRATION")
    print("-" * 40)
    
    # Simulate different types of crises
    crisis_scenarios = [
        {
            "type": CrisisType.SECURITY_BREACH,
            "signals": [
                Signal(
                    source="security_monitor",
                    signal_type="high_failed_logins",
                    value=500,
                    timestamp=datetime.now(),
                    confidence=0.9,
                    metadata={"window": "1hour", "threshold": 100}
                ),
                Signal(
                    source="network_monitor",
                    signal_type="network_anomaly",
                    value={"anomaly_type": "traffic_spike", "magnitude": 300},
                    timestamp=datetime.now(),
                    confidence=0.8,
                    metadata={"baseline_deviation": 300}
                )
            ]
        },
        {
            "type": CrisisType.FINANCIAL_CRISIS,
            "signals": [
                Signal(
                    source="financial_monitor",
                    signal_type="revenue_drop",
                    value=-25.0,
                    timestamp=datetime.now(),
                    confidence=0.95,
                    metadata={"threshold": -20, "period": "week"}
                ),
                Signal(
                    source="financial_monitor",
                    signal_type="low_cash_flow",
                    value=15,
                    timestamp=datetime.now(),
                    confidence=0.9,
                    metadata={"runway_days": 15}
                )
            ]
        }
    ]
    
    print("Simulating various crisis scenarios...")
    
    for i, scenario in enumerate(crisis_scenarios, 1):
        print(f"\n   Scenario {i}: {scenario['type'].value.replace('_', ' ').title()}")
        
        # Detect potential crises from scenario signals
        potential_crises = await engine.early_warning_system.detect_potential_crises(scenario['signals'])
        
        for potential in potential_crises:
            if potential.probability > 0.7:
                # Create and process crisis
                crisis = engine._create_crisis_from_potential(potential)
                processed = await engine._process_crisis(crisis)
                
                print(f"      ✅ Crisis Created: {processed.id}")
                print(f"      📊 Severity: {processed.severity_level.name}")
                print(f"      💰 Estimated Cost: ${processed.impact_assessment.financial_impact['total_estimated_cost']:,.2f}")
                print(f"      ⏱️  Recovery Time: {processed.impact_assessment.recovery_estimate}")
                
                # Add to active crises
                engine.active_crises[processed.id] = processed
    
    print("\n7. 📈 CRISIS METRICS AND ANALYTICS")
    print("-" * 40)
    
    # Get comprehensive metrics
    metrics = await engine.get_crisis_metrics()
    
    print("Crisis Management Metrics:")
    print(f"   🔴 Active Crises: {metrics['active_crises']}")
    print(f"   ✅ Total Resolved: {metrics['total_resolved']}")
    print(f"   ⏱️  Average Resolution Time: {metrics['average_resolution_time_seconds']:.1f} seconds")
    print(f"   📊 Escalation Rate: {metrics['escalation_rate']:.1%}")
    
    print("\n   Crisis Type Distribution:")
    for crisis_type, count in metrics['crisis_type_distribution'].items():
        print(f"      {crisis_type.replace('_', ' ').title()}: {count}")
    
    # Show active crises
    active_crises = await engine.get_active_crises()
    print(f"\n   📋 Active Crisis Details:")
    for crisis in active_crises:
        print(f"      • {crisis.id}: {crisis.crisis_type.value} ({crisis.severity_level.name})")
        print(f"        Status: {crisis.current_status.value}")
        print(f"        Duration: {datetime.now() - crisis.start_time}")
        print(f"        Stakeholders: {len(crisis.stakeholders_impacted)} groups affected")
    
    print("\n8. 🔧 CRISIS RESOLUTION DEMONSTRATION")
    print("-" * 40)
    
    # Resolve one of the active crises
    if active_crises:
        crisis_to_resolve = active_crises[0]
        print(f"Resolving crisis: {crisis_to_resolve.id}")
        
        success = await engine.resolve_crisis(crisis_to_resolve.id)
        
        if success:
            print("✅ Crisis successfully resolved!")
            
            # Show updated metrics
            updated_metrics = await engine.get_crisis_metrics()
            print(f"   Updated Active Crises: {updated_metrics['active_crises']}")
            print(f"   Updated Resolved Count: {updated_metrics['total_resolved']}")
        else:
            print("❌ Failed to resolve crisis")
    
    print("\n9. 🎯 CRISIS LEADERSHIP EXCELLENCE SUMMARY")
    print("-" * 40)
    
    print("Crisis Leadership Capabilities Demonstrated:")
    print("   ✅ Proactive Crisis Detection - Early warning system monitoring")
    print("   ✅ Intelligent Classification - Automated crisis type and severity assessment")
    print("   ✅ Comprehensive Impact Analysis - Financial, operational, and reputation impact")
    print("   ✅ Smart Escalation Management - Automated stakeholder notification")
    print("   ✅ Real-time Crisis Tracking - Active crisis monitoring and management")
    print("   ✅ Performance Analytics - Crisis response metrics and optimization")
    print("   ✅ Multi-crisis Handling - Simultaneous crisis management capability")
    print("   ✅ Recovery Planning - Systematic crisis resolution and learning")
    
    print("\n🏆 ScrollIntel Crisis Leadership Excellence Engine is ready to handle")
    print("   any crisis situation with superhuman composure and strategic thinking!")
    
    return {
        "demo_completed": True,
        "crises_processed": len(engine.active_crises) + len(engine.crisis_history),
        "active_crises": len(engine.active_crises),
        "resolved_crises": len(engine.crisis_history),
        "escalations_performed": len(engine.escalation_manager.escalation_history),
        "metrics": metrics
    }


async def demonstrate_crisis_scenarios():
    """Demonstrate specific crisis scenarios"""
    
    print("\n🎭 CRISIS SCENARIO DEMONSTRATIONS")
    print("=" * 50)
    
    engine = CrisisDetectionEngine()
    
    scenarios = [
        {
            "name": "Major System Outage",
            "description": "Critical production systems experiencing failures",
            "signals": [
                Signal("system_monitor", "high_cpu_usage", 98.0, datetime.now(), 0.95, {}),
                Signal("application_monitor", "high_error_rate", 25.0, datetime.now(), 0.9, {}),
                Signal("system_monitor", "high_memory_usage", 92.0, datetime.now(), 0.85, {})
            ]
        },
        {
            "name": "Security Breach",
            "description": "Suspected unauthorized access to systems",
            "signals": [
                Signal("security_monitor", "high_failed_logins", 1000, datetime.now(), 0.95, {}),
                Signal("network_monitor", "network_anomaly", {"type": "intrusion"}, datetime.now(), 0.8, {}),
                Signal("security_monitor", "suspicious_activity", "admin_access", datetime.now(), 0.9, {})
            ]
        },
        {
            "name": "Financial Crisis",
            "description": "Severe financial difficulties threatening operations",
            "signals": [
                Signal("financial_monitor", "revenue_drop", -40.0, datetime.now(), 0.95, {}),
                Signal("financial_monitor", "low_cash_flow", 10, datetime.now(), 0.9, {}),
                Signal("market_monitor", "high_volatility", 0.5, datetime.now(), 0.7, {})
            ]
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print("-" * 30)
        
        # Process scenario signals
        potential_crises = await engine.early_warning_system.detect_potential_crises(scenario['signals'])
        
        for potential in potential_crises:
            if potential.probability > 0.6:
                # Create and fully process crisis
                crisis = engine._create_crisis_from_potential(potential)
                processed_crisis = await engine._process_crisis(crisis)
                
                print(f"   🚨 Crisis Activated: {processed_crisis.id}")
                print(f"   📊 Type: {processed_crisis.crisis_type.value}")
                print(f"   📈 Severity: {processed_crisis.severity_level.name}")
                print(f"   🎯 Confidence: {processed_crisis.classification.confidence:.1%}")
                
                # Show key impact metrics
                if processed_crisis.impact_assessment:
                    financial_impact = processed_crisis.impact_assessment.financial_impact
                    print(f"   💰 Financial Impact: ${financial_impact.get('total_estimated_cost', 0):,.2f}")
                    print(f"   ⏱️  Recovery Estimate: {processed_crisis.impact_assessment.recovery_estimate}")
                    print(f"   🚨 Urgency: {processed_crisis.impact_assessment.mitigation_urgency.name}")
                
                # Show escalation status
                if processed_crisis.escalation_history:
                    latest_escalation = processed_crisis.escalation_history[-1]
                    print(f"   📢 Escalated to Level {latest_escalation['level']}")
                    print(f"   📬 Notifications: {len(latest_escalation['notifications'])}")
                
                print(f"   👥 Stakeholders Affected: {len(processed_crisis.stakeholders_impacted)}")
                print(f"   🏢 Areas Impacted: {', '.join(processed_crisis.affected_areas)}")


if __name__ == "__main__":
    async def main():
        # Run main demo
        demo_results = await demo_crisis_detection_engine()
        
        # Run scenario demonstrations
        await demonstrate_crisis_scenarios()
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"Results: {json.dumps(demo_results, indent=2, default=str)}")
    
    asyncio.run(main())