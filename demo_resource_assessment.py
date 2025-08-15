"""
Demo script for Resource Assessment System

Demonstrates the comprehensive resource assessment capabilities including
rapid inventory, capacity tracking, gap identification, and alternative sourcing.
"""

import asyncio
import json
from datetime import datetime, timedelta
from scrollintel.engines.resource_assessment_engine import ResourceAssessmentEngine
from scrollintel.models.resource_mobilization_models import (
    ResourceRequirement, ResourceType, ResourcePriority
)
from scrollintel.models.crisis_models_simple import Crisis, CrisisType, SeverityLevel


async def demo_resource_assessment():
    """Demonstrate comprehensive resource assessment capabilities"""
    print("🔍 ScrollIntel Resource Assessment System Demo")
    print("=" * 60)
    
    # Initialize the resource assessment engine
    engine = ResourceAssessmentEngine()
    
    # Create a sample crisis scenario
    crisis = Crisis(
        id="demo_crisis_001",
        crisis_type=CrisisType.SECURITY_BREACH,
        severity_level=SeverityLevel.CRITICAL,
        description="Critical security breach requiring immediate resource mobilization",
        affected_areas=["production_systems", "customer_data", "payment_processing"],
        stakeholders_impacted=["customers", "employees", "partners", "regulators"]
    )
    
    print(f"📋 Crisis Scenario: {crisis.description}")
    print(f"   Type: {crisis.crisis_type.value}")
    print(f"   Severity: {crisis.severity_level.value}")
    print(f"   Affected Areas: {', '.join(crisis.affected_areas)}")
    print()
    
    # Step 1: Perform comprehensive resource assessment
    print("🔍 Step 1: Performing Comprehensive Resource Assessment")
    print("-" * 50)
    
    inventory = await engine.assess_available_resources(crisis)
    
    print(f"✅ Assessment completed at: {inventory.assessment_time}")
    print(f"📊 Total resources in inventory: {inventory.total_resources}")
    print()
    
    print("📈 Resources by Type:")
    for resource_type, count in inventory.resources_by_type.items():
        print(f"   {resource_type.value}: {count} resources")
    print()
    
    print("⚡ Capacity Summary:")
    for resource_type, capacity in inventory.total_capacity.items():
        available = inventory.available_capacity.get(resource_type, 0)
        utilization = inventory.utilization_rates.get(resource_type, 0)
        print(f"   {resource_type.value}:")
        print(f"     Total Capacity: {capacity:.1f}")
        print(f"     Available: {available:.1f}")
        print(f"     Utilization: {utilization:.1f}%")
    print()
    
    print("🚨 Critical Shortages:")
    if inventory.critical_shortages:
        for shortage in inventory.critical_shortages:
            print(f"   {shortage.resource_type.value}: {shortage.gap_quantity:.1f} units short")
            print(f"     Impact: {shortage.impact_description}")
            print(f"     Estimated Cost: ${shortage.estimated_cost:,.2f}")
            print(f"     Time to Acquire: {shortage.time_to_acquire}")
    else:
        print("   No critical shortages identified")
    print()
    
    # Step 2: Track specific resource capacity
    print("📊 Step 2: Tracking Resource Capacity")
    print("-" * 50)
    
    # Get some resource IDs for capacity tracking
    sample_resource_ids = [r.id for r in inventory.available_resources[:3]]
    
    if sample_resource_ids:
        capacity_info = await engine.track_resource_capacity(sample_resource_ids)
        
        print("🔍 Resource Capacity Details:")
        for resource_id, info in capacity_info.items():
            if 'error' not in info:
                print(f"   Resource {resource_id}:")
                print(f"     Total Capacity: {info['total_capacity']:.1f}")
                print(f"     Current Utilization: {info['current_utilization']:.1f}")
                print(f"     Available: {info['available_capacity']:.1f}")
                print(f"     Utilization: {info['utilization_percentage']:.1f}%")
                print(f"     Status: {info['status']}")
            else:
                print(f"   Resource {resource_id}: {info['error']}")
        print()
    
    # Step 3: Define resource requirements for crisis response
    print("📋 Step 3: Defining Crisis Resource Requirements")
    print("-" * 50)
    
    requirements = [
        ResourceRequirement(
            crisis_id=crisis.id,
            resource_type=ResourceType.HUMAN_RESOURCES,
            required_capabilities=["incident_response", "security_analysis", "forensic_investigation"],
            quantity_needed=60.0,  # 60 person-hours
            priority=ResourcePriority.CRITICAL,
            duration_needed=timedelta(hours=24),
            justification="Critical incident response team for security breach containment",
            requested_by="crisis_management_system"
        ),
        ResourceRequirement(
            crisis_id=crisis.id,
            resource_type=ResourceType.TECHNICAL_INFRASTRUCTURE,
            required_capabilities=["high_availability", "isolation_capability", "backup_systems"],
            quantity_needed=800.0,  # 800 processing units
            priority=ResourcePriority.CRITICAL,
            duration_needed=timedelta(hours=12),
            justification="Infrastructure isolation and backup system activation",
            requested_by="crisis_management_system"
        ),
        ResourceRequirement(
            crisis_id=crisis.id,
            resource_type=ResourceType.EXTERNAL_SERVICES,
            required_capabilities=["forensic_analysis", "legal_consultation", "pr_management"],
            quantity_needed=15.0,  # 15 service hours
            priority=ResourcePriority.HIGH,
            duration_needed=timedelta(hours=48),
            justification="External expertise for forensic analysis and crisis communication",
            requested_by="crisis_management_system"
        ),
        ResourceRequirement(
            crisis_id=crisis.id,
            resource_type=ResourceType.CLOUD_COMPUTE,
            required_capabilities=["elastic_scaling", "geographic_distribution", "enhanced_security"],
            quantity_needed=5000.0,  # 5000 compute units
            priority=ResourcePriority.HIGH,
            duration_needed=timedelta(hours=72),
            justification="Additional compute capacity for system recovery and enhanced monitoring",
            requested_by="crisis_management_system"
        )
    ]
    
    print("📝 Resource Requirements:")
    for i, req in enumerate(requirements, 1):
        print(f"   {i}. {req.resource_type.value}")
        print(f"      Quantity: {req.quantity_needed}")
        print(f"      Priority: {req.priority.value}")
        print(f"      Duration: {req.duration_needed}")
        print(f"      Capabilities: {', '.join(req.required_capabilities)}")
        print(f"      Justification: {req.justification}")
    print()
    
    # Step 4: Identify resource gaps
    print("🔍 Step 4: Identifying Resource Gaps")
    print("-" * 50)
    
    gaps = await engine.identify_resource_gaps(requirements, inventory)
    
    if gaps:
        print(f"⚠️  Identified {len(gaps)} resource gaps:")
        total_gap_cost = 0
        
        for i, gap in enumerate(gaps, 1):
            print(f"   Gap {i}: {gap.resource_type.value}")
            print(f"     Shortage: {gap.gap_quantity:.1f} units")
            print(f"     Severity: {gap.severity.value}")
            print(f"     Missing Capabilities: {', '.join(gap.gap_capabilities) if gap.gap_capabilities else 'None'}")
            print(f"     Impact: {gap.impact_description}")
            print(f"     Estimated Cost: ${gap.estimated_cost:,.2f}")
            print(f"     Time to Acquire: {gap.time_to_acquire}")
            print(f"     Alternative Options: {', '.join(gap.alternative_options[:3])}...")
            print()
            total_gap_cost += gap.estimated_cost
        
        print(f"💰 Total Estimated Cost to Fill Gaps: ${total_gap_cost:,.2f}")
    else:
        print("✅ No resource gaps identified - all requirements can be met with current resources")
    print()
    
    # Step 5: Find alternative sources for gaps
    if gaps:
        print("🔄 Step 5: Finding Alternative Sources")
        print("-" * 50)
        
        for i, gap in enumerate(gaps[:2], 1):  # Show alternatives for first 2 gaps
            print(f"🔍 Alternatives for Gap {i} ({gap.resource_type.value}):")
            
            alternatives = await engine.find_alternative_sources(gap)
            
            for j, alt in enumerate(alternatives, 1):
                print(f"   Alternative {j}: {alt['type']}")
                print(f"     Description: {alt['description']}")
                print(f"     Estimated Cost: ${alt['estimated_cost']:,.2f}")
                print(f"     Implementation Time: {alt['time_to_implement']}")
                print(f"     Reliability: {alt['reliability']:.1%}")
                print(f"     Capacity Provided: {alt['capacity_provided']:.1f}")
                print()
    
    # Step 6: Generate assessment summary
    print("📊 Step 6: Assessment Summary")
    print("-" * 50)
    
    print("🎯 Crisis Response Readiness Assessment:")
    print(f"   Total Resources Available: {len(inventory.available_resources)}")
    print(f"   Resources Currently Allocated: {len(inventory.allocated_resources)}")
    print(f"   Resources Unavailable: {len(inventory.unavailable_resources)}")
    print(f"   Critical Shortages: {len(inventory.critical_shortages)}")
    print(f"   Resource Gaps Identified: {len(gaps)}")
    print()
    
    # Calculate readiness score
    total_requirements = len(requirements)
    gaps_count = len(gaps)
    critical_shortages = len(inventory.critical_shortages)
    
    readiness_score = max(0, 100 - (gaps_count * 15) - (critical_shortages * 10))
    
    print(f"🏆 Crisis Response Readiness Score: {readiness_score}/100")
    
    if readiness_score >= 90:
        print("   Status: ✅ EXCELLENT - Fully prepared for crisis response")
    elif readiness_score >= 75:
        print("   Status: ✅ GOOD - Well prepared with minor gaps")
    elif readiness_score >= 60:
        print("   Status: ⚠️  ADEQUATE - Prepared but with notable gaps")
    elif readiness_score >= 40:
        print("   Status: ⚠️  CONCERNING - Significant resource gaps exist")
    else:
        print("   Status: 🚨 CRITICAL - Major resource shortages, immediate action required")
    
    print()
    
    # Step 7: Recommendations
    print("💡 Step 7: Recommendations")
    print("-" * 50)
    
    recommendations = []
    
    if gaps:
        recommendations.append("🔄 Implement alternative sourcing for identified resource gaps")
        recommendations.append("📞 Activate external partner agreements for additional resources")
        recommendations.append("⚡ Consider emergency procurement for critical shortages")
    
    if inventory.critical_shortages:
        recommendations.append("🚨 Address critical resource shortages immediately")
        recommendations.append("📈 Increase resource capacity in shortage areas")
    
    if len(inventory.unavailable_resources) > len(inventory.available_resources) * 0.2:
        recommendations.append("🔧 Review and address resource availability issues")
        recommendations.append("📅 Optimize maintenance schedules to improve availability")
    
    if not recommendations:
        recommendations.append("✅ Resource assessment shows good crisis preparedness")
        recommendations.append("🔄 Continue regular resource assessments to maintain readiness")
        recommendations.append("📊 Monitor resource utilization trends for optimization opportunities")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    print()
    print("🎉 Resource Assessment Demo Completed Successfully!")
    print("=" * 60)


async def demo_capacity_monitoring():
    """Demonstrate real-time capacity monitoring"""
    print("\n🔍 Real-time Capacity Monitoring Demo")
    print("-" * 40)
    
    engine = ResourceAssessmentEngine()
    
    # Get all resources
    resources = await engine.resource_registry.get_all_resources()
    
    print("📊 Current Resource Capacity Status:")
    print()
    
    for resource in resources[:5]:  # Show first 5 resources
        capacity_info = await engine.track_resource_capacity([resource.id])
        info = capacity_info[resource.id]
        
        if 'error' not in info:
            utilization_bar = "█" * int(info['utilization_percentage'] / 10) + "░" * (10 - int(info['utilization_percentage'] / 10))
            
            print(f"🔧 {resource.name}")
            print(f"   Type: {resource.resource_type.value}")
            print(f"   Capacity: {info['total_capacity']:.1f} | Available: {info['available_capacity']:.1f}")
            print(f"   Utilization: [{utilization_bar}] {info['utilization_percentage']:.1f}%")
            print(f"   Status: {info['status']}")
            print()


if __name__ == "__main__":
    print("🚀 Starting ScrollIntel Resource Assessment System Demo")
    print()
    
    # Run the main demo
    asyncio.run(demo_resource_assessment())
    
    # Run capacity monitoring demo
    asyncio.run(demo_capacity_monitoring())
    
    print("\n✨ All demos completed successfully!")