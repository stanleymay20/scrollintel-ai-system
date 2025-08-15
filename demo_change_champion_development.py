"""
Change Champion Development Demo

Demonstrates the change champion identification, development, and network management capabilities.
"""

import asyncio
import json
from datetime import datetime, timedelta
from scrollintel.engines.change_champion_development_engine import ChangeChampionDevelopmentEngine
from scrollintel.models.change_champion_models import (
    ChangeCapability, ChampionLevel, ChampionRole, NetworkStatus
)


async def demo_change_champion_development():
    """Demonstrate change champion development system"""
    print("🚀 Change Champion Development Demo")
    print("=" * 50)
    
    # Initialize champion development engine
    champion_engine = ChangeChampionDevelopmentEngine()
    
    # Demo 1: Identify Potential Change Champions
    print("\n1. Identifying Potential Change Champions")
    print("-" * 42)
    
    # Sample employee data from organization
    employee_data = [
        {
            "id": "emp_001",
            "name": "Sarah Johnson",
            "role": "Senior Manager",
            "department": "Operations",
            "organization_id": "techcorp_inc",
            "capabilities": {
                "change_advocacy": 88,
                "influence_building": 82,
                "communication": 92,
                "training_delivery": 75,
                "resistance_management": 70,
                "network_building": 85,
                "feedback_collection": 78,
                "coaching_mentoring": 80,
                "project_coordination": 73,
                "cultural_sensitivity": 87
            },
            "network_size": 28,
            "cross_department_connections": 5,
            "credibility_score": 90,
            "cultural_alignment_score": 88,
            "experience": [
                "Led digital transformation initiative",
                "Mentored 8 junior staff members",
                "Facilitated cross-department workshops",
                "Managed change resistance successfully"
            ]
        },
        {
            "id": "emp_002",
            "name": "Michael Chen",
            "role": "Team Lead",
            "department": "IT",
            "organization_id": "techcorp_inc",
            "capabilities": {
                "change_advocacy": 75,
                "influence_building": 68,
                "communication": 80,
                "training_delivery": 65,
                "resistance_management": 60,
                "network_building": 72,
                "feedback_collection": 70,
                "coaching_mentoring": 67,
                "project_coordination": 78,
                "cultural_sensitivity": 75
            },
            "network_size": 20,
            "cross_department_connections": 3,
            "credibility_score": 78,
            "cultural_alignment_score": 82,
            "experience": [
                "Led agile transformation in team",
                "Participated in company-wide change initiative",
                "Trained team on new technologies"
            ]
        },
        {
            "id": "emp_003",
            "name": "Lisa Rodriguez",
            "role": "HR Business Partner",
            "department": "HR",
            "organization_id": "techcorp_inc",
            "capabilities": {
                "change_advocacy": 70,
                "influence_building": 65,
                "communication": 88,
                "training_delivery": 85,
                "resistance_management": 78,
                "network_building": 70,
                "feedback_collection": 82,
                "coaching_mentoring": 83,
                "project_coordination": 68,
                "cultural_sensitivity": 92
            },
            "network_size": 22,
            "cross_department_connections": 6,
            "credibility_score": 85,
            "cultural_alignment_score": 95,
            "experience": [
                "Designed culture change programs",
                "Facilitated diversity and inclusion initiatives",
                "Coached managers through organizational changes"
            ]
        },
        {
            "id": "emp_004",
            "name": "David Kim",
            "role": "Product Manager",
            "department": "Product",
            "organization_id": "techcorp_inc",
            "capabilities": {
                "change_advocacy": 82,
                "influence_building": 78,
                "communication": 85,
                "training_delivery": 60,
                "resistance_management": 65,
                "network_building": 80,
                "feedback_collection": 75,
                "coaching_mentoring": 62,
                "project_coordination": 88,
                "cultural_sensitivity": 70
            },
            "network_size": 25,
            "cross_department_connections": 4,
            "credibility_score": 82,
            "cultural_alignment_score": 78,
            "experience": [
                "Led product transformation initiatives",
                "Managed stakeholder alignment",
                "Facilitated user feedback sessions"
            ]
        },
        {
            "id": "emp_005",
            "name": "Jennifer Walsh",
            "role": "Marketing Director",
            "department": "Marketing",
            "organization_id": "techcorp_inc",
            "capabilities": {
                "change_advocacy": 65,
                "influence_building": 58,
                "communication": 75,
                "training_delivery": 55,
                "resistance_management": 50,
                "network_building": 68,
                "feedback_collection": 72,
                "coaching_mentoring": 58,
                "project_coordination": 70,
                "cultural_sensitivity": 65
            },
            "network_size": 15,
            "cross_department_connections": 2,
            "credibility_score": 72,
            "cultural_alignment_score": 75,
            "experience": [
                "Participated in brand transformation",
                "Led marketing team restructure"
            ]
        }
    ]
    
    # Identify potential champions using standard criteria
    candidates = champion_engine.identify_potential_champions(
        organization_id="techcorp_inc",
        employee_data=employee_data,
        criteria_type="standard",
        target_count=4
    )
    
    print(f"Identified {len(candidates)} potential change champions:")
    print()
    
    for i, candidate in enumerate(candidates, 1):
        print(f"{i}. {candidate['name']} ({candidate['role']})")
        print(f"   Department: {candidate['department']}")
        print(f"   Champion Score: {candidate['champion_score']:.1f}/100")
        print(f"   Recommended Level: {candidate['recommended_level'].value.title()}")
        print(f"   Recommended Roles: {[role.value.title() for role in candidate['recommended_roles']]}")
        print(f"   Top Strengths:")
        for strength in candidate['strengths'][:2]:
            print(f"     • {strength}")
        print(f"   Development Areas:")
        for area in candidate['development_areas'][:2]:
            print(f"     • {area}")
        print()
    
    # Demo 2: Create Champion Profiles
    print("\n2. Creating Detailed Champion Profiles")
    print("-" * 38)
    
    # Create detailed profile for top candidate
    top_candidate = candidates[0]
    top_employee = next(emp for emp in employee_data if emp["id"] == top_candidate["employee_id"])
    
    champion_assessment = {
        "capabilities": top_employee["capabilities"],
        "influence_network": ["emp_002", "emp_003", "emp_006", "emp_007", "emp_008"],
        "credibility_score": top_employee["credibility_score"],
        "engagement_score": 88,
        "availability_score": 75,
        "motivation_score": 92,
        "cultural_fit_score": top_employee["cultural_alignment_score"],
        "change_experience": top_employee["experience"]
    }
    
    champion_profile = champion_engine.create_champion_profile(
        employee_data=top_employee,
        champion_assessment=champion_assessment
    )
    
    print(f"Champion Profile Created: {champion_profile.name}")
    print(f"Champion ID: {champion_profile.id}")
    print(f"Champion Level: {champion_profile.champion_level.value.title()}")
    print(f"Champion Roles: {[role.value.title() for role in champion_profile.champion_roles]}")
    print(f"Influence Network Size: {len(champion_profile.influence_network)}")
    print(f"Overall Capability Score: {sum(champion_profile.capabilities.values()) / len(champion_profile.capabilities):.1f}")
    
    print("\nTop Capabilities:")
    sorted_capabilities = sorted(
        champion_profile.capabilities.items(),
        key=lambda x: x[1],
        reverse=True
    )
    for capability, score in sorted_capabilities[:5]:
        print(f"  {capability.value.replace('_', ' ').title()}: {score}/100")
    
    print(f"\nKey Scores:")
    print(f"  Credibility: {champion_profile.credibility_score}/100")
    print(f"  Engagement: {champion_profile.engagement_score}/100")
    print(f"  Motivation: {champion_profile.motivation_score}/100")
    print(f"  Cultural Fit: {champion_profile.cultural_fit_score}/100")
    
    # Demo 3: Design Development Program
    print("\n\n3. Designing Champion Development Program")
    print("-" * 41)
    
    # Create champion profiles for program design
    selected_champions = []
    for candidate in candidates[:3]:  # Top 3 candidates
        emp_data = next(emp for emp in employee_data if emp["id"] == candidate["employee_id"])
        
        assessment = {
            "capabilities": emp_data["capabilities"],
            "influence_network": [f"emp_{i:03d}" for i in range(3, 8)],
            "credibility_score": emp_data["credibility_score"],
            "engagement_score": 85,
            "availability_score": 80,
            "motivation_score": 88,
            "cultural_fit_score": emp_data["cultural_alignment_score"],
            "change_experience": emp_data["experience"]
        }
        
        profile = champion_engine.create_champion_profile(emp_data, assessment)
        selected_champions.append(profile)
    
    program_objectives = [
        "Develop advanced change advocacy skills",
        "Build cross-functional influence networks",
        "Master resistance management techniques",
        "Create effective change communication strategies",
        "Establish mentoring and coaching capabilities"
    ]
    
    constraints = {
        "budget": "medium",
        "timeline": "4 months",
        "delivery_preference": "blended",
        "group_size": len(selected_champions)
    }
    
    development_program = champion_engine.design_development_program(
        champions=selected_champions,
        program_objectives=program_objectives,
        constraints=constraints
    )
    
    print(f"Development Program: {development_program.name}")
    print(f"Target Level: {development_program.target_level.value.title()}")
    print(f"Duration: {development_program.duration_weeks} weeks")
    print(f"Target Roles: {[role.value.title() for role in development_program.target_roles]}")
    print(f"Mentorship Component: {'Yes' if development_program.mentorship_component else 'No'}")
    print(f"Peer Learning Groups: {'Yes' if development_program.peer_learning_groups else 'No'}")
    print(f"Certification Available: {'Yes' if development_program.certification_available else 'No'}")
    
    print(f"\nLearning Modules ({len(development_program.learning_modules)} total):")
    for i, module in enumerate(development_program.learning_modules, 1):
        print(f"  {i}. {module.title}")
        print(f"     Type: {module.content_type.title()}, Duration: {module.duration_hours}h")
        print(f"     Delivery: {module.delivery_method.title()}")
        print(f"     Target Capabilities: {[cap.value.replace('_', ' ').title() for cap in module.target_capabilities]}")
        print(f"     Objectives: {module.learning_objectives[0]}")
        print()
    
    print(f"Practical Assignments ({len(development_program.practical_assignments)} total):")
    for i, assignment in enumerate(development_program.practical_assignments, 1):
        print(f"  {i}. {assignment.title}")
        print(f"     Type: {assignment.assignment_type.title()}, Duration: {assignment.duration_weeks} weeks")
        print(f"     Key Deliverable: {assignment.deliverables[0]}")
        print(f"     Success Metric: {assignment.success_metrics[0]}")
        print()
    
    print("Success Criteria:")
    for criterion in development_program.success_criteria:
        print(f"  • {criterion}")
    
    # Demo 4: Create Champion Network
    print("\n\n4. Creating Change Champion Network")
    print("-" * 35)
    
    network_objectives = [
        "Support organization-wide digital transformation",
        "Build change capability across all departments",
        "Facilitate knowledge sharing and best practices",
        "Provide change support to employees",
        "Drive cultural transformation initiatives"
    ]
    
    champion_network = champion_engine.create_champion_network(
        champions=selected_champions,
        network_type="cross_functional",
        objectives=network_objectives
    )
    
    print(f"Network Created: {champion_network.name}")
    print(f"Network ID: {champion_network.id}")
    print(f"Network Type: {champion_network.network_type.replace('_', ' ').title()}")
    print(f"Network Status: {champion_network.network_status.value.title()}")
    print(f"Formation Date: {champion_network.formation_date.strftime('%Y-%m-%d')}")
    print(f"Champion Count: {len(champion_network.champions)}")
    print(f"Network Lead: {champion_network.network_lead}")
    print(f"Coordinators: {len(champion_network.coordinators)}")
    print(f"Coverage Areas: {', '.join(champion_network.coverage_areas)}")
    
    print("\nNetwork Objectives:")
    for i, objective in enumerate(champion_network.objectives, 1):
        print(f"  {i}. {objective}")
    
    print("\nCommunication Channels:")
    for channel in champion_network.communication_channels:
        print(f"  • {channel}")
    
    print(f"\nMeeting Schedule: {champion_network.meeting_schedule}")
    
    print("\nSuccess Metrics:")
    for metric in champion_network.success_metrics:
        print(f"  • {metric}")
    
    # Demo 5: Network Coordination Planning
    print("\n\n5. Network Coordination Planning")
    print("-" * 33)
    
    key_initiatives = [
        "Digital transformation support program",
        "Culture change facilitation workshops",
        "Change readiness assessment rollout",
        "Resistance management training delivery",
        "Cross-department collaboration enhancement"
    ]
    
    coordination_plan = champion_engine.plan_network_coordination(
        network=champion_network,
        coordination_period="Q1_2024",
        key_initiatives=key_initiatives
    )
    
    print(f"Coordination Plan: {coordination_plan.coordination_period}")
    print(f"Plan ID: {coordination_plan.id}")
    
    print("\nKey Initiatives:")
    for i, initiative in enumerate(coordination_plan.key_initiatives, 1):
        print(f"  {i}. {initiative}")
    
    print("\nResource Allocation:")
    for resource, allocation in coordination_plan.resource_allocation.items():
        print(f"  {resource.replace('_', ' ').title()}: {allocation}")
    
    print("\nCommunication Strategy:")
    for strategy, details in coordination_plan.communication_strategy.items():
        print(f"  {strategy.replace('_', ' ').title()}: {details}")
    
    print("\nTraining Schedule:")
    for training in coordination_plan.training_schedule:
        print(f"  Month {training['month']}: {training['focus']} ({training['duration']})")
    
    print("\nPerformance Targets:")
    for target, value in coordination_plan.performance_targets.items():
        print(f"  {target.replace('_', ' ').title()}: {value}{'%' if isinstance(value, (int, float)) and value <= 100 else ''}")
    
    print("\nRisk Mitigation Strategies:")
    for strategy in coordination_plan.risk_mitigation:
        print(f"  • {strategy}")
    
    # Demo 6: Champion Performance Measurement
    print("\n\n6. Champion Performance Measurement")
    print("-" * 37)
    
    # Simulate performance data after 3 months
    performance_data = {
        "change_initiatives_supported": 7,
        "training_sessions_delivered": 12,
        "employees_influenced": 65,
        "resistance_cases_resolved": 5,
        "feedback_sessions_conducted": 18,
        "network_engagement_score": 92,
        "peer_rating": 88,
        "manager_rating": 91,
        "change_success_contribution": 85,
        "knowledge_sharing_score": 82,
        "mentorship_effectiveness": 87,
        "innovation_contributions": 3,
        "cultural_alignment_score": 90,
        "recognition_received": [
            "Change Champion of the Quarter",
            "Excellence in Leadership Award"
        ],
        "development_areas": [
            "Advanced project coordination",
            "Strategic change planning"
        ]
    }
    
    performance_metrics = champion_engine.measure_champion_performance(
        champion_id=champion_profile.id,
        measurement_period="Q1_2024",
        performance_data=performance_data
    )
    
    print(f"Performance Measurement for: {champion_profile.name}")
    print(f"Measurement Period: {performance_metrics.measurement_period}")
    print(f"Overall Performance Score: {performance_metrics.overall_performance_score:.1f}/100")
    
    print("\nKey Performance Indicators:")
    print(f"  Change Initiatives Supported: {performance_metrics.change_initiatives_supported}")
    print(f"  Training Sessions Delivered: {performance_metrics.training_sessions_delivered}")
    print(f"  Employees Influenced: {performance_metrics.employees_influenced}")
    print(f"  Resistance Cases Resolved: {performance_metrics.resistance_cases_resolved}")
    print(f"  Feedback Sessions Conducted: {performance_metrics.feedback_sessions_conducted}")
    
    print("\nRatings & Scores:")
    print(f"  Network Engagement: {performance_metrics.network_engagement_score}/100")
    print(f"  Peer Rating: {performance_metrics.peer_rating}/100")
    print(f"  Manager Rating: {performance_metrics.manager_rating}/100")
    print(f"  Change Success Contribution: {performance_metrics.change_success_contribution}/100")
    print(f"  Knowledge Sharing: {performance_metrics.knowledge_sharing_score}/100")
    print(f"  Mentorship Effectiveness: {performance_metrics.mentorship_effectiveness}/100")
    print(f"  Cultural Alignment: {performance_metrics.cultural_alignment_score}/100")
    
    print(f"\nInnovation Contributions: {performance_metrics.innovation_contributions}")
    
    print("\nRecognition Received:")
    for recognition in performance_metrics.recognition_received:
        print(f"  • {recognition}")
    
    print("\nDevelopment Areas:")
    for area in performance_metrics.development_areas:
        print(f"  • {area}")
    
    # Demo 7: Change Champion Capabilities Overview
    print("\n\n7. Change Champion Capabilities Framework")
    print("-" * 43)
    
    print("Core Change Champion Capabilities:")
    
    capability_descriptions = {
        ChangeCapability.CHANGE_ADVOCACY: "Promoting and supporting organizational change initiatives",
        ChangeCapability.INFLUENCE_BUILDING: "Building influence without formal authority",
        ChangeCapability.COMMUNICATION: "Effective communication across all levels and contexts",
        ChangeCapability.TRAINING_DELIVERY: "Designing and delivering effective training programs",
        ChangeCapability.RESISTANCE_MANAGEMENT: "Identifying and addressing change resistance",
        ChangeCapability.NETWORK_BUILDING: "Building and maintaining professional networks",
        ChangeCapability.FEEDBACK_COLLECTION: "Gathering and analyzing stakeholder feedback",
        ChangeCapability.COACHING_MENTORING: "Coaching and mentoring others through change",
        ChangeCapability.PROJECT_COORDINATION: "Coordinating change projects and initiatives",
        ChangeCapability.CULTURAL_SENSITIVITY: "Understanding and respecting cultural diversity"
    }
    
    for capability, description in capability_descriptions.items():
        weight = champion_engine.capability_weights[capability]
        print(f"\n{capability.value.replace('_', ' ').title()} (Weight: {weight:.1%})")
        print(f"  {description}")
    
    # Demo 8: Champion Development Journey
    print("\n\n8. Champion Development Journey")
    print("-" * 32)
    
    print("Champion Level Progression:")
    level_descriptions = {
        ChampionLevel.EMERGING: "New to change champion role, learning basic skills",
        ChampionLevel.DEVELOPING: "Building change champion capabilities and confidence",
        ChampionLevel.ACTIVE: "Actively supporting change initiatives with proven skills",
        ChampionLevel.SENIOR: "Leading change efforts and mentoring other champions",
        ChampionLevel.MASTER: "Expert change leader driving organizational transformation"
    }
    
    for level, description in level_descriptions.items():
        print(f"\n{level.value.title()} Champion:")
        print(f"  {description}")
    
    print("\nChampion Role Types:")
    role_descriptions = {
        ChampionRole.ADVOCATE: "Promotes change and builds support",
        ChampionRole.FACILITATOR: "Facilitates change processes and meetings",
        ChampionRole.TRAINER: "Delivers training and skill development",
        ChampionRole.MENTOR: "Mentors and coaches others through change",
        ChampionRole.COORDINATOR: "Coordinates change activities and projects",
        ChampionRole.STRATEGIST: "Develops change strategies and plans"
    }
    
    for role, description in role_descriptions.items():
        print(f"\n{role.value.title()}:")
        print(f"  {description}")
    
    print("\n" + "=" * 50)
    print("✅ Change Champion Development Demo Complete!")
    print("\nKey Capabilities Demonstrated:")
    print("• Potential change champion identification")
    print("• Comprehensive champion profiling")
    print("• Customized development program design")
    print("• Champion network creation and management")
    print("• Network coordination planning")
    print("• Performance measurement and tracking")
    print("• Multi-level capability framework")
    print("• Role-based champion development")
    print("• Cross-functional network support")
    print("• Continuous improvement and recognition")


if __name__ == "__main__":
    asyncio.run(demo_change_champion_development())