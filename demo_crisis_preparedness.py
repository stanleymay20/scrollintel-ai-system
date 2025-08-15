#!/usr/bin/env python3
"""
Crisis Preparedness Enhancement System Demo

Demonstrates crisis preparedness assessment, simulation, and training capabilities.
"""

import asyncio
from datetime import datetime, timedelta
from scrollintel.engines.crisis_preparedness_engine import CrisisPreparednessEngine
from scrollintel.models.crisis_preparedness_models import (
    SimulationType, TrainingType, CapabilityArea, PreparednessLevel
)


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_subsection(title: str):
    """Print a formatted subsection header"""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")


async def demo_crisis_preparedness():
    """Demonstrate crisis preparedness enhancement capabilities"""
    
    print_section("CRISIS PREPAREDNESS ENHANCEMENT SYSTEM DEMO")
    
    # Initialize the preparedness engine
    engine = CrisisPreparednessEngine()
    print("✅ Crisis Preparedness Engine initialized")
    
    # Sample organization data
    organization_data = {
        "industry": "technology",
        "organizational_maturity": 0.85,
        "employee_count": 750,
        "annual_revenue": 100000000,
        "previous_crises": 3,
        "training_budget": 150000,
        "crisis_detection_score": 78.0,
        "decision_making_score": 82.0,
        "communication_score": 88.0,
        "resource_mobilization_score": 74.0,
        "team_coordination_score": 80.0,
        "stakeholder_management_score": 85.0,
        "recovery_planning_score": 76.0,
        "has_crisis_plan": True,
        "last_assessment_date": "2023-06-01",
        "crisis_team_size": 12,
        "backup_systems": True,
        "communication_channels": 5
    }
    
    print_subsection("1. CRISIS PREPAREDNESS ASSESSMENT")
    
    # Conduct comprehensive preparedness assessment
    assessment = engine.assess_crisis_preparedness(
        organization_data=organization_data,
        assessor_id="senior_assessor_001"
    )
    
    print(f"✅ Preparedness assessment completed: {assessment.id}")
    print(f"   Overall Preparedness Level: {assessment.overall_preparedness_level.value.upper()}")
    print(f"   Overall Score: {assessment.overall_score:.1f}%")
    print(f"   Assessment Confidence: {assessment.confidence_level:.1f}%")
    
    print(f"\n📊 Capability Assessment Results:")
    for capability, score in assessment.capability_scores.items():
        level = assessment.capability_levels[capability]
        level_icon = "🟢" if level.value == "excellent" else "🔵" if level.value == "good" else "🟡" if level.value == "adequate" else "🟠" if level.value == "poor" else "🔴"
        print(f"   {level_icon} {capability.value.replace('_', ' ').title()}: {score:.1f}% ({level.value.title()})")
    
    print(f"\n💪 Identified Strengths ({len(assessment.strengths)}):")
    for strength in assessment.strengths:
        print(f"   ✅ {strength}")
    
    print(f"\n⚠️ Areas for Improvement ({len(assessment.weaknesses)}):")
    for weakness in assessment.weaknesses:
        print(f"   ❌ {weakness}")
    
    print(f"\n🔍 Preparedness Gaps ({len(assessment.gaps_identified)}):")
    for gap in assessment.gaps_identified:
        print(f"   🔸 {gap}")
    
    print(f"\n🚨 High-Risk Scenarios ({len(assessment.high_risk_scenarios)}):")
    for scenario in assessment.high_risk_scenarios:
        print(f"   ⚡ {scenario}")
    
    print(f"\n🎯 Vulnerability Areas ({len(assessment.vulnerability_areas)}):")
    for vulnerability in assessment.vulnerability_areas:
        print(f"   🔻 {vulnerability}")
    
    print_subsection("2. CRISIS SIMULATION EXERCISES")
    
    # Create different types of crisis simulations
    simulations = []
    
    # System outage simulation
    system_outage_sim = engine.create_crisis_simulation(
        simulation_type=SimulationType.SYSTEM_OUTAGE,
        participants=["cto", "ops_manager", "comm_lead", "customer_success", "security_lead"],
        facilitator_id="simulation_expert_001",
        custom_parameters={
            "title": "Major System Outage Response Drill",
            "complexity_level": "High",
            "duration_minutes": 180,
            "scenario_details": "Critical production systems have failed during peak business hours, affecting 80% of customers"
        }
    )
    simulations.append(system_outage_sim)
    
    # Security breach simulation
    security_breach_sim = engine.create_crisis_simulation(
        simulation_type=SimulationType.SECURITY_BREACH,
        participants=["ciso", "legal_counsel", "pr_manager", "cto", "hr_director"],
        facilitator_id="security_expert_001",
        custom_parameters={
            "title": "Data Breach Response Exercise",
            "complexity_level": "Critical",
            "duration_minutes": 240,
            "scenario_details": "Potential customer data breach detected, requiring immediate investigation and response"
        }
    )
    simulations.append(security_breach_sim)
    
    print(f"✅ Created {len(simulations)} crisis simulations")
    
    for i, sim in enumerate(simulations, 1):
        print(f"\n🎭 Simulation {i}: {sim.title}")
        print(f"   Type: {sim.simulation_type.value.replace('_', ' ').title()}")
        print(f"   Complexity: {sim.complexity_level}")
        print(f"   Duration: {sim.duration_minutes} minutes")
        print(f"   Participants: {len(sim.participants)}")
        print(f"   Facilitator: {sim.facilitator_id}")
        print(f"   Status: {sim.simulation_status}")
        print(f"   Scheduled: {sim.scheduled_date.strftime('%Y-%m-%d %H:%M')}")
        
        print(f"   📋 Learning Objectives:")
        for objective in sim.learning_objectives:
            print(f"      • {objective}")
        
        print(f"   ✅ Success Criteria:")
        for criteria in sim.success_criteria:
            print(f"      • {criteria}")
    
    print_subsection("3. SIMULATION EXECUTION")
    
    # Execute the first simulation
    selected_sim = simulations[0]
    print(f"🎬 Executing simulation: {selected_sim.title}")
    
    execution_results = engine.execute_simulation(selected_sim)
    
    print(f"✅ Simulation execution completed")
    print(f"   Status: {selected_sim.simulation_status}")
    print(f"   Duration: {(selected_sim.actual_end_time - selected_sim.actual_start_time).total_seconds() / 60:.1f} minutes")
    
    print(f"\n👥 Participant Performance:")
    for participant, score in selected_sim.participant_performance.items():
        performance_icon = "🏆" if score >= 90 else "🥇" if score >= 80 else "🥈" if score >= 70 else "🥉"
        print(f"   {performance_icon} {participant}: {score:.1f}%")
    
    print(f"\n🎯 Objectives Achieved ({len(selected_sim.objectives_achieved)}):")
    for objective in selected_sim.objectives_achieved:
        print(f"   ✅ {objective}")
    
    print(f"\n📚 Lessons Learned ({len(selected_sim.lessons_learned)}):")
    for lesson in selected_sim.lessons_learned:
        print(f"   💡 {lesson}")
    
    print(f"\n🔧 Improvement Areas ({len(selected_sim.improvement_areas)}):")
    for area in selected_sim.improvement_areas:
        print(f"   🔸 {area}")
    
    print_subsection("4. TRAINING PROGRAM DEVELOPMENT")
    
    # Develop training programs for different capability areas
    training_programs = []
    
    # Crisis detection training
    detection_training = engine.develop_training_program(
        capability_area=CapabilityArea.CRISIS_DETECTION,
        target_audience=["managers", "team_leads", "operations_staff"],
        training_type=TrainingType.TABLETOP_EXERCISE
    )
    training_programs.append(detection_training)
    
    # Communication training
    communication_training = engine.develop_training_program(
        capability_area=CapabilityArea.COMMUNICATION,
        target_audience=["executives", "pr_team", "customer_success"],
        training_type=TrainingType.SIMULATION_DRILL
    )
    training_programs.append(communication_training)
    
    # Leadership development
    leadership_training = engine.develop_training_program(
        capability_area=CapabilityArea.TEAM_COORDINATION,
        target_audience=["senior_managers", "directors", "c_suite"],
        training_type=TrainingType.LEADERSHIP_DEVELOPMENT
    )
    training_programs.append(leadership_training)
    
    print(f"✅ Developed {len(training_programs)} training programs")
    
    for i, program in enumerate(training_programs, 1):
        print(f"\n📚 Training Program {i}: {program.program_name}")
        print(f"   Type: {program.training_type.value.replace('_', ' ').title()}")
        print(f"   Duration: {program.duration_hours} hours")
        print(f"   Target Audience: {', '.join(program.target_audience)}")
        print(f"   Status: {program.approval_status}")
        print(f"   Version: {program.version}")
        
        print(f"   🎯 Learning Objectives:")
        for objective in program.learning_objectives:
            print(f"      • {objective}")
        
        print(f"   🏆 Competencies Developed:")
        for competency in program.competencies_developed:
            print(f"      • {competency}")
    
    print_subsection("5. CAPABILITY DEVELOPMENT PLANNING")
    
    # Create capability development plans for areas needing improvement
    development_plans = []
    
    # Find capabilities that need improvement (score < 80)
    improvement_capabilities = [
        (capability, score) for capability, score in assessment.capability_scores.items()
        if score < 80
    ]
    
    for capability, current_score in improvement_capabilities[:3]:  # Top 3 priorities
        target_level = PreparednessLevel.EXCELLENT if current_score > 70 else PreparednessLevel.GOOD
        
        plan = engine.create_capability_development_plan(
            capability_area=capability,
            current_assessment=assessment,
            target_level=target_level
        )
        development_plans.append(plan)
    
    print(f"✅ Created {len(development_plans)} capability development plans")
    
    for i, plan in enumerate(development_plans, 1):
        print(f"\n🎯 Development Plan {i}: {plan.capability_area.value.replace('_', ' ').title()}")
        print(f"   Current Level: {plan.current_level.value.title()}")
        print(f"   Target Level: {plan.target_level.value.title()}")
        print(f"   Progress: {plan.current_progress:.1f}%")
        print(f"   Status: {plan.status}")
        print(f"   Budget: ${plan.budget_allocated:,.0f}")
        print(f"   Timeline: {plan.start_date.strftime('%Y-%m-%d')} to {plan.target_completion_date.strftime('%Y-%m-%d')}")
        
        print(f"   📋 Development Objectives:")
        for objective in plan.development_objectives:
            print(f"      • {objective}")
        
        print(f"   🔧 Improvement Actions:")
        for action in plan.improvement_actions:
            print(f"      • {action}")
        
        print(f"   📚 Training Requirements:")
        for requirement in plan.training_requirements:
            print(f"      • {requirement}")
        
        print(f"   🏁 Milestones ({len(plan.milestones)}):")
        for milestone in plan.milestones:
            print(f"      • {milestone['milestone']} (Target: {milestone['target_date'][:10]})")
    
    print_subsection("6. COMPREHENSIVE PREPAREDNESS REPORT")
    
    # Generate comprehensive preparedness report
    report = engine.generate_preparedness_report(
        assessment=assessment,
        simulations=simulations,
        training_programs=training_programs
    )
    
    print(f"✅ Generated preparedness report: {report.id}")
    print(f"   Title: {report.report_title}")
    print(f"   Type: {report.report_type}")
    print(f"   Generated: {report.generated_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Status: {report.review_status}")
    
    print(f"\n📄 Executive Summary:")
    print(f"   {report.executive_summary}")
    
    print(f"\n📊 Current State Assessment:")
    print(f"   {report.current_state_assessment}")
    
    print(f"\n🔍 Gap Analysis:")
    print(f"   {report.gap_analysis}")
    
    print(f"\n💡 Improvement Recommendations:")
    print(f"   {report.improvement_recommendations}")
    
    print(f"\n🗺️ Implementation Roadmap:")
    print(f"   {report.implementation_roadmap}")
    
    print(f"\n📈 Supporting Data:")
    print(f"   Assessment Data: {len(report.assessment_data)} key metrics")
    print(f"   Simulation Results: {len(report.simulation_results)} exercises")
    print(f"   Training Outcomes: {len(report.training_outcomes)} programs")
    
    print_subsection("7. PREPAREDNESS INSIGHTS & RECOMMENDATIONS")
    
    # Calculate key insights
    high_performing_capabilities = [
        capability for capability, score in assessment.capability_scores.items()
        if score >= 85
    ]
    
    critical_capabilities = [
        capability for capability, score in assessment.capability_scores.items()
        if score < 70
    ]
    
    total_training_hours = sum(program.duration_hours for program in training_programs)
    total_development_budget = sum(plan.budget_allocated for plan in development_plans)
    
    print(f"📈 Key Preparedness Insights:")
    print(f"   • Overall preparedness maturity: {assessment.overall_preparedness_level.value.title()}")
    print(f"   • High-performing capabilities: {len(high_performing_capabilities)}")
    print(f"   • Critical improvement areas: {len(critical_capabilities)}")
    print(f"   • Simulations planned: {len(simulations)}")
    print(f"   • Training programs developed: {len(training_programs)}")
    print(f"   • Total training hours: {total_training_hours:.1f}")
    print(f"   • Development investment: ${total_development_budget:,.0f}")
    
    print(f"\n🎯 Strategic Recommendations:")
    print(f"   • Prioritize {len(critical_capabilities)} critical capability areas")
    print(f"   • Execute {len(simulations)} crisis simulations quarterly")
    print(f"   • Implement {len(training_programs)} specialized training programs")
    print(f"   • Establish continuous improvement cycle with quarterly assessments")
    print(f"   • Build crisis response muscle memory through regular drills")
    
    # Preparedness maturity assessment
    if assessment.overall_score >= 90:
        maturity_assessment = "🏆 Crisis Leadership Excellence - Organization demonstrates exceptional preparedness"
    elif assessment.overall_score >= 80:
        maturity_assessment = "🥇 Advanced Preparedness - Strong crisis response capabilities with minor gaps"
    elif assessment.overall_score >= 70:
        maturity_assessment = "🥈 Adequate Preparedness - Basic capabilities in place, improvement needed"
    else:
        maturity_assessment = "🚨 Critical Gaps - Immediate action required to build crisis response capabilities"
    
    print(f"\n🏅 Preparedness Maturity Assessment:")
    print(f"   {maturity_assessment}")
    
    print_subsection("8. IMPLEMENTATION TIMELINE")
    
    print("📅 Recommended Implementation Timeline:")
    print("   Phase 1 (Months 1-2): Address critical gaps and conduct initial simulations")
    print("   Phase 2 (Months 3-4): Implement training programs and capability development")
    print("   Phase 3 (Months 5-6): Advanced simulations and leadership development")
    print("   Phase 4 (Ongoing): Continuous improvement and quarterly assessments")
    
    print(f"\n🔄 Continuous Improvement Cycle:")
    print(f"   • Quarterly preparedness assessments")
    print(f"   • Monthly crisis simulation exercises")
    print(f"   • Bi-annual training program updates")
    print(f"   • Annual comprehensive preparedness review")
    
    print_section("DEMO COMPLETED SUCCESSFULLY")
    print("🎉 Crisis Preparedness Enhancement System demonstration completed!")
    print("\n📋 Summary:")
    print(f"   • Overall preparedness score: {assessment.overall_score:.1f}%")
    print(f"   • Preparedness level: {assessment.overall_preparedness_level.value.title()}")
    print(f"   • Simulations created: {len(simulations)}")
    print(f"   • Training programs developed: {len(training_programs)}")
    print(f"   • Development plans created: {len(development_plans)}")
    print(f"   • Total investment required: ${total_development_budget:,.0f}")
    
    return {
        "assessment": assessment,
        "simulations": simulations,
        "training_programs": training_programs,
        "development_plans": development_plans,
        "report": report,
        "preparedness_score": assessment.overall_score
    }


if __name__ == "__main__":
    asyncio.run(demo_crisis_preparedness())