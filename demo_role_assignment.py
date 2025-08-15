#!/usr/bin/env python3
"""
Demo script for Role Assignment Engine

This script demonstrates the crisis role assignment capabilities,
showing how the system assigns roles based on individual strengths
and provides clear role communication.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scrollintel.engines.role_assignment_engine import (
    RoleAssignmentEngine, Person, PersonSkill, RoleType, SkillLevel
)
import json
from datetime import datetime


def create_demo_people():
    """Create demo people with various skills and backgrounds"""
    return [
        Person(
            id="alice_johnson",
            name="Alice Johnson",
            current_availability=1.0,
            skills=[
                PersonSkill("leadership", SkillLevel.EXPERT, 10, 0.95, True),
                PersonSkill("decision_making", SkillLevel.EXPERT, 8, 0.92, True),
                PersonSkill("crisis_management", SkillLevel.EXPERT, 12, 0.98, True),
                PersonSkill("stakeholder_management", SkillLevel.ADVANCED, 6, 0.88, True)
            ],
            preferred_roles=[RoleType.CRISIS_COMMANDER],
            stress_tolerance=0.95,
            leadership_experience=10,
            crisis_history=["system_outage_2023", "security_breach_2022", "pr_crisis_2021"],
            current_workload=0.1
        ),
        Person(
            id="bob_smith",
            name="Bob Smith",
            current_availability=0.9,
            skills=[
                PersonSkill("technical_expertise", SkillLevel.EXPERT, 15, 0.94, True),
                PersonSkill("problem_solving", SkillLevel.EXPERT, 12, 0.91, True),
                PersonSkill("system_architecture", SkillLevel.EXPERT, 10, 0.96, True),
                PersonSkill("leadership", SkillLevel.INTERMEDIATE, 3, 0.75, False)
            ],
            preferred_roles=[RoleType.TECHNICAL_LEAD],
            stress_tolerance=0.88,
            leadership_experience=5,
            crisis_history=["system_outage_2023", "data_corruption_2022"],
            current_workload=0.2
        ),
        Person(
            id="carol_davis",
            name="Carol Davis",
            current_availability=0.95,
            skills=[
                PersonSkill("communication", SkillLevel.EXPERT, 9, 0.93, True),
                PersonSkill("public_relations", SkillLevel.EXPERT, 11, 0.97, True),
                PersonSkill("stakeholder_management", SkillLevel.ADVANCED, 7, 0.89, True),
                PersonSkill("media_relations", SkillLevel.EXPERT, 8, 0.94, True)
            ],
            preferred_roles=[RoleType.COMMUNICATION_LEAD, RoleType.MEDIA_HANDLER],
            stress_tolerance=0.85,
            leadership_experience=6,
            crisis_history=["pr_crisis_2022", "customer_complaint_2021"],
            current_workload=0.15
        ),
        Person(
            id="david_wilson",
            name="David Wilson",
            current_availability=0.8,
            skills=[
                PersonSkill("resource_management", SkillLevel.ADVANCED, 8, 0.87, True),
                PersonSkill("logistics", SkillLevel.EXPERT, 10, 0.92, True),
                PersonSkill("coordination", SkillLevel.ADVANCED, 6, 0.84, False),
                PersonSkill("vendor_management", SkillLevel.ADVANCED, 7, 0.88, True)
            ],
            preferred_roles=[RoleType.RESOURCE_COORDINATOR],
            stress_tolerance=0.78,
            leadership_experience=3,
            crisis_history=["supply_chain_2023"],
            current_workload=0.3
        ),
        Person(
            id="eva_martinez",
            name="Eva Martinez",
            current_availability=1.0,
            skills=[
                PersonSkill("stakeholder_management", SkillLevel.EXPERT, 9, 0.91, True),
                PersonSkill("communication", SkillLevel.ADVANCED, 7, 0.86, True),
                PersonSkill("negotiation", SkillLevel.EXPERT, 8, 0.93, True),
                PersonSkill("relationship_building", SkillLevel.EXPERT, 10, 0.95, False)
            ],
            preferred_roles=[RoleType.STAKEHOLDER_LIAISON],
            stress_tolerance=0.82,
            leadership_experience=4,
            crisis_history=["investor_relations_2022"],
            current_workload=0.1
        ),
        Person(
            id="frank_brown",
            name="Frank Brown",
            current_availability=0.7,
            skills=[
                PersonSkill("operations_management", SkillLevel.ADVANCED, 12, 0.89, True),
                PersonSkill("process_optimization", SkillLevel.EXPERT, 8, 0.92, False),
                PersonSkill("team_coordination", SkillLevel.ADVANCED, 6, 0.85, True),
                PersonSkill("quality_control", SkillLevel.ADVANCED, 9, 0.88, False)
            ],
            preferred_roles=[RoleType.OPERATIONS_MANAGER],
            stress_tolerance=0.80,
            leadership_experience=7,
            crisis_history=["operational_failure_2023"],
            current_workload=0.4
        )
    ]


def print_separator(title):
    """Print a formatted separator"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def print_person_summary(people):
    """Print summary of available people"""
    print_separator("AVAILABLE PERSONNEL")
    
    for person in people:
        print(f"\n👤 {person.name} (ID: {person.id})")
        print(f"   Availability: {person.current_availability:.0%}")
        print(f"   Current Workload: {person.current_workload:.0%}")
        print(f"   Stress Tolerance: {person.stress_tolerance:.0%}")
        print(f"   Leadership Experience: {person.leadership_experience} years")
        print(f"   Crisis History: {len(person.crisis_history)} previous crises")
        
        print("   🎯 Preferred Roles:")
        for role in person.preferred_roles:
            print(f"      • {role.value.replace('_', ' ').title()}")
        
        print("   💪 Top Skills:")
        top_skills = sorted(person.skills, key=lambda s: s.level.value, reverse=True)[:3]
        for skill in top_skills:
            print(f"      • {skill.skill_name.replace('_', ' ').title()}: "
                  f"Level {skill.level.value}/5 ({skill.years_experience}y exp)")


def print_assignment_results(result):
    """Print detailed assignment results"""
    print_separator("ROLE ASSIGNMENT RESULTS")
    
    print(f"📊 Assignment Quality Score: {result.assignment_quality_score:.1%}")
    print(f"✅ Roles Assigned: {len(result.assignments)}")
    print(f"❌ Unassigned Roles: {len(result.unassigned_roles)}")
    
    if result.assignments:
        print("\n🎯 ROLE ASSIGNMENTS:")
        for assignment in result.assignments:
            print(f"\n   Role: {assignment.role_type.value.replace('_', ' ').title()}")
            print(f"   Assigned to: Person {assignment.person_id}")
            print(f"   Confidence: {assignment.assignment_confidence:.1%}")
            print(f"   Assignment Time: {assignment.assignment_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            print("   📋 Key Responsibilities:")
            for i, responsibility in enumerate(assignment.responsibilities[:3], 1):
                print(f"      {i}. {responsibility}")
            
            print("   📊 Reporting Structure:")
            for key, value in assignment.reporting_structure.items():
                print(f"      • {key.replace('_', ' ').title()}: {value}")
    
    if result.unassigned_roles:
        print(f"\n❌ UNASSIGNED ROLES:")
        for role in result.unassigned_roles:
            print(f"   • {role.value.replace('_', ' ').title()}")
    
    if result.backup_assignments:
        print(f"\n🔄 BACKUP ASSIGNMENTS:")
        for role, backups in result.backup_assignments.items():
            if backups:
                print(f"   {role.value.replace('_', ' ').title()}:")
                for i, backup_id in enumerate(backups, 1):
                    print(f"      {i}. Person {backup_id}")
    
    if result.recommendations:
        print(f"\n💡 RECOMMENDATIONS:")
        for i, recommendation in enumerate(result.recommendations, 1):
            print(f"   {i}. {recommendation}")


def demonstrate_role_clarity(engine, assignment):
    """Demonstrate role clarity communication"""
    print_separator("ROLE CLARITY COMMUNICATION")
    
    clarity = engine.get_role_clarity_communication(assignment)
    
    print(f"📋 Assignment Summary:")
    summary = clarity["assignment_summary"]
    print(f"   Person: {summary['person']}")
    print(f"   Role: {summary['role']}")
    print(f"   Confidence: {summary['confidence']}")
    print(f"   Assignment Time: {summary['assignment_time']}")
    
    print(f"\n📝 Responsibilities:")
    for i, responsibility in enumerate(clarity["responsibilities"], 1):
        print(f"   {i}. {responsibility}")
    
    print(f"\n📊 Reporting Structure:")
    for key, value in clarity["reporting_structure"].items():
        print(f"   • {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n🎯 Success Criteria:")
    for i, criterion in enumerate(clarity["success_criteria"], 1):
        print(f"   {i}. {criterion}")
    
    print(f"\n📞 Communication Protocols:")
    protocols = clarity["communication_protocols"]
    for key, value in protocols.items():
        print(f"   • {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n🚨 Escalation Procedures:")
    for i, procedure in enumerate(clarity["escalation_procedures"], 1):
        print(f"   {i}. {procedure}")


def demonstrate_assignment_confirmation(engine, assignment_id):
    """Demonstrate assignment confirmation process"""
    print_separator("ASSIGNMENT CONFIRMATION")
    
    # Simulate confirmation
    print("🤝 Simulating assignment confirmation...")
    
    # Confirmed assignment
    confirmed_result = engine.confirm_role_assignment(assignment_id, True)
    print(f"\n✅ Assignment Confirmed:")
    print(f"   Confirmation Time: {confirmed_result['confirmation_time']}")
    print(f"   Next Steps:")
    for i, step in enumerate(confirmed_result['next_steps'], 1):
        print(f"      {i}. {step}")
    
    # Declined assignment
    declined_result = engine.confirm_role_assignment(assignment_id, False)
    print(f"\n❌ Assignment Declined (Alternative Scenario):")
    print(f"   Next Steps:")
    for i, step in enumerate(declined_result['next_steps'], 1):
        print(f"      {i}. {step}")


def run_crisis_scenarios():
    """Run different crisis scenarios"""
    engine = RoleAssignmentEngine()
    people = create_demo_people()
    
    scenarios = [
        {
            "name": "Major System Outage",
            "crisis_id": "system_outage_2024_001",
            "required_roles": [
                RoleType.CRISIS_COMMANDER,
                RoleType.TECHNICAL_LEAD,
                RoleType.COMMUNICATION_LEAD,
                RoleType.STAKEHOLDER_LIAISON
            ],
            "severity": 0.9
        },
        {
            "name": "Security Breach",
            "crisis_id": "security_breach_2024_001",
            "required_roles": [
                RoleType.CRISIS_COMMANDER,
                RoleType.TECHNICAL_LEAD,
                RoleType.COMMUNICATION_LEAD,
                RoleType.SECURITY_SPECIALIST
            ],
            "severity": 0.8
        },
        {
            "name": "PR Crisis",
            "crisis_id": "pr_crisis_2024_001",
            "required_roles": [
                RoleType.CRISIS_COMMANDER,
                RoleType.COMMUNICATION_LEAD,
                RoleType.MEDIA_HANDLER,
                RoleType.STAKEHOLDER_LIAISON
            ],
            "severity": 0.6
        }
    ]
    
    print_person_summary(people)
    
    for scenario in scenarios:
        print_separator(f"CRISIS SCENARIO: {scenario['name']}")
        print(f"Crisis ID: {scenario['crisis_id']}")
        print(f"Severity Level: {scenario['severity']:.0%}")
        print(f"Required Roles: {len(scenario['required_roles'])}")
        
        # Perform role assignment
        result = engine.assign_roles(
            crisis_id=scenario['crisis_id'],
            available_people=people,
            required_roles=scenario['required_roles'],
            crisis_severity=scenario['severity']
        )
        
        print_assignment_results(result)
        
        # Demonstrate role clarity for first assignment
        if result.assignments:
            demonstrate_role_clarity(engine, result.assignments[0])
            demonstrate_assignment_confirmation(engine, f"{scenario['crisis_id']}_assignment_1")
        
        # Update people availability for next scenario
        for person in people:
            if any(a.person_id == person.id for a in result.assignments):
                person.current_workload += 0.3  # Increase workload for assigned people
                person.current_availability = max(0.1, person.current_availability - 0.2)


def main():
    """Main demo function"""
    print("🚨 CRISIS LEADERSHIP EXCELLENCE - ROLE ASSIGNMENT ENGINE DEMO")
    print("=" * 70)
    print("This demo showcases the role assignment engine's ability to:")
    print("• Assign crisis roles based on individual strengths")
    print("• Optimize assignments for crisis effectiveness")
    print("• Provide clear role communication and confirmation")
    print("• Generate backup assignments and recommendations")
    
    try:
        run_crisis_scenarios()
        
        print_separator("DEMO COMPLETED SUCCESSFULLY")
        print("✅ Role Assignment Engine demonstrated successfully!")
        print("🎯 Key capabilities shown:")
        print("   • Intelligent role-person matching")
        print("   • Crisis severity consideration")
        print("   • Role clarity communication")
        print("   • Assignment confirmation process")
        print("   • Backup assignment generation")
        print("   • Quality scoring and recommendations")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())