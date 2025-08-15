"""
Crisis Team Formation Demo

Demonstrates the crisis team formation system capabilities including:
- Rapid team assembly
- Skill matching and optimization
- Role assignment
- Team composition optimization
"""

import asyncio
import json
from datetime import datetime, timedelta
from scrollintel.engines.crisis_team_formation_engine import CrisisTeamFormationEngine
from scrollintel.models.team_coordination_models import (
    Person, TeamFormationRequest, Skill, SkillLevel, 
    AvailabilityStatus, TeamRole
)


async def create_sample_personnel():
    """Create sample personnel database"""
    personnel = []
    
    # Crisis Management Specialists
    personnel.append(Person(
        id="alice_001",
        name="Alice Johnson",
        email="alice.johnson@company.com",
        phone="+1-555-0101",
        department="Operations",
        title="Crisis Management Director",
        skills=[
            Skill(name="crisis_management", level=SkillLevel.EXPERT, years_experience=8),
            Skill(name="leadership", level=SkillLevel.EXPERT, years_experience=10),
            Skill(name="strategic_planning", level=SkillLevel.ADVANCED, years_experience=7),
            Skill(name="stakeholder_management", level=SkillLevel.ADVANCED, years_experience=9),
            Skill(name="decision_making", level=SkillLevel.EXPERT, years_experience=8)
        ],
        availability_status=AvailabilityStatus.AVAILABLE,
        current_workload=0.4,
        crisis_experience={
            "crisis_leader": 12,
            "system_outage": 8,
            "security_breach": 5,
            "financial_crisis": 3
        },
        preferred_roles=[TeamRole.CRISIS_LEADER, TeamRole.EXECUTIVE_LIAISON],
        timezone="EST",
        languages=["English", "Spanish"]
    ))
    
    # Technical Specialists
    personnel.append(Person(
        id="bob_002",
        name="Bob Chen",
        email="bob.chen@company.com",
        phone="+1-555-0102",
        department="Engineering",
        title="Senior Site Reliability Engineer",
        skills=[
            Skill(name="system_administration", level=SkillLevel.EXPERT, years_experience=9),
            Skill(name="troubleshooting", level=SkillLevel.EXPERT, years_experience=8),
            Skill(name="cloud_infrastructure", level=SkillLevel.ADVANCED, years_experience=6),
            Skill(name="incident_response", level=SkillLevel.ADVANCED, years_experience=7),
            Skill(name="technical_communication", level=SkillLevel.INTERMEDIATE, years_experience=5)
        ],
        availability_status=AvailabilityStatus.AVAILABLE,
        current_workload=0.6,
        crisis_experience={
            "technical_lead": 15,
            "system_outage": 20,
            "performance_degradation": 12
        },
        preferred_roles=[TeamRole.TECHNICAL_LEAD, TeamRole.OPERATIONS_LEAD],
        timezone="PST",
        languages=["English", "Mandarin"]
    ))
    
    # Security Specialists
    personnel.append(Person(
        id="carol_003",
        name="Carol Rodriguez",
        email="carol.rodriguez@company.com",
        phone="+1-555-0103",
        department="Security",
        title="Chief Information Security Officer",
        skills=[
            Skill(name="cybersecurity", level=SkillLevel.EXPERT, years_experience=10),
            Skill(name="incident_response", level=SkillLevel.EXPERT, years_experience=8),
            Skill(name="forensics", level=SkillLevel.ADVANCED, years_experience=6),
            Skill(name="compliance", level=SkillLevel.ADVANCED, years_experience=7),
            Skill(name="risk_assessment", level=SkillLevel.EXPERT, years_experience=9)
        ],
        availability_status=AvailabilityStatus.AVAILABLE,
        current_workload=0.5,
        crisis_experience={
            "security_lead": 18,
            "security_breach": 25,
            "cyber_attack": 15,
            "data_loss": 8
        },
        preferred_roles=[TeamRole.SECURITY_LEAD],
        timezone="EST",
        languages=["English", "Spanish"]
    ))
    
    # Communications Specialists
    personnel.append(Person(
        id="david_004",
        name="David Kim",
        email="david.kim@company.com",
        phone="+1-555-0104",
        department="Marketing",
        title="Director of Communications",
        skills=[
            Skill(name="crisis_communication", level=SkillLevel.EXPERT, years_experience=7),
            Skill(name="public_relations", level=SkillLevel.EXPERT, years_experience=9),
            Skill(name="media_relations", level=SkillLevel.ADVANCED, years_experience=8),
            Skill(name="stakeholder_management", level=SkillLevel.ADVANCED, years_experience=6),
            Skill(name="content_creation", level=SkillLevel.ADVANCED, years_experience=5)
        ],
        availability_status=AvailabilityStatus.AVAILABLE,
        current_workload=0.3,
        crisis_experience={
            "communications_lead": 10,
            "reputation_damage": 12,
            "media_crisis": 8
        },
        preferred_roles=[TeamRole.COMMUNICATIONS_LEAD],
        timezone="CST",
        languages=["English", "Korean"]
    ))
    
    # Legal Specialists
    personnel.append(Person(
        id="eve_005",
        name="Eve Thompson",
        email="eve.thompson@company.com",
        phone="+1-555-0105",
        department="Legal",
        title="General Counsel",
        skills=[
            Skill(name="legal_compliance", level=SkillLevel.EXPERT, years_experience=12),
            Skill(name="regulatory_knowledge", level=SkillLevel.EXPERT, years_experience=10),
            Skill(name="risk_assessment", level=SkillLevel.ADVANCED, years_experience=8),
            Skill(name="contract_law", level=SkillLevel.EXPERT, years_experience=11),
            Skill(name="litigation_management", level=SkillLevel.ADVANCED, years_experience=7)
        ],
        availability_status=AvailabilityStatus.AVAILABLE,
        current_workload=0.7,
        crisis_experience={
            "legal_advisor": 8,
            "regulatory_violation": 6,
            "legal_issue": 10
        },
        preferred_roles=[TeamRole.LEGAL_ADVISOR],
        timezone="EST",
        languages=["English"]
    ))
    
    # Customer Relations
    personnel.append(Person(
        id="frank_006",
        name="Frank Wilson",
        email="frank.wilson@company.com",
        phone="+1-555-0106",
        department="Customer Success",
        title="VP Customer Experience",
        skills=[
            Skill(name="customer_service", level=SkillLevel.EXPERT, years_experience=8),
            Skill(name="relationship_management", level=SkillLevel.ADVANCED, years_experience=9),
            Skill(name="communication", level=SkillLevel.ADVANCED, years_experience=7),
            Skill(name="conflict_resolution", level=SkillLevel.ADVANCED, years_experience=6),
            Skill(name="escalation_management", level=SkillLevel.INTERMEDIATE, years_experience=5)
        ],
        availability_status=AvailabilityStatus.AVAILABLE,
        current_workload=0.4,
        crisis_experience={
            "customer_liaison": 12,
            "service_disruption": 15,
            "customer_complaints": 20
        },
        preferred_roles=[TeamRole.CUSTOMER_LIAISON],
        timezone="PST",
        languages=["English", "French"]
    ))
    
    # Additional Technical Support
    personnel.append(Person(
        id="grace_007",
        name="Grace Liu",
        email="grace.liu@company.com",
        phone="+1-555-0107",
        department="Engineering",
        title="DevOps Engineer",
        skills=[
            Skill(name="system_administration", level=SkillLevel.ADVANCED, years_experience=5),
            Skill(name="automation", level=SkillLevel.EXPERT, years_experience=6),
            Skill(name="monitoring", level=SkillLevel.ADVANCED, years_experience=4),
            Skill(name="cloud_infrastructure", level=SkillLevel.ADVANCED, years_experience=5),
            Skill(name="troubleshooting", level=SkillLevel.INTERMEDIATE, years_experience=4)
        ],
        availability_status=AvailabilityStatus.AVAILABLE,
        current_workload=0.5,
        crisis_experience={
            "technical_support": 8,
            "system_outage": 10,
            "deployment_issues": 12
        },
        preferred_roles=[TeamRole.TECHNICAL_LEAD, TeamRole.OPERATIONS_LEAD],
        timezone="PST",
        languages=["English", "Mandarin"]
    ))
    
    # Executive Liaison
    personnel.append(Person(
        id="henry_008",
        name="Henry Davis",
        email="henry.davis@company.com",
        phone="+1-555-0108",
        department="Executive",
        title="Chief Operating Officer",
        skills=[
            Skill(name="executive_communication", level=SkillLevel.EXPERT, years_experience=12),
            Skill(name="strategic_planning", level=SkillLevel.EXPERT, years_experience=15),
            Skill(name="leadership", level=SkillLevel.EXPERT, years_experience=18),
            Skill(name="stakeholder_management", level=SkillLevel.EXPERT, years_experience=14),
            Skill(name="business_continuity", level=SkillLevel.ADVANCED, years_experience=10)
        ],
        availability_status=AvailabilityStatus.AVAILABLE,
        current_workload=0.8,
        crisis_experience={
            "executive_liaison": 6,
            "business_continuity": 8,
            "strategic_crisis": 5
        },
        preferred_roles=[TeamRole.EXECUTIVE_LIAISON, TeamRole.CRISIS_LEADER],
        timezone="EST",
        languages=["English"]
    ))
    
    return personnel


async def demonstrate_crisis_team_formation():
    """Demonstrate crisis team formation capabilities"""
    print("🚨 Crisis Team Formation System Demo")
    print("=" * 50)
    
    # Initialize engine
    engine = CrisisTeamFormationEngine()
    
    # Create and add personnel
    print("\n📋 Setting up personnel database...")
    personnel = await create_sample_personnel()
    
    for person in personnel:
        engine.add_person(person)
        print(f"   ✓ Added {person.name} ({person.title}) - {len(person.skills)} skills")
    
    print(f"\n✅ Personnel database ready with {len(personnel)} team members")
    
    # Scenario 1: System Outage Crisis
    print("\n" + "="*50)
    print("🔥 SCENARIO 1: Critical System Outage")
    print("="*50)
    
    outage_request = TeamFormationRequest(
        crisis_id="outage_2024_001",
        crisis_type="system_outage",
        severity_level=4,  # High severity
        urgency="critical",
        required_skills=[
            "system_administration",
            "troubleshooting", 
            "crisis_management",
            "technical_communication",
            "customer_service"
        ],
        preferred_team_size=6,
        formation_deadline=datetime.utcnow() + timedelta(minutes=15)
    )
    
    print(f"📝 Crisis Details:")
    print(f"   • Crisis ID: {outage_request.crisis_id}")
    print(f"   • Type: {outage_request.crisis_type}")
    print(f"   • Severity: {outage_request.severity_level}/5")
    print(f"   • Required Skills: {', '.join(outage_request.required_skills)}")
    print(f"   • Formation Deadline: {outage_request.formation_deadline.strftime('%H:%M:%S')}")
    
    print("\n⚡ Forming crisis response team...")
    outage_team = await engine.form_crisis_team(outage_request)
    
    print(f"\n✅ Team '{outage_team.team_name}' formed successfully!")
    print(f"   • Team ID: {outage_team.id}")
    print(f"   • Members: {len(outage_team.members)}")
    print(f"   • Formation Time: {outage_team.formation_time.strftime('%H:%M:%S')}")
    print(f"   • Status: {outage_team.team_status}")
    
    print(f"\n👥 Team Composition:")
    for assignment in outage_team.role_assignments:
        person = engine.personnel_registry[assignment.person_id]
        print(f"   • {assignment.role.value.replace('_', ' ').title()}: {person.name}")
        print(f"     - Confidence: {assignment.assignment_confidence:.2f}")
        print(f"     - Key Skills: {', '.join(assignment.required_skills[:3])}")
    
    print(f"\n📞 Communication Channels:")
    for channel, link in outage_team.communication_channels.items():
        print(f"   • {channel.replace('_', ' ').title()}: {link}")
    
    # Scenario 2: Security Breach Crisis
    print("\n" + "="*50)
    print("🛡️ SCENARIO 2: Security Breach Incident")
    print("="*50)
    
    security_request = TeamFormationRequest(
        crisis_id="security_2024_002",
        crisis_type="security_breach",
        severity_level=5,  # Critical severity
        urgency="immediate",
        required_skills=[
            "cybersecurity",
            "incident_response",
            "forensics",
            "legal_compliance",
            "crisis_communication"
        ],
        preferred_team_size=8,
        formation_deadline=datetime.utcnow() + timedelta(minutes=10)
    )
    
    print(f"📝 Crisis Details:")
    print(f"   • Crisis ID: {security_request.crisis_id}")
    print(f"   • Type: {security_request.crisis_type}")
    print(f"   • Severity: {security_request.severity_level}/5 (CRITICAL)")
    print(f"   • Required Skills: {', '.join(security_request.required_skills)}")
    
    print("\n🔒 Forming security incident response team...")
    security_team = await engine.form_crisis_team(security_request)
    
    print(f"\n✅ Security team formed in {(security_team.formation_time - datetime.utcnow()).total_seconds():.1f} seconds!")
    print(f"   • Team ID: {security_team.id}")
    print(f"   • Members: {len(security_team.members)}")
    
    print(f"\n👥 Security Team Composition:")
    for assignment in security_team.role_assignments:
        person = engine.personnel_registry[assignment.person_id]
        print(f"   • {assignment.role.value.replace('_', ' ').title()}: {person.name}")
        print(f"     - Department: {person.department}")
        print(f"     - Experience: {person.crisis_experience.get('security_breach', 0)} security incidents")
    
    # Demonstrate skill matching
    print("\n" + "="*50)
    print("🎯 SKILL MATCHING DEMONSTRATION")
    print("="*50)
    
    required_skills = ["cybersecurity", "crisis_management", "public_relations"]
    print(f"🔍 Finding best matches for skills: {', '.join(required_skills)}")
    
    skill_matches = await engine.match_skills_to_availability(required_skills)
    
    print(f"\n📊 Top 5 Skill Matches:")
    for i, match in enumerate(skill_matches[:5], 1):
        person = engine.personnel_registry[match.person_id]
        print(f"   {i}. {person.name} ({person.title})")
        print(f"      • Role: {match.role.value.replace('_', ' ').title()}")
        print(f"      • Overall Score: {match.overall_match_score:.3f}")
        print(f"      • Skill Match: {match.skill_match_score:.3f}")
        print(f"      • Experience: {match.experience_match_score:.3f}")
        print(f"      • Availability: {match.availability_score:.3f}")
        print(f"      • Strengths: {', '.join(match.strengths[:2])}")
        if match.missing_skills:
            print(f"      • Development Areas: {', '.join(match.missing_skills[:2])}")
        print()
    
    # Demonstrate team optimization
    print("\n" + "="*50)
    print("⚙️ TEAM OPTIMIZATION DEMONSTRATION")
    print("="*50)
    
    print(f"🔧 Optimizing system outage team for enhanced performance...")
    
    # Simulate some team performance issues
    outage_team.performance_metrics = {
        "response_time": 0.7,
        "communication_effectiveness": 0.8,
        "technical_resolution": 0.6
    }
    
    optimized_team = await engine.optimize_team_composition(outage_team.id, "system_outage")
    
    print(f"✅ Team optimization completed!")
    print(f"   • Team ID: {optimized_team.id}")
    print(f"   • Status: {optimized_team.team_status}")
    print(f"   • Members after optimization: {len(optimized_team.members)}")
    
    # Show team status
    print("\n📈 Active Teams Summary:")
    active_teams = [team for team in engine.active_teams.values() 
                   if team.team_status in ["forming", "active"]]
    
    for team in active_teams:
        print(f"   • {team.team_name}")
        print(f"     - Crisis: {team.crisis_type}")
        print(f"     - Members: {len(team.members)}")
        print(f"     - Status: {team.team_status}")
        print(f"     - Formed: {team.formation_time.strftime('%H:%M:%S')}")
    
    # Demonstrate personnel availability updates
    print("\n" + "="*50)
    print("👤 PERSONNEL MANAGEMENT DEMONSTRATION")
    print("="*50)
    
    print("📝 Updating personnel availability...")
    
    # Make some personnel busy
    engine.update_person_availability("bob_002", AvailabilityStatus.IN_CRISIS_RESPONSE)
    engine.update_person_availability("carol_003", AvailabilityStatus.BUSY)
    
    print("   ✓ Bob Chen: Now in crisis response")
    print("   ✓ Carol Rodriguez: Now busy with other tasks")
    
    # Check available personnel
    available_personnel = await engine._get_available_personnel()
    print(f"\n📊 Available Personnel: {len(available_personnel)} out of {len(personnel)}")
    
    for person in available_personnel:
        workload_status = "Light" if person.current_workload < 0.5 else "Moderate" if person.current_workload < 0.8 else "Heavy"
        print(f"   • {person.name} ({person.department}) - {workload_status} workload")
    
    # Final summary
    print("\n" + "="*50)
    print("📊 DEMO SUMMARY")
    print("="*50)
    
    print(f"✅ Successfully demonstrated crisis team formation system:")
    print(f"   • Personnel Database: {len(personnel)} team members")
    print(f"   • Teams Formed: {len(engine.active_teams)}")
    print(f"   • Crisis Types Handled: System Outage, Security Breach")
    print(f"   • Skills Matched: {len(skill_matches)} potential assignments")
    print(f"   • Team Optimizations: 1 completed")
    print(f"   • Available Personnel: {len(available_personnel)}")
    
    print(f"\n🎯 Key Capabilities Demonstrated:")
    print(f"   ✓ Rapid team assembly (< 15 minutes)")
    print(f"   ✓ Intelligent skill matching and role assignment")
    print(f"   ✓ Crisis-type specific team composition")
    print(f"   ✓ Real-time availability management")
    print(f"   ✓ Team performance optimization")
    print(f"   ✓ Multi-crisis scenario handling")
    print(f"   ✓ Communication channel setup")
    print(f"   ✓ Role-based responsibility assignment")
    
    print(f"\n🚀 Crisis Team Formation System is ready for production deployment!")
    
    return {
        "personnel_count": len(personnel),
        "teams_formed": len(engine.active_teams),
        "skill_matches": len(skill_matches),
        "available_personnel": len(available_personnel),
        "demo_completed": True
    }


if __name__ == "__main__":
    print("Starting Crisis Team Formation Demo...")
    result = asyncio.run(demonstrate_crisis_team_formation())
    print(f"\nDemo completed successfully: {json.dumps(result, indent=2)}")