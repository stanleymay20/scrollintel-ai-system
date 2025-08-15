"""
Tests for Crisis Team Formation Engine

Test suite for crisis team formation, optimization, and skill matching functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.crisis_team_formation_engine import CrisisTeamFormationEngine
from scrollintel.models.team_coordination_models import (
    Person, TeamFormationRequest, CrisisTeam, Skill, SkillLevel,
    AvailabilityStatus, TeamRole, RoleAssignment
)


class TestCrisisTeamFormationEngine:
    """Test cases for crisis team formation engine"""
    
    @pytest.fixture
    def engine(self):
        """Create engine instance for testing"""
        return CrisisTeamFormationEngine()
    
    @pytest.fixture
    def sample_personnel(self):
        """Create sample personnel for testing"""
        personnel = []
        
        # Crisis leader
        leader = Person(
            id="person_1",
            name="Alice Johnson",
            email="alice@company.com",
            department="Operations",
            title="Operations Manager",
            skills=[
                Skill(name="crisis_management", level=SkillLevel.EXPERT, years_experience=5),
                Skill(name="leadership", level=SkillLevel.ADVANCED, years_experience=7),
                Skill(name="communication", level=SkillLevel.ADVANCED, years_experience=6)
            ],
            availability_status=AvailabilityStatus.AVAILABLE,
            current_workload=0.3,
            crisis_experience={"crisis_leader": 8, "system_outage": 5}
        )
        personnel.append(leader)
        
        # Technical lead
        tech_lead = Person(
            id="person_2",
            name="Bob Smith",
            email="bob@company.com",
            department="Engineering",
            title="Senior Engineer",
            skills=[
                Skill(name="system_administration", level=SkillLevel.EXPERT, years_experience=8),
                Skill(name="troubleshooting", level=SkillLevel.ADVANCED, years_experience=6),
                Skill(name="technical_communication", level=SkillLevel.INTERMEDIATE, years_experience=4)
            ],
            availability_status=AvailabilityStatus.AVAILABLE,
            current_workload=0.5,
            crisis_experience={"technical_lead": 6, "system_outage": 10}
        )
        personnel.append(tech_lead)
        
        # Communications lead
        comm_lead = Person(
            id="person_3",
            name="Carol Davis",
            email="carol@company.com",
            department="Marketing",
            title="Communications Director",
            skills=[
                Skill(name="crisis_communication", level=SkillLevel.EXPERT, years_experience=6),
                Skill(name="public_relations", level=SkillLevel.ADVANCED, years_experience=8),
                Skill(name="stakeholder_management", level=SkillLevel.ADVANCED, years_experience=7)
            ],
            availability_status=AvailabilityStatus.AVAILABLE,
            current_workload=0.4,
            crisis_experience={"communications_lead": 5, "reputation_damage": 3}
        )
        personnel.append(comm_lead)
        
        # Security specialist
        security_lead = Person(
            id="person_4",
            name="David Wilson",
            email="david@company.com",
            department="Security",
            title="Security Analyst",
            skills=[
                Skill(name="cybersecurity", level=SkillLevel.EXPERT, years_experience=7),
                Skill(name="incident_response", level=SkillLevel.ADVANCED, years_experience=5),
                Skill(name="forensics", level=SkillLevel.INTERMEDIATE, years_experience=3)
            ],
            availability_status=AvailabilityStatus.AVAILABLE,
            current_workload=0.6,
            crisis_experience={"security_lead": 4, "security_breach": 8}
        )
        personnel.append(security_lead)
        
        return personnel
    
    @pytest.fixture
    def sample_request(self):
        """Create sample team formation request"""
        return TeamFormationRequest(
            crisis_id="crisis_123",
            crisis_type="system_outage",
            severity_level=3,
            urgency="high",
            required_skills=["system_administration", "crisis_management", "communication"],
            preferred_team_size=5,
            formation_deadline=datetime.utcnow() + timedelta(minutes=30)
        )
    
    def test_engine_initialization(self, engine):
        """Test engine initializes correctly"""
        assert engine is not None
        assert isinstance(engine.personnel_registry, dict)
        assert isinstance(engine.active_teams, dict)
        assert len(engine.personnel_registry) == 0
        assert len(engine.active_teams) == 0
    
    def test_add_personnel(self, engine, sample_personnel):
        """Test adding personnel to registry"""
        person = sample_personnel[0]
        engine.add_person(person)
        
        assert person.id in engine.personnel_registry
        assert engine.personnel_registry[person.id] == person
    
    def test_update_personnel_availability(self, engine, sample_personnel):
        """Test updating personnel availability"""
        person = sample_personnel[0]
        engine.add_person(person)
        
        # Update availability
        engine.update_person_availability(person.id, AvailabilityStatus.BUSY)
        
        updated_person = engine.personnel_registry[person.id]
        assert updated_person.availability_status == AvailabilityStatus.BUSY
    
    @pytest.mark.asyncio
    async def test_get_available_personnel(self, engine, sample_personnel):
        """Test getting available personnel"""
        # Add personnel to registry
        for person in sample_personnel:
            engine.add_person(person)
        
        # Make one person unavailable
        engine.update_person_availability(sample_personnel[0].id, AvailabilityStatus.UNAVAILABLE)
        
        available = await engine._get_available_personnel()
        
        # Should have 3 available (4 total - 1 unavailable)
        assert len(available) == 3
        assert all(p.availability_status == AvailabilityStatus.AVAILABLE for p in available)
        assert all(p.current_workload < 0.8 for p in available)
    
    @pytest.mark.asyncio
    async def test_calculate_skill_match(self, engine, sample_personnel):
        """Test skill matching calculation"""
        person = sample_personnel[0]  # Alice - crisis management expert
        role = TeamRole.CRISIS_LEADER
        required_skills = ["crisis_management", "leadership"]
        
        match = await engine._calculate_skill_match(person, role, required_skills)
        
        assert match.person_id == person.id
        assert match.role == role
        assert match.skill_match_score > 0.7  # Should be high for expert (adjusted for missing skills)
        assert match.overall_match_score > 0.7  # Should be good overall match
        assert "crisis_management" in match.strengths
    
    @pytest.mark.asyncio
    async def test_form_crisis_team(self, engine, sample_personnel, sample_request):
        """Test complete crisis team formation"""
        # Add personnel to registry
        for person in sample_personnel:
            engine.add_person(person)
        
        # Form team
        team = await engine.form_crisis_team(sample_request)
        
        assert team is not None
        assert team.crisis_id == sample_request.crisis_id
        assert team.crisis_type == sample_request.crisis_type
        assert len(team.members) > 0
        assert len(team.role_assignments) > 0
        assert team.team_status == "forming"
        assert team.id in engine.active_teams
        
        # Check that team has required roles
        assigned_roles = [assignment.role for assignment in team.role_assignments]
        assert TeamRole.CRISIS_LEADER in assigned_roles or team.team_lead_id
    
    @pytest.mark.asyncio
    async def test_match_skills_to_availability(self, engine, sample_personnel):
        """Test skill matching to availability"""
        # Add personnel to registry
        for person in sample_personnel:
            engine.add_person(person)
        
        required_skills = ["system_administration", "cybersecurity"]
        matches = await engine.match_skills_to_availability(required_skills)
        
        assert len(matches) > 0
        # Should find matches for technical and security personnel
        person_ids = [match.person_id for match in matches]
        assert "person_2" in person_ids  # Bob - system admin
        assert "person_4" in person_ids  # David - security
    
    def test_get_team_composition_system_outage(self, engine):
        """Test team composition for system outage"""
        composition = engine._get_team_composition("system_outage", 3)
        
        assert composition.crisis_type == "system_outage"
        assert TeamRole.CRISIS_LEADER in composition.required_roles
        assert TeamRole.TECHNICAL_LEAD in composition.required_roles
        assert TeamRole.COMMUNICATIONS_LEAD in composition.required_roles
        assert composition.team_size_range[0] >= 4
        assert "system_administration" in composition.skill_requirements
    
    def test_get_team_composition_security_breach(self, engine):
        """Test team composition for security breach"""
        composition = engine._get_team_composition("security_breach", 4)
        
        assert composition.crisis_type == "security_breach"
        assert TeamRole.SECURITY_LEAD in composition.required_roles
        assert TeamRole.LEGAL_ADVISOR in composition.required_roles
        assert "cybersecurity" in composition.skill_requirements
        assert composition.skill_requirements["cybersecurity"] == SkillLevel.EXPERT
    
    def test_get_team_composition_high_severity(self, engine):
        """Test team composition adjusts for high severity"""
        low_severity = engine._get_team_composition("system_outage", 2)
        high_severity = engine._get_team_composition("system_outage", 5)
        
        # High severity should have larger team size
        assert high_severity.team_size_range[0] > low_severity.team_size_range[0]
        assert high_severity.team_size_range[1] > low_severity.team_size_range[1]
        
        # High severity should have additional roles
        assert len(high_severity.required_roles) >= len(low_severity.required_roles)
    
    def test_get_role_skill_requirements(self, engine):
        """Test role skill requirements"""
        crisis_leader_skills = engine._get_role_skill_requirements(TeamRole.CRISIS_LEADER)
        
        assert "crisis_management" in crisis_leader_skills
        assert "leadership" in crisis_leader_skills
        assert crisis_leader_skills["crisis_management"] == SkillLevel.EXPERT
        
        tech_lead_skills = engine._get_role_skill_requirements(TeamRole.TECHNICAL_LEAD)
        assert "system_administration" in tech_lead_skills
        assert tech_lead_skills["system_administration"] == SkillLevel.EXPERT
    
    def test_get_role_responsibilities(self, engine):
        """Test role responsibilities"""
        crisis_leader_resp = engine._get_role_responsibilities(TeamRole.CRISIS_LEADER)
        
        assert len(crisis_leader_resp) > 0
        assert any("coordination" in resp.lower() for resp in crisis_leader_resp)
        assert any("decision" in resp.lower() for resp in crisis_leader_resp)
        
        tech_lead_resp = engine._get_role_responsibilities(TeamRole.TECHNICAL_LEAD)
        assert any("technical" in resp.lower() for resp in tech_lead_resp)
    
    def test_find_person_skill(self, engine, sample_personnel):
        """Test finding specific skill in person"""
        person = sample_personnel[0]  # Alice with crisis_management skill
        
        skill = engine._find_person_skill(person, "crisis_management")
        assert skill is not None
        assert skill.name == "crisis_management"
        assert skill.level == SkillLevel.EXPERT
        
        # Test non-existent skill
        no_skill = engine._find_person_skill(person, "nonexistent_skill")
        assert no_skill is None
    
    def test_calculate_skill_level_score(self, engine):
        """Test skill level scoring"""
        # Exact match
        score = engine._calculate_skill_level_score(SkillLevel.EXPERT, SkillLevel.EXPERT)
        assert score == 1.0
        
        # Exceeds requirement
        score = engine._calculate_skill_level_score(SkillLevel.MASTER, SkillLevel.ADVANCED)
        assert score == 1.0
        
        # Below requirement
        score = engine._calculate_skill_level_score(SkillLevel.INTERMEDIATE, SkillLevel.EXPERT)
        assert score == 0.5  # 2/4
    
    def test_person_has_skill(self, engine, sample_personnel):
        """Test checking if person has required skill level"""
        person = sample_personnel[0]  # Alice with expert crisis_management
        
        # Has skill at required level
        assert engine._person_has_skill(person, "crisis_management", SkillLevel.EXPERT)
        assert engine._person_has_skill(person, "crisis_management", SkillLevel.ADVANCED)
        
        # Doesn't have skill at required level
        assert not engine._person_has_skill(person, "crisis_management", SkillLevel.MASTER)
        assert not engine._person_has_skill(person, "nonexistent_skill", SkillLevel.BEGINNER)
    
    def test_get_role_priority(self, engine):
        """Test role priority ordering"""
        roles = [TeamRole.CUSTOMER_LIAISON, TeamRole.CRISIS_LEADER, TeamRole.TECHNICAL_LEAD]
        prioritized = engine._get_role_priority(roles)
        
        # Crisis leader should come first
        assert prioritized[0] == TeamRole.CRISIS_LEADER
        # Technical lead should come before customer liaison
        assert prioritized.index(TeamRole.TECHNICAL_LEAD) < prioritized.index(TeamRole.CUSTOMER_LIAISON)
    
    def test_get_leadership_score(self, engine, sample_personnel):
        """Test leadership score calculation"""
        # Add personnel to registry
        for person in sample_personnel:
            engine.add_person(person)
        
        # Alice has leadership skill and crisis experience
        alice_score = engine._get_leadership_score("person_1")
        
        # Bob has no leadership skill
        bob_score = engine._get_leadership_score("person_2")
        
        assert alice_score > bob_score
        assert alice_score > 0.5  # Should be reasonably high
    
    @pytest.mark.asyncio
    async def test_optimize_team_composition(self, engine, sample_personnel, sample_request):
        """Test team composition optimization"""
        # Add personnel and form initial team
        for person in sample_personnel:
            engine.add_person(person)
        
        team = await engine.form_crisis_team(sample_request)
        
        # Optimize the team
        optimized_team = await engine.optimize_team_composition(team.id, "system_outage")
        
        assert optimized_team is not None
        assert optimized_team.id == team.id
        # Team should still be functional after optimization
        assert len(optimized_team.members) > 0
    
    def test_get_team_by_id(self, engine, sample_personnel, sample_request):
        """Test retrieving team by ID"""
        # Add personnel and form team
        for person in sample_personnel:
            engine.add_person(person)
        
        # Form team using asyncio.run since we're not in async context
        team = asyncio.run(engine.form_crisis_team(sample_request))
        
        # Retrieve team
        retrieved_team = engine.get_team_by_id(team.id)
        assert retrieved_team is not None
        assert retrieved_team.id == team.id
        
        # Test non-existent team
        no_team = engine.get_team_by_id("nonexistent_id")
        assert no_team is None
    
    def test_deactivate_team(self, engine, sample_personnel, sample_request):
        """Test team deactivation"""
        # Add personnel and form team
        for person in sample_personnel:
            engine.add_person(person)
        
        team = asyncio.run(engine.form_crisis_team(sample_request))
        
        # Deactivate team
        engine.deactivate_team(team.id)
        
        deactivated_team = engine.get_team_by_id(team.id)
        assert deactivated_team.team_status == "disbanded"
        assert deactivated_team.deactivation_time is not None
    
    @pytest.mark.asyncio
    async def test_team_formation_with_insufficient_personnel(self, engine):
        """Test team formation with insufficient personnel"""
        # Add only one person
        person = Person(
            id="person_1",
            name="Solo Worker",
            skills=[Skill(name="basic_skill", level=SkillLevel.BEGINNER, years_experience=1)],
            availability_status=AvailabilityStatus.AVAILABLE
        )
        engine.add_person(person)
        
        request = TeamFormationRequest(
            crisis_id="crisis_456",
            crisis_type="system_outage",
            severity_level=4,
            required_skills=["system_administration", "crisis_management"]
        )
        
        # Should still form a team, even if not optimal
        team = await engine.form_crisis_team(request)
        assert team is not None
        assert len(team.members) >= 1
    
    @pytest.mark.asyncio
    async def test_team_formation_with_overloaded_personnel(self, engine, sample_personnel):
        """Test team formation when personnel are overloaded"""
        # Make all personnel overloaded
        for person in sample_personnel:
            person.current_workload = 0.95  # Very high workload
            engine.add_person(person)
        
        request = TeamFormationRequest(
            crisis_id="crisis_789",
            crisis_type="security_breach",
            severity_level=5
        )
        
        # Should still form team but with lower availability scores
        team = await engine.form_crisis_team(request)
        assert team is not None
        # Team might be smaller due to availability constraints
        assert len(team.members) > 0


@pytest.mark.asyncio
async def test_integration_full_team_lifecycle():
    """Integration test for complete team lifecycle"""
    engine = CrisisTeamFormationEngine()
    
    # Create comprehensive personnel
    personnel = [
        Person(
            id=f"person_{i}",
            name=f"Person {i}",
            email=f"person{i}@company.com",
            department="Operations" if i % 2 == 0 else "Engineering",
            skills=[
                Skill(name="crisis_management", level=SkillLevel.ADVANCED, years_experience=3),
                Skill(name="system_administration", level=SkillLevel.INTERMEDIATE, years_experience=2),
                Skill(name="communication", level=SkillLevel.INTERMEDIATE, years_experience=2)
            ],
            availability_status=AvailabilityStatus.AVAILABLE,
            current_workload=0.3 + (i * 0.1),
            crisis_experience={"system_outage": i + 1}
        )
        for i in range(6)
    ]
    
    # Add personnel
    for person in personnel:
        engine.add_person(person)
    
    # Form team
    request = TeamFormationRequest(
        crisis_id="integration_test_crisis",
        crisis_type="system_outage",
        severity_level=3,
        required_skills=["system_administration", "crisis_management"]
    )
    
    team = await engine.form_crisis_team(request)
    
    # Verify team formation
    assert team is not None
    assert len(team.members) > 0
    assert team.crisis_id == request.crisis_id
    
    # Activate team
    team.team_status = "active"
    team.activation_time = datetime.utcnow()
    
    # Optimize team
    optimized_team = await engine.optimize_team_composition(team.id, "system_outage")
    assert optimized_team is not None
    
    # Deactivate team
    engine.deactivate_team(team.id)
    final_team = engine.get_team_by_id(team.id)
    assert final_team.team_status == "disbanded"
    
    print(f"Integration test completed successfully. Team {team.id} went through full lifecycle.")