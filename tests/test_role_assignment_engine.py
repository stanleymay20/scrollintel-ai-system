"""
Tests for Role Assignment Engine
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from scrollintel.engines.role_assignment_engine import (
    RoleAssignmentEngine, Person, PersonSkill, RoleType, SkillLevel,
    RoleRequirement, RoleAssignment, AssignmentResult
)


class TestRoleAssignmentEngine:
    """Test cases for Role Assignment Engine"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = RoleAssignmentEngine()
        
        # Create test people
        self.test_people = [
            Person(
                id="person_1",
                name="Alice Johnson",
                current_availability=1.0,
                skills=[
                    PersonSkill("leadership", SkillLevel.EXPERT, 8, 0.9, True),
                    PersonSkill("decision_making", SkillLevel.ADVANCED, 6, 0.85, True),
                    PersonSkill("crisis_management", SkillLevel.EXPERT, 10, 0.95, True)
                ],
                preferred_roles=[RoleType.CRISIS_COMMANDER],
                stress_tolerance=0.95,
                leadership_experience=8,
                crisis_history=["system_outage_2023", "security_breach_2022"],
                current_workload=0.1
            ),
            Person(
                id="person_2",
                name="Bob Smith",
                current_availability=0.8,
                skills=[
                    PersonSkill("technical_expertise", SkillLevel.EXPERT, 12, 0.9, True),
                    PersonSkill("problem_solving", SkillLevel.ADVANCED, 10, 0.88, True),
                    PersonSkill("system_architecture", SkillLevel.EXPERT, 8, 0.92, False)
                ],
                preferred_roles=[RoleType.TECHNICAL_LEAD],
                stress_tolerance=0.85,
                leadership_experience=5,
                crisis_history=["system_outage_2023"],
                current_workload=0.2
            ),
            Person(
                id="person_3",
                name="Carol Davis",
                current_availability=0.9,
                skills=[
                    PersonSkill("communication", SkillLevel.ADVANCED, 7, 0.87, True),
                    PersonSkill("public_relations", SkillLevel.EXPERT, 9, 0.91, True),
                    PersonSkill("stakeholder_management", SkillLevel.ADVANCED, 6, 0.83, False)
                ],
                preferred_roles=[RoleType.COMMUNICATION_LEAD],
                stress_tolerance=0.82,
                leadership_experience=4,
                crisis_history=["pr_crisis_2022"],
                current_workload=0.15
            )
        ]
        
        self.test_roles = [
            RoleType.CRISIS_COMMANDER,
            RoleType.TECHNICAL_LEAD,
            RoleType.COMMUNICATION_LEAD
        ]
    
    def test_engine_initialization(self):
        """Test engine initializes correctly"""
        assert isinstance(self.engine, RoleAssignmentEngine)
        assert len(self.engine.role_definitions) > 0
        assert RoleType.CRISIS_COMMANDER in self.engine.role_definitions
        assert self.engine.assignment_history == []
    
    def test_assign_roles_basic(self):
        """Test basic role assignment functionality"""
        result = self.engine.assign_roles(
            crisis_id="test_crisis_001",
            available_people=self.test_people,
            required_roles=self.test_roles,
            crisis_severity=0.7
        )
        
        assert isinstance(result, AssignmentResult)
        assert len(result.assignments) > 0
        assert result.assignment_quality_score > 0.0
        assert isinstance(result.recommendations, list)
        assert isinstance(result.backup_assignments, dict)
    
    def test_assign_roles_optimal_matching(self):
        """Test that roles are assigned to best-matched people"""
        result = self.engine.assign_roles(
            crisis_id="test_crisis_002",
            available_people=self.test_people,
            required_roles=self.test_roles,
            crisis_severity=0.5
        )
        
        # Check that Alice (best leadership skills) gets Crisis Commander
        commander_assignment = next(
            (a for a in result.assignments if a.role_type == RoleType.CRISIS_COMMANDER),
            None
        )
        assert commander_assignment is not None
        assert commander_assignment.person_id == "person_1"
        
        # Check that Bob (best technical skills) gets Technical Lead
        tech_assignment = next(
            (a for a in result.assignments if a.role_type == RoleType.TECHNICAL_LEAD),
            None
        )
        assert tech_assignment is not None
        assert tech_assignment.person_id == "person_2"
    
    def test_calculate_person_role_score(self):
        """Test person-role compatibility scoring"""
        person = self.test_people[0]  # Alice
        role_req = self.engine.role_definitions[RoleType.CRISIS_COMMANDER]
        
        score = self.engine._calculate_person_role_score(person, role_req, 0.5)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.7  # Alice should score high for Crisis Commander
    
    def test_calculate_skill_match(self):
        """Test skill matching calculation"""
        person = self.test_people[0]  # Alice
        role_req = self.engine.role_definitions[RoleType.CRISIS_COMMANDER]
        
        skill_score = self.engine._calculate_skill_match(person, role_req)
        
        assert 0.0 <= skill_score <= 1.0
        assert skill_score > 0.8  # Alice has excellent crisis management skills
    
    def test_optimize_assignments_priority(self):
        """Test that high-priority roles are assigned first"""
        compatibility_matrix = {}
        
        # Create compatibility scores
        for person in self.test_people:
            for role in self.test_roles:
                compatibility_matrix[(person.id, role)] = 0.7
        
        assignments = self.engine._optimize_assignments(
            compatibility_matrix, self.test_people, self.test_roles
        )
        
        assert len(assignments) > 0
        
        # Crisis Commander should be assigned (highest priority)
        assigned_roles = {a.role_type for a in assignments}
        assert RoleType.CRISIS_COMMANDER in assigned_roles
    
    def test_create_role_assignment(self):
        """Test role assignment creation"""
        person = self.test_people[0]
        role = RoleType.CRISIS_COMMANDER
        confidence = 0.85
        
        assignment = self.engine._create_role_assignment(person, role, confidence)
        
        assert isinstance(assignment, RoleAssignment)
        assert assignment.person_id == person.id
        assert assignment.role_type == role
        assert assignment.assignment_confidence == confidence
        assert len(assignment.responsibilities) > 0
        assert isinstance(assignment.reporting_structure, dict)
        assert isinstance(assignment.assignment_time, datetime)
    
    def test_get_role_responsibilities(self):
        """Test role responsibility generation"""
        responsibilities = self.engine._get_role_responsibilities(RoleType.CRISIS_COMMANDER)
        
        assert isinstance(responsibilities, list)
        assert len(responsibilities) > 0
        assert all(isinstance(r, str) for r in responsibilities)
        assert "leadership" in responsibilities[0].lower()
    
    def test_determine_reporting_structure(self):
        """Test reporting structure determination"""
        structure = self.engine._determine_reporting_structure(RoleType.CRISIS_COMMANDER)
        
        assert isinstance(structure, dict)
        assert "reports_to" in structure
        assert "manages" in structure
    
    def test_generate_backup_assignments(self):
        """Test backup assignment generation"""
        # Create mock primary assignments
        primary_assignments = [
            RoleAssignment(
                person_id="person_1",
                role_type=RoleType.CRISIS_COMMANDER,
                assignment_confidence=0.9,
                responsibilities=[],
                reporting_structure={},
                assignment_time=datetime.now()
            )
        ]
        
        compatibility_matrix = {
            ("person_2", RoleType.CRISIS_COMMANDER): 0.7,
            ("person_3", RoleType.CRISIS_COMMANDER): 0.6
        }
        
        backups = self.engine._generate_backup_assignments(
            compatibility_matrix, primary_assignments, self.test_people
        )
        
        assert isinstance(backups, dict)
        assert RoleType.CRISIS_COMMANDER in backups
        assert len(backups[RoleType.CRISIS_COMMANDER]) <= 2
    
    def test_calculate_assignment_quality(self):
        """Test assignment quality calculation"""
        assignments = [
            RoleAssignment(
                person_id="person_1",
                role_type=RoleType.CRISIS_COMMANDER,
                assignment_confidence=0.9,
                responsibilities=[],
                reporting_structure={},
                assignment_time=datetime.now()
            ),
            RoleAssignment(
                person_id="person_2",
                role_type=RoleType.TECHNICAL_LEAD,
                assignment_confidence=0.8,
                responsibilities=[],
                reporting_structure={},
                assignment_time=datetime.now()
            )
        ]
        
        quality = self.engine._calculate_assignment_quality(assignments, {})
        
        assert 0.0 <= quality <= 1.0
        assert quality == 0.85  # Average of 0.9 and 0.8
    
    def test_generate_recommendations(self):
        """Test recommendation generation"""
        assignments = [
            RoleAssignment(
                person_id="person_1",
                role_type=RoleType.CRISIS_COMMANDER,
                assignment_confidence=0.6,  # Low confidence
                responsibilities=[],
                reporting_structure={},
                assignment_time=datetime.now()
            )
        ]
        
        recommendations = self.engine._generate_recommendations(
            assignments, self.test_people, self.test_roles, 0.6
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("training" in rec.lower() for rec in recommendations)
    
    def test_get_role_clarity_communication(self):
        """Test role clarity communication generation"""
        assignment = RoleAssignment(
            person_id="person_1",
            role_type=RoleType.CRISIS_COMMANDER,
            assignment_confidence=0.9,
            responsibilities=["Lead crisis response"],
            reporting_structure={"reports_to": "CEO"},
            assignment_time=datetime.now()
        )
        
        clarity = self.engine.get_role_clarity_communication(assignment)
        
        assert isinstance(clarity, dict)
        assert "assignment_summary" in clarity
        assert "responsibilities" in clarity
        assert "reporting_structure" in clarity
        assert "success_criteria" in clarity
        assert "communication_protocols" in clarity
        assert "escalation_procedures" in clarity
    
    def test_confirm_role_assignment(self):
        """Test role assignment confirmation"""
        result = self.engine.confirm_role_assignment("assignment_1", True)
        
        assert isinstance(result, dict)
        assert "assignment_confirmed" in result
        assert "confirmation_time" in result
        assert "next_steps" in result
        assert result["assignment_confirmed"] is True
    
    def test_confirm_role_assignment_declined(self):
        """Test role assignment confirmation when declined"""
        result = self.engine.confirm_role_assignment("assignment_1", False)
        
        assert isinstance(result, dict)
        assert result["assignment_confirmed"] is False
        assert "alternative" in result["next_steps"][0].lower()
    
    def test_empty_people_list(self):
        """Test handling of empty people list"""
        result = self.engine.assign_roles(
            crisis_id="test_crisis_003",
            available_people=[],
            required_roles=self.test_roles,
            crisis_severity=0.5
        )
        
        assert len(result.assignments) == 0
        assert len(result.unassigned_roles) == len(self.test_roles)
        assert result.assignment_quality_score == 0.0
    
    def test_empty_roles_list(self):
        """Test handling of empty roles list"""
        result = self.engine.assign_roles(
            crisis_id="test_crisis_004",
            available_people=self.test_people,
            required_roles=[],
            crisis_severity=0.5
        )
        
        assert len(result.assignments) == 0
        assert len(result.unassigned_roles) == 0
        assert result.assignment_quality_score == 0.0
    
    def test_high_crisis_severity_impact(self):
        """Test that high crisis severity affects scoring"""
        # Test with low severity
        result_low = self.engine.assign_roles(
            crisis_id="test_crisis_005a",
            available_people=self.test_people[:1],
            required_roles=[RoleType.CRISIS_COMMANDER],
            crisis_severity=0.1
        )
        
        # Test with high severity
        result_high = self.engine.assign_roles(
            crisis_id="test_crisis_005b",
            available_people=self.test_people[:1],
            required_roles=[RoleType.CRISIS_COMMANDER],
            crisis_severity=0.9
        )
        
        # High severity should result in higher confidence scores
        if result_low.assignments and result_high.assignments:
            assert result_high.assignments[0].assignment_confidence >= result_low.assignments[0].assignment_confidence
    
    def test_assignment_history_tracking(self):
        """Test that assignment history is tracked"""
        initial_count = len(self.engine.assignment_history)
        
        self.engine.assign_roles(
            crisis_id="test_crisis_006",
            available_people=self.test_people,
            required_roles=self.test_roles,
            crisis_severity=0.5
        )
        
        assert len(self.engine.assignment_history) > initial_count
    
    def test_person_workload_consideration(self):
        """Test that person workload is considered in assignments"""
        # Create person with high workload
        overloaded_person = Person(
            id="overloaded",
            name="Overloaded Person",
            current_availability=1.0,
            skills=[
                PersonSkill("leadership", SkillLevel.EXPERT, 10, 0.95, True),
                PersonSkill("decision_making", SkillLevel.EXPERT, 8, 0.9, True),
                PersonSkill("crisis_management", SkillLevel.EXPERT, 12, 0.98, True)
            ],
            preferred_roles=[RoleType.CRISIS_COMMANDER],
            stress_tolerance=0.95,
            leadership_experience=10,
            current_workload=0.9  # Very high workload
        )
        
        people_with_overloaded = self.test_people + [overloaded_person]
        
        result = self.engine.assign_roles(
            crisis_id="test_crisis_007",
            available_people=people_with_overloaded,
            required_roles=[RoleType.CRISIS_COMMANDER],
            crisis_severity=0.5
        )
        
        # Should prefer person with lower workload despite higher skills
        if result.assignments:
            assigned_person_id = result.assignments[0].person_id
            assert assigned_person_id != "overloaded"
    
    def test_role_preference_bonus(self):
        """Test that role preferences provide scoring bonus"""
        person_with_preference = self.test_people[0]  # Alice prefers Crisis Commander
        role_req = self.engine.role_definitions[RoleType.CRISIS_COMMANDER]
        
        # Calculate score for preferred role
        preferred_score = self.engine._calculate_person_role_score(
            person_with_preference, role_req, 0.5
        )
        
        # Create person without preference
        person_without_preference = Person(
            id="no_pref",
            name="No Preference",
            current_availability=person_with_preference.current_availability,
            skills=person_with_preference.skills,
            preferred_roles=[],  # No preferences
            stress_tolerance=person_with_preference.stress_tolerance,
            leadership_experience=person_with_preference.leadership_experience,
            current_workload=person_with_preference.current_workload
        )
        
        non_preferred_score = self.engine._calculate_person_role_score(
            person_without_preference, role_req, 0.5
        )
        
        # Preferred role should score higher
        assert preferred_score > non_preferred_score


@pytest.fixture
def sample_people():
    """Fixture providing sample people for testing"""
    return [
        Person(
            id="test_person_1",
            name="Test Person 1",
            current_availability=1.0,
            skills=[
                PersonSkill("leadership", SkillLevel.ADVANCED, 5, 0.8, True)
            ],
            preferred_roles=[RoleType.CRISIS_COMMANDER],
            stress_tolerance=0.8,
            leadership_experience=5
        )
    ]


@pytest.fixture
def sample_roles():
    """Fixture providing sample roles for testing"""
    return [RoleType.CRISIS_COMMANDER, RoleType.TECHNICAL_LEAD]


def test_integration_with_fixtures(sample_people, sample_roles):
    """Test integration using pytest fixtures"""
    engine = RoleAssignmentEngine()
    
    result = engine.assign_roles(
        crisis_id="fixture_test",
        available_people=sample_people,
        required_roles=sample_roles,
        crisis_severity=0.5
    )
    
    assert isinstance(result, AssignmentResult)
    assert len(result.assignments) <= len(sample_people)