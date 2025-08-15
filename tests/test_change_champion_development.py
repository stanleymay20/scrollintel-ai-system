"""
Tests for Change Champion Development Engine

Test suite for change champion identification, development, and network management.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.change_champion_development_engine import ChangeChampionDevelopmentEngine
from scrollintel.models.change_champion_models import (
    ChangeChampionProfile, ChampionLevel, ChampionRole, ChangeCapability,
    ChampionDevelopmentProgram, ChampionNetwork, NetworkStatus,
    ChampionPerformanceMetrics, NetworkCoordinationPlan
)


class TestChangeChampionDevelopmentEngine:
    """Test cases for Change Champion Development Engine"""
    
    @pytest.fixture
    def champion_engine(self):
        """Create champion development engine instance"""
        return ChangeChampionDevelopmentEngine()
    
    @pytest.fixture
    def sample_employee_data(self):
        """Sample employee data for testing"""
        return [
            {
                "id": "emp_001",
                "name": "Sarah Johnson",
                "role": "Senior Manager",
                "department": "Operations",
                "organization_id": "org_123",
                "capabilities": {
                    "change_advocacy": 85,
                    "influence_building": 78,
                    "communication": 90,
                    "training_delivery": 70,
                    "resistance_management": 65,
                    "network_building": 80,
                    "coaching_mentoring": 75
                },
                "network_size": 25,
                "cross_department_connections": 4,
                "credibility_score": 88,
                "cultural_alignment_score": 85,
                "experience": [
                    "Led digital transformation project",
                    "Mentored junior staff",
                    "Facilitated change workshops"
                ]
            },
            {
                "id": "emp_002",
                "name": "Michael Chen",
                "role": "Team Lead",
                "department": "IT",
                "organization_id": "org_123",
                "capabilities": {
                    "change_advocacy": 70,
                    "influence_building": 65,
                    "communication": 75,
                    "training_delivery": 60,
                    "resistance_management": 55,
                    "network_building": 68,
                    "coaching_mentoring": 62
                },
                "network_size": 18,
                "cross_department_connections": 2,
                "credibility_score": 75,
                "cultural_alignment_score": 80,
                "experience": [
                    "Participated in agile transformation",
                    "Led team through system migration"
                ]
            },
            {
                "id": "emp_003",
                "name": "Lisa Rodriguez",
                "role": "HR Specialist",
                "department": "HR",
                "organization_id": "org_123",
                "capabilities": {
                    "change_advocacy": 60,
                    "influence_building": 55,
                    "communication": 85,
                    "training_delivery": 80,
                    "resistance_management": 70,
                    "network_building": 65,
                    "coaching_mentoring": 78
                },
                "network_size": 20,
                "cross_department_connections": 5,
                "credibility_score": 82,
                "cultural_alignment_score": 90,
                "experience": [
                    "Designed training programs",
                    "Facilitated culture change initiatives"
                ]
            }
        ]
    
    def test_identify_potential_champions_success(self, champion_engine, sample_employee_data):
        """Test successful identification of potential change champions"""
        candidates = champion_engine.identify_potential_champions(
            organization_id="org_123",
            employee_data=sample_employee_data,
            criteria_type="standard",
            target_count=2
        )
        
        # Verify results
        assert len(candidates) <= 2  # Respects target count
        assert all(candidate["champion_score"] >= 70 for candidate in candidates)
        
        # Check candidate structure
        for candidate in candidates:
            assert "employee_id" in candidate
            assert "name" in candidate
            assert "role" in candidate
            assert "department" in candidate
            assert "champion_score" in candidate
            assert "strengths" in candidate
            assert "development_areas" in candidate
            assert "recommended_level" in candidate
            assert "recommended_roles" in candidate
            
            # Verify score is reasonable
            assert 0 <= candidate["champion_score"] <= 100
            assert isinstance(candidate["strengths"], list)
            assert isinstance(candidate["development_areas"], list)
            assert isinstance(candidate["recommended_roles"], list)
    
    def test_identify_champions_with_senior_criteria(self, champion_engine, sample_employee_data):
        """Test champion identification with senior criteria"""
        candidates = champion_engine.identify_potential_champions(
            organization_id="org_123",
            employee_data=sample_employee_data,
            criteria_type="senior"
        )
        
        # Senior criteria should be more selective
        senior_candidates = [c for c in candidates if c["recommended_level"] == ChampionLevel.SENIOR]
        
        # Verify senior candidates meet higher standards
        for candidate in senior_candidates:
            assert candidate["champion_score"] >= 75  # Higher threshold for senior
    
    def test_evaluate_champion_potential(self, champion_engine):
        """Test individual champion potential evaluation"""
        employee = {
            "capabilities": {
                "change_advocacy": 80,
                "influence_building": 75,
                "communication": 85,
                "cultural_sensitivity": 70
            },
            "network_size": 20,
            "cross_department_connections": 3,
            "credibility_score": 85,
            "cultural_alignment_score": 80,
            "experience": ["Led change project", "Mentored others"]
        }
        
        criteria = champion_engine.identification_criteria["standard"]
        score = champion_engine._evaluate_champion_potential(employee, criteria)
        
        assert 0 <= score <= 100
        assert isinstance(score, float)
    
    def test_create_champion_profile(self, champion_engine):
        """Test champion profile creation"""
        employee_data = {
            "id": "emp_001",
            "name": "Sarah Johnson",
            "role": "Senior Manager",
            "department": "Operations",
            "organization_id": "org_123"
        }
        
        champion_assessment = {
            "capabilities": {
                "change_advocacy": 85,
                "influence_building": 78,
                "communication": 90,
                "training_delivery": 70,
                "resistance_management": 65,
                "network_building": 80,
                "feedback_collection": 72,
                "coaching_mentoring": 75,
                "project_coordination": 68,
                "cultural_sensitivity": 82
            },
            "influence_network": ["emp_002", "emp_003", "emp_004"],
            "credibility_score": 88,
            "engagement_score": 85,
            "availability_score": 75,
            "motivation_score": 90,
            "cultural_fit_score": 85,
            "change_experience": ["Digital transformation", "Process improvement"],
        }
        
        profile = champion_engine.create_champion_profile(employee_data, champion_assessment)
        
        # Verify profile structure
        assert isinstance(profile, ChangeChampionProfile)
        assert profile.employee_id == "emp_001"
        assert profile.name == "Sarah Johnson"
        assert profile.role == "Senior Manager"
        assert profile.department == "Operations"
        assert profile.organization_id == "org_123"
        assert profile.champion_level in ChampionLevel
        assert len(profile.champion_roles) > 0
        assert len(profile.capabilities) == len(ChangeCapability)
        assert profile.credibility_score == 88
        assert profile.engagement_score == 85
        assert profile.status == "active"
    
    def test_design_development_program(self, champion_engine):
        """Test development program design"""
        # Create sample champion profiles
        champions = []
        for i in range(3):
            capabilities = {cap: 60 + i * 10 for cap in ChangeCapability}
            
            profile = ChangeChampionProfile(
                id=f"champ_{i}",
                employee_id=f"emp_{i}",
                name=f"Champion {i}",
                role="Manager",
                department="Operations",
                organization_id="org_123",
                champion_level=ChampionLevel.DEVELOPING,
                champion_roles=[ChampionRole.ADVOCATE],
                capabilities=capabilities,
                influence_network=[],
                credibility_score=70,
                engagement_score=75,
                availability_score=80,
                motivation_score=85,
                cultural_fit_score=80,
                change_experience=[],
                training_completed=[],
                certifications=[],
                mentorship_relationships=[],
                success_metrics={}
            )
            champions.append(profile)
        
        program_objectives = [
            "Develop change advocacy skills",
            "Build influence networks",
            "Improve communication effectiveness"
        ]
        
        constraints = {
            "budget": "medium",
            "timeline": "3 months",
            "delivery_preference": "blended"
        }
        
        program = champion_engine.design_development_program(
            champions=champions,
            program_objectives=program_objectives,
            constraints=constraints
        )
        
        # Verify program structure
        assert isinstance(program, ChampionDevelopmentProgram)
        assert program.name is not None
        assert program.description is not None
        assert program.target_level in ChampionLevel
        assert len(program.target_roles) > 0
        assert program.duration_weeks > 0
        assert len(program.learning_modules) > 0
        assert len(program.practical_assignments) > 0
        assert len(program.success_criteria) > 0
        assert len(program.resources_required) > 0
    
    def test_create_champion_network(self, champion_engine):
        """Test champion network creation"""
        # Create sample champion profiles
        champions = []
        for i in range(5):
            capabilities = {cap: 70 + i * 5 for cap in ChangeCapability}
            
            profile = ChangeChampionProfile(
                id=f"champ_{i}",
                employee_id=f"emp_{i}",
                name=f"Champion {i}",
                role="Manager" if i == 0 else "Team Lead",
                department=f"Dept_{i % 3}",
                organization_id="org_123",
                champion_level=ChampionLevel.SENIOR if i == 0 else ChampionLevel.ACTIVE,
                champion_roles=[ChampionRole.ADVOCATE, ChampionRole.FACILITATOR],
                capabilities=capabilities,
                influence_network=[],
                credibility_score=75 + i * 5,
                engagement_score=80,
                availability_score=85,
                motivation_score=90,
                cultural_fit_score=85,
                change_experience=[],
                training_completed=[],
                certifications=[],
                mentorship_relationships=[],
                success_metrics={}
            )
            champions.append(profile)
        
        objectives = [
            "Support organizational transformation",
            "Build change capability across departments",
            "Facilitate knowledge sharing"
        ]
        
        network = champion_engine.create_champion_network(
            champions=champions,
            network_type="cross_functional",
            objectives=objectives
        )
        
        # Verify network structure
        assert isinstance(network, ChampionNetwork)
        assert network.name is not None
        assert network.organization_id == "org_123"
        assert network.network_type == "cross_functional"
        assert len(network.champions) == 5
        assert network.network_lead is not None
        assert network.network_status == NetworkStatus.FORMING
        assert len(network.objectives) == 3
        assert len(network.success_metrics) > 0
        assert len(network.communication_channels) > 0
    
    def test_plan_network_coordination(self, champion_engine):
        """Test network coordination planning"""
        # Create mock network
        network = ChampionNetwork(
            id="network_123",
            name="Test Network",
            organization_id="org_123",
            network_type="cross_functional",
            champions=["champ_1", "champ_2", "champ_3"],
            network_lead="champ_1",
            coordinators=["champ_2"],
            coverage_areas=["Operations", "IT", "HR"],
            network_status=NetworkStatus.PERFORMING,
            formation_date=datetime.now(),
            objectives=["Support transformation", "Build capability"],
            success_metrics=["Engagement > 80%"],
            communication_channels=["Monthly meetings"],
            meeting_schedule="Monthly",
            governance_structure={"lead": "champ_1"},
            performance_metrics={}
        )
        
        key_initiatives = [
            "Digital transformation support",
            "Culture change facilitation",
            "Training program delivery"
        ]
        
        plan = champion_engine.plan_network_coordination(
            network=network,
            coordination_period="Q1_2024",
            key_initiatives=key_initiatives
        )
        
        # Verify coordination plan
        assert isinstance(plan, NetworkCoordinationPlan)
        assert plan.network_id == "network_123"
        assert plan.coordination_period == "Q1_2024"
        assert len(plan.key_initiatives) == 3
        assert "resource_allocation" in plan.__dict__
        assert "communication_strategy" in plan.__dict__
        assert len(plan.training_schedule) > 0
        assert len(plan.performance_targets) > 0
        assert len(plan.success_metrics) > 0
    
    def test_measure_champion_performance(self, champion_engine):
        """Test champion performance measurement"""
        performance_data = {
            "change_initiatives_supported": 5,
            "training_sessions_delivered": 8,
            "employees_influenced": 45,
            "resistance_cases_resolved": 3,
            "feedback_sessions_conducted": 12,
            "network_engagement_score": 88,
            "peer_rating": 85,
            "manager_rating": 90,
            "change_success_contribution": 82,
            "knowledge_sharing_score": 78,
            "mentorship_effectiveness": 85,
            "innovation_contributions": 2,
            "cultural_alignment_score": 87,
            "recognition_received": ["Employee of the Month"],
            "development_areas": ["Project coordination"]
        }
        
        metrics = champion_engine.measure_champion_performance(
            champion_id="champ_123",
            measurement_period="Q1_2024",
            performance_data=performance_data
        )
        
        # Verify performance metrics
        assert isinstance(metrics, ChampionPerformanceMetrics)
        assert metrics.champion_id == "champ_123"
        assert metrics.measurement_period == "Q1_2024"
        assert metrics.change_initiatives_supported == 5
        assert metrics.training_sessions_delivered == 8
        assert metrics.employees_influenced == 45
        assert metrics.resistance_cases_resolved == 3
        assert metrics.network_engagement_score == 88
        assert metrics.peer_rating == 85
        assert metrics.manager_rating == 90
        assert 0 <= metrics.overall_performance_score <= 100
        assert len(metrics.recognition_received) == 1
        assert len(metrics.development_areas) == 1
    
    def test_calculate_overall_performance_score(self, champion_engine):
        """Test overall performance score calculation"""
        performance_data = {
            "change_initiatives_supported": 5,
            "training_sessions_delivered": 10,
            "employees_influenced": 50,
            "resistance_cases_resolved": 8,
            "network_engagement_score": 85,
            "peer_rating": 80,
            "manager_rating": 88,
            "change_success_contribution": 90
        }
        
        score = champion_engine._calculate_overall_performance_score(performance_data)
        
        assert 0 <= score <= 100
        assert isinstance(score, float)
    
    def test_recommend_champion_level(self, champion_engine):
        """Test champion level recommendation"""
        # Test different score ranges
        assert champion_engine._recommend_champion_level(95) == ChampionLevel.SENIOR
        assert champion_engine._recommend_champion_level(85) == ChampionLevel.ACTIVE
        assert champion_engine._recommend_champion_level(75) == ChampionLevel.DEVELOPING
        assert champion_engine._recommend_champion_level(65) == ChampionLevel.EMERGING
    
    def test_recommend_champion_roles(self, champion_engine):
        """Test champion role recommendations"""
        employee = {
            "capabilities": {
                "change_advocacy": 80,
                "communication": 85,
                "training_delivery": 75,
                "coaching_mentoring": 78,
                "project_coordination": 72
            }
        }
        
        roles = champion_engine._recommend_champion_roles(employee, 82)
        
        assert len(roles) > 0
        assert all(isinstance(role, ChampionRole) for role in roles)
        
        # Should include advocate for high change advocacy
        assert ChampionRole.ADVOCATE in roles
        
        # Should include facilitator for high communication
        assert ChampionRole.FACILITATOR in roles
    
    def test_identify_champion_strengths(self, champion_engine):
        """Test champion strengths identification"""
        employee = {
            "capabilities": {
                "change_advocacy": 85,
                "communication": 90,
                "influence_building": 75
            },
            "network_size": 25,
            "credibility_score": 88
        }
        
        criteria = champion_engine.identification_criteria["standard"]
        strengths = champion_engine._identify_champion_strengths(employee, criteria)
        
        assert isinstance(strengths, list)
        assert len(strengths) > 0
        assert any("communication" in strength.lower() for strength in strengths)
        assert any("network" in strength.lower() for strength in strengths)
    
    def test_identify_development_areas(self, champion_engine):
        """Test development areas identification"""
        employee = {
            "capabilities": {
                "change_advocacy": 65,  # Just above minimum
                "influence_building": 58,  # Below minimum + 20
                "communication": 70
            },
            "cross_department_connections": 2  # Below threshold
        }
        
        criteria = champion_engine.identification_criteria["standard"]
        development_areas = champion_engine._identify_development_areas(employee, criteria)
        
        assert isinstance(development_areas, list)
        assert len(development_areas) > 0
        assert any("influence" in area.lower() for area in development_areas)
        assert any("cross-departmental" in area.lower() for area in development_areas)
    
    def test_create_specialized_module(self, champion_engine):
        """Test specialized learning module creation"""
        module = champion_engine._create_specialized_module(ChangeCapability.RESISTANCE_MANAGEMENT)
        
        assert module.title is not None
        assert module.description is not None
        assert ChangeCapability.RESISTANCE_MANAGEMENT in module.target_capabilities
        assert module.duration_hours > 0
        assert len(module.learning_objectives) > 0
        assert len(module.completion_criteria) > 0
    
    def test_calculate_resource_requirements(self, champion_engine):
        """Test resource requirements calculation"""
        champions = [Mock() for _ in range(5)]  # 5 mock champions
        modules = [
            Mock(duration_hours=8, delivery_method="in_person"),
            Mock(duration_hours=12, delivery_method="virtual"),
            Mock(duration_hours=6, delivery_method="blended")
        ]
        
        requirements = champion_engine._calculate_resource_requirements(champions, modules)
        
        assert isinstance(requirements, list)
        assert len(requirements) > 0
        assert any("training budget" in req.lower() for req in requirements)
        assert any("trainer resources" in req.lower() for req in requirements)
    
    def test_error_handling(self, champion_engine):
        """Test error handling in champion development engine"""
        # Test with invalid criteria type
        with pytest.raises(ValueError):
            champion_engine.identify_potential_champions(
                organization_id="org_123",
                employee_data=[],
                criteria_type="invalid_criteria"
            )
        
        # Test with empty employee data
        candidates = champion_engine.identify_potential_champions(
            organization_id="org_123",
            employee_data=[],
            criteria_type="standard"
        )
        assert len(candidates) == 0
    
    def test_capability_weights_initialization(self, champion_engine):
        """Test capability weights initialization"""
        weights = champion_engine.capability_weights
        
        assert len(weights) == len(ChangeCapability)
        assert all(0 <= weight <= 1 for weight in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 0.01  # Should sum to approximately 1
    
    def test_identification_criteria_initialization(self, champion_engine):
        """Test identification criteria initialization"""
        criteria = champion_engine.identification_criteria
        
        assert "standard" in criteria
        assert "senior" in criteria
        
        standard_criteria = criteria["standard"]
        assert len(standard_criteria.required_capabilities) > 0
        assert len(standard_criteria.minimum_scores) > 0
        assert len(standard_criteria.weight_factors) > 0
    
    def test_development_programs_initialization(self, champion_engine):
        """Test development programs initialization"""
        programs = champion_engine.development_programs
        
        assert "foundation" in programs
        assert "advanced" in programs
        
        foundation_program = programs["foundation"]
        assert foundation_program.target_level == ChampionLevel.DEVELOPING
        assert len(foundation_program.learning_modules) > 0
        assert len(foundation_program.practical_assignments) > 0