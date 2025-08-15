"""
Tests for Cultural Vision Development Engine

Tests the cultural vision creation, alignment, and communication systems.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.cultural_vision_engine import CulturalVisionEngine
from scrollintel.models.cultural_vision_models import (
    VisionDevelopmentRequest, CulturalVision, StrategicObjective,
    VisionScope, StakeholderType, CulturalValue, AlignmentLevel
)


class TestCulturalVisionEngine:
    """Test cases for Cultural Vision Engine"""
    
    @pytest.fixture
    def vision_engine(self):
        """Create vision engine instance"""
        return CulturalVisionEngine()
    
    @pytest.fixture
    def sample_strategic_objectives(self):
        """Create sample strategic objectives"""
        return [
            StrategicObjective(
                id="obj1",
                title="Increase Innovation",
                description="Drive breakthrough innovation across all products",
                priority=1,
                target_date=datetime.now() + timedelta(days=365),
                success_metrics=["Innovation index", "New product launches"],
                cultural_requirements=["creativity", "risk-taking", "collaboration"]
            ),
            StrategicObjective(
                id="obj2",
                title="Improve Customer Satisfaction",
                description="Achieve industry-leading customer satisfaction",
                priority=2,
                target_date=datetime.now() + timedelta(days=180),
                success_metrics=["NPS score", "Customer retention"],
                cultural_requirements=["customer-focus", "quality", "responsiveness"]
            )
        ]
    
    @pytest.fixture
    def sample_vision_request(self, sample_strategic_objectives):
        """Create sample vision development request"""
        return VisionDevelopmentRequest(
            organization_id="org123",
            scope=VisionScope.ORGANIZATIONAL,
            strategic_objectives=sample_strategic_objectives,
            current_culture_assessment={
                "innovation_score": 0.6,
                "collaboration_score": 0.7,
                "customer_focus_score": 0.5
            },
            stakeholder_requirements={
                StakeholderType.EXECUTIVE: ["ROI focus", "strategic alignment"],
                StakeholderType.EMPLOYEE: ["career growth", "work-life balance"],
                StakeholderType.CUSTOMER: ["quality", "service excellence"]
            },
            constraints=["Budget limitations", "Timeline constraints"],
            timeline=datetime.now() + timedelta(days=180)
        )
    
    def test_develop_cultural_vision_success(self, vision_engine, sample_vision_request):
        """Test successful cultural vision development"""
        # Act
        result = vision_engine.develop_cultural_vision(sample_vision_request)
        
        # Assert
        assert result is not None
        assert result.vision is not None
        assert result.vision.organization_id == "org123"
        assert result.vision.scope == VisionScope.ORGANIZATIONAL
        assert len(result.vision.core_values) > 0
        assert result.vision.vision_statement != ""
        assert len(result.alignment_analysis) == 2  # Two objectives
        assert len(result.communication_strategies) == 3  # Three stakeholder types
        assert len(result.implementation_recommendations) > 0
        assert 0 <= result.success_probability <= 1
    
    def test_develop_core_values(self, vision_engine, sample_vision_request):
        """Test core values development"""
        # Act
        result = vision_engine.develop_cultural_vision(sample_vision_request)
        
        # Assert
        values = result.vision.core_values
        assert len(values) <= 7  # Max 7 core values
        assert all(isinstance(v, CulturalValue) for v in values)
        assert all(v.name and v.description for v in values)
        assert all(0 <= v.importance_score <= 1 for v in values)
        assert all(0 <= v.measurability <= 1 for v in values)
    
    def test_craft_vision_statement(self, vision_engine, sample_vision_request):
        """Test vision statement crafting"""
        # Act
        result = vision_engine.develop_cultural_vision(sample_vision_request)
        
        # Assert
        vision_statement = result.vision.vision_statement
        assert vision_statement is not None
        assert len(vision_statement) > 20  # Reasonable length
        assert any(value.name.lower() in vision_statement.lower() 
                  for value in result.vision.core_values)
    
    def test_align_with_strategic_objectives(self, vision_engine, sample_strategic_objectives):
        """Test strategic alignment analysis"""
        # Arrange
        vision = CulturalVision(
            id="vision1",
            organization_id="org123",
            title="Test Vision",
            vision_statement="We aspire to innovate and serve customers excellently",
            mission_alignment="Aligned with company mission",
            core_values=[
                CulturalValue(
                    name="Innovation",
                    description="Drive creative solutions",
                    behavioral_indicators=["Creative thinking", "Experimentation"],
                    importance_score=0.9,
                    measurability=0.8
                )
            ],
            scope=VisionScope.ORGANIZATIONAL,
            target_behaviors=["Innovative thinking", "Customer focus"],
            success_indicators=["Innovation metrics", "Customer satisfaction"],
            created_date=datetime.now(),
            target_implementation=datetime.now() + timedelta(days=180)
        )
        
        # Act
        alignments = vision_engine.align_with_strategic_objectives(vision, sample_strategic_objectives)
        
        # Assert
        assert len(alignments) == 2
        for alignment in alignments:
            assert alignment.vision_id == "vision1"
            assert alignment.objective_id in ["obj1", "obj2"]
            assert isinstance(alignment.alignment_level, AlignmentLevel)
            assert 0 <= alignment.alignment_score <= 1
            assert isinstance(alignment.supporting_evidence, list)
            assert isinstance(alignment.gaps_identified, list)
            assert isinstance(alignment.recommendations, list)
    
    def test_develop_stakeholder_buy_in_strategy(self, vision_engine):
        """Test stakeholder buy-in strategy development"""
        # Arrange
        vision = CulturalVision(
            id="vision1",
            organization_id="org123",
            title="Test Vision",
            vision_statement="We aspire to excellence through collaboration",
            mission_alignment="Aligned",
            core_values=[
                CulturalValue(
                    name="Excellence",
                    description="Strive for the best",
                    behavioral_indicators=["High standards"],
                    importance_score=0.9,
                    measurability=0.8
                )
            ],
            scope=VisionScope.ORGANIZATIONAL,
            target_behaviors=[],
            success_indicators=[],
            created_date=datetime.now(),
            target_implementation=datetime.now() + timedelta(days=180)
        )
        
        stakeholder_requirements = {
            StakeholderType.EXECUTIVE: ["Strategic alignment", "ROI"],
            StakeholderType.EMPLOYEE: ["Career development", "Recognition"]
        }
        
        # Act
        strategies = vision_engine.develop_stakeholder_buy_in_strategy(
            vision, stakeholder_requirements
        )
        
        # Assert
        assert len(strategies) == 2
        for strategy in strategies:
            assert strategy.vision_id == "vision1"
            assert strategy.target_audience in stakeholder_requirements.keys()
            assert len(strategy.key_messages) > 0
            assert len(strategy.communication_channels) > 0
            assert strategy.frequency is not None
            assert len(strategy.success_metrics) > 0
    
    def test_calculate_success_probability(self, vision_engine, sample_vision_request):
        """Test success probability calculation"""
        # Act
        result = vision_engine.develop_cultural_vision(sample_vision_request)
        
        # Assert
        assert 0 <= result.success_probability <= 1
        
        # Test with different scopes
        sample_vision_request.scope = VisionScope.TEAM
        team_result = vision_engine.develop_cultural_vision(sample_vision_request)
        
        sample_vision_request.scope = VisionScope.ORGANIZATIONAL
        org_result = vision_engine.develop_cultural_vision(sample_vision_request)
        
        # Team scope should have higher success probability
        assert team_result.success_probability >= org_result.success_probability
    
    def test_risk_assessment(self, vision_engine, sample_vision_request):
        """Test risk factor assessment"""
        # Act
        result = vision_engine.develop_cultural_vision(sample_vision_request)
        
        # Assert
        assert isinstance(result.risk_factors, list)
        assert len(result.risk_factors) > 0
        
        # Test with aggressive timeline
        sample_vision_request.timeline = datetime.now() + timedelta(days=30)
        aggressive_result = vision_engine.develop_cultural_vision(sample_vision_request)
        
        # Should identify timeline risk
        timeline_risks = [r for r in aggressive_result.risk_factors if "timeline" in r.lower()]
        assert len(timeline_risks) > 0
    
    def test_communication_channel_selection(self, vision_engine):
        """Test communication channel selection for different stakeholders"""
        # Test executive channels
        exec_channels = vision_engine._select_communication_channels(StakeholderType.EXECUTIVE)
        assert "Board presentations" in exec_channels
        assert "Executive briefings" in exec_channels
        
        # Test employee channels
        emp_channels = vision_engine._select_communication_channels(StakeholderType.EMPLOYEE)
        assert "All-hands meetings" in emp_channels
        assert "Team meetings" in emp_channels
        
        # Test customer channels
        cust_channels = vision_engine._select_communication_channels(StakeholderType.CUSTOMER)
        assert "Customer communications" in cust_channels
    
    def test_message_tailoring(self, vision_engine):
        """Test message tailoring for different stakeholders"""
        # Arrange
        vision = CulturalVision(
            id="vision1",
            organization_id="org123",
            title="Test Vision",
            vision_statement="We aspire to excellence",
            mission_alignment="Aligned",
            core_values=[
                CulturalValue(
                    name="Excellence",
                    description="Strive for the best",
                    behavioral_indicators=["High standards"],
                    importance_score=0.9,
                    measurability=0.8
                )
            ],
            scope=VisionScope.ORGANIZATIONAL,
            target_behaviors=[],
            success_indicators=[],
            created_date=datetime.now(),
            target_implementation=datetime.now() + timedelta(days=180)
        )
        
        # Test executive messages
        exec_messages = vision_engine._tailor_messages_for_stakeholder(
            vision, StakeholderType.EXECUTIVE, ["ROI", "Strategic alignment"]
        )
        assert any("ROI" in msg for msg in exec_messages)
        
        # Test employee messages
        emp_messages = vision_engine._tailor_messages_for_stakeholder(
            vision, StakeholderType.EMPLOYEE, ["Career growth", "Work environment"]
        )
        assert any("work experience" in msg.lower() for msg in emp_messages)
    
    def test_value_alignment_calculation(self, vision_engine):
        """Test value alignment calculation"""
        # Arrange
        values = [
            CulturalValue(
                name="Innovation",
                description="Drive innovation",
                behavioral_indicators=["Creative thinking", "Innovation mindset"],
                importance_score=0.9,
                measurability=0.8
            )
        ]
        
        objective = StrategicObjective(
            id="obj1",
            title="Increase Innovation",
            description="Drive innovation",
            priority=1,
            target_date=datetime.now() + timedelta(days=365),
            success_metrics=["Innovation index"],
            cultural_requirements=["creativity"]
        )
        
        # Act
        alignment_score = vision_engine._calculate_value_alignment(values, objective)
        
        # Assert
        assert 0 <= alignment_score <= 1
        assert alignment_score > 0  # Should have some alignment due to "innovation" keyword
    
    def test_vision_optimization(self, vision_engine):
        """Test vision optimization based on alignment"""
        # Arrange
        vision = CulturalVision(
            id="vision1",
            organization_id="org123",
            title="Test Vision",
            vision_statement="We aspire to excellence",
            mission_alignment="Aligned",
            core_values=[],
            scope=VisionScope.ORGANIZATIONAL,
            target_behaviors=[],
            success_indicators=[],
            created_date=datetime.now(),
            target_implementation=datetime.now() + timedelta(days=180)
        )
        
        # Create mock alignments with recommendations
        alignments = [
            Mock(alignment_score=0.5, recommendations=["Add innovation focus"]),
            Mock(alignment_score=0.8, recommendations=["Strengthen collaboration"])
        ]
        
        # Act
        vision_engine._optimize_vision_alignment(vision, alignments)
        
        # Assert
        assert vision.alignment_score == 0.65  # Average of 0.5 and 0.8
        assert "Add innovation focus" in vision.target_behaviors
        assert "Strengthen collaboration" in vision.target_behaviors
    
    def test_error_handling(self, vision_engine):
        """Test error handling in vision development"""
        # Test with invalid request
        invalid_request = VisionDevelopmentRequest(
            organization_id="",  # Invalid empty org ID
            scope=VisionScope.ORGANIZATIONAL,
            strategic_objectives=[],
            current_culture_assessment={},
            stakeholder_requirements={}
        )
        
        # Should handle gracefully and still return a result
        result = vision_engine.develop_cultural_vision(invalid_request)
        assert result is not None
        assert result.vision is not None
    
    def test_template_loading(self, vision_engine):
        """Test template and framework loading"""
        # Test vision templates
        templates = vision_engine._load_vision_templates()
        assert isinstance(templates, dict)
        assert len(templates) > 0
        
        # Test value frameworks
        frameworks = vision_engine._load_value_frameworks()
        assert isinstance(frameworks, dict)
        assert "foundational" in frameworks
        assert len(frameworks["foundational"]) > 0
        
        # Test alignment criteria
        criteria = vision_engine._load_alignment_criteria()
        assert isinstance(criteria, dict)
        assert all(0 <= v <= 1 for v in criteria.values())
        assert sum(criteria.values()) == 1.0  # Should sum to 1.0 for weighting


@pytest.mark.integration
class TestCulturalVisionIntegration:
    """Integration tests for cultural vision system"""
    
    def test_end_to_end_vision_development(self):
        """Test complete vision development workflow"""
        # Arrange
        engine = CulturalVisionEngine()
        
        request = VisionDevelopmentRequest(
            organization_id="test_org",
            scope=VisionScope.DEPARTMENTAL,
            strategic_objectives=[
                StrategicObjective(
                    id="obj1",
                    title="Digital Transformation",
                    description="Transform digital capabilities",
                    priority=1,
                    target_date=datetime.now() + timedelta(days=365),
                    success_metrics=["Digital maturity score"],
                    cultural_requirements=["innovation", "adaptability", "learning"]
                )
            ],
            current_culture_assessment={"digital_readiness": 0.4},
            stakeholder_requirements={
                StakeholderType.EXECUTIVE: ["Business value", "Risk management"],
                StakeholderType.EMPLOYEE: ["Skill development", "Support"]
            }
        )
        
        # Act
        result = engine.develop_cultural_vision(request)
        
        # Assert - Complete workflow validation
        assert result.vision.organization_id == "test_org"
        assert result.vision.scope == VisionScope.DEPARTMENTAL
        assert len(result.vision.core_values) > 0
        assert result.vision.vision_statement != ""
        assert len(result.alignment_analysis) == 1
        assert len(result.communication_strategies) == 2
        assert result.success_probability > 0
        
        # Validate alignment quality
        alignment = result.alignment_analysis[0]
        assert alignment.objective_id == "obj1"
        assert alignment.alignment_score > 0
        
        # Validate communication strategies
        exec_strategy = next(s for s in result.communication_strategies 
                           if s.target_audience == StakeholderType.EXECUTIVE)
        assert "Business value" in " ".join(exec_strategy.key_messages)
        
        emp_strategy = next(s for s in result.communication_strategies 
                          if s.target_audience == StakeholderType.EMPLOYEE)
        assert "Skill development" in " ".join(emp_strategy.key_messages) or \
               "development" in " ".join(emp_strategy.key_messages).lower()
    
    def test_vision_refinement_cycle(self):
        """Test iterative vision refinement"""
        # Arrange
        engine = CulturalVisionEngine()
        
        # Initial vision development
        initial_request = VisionDevelopmentRequest(
            organization_id="refine_org",
            scope=VisionScope.ORGANIZATIONAL,
            strategic_objectives=[
                StrategicObjective(
                    id="obj1",
                    title="Market Leadership",
                    description="Become market leader",
                    priority=1,
                    target_date=datetime.now() + timedelta(days=365),
                    success_metrics=["Market share"],
                    cultural_requirements=["competitiveness", "excellence"]
                )
            ],
            current_culture_assessment={"competitiveness": 0.3},
            stakeholder_requirements={
                StakeholderType.EXECUTIVE: ["Market position", "Competitive advantage"]
            }
        )
        
        # Act - Initial development
        initial_result = engine.develop_cultural_vision(initial_request)
        
        # Simulate refinement based on feedback
        refined_objectives = initial_request.strategic_objectives + [
            StrategicObjective(
                id="obj2",
                title="Customer Excellence",
                description="Deliver exceptional customer experience",
                priority=2,
                target_date=datetime.now() + timedelta(days=180),
                success_metrics=["Customer satisfaction"],
                cultural_requirements=["customer-focus", "service-excellence"]
            )
        ]
        
        refined_request = VisionDevelopmentRequest(
            organization_id="refine_org",
            scope=VisionScope.ORGANIZATIONAL,
            strategic_objectives=refined_objectives,
            current_culture_assessment={"competitiveness": 0.3, "customer_focus": 0.6},
            stakeholder_requirements={
                StakeholderType.EXECUTIVE: ["Market position", "Customer loyalty"],
                StakeholderType.CUSTOMER: ["Service quality", "Responsiveness"]
            }
        )
        
        # Act - Refined development
        refined_result = engine.develop_cultural_vision(refined_request)
        
        # Assert - Refinement improvements
        assert len(refined_result.alignment_analysis) == 2  # More objectives
        assert len(refined_result.communication_strategies) == 2  # More stakeholders
        
        # Should have customer-focused values in refined version
        customer_values = [v for v in refined_result.vision.core_values 
                          if "customer" in v.name.lower() or "service" in v.name.lower()]
        assert len(customer_values) > 0 or any("customer" in v.description.lower() 
                                               for v in refined_result.vision.core_values)