"""
Tests for Influence Strategy Development Engine
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.influence_strategy_engine import InfluenceStrategyEngine
from scrollintel.models.influence_strategy_models import (
    InfluenceStrategy, InfluenceTactic, InfluenceTarget, InfluenceType,
    InfluenceContext, InfluenceObjective, InfluenceExecution,
    InfluenceEffectivenessMetrics, InfluenceOptimization
)
from scrollintel.models.stakeholder_influence_models import Stakeholder


class TestInfluenceStrategyEngine:
    """Test suite for InfluenceStrategyEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create engine instance for testing"""
        return InfluenceStrategyEngine()
    
    @pytest.fixture
    def sample_stakeholders(self):
        """Create sample stakeholders for testing"""
        from scrollintel.models.stakeholder_influence_models import (
            StakeholderType, InfluenceLevel, CommunicationStyle, 
            Background, Priority, DecisionPattern
        )
        
        return [
            Stakeholder(
                id="stakeholder_1",
                name="John Board",
                title="Board Chair",
                organization="TestCorp",
                stakeholder_type=StakeholderType.BOARD_MEMBER,
                background=Background(
                    industry_experience=["technology"],
                    functional_expertise=["governance"],
                    education=["MBA"],
                    previous_roles=["CEO"],
                    achievements=["Leadership Award"]
                ),
                influence_level=InfluenceLevel.CRITICAL,
                communication_style=CommunicationStyle.ANALYTICAL,
                decision_pattern=DecisionPattern(
                    decision_style="analytical",
                    key_factors=["data_quality"],
                    typical_concerns=["market_volatility", "regulatory_compliance"],
                    influence_tactics=["rational_persuasion"]
                ),
                priorities=[
                    Priority(name="company_growth", description="Growth", importance=0.9, category="strategic"),
                    Priority(name="risk_management", description="Risk", importance=0.8, category="governance")
                ],
                relationships=[],
                contact_preferences={"email": "john@company.com"}
            ),
            Stakeholder(
                id="stakeholder_2",
                name="Sarah Executive",
                title="CEO",
                organization="TestCorp",
                stakeholder_type=StakeholderType.EXECUTIVE,
                background=Background(
                    industry_experience=["technology"],
                    functional_expertise=["leadership"],
                    education=["MBA"],
                    previous_roles=["VP"],
                    achievements=["Innovation Award"]
                ),
                influence_level=InfluenceLevel.CRITICAL,
                communication_style=CommunicationStyle.VISIONARY,
                decision_pattern=DecisionPattern(
                    decision_style="collaborative",
                    key_factors=["team_input"],
                    typical_concerns=["competitive_pressure", "talent_retention"],
                    influence_tactics=["consultation"]
                ),
                priorities=[
                    Priority(name="innovation", description="Innovation", importance=0.95, category="strategic"),
                    Priority(name="market_leadership", description="Leadership", importance=0.9, category="market")
                ],
                relationships=[],
                contact_preferences={"email": "sarah@company.com"}
            )
        ]
    
    @pytest.fixture
    def sample_influence_targets(self):
        """Create sample influence targets for testing"""
        return [
            InfluenceTarget(
                stakeholder_id="target_1",
                name="John Board",
                role="Board Chair",
                influence_level=0.9,
                decision_making_style="analytical",
                communication_preferences=["data_driven", "formal"],
                key_motivators=["company_growth", "risk_management"],
                concerns=["market_volatility", "regulatory_compliance"],
                relationship_strength=0.7
            ),
            InfluenceTarget(
                stakeholder_id="target_2",
                name="Sarah Executive",
                role="CEO",
                influence_level=0.95,
                decision_making_style="collaborative",
                communication_preferences=["consultative", "strategic"],
                key_motivators=["innovation", "market_leadership"],
                concerns=["competitive_pressure", "talent_retention"],
                relationship_strength=0.8
            )
        ]
    
    def test_engine_initialization(self, engine):
        """Test engine initializes correctly"""
        assert engine is not None
        assert hasattr(engine, 'tactic_library')
        assert hasattr(engine, 'effectiveness_history')
        assert hasattr(engine, 'optimization_rules')
        assert len(engine.tactic_library) > 0
    
    def test_develop_influence_strategy(self, engine, sample_stakeholders):
        """Test influence strategy development"""
        strategy = engine.develop_influence_strategy(
            objective=InfluenceObjective.BUILD_SUPPORT,
            target_stakeholders=sample_stakeholders,
            context=InfluenceContext.BOARD_MEETING
        )
        
        assert isinstance(strategy, InfluenceStrategy)
        assert strategy.id is not None
        assert strategy.objective == InfluenceObjective.BUILD_SUPPORT
        assert strategy.context == InfluenceContext.BOARD_MEETING
        assert len(strategy.target_stakeholders) == 2
        assert len(strategy.primary_tactics) > 0
        assert 0 <= strategy.expected_effectiveness <= 1
        assert len(strategy.success_metrics) > 0
        assert len(strategy.resource_requirements) > 0
    
    def test_develop_strategy_with_constraints(self, engine, sample_stakeholders):
        """Test strategy development with constraints"""
        constraints = {
            "time_limit": "1_week",
            "resource_limit": "low",
            "formality_level": "high"
        }
        
        strategy = engine.develop_influence_strategy(
            objective=InfluenceObjective.GAIN_CONSENSUS,
            target_stakeholders=sample_stakeholders,
            context=InfluenceContext.COMMITTEE_MEETING,
            constraints=constraints
        )
        
        assert isinstance(strategy, InfluenceStrategy)
        assert strategy.objective == InfluenceObjective.GAIN_CONSENSUS
        assert strategy.context == InfluenceContext.COMMITTEE_MEETING
    
    def test_select_optimal_tactics(self, engine, sample_influence_targets):
        """Test optimal tactic selection"""
        primary_tactics, secondary_tactics = engine.select_optimal_tactics(
            objective=InfluenceObjective.BUILD_SUPPORT,
            targets=sample_influence_targets,
            context=InfluenceContext.BOARD_MEETING
        )
        
        assert isinstance(primary_tactics, list)
        assert isinstance(secondary_tactics, list)
        assert len(primary_tactics) > 0
        assert all(isinstance(tactic, InfluenceTactic) for tactic in primary_tactics)
        assert all(isinstance(tactic, InfluenceTactic) for tactic in secondary_tactics)
        
        # Check that tactics are suitable for context
        for tactic in primary_tactics:
            assert InfluenceContext.BOARD_MEETING in tactic.context_suitability
    
    def test_select_tactics_with_constraints(self, engine, sample_influence_targets):
        """Test tactic selection with constraints"""
        constraints = {"exclude_pressure": True, "formal_only": True}
        
        primary_tactics, secondary_tactics = engine.select_optimal_tactics(
            objective=InfluenceObjective.CHANGE_OPINION,
            targets=sample_influence_targets,
            context=InfluenceContext.ONE_ON_ONE,
            constraints=constraints
        )
        
        assert len(primary_tactics) > 0
        assert len(secondary_tactics) >= 0
    
    def test_measure_influence_effectiveness(self, engine, sample_influence_targets):
        """Test influence effectiveness measurement"""
        # Create sample strategy
        strategy = InfluenceStrategy(
            id="test_strategy",
            name="Test Strategy",
            objective=InfluenceObjective.BUILD_SUPPORT,
            target_stakeholders=sample_influence_targets,
            primary_tactics=engine.tactic_library[:2],
            secondary_tactics=engine.tactic_library[2:3],
            context=InfluenceContext.BOARD_MEETING,
            timeline={},
            success_metrics=["support_level", "consensus"],
            risk_mitigation={},
            resource_requirements=[],
            expected_effectiveness=0.8
        )
        
        # Create sample execution
        execution = InfluenceExecution(
            id="test_execution",
            strategy_id="test_strategy",
            execution_date=datetime.now(),
            context_details={"meeting_type": "board_meeting"},
            tactics_used=["rational_persuasion_data"],
            target_responses={"target_1": "positive", "target_2": "neutral"},
            immediate_outcomes=["increased_support", "questions_raised"],
            effectiveness_rating=0.75,
            lessons_learned=["need_more_data"],
            follow_up_actions=["schedule_follow_up"]
        )
        
        metrics = engine.measure_influence_effectiveness(execution, strategy)
        
        assert isinstance(metrics, InfluenceEffectivenessMetrics)
        assert metrics.strategy_id == "test_strategy"
        assert metrics.execution_id == "test_execution"
        assert 0 <= metrics.objective_achievement <= 1
        assert 0 <= metrics.consensus_level <= 1
        assert 0 <= metrics.support_gained <= 1
        assert len(metrics.stakeholder_satisfaction) > 0
    
    def test_optimize_influence_strategy(self, engine, sample_influence_targets):
        """Test influence strategy optimization"""
        # Create sample strategy
        strategy = InfluenceStrategy(
            id="test_strategy",
            name="Test Strategy",
            objective=InfluenceObjective.BUILD_SUPPORT,
            target_stakeholders=sample_influence_targets,
            primary_tactics=engine.tactic_library[:2],
            secondary_tactics=engine.tactic_library[2:3],
            context=InfluenceContext.BOARD_MEETING,
            timeline={},
            success_metrics=["support_level"],
            risk_mitigation={},
            resource_requirements=[],
            expected_effectiveness=0.6
        )
        
        # Create sample metrics
        metrics = [
            InfluenceEffectivenessMetrics(
                strategy_id="test_strategy",
                execution_id="exec_1",
                objective_achievement=0.6,
                stakeholder_satisfaction={"target_1": 0.7, "target_2": 0.5},
                relationship_impact={"target_1": 0.05, "target_2": 0.02},
                consensus_level=0.5,
                support_gained=0.6,
                opposition_reduced=0.4,
                long_term_relationship_health=0.7
            ),
            InfluenceEffectivenessMetrics(
                strategy_id="test_strategy",
                execution_id="exec_2",
                objective_achievement=0.65,
                stakeholder_satisfaction={"target_1": 0.75, "target_2": 0.6},
                relationship_impact={"target_1": 0.03, "target_2": 0.04},
                consensus_level=0.55,
                support_gained=0.65,
                opposition_reduced=0.45,
                long_term_relationship_health=0.72
            )
        ]
        
        optimization = engine.optimize_influence_strategy(strategy, metrics)
        
        assert isinstance(optimization, InfluenceOptimization)
        assert optimization.strategy_id == "test_strategy"
        assert 0 <= optimization.current_effectiveness <= 1
        assert isinstance(optimization.optimization_opportunities, list)
        assert isinstance(optimization.recommended_tactic_changes, list)
        assert isinstance(optimization.timing_adjustments, list)
        assert 0 <= optimization.confidence_level <= 1
    
    def test_convert_stakeholders_to_targets(self, engine, sample_stakeholders):
        """Test conversion of stakeholders to influence targets"""
        targets = engine._convert_to_influence_targets(sample_stakeholders)
        
        assert len(targets) == len(sample_stakeholders)
        assert all(isinstance(target, InfluenceTarget) for target in targets)
        
        # Check data mapping
        for i, target in enumerate(targets):
            stakeholder = sample_stakeholders[i]
            assert target.stakeholder_id == stakeholder.id
            assert target.name == stakeholder.name
            assert target.role == stakeholder.title
            assert isinstance(target.influence_level, float)
            assert 0 <= target.influence_level <= 1
    
    def test_analyze_stakeholder_characteristics(self, engine, sample_influence_targets):
        """Test stakeholder characteristics analysis"""
        analysis = engine._analyze_stakeholder_characteristics(sample_influence_targets)
        
        assert isinstance(analysis, dict)
        assert "dominant_decision_styles" in analysis
        assert "common_motivators" in analysis
        assert "shared_concerns" in analysis
        assert "relationship_strengths" in analysis
        assert "influence_distribution" in analysis
        
        # Check analysis content
        assert isinstance(analysis["dominant_decision_styles"], dict)
        assert isinstance(analysis["common_motivators"], list)
    
    def test_calculate_expected_effectiveness(self, engine):
        """Test expected effectiveness calculation"""
        primary_tactics = engine.tactic_library[:2]
        secondary_tactics = engine.tactic_library[2:3]
        targets = [
            InfluenceTarget(
                stakeholder_id="target_1",
                name="Test Target",
                role="Board Member",
                influence_level=0.8,
                decision_making_style="analytical",
                communication_preferences=["data_driven"],
                key_motivators=["growth"],
                concerns=["risk"],
                relationship_strength=0.7
            )
        ]
        
        effectiveness = engine._calculate_expected_effectiveness(
            primary_tactics, secondary_tactics, targets, InfluenceContext.BOARD_MEETING
        )
        
        assert isinstance(effectiveness, float)
        assert 0 <= effectiveness <= 1
    
    def test_score_tactic_for_target(self, engine):
        """Test tactic scoring for specific target"""
        tactic = engine.tactic_library[0]
        target = InfluenceTarget(
            stakeholder_id="target_1",
            name="Test Target",
            role="Board Member",
            influence_level=0.8,
            decision_making_style="analytical",
            communication_preferences=["data_driven"],
            key_motivators=["growth"],
            concerns=["risk"],
            relationship_strength=0.7
        )
        
        score = engine._score_tactic_for_target(
            tactic, target, InfluenceObjective.BUILD_SUPPORT
        )
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
    
    def test_create_influence_timeline(self, engine):
        """Test influence timeline creation"""
        primary_tactics = engine.tactic_library[:2]
        secondary_tactics = engine.tactic_library[2:3]
        
        timeline = engine._create_influence_timeline(primary_tactics, secondary_tactics)
        
        assert isinstance(timeline, dict)
        assert "preparation_start" in timeline
        assert "preparation_complete" in timeline
        assert "primary_execution_start" in timeline
        assert "final_assessment" in timeline
        
        # Check timeline ordering
        prep_start = timeline["preparation_start"]
        prep_complete = timeline["preparation_complete"]
        assert prep_complete > prep_start
    
    def test_identify_risk_mitigation(self, engine, sample_influence_targets):
        """Test risk identification and mitigation"""
        tactics = engine.tactic_library[:2]
        
        risk_mitigation = engine._identify_risk_mitigation(
            tactics, sample_influence_targets, InfluenceContext.BOARD_MEETING
        )
        
        assert isinstance(risk_mitigation, dict)
        # Should have mitigation strategies for identified risks
        for risk, mitigation in risk_mitigation.items():
            assert isinstance(risk, str)
            assert isinstance(mitigation, str)
            assert len(mitigation) > 0
    
    def test_effectiveness_measurement_components(self, engine):
        """Test individual effectiveness measurement components"""
        execution = InfluenceExecution(
            id="test_execution",
            strategy_id="test_strategy",
            execution_date=datetime.now(),
            context_details={},
            tactics_used=["rational_persuasion"],
            target_responses={"target_1": "positive"},
            immediate_outcomes=["support_gained"],
            effectiveness_rating=0.8,
            lessons_learned=[],
            follow_up_actions=[]
        )
        
        # Test objective achievement calculation
        achievement = engine._calculate_objective_achievement(execution, None)
        assert isinstance(achievement, float)
        assert 0 <= achievement <= 1
        
        # Test consensus level calculation
        consensus = engine._calculate_consensus_level(execution)
        assert isinstance(consensus, float)
        assert 0 <= consensus <= 1
        
        # Test support measurement
        support = engine._measure_support_gained(execution)
        assert isinstance(support, float)
        assert 0 <= support <= 1
    
    def test_optimization_components(self, engine, sample_influence_targets):
        """Test optimization recommendation components"""
        strategy = InfluenceStrategy(
            id="test_strategy",
            name="Test Strategy",
            objective=InfluenceObjective.BUILD_SUPPORT,
            target_stakeholders=sample_influence_targets,
            primary_tactics=engine.tactic_library[:1],
            secondary_tactics=[],
            context=InfluenceContext.BOARD_MEETING,
            timeline={},
            success_metrics=[],
            risk_mitigation={},
            resource_requirements=[],
            expected_effectiveness=0.5
        )
        
        metrics = [
            InfluenceEffectivenessMetrics(
                strategy_id="test_strategy",
                execution_id="exec_1",
                objective_achievement=0.5,
                stakeholder_satisfaction={"target_1": 0.6},
                relationship_impact={"target_1": 0.02},
                consensus_level=0.4,
                support_gained=0.5,
                opposition_reduced=0.3,
                long_term_relationship_health=0.6
            )
        ]
        
        # Test opportunity identification
        opportunities = engine._identify_optimization_opportunities(strategy, metrics)
        assert isinstance(opportunities, list)
        
        # Test tactic change recommendations
        changes = engine._recommend_tactic_changes(strategy, metrics)
        assert isinstance(changes, list)
        
        # Test timing adjustments
        adjustments = engine._suggest_timing_adjustments(strategy, metrics)
        assert isinstance(adjustments, list)
    
    def test_error_handling(self, engine):
        """Test error handling in engine methods"""
        # Test with invalid objective
        with pytest.raises(Exception):
            engine.develop_influence_strategy(
                objective="invalid_objective",
                target_stakeholders=[],
                context=InfluenceContext.BOARD_MEETING
            )
        
        # Test with empty stakeholders
        with pytest.raises(Exception):
            engine.develop_influence_strategy(
                objective=InfluenceObjective.BUILD_SUPPORT,
                target_stakeholders=[],
                context=InfluenceContext.BOARD_MEETING
            )
    
    def test_tactic_library_completeness(self, engine):
        """Test that tactic library is properly initialized"""
        assert len(engine.tactic_library) >= 3
        
        for tactic in engine.tactic_library:
            assert isinstance(tactic, InfluenceTactic)
            assert tactic.id is not None
            assert tactic.name is not None
            assert isinstance(tactic.influence_type, InfluenceType)
            assert 0 <= tactic.effectiveness_score <= 1
            assert len(tactic.context_suitability) > 0
            assert len(tactic.required_preparation) > 0
    
    def test_optimization_rules_initialization(self, engine):
        """Test optimization rules are properly initialized"""
        rules = engine.optimization_rules
        
        assert isinstance(rules, dict)
        assert "effectiveness_threshold" in rules
        assert "relationship_impact_weight" in rules
        assert "objective_achievement_weight" in rules
        assert "consensus_weight" in rules
        assert "minimum_confidence" in rules
        
        # Check weight values sum appropriately
        weights = (
            rules["relationship_impact_weight"] +
            rules["objective_achievement_weight"] +
            rules["consensus_weight"]
        )
        assert abs(weights - 1.0) < 0.01  # Allow for floating point precision