"""
Tests for Decision Tree Engine
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from scrollintel.engines.decision_tree_engine import DecisionTreeEngine
from scrollintel.models.decision_tree_models import (
    CrisisType, SeverityLevel, DecisionNodeType, ConfidenceLevel
)


class TestDecisionTreeEngine:
    
    @pytest.fixture
    def engine(self):
        """Create a decision tree engine instance for testing"""
        return DecisionTreeEngine()
    
    def test_initialization(self, engine):
        """Test that the engine initializes with default trees"""
        assert len(engine.decision_trees) >= 3
        assert "system_outage_tree" in engine.decision_trees
        assert "security_breach_tree" in engine.decision_trees
        assert "financial_crisis_tree" in engine.decision_trees
    
    def test_find_applicable_trees_system_outage(self, engine):
        """Test finding applicable trees for system outage"""
        applicable_trees = engine.find_applicable_trees(
            CrisisType.SYSTEM_OUTAGE,
            SeverityLevel.HIGH
        )
        
        assert len(applicable_trees) >= 1
        assert applicable_trees[0].tree_id == "system_outage_tree"
        assert CrisisType.SYSTEM_OUTAGE in applicable_trees[0].crisis_types
    
    def test_find_applicable_trees_security_breach(self, engine):
        """Test finding applicable trees for security breach"""
        applicable_trees = engine.find_applicable_trees(
            CrisisType.SECURITY_BREACH,
            SeverityLevel.CRITICAL
        )
        
        assert len(applicable_trees) >= 1
        assert applicable_trees[0].tree_id == "security_breach_tree"
        assert CrisisType.SECURITY_BREACH in applicable_trees[0].crisis_types
    
    def test_find_applicable_trees_financial_crisis(self, engine):
        """Test finding applicable trees for financial crisis"""
        applicable_trees = engine.find_applicable_trees(
            CrisisType.FINANCIAL_CRISIS,
            SeverityLevel.HIGH
        )
        
        assert len(applicable_trees) >= 1
        assert applicable_trees[0].tree_id == "financial_crisis_tree"
        assert CrisisType.FINANCIAL_CRISIS in applicable_trees[0].crisis_types
    
    def test_find_applicable_trees_no_match(self, engine):
        """Test finding applicable trees when no match exists"""
        applicable_trees = engine.find_applicable_trees(
            CrisisType.NATURAL_DISASTER,
            SeverityLevel.LOW
        )
        
        assert len(applicable_trees) == 0
    
    def test_start_decision_path(self, engine):
        """Test starting a new decision path"""
        crisis_id = "test_crisis_001"
        tree_id = "system_outage_tree"
        
        path = engine.start_decision_path(crisis_id, tree_id)
        
        assert path.crisis_id == crisis_id
        assert path.tree_id == tree_id
        assert path.path_id in engine.active_paths
        assert len(path.nodes_traversed) == 0
        assert len(path.decisions_made) == 0
        assert path.start_time is not None
    
    def test_start_decision_path_invalid_tree(self, engine):
        """Test starting a decision path with invalid tree ID"""
        with pytest.raises(ValueError, match="Decision tree invalid_tree not found"):
            engine.start_decision_path("test_crisis", "invalid_tree")
    
    def test_evaluate_outage_scope_critical(self, engine):
        """Test evaluating outage scope as critical"""
        context = {
            "affected_systems": ["web", "api", "database", "cache"],
            "user_count": 5000,
            "business_impact": "high"
        }
        
        is_critical, confidence = engine._evaluate_outage_scope(context)
        
        assert is_critical is True
        assert confidence == 0.9
    
    def test_evaluate_outage_scope_non_critical(self, engine):
        """Test evaluating outage scope as non-critical"""
        context = {
            "affected_systems": ["cache"],
            "user_count": 100,
            "business_impact": "low"
        }
        
        is_critical, confidence = engine._evaluate_outage_scope(context)
        
        assert is_critical is False
        assert confidence == 0.9
    
    def test_evaluate_breach_severity_severe(self, engine):
        """Test evaluating breach severity as severe"""
        context = {
            "compromised_systems": ["web", "database"],
            "data_exposure": "pii",
            "attack_vector": "advanced_persistent_threat"
        }
        
        is_severe, confidence = engine._evaluate_breach_severity(context)
        
        assert is_severe is True
        assert confidence == 0.8
    
    def test_evaluate_breach_severity_non_severe(self, engine):
        """Test evaluating breach severity as non-severe"""
        context = {
            "compromised_systems": ["web"],
            "data_exposure": "none",
            "attack_vector": "script_kiddie"
        }
        
        is_severe, confidence = engine._evaluate_breach_severity(context)
        
        assert is_severe is False
        assert confidence == 0.8
    
    def test_evaluate_financial_impact_severe(self, engine):
        """Test evaluating financial impact as severe"""
        context = {
            "cash_reserves": 500000,  # Less than $1M
            "revenue_impact": 0.3,    # 30% impact
            "funding_runway": 4       # 4 months
        }
        
        is_severe, confidence = engine._evaluate_financial_impact(context)
        
        assert is_severe is True
        assert confidence == 0.9
    
    def test_evaluate_financial_impact_non_severe(self, engine):
        """Test evaluating financial impact as non-severe"""
        context = {
            "cash_reserves": 5000000,  # $5M
            "revenue_impact": 0.05,    # 5% impact
            "funding_runway": 18       # 18 months
        }
        
        is_severe, confidence = engine._evaluate_financial_impact(context)
        
        assert is_severe is False
        assert confidence == 0.9
    
    def test_navigate_tree_system_outage(self, engine):
        """Test navigating through system outage decision tree"""
        # Start a decision path
        path = engine.start_decision_path("test_crisis", "system_outage_tree")
        
        # Navigate with critical outage context
        context = {
            "affected_systems": ["web", "api", "database"],
            "user_count": 2000,
            "business_impact": "critical"
        }
        
        recommendation = engine.navigate_tree(path.path_id, context)
        
        assert recommendation.crisis_id == "test_crisis"
        assert recommendation.tree_id == "system_outage_tree"
        assert recommendation.recommended_action is not None
        assert recommendation.confidence_level in [level for level in ConfidenceLevel]
    
    def test_navigate_tree_invalid_path(self, engine):
        """Test navigating with invalid path ID"""
        with pytest.raises(ValueError, match="Decision path invalid_path not found"):
            engine.navigate_tree("invalid_path", {})
    
    def test_complete_path(self, engine):
        """Test completing a decision path"""
        # Start a decision path
        path = engine.start_decision_path("test_crisis", "system_outage_tree")
        path_id = path.path_id
        
        # Complete the path
        engine.complete_path(path_id, "crisis_resolved", True)
        
        # Verify path is removed from active paths
        assert path_id not in engine.active_paths
        
        # Verify tree metrics are updated
        tree = engine.decision_trees["system_outage_tree"]
        assert tree.usage_count >= 1
    
    def test_complete_path_invalid_path(self, engine):
        """Test completing an invalid path"""
        with pytest.raises(ValueError, match="Decision path invalid_path not found"):
            engine.complete_path("invalid_path", "outcome", True)
    
    def test_learn_from_outcome(self, engine):
        """Test learning from decision path outcome"""
        path_id = "test_path"
        feedback_score = 0.8
        lessons_learned = ["Faster response needed", "Better communication required"]
        
        initial_learning_count = len(engine.learning_data)
        
        engine.learn_from_outcome(path_id, feedback_score, lessons_learned)
        
        assert len(engine.learning_data) == initial_learning_count + 1
        assert engine.learning_data[-1].feedback_score == feedback_score
        assert engine.learning_data[-1].lessons_learned == lessons_learned
    
    def test_get_tree_metrics(self, engine):
        """Test getting tree performance metrics"""
        tree_id = "system_outage_tree"
        
        metrics = engine.get_tree_metrics(tree_id)
        
        assert metrics.tree_id == tree_id
        assert metrics.total_usage >= 0
        assert 0.0 <= metrics.success_rate <= 1.0
        assert metrics.average_decision_time >= 0.0
    
    def test_get_tree_metrics_invalid_tree(self, engine):
        """Test getting metrics for invalid tree"""
        with pytest.raises(ValueError, match="Decision tree invalid_tree not found"):
            engine.get_tree_metrics("invalid_tree")
    
    def test_optimize_tree(self, engine):
        """Test optimizing a decision tree"""
        tree_id = "system_outage_tree"
        
        suggestions = engine.optimize_tree(tree_id)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert all(isinstance(suggestion, str) for suggestion in suggestions)
    
    def test_map_confidence_levels(self, engine):
        """Test mapping confidence scores to levels"""
        assert engine._map_confidence(0.95) == ConfidenceLevel.VERY_HIGH
        assert engine._map_confidence(0.8) == ConfidenceLevel.HIGH
        assert engine._map_confidence(0.6) == ConfidenceLevel.MEDIUM
        assert engine._map_confidence(0.4) == ConfidenceLevel.LOW
        assert engine._map_confidence(0.2) == ConfidenceLevel.VERY_LOW
    
    def test_find_node(self, engine):
        """Test finding a node in the decision tree"""
        tree = engine.decision_trees["system_outage_tree"]
        
        # Find root node
        root_node = engine._find_node(tree.root_node, "root_system_outage")
        assert root_node is not None
        assert root_node.node_id == "root_system_outage"
        
        # Find non-existent node
        missing_node = engine._find_node(tree.root_node, "non_existent_node")
        assert missing_node is None
    
    def test_decision_tree_structure(self, engine):
        """Test the structure of created decision trees"""
        # Test system outage tree structure
        system_tree = engine.decision_trees["system_outage_tree"]
        assert system_tree.root_node.node_type == DecisionNodeType.CONDITION
        assert len(system_tree.root_node.children) >= 1
        
        # Test security breach tree structure
        security_tree = engine.decision_trees["security_breach_tree"]
        assert security_tree.root_node.node_type == DecisionNodeType.CONDITION
        assert len(security_tree.root_node.children) >= 1
        
        # Test financial crisis tree structure
        financial_tree = engine.decision_trees["financial_crisis_tree"]
        assert financial_tree.root_node.node_type == DecisionNodeType.CONDITION
        assert len(financial_tree.root_node.children) >= 1
    
    @patch('scrollintel.engines.decision_tree_engine.datetime')
    def test_path_timing(self, mock_datetime, engine):
        """Test that decision paths track timing correctly"""
        mock_now = datetime(2024, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = mock_now
        
        path = engine.start_decision_path("test_crisis", "system_outage_tree")
        
        assert path.start_time == mock_now
        assert path.end_time is None
        
        # Complete the path
        mock_end_time = datetime(2024, 1, 1, 12, 30, 0)
        mock_datetime.now.return_value = mock_end_time
        
        engine.complete_path(path.path_id, "resolved", True)
        
        # Path should be removed from active paths, so we can't check end_time directly
        # But we can verify the path was completed successfully
        tree = engine.decision_trees["system_outage_tree"]
        assert tree.usage_count >= 1