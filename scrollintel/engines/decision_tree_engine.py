"""
Decision Tree Engine for Crisis Leadership Excellence System
"""

import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
import uuid

from ..models.decision_tree_models import (
    DecisionTree, DecisionNode, DecisionCondition, DecisionAction,
    DecisionPath, DecisionRecommendation, DecisionTreeMetrics,
    DecisionTreeLearning, CrisisType, SeverityLevel, DecisionNodeType,
    ConfidenceLevel
)


class DecisionTreeEngine:
    """Engine for managing and executing decision trees in crisis scenarios"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.decision_trees: Dict[str, DecisionTree] = {}
        self.active_paths: Dict[str, DecisionPath] = {}
        self.learning_data: List[DecisionTreeLearning] = []
        self._load_default_trees()
    
    def _load_default_trees(self):
        """Load default decision trees for common crisis scenarios"""
        # System Outage Decision Tree
        system_outage_tree = self._create_system_outage_tree()
        self.decision_trees[system_outage_tree.tree_id] = system_outage_tree
        
        # Security Breach Decision Tree
        security_breach_tree = self._create_security_breach_tree()
        self.decision_trees[security_breach_tree.tree_id] = security_breach_tree
        
        # Financial Crisis Decision Tree
        financial_crisis_tree = self._create_financial_crisis_tree()
        self.decision_trees[financial_crisis_tree.tree_id] = financial_crisis_tree
    
    def _create_system_outage_tree(self) -> DecisionTree:
        """Create decision tree for system outage scenarios"""
        # Root node - assess outage scope
        assess_scope_condition = DecisionCondition(
            condition_id="assess_outage_scope",
            description="Determine the scope and impact of the system outage",
            evaluation_criteria="Check affected systems, user impact, and business functions",
            required_information=["affected_systems", "user_count", "business_impact"],
            evaluation_method="automatic"
        )
        
        root_node = DecisionNode(
            node_id="root_system_outage",
            node_type=DecisionNodeType.CONDITION,
            title="Assess System Outage Scope",
            description="Evaluate the extent and impact of the system outage",
            condition=assess_scope_condition
        )
        
        # Critical outage path
        critical_action = DecisionAction(
            action_id="activate_crisis_team",
            title="Activate Crisis Response Team",
            description="Immediately activate the crisis response team and war room",
            required_resources=["crisis_team", "war_room", "communication_channels"],
            estimated_duration=15,
            success_probability=0.95,
            risk_level="low"
        )
        
        critical_node = DecisionNode(
            node_id="critical_outage_response",
            node_type=DecisionNodeType.ACTION,
            title="Critical Outage Response",
            description="Execute immediate response for critical system outage",
            action=critical_action,
            parent_id="root_system_outage"
        )
        
        # Minor outage path
        minor_action = DecisionAction(
            action_id="standard_incident_response",
            title="Standard Incident Response",
            description="Follow standard incident response procedures",
            required_resources=["on_call_team", "monitoring_tools"],
            estimated_duration=30,
            success_probability=0.90,
            risk_level="low"
        )
        
        minor_node = DecisionNode(
            node_id="minor_outage_response",
            node_type=DecisionNodeType.ACTION,
            title="Minor Outage Response",
            description="Execute standard response for minor system outage",
            action=minor_action,
            parent_id="root_system_outage"
        )
        
        root_node.children = [critical_node, minor_node]
        
        return DecisionTree(
            tree_id="system_outage_tree",
            name="System Outage Response",
            description="Decision tree for handling system outages",
            crisis_types=[CrisisType.SYSTEM_OUTAGE],
            severity_levels=[SeverityLevel.MEDIUM, SeverityLevel.HIGH, SeverityLevel.CRITICAL],
            root_node=root_node,
            tags=["system", "outage", "infrastructure"]
        )
    
    def _create_security_breach_tree(self) -> DecisionTree:
        """Create decision tree for security breach scenarios"""
        # Root node - assess breach severity
        assess_breach_condition = DecisionCondition(
            condition_id="assess_breach_severity",
            description="Determine the severity and scope of the security breach",
            evaluation_criteria="Check data exposure, system compromise, and attack vector",
            required_information=["compromised_systems", "data_exposure", "attack_vector"],
            evaluation_method="hybrid"
        )
        
        root_node = DecisionNode(
            node_id="root_security_breach",
            node_type=DecisionNodeType.CONDITION,
            title="Assess Security Breach",
            description="Evaluate the severity and scope of the security breach",
            condition=assess_breach_condition
        )
        
        # Immediate containment action
        containment_action = DecisionAction(
            action_id="immediate_containment",
            title="Immediate Breach Containment",
            description="Isolate affected systems and prevent further damage",
            required_resources=["security_team", "network_tools", "forensics_team"],
            estimated_duration=45,
            success_probability=0.85,
            risk_level="medium"
        )
        
        containment_node = DecisionNode(
            node_id="breach_containment",
            node_type=DecisionNodeType.ACTION,
            title="Breach Containment",
            description="Execute immediate containment procedures",
            action=containment_action,
            parent_id="root_security_breach"
        )
        
        root_node.children = [containment_node]
        
        return DecisionTree(
            tree_id="security_breach_tree",
            name="Security Breach Response",
            description="Decision tree for handling security breaches",
            crisis_types=[CrisisType.SECURITY_BREACH, CrisisType.CYBER_ATTACK],
            severity_levels=[SeverityLevel.HIGH, SeverityLevel.CRITICAL, SeverityLevel.CATASTROPHIC],
            root_node=root_node,
            tags=["security", "breach", "cyber"]
        )
    
    def _create_financial_crisis_tree(self) -> DecisionTree:
        """Create decision tree for financial crisis scenarios"""
        # Root node - assess financial impact
        assess_financial_condition = DecisionCondition(
            condition_id="assess_financial_impact",
            description="Determine the financial impact and cash flow implications",
            evaluation_criteria="Check cash reserves, revenue impact, and funding needs",
            required_information=["cash_reserves", "revenue_impact", "funding_runway"],
            evaluation_method="manual"
        )
        
        root_node = DecisionNode(
            node_id="root_financial_crisis",
            node_type=DecisionNodeType.CONDITION,
            title="Assess Financial Crisis",
            description="Evaluate the financial impact and implications",
            condition=assess_financial_condition
        )
        
        # Cash preservation action
        cash_preservation_action = DecisionAction(
            action_id="cash_preservation",
            title="Activate Cash Preservation Mode",
            description="Implement immediate cash preservation measures",
            required_resources=["finance_team", "executive_approval", "legal_review"],
            estimated_duration=120,
            success_probability=0.80,
            risk_level="high"
        )
        
        cash_preservation_node = DecisionNode(
            node_id="cash_preservation_mode",
            node_type=DecisionNodeType.ACTION,
            title="Cash Preservation",
            description="Execute cash preservation strategies",
            action=cash_preservation_action,
            parent_id="root_financial_crisis"
        )
        
        root_node.children = [cash_preservation_node]
        
        return DecisionTree(
            tree_id="financial_crisis_tree",
            name="Financial Crisis Response",
            description="Decision tree for handling financial crises",
            crisis_types=[CrisisType.FINANCIAL_CRISIS],
            severity_levels=[SeverityLevel.HIGH, SeverityLevel.CRITICAL, SeverityLevel.CATASTROPHIC],
            root_node=root_node,
            tags=["financial", "cash", "funding"]
        )
    
    def find_applicable_trees(self, crisis_type: CrisisType, severity: SeverityLevel) -> List[DecisionTree]:
        """Find decision trees applicable to the given crisis type and severity"""
        applicable_trees = []
        
        for tree in self.decision_trees.values():
            if crisis_type in tree.crisis_types and severity in tree.severity_levels:
                applicable_trees.append(tree)
        
        # Sort by success rate and usage count
        applicable_trees.sort(key=lambda t: (t.success_rate, t.usage_count), reverse=True)
        
        return applicable_trees
    
    def start_decision_path(self, crisis_id: str, tree_id: str) -> DecisionPath:
        """Start a new decision path through a decision tree"""
        if tree_id not in self.decision_trees:
            raise ValueError(f"Decision tree {tree_id} not found")
        
        path = DecisionPath(
            path_id=str(uuid.uuid4()),
            tree_id=tree_id,
            crisis_id=crisis_id,
            nodes_traversed=[],
            decisions_made=[],
            start_time=datetime.now()
        )
        
        self.active_paths[path.path_id] = path
        self.logger.info(f"Started decision path {path.path_id} for crisis {crisis_id}")
        
        return path
    
    def evaluate_condition(self, condition: DecisionCondition, context: Dict[str, Any]) -> Tuple[bool, float]:
        """Evaluate a decision condition based on available context"""
        try:
            # Check if all required information is available
            missing_info = []
            for info in condition.required_information:
                if info not in context:
                    missing_info.append(info)
            
            if missing_info:
                self.logger.warning(f"Missing information for condition {condition.condition_id}: {missing_info}")
                return False, 0.3  # Low confidence due to missing information
            
            # Evaluate based on condition type
            if condition.condition_id == "assess_outage_scope":
                return self._evaluate_outage_scope(context)
            elif condition.condition_id == "assess_breach_severity":
                return self._evaluate_breach_severity(context)
            elif condition.condition_id == "assess_financial_impact":
                return self._evaluate_financial_impact(context)
            else:
                # Generic evaluation
                return True, 0.7
                
        except Exception as e:
            self.logger.error(f"Error evaluating condition {condition.condition_id}: {str(e)}")
            return False, 0.1
    
    def _evaluate_outage_scope(self, context: Dict[str, Any]) -> Tuple[bool, float]:
        """Evaluate system outage scope"""
        affected_systems = context.get("affected_systems", [])
        user_count = context.get("user_count", 0)
        business_impact = context.get("business_impact", "low")
        
        # Determine if this is a critical outage
        is_critical = (
            len(affected_systems) > 3 or
            user_count > 1000 or
            business_impact in ["high", "critical"]
        )
        
        confidence = 0.9 if all([affected_systems, user_count, business_impact]) else 0.6
        
        return is_critical, confidence
    
    def _evaluate_breach_severity(self, context: Dict[str, Any]) -> Tuple[bool, float]:
        """Evaluate security breach severity"""
        compromised_systems = context.get("compromised_systems", [])
        data_exposure = context.get("data_exposure", "none")
        attack_vector = context.get("attack_vector", "unknown")
        
        # Determine if this is a severe breach
        is_severe = (
            len(compromised_systems) > 1 or
            data_exposure in ["pii", "financial", "confidential"] or
            attack_vector in ["advanced_persistent_threat", "insider_threat"]
        )
        
        confidence = 0.8 if all([compromised_systems, data_exposure, attack_vector]) else 0.5
        
        return is_severe, confidence
    
    def _evaluate_financial_impact(self, context: Dict[str, Any]) -> Tuple[bool, float]:
        """Evaluate financial crisis impact"""
        cash_reserves = context.get("cash_reserves", 0)
        revenue_impact = context.get("revenue_impact", 0)
        funding_runway = context.get("funding_runway", 12)  # months
        
        # Determine if this is a severe financial crisis
        is_severe = (
            cash_reserves < 1000000 or  # Less than $1M
            revenue_impact > 0.2 or     # More than 20% revenue impact
            funding_runway < 6          # Less than 6 months runway
        )
        
        confidence = 0.9 if all([cash_reserves, revenue_impact, funding_runway]) else 0.4
        
        return is_severe, confidence
    
    def navigate_tree(self, path_id: str, context: Dict[str, Any]) -> DecisionRecommendation:
        """Navigate through a decision tree and provide recommendations"""
        if path_id not in self.active_paths:
            raise ValueError(f"Decision path {path_id} not found")
        
        path = self.active_paths[path_id]
        tree = self.decision_trees[path.tree_id]
        
        # Start from root if no nodes traversed yet
        current_node = tree.root_node if not path.nodes_traversed else self._find_node(tree.root_node, path.nodes_traversed[-1])
        
        if not current_node:
            raise ValueError("Cannot find current node in decision tree")
        
        # Navigate based on node type
        if current_node.node_type == DecisionNodeType.CONDITION:
            return self._handle_condition_node(path, current_node, context)
        elif current_node.node_type == DecisionNodeType.ACTION:
            return self._handle_action_node(path, current_node, context)
        else:
            # Default recommendation
            return self._create_default_recommendation(path, current_node, context)
    
    def _find_node(self, root: DecisionNode, node_id: str) -> Optional[DecisionNode]:
        """Find a node in the decision tree by ID"""
        if root.node_id == node_id:
            return root
        
        for child in root.children:
            found = self._find_node(child, node_id)
            if found:
                return found
        
        return None
    
    def _handle_condition_node(self, path: DecisionPath, node: DecisionNode, context: Dict[str, Any]) -> DecisionRecommendation:
        """Handle a condition node in the decision tree"""
        if not node.condition:
            raise ValueError("Condition node missing condition")
        
        # Evaluate the condition
        result, confidence = self.evaluate_condition(node.condition, context)
        
        # Record the decision
        decision_record = {
            "node_id": node.node_id,
            "condition_id": node.condition.condition_id,
            "result": result,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        
        path.nodes_traversed.append(node.node_id)
        path.decisions_made.append(decision_record)
        
        # Choose next node based on result
        next_node = node.children[0] if result and node.children else None
        if not next_node and len(node.children) > 1:
            next_node = node.children[1]  # Alternative path
        
        if next_node:
            if next_node.action:
                return DecisionRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    crisis_id=path.crisis_id,
                    tree_id=path.tree_id,
                    recommended_action=next_node.action,
                    confidence_level=self._map_confidence(confidence),
                    reasoning=f"Based on condition evaluation: {node.condition.description}",
                    estimated_impact={"success_probability": next_node.action.success_probability}
                )
        
        return self._create_default_recommendation(path, node, context)
    
    def _handle_action_node(self, path: DecisionPath, node: DecisionNode, context: Dict[str, Any]) -> DecisionRecommendation:
        """Handle an action node in the decision tree"""
        if not node.action:
            raise ValueError("Action node missing action")
        
        path.nodes_traversed.append(node.node_id)
        
        return DecisionRecommendation(
            recommendation_id=str(uuid.uuid4()),
            crisis_id=path.crisis_id,
            tree_id=path.tree_id,
            recommended_action=node.action,
            confidence_level=ConfidenceLevel.HIGH,
            reasoning=f"Direct action recommendation: {node.description}",
            estimated_impact={"success_probability": node.action.success_probability}
        )
    
    def _create_default_recommendation(self, path: DecisionPath, node: DecisionNode, context: Dict[str, Any]) -> DecisionRecommendation:
        """Create a default recommendation when no specific action is available"""
        default_action = DecisionAction(
            action_id="gather_more_information",
            title="Gather More Information",
            description="Collect additional information before proceeding",
            required_resources=["analysis_team"],
            estimated_duration=30,
            success_probability=0.8,
            risk_level="low"
        )
        
        return DecisionRecommendation(
            recommendation_id=str(uuid.uuid4()),
            crisis_id=path.crisis_id,
            tree_id=path.tree_id,
            recommended_action=default_action,
            confidence_level=ConfidenceLevel.MEDIUM,
            reasoning="Insufficient information to make specific recommendation",
            estimated_impact={"success_probability": 0.8}
        )
    
    def _map_confidence(self, confidence_score: float) -> ConfidenceLevel:
        """Map numeric confidence score to confidence level enum"""
        if confidence_score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def complete_path(self, path_id: str, outcome: str, success: bool) -> None:
        """Complete a decision path and record the outcome"""
        if path_id not in self.active_paths:
            raise ValueError(f"Decision path {path_id} not found")
        
        path = self.active_paths[path_id]
        path.end_time = datetime.now()
        path.outcome = outcome
        path.success = success
        
        # Update tree metrics
        tree = self.decision_trees[path.tree_id]
        tree.usage_count += 1
        
        # Calculate new success rate
        if tree.usage_count == 1:
            tree.success_rate = 1.0 if success else 0.0
        else:
            # Weighted average with previous success rate
            tree.success_rate = (tree.success_rate * (tree.usage_count - 1) + (1.0 if success else 0.0)) / tree.usage_count
        
        # Remove from active paths
        del self.active_paths[path_id]
        
        self.logger.info(f"Completed decision path {path_id} with outcome: {outcome}")
    
    def learn_from_outcome(self, path_id: str, feedback_score: float, lessons_learned: List[str]) -> None:
        """Learn from the outcome of a decision path to improve future recommendations"""
        # This would be implemented with machine learning algorithms
        # For now, we'll store the learning data for future analysis
        learning_data = DecisionTreeLearning(
            learning_id=str(uuid.uuid4()),
            tree_id="",  # Would be filled from path data
            crisis_scenario={},  # Would be filled from context
            actual_outcome="",  # Would be filled from path outcome
            predicted_outcome="",  # Would be filled from recommendations
            decision_path=[],  # Would be filled from path
            feedback_score=feedback_score,
            lessons_learned=lessons_learned,
            suggested_improvements=[]
        )
        
        self.learning_data.append(learning_data)
        self.logger.info(f"Recorded learning data for path {path_id}")
    
    def get_tree_metrics(self, tree_id: str) -> DecisionTreeMetrics:
        """Get performance metrics for a decision tree"""
        if tree_id not in self.decision_trees:
            raise ValueError(f"Decision tree {tree_id} not found")
        
        tree = self.decision_trees[tree_id]
        
        return DecisionTreeMetrics(
            tree_id=tree_id,
            total_usage=tree.usage_count,
            success_rate=tree.success_rate,
            average_decision_time=0.0,  # Would be calculated from historical data
            confidence_distribution={},  # Would be calculated from historical data
            most_common_paths=[],  # Would be calculated from historical data
            failure_points=[],  # Would be calculated from historical data
            optimization_suggestions=[]  # Would be generated from analysis
        )
    
    def optimize_tree(self, tree_id: str) -> List[str]:
        """Optimize a decision tree based on learning data and performance metrics"""
        # This would implement tree optimization algorithms
        # For now, return basic suggestions
        suggestions = [
            "Consider adding more condition branches for edge cases",
            "Review action success probabilities based on historical data",
            "Add timeout handling for long-running conditions",
            "Implement parallel decision paths for complex scenarios"
        ]
        
        self.logger.info(f"Generated optimization suggestions for tree {tree_id}")
        return suggestions