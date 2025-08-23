"""
Tests for Intelligence and Decision Engine

This module contains comprehensive tests for the Intelligence and Decision Engine,
including decision making, risk assessment, ML pipeline, and knowledge graph functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

from scrollintel.engines.intelligence_engine import (
    IntelligenceEngine, BusinessDecisionTree, ContinuousLearningPipeline,
    RiskAssessmentEngine, BusinessKnowledgeGraph, BusinessContext,
    DecisionOption, Decision, DecisionConfidence, RiskLevel
)


class TestIntelligenceEngine:
    """Test suite for the main Intelligence Engine"""
    
    @pytest.fixture
    async def intelligence_engine(self):
        """Create and initialize intelligence engine for testing"""
        engine = IntelligenceEngine()
        await engine.start()
        yield engine
        await engine.stop()
    
    @pytest.fixture
    def sample_business_context(self):
        """Create sample business context for testing"""
        return BusinessContext(
            industry="technology",
            business_unit="product_development",
            stakeholders=["cto", "product_manager", "engineering_team"],
            constraints=[
                {"type": "budget", "value": 500000},
                {"type": "timeline", "value": 180}
            ],
            objectives=[
                {"type": "revenue", "target": 1000000, "timeline": "12_months"},
                {"type": "market_share", "target": 0.15, "timeline": "18_months"}
            ],
            current_state={
                "revenue": 750000,
                "market_share": 0.12,
                "team_size": 25,
                "product_maturity": "growth"
            },
            historical_data={
                "previous_launches": 3,
                "success_rate": 0.67,
                "average_roi": 2.3
            },
            time_horizon="medium_term",
            budget_constraints={"total": 500000, "development": 300000, "marketing": 200000},
            regulatory_requirements=["gdpr", "sox", "pci_dss"]
        )
    
    @pytest.fixture
    def sample_decision_options(self):
        """Create sample decision options for testing"""
        return [
            DecisionOption(
                id="option_1",
                name="Aggressive Market Expansion",
                description="Rapid expansion into new markets with significant investment",
                expected_outcomes={"revenue_increase": 2000000, "market_share_gain": 0.08},
                costs={"development": 200000, "marketing": 300000, "operations": 100000},
                benefits={"revenue": 2000000, "brand_recognition": 500000},
                risks=[
                    {"type": "market", "probability": 0.3, "impact": 800000},
                    {"type": "competition", "probability": 0.4, "impact": 500000}
                ],
                implementation_complexity=8.5,
                time_to_implement=120,
                resource_requirements={"engineers": 15, "marketers": 8, "budget": 600000}
            ),
            DecisionOption(
                id="option_2",
                name="Conservative Product Enhancement",
                description="Incremental improvements to existing product with lower risk",
                expected_outcomes={"revenue_increase": 800000, "customer_satisfaction": 0.15},
                costs={"development": 150000, "marketing": 100000, "operations": 50000},
                benefits={"revenue": 800000, "customer_retention": 200000},
                risks=[
                    {"type": "competitive", "probability": 0.2, "impact": 300000}
                ],
                implementation_complexity=4.0,
                time_to_implement=60,
                resource_requirements={"engineers": 8, "marketers": 3, "budget": 300000}
            ),
            DecisionOption(
                id="option_3",
                name="Innovation-Focused R&D",
                description="Heavy investment in research and development for breakthrough innovation",
                expected_outcomes={"innovation_score": 0.9, "future_revenue_potential": 5000000},
                costs={"research": 400000, "development": 200000, "talent": 150000},
                benefits={"ip_value": 2000000, "competitive_advantage": 1000000},
                risks=[
                    {"type": "technical", "probability": 0.5, "impact": 600000},
                    {"type": "market_timing", "probability": 0.3, "impact": 400000}
                ],
                implementation_complexity=9.0,
                time_to_implement=180,
                resource_requirements={"researchers": 10, "engineers": 12, "budget": 750000}
            )
        ]
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, intelligence_engine):
        """Test that the intelligence engine initializes correctly"""
        assert intelligence_engine.status.value == "ready"
        assert intelligence_engine.decision_tree is not None
        assert intelligence_engine.learning_pipeline is not None
        assert intelligence_engine.risk_engine is not None
        assert intelligence_engine.knowledge_graph is not None
    
    @pytest.mark.asyncio
    async def test_make_decision(self, intelligence_engine, sample_business_context, sample_decision_options):
        """Test the main decision-making functionality"""
        
        # Make a decision
        decision = await intelligence_engine.make_decision(sample_business_context, sample_decision_options)
        
        # Verify decision structure
        assert isinstance(decision, Decision)
        assert decision.id is not None
        assert decision.context == sample_business_context
        assert decision.selected_option in sample_decision_options
        assert isinstance(decision.confidence, DecisionConfidence)
        assert len(decision.reasoning) > 0
        assert decision.risk_assessment is not None
        assert decision.expected_outcome is not None
        assert len(decision.decision_tree_path) > 0
        assert len(decision.ml_predictions) > 0
        assert decision.timestamp is not None
    
    @pytest.mark.asyncio
    async def test_decision_confidence_levels(self, intelligence_engine, sample_business_context, sample_decision_options):
        """Test that decision confidence is calculated appropriately"""
        
        decision = await intelligence_engine.make_decision(sample_business_context, sample_decision_options)
        
        # Confidence should be one of the valid enum values
        valid_confidences = [conf.value for conf in DecisionConfidence]
        assert decision.confidence.value in valid_confidences
        
        # Decision should have reasoning that explains the confidence
        assert any("confidence" in reason.lower() for reason in decision.reasoning)
    
    @pytest.mark.asyncio
    async def test_risk_assessment(self, intelligence_engine, sample_business_context):
        """Test risk assessment functionality"""
        
        scenario = {
            "type": "market_expansion",
            "impact_areas": ["financial", "operational", "strategic"],
            "timeline": "medium_term",
            "investment_required": 1000000,
            "market_uncertainty": 0.6
        }
        
        risk_result = await intelligence_engine.assess_risk(scenario, sample_business_context)
        
        # Verify risk assessment structure
        assert "overall_risk_score" in risk_result
        assert "risk_level" in risk_result
        assert "risk_categories" in risk_result
        assert "mitigation_strategies" in risk_result
        assert "confidence" in risk_result
        
        # Risk score should be between 0 and 1
        assert 0 <= risk_result["overall_risk_score"] <= 1
        
        # Should have mitigation strategies for significant risks
        if risk_result["overall_risk_score"] > 0.5:
            assert len(risk_result["mitigation_strategies"]) > 0
    
    @pytest.mark.asyncio
    async def test_knowledge_query(self, intelligence_engine):
        """Test knowledge graph query functionality"""
        
        query = "market analysis best practices"
        context = {"domain": "strategic_planning", "urgency": "high"}
        
        results = await intelligence_engine.query_knowledge(query, context)
        
        # Verify query results
        assert isinstance(results, list)
        assert len(results) >= 0
        
        # If results exist, they should have proper structure
        if results:
            for result in results:
                assert "id" in result
                assert "name" in result
                assert "description" in result
                assert "relevance_score" in result or "final_relevance" in result
    
    @pytest.mark.asyncio
    async def test_learn_from_outcome(self, intelligence_engine, sample_business_context, sample_decision_options):
        """Test learning from business outcomes"""
        
        # First make a decision
        decision = await intelligence_engine.make_decision(sample_business_context, sample_decision_options)
        
        # Create a sample outcome
        outcome = {
            "success_score": 0.8,
            "financial_impact": 1500000,
            "roi_achieved": 2.5,
            "timeline_variance": -10,  # 10 days early
            "stakeholder_satisfaction": 0.85,
            "lessons_learned": [
                "Market research was crucial for success",
                "Early stakeholder engagement improved outcomes"
            ]
        }
        
        # Learn from the outcome
        await intelligence_engine.learn_from_outcome(decision.id, outcome)
        
        # Verify that learning was processed (no exceptions thrown)
        assert True  # If we get here, learning was successful
    
    @pytest.mark.asyncio
    async def test_engine_status(self, intelligence_engine):
        """Test engine status reporting"""
        
        status = intelligence_engine.get_status()
        
        # Verify status structure
        assert "healthy" in status
        assert "decision_tree_status" in status
        assert "ml_pipeline_status" in status
        assert "risk_engine_status" in status
        assert "knowledge_graph_status" in status
        assert "active_models" in status
        
        # Engine should be healthy after initialization
        assert status["healthy"] is True
    
    @pytest.mark.asyncio
    async def test_engine_metrics(self, intelligence_engine):
        """Test engine metrics collection"""
        
        metrics = intelligence_engine.get_metrics()
        
        # Verify metrics structure
        assert "engine_id" in metrics
        assert "name" in metrics
        assert "status" in metrics
        assert "capabilities" in metrics
        assert "usage_count" in metrics
        assert "error_count" in metrics
        assert "error_rate" in metrics
        assert "created_at" in metrics
        
        # Metrics should have reasonable values
        assert metrics["engine_id"] == "intelligence_engine"
        assert metrics["usage_count"] >= 0
        assert metrics["error_count"] >= 0
        assert 0 <= metrics["error_rate"] <= 1


class TestBusinessDecisionTree:
    """Test suite for the Business Decision Tree component"""
    
    @pytest.fixture
    async def decision_tree(self):
        """Create and initialize decision tree for testing"""
        tree = BusinessDecisionTree()
        await tree.initialize()
        return tree
    
    @pytest.fixture
    def sample_context_and_options(self):
        """Create sample context and options for decision tree testing"""
        context = BusinessContext(
            industry="finance",
            business_unit="investment",
            stakeholders=["cfo", "investment_committee"],
            constraints=[{"type": "risk_tolerance", "value": 0.3}],
            objectives=[{"type": "return", "target": 0.15}],
            current_state={"portfolio_value": 10000000},
            historical_data={"avg_return": 0.12},
            time_horizon="long_term",
            budget_constraints={"investment": 2000000},
            regulatory_requirements=["sec", "finra"]
        )
        
        options = [
            DecisionOption(
                id="conservative",
                name="Conservative Investment",
                description="Low-risk, stable return investment",
                expected_outcomes={"return": 0.08, "volatility": 0.05},
                costs={"management_fee": 20000},
                benefits={"stable_income": 160000},
                risks=[{"type": "inflation", "probability": 0.2, "impact": 50000}],
                implementation_complexity=2.0,
                time_to_implement=30,
                resource_requirements={"capital": 2000000}
            ),
            DecisionOption(
                id="aggressive",
                name="Aggressive Growth",
                description="High-risk, high-return investment strategy",
                expected_outcomes={"return": 0.18, "volatility": 0.15},
                costs={"management_fee": 40000, "research": 10000},
                benefits={"growth_potential": 360000},
                risks=[
                    {"type": "market", "probability": 0.4, "impact": 400000},
                    {"type": "liquidity", "probability": 0.2, "impact": 200000}
                ],
                implementation_complexity=7.0,
                time_to_implement=60,
                resource_requirements={"capital": 2000000, "analysts": 3}
            )
        ]
        
        return context, options
    
    @pytest.mark.asyncio
    async def test_decision_tree_initialization(self, decision_tree):
        """Test decision tree initialization"""
        assert decision_tree.tree_nodes is not None
        assert len(decision_tree.tree_nodes) > 0
        assert "root" in decision_tree.tree_nodes
        assert decision_tree.current_version is not None
    
    @pytest.mark.asyncio
    async def test_evaluate_options(self, decision_tree, sample_context_and_options):
        """Test option evaluation using decision tree"""
        context, options = sample_context_and_options
        
        # Mock ML predictions and risk assessments
        ml_predictions = [
            {"option_id": "conservative", "roi_prediction": 0.08, "confidence": 0.9},
            {"option_id": "aggressive", "roi_prediction": 0.18, "confidence": 0.6}
        ]
        
        risk_assessments = [
            {"option_id": "conservative", "risk_score": 0.2},
            {"option_id": "aggressive", "risk_score": 0.7}
        ]
        
        knowledge = [
            {"type": "best_practice", "content": "Diversification reduces risk"}
        ]
        
        # Evaluate options
        result = await decision_tree.evaluate_options(
            context, options, ml_predictions, risk_assessments, knowledge
        )
        
        # Verify result structure
        assert "selected_option" in result
        assert "reasoning" in result
        assert "confidence" in result
        assert "tree_path" in result
        assert "risk_summary" in result
        assert "expected_outcome" in result
        
        # Selected option should be one of the input options
        assert result["selected_option"] in options
        
        # Should have reasoning and tree path
        assert len(result["reasoning"]) > 0
        assert len(result["tree_path"]) > 0
    
    @pytest.mark.asyncio
    async def test_update_from_outcome(self, decision_tree):
        """Test updating decision tree from outcomes"""
        decision_id = "test_decision_123"
        outcome = {
            "success_score": 0.9,
            "actual_roi": 0.15,
            "variance_from_prediction": 0.02
        }
        
        # Update tree from outcome (should not raise exceptions)
        await decision_tree.update_from_outcome(decision_id, outcome)
        
        # Verify update was processed
        assert True  # If we get here, update was successful


class TestContinuousLearningPipeline:
    """Test suite for the Continuous Learning Pipeline"""
    
    @pytest.fixture
    async def learning_pipeline(self):
        """Create and initialize learning pipeline for testing"""
        pipeline = ContinuousLearningPipeline()
        await pipeline.initialize()
        return pipeline
    
    @pytest.fixture
    def sample_option_and_context(self):
        """Create sample option and context for ML testing"""
        option = DecisionOption(
            id="ml_test_option",
            name="ML Test Option",
            description="Option for testing ML predictions",
            expected_outcomes={"revenue": 1000000},
            costs={"development": 200000, "marketing": 100000},
            benefits={"revenue": 1000000, "market_share": 0.05},
            risks=[{"type": "market", "probability": 0.3, "impact": 300000}],
            implementation_complexity=6.0,
            time_to_implement=90,
            resource_requirements={"team": 10, "budget": 300000}
        )
        
        context = BusinessContext(
            industry="retail",
            business_unit="ecommerce",
            stakeholders=["ceo", "cmo"],
            constraints=[{"type": "budget", "value": 500000}],
            objectives=[{"type": "growth", "target": 0.2}],
            current_state={"revenue": 5000000},
            historical_data={"growth_rate": 0.15},
            time_horizon="short_term",
            budget_constraints={"total": 500000},
            regulatory_requirements=["gdpr"]
        )
        
        return option, context
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, learning_pipeline):
        """Test learning pipeline initialization"""
        assert learning_pipeline.models is not None
        assert len(learning_pipeline.models) > 0
        assert "roi_predictor" in learning_pipeline.models
        assert "success_predictor" in learning_pipeline.models
        assert "risk_predictor" in learning_pipeline.models
    
    @pytest.mark.asyncio
    async def test_predict_outcome(self, learning_pipeline, sample_option_and_context):
        """Test ML outcome prediction"""
        option, context = sample_option_and_context
        
        prediction = await learning_pipeline.predict_outcome(option, context)
        
        # Verify prediction structure
        assert "option_id" in prediction
        assert "roi_prediction" in prediction
        assert "success_probability" in prediction
        assert "confidence" in prediction
        assert "risk_factors" in prediction
        assert "model_predictions" in prediction
        
        # Verify prediction values are reasonable
        assert prediction["option_id"] == option.id
        assert 0 <= prediction["roi_prediction"] <= 10  # ROI can be > 1
        assert 0 <= prediction["success_probability"] <= 1
        assert 0 <= prediction["confidence"] <= 1
        assert isinstance(prediction["risk_factors"], list)
        assert isinstance(prediction["model_predictions"], dict)
    
    @pytest.mark.asyncio
    async def test_learn_from_outcome(self, learning_pipeline):
        """Test learning from business outcomes"""
        decision_id = "test_decision_ml_123"
        outcome = {
            "success_score": 0.85,
            "actual_roi": 1.8,
            "timeline_accuracy": 0.9,
            "cost_accuracy": 0.95
        }
        
        # Learn from outcome (should not raise exceptions)
        await learning_pipeline.learn_from_outcome(decision_id, outcome)
        
        # Verify learning was processed
        assert True  # If we get here, learning was successful


class TestRiskAssessmentEngine:
    """Test suite for the Risk Assessment Engine"""
    
    @pytest.fixture
    async def risk_engine(self):
        """Create and initialize risk engine for testing"""
        engine = RiskAssessmentEngine()
        await engine.initialize()
        return engine
    
    @pytest.fixture
    def sample_risk_scenario(self):
        """Create sample risk scenario for testing"""
        option = DecisionOption(
            id="risk_test_option",
            name="High-Risk Venture",
            description="New market entry with significant uncertainty",
            expected_outcomes={"market_share": 0.1, "revenue": 2000000},
            costs={"entry_cost": 500000, "marketing": 300000},
            benefits={"revenue": 2000000, "brand_value": 500000},
            risks=[
                {"type": "regulatory", "probability": 0.4, "impact": 600000},
                {"type": "competitive", "probability": 0.6, "impact": 400000}
            ],
            implementation_complexity=8.5,
            time_to_implement=150,
            resource_requirements={"team": 20, "budget": 800000}
        )
        
        context = BusinessContext(
            industry="healthcare",
            business_unit="medical_devices",
            stakeholders=["ceo", "regulatory_affairs", "r_and_d"],
            constraints=[{"type": "regulatory_approval", "timeline": 365}],
            objectives=[{"type": "market_entry", "timeline": "18_months"}],
            current_state={"regulatory_status": "pre_submission"},
            historical_data={"approval_rate": 0.7},
            time_horizon="long_term",
            budget_constraints={"total": 1000000},
            regulatory_requirements=["fda", "ce_mark", "iso_13485"]
        )
        
        return option, context
    
    @pytest.mark.asyncio
    async def test_risk_engine_initialization(self, risk_engine):
        """Test risk engine initialization"""
        assert risk_engine.risk_models is not None
        assert risk_engine.risk_history is not None
    
    @pytest.mark.asyncio
    async def test_assess_option_risk(self, risk_engine, sample_risk_scenario):
        """Test risk assessment for decision options"""
        option, context = sample_risk_scenario
        
        risk_assessment = await risk_engine.assess_option_risk(option, context)
        
        # Verify risk assessment structure
        assert "option_id" in risk_assessment
        assert "risk_score" in risk_assessment
        assert "risk_level" in risk_assessment
        assert "financial_risk" in risk_assessment
        assert "operational_risk" in risk_assessment
        assert "strategic_risk" in risk_assessment
        assert "regulatory_risk" in risk_assessment
        assert "mitigation_strategies" in risk_assessment
        assert "confidence" in risk_assessment
        
        # Verify risk values are reasonable
        assert risk_assessment["option_id"] == option.id
        assert 0 <= risk_assessment["risk_score"] <= 1
        assert 0 <= risk_assessment["confidence"] <= 1
        
        # Should have mitigation strategies for high-risk options
        if risk_assessment["risk_score"] > 0.5:
            assert len(risk_assessment["mitigation_strategies"]) > 0
    
    @pytest.mark.asyncio
    async def test_assess_scenario_risk(self, risk_engine, sample_risk_scenario):
        """Test scenario risk assessment"""
        option, context = sample_risk_scenario
        
        scenario = {
            "id": "test_scenario",
            "type": "market_entry",
            "impact_areas": ["financial", "operational", "regulatory"],
            "timeline": "long_term",
            "investment": 1000000
        }
        
        risk_result = await risk_engine.assess_scenario_risk(scenario, context)
        
        # Verify scenario risk assessment
        assert "scenario_id" in risk_result
        assert "overall_risk" in risk_result
        assert "area_risks" in risk_result
        assert "timeline" in risk_result
        assert "assessment_timestamp" in risk_result
        
        # Should have risk assessments for each impact area
        assert len(risk_result["area_risks"]) > 0


class TestBusinessKnowledgeGraph:
    """Test suite for the Business Knowledge Graph"""
    
    @pytest.fixture
    async def knowledge_graph(self):
        """Create and initialize knowledge graph for testing"""
        graph = BusinessKnowledgeGraph()
        await graph.initialize()
        return graph
    
    @pytest.fixture
    def sample_business_context_kg(self):
        """Create sample business context for knowledge graph testing"""
        return BusinessContext(
            industry="manufacturing",
            business_unit="supply_chain",
            stakeholders=["coo", "supply_chain_manager"],
            constraints=[{"type": "supplier_diversity", "requirement": 0.3}],
            objectives=[{"type": "cost_reduction", "target": 0.15}],
            current_state={"supplier_count": 50, "cost_base": 10000000},
            historical_data={"cost_trends": "increasing"},
            time_horizon="medium_term",
            budget_constraints={"optimization": 200000},
            regulatory_requirements=["iso_9001", "environmental"]
        )
    
    @pytest.mark.asyncio
    async def test_knowledge_graph_initialization(self, knowledge_graph):
        """Test knowledge graph initialization"""
        assert knowledge_graph.entities is not None
        assert knowledge_graph.relationships is not None
        assert knowledge_graph.indexes is not None
        assert len(knowledge_graph.entities) > 0
    
    @pytest.mark.asyncio
    async def test_get_relevant_knowledge(self, knowledge_graph, sample_business_context_kg):
        """Test retrieving relevant knowledge for business context"""
        
        relevant_knowledge = await knowledge_graph.get_relevant_knowledge(sample_business_context_kg)
        
        # Verify knowledge retrieval
        assert isinstance(relevant_knowledge, list)
        assert len(relevant_knowledge) >= 0
        
        # If knowledge exists, verify structure
        if relevant_knowledge:
            for knowledge_item in relevant_knowledge:
                assert "id" in knowledge_item
                assert "name" in knowledge_item
                assert "description" in knowledge_item
                assert "final_relevance" in knowledge_item or "relevance_score" in knowledge_item
    
    @pytest.mark.asyncio
    async def test_semantic_search(self, knowledge_graph):
        """Test semantic search functionality"""
        
        query = "risk management best practices"
        context = {"domain": "finance", "urgency": "medium"}
        
        results = await knowledge_graph.semantic_search(query, context)
        
        # Verify search results
        assert isinstance(results, list)
        assert len(results) >= 0
        
        # If results exist, verify structure
        if results:
            for result in results:
                assert "id" in result
                assert "name" in result
                assert "description" in result
                assert "relevance_score" in result
    
    @pytest.mark.asyncio
    async def test_update_from_outcome(self, knowledge_graph):
        """Test updating knowledge graph from outcomes"""
        
        decision_id = "test_decision_kg_123"
        outcome = {
            "success_score": 0.9,
            "key_factors": ["stakeholder_engagement", "data_quality"],
            "insights": ["Early engagement improves outcomes", "Data quality is critical"]
        }
        
        # Update knowledge graph from outcome
        await knowledge_graph.update_from_outcome(decision_id, outcome)
        
        # Verify update was processed
        assert True  # If we get here, update was successful


class TestIntegrationScenarios:
    """Integration tests for complete intelligence engine workflows"""
    
    @pytest.fixture
    async def full_intelligence_system(self):
        """Create complete intelligence system for integration testing"""
        engine = IntelligenceEngine()
        await engine.start()
        yield engine
        await engine.stop()
    
    @pytest.mark.asyncio
    async def test_complete_decision_workflow(self, full_intelligence_system):
        """Test complete decision-making workflow from start to finish"""
        
        # Step 1: Create business context
        context = BusinessContext(
            industry="technology",
            business_unit="ai_research",
            stakeholders=["cto", "research_director", "product_manager"],
            constraints=[
                {"type": "budget", "value": 2000000},
                {"type": "timeline", "value": 365},
                {"type": "team_size", "value": 25}
            ],
            objectives=[
                {"type": "breakthrough_innovation", "target": 1, "timeline": "12_months"},
                {"type": "patent_applications", "target": 5, "timeline": "18_months"},
                {"type": "revenue_potential", "target": 10000000, "timeline": "24_months"}
            ],
            current_state={
                "research_maturity": "advanced",
                "team_expertise": "high",
                "market_position": "leader",
                "ip_portfolio": 45
            },
            historical_data={
                "successful_projects": 8,
                "total_projects": 12,
                "average_roi": 3.2,
                "time_to_market": 18
            },
            time_horizon="long_term",
            budget_constraints={
                "total": 2000000,
                "research": 1200000,
                "development": 600000,
                "operations": 200000
            },
            regulatory_requirements=["ai_ethics", "data_privacy", "export_control"]
        )
        
        # Step 2: Create decision options
        options = [
            DecisionOption(
                id="quantum_ai_research",
                name="Quantum AI Research Initiative",
                description="Invest in quantum computing applications for AI acceleration",
                expected_outcomes={
                    "breakthrough_probability": 0.3,
                    "patent_potential": 8,
                    "revenue_potential": 50000000,
                    "competitive_advantage": 0.9
                },
                costs={
                    "quantum_hardware": 800000,
                    "specialized_talent": 600000,
                    "research_infrastructure": 400000,
                    "partnerships": 200000
                },
                benefits={
                    "ip_value": 20000000,
                    "market_leadership": 10000000,
                    "talent_attraction": 2000000
                },
                risks=[
                    {"type": "technical_feasibility", "probability": 0.4, "impact": 1500000},
                    {"type": "talent_acquisition", "probability": 0.3, "impact": 800000},
                    {"type": "market_timing", "probability": 0.2, "impact": 5000000}
                ],
                implementation_complexity=9.5,
                time_to_implement=365,
                resource_requirements={
                    "quantum_researchers": 8,
                    "ai_engineers": 12,
                    "infrastructure_specialists": 5,
                    "budget": 2000000,
                    "lab_space": 500,
                    "quantum_access": True
                }
            ),
            DecisionOption(
                id="edge_ai_platform",
                name="Edge AI Platform Development",
                description="Develop comprehensive edge AI platform for IoT applications",
                expected_outcomes={
                    "market_penetration": 0.15,
                    "revenue_potential": 15000000,
                    "customer_base": 500,
                    "platform_adoption": 0.25
                },
                costs={
                    "platform_development": 700000,
                    "edge_hardware": 300000,
                    "cloud_infrastructure": 200000,
                    "go_to_market": 400000
                },
                benefits={
                    "recurring_revenue": 8000000,
                    "ecosystem_value": 5000000,
                    "data_insights": 2000000
                },
                risks=[
                    {"type": "competitive_response", "probability": 0.6, "impact": 3000000},
                    {"type": "technology_obsolescence", "probability": 0.2, "impact": 1000000},
                    {"type": "customer_adoption", "probability": 0.3, "impact": 2000000}
                ],
                implementation_complexity=7.0,
                time_to_implement=270,
                resource_requirements={
                    "platform_engineers": 15,
                    "hardware_specialists": 6,
                    "cloud_architects": 4,
                    "budget": 1600000,
                    "testing_infrastructure": True
                }
            ),
            DecisionOption(
                id="ai_ethics_framework",
                name="AI Ethics and Governance Framework",
                description="Develop industry-leading AI ethics and governance capabilities",
                expected_outcomes={
                    "regulatory_compliance": 1.0,
                    "industry_leadership": 0.8,
                    "risk_mitigation": 0.9,
                    "stakeholder_trust": 0.85
                },
                costs={
                    "ethics_research": 300000,
                    "governance_tools": 200000,
                    "compliance_systems": 250000,
                    "training_programs": 150000
                },
                benefits={
                    "risk_reduction": 5000000,
                    "regulatory_advantage": 3000000,
                    "brand_value": 4000000
                },
                risks=[
                    {"type": "regulatory_changes", "probability": 0.3, "impact": 500000},
                    {"type": "implementation_complexity", "probability": 0.2, "impact": 300000}
                ],
                implementation_complexity=6.0,
                time_to_implement=180,
                resource_requirements={
                    "ethics_researchers": 5,
                    "compliance_specialists": 4,
                    "software_engineers": 8,
                    "budget": 900000,
                    "legal_support": True
                }
            )
        ]
        
        # Step 3: Make decision
        decision = await full_intelligence_system.make_decision(context, options)
        
        # Verify decision was made successfully
        assert isinstance(decision, Decision)
        assert decision.selected_option in options
        assert isinstance(decision.confidence, DecisionConfidence)
        
        # Step 4: Simulate outcome and learn
        outcome = {
            "success_score": 0.85,
            "financial_impact": decision.selected_option.benefits.get("recurring_revenue", 0) * 0.8,
            "roi_achieved": 2.8,
            "timeline_variance": 15,  # 15 days over estimate
            "stakeholder_satisfaction": 0.9,
            "unexpected_benefits": [
                "Improved team morale and retention",
                "Unexpected partnership opportunities",
                "Enhanced market reputation"
            ],
            "challenges_faced": [
                "Initial talent acquisition delays",
                "Integration complexity higher than expected"
            ],
            "lessons_learned": [
                "Early stakeholder engagement is crucial",
                "Phased implementation reduces risk",
                "Cross-functional teams improve outcomes"
            ],
            "kpi_results": {
                "budget_variance": -0.05,  # 5% under budget
                "timeline_variance": 0.06,  # 6% over timeline
                "quality_score": 0.92,
                "customer_satisfaction": 0.88
            }
        }
        
        # Step 5: Learn from outcome
        await full_intelligence_system.learn_from_outcome(decision.id, outcome)
        
        # Verify the complete workflow executed successfully
        assert True  # If we reach here, the complete workflow succeeded
    
    @pytest.mark.asyncio
    async def test_multiple_decision_learning_cycle(self, full_intelligence_system):
        """Test multiple decisions with learning feedback loop"""
        
        # Create a series of related decisions to test learning
        base_context = BusinessContext(
            industry="fintech",
            business_unit="payments",
            stakeholders=["ceo", "cto", "head_of_product"],
            constraints=[{"type": "regulatory", "value": "strict"}],
            objectives=[{"type": "market_share", "target": 0.2}],
            current_state={"market_share": 0.05},
            historical_data={"growth_rate": 0.3},
            time_horizon="medium_term",
            budget_constraints={"total": 5000000},
            regulatory_requirements=["pci_dss", "gdpr", "psd2"]
        )
        
        decisions_and_outcomes = []
        
        # Make multiple decisions and learn from outcomes
        for i in range(3):
            # Create options for this iteration
            options = [
                DecisionOption(
                    id=f"option_a_{i}",
                    name=f"Conservative Approach {i+1}",
                    description="Low-risk incremental improvement",
                    expected_outcomes={"growth": 0.1},
                    costs={"development": 500000},
                    benefits={"revenue": 1000000},
                    risks=[{"type": "competitive", "probability": 0.2, "impact": 200000}],
                    implementation_complexity=3.0 + i * 0.5,
                    time_to_implement=60 + i * 10,
                    resource_requirements={"team": 10 + i * 2}
                ),
                DecisionOption(
                    id=f"option_b_{i}",
                    name=f"Aggressive Strategy {i+1}",
                    description="High-risk, high-reward approach",
                    expected_outcomes={"growth": 0.3},
                    costs={"development": 1500000},
                    benefits={"revenue": 4000000},
                    risks=[
                        {"type": "market", "probability": 0.4, "impact": 1000000},
                        {"type": "technical", "probability": 0.3, "impact": 800000}
                    ],
                    implementation_complexity=7.0 + i * 0.5,
                    time_to_implement=120 + i * 15,
                    resource_requirements={"team": 25 + i * 5}
                )
            ]
            
            # Make decision
            decision = await full_intelligence_system.make_decision(base_context, options)
            
            # Simulate outcome (gradually improving success as system learns)
            base_success = 0.6 + (i * 0.1)  # Learning improves outcomes
            outcome = {
                "success_score": base_success + (0.1 if decision.selected_option.id.endswith("a") else 0.05),
                "financial_impact": decision.selected_option.benefits.get("revenue", 0) * base_success,
                "roi_achieved": 1.5 + (i * 0.2),
                "lessons_learned": [f"Lesson from iteration {i+1}"]
            }
            
            # Learn from outcome
            await full_intelligence_system.learn_from_outcome(decision.id, outcome)
            
            decisions_and_outcomes.append((decision, outcome))
        
        # Verify that we made multiple decisions and learned from each
        assert len(decisions_and_outcomes) == 3
        
        # Verify that each decision was valid
        for decision, outcome in decisions_and_outcomes:
            assert isinstance(decision, Decision)
            assert decision.selected_option is not None
            assert outcome["success_score"] > 0
    
    @pytest.mark.asyncio
    async def test_high_complexity_scenario(self, full_intelligence_system):
        """Test intelligence engine with highly complex business scenario"""
        
        # Create a complex multi-stakeholder, multi-constraint scenario
        complex_context = BusinessContext(
            industry="healthcare",
            business_unit="digital_therapeutics",
            stakeholders=[
                "ceo", "cmo", "head_of_clinical", "regulatory_affairs",
                "data_science_lead", "patient_advocacy", "investors"
            ],
            constraints=[
                {"type": "regulatory_timeline", "value": 730, "criticality": "high"},
                {"type": "clinical_trial_budget", "value": 3000000, "criticality": "high"},
                {"type": "patient_safety", "value": "paramount", "criticality": "critical"},
                {"type": "data_privacy", "value": "hipaa_gdpr", "criticality": "high"},
                {"type": "market_window", "value": 18, "criticality": "medium"},
                {"type": "competitive_pressure", "value": "intense", "criticality": "medium"}
            ],
            objectives=[
                {
                    "type": "regulatory_approval",
                    "target": "fda_ce_mark",
                    "timeline": "24_months",
                    "priority": "critical"
                },
                {
                    "type": "clinical_efficacy",
                    "target": 0.85,
                    "timeline": "18_months",
                    "priority": "high"
                },
                {
                    "type": "market_penetration",
                    "target": 0.1,
                    "timeline": "36_months",
                    "priority": "medium"
                },
                {
                    "type": "revenue_target",
                    "target": 50000000,
                    "timeline": "60_months",
                    "priority": "high"
                }
            ],
            current_state={
                "development_stage": "prototype",
                "clinical_data": "preclinical",
                "regulatory_status": "pre_submission",
                "team_size": 45,
                "funding_runway": 18,
                "ip_portfolio": 12,
                "partnership_status": "exploring"
            },
            historical_data={
                "previous_approvals": 0,
                "clinical_success_rate": 0.7,
                "average_development_time": 36,
                "competitor_approval_rate": 0.4,
                "market_growth_rate": 0.25
            },
            time_horizon="long_term",
            budget_constraints={
                "total": 10000000,
                "clinical_trials": 4000000,
                "regulatory": 1500000,
                "development": 3000000,
                "operations": 1500000
            },
            regulatory_requirements=[
                "fda_510k", "ce_mark_mdd", "hipaa", "gdpr",
                "iso_14155", "iso_27001", "gcp", "clinical_trial_regulation"
            ]
        )
        
        # Create complex decision options
        complex_options = [
            DecisionOption(
                id="accelerated_pathway",
                name="Accelerated Regulatory Pathway",
                description="Pursue breakthrough device designation and accelerated approval",
                expected_outcomes={
                    "approval_probability": 0.7,
                    "time_to_market": 18,
                    "market_advantage": 0.8,
                    "regulatory_risk": 0.4
                },
                costs={
                    "regulatory_consulting": 800000,
                    "accelerated_trials": 2500000,
                    "fda_interactions": 300000,
                    "quality_systems": 400000
                },
                benefits={
                    "first_mover_advantage": 15000000,
                    "premium_pricing": 8000000,
                    "partnership_value": 5000000
                },
                risks=[
                    {"type": "regulatory_rejection", "probability": 0.3, "impact": 5000000},
                    {"type": "clinical_failure", "probability": 0.25, "impact": 4000000},
                    {"type": "competitive_response", "probability": 0.6, "impact": 3000000}
                ],
                implementation_complexity=9.0,
                time_to_implement=540,
                resource_requirements={
                    "regulatory_specialists": 8,
                    "clinical_researchers": 15,
                    "data_scientists": 10,
                    "quality_engineers": 6,
                    "budget": 4000000,
                    "clinical_sites": 25
                }
            ),
            DecisionOption(
                id="traditional_pathway",
                name="Traditional Development Pathway",
                description="Follow standard regulatory pathway with comprehensive trials",
                expected_outcomes={
                    "approval_probability": 0.85,
                    "time_to_market": 30,
                    "market_advantage": 0.5,
                    "regulatory_risk": 0.2
                },
                costs={
                    "comprehensive_trials": 3500000,
                    "regulatory_standard": 500000,
                    "extended_development": 1000000,
                    "quality_systems": 300000
                },
                benefits={
                    "high_approval_confidence": 10000000,
                    "comprehensive_data": 3000000,
                    "regulatory_precedent": 2000000
                },
                risks=[
                    {"type": "market_timing", "probability": 0.4, "impact": 8000000},
                    {"type": "competitive_preemption", "probability": 0.5, "impact": 6000000},
                    {"type": "funding_gap", "probability": 0.3, "impact": 2000000}
                ],
                implementation_complexity=6.5,
                time_to_implement=900,
                resource_requirements={
                    "clinical_researchers": 20,
                    "regulatory_specialists": 5,
                    "data_scientists": 8,
                    "budget": 5300000,
                    "clinical_sites": 40
                }
            ),
            DecisionOption(
                id="partnership_strategy",
                name="Strategic Partnership Development",
                description="Partner with established healthcare company for co-development",
                expected_outcomes={
                    "approval_probability": 0.8,
                    "time_to_market": 24,
                    "market_advantage": 0.6,
                    "partnership_success": 0.7
                },
                costs={
                    "partnership_development": 500000,
                    "shared_trials": 2000000,
                    "integration_costs": 300000,
                    "revenue_sharing": 0.4  # 40% revenue share
                },
                benefits={
                    "reduced_risk": 3000000,
                    "market_access": 8000000,
                    "regulatory_expertise": 2000000,
                    "funding_support": 4000000
                },
                risks=[
                    {"type": "partnership_failure", "probability": 0.3, "impact": 3000000},
                    {"type": "control_loss", "probability": 0.4, "impact": 2000000},
                    {"type": "ip_disputes", "probability": 0.2, "impact": 1500000}
                ],
                implementation_complexity=7.5,
                time_to_implement=180,
                resource_requirements={
                    "business_development": 5,
                    "legal_specialists": 3,
                    "clinical_liaison": 8,
                    "budget": 2800000,
                    "partnership_management": True
                }
            )
        ]
        
        # Make decision on complex scenario
        complex_decision = await full_intelligence_system.make_decision(complex_context, complex_options)
        
        # Verify complex decision was handled successfully
        assert isinstance(complex_decision, Decision)
        assert complex_decision.selected_option in complex_options
        assert len(complex_decision.reasoning) > 0
        assert complex_decision.risk_assessment is not None
        
        # The system should handle high complexity scenarios
        # by providing detailed reasoning and risk analysis
        assert len(complex_decision.reasoning) >= 3  # Should have substantial reasoning
        assert complex_decision.confidence is not None
        
        # Verify that the decision considers the complexity appropriately
        # (e.g., higher complexity might lead to more conservative choices or lower confidence)
        selected_complexity = complex_decision.selected_option.implementation_complexity
        assert 0 <= selected_complexity <= 10


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])