"""
Integration Tests for Strategic Planning Implementation

This module tests the complete strategic planning system including
roadmap creation, investment optimization, and competitive analysis.
"""

import pytest
import asyncio
from datetime import datetime, date, timedelta
from typing import List, Dict, Any

from scrollintel.engines.strategic_planner import StrategicPlanner
from scrollintel.engines.industry_disruption_predictor import IndustryDisruptionPredictor
from scrollintel.engines.competitive_intelligence_analyzer import CompetitiveIntelligenceAnalyzer
from scrollintel.engines.technology_investment_optimizer import TechnologyInvestmentOptimizer
from scrollintel.models.strategic_planning_models import (
    TechnologyVision, TechnologyBet, TechnologyDomain, InvestmentRisk,
    MarketImpact, MarketChange
)


class TestStrategicPlanningIntegration:
    """Test strategic planning system integration"""
    
    @pytest.fixture
    def strategic_planner(self):
        """Create strategic planner instance"""
        return StrategicPlanner()
    
    @pytest.fixture
    def disruption_predictor(self):
        """Create disruption predictor instance"""
        return IndustryDisruptionPredictor()
    
    @pytest.fixture
    def competitive_analyzer(self):
        """Create competitive analyzer instance"""
        return CompetitiveIntelligenceAnalyzer()
    
    @pytest.fixture
    def investment_optimizer(self):
        """Create investment optimizer instance"""
        return TechnologyInvestmentOptimizer()
    
    @pytest.fixture
    def sample_vision(self):
        """Create sample technology vision"""
        return TechnologyVision(
            id="vision_test_001",
            title="AI-First Technology Leadership",
            description="Establish market leadership through AI innovation",
            time_horizon=10,
            key_principles=[
                "AI-first approach to all products",
                "Ethical AI development",
                "Human-AI collaboration"
            ],
            strategic_objectives=[
                "Achieve 40% market share in AI services",
                "Develop breakthrough AI capabilities",
                "Build sustainable competitive advantages"
            ],
            success_criteria=[
                "Revenue growth > 25% annually",
                "AI patent portfolio > 1000 patents",
                "Customer satisfaction > 90%"
            ],
            market_assumptions=[
                "AI market grows 30% annually",
                "Regulatory environment remains favorable",
                "Talent availability improves"
            ]
        )
    
    @pytest.fixture
    def sample_technology_bets(self):
        """Create sample technology investment bets"""
        return [
            TechnologyBet(
                id="bet_ai_001",
                name="Advanced AI Research",
                description="Investment in next-generation AI capabilities",
                domain=TechnologyDomain.ARTIFICIAL_INTELLIGENCE,
                investment_amount=2e9,  # $2B
                time_horizon=8,
                risk_level=InvestmentRisk.HIGH,
                expected_roi=4.5,
                market_impact=MarketImpact.REVOLUTIONARY,
                competitive_advantage=0.85,
                technical_feasibility=0.70,
                market_readiness=0.60,
                regulatory_risk=0.40,
                talent_requirements={"ai_researchers": 500, "ml_engineers": 1000},
                key_milestones=[
                    {"year": 2, "milestone": "Advanced reasoning capabilities"},
                    {"year": 5, "milestone": "Human-level performance"},
                    {"year": 8, "milestone": "Autonomous research systems"}
                ],
                success_metrics=["AI benchmark scores", "Patent filings", "Revenue impact"],
                dependencies=["quantum_computing", "advanced_hardware"],
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            TechnologyBet(
                id="bet_quantum_001",
                name="Quantum Computing Platform",
                description="Development of practical quantum systems",
                domain=TechnologyDomain.QUANTUM_COMPUTING,
                investment_amount=1.5e9,  # $1.5B
                time_horizon=12,
                risk_level=InvestmentRisk.EXTREME,
                expected_roi=8.0,
                market_impact=MarketImpact.REVOLUTIONARY,
                competitive_advantage=0.95,
                technical_feasibility=0.45,
                market_readiness=0.30,
                regulatory_risk=0.20,
                talent_requirements={"quantum_physicists": 200, "quantum_engineers": 300},
                key_milestones=[
                    {"year": 3, "milestone": "100-qubit stable system"},
                    {"year": 7, "milestone": "Quantum advantage demonstration"},
                    {"year": 12, "milestone": "Commercial quantum platform"}
                ],
                success_metrics=["Quantum volume", "Error rates", "Commercial applications"],
                dependencies=["advanced_materials", "cryogenic_systems"],
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            TechnologyBet(
                id="bet_robotics_001",
                name="Autonomous Robotics Systems",
                description="Advanced robotics for industrial automation",
                domain=TechnologyDomain.ROBOTICS,
                investment_amount=1.2e9,  # $1.2B
                time_horizon=6,
                risk_level=InvestmentRisk.MEDIUM,
                expected_roi=3.5,
                market_impact=MarketImpact.TRANSFORMATIVE,
                competitive_advantage=0.75,
                technical_feasibility=0.80,
                market_readiness=0.70,
                regulatory_risk=0.35,
                talent_requirements={"robotics_engineers": 400, "ai_specialists": 300},
                key_milestones=[
                    {"year": 2, "milestone": "Prototype systems"},
                    {"year": 4, "milestone": "Commercial deployment"},
                    {"year": 6, "milestone": "Market leadership"}
                ],
                success_metrics=["Deployment scale", "Performance metrics", "Market share"],
                dependencies=["ai_advancement", "sensor_technology"],
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        ]
    
    @pytest.mark.asyncio
    async def test_strategic_roadmap_creation(self, strategic_planner, sample_vision):
        """Test comprehensive strategic roadmap creation"""
        
        # Create strategic roadmap
        roadmap = await strategic_planner.create_longterm_roadmap(sample_vision, 10)
        
        # Verify roadmap structure
        assert roadmap is not None
        assert roadmap.name is not None
        assert roadmap.time_horizon == 10
        assert roadmap.vision.title == sample_vision.title
        
        # Verify milestones
        assert len(roadmap.milestones) > 0
        for milestone in roadmap.milestones:
            assert milestone.name is not None
            assert milestone.target_date > date.today()
            assert len(milestone.completion_criteria) > 0
        
        # Verify technology bets
        assert len(roadmap.technology_bets) > 0
        for bet in roadmap.technology_bets:
            assert bet.investment_amount > 0
            assert bet.expected_roi > 0
            assert bet.time_horizon > 0
        
        # Verify success metrics
        assert len(roadmap.success_metrics) > 0
        for metric in roadmap.success_metrics:
            assert metric.target_value > 0
            assert metric.measurement_unit is not None
        
        # Verify resource allocation
        assert roadmap.resource_allocation is not None
        assert len(roadmap.resource_allocation) > 0
        
        # Verify scenario plans
        assert len(roadmap.scenario_plans) > 0
        for scenario in roadmap.scenario_plans:
            assert "name" in scenario
            assert "probability" in scenario
            assert scenario["probability"] > 0
    
    @pytest.mark.asyncio
    async def test_investment_analysis(self, strategic_planner, sample_technology_bets):
        """Test technology investment analysis"""
        
        # Analyze technology investments
        analysis = await strategic_planner.evaluate_technology_bets(sample_technology_bets)
        
        # Verify analysis results
        assert analysis is not None
        assert analysis.total_investment > 0
        assert 0 <= analysis.portfolio_risk <= 1
        assert analysis.expected_return > 0
        assert 0 <= analysis.diversification_score <= 1
        
        # Verify technology coverage
        assert analysis.technology_coverage is not None
        assert len(analysis.technology_coverage) > 0
        
        # Verify time horizon distribution
        assert analysis.time_horizon_distribution is not None
        assert len(analysis.time_horizon_distribution) > 0
        
        # Verify recommendations
        assert len(analysis.recommendations) > 0
        assert len(analysis.optimization_opportunities) > 0
    
    @pytest.mark.asyncio
    async def test_industry_evolution_prediction(self, strategic_planner):
        """Test industry evolution prediction"""
        
        # Predict industry evolution
        forecast = await strategic_planner.predict_industry_evolution("artificial_intelligence", 10)
        
        # Verify forecast structure
        assert forecast is not None
        assert forecast.industry == "artificial_intelligence"
        assert forecast.time_horizon == 10
        
        # Verify growth projections
        assert forecast.growth_projections is not None
        assert len(forecast.growth_projections) > 0
        
        # Verify technology trends
        assert len(forecast.technology_trends) > 0
        
        # Verify market dynamics
        assert forecast.market_dynamics is not None
        assert "market_size_projection" in forecast.market_dynamics
        
        # Verify competitive landscape
        assert forecast.competitive_landscape is not None
        
        # Verify disruption risks
        assert len(forecast.disruption_risks) >= 0  # May be empty for some industries
        
        # Verify investment opportunities
        assert len(forecast.investment_opportunities) > 0
    
    @pytest.mark.asyncio
    async def test_disruption_prediction(self, disruption_predictor):
        """Test industry disruption prediction"""
        
        # Predict disruptions
        disruptions = await disruption_predictor.predict_industry_disruption(
            "artificial_intelligence", 10
        )
        
        # Verify disruption predictions
        assert disruptions is not None
        assert len(disruptions) > 0
        
        for disruption in disruptions:
            assert disruption.industry == "artificial_intelligence"
            assert 0 <= disruption.probability <= 1
            assert disruption.time_horizon > 0
            assert disruption.impact_magnitude > 0
            assert len(disruption.key_drivers) > 0
            assert len(disruption.affected_sectors) > 0
            assert len(disruption.opportunities) > 0
            assert len(disruption.threats) > 0
            assert len(disruption.recommended_actions) > 0
    
    @pytest.mark.asyncio
    async def test_competitive_analysis(self, competitive_analyzer):
        """Test competitive intelligence analysis"""
        
        # Analyze competitor
        intelligence = await competitive_analyzer.analyze_competitor("Google")
        
        # Verify intelligence structure
        assert intelligence is not None
        assert intelligence.competitor_name is not None
        assert intelligence.market_position is not None
        
        # Verify technology capabilities
        assert intelligence.technology_capabilities is not None
        assert len(intelligence.technology_capabilities) > 0
        for domain, capability in intelligence.technology_capabilities.items():
            assert 0 <= capability <= 1
        
        # Verify investment patterns
        assert intelligence.investment_patterns is not None
        assert len(intelligence.investment_patterns) > 0
        
        # Verify strategic moves
        assert len(intelligence.strategic_moves) > 0
        
        # Verify SWOT analysis
        assert len(intelligence.strengths) > 0
        assert len(intelligence.weaknesses) > 0
        assert len(intelligence.opportunities) > 0
        assert len(intelligence.threats) > 0
        
        # Verify predictions and counter-strategies
        assert len(intelligence.predicted_actions) > 0
        assert len(intelligence.counter_strategies) > 0
    
    @pytest.mark.asyncio
    async def test_competitive_landscape_analysis(self, competitive_analyzer):
        """Test competitive landscape analysis"""
        
        # Analyze competitive landscape
        landscape = await competitive_analyzer.analyze_competitive_landscape(
            "artificial_intelligence", ["Google", "Microsoft", "OpenAI"]
        )
        
        # Verify landscape structure
        assert landscape is not None
        assert "landscape_overview" in landscape
        assert "competitor_details" in landscape
        
        overview = landscape["landscape_overview"]
        assert overview["industry"] == "artificial_intelligence"
        assert overview["competitor_count"] == 3
        assert "market_concentration" in overview
        assert "technology_leadership" in overview
        assert "competitive_intensity" in overview
        
        # Verify competitor details
        competitor_details = landscape["competitor_details"]
        assert len(competitor_details) == 3
        for competitor, details in competitor_details.items():
            assert details.competitor_name is not None
            assert details.market_position is not None
    
    @pytest.mark.asyncio
    async def test_investment_optimization(self, investment_optimizer, sample_technology_bets):
        """Test investment portfolio optimization"""
        
        # Optimize investment portfolio
        optimization = await investment_optimizer.optimize_investment_portfolio(
            sample_technology_bets,
            total_budget=5e9,  # $5B
            time_horizon=10,
            risk_tolerance=0.30
        )
        
        # Verify optimization results
        assert optimization is not None
        assert "optimized_allocations" in optimization
        assert "portfolio_metrics" in optimization
        assert "investment_analysis" in optimization
        assert "implementation_plan" in optimization
        
        # Verify allocations
        allocations = optimization["optimized_allocations"]
        assert len(allocations) > 0
        
        total_allocation = sum(alloc["amount"] for alloc in allocations.values())
        assert total_allocation <= 5e9 * 1.01  # Allow 1% tolerance
        
        # Verify portfolio metrics
        metrics = optimization["portfolio_metrics"]
        assert metrics["expected_return"] > 0
        assert 0 <= metrics["portfolio_risk"] <= 1
        assert metrics["total_budget"] == 5e9
        
        # Verify investment analysis
        analysis = optimization["investment_analysis"]
        assert "domain_distribution" in analysis
        assert "risk_distribution" in analysis
        assert "time_distribution" in analysis
        
        # Verify implementation plan
        plan = optimization["implementation_plan"]
        assert "implementation_phases" in plan
        assert "key_milestones" in plan
        assert len(plan["implementation_phases"]) > 0
    
    @pytest.mark.asyncio
    async def test_strategic_pivot_recommendations(self, strategic_planner):
        """Test strategic pivot recommendations"""
        
        # Create sample market changes
        market_changes = [
            MarketChange(
                change_type="AI Regulation",
                description="New AI governance regulations introduced",
                impact_magnitude=8.0,
                affected_markets=["AI services", "Data analytics"],
                time_horizon=2,
                probability=0.75,
                strategic_implications=["Compliance requirements", "Market access changes"]
            ),
            MarketChange(
                change_type="Quantum Breakthrough",
                description="Major quantum computing breakthrough achieved",
                impact_magnitude=9.0,
                affected_markets=["Cryptography", "Optimization", "AI"],
                time_horizon=5,
                probability=0.60,
                strategic_implications=["Technology disruption", "New opportunities"]
            )
        ]
        
        # Generate pivot recommendations
        pivots = await strategic_planner.recommend_strategic_pivots(market_changes)
        
        # Verify pivot recommendations
        assert pivots is not None
        assert len(pivots) > 0
        
        for pivot in pivots:
            assert pivot.name is not None
            assert pivot.description is not None
            assert len(pivot.trigger_conditions) > 0
            assert pivot.implementation_timeline > 0
            assert pivot.resource_requirements is not None
            assert len(pivot.expected_outcomes) > 0
            assert 0 <= pivot.success_probability <= 1
            assert pivot.roi_projection > 0
    
    @pytest.mark.asyncio
    async def test_portfolio_rebalancing(self, investment_optimizer):
        """Test portfolio rebalancing"""
        
        # Sample current portfolio
        current_portfolio = {
            "total_value": 5e9,
            "allocations": {
                "ai_investment": {"amount": 2e9, "performance": 1.25},
                "quantum_investment": {"amount": 1.5e9, "performance": 0.85},
                "robotics_investment": {"amount": 1.5e9, "performance": 1.10}
            }
        }
        
        # Sample market changes
        market_changes = [
            {"type": "regulatory_change", "impact": 0.15, "affected_areas": ["ai"]},
            {"type": "technology_breakthrough", "impact": 0.25, "affected_areas": ["quantum"]}
        ]
        
        # Sample performance data
        performance_data = {
            "ai_investment": {"return": 0.25, "risk": 0.20},
            "quantum_investment": {"return": -0.15, "risk": 0.35},
            "robotics_investment": {"return": 0.10, "risk": 0.15}
        }
        
        # Rebalance portfolio
        rebalancing = await investment_optimizer.rebalance_portfolio(
            current_portfolio, market_changes, performance_data
        )
        
        # Verify rebalancing results
        assert rebalancing is not None
        assert "performance_analysis" in rebalancing
        assert "market_impact_assessment" in rebalancing
        assert "rebalancing_recommendations" in rebalancing
        
        # Verify recommendations
        recommendations = rebalancing["rebalancing_recommendations"]
        assert len(recommendations) > 0
        
        for recommendation in recommendations:
            assert "action" in recommendation
            assert "rationale" in recommendation
            assert "impact" in recommendation
            assert "timeline" in recommendation
    
    @pytest.mark.asyncio
    async def test_end_to_end_strategic_planning(
        self, strategic_planner, disruption_predictor, 
        competitive_analyzer, investment_optimizer,
        sample_vision, sample_technology_bets
    ):
        """Test end-to-end strategic planning workflow"""
        
        # Step 1: Create strategic roadmap
        roadmap = await strategic_planner.create_longterm_roadmap(sample_vision, 10)
        assert roadmap is not None
        
        # Step 2: Analyze technology investments
        investment_analysis = await strategic_planner.evaluate_technology_bets(sample_technology_bets)
        assert investment_analysis is not None
        
        # Step 3: Predict industry disruptions
        disruptions = await disruption_predictor.predict_industry_disruption(
            "artificial_intelligence", 10
        )
        assert len(disruptions) > 0
        
        # Step 4: Analyze competitive landscape
        landscape = await competitive_analyzer.analyze_competitive_landscape(
            "artificial_intelligence", ["Google", "Microsoft", "OpenAI"]
        )
        assert landscape is not None
        
        # Step 5: Optimize investment portfolio
        optimization = await investment_optimizer.optimize_investment_portfolio(
            sample_technology_bets,
            total_budget=5e9,
            time_horizon=10,
            risk_tolerance=0.30
        )
        assert optimization is not None
        
        # Verify integration consistency
        # Technology bets in roadmap should align with optimization inputs
        roadmap_domains = {bet.domain for bet in roadmap.technology_bets}
        optimization_domains = {bet.domain for bet in sample_technology_bets}
        assert len(roadmap_domains.intersection(optimization_domains)) > 0
        
        # Investment analysis should be consistent with optimization
        assert investment_analysis.total_investment > 0
        assert optimization["portfolio_metrics"]["total_budget"] > 0
        
        # Disruption predictions should inform strategic planning
        disruption_industries = {d.industry for d in disruptions}
        assert "artificial_intelligence" in disruption_industries
    
    @pytest.mark.asyncio
    async def test_scenario_analysis_integration(self, investment_optimizer, sample_technology_bets):
        """Test scenario analysis integration"""
        
        # Optimize portfolio with scenario analysis
        optimization = await investment_optimizer.optimize_investment_portfolio(
            sample_technology_bets,
            total_budget=5e9,
            time_horizon=10,
            risk_tolerance=0.30
        )
        
        # Verify scenario analysis
        assert "scenario_analysis" in optimization
        scenario_analysis = optimization["scenario_analysis"]
        
        assert "scenarios" in scenario_analysis
        scenarios = scenario_analysis["scenarios"]
        
        # Should have optimistic, base case, and pessimistic scenarios
        scenario_names = {name for name in scenarios.keys()}
        expected_scenarios = {"optimistic", "base_case", "pessimistic"}
        assert expected_scenarios.issubset(scenario_names)
        
        # Verify scenario probabilities sum to 1.0
        total_probability = sum(
            scenario["probability"] for scenario in scenarios.values()
        )
        assert abs(total_probability - 1.0) < 0.01  # Allow small floating point error
        
        # Verify expected value analysis
        assert "expected_value_analysis" in scenario_analysis
        expected_value = scenario_analysis["expected_value_analysis"]
        assert expected_value["probability_weighted_return"] > 0
        assert expected_value["probability_weighted_risk"] > 0
    
    @pytest.mark.asyncio
    async def test_risk_assessment_integration(self, investment_optimizer, sample_technology_bets):
        """Test risk assessment integration"""
        
        # Optimize portfolio with risk assessment
        optimization = await investment_optimizer.optimize_investment_portfolio(
            sample_technology_bets,
            total_budget=5e9,
            time_horizon=10,
            risk_tolerance=0.25  # Lower risk tolerance
        )
        
        # Verify risk assessment
        assert "risk_assessment" in optimization
        risk_assessment = optimization["risk_assessment"]
        
        assert "total_portfolio_risk" in risk_assessment
        assert risk_assessment["total_portfolio_risk"] <= 0.25 * 1.1  # Allow 10% tolerance
        
        assert "risk_contributions" in risk_assessment
        assert "top_risk_contributors" in risk_assessment
        assert "mitigation_strategies" in risk_assessment
        
        # Verify risk contributions
        risk_contributions = risk_assessment["risk_contributions"]
        assert len(risk_contributions) > 0
        
        # Verify mitigation strategies
        mitigation_strategies = risk_assessment["mitigation_strategies"]
        assert len(mitigation_strategies) > 0
    
    @pytest.mark.asyncio
    async def test_performance_and_scalability(
        self, strategic_planner, investment_optimizer, sample_technology_bets
    ):
        """Test performance and scalability of strategic planning system"""
        
        # Test with larger dataset
        large_investment_set = sample_technology_bets * 10  # 30 investments
        
        # Update IDs to avoid duplicates
        for i, bet in enumerate(large_investment_set):
            bet.id = f"bet_{i:03d}"
        
        start_time = datetime.now()
        
        # Optimize large portfolio
        optimization = await investment_optimizer.optimize_investment_portfolio(
            large_investment_set,
            total_budget=50e9,  # $50B
            time_horizon=15,
            risk_tolerance=0.35
        )
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Verify performance (should complete within reasonable time)
        assert execution_time < 30  # Should complete within 30 seconds
        
        # Verify optimization results are still valid
        assert optimization is not None
        assert len(optimization["optimized_allocations"]) > 0
        
        # Verify scalability - results should be consistent
        metrics = optimization["portfolio_metrics"]
        assert metrics["expected_return"] > 0
        assert 0 <= metrics["portfolio_risk"] <= 1
        assert metrics["total_budget"] == 50e9