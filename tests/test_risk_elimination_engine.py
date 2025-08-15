"""
Test suite for Risk Elimination Engine.

Tests all components of the risk elimination system including:
- Multi-dimensional risk analysis
- Redundant mitigation strategies
- Predictive risk modeling
- Adaptive response framework
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.risk_elimination_engine import (
    RiskEliminationEngine,
    Risk,
    MitigationStrategy,
    RiskCategory,
    RiskSeverity,
    RiskStatus,
    StrategyType,
    ImplementationStatus,
    TechnicalRiskAnalyzer,
    MarketRiskAnalyzer,
    FinancialRiskAnalyzer,
    RegulatoryRiskAnalyzer,
    ExecutionRiskAnalyzer,
    CompetitiveRiskAnalyzer,
    TalentRiskAnalyzer,
    TimingRiskAnalyzer,
    PredictiveRiskModel,
    MitigationStrategyGenerator,
    AdaptiveResponseFramework,
    create_risk_elimination_engine
)


class TestRiskEliminationEngine:
    """Test the main Risk Elimination Engine."""
    
    @pytest.fixture
    def engine(self):
        """Create a Risk Elimination Engine instance for testing."""
        return create_risk_elimination_engine()
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, engine):
        """Test that the engine initializes with all required components."""
        assert len(engine.risk_analyzers) == 8
        assert engine.predictive_model is not None
        assert engine.strategy_generator is not None
        assert engine.adaptive_framework is not None
        assert isinstance(engine.identified_risks, dict)
        assert isinstance(engine.mitigation_strategies, dict)
        assert isinstance(engine.active_mitigations, list)
    
    @pytest.mark.asyncio
    async def test_analyze_all_risks(self, engine):
        """Test comprehensive risk analysis across all categories."""
        risks_by_category = await engine.analyze_all_risks()
        
        # Should have risks for each category
        assert len(risks_by_category) == 8
        
        # Check that risks are properly categorized
        expected_categories = [cat.value for cat in RiskCategory]
        for category in expected_categories:
            assert category in risks_by_category
            
        # Verify risks are stored in engine
        assert len(engine.identified_risks) > 0
        assert engine.last_analysis_time is not None
    
    @pytest.mark.asyncio
    async def test_generate_mitigation_strategies(self, engine):
        """Test generation of redundant mitigation strategies."""
        # First analyze risks
        await engine.analyze_all_risks()
        
        # Generate strategies
        strategies = await engine.generate_mitigation_strategies()
        
        # Should have strategies for each risk
        assert len(strategies) == len(engine.identified_risks)
        
        # Each risk should have 3-5 strategies
        for risk_id, risk_strategies in strategies.items():
            assert 3 <= len(risk_strategies) <= 5
            
            # Should have backup strategies
            primary_strategies = [s for s in risk_strategies if s.priority <= 3]
            backup_strategies = [s for s in risk_strategies if s.priority > 3]
            
            assert len(primary_strategies) >= 3
            assert len(backup_strategies) >= 2
            
            # Primary strategies should reference backups
            for strategy in primary_strategies:
                assert len(strategy.backup_strategies) > 0
    
    @pytest.mark.asyncio
    async def test_deploy_mitigation_strategies(self, engine):
        """Test deployment of mitigation strategies."""
        # Setup: analyze risks and generate strategies
        await engine.analyze_all_risks()
        await engine.generate_mitigation_strategies()
        
        # Deploy strategies for first risk
        risk_id = list(engine.identified_risks.keys())[0]
        result = await engine.deploy_mitigation_strategies(risk_id)
        
        assert result["risk_id"] == risk_id
        assert result["strategies_deployed"] > 0
        assert len(result["deployment_results"]) > 0
        assert result["backup_strategies_available"] >= 2
        
        # Check that strategies are marked as active
        active_count = len([s for s in engine.active_mitigations 
                           if s.risk_id == risk_id])
        assert active_count > 0
    
    @pytest.mark.asyncio
    async def test_predict_future_risks(self, engine):
        """Test predictive risk modeling."""
        # Setup: analyze current risks
        await engine.analyze_all_risks()
        
        # Predict future risks
        prediction = await engine.predict_future_risks(timeframe_days=30)
        
        assert prediction["prediction_timeframe_days"] == 30
        assert "new_risks_predicted" in prediction
        assert "future_risks" in prediction
        assert "risk_evolution_forecast" in prediction
        assert prediction["model_accuracy"] > 0.8
        
        # Check forecast structure
        forecast = prediction["risk_evolution_forecast"]
        assert "timeframe_days" in forecast
        assert "risk_evolution" in forecast
        assert "new_risks_predicted" in forecast
    
    @pytest.mark.asyncio
    async def test_adaptive_response(self, engine):
        """Test adaptive response framework."""
        # Setup: create high-probability risk
        high_risk = Risk(
            id="test_high_risk",
            category=RiskCategory.TECHNICAL,
            description="High probability test risk",
            probability=0.85,  # Above threshold
            impact=0.8,
            severity=RiskSeverity.CRITICAL
        )
        engine.identified_risks[high_risk.id] = high_risk
        
        # Execute adaptive response
        response = await engine.execute_adaptive_response()
        
        assert "adaptations_executed" in response
        assert "adaptation_results" in response
        assert "strategies_adapted" in response
        assert response["response_time_seconds"] <= 1.0  # Real-time response
    
    @pytest.mark.asyncio
    async def test_success_probability_calculation(self, engine):
        """Test success probability calculation."""
        # Test with no risks (should be 100%)
        prob = await engine.calculate_success_probability()
        assert prob == 1.0
        
        # Add risks and strategies
        await engine.analyze_all_risks()
        await engine.generate_mitigation_strategies()
        
        # Deploy some strategies
        for risk_id in list(engine.identified_risks.keys())[:2]:
            await engine.deploy_mitigation_strategies(risk_id)
        
        # Calculate probability
        prob = await engine.calculate_success_probability()
        assert 0.0 <= prob <= 1.0
        assert engine.risk_elimination_rate >= 0.0
    
    @pytest.mark.asyncio
    async def test_complete_risk_elimination_cycle(self, engine):
        """Test complete risk elimination cycle."""
        result = await engine.run_complete_risk_elimination_cycle()
        
        # Verify cycle completion
        assert result["cycle_completed"] is True
        assert result["cycle_duration_seconds"] > 0
        
        # Verify risk analysis
        risks_analyzed = result["risks_analyzed"]
        assert risks_analyzed["total"] > 0
        assert len(risks_analyzed["by_category"]) == 8
        
        # Verify mitigation strategies
        mitigation = result["mitigation_strategies"]
        assert mitigation["total_generated"] > 0
        assert mitigation["deployed"] >= 0
        assert mitigation["backup_available"] >= 0
        
        # Verify success probability
        assert 0.0 <= result["final_success_probability"] <= 1.0
        
        # Should achieve high success probability
        assert result["final_success_probability"] > 0.7
    
    @pytest.mark.asyncio
    async def test_engine_status(self, engine):
        """Test engine status reporting."""
        # Run some operations first
        await engine.analyze_all_risks()
        await engine.generate_mitigation_strategies()
        
        status = await engine.get_engine_status()
        
        assert status["engine_status"] == "active"
        assert status["risk_analyzers_active"] == 8
        assert status["total_risks_identified"] > 0
        assert status["total_strategies_generated"] > 0
        assert 0.0 <= status["success_probability"] <= 1.0
        assert status["predictive_model_accuracy"] > 0.8


class TestRiskAnalyzers:
    """Test individual risk analyzers."""
    
    @pytest.mark.asyncio
    async def test_technical_risk_analyzer(self):
        """Test technical risk analysis."""
        analyzer = TechnicalRiskAnalyzer()
        
        assert analyzer.category == RiskCategory.TECHNICAL
        assert analyzer.active is True
        
        risks = await analyzer.analyze_risks()
        assert len(risks) > 0
        
        for risk in risks:
            assert risk.category == RiskCategory.TECHNICAL
            assert 0.0 <= risk.probability <= 1.0
            assert 0.0 <= risk.impact <= 1.0
            assert risk.severity in RiskSeverity
        
        future_risks = await analyzer.predict_future_risks(30)
        assert isinstance(future_risks, list)
    
    @pytest.mark.asyncio
    async def test_market_risk_analyzer(self):
        """Test market risk analysis."""
        analyzer = MarketRiskAnalyzer()
        
        assert analyzer.category == RiskCategory.MARKET
        
        risks = await analyzer.analyze_risks()
        assert len(risks) > 0
        
        for risk in risks:
            assert risk.category == RiskCategory.MARKET
            assert risk.severity in [RiskSeverity.HIGH, RiskSeverity.CRITICAL]
    
    @pytest.mark.asyncio
    async def test_financial_risk_analyzer(self):
        """Test financial risk analysis."""
        analyzer = FinancialRiskAnalyzer()
        
        assert analyzer.category == RiskCategory.FINANCIAL
        
        risks = await analyzer.analyze_risks()
        assert len(risks) > 0
        
        # Financial risks should be critical
        critical_risks = [r for r in risks if r.severity == RiskSeverity.CRITICAL]
        assert len(critical_risks) > 0
    
    @pytest.mark.asyncio
    async def test_all_risk_analyzers(self):
        """Test that all 8 risk analyzers work correctly."""
        analyzers = [
            TechnicalRiskAnalyzer(),
            MarketRiskAnalyzer(),
            FinancialRiskAnalyzer(),
            RegulatoryRiskAnalyzer(),
            ExecutionRiskAnalyzer(),
            CompetitiveRiskAnalyzer(),
            TalentRiskAnalyzer(),
            TimingRiskAnalyzer()
        ]
        
        assert len(analyzers) == 8
        
        for analyzer in analyzers:
            risks = await analyzer.analyze_risks()
            assert len(risks) > 0
            
            for risk in risks:
                assert risk.category == analyzer.category
                assert risk.id.startswith(analyzer.category.value)


class TestPredictiveRiskModel:
    """Test predictive risk modeling."""
    
    @pytest.fixture
    def model(self):
        return PredictiveRiskModel()
    
    @pytest.fixture
    def sample_risk(self):
        return Risk(
            id="test_risk",
            category=RiskCategory.TECHNICAL,
            description="Test risk",
            probability=0.5,
            impact=0.7,
            severity=RiskSeverity.MEDIUM
        )
    
    @pytest.mark.asyncio
    async def test_predict_risk_manifestation(self, model, sample_risk):
        """Test risk manifestation prediction."""
        manifestation_date, probability = await model.predict_risk_manifestation(sample_risk)
        
        assert isinstance(manifestation_date, datetime)
        assert manifestation_date > datetime.now()
        assert 0.0 <= probability <= 1.0
        assert probability >= sample_risk.probability  # Should be adjusted upward
    
    @pytest.mark.asyncio
    async def test_forecast_risk_evolution(self, model, sample_risk):
        """Test risk evolution forecasting."""
        risks = [sample_risk]
        forecast = await model.forecast_risk_evolution(risks, days=30)
        
        assert forecast["timeframe_days"] == 30
        assert "risk_evolution" in forecast
        assert sample_risk.id in forecast["risk_evolution"]
        
        evolution = forecast["risk_evolution"][sample_risk.id]
        assert "current_probability" in evolution
        assert "predicted_probability" in evolution
        assert "trend" in evolution


class TestMitigationStrategyGenerator:
    """Test mitigation strategy generation."""
    
    @pytest.fixture
    def generator(self):
        return MitigationStrategyGenerator()
    
    @pytest.fixture
    def sample_risks(self):
        return [
            Risk(
                id="tech_risk",
                category=RiskCategory.TECHNICAL,
                description="Technical risk",
                probability=0.6,
                impact=0.8,
                severity=RiskSeverity.HIGH
            ),
            Risk(
                id="market_risk",
                category=RiskCategory.MARKET,
                description="Market risk",
                probability=0.7,
                impact=0.9,
                severity=RiskSeverity.CRITICAL
            )
        ]
    
    @pytest.mark.asyncio
    async def test_generate_strategies(self, generator, sample_risks):
        """Test strategy generation for different risk types."""
        for risk in sample_risks:
            strategies = await generator.generate_strategies(risk)
            
            # Should generate 3-5 strategies
            assert 3 <= len(strategies) <= 5
            
            # Should have primary and backup strategies
            primary = [s for s in strategies if s.priority <= 3]
            backup = [s for s in strategies if s.priority > 3]
            
            assert len(primary) >= 3
            assert len(backup) >= 2
            
            # Primary strategies should reference backups
            for strategy in primary:
                assert len(strategy.backup_strategies) > 0
                assert all(backup_id in [b.id for b in backup] 
                          for backup_id in strategy.backup_strategies)
            
            # All strategies should be for the correct risk
            for strategy in strategies:
                assert strategy.risk_id == risk.id
                assert strategy.effectiveness_score > 0.0
                assert strategy.resources_required is not None


class TestAdaptiveResponseFramework:
    """Test adaptive response framework."""
    
    @pytest.fixture
    def framework(self):
        return AdaptiveResponseFramework()
    
    @pytest.fixture
    def high_risk(self):
        return Risk(
            id="high_risk",
            category=RiskCategory.TECHNICAL,
            description="High probability risk",
            probability=0.85,  # Above threshold
            impact=0.8,
            severity=RiskSeverity.CRITICAL
        )
    
    @pytest.fixture
    def low_risk(self):
        return Risk(
            id="low_risk",
            category=RiskCategory.MARKET,
            description="Low probability risk",
            probability=0.3,  # Below threshold
            impact=0.5,
            severity=RiskSeverity.LOW
        )
    
    @pytest.mark.asyncio
    async def test_monitor_risk_changes(self, framework, high_risk, low_risk):
        """Test risk change monitoring."""
        risks = [high_risk, low_risk]
        adaptations = await framework.monitor_risk_changes(risks)
        
        # Should detect high risk needing adaptation
        assert len(adaptations) >= 1
        
        high_risk_adaptation = next(
            (a for a in adaptations if a["risk_id"] == high_risk.id), 
            None
        )
        assert high_risk_adaptation is not None
        assert high_risk_adaptation["trigger"] == "probability_threshold_exceeded"
        assert high_risk_adaptation["urgency"] == "high"
    
    @pytest.mark.asyncio
    async def test_adapt_strategies(self, framework, high_risk):
        """Test strategy adaptation."""
        # Create sample strategies
        strategies = [
            MitigationStrategy(
                id="strategy_1",
                risk_id=high_risk.id,
                strategy_type=StrategyType.PREVENTIVE,
                description="Test strategy",
                implementation_plan="Test plan",
                resources_required={"budget": 100000, "personnel": 5},
                effectiveness_score=0.6,
                priority=1
            )
        ]
        
        adapted = await framework.adapt_strategies(high_risk, strategies)
        
        # Should increase resources for high-probability risk
        assert adapted[0].resources_required["budget"] > 100000
        assert adapted[0].resources_required["personnel"] > 5
    
    @pytest.mark.asyncio
    async def test_execute_real_time_adjustment(self, framework):
        """Test real-time adjustment execution."""
        adaptation = {
            "risk_id": "test_risk",
            "trigger": "probability_threshold_exceeded",
            "recommended_action": "escalate_mitigation_strategies",
            "urgency": "high"
        }
        
        result = await framework.execute_real_time_adjustment(adaptation)
        
        assert result["risk_id"] == "test_risk"
        assert result["action_taken"] == "escalate_mitigation_strategies"
        assert result["success"] is True
        assert isinstance(result["timestamp"], datetime)
        
        # Should be recorded in history
        assert len(framework.adaptation_history) == 1


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_risk_elimination(self):
        """Test complete end-to-end risk elimination process."""
        engine = create_risk_elimination_engine()
        
        # Run complete cycle
        result = await engine.run_complete_risk_elimination_cycle()
        
        # Verify comprehensive risk elimination
        assert result["cycle_completed"] is True
        assert result["risks_analyzed"]["total"] > 0
        assert result["mitigation_strategies"]["total_generated"] > 0
        
        # Should achieve high success probability
        assert result["final_success_probability"] > 0.8
        
        # Should have backup strategies available
        assert result["mitigation_strategies"]["backup_available"] > 0
        
        # Adaptive response should be functional
        assert result["adaptive_response"]["adaptations_executed"] >= 0
    
    @pytest.mark.asyncio
    async def test_redundancy_and_failover(self):
        """Test redundancy and failover capabilities."""
        engine = create_risk_elimination_engine()
        
        # Analyze risks and generate strategies
        await engine.analyze_all_risks()
        await engine.generate_mitigation_strategies()
        
        # Verify redundancy
        for risk_id, strategies in engine.mitigation_strategies.items():
            # Should have multiple strategies per risk
            assert len(strategies) >= 3
            
            # Should have backup strategies
            backup_count = len([s for s in strategies if s.priority > 3])
            assert backup_count >= 2
            
            # Primary strategies should reference backups
            primary_strategies = [s for s in strategies if s.priority <= 3]
            for strategy in primary_strategies:
                assert len(strategy.backup_strategies) > 0
    
    @pytest.mark.asyncio
    async def test_real_time_adaptation(self):
        """Test real-time adaptation capabilities."""
        engine = create_risk_elimination_engine()
        
        # Create dynamic risk scenario
        await engine.analyze_all_risks()
        
        # Simulate risk probability increase
        for risk in engine.identified_risks.values():
            risk.probability = min(risk.probability * 1.5, 1.0)
        
        # Execute adaptive response
        response = await engine.execute_adaptive_response()
        
        # Should detect and respond to changes
        assert response["response_time_seconds"] <= 1.0
        
        # Should adapt strategies if needed
        if response["adaptations_executed"] > 0:
            assert response["strategies_adapted"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])