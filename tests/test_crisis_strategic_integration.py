"""
Tests for Crisis-Strategic Planning Integration

This module tests the integration between crisis leadership capabilities
and strategic planning systems.
"""

import pytest
import asyncio
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.crisis_strategic_integration import (
    CrisisStrategicIntegration, CrisisImpactLevel, CrisisStrategicImpact,
    CrisisAwareAdjustment, RecoveryIntegrationPlan
)
from scrollintel.models.crisis_detection_models import Crisis, CrisisType, SeverityLevel
from scrollintel.models.strategic_planning_models import (
    StrategicRoadmap, TechnologyBet, StrategicMilestone, TechnologyVision,
    RiskAssessment, SuccessMetric, TechnologyDomain, InvestmentRisk,
    MarketImpact, CompetitivePosition
)


class TestCrisisStrategicIntegration:
    """Test suite for crisis-strategic integration functionality"""
    
    @pytest.fixture
    def integration_engine(self):
        """Create integration engine for testing"""
        return CrisisStrategicIntegration()
    
    @pytest.fixture
    def sample_crisis(self):
        """Create sample crisis for testing"""
        return Crisis(
            id="crisis_001",
            crisis_type=CrisisType.SECURITY_BREACH,
            severity_level=SeverityLevel.HIGH,
            title="Major Security Incident",
            description="Critical security breach affecting customer data",
            start_time=datetime.now(),
            affected_areas=["security", "customer_data", "operations"],
            stakeholders_impacted=["customers", "employees", "regulators"],
            current_status="active",
            response_actions=[],
            estimated_resolution_time=72
        )
    
    @pytest.fixture
    def sample_strategic_roadmap(self):
        """Create sample strategic roadmap for testing"""
        vision = TechnologyVision(
            id="vision_001",
            title="AI-First Technology Leadership",
            description="Become the leading AI technology company",
            time_horizon=10,
            key_principles=["innovation", "security", "scalability"],
            strategic_objectives=["AI leadership", "market expansion"],
            success_criteria=["market share > 30%", "AI patents > 1000"],
            market_assumptions=["AI adoption accelerates", "Regulatory stability"]
        )
        
        milestone = StrategicMilestone(
            id="milestone_001",
            name="AI Platform Launch",
            description="Launch next-generation AI platform",
            target_date=date.today() + timedelta(days=180),
            completion_criteria=["Platform deployed", "Customer onboarding"],
            success_metrics=["User adoption rate", "Performance benchmarks"],
            dependencies=["ai_research", "infrastructure"],
            risk_factors=["technical_challenges", "market_readiness"],
            resource_requirements={"budget": 500e6, "headcount": 200}
        )
        
        tech_bet = TechnologyBet(
            id="bet_001",
            name="Advanced AI Research",
            description="Investment in next-gen AI capabilities",
            domain=TechnologyDomain.ARTIFICIAL_INTELLIGENCE,
            investment_amount=2e9,
            time_horizon=5,
            risk_level=InvestmentRisk.HIGH,
            expected_roi=4.0,
            market_impact=MarketImpact.TRANSFORMATIVE,
            competitive_advantage=0.8,
            technical_feasibility=0.7,
            market_readiness=0.6,
            regulatory_risk=0.3,
            talent_requirements={"ai_researchers": 100, "ml_engineers": 200},
            key_milestones=[{"year": 2, "milestone": "Breakthrough discovery"}],
            success_metrics=["Research publications", "Patent filings"],
            dependencies=["quantum_computing"],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        risk_assessment = RiskAssessment(
            id="risk_001",
            risk_type="Technology Risk",
            description="AI technology may not mature as expected",
            probability=0.3,
            impact=0.7,
            mitigation_strategies=["Diversified research", "External partnerships"],
            contingency_plans=["Alternative technologies", "Acquisition strategy"],
            monitoring_indicators=["Research progress", "Competitive advances"]
        )
        
        success_metric = SuccessMetric(
            id="metric_001",
            name="AI Market Share",
            description="Market share in AI services",
            target_value=0.3,
            current_value=0.15,
            measurement_unit="percentage",
            measurement_frequency="quarterly",
            data_source="market_research"
        )
        
        return StrategicRoadmap(
            id="roadmap_001",
            name="10-Year AI Strategy",
            description="Comprehensive AI technology roadmap",
            vision=vision,
            time_horizon=10,
            milestones=[milestone],
            technology_bets=[tech_bet],
            risk_assessments=[risk_assessment],
            success_metrics=[success_metric],
            competitive_positioning=CompetitivePosition.LEADER,
            market_assumptions=["AI growth continues", "Talent availability"],
            resource_allocation={"R&D": 0.4, "Operations": 0.6},
            scenario_plans=[],
            review_schedule=[date.today() + timedelta(days=90)],
            stakeholders=["CTO", "CEO", "Board"],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    @pytest.mark.asyncio
    async def test_assess_crisis_impact_on_strategy(
        self, integration_engine, sample_crisis, sample_strategic_roadmap
    ):
        """Test crisis impact assessment on strategic plans"""
        
        # Assess impact
        impact_assessment = await integration_engine.assess_crisis_impact_on_strategy(
            sample_crisis, sample_strategic_roadmap
        )
        
        # Verify assessment structure
        assert isinstance(impact_assessment, CrisisStrategicImpact)
        assert impact_assessment.crisis_id == sample_crisis.id
        assert impact_assessment.strategic_plan_id == sample_strategic_roadmap.id
        assert isinstance(impact_assessment.impact_level, CrisisImpactLevel)
        
        # Verify impact analysis
        assert impact_assessment.resource_reallocation_needed > 0
        assert len(impact_assessment.strategic_recommendations) > 0
        assert impact_assessment.recovery_timeline > 0
        
        # Security breach should have significant impact
        assert impact_assessment.impact_level in [
            CrisisImpactLevel.SIGNIFICANT, CrisisImpactLevel.SEVERE
        ]
    
    @pytest.mark.asyncio
    async def test_generate_crisis_aware_adjustments(
        self, integration_engine, sample_crisis, sample_strategic_roadmap
    ):
        """Test generation of crisis-aware strategic adjustments"""
        
        # First get impact assessment
        impact_assessment = await integration_engine.assess_crisis_impact_on_strategy(
            sample_crisis, sample_strategic_roadmap
        )
        
        # Generate adjustments
        adjustments = await integration_engine.generate_crisis_aware_adjustments(
            sample_crisis, sample_strategic_roadmap, impact_assessment
        )
        
        # Verify adjustments
        assert isinstance(adjustments, list)
        assert len(adjustments) > 0
        
        for adjustment in adjustments:
            assert isinstance(adjustment, CrisisAwareAdjustment)
            assert adjustment.crisis_id == sample_crisis.id
            assert adjustment.priority >= 1 and adjustment.priority <= 5
            assert adjustment.implementation_timeline > 0
            assert len(adjustment.expected_benefits) > 0
            assert len(adjustment.success_metrics) > 0
    
    @pytest.mark.asyncio
    async def test_create_recovery_integration_plan(
        self, integration_engine, sample_crisis, sample_strategic_roadmap
    ):
        """Test creation of recovery integration plan"""
        
        # Get impact assessment
        impact_assessment = await integration_engine.assess_crisis_impact_on_strategy(
            sample_crisis, sample_strategic_roadmap
        )
        
        # Create recovery plan
        recovery_plan = await integration_engine.create_recovery_integration_plan(
            sample_crisis, sample_strategic_roadmap, impact_assessment
        )
        
        # Verify recovery plan
        assert isinstance(recovery_plan, RecoveryIntegrationPlan)
        assert recovery_plan.crisis_id == sample_crisis.id
        assert recovery_plan.strategic_roadmap_id == sample_strategic_roadmap.id
        
        # Verify recovery phases
        assert len(recovery_plan.recovery_phases) > 0
        phase_names = [phase["phase"] for phase in recovery_plan.recovery_phases]
        assert "immediate_stabilization" in phase_names
        
        # Verify other components
        assert len(recovery_plan.success_criteria) > 0
        assert "monitoring_framework" in recovery_plan.monitoring_framework
        assert len(recovery_plan.stakeholder_communication_plan) > 0
    
    def test_calculate_impact_level(self, integration_engine, sample_strategic_roadmap):
        """Test impact level calculation"""
        
        # Test different crisis severities
        low_crisis = Crisis(
            id="crisis_low", crisis_type=CrisisType.SYSTEM_OUTAGE,
            severity_level=SeverityLevel.LOW, title="Minor Outage",
            description="Brief system outage", start_time=datetime.now(),
            affected_areas=["operations"], stakeholders_impacted=["users"],
            current_status="active", response_actions=[],
            estimated_resolution_time=4
        )
        
        high_crisis = Crisis(
            id="crisis_high", crisis_type=CrisisType.SECURITY_BREACH,
            severity_level=SeverityLevel.CRITICAL, title="Major Breach",
            description="Critical security incident", start_time=datetime.now(),
            affected_areas=["security", "data"], stakeholders_impacted=["all"],
            current_status="active", response_actions=[],
            estimated_resolution_time=168
        )
        
        # Calculate impact levels
        low_impact = integration_engine._calculate_impact_level(low_crisis, sample_strategic_roadmap)
        high_impact = integration_engine._calculate_impact_level(high_crisis, sample_strategic_roadmap)
        
        # Verify impact scaling
        impact_levels = list(CrisisImpactLevel)
        low_index = impact_levels.index(low_impact)
        high_index = impact_levels.index(high_impact)
        
        assert high_index >= low_index  # Higher severity should have higher impact
    
    def test_identify_affected_milestones(self, integration_engine, sample_crisis, sample_strategic_roadmap):
        """Test identification of affected milestones"""
        
        affected_milestones = integration_engine._identify_affected_milestones(
            sample_crisis, sample_strategic_roadmap.milestones
        )
        
        # Should identify milestones based on timeline and risk factors
        assert isinstance(affected_milestones, list)
        
        # Milestone within 1 year with high budget should be affected
        if sample_strategic_roadmap.milestones[0].resource_requirements.get("budget", 0) > 100e6:
            assert sample_strategic_roadmap.milestones[0].id in affected_milestones
    
    def test_identify_affected_technology_bets(self, integration_engine, sample_crisis, sample_strategic_roadmap):
        """Test identification of affected technology bets"""
        
        affected_bets = integration_engine._identify_affected_technology_bets(
            sample_crisis, sample_strategic_roadmap.technology_bets
        )
        
        # Should identify high-risk or high-investment bets
        assert isinstance(affected_bets, list)
        
        # High-risk bet should be affected
        high_risk_bet = sample_strategic_roadmap.technology_bets[0]
        if high_risk_bet.risk_level == InvestmentRisk.HIGH:
            assert high_risk_bet.id in affected_bets
    
    def test_calculate_resource_reallocation(self, integration_engine, sample_crisis):
        """Test resource reallocation calculation"""
        
        # Test different impact levels
        for impact_level in CrisisImpactLevel:
            reallocation = integration_engine._calculate_resource_reallocation(
                sample_crisis, impact_level
            )
            
            assert 0 <= reallocation <= 0.8  # Should be between 0% and 80%
            
            # Higher impact should generally require more reallocation
            if impact_level == CrisisImpactLevel.CATASTROPHIC:
                assert reallocation > 0.5  # Should be significant for catastrophic
    
    def test_calculate_timeline_adjustments(self, integration_engine, sample_crisis):
        """Test timeline adjustment calculations"""
        
        affected_milestones = ["milestone_001", "milestone_002"]
        impact_level = CrisisImpactLevel.SIGNIFICANT
        
        adjustments = integration_engine._calculate_timeline_adjustments(
            sample_crisis, affected_milestones, impact_level
        )
        
        # Verify adjustments
        assert isinstance(adjustments, dict)
        assert len(adjustments) == len(affected_milestones)
        
        for milestone_id, delay in adjustments.items():
            assert milestone_id in affected_milestones
            assert delay > 0  # Should have some delay
            assert delay <= 365  # Should be reasonable
    
    def test_assess_risk_level_changes(self, integration_engine, sample_crisis, sample_strategic_roadmap):
        """Test risk level change assessment"""
        
        risk_changes = integration_engine._assess_risk_level_changes(
            sample_crisis, sample_strategic_roadmap.risk_assessments
        )
        
        # Verify risk changes
        assert isinstance(risk_changes, dict)
        
        for risk_id, change in risk_changes.items():
            assert change >= 0  # Risk should increase or stay same during crisis
            assert change <= 0.5  # Should be reasonable increase
    
    def test_generate_strategic_recommendations(self, integration_engine, sample_crisis, sample_strategic_roadmap):
        """Test strategic recommendation generation"""
        
        impact_level = CrisisImpactLevel.SEVERE
        
        recommendations = integration_engine._generate_strategic_recommendations(
            sample_crisis, impact_level, sample_strategic_roadmap
        )
        
        # Verify recommendations
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should have crisis-specific recommendations
        recommendation_text = " ".join(recommendations).lower()
        assert any(keyword in recommendation_text for keyword in [
            "crisis", "resource", "strategic", "risk", "stakeholder"
        ])
    
    def test_estimate_recovery_timeline(self, integration_engine, sample_crisis):
        """Test recovery timeline estimation"""
        
        # Test different impact levels
        for impact_level in CrisisImpactLevel:
            recovery_time = integration_engine._estimate_recovery_timeline(
                sample_crisis, impact_level
            )
            
            assert recovery_time > 0  # Should have positive recovery time
            assert recovery_time <= 730  # Should be reasonable (max 2 years)
            
            # Higher impact should generally take longer to recover
            if impact_level == CrisisImpactLevel.CATASTROPHIC:
                assert recovery_time > 365  # Should take more than a year


class TestCrisisStrategicIntegrationEdgeCases:
    """Test edge cases and error conditions"""
    
    @pytest.fixture
    def integration_engine(self):
        return CrisisStrategicIntegration()
    
    @pytest.mark.asyncio
    async def test_empty_strategic_roadmap(self, integration_engine):
        """Test handling of empty strategic roadmap"""
        
        crisis = Crisis(
            id="crisis_001", crisis_type=CrisisType.SYSTEM_OUTAGE,
            severity_level=SeverityLevel.MEDIUM, title="Test Crisis",
            description="Test crisis", start_time=datetime.now(),
            affected_areas=["ops"], stakeholders_impacted=["users"],
            current_status="active", response_actions=[],
            estimated_resolution_time=24
        )
        
        # Empty roadmap
        empty_roadmap = StrategicRoadmap(
            id="empty_roadmap", name="Empty", description="Empty roadmap",
            vision=TechnologyVision(
                id="vision", title="Test", description="Test vision",
                time_horizon=5, key_principles=[], strategic_objectives=[],
                success_criteria=[], market_assumptions=[]
            ),
            time_horizon=5, milestones=[], technology_bets=[],
            risk_assessments=[], success_metrics=[],
            competitive_positioning=CompetitivePosition.FAST_FOLLOWER,
            market_assumptions=[], resource_allocation={},
            scenario_plans=[], review_schedule=[], stakeholders=[],
            created_at=datetime.now(), updated_at=datetime.now()
        )
        
        # Should handle empty roadmap gracefully
        impact_assessment = await integration_engine.assess_crisis_impact_on_strategy(
            crisis, empty_roadmap
        )
        
        assert isinstance(impact_assessment, CrisisStrategicImpact)
        assert len(impact_assessment.affected_milestones) == 0
        assert len(impact_assessment.affected_technology_bets) == 0
    
    @pytest.mark.asyncio
    async def test_minimal_crisis_impact(self, integration_engine):
        """Test handling of minimal crisis impact"""
        
        # Very low severity crisis
        minimal_crisis = Crisis(
            id="minimal_crisis", crisis_type=CrisisType.MINOR_INCIDENT,
            severity_level=SeverityLevel.LOW, title="Minor Issue",
            description="Very minor issue", start_time=datetime.now(),
            affected_areas=["minor"], stakeholders_impacted=["few"],
            current_status="resolving", response_actions=[],
            estimated_resolution_time=2
        )
        
        # Simple roadmap
        simple_vision = TechnologyVision(
            id="simple_vision", title="Simple Vision", description="Simple",
            time_horizon=3, key_principles=["simple"], strategic_objectives=["grow"],
            success_criteria=["success"], market_assumptions=["stable"]
        )
        
        simple_roadmap = StrategicRoadmap(
            id="simple_roadmap", name="Simple", description="Simple roadmap",
            vision=simple_vision, time_horizon=3, milestones=[], technology_bets=[],
            risk_assessments=[], success_metrics=[],
            competitive_positioning=CompetitivePosition.FAST_FOLLOWER,
            market_assumptions=[], resource_allocation={},
            scenario_plans=[], review_schedule=[], stakeholders=[],
            created_at=datetime.now(), updated_at=datetime.now()
        )
        
        impact_assessment = await integration_engine.assess_crisis_impact_on_strategy(
            minimal_crisis, simple_roadmap
        )
        
        # Should have minimal impact
        assert impact_assessment.impact_level == CrisisImpactLevel.MINIMAL
        assert impact_assessment.resource_reallocation_needed < 0.1
        assert impact_assessment.recovery_timeline < 60  # Less than 2 months


if __name__ == "__main__":
    pytest.main([__file__])