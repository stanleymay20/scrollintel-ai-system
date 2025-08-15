"""
Test Suite for ScrollIntel Competitive Intelligence System
Comprehensive testing of competitive analysis and market positioning capabilities
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime

from scrollintel.core.competitive_intelligence_system import (
    ScrollIntelCompetitiveIntelligence,
    CompetitorTier,
    ThreatLevel,
    MarketPosition,
    CompetitorProfile,
    MarketIntelligence,
    PositioningStrategy
)

class TestScrollIntelCompetitiveIntelligence:
    """Test ScrollIntel competitive intelligence system"""
    
    @pytest.fixture
    def intelligence_system(self):
        """Create competitive intelligence system instance"""
        return ScrollIntelCompetitiveIntelligence()
    
    def test_initialization(self, intelligence_system):
        """Test system initialization"""
        assert len(intelligence_system.competitors) > 0
        assert intelligence_system.market_intelligence is not None
        assert intelligence_system.positioning_strategy is not None
        assert isinstance(intelligence_system.threat_alerts, list)
        assert isinstance(intelligence_system.market_opportunities, list)
    
    def test_competitor_profiles_loaded(self, intelligence_system):
        """Test that competitor profiles are properly loaded"""
        # Check that key competitors are loaded
        expected_competitors = ["openai_gpt", "anthropic_claude", "palantir", "mckinsey"]
        
        for competitor_id in expected_competitors:
            assert competitor_id in intelligence_system.competitors
            competitor = intelligence_system.competitors[competitor_id]
            
            # Validate competitor profile structure
            assert competitor.competitor_id == competitor_id
            assert competitor.company_name
            assert isinstance(competitor.tier, CompetitorTier)
            assert isinstance(competitor.threat_level, ThreatLevel)
            assert isinstance(competitor.market_position, MarketPosition)
            assert competitor.annual_revenue > 0
            assert competitor.employee_count > 0
            assert len(competitor.key_products) > 0
            assert len(competitor.competitive_advantages) > 0
            assert len(competitor.weaknesses) > 0
    
    def test_market_intelligence_structure(self, intelligence_system):
        """Test market intelligence data structure"""
        market_intel = intelligence_system.market_intelligence
        
        assert market_intel.market_size > 0
        assert 0 < market_intel.growth_rate < 1
        assert len(market_intel.key_trends) > 0
        assert len(market_intel.adoption_barriers) > 0
        assert len(market_intel.buyer_personas) > 0
        assert len(market_intel.decision_criteria) > 0
        assert len(market_intel.budget_allocation_patterns) > 0
        assert market_intel.technology_readiness
        assert len(market_intel.regulatory_environment) > 0
        assert market_intel.competitive_landscape_summary
    
    def test_positioning_strategy_structure(self, intelligence_system):
        """Test positioning strategy data structure"""
        positioning = intelligence_system.positioning_strategy
        
        assert positioning.unique_value_proposition
        assert len(positioning.key_differentiators) > 0
        assert len(positioning.target_segments) > 0
        assert positioning.messaging_framework
        assert len(positioning.competitive_advantages) > 0
        assert len(positioning.proof_points) > 0
        assert len(positioning.objection_handling) > 0
        assert positioning.pricing_strategy
        assert positioning.go_to_market_approach
    
    @pytest.mark.asyncio
    async def test_analyze_competitive_threats(self, intelligence_system):
        """Test competitive threat analysis"""
        threat_analysis = await intelligence_system.analyze_competitive_threats()
        
        # Validate threat analysis structure
        assert "immediate_threats" in threat_analysis
        assert "emerging_threats" in threat_analysis
        assert "market_opportunities" in threat_analysis
        assert "strategic_recommendations" in threat_analysis
        
        # Check immediate threats
        for threat in threat_analysis["immediate_threats"]:
            assert "competitor" in threat
            assert "threat_level" in threat
            assert "threat_score" in threat
            assert "key_concerns" in threat
            assert "mitigation_strategies" in threat
            assert "market_impact" in threat
            assert 0 <= threat["threat_score"] <= 100
        
        # Check market opportunities
        for opportunity in threat_analysis["market_opportunities"]:
            assert "opportunity" in opportunity
            assert "market_size" in opportunity
            assert "competitive_gap" in opportunity
            assert "time_to_market" in opportunity
            assert "success_probability" in opportunity
        
        # Check strategic recommendations
        for recommendation in threat_analysis["strategic_recommendations"]:
            assert "priority" in recommendation
            assert "recommendation" in recommendation
            assert "rationale" in recommendation
            assert "timeline" in recommendation
            assert "investment" in recommendation
            assert "expected_impact" in recommendation
    
    def test_calculate_threat_score(self, intelligence_system):
        """Test threat score calculation"""
        # Create test competitor
        test_competitor = CompetitorProfile(
            competitor_id="test_competitor",
            company_name="Test Company",
            tier=CompetitorTier.TIER_1_DIRECT,
            market_position=MarketPosition.CHALLENGER,
            threat_level=ThreatLevel.HIGH,
            annual_revenue=1000000000,  # $1B
            employee_count=1000,
            funding_raised=500000000,  # $500M
            key_products=["Product A", "Product B"],
            target_market=["Enterprise"],
            pricing_model="Subscription",
            key_differentiators=["Feature 1", "Feature 2"],
            weaknesses=["Weakness 1"],
            recent_developments=[],
            market_share=0.15,  # 15%
            customer_count=1000,
            geographic_presence=["North America"],
            technology_stack=["Tech 1"],
            leadership_team=[],
            financial_health="Strong",
            growth_trajectory="300% YoY growth",
            competitive_advantages=["Advantage 1", "Advantage 2", "Advantage 3"],
            strategic_partnerships=["Partner 1"]
        )
        
        threat_score = intelligence_system._calculate_threat_score(test_competitor)
        
        # Threat score should be between 0 and 100
        assert 0 <= threat_score <= 100
        
        # High revenue, market share, and growth should result in higher score
        assert threat_score > 50  # Should be relatively high threat
    
    def test_generate_mitigation_strategies(self, intelligence_system):
        """Test mitigation strategy generation"""
        # Create test competitor with specific advantages and weaknesses
        test_competitor = CompetitorProfile(
            competitor_id="test_competitor",
            company_name="Test Company",
            tier=CompetitorTier.TIER_1_DIRECT,
            market_position=MarketPosition.LEADER,
            threat_level=ThreatLevel.CRITICAL,
            annual_revenue=2000000000,
            employee_count=2000,
            funding_raised=1000000000,
            key_products=["AI Platform"],
            target_market=["Enterprise AI"],
            pricing_model="Usage-based",
            key_differentiators=["Strong brand recognition"],
            weaknesses=["No CTO-specific functionality", "Limited strategic planning"],
            recent_developments=[],
            market_share=0.35,
            customer_count=10000,
            geographic_presence=["Global"],
            technology_stack=["AI/ML"],
            leadership_team=[],
            financial_health="Excellent",
            growth_trajectory="High growth",
            competitive_advantages=["Strong brand recognition", "Enterprise adoption"],
            strategic_partnerships=["Major cloud providers"]
        )
        
        strategies = intelligence_system._generate_mitigation_strategies(test_competitor)
        
        assert len(strategies) > 0
        assert isinstance(strategies, list)
        
        # Should generate strategies based on competitor advantages and weaknesses
        strategy_text = " ".join(strategies).lower()
        assert any(keyword in strategy_text for keyword in ["brand", "enterprise", "cto", "strategic"])
    
    @pytest.mark.asyncio
    async def test_generate_market_positioning_report(self, intelligence_system):
        """Test market positioning report generation"""
        report = await intelligence_system.generate_market_positioning_report()
        
        # Validate report structure
        required_sections = [
            "executive_summary",
            "market_analysis",
            "competitive_landscape",
            "positioning_strategy",
            "strategic_recommendations",
            "market_opportunities",
            "success_metrics",
            "risk_assessment"
        ]
        
        for section in required_sections:
            assert section in report
        
        # Validate executive summary
        exec_summary = report["executive_summary"]
        assert "market_opportunity" in exec_summary
        assert "competitive_position" in exec_summary
        assert "key_differentiators" in exec_summary
        assert "threat_level" in exec_summary
        assert "recommended_action" in exec_summary
        
        # Validate market analysis
        market_analysis = report["market_analysis"]
        assert "market_size" in market_analysis
        assert "growth_rate" in market_analysis
        assert "key_trends" in market_analysis
        assert "buyer_personas" in market_analysis
        assert "adoption_barriers" in market_analysis
        
        # Validate competitive landscape
        comp_landscape = report["competitive_landscape"]
        assert "total_competitors" in comp_landscape
        assert "tier_1_direct" in comp_landscape
        assert "immediate_threats" in comp_landscape
        assert "market_share_distribution" in comp_landscape
    
    def test_get_competitive_dashboard(self, intelligence_system):
        """Test competitive dashboard generation"""
        dashboard = intelligence_system.get_competitive_dashboard()
        
        # Validate dashboard structure
        required_sections = [
            "competitive_overview",
            "market_intelligence",
            "positioning_strength",
            "threat_monitoring",
            "strategic_priorities",
            "success_indicators"
        ]
        
        for section in required_sections:
            assert section in dashboard
        
        # Validate competitive overview
        comp_overview = dashboard["competitive_overview"]
        assert "total_competitors_tracked" in comp_overview
        assert "critical_threats" in comp_overview
        assert "high_threats" in comp_overview
        assert "market_opportunity" in comp_overview
        assert "competitive_intensity" in comp_overview
        
        # Validate market intelligence
        market_intel = dashboard["market_intelligence"]
        assert "market_size" in market_intel
        assert "growth_rate" in market_intel
        assert "key_trends_count" in market_intel
        assert "buyer_personas_count" in market_intel
        
        # Validate positioning strength
        positioning = dashboard["positioning_strength"]
        assert "unique_value_proposition" in positioning
        assert "key_differentiators_count" in positioning
        assert "competitive_advantages_count" in positioning
        assert "positioning_confidence" in positioning
        
        # Validate strategic priorities
        assert isinstance(dashboard["strategic_priorities"], list)
        assert len(dashboard["strategic_priorities"]) > 0
    
    def test_competitor_tier_classification(self, intelligence_system):
        """Test competitor tier classification"""
        tier_counts = {
            CompetitorTier.TIER_1_DIRECT: 0,
            CompetitorTier.TIER_2_ADJACENT: 0,
            CompetitorTier.TIER_3_TRADITIONAL: 0,
            CompetitorTier.EMERGING_THREAT: 0
        }
        
        for competitor in intelligence_system.competitors.values():
            tier_counts[competitor.tier] += 1
        
        # Should have competitors in each tier
        assert tier_counts[CompetitorTier.TIER_1_DIRECT] > 0
        assert tier_counts[CompetitorTier.TIER_2_ADJACENT] > 0
        assert tier_counts[CompetitorTier.TIER_3_TRADITIONAL] > 0
        assert tier_counts[CompetitorTier.EMERGING_THREAT] > 0
    
    def test_threat_level_distribution(self, intelligence_system):
        """Test threat level distribution"""
        threat_counts = {
            ThreatLevel.CRITICAL: 0,
            ThreatLevel.HIGH: 0,
            ThreatLevel.MEDIUM: 0,
            ThreatLevel.LOW: 0,
            ThreatLevel.NEGLIGIBLE: 0
        }
        
        for competitor in intelligence_system.competitors.values():
            threat_counts[competitor.threat_level] += 1
        
        # Should have varied threat levels
        total_threats = sum(threat_counts.values())
        assert total_threats > 0
        
        # Should have some high-priority threats
        high_priority_threats = threat_counts[ThreatLevel.CRITICAL] + threat_counts[ThreatLevel.HIGH]
        assert high_priority_threats > 0
    
    def test_market_share_calculation(self, intelligence_system):
        """Test market share calculations"""
        total_market_share = sum(
            competitor.market_share 
            for competitor in intelligence_system.competitors.values()
        )
        
        # Total competitor market share should be less than 100%
        assert total_market_share < 1.0
        
        # Should leave significant opportunity for ScrollIntel
        scrollintel_opportunity = 1.0 - total_market_share
        assert scrollintel_opportunity > 0.3  # At least 30% opportunity
    
    def test_competitive_advantages_analysis(self, intelligence_system):
        """Test competitive advantages analysis"""
        for competitor in intelligence_system.competitors.values():
            # Each competitor should have advantages and weaknesses
            assert len(competitor.competitive_advantages) > 0
            assert len(competitor.weaknesses) > 0
            
            # Advantages and weaknesses should be different
            advantages_set = set(competitor.competitive_advantages)
            weaknesses_set = set(competitor.weaknesses)
            assert advantages_set.isdisjoint(weaknesses_set)
    
    def test_recent_developments_tracking(self, intelligence_system):
        """Test recent developments tracking"""
        for competitor in intelligence_system.competitors.values():
            if competitor.recent_developments:
                for development in competitor.recent_developments:
                    assert "date" in development
                    assert "event" in development
                    assert development["date"]  # Should have date
                    assert development["event"]  # Should have event description
    
    def test_financial_health_assessment(self, intelligence_system):
        """Test financial health assessment"""
        valid_health_indicators = [
            "excellent", "strong", "good", "moderate", "weak", "poor",
            "profitable", "funded", "stable", "growing", "declining"
        ]
        
        for competitor in intelligence_system.competitors.values():
            health_lower = competitor.financial_health.lower()
            assert any(indicator in health_lower for indicator in valid_health_indicators)
    
    def test_growth_trajectory_analysis(self, intelligence_system):
        """Test growth trajectory analysis"""
        growth_indicators = ["growth", "yoy", "%", "rapid", "explosive", "steady", "moderate", "slow"]
        
        for competitor in intelligence_system.competitors.values():
            trajectory_lower = competitor.growth_trajectory.lower()
            assert any(indicator in trajectory_lower for indicator in growth_indicators)
    
    @pytest.mark.asyncio
    async def test_system_performance(self, intelligence_system):
        """Test system performance under load"""
        # Test multiple concurrent operations
        tasks = []
        
        for _ in range(10):
            tasks.append(intelligence_system.analyze_competitive_threats())
            tasks.append(intelligence_system.generate_market_positioning_report())
        
        # All tasks should complete successfully
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            assert not isinstance(result, Exception)
    
    def test_data_consistency(self, intelligence_system):
        """Test data consistency across system components"""
        # Market intelligence should be consistent with competitor data
        total_competitor_customers = sum(
            competitor.customer_count 
            for competitor in intelligence_system.competitors.values()
        )
        
        # Should have reasonable customer distribution
        assert total_competitor_customers > 0
        
        # Market size should be reasonable compared to competitor revenues
        total_competitor_revenue = sum(
            competitor.annual_revenue 
            for competitor in intelligence_system.competitors.values()
        )
        
        market_size = intelligence_system.market_intelligence.market_size
        assert market_size > total_competitor_revenue  # Market should be larger than current players

@pytest.mark.integration
class TestCompetitiveIntelligenceIntegration:
    """Integration tests for competitive intelligence system"""
    
    @pytest.mark.asyncio
    async def test_full_analysis_workflow(self):
        """Test complete competitive analysis workflow"""
        intelligence_system = ScrollIntelCompetitiveIntelligence()
        
        # Step 1: Analyze threats
        threat_analysis = await intelligence_system.analyze_competitive_threats()
        assert len(threat_analysis["immediate_threats"]) > 0
        
        # Step 2: Generate positioning report
        positioning_report = await intelligence_system.generate_market_positioning_report()
        assert positioning_report["executive_summary"]["market_opportunity"]
        
        # Step 3: Get dashboard
        dashboard = intelligence_system.get_competitive_dashboard()
        assert dashboard["competitive_overview"]["total_competitors_tracked"] > 0
        
        # All components should be consistent
        assert len(intelligence_system.competitors) == dashboard["competitive_overview"]["total_competitors_tracked"]
    
    def test_real_world_competitor_data(self):
        """Test with real-world competitor data"""
        intelligence_system = ScrollIntelCompetitiveIntelligence()
        
        # Check OpenAI data
        openai = intelligence_system.competitors.get("openai_gpt")
        if openai:
            assert openai.company_name == "OpenAI (GPT Enterprise)"
            assert openai.threat_level == ThreatLevel.CRITICAL
            assert openai.annual_revenue > 1000000000  # Should be > $1B
        
        # Check Anthropic data
        anthropic = intelligence_system.competitors.get("anthropic_claude")
        if anthropic:
            assert anthropic.company_name == "Anthropic (Claude for Enterprise)"
            assert anthropic.threat_level == ThreatLevel.HIGH
            assert len(anthropic.weaknesses) > 0
    
    @pytest.mark.asyncio
    async def test_market_opportunity_identification(self):
        """Test market opportunity identification"""
        intelligence_system = ScrollIntelCompetitiveIntelligence()
        
        threat_analysis = await intelligence_system.analyze_competitive_threats()
        opportunities = threat_analysis["market_opportunities"]
        
        # Should identify AI CTO category creation opportunity
        ai_cto_opportunity = None
        for opp in opportunities:
            if "AI CTO" in opp["opportunity"]:
                ai_cto_opportunity = opp
                break
        
        assert ai_cto_opportunity is not None
        assert "50B" in ai_cto_opportunity["market_size"] or "25B" in ai_cto_opportunity["market_size"]
        assert "95%" in ai_cto_opportunity["success_probability"] or "90%" in ai_cto_opportunity["success_probability"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])