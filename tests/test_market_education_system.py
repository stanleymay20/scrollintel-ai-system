"""
Tests for Market Education System
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.market_education_engine import (
    MarketEducationEngine,
    CampaignType,
    TargetSegment,
    ContentType,
    EducationContent,
    MarketingCampaign
)
from scrollintel.core.content_delivery_system import (
    ContentDeliverySystem,
    DeliveryChannel,
    EngagementLevel,
    DeliveryTarget
)
from scrollintel.core.market_readiness_assessment import (
    MarketReadinessAssessment,
    MarketSegment,
    ReadinessLevel
)

class TestMarketEducationEngine:
    """Test cases for Market Education Engine"""
    
    @pytest.fixture
    def engine(self):
        return MarketEducationEngine()
    
    @pytest.mark.asyncio
    async def test_create_campaign_from_template(self, engine):
        """Test creating campaign from template"""
        campaign = await engine.create_campaign("year_1_awareness")
        
        assert campaign is not None
        assert campaign.name == "AI CTO Awareness Foundation"
        assert campaign.campaign_type == CampaignType.AWARENESS
        assert TargetSegment.ENTERPRISE_CTOS in campaign.target_segments
        assert len(campaign.content_pieces) > 0
        assert campaign.id in engine.campaigns
    
    @pytest.mark.asyncio
    async def test_create_campaign_with_customizations(self, engine):
        """Test creating campaign with customizations"""
        customizations = {
            "name": "Custom Campaign",
            "budget": 2000000
        }
        
        campaign = await engine.create_campaign("year_2_education", customizations)
        
        assert campaign.name == "Custom Campaign"
        assert campaign.budget == 2000000
        assert campaign.campaign_type == CampaignType.EDUCATION
    
    @pytest.mark.asyncio
    async def test_assess_market_readiness(self, engine):
        """Test market readiness assessment"""
        metrics = await engine.assess_market_readiness(TargetSegment.ENTERPRISE_CTOS)
        
        assert metrics is not None
        assert metrics.segment == TargetSegment.ENTERPRISE_CTOS
        assert 0 <= metrics.awareness_level <= 100
        assert 0 <= metrics.understanding_level <= 100
        assert 0 <= metrics.acceptance_level <= 100
        assert 0 <= metrics.adoption_readiness <= 100
        assert len(metrics.resistance_factors) > 0
    
    @pytest.mark.asyncio
    async def test_track_content_engagement(self, engine):
        """Test content engagement tracking"""
        # Create campaign first
        campaign = await engine.create_campaign("year_1_awareness")
        content_id = campaign.content_pieces[0].id
        
        metrics = {
            "views": 1000,
            "shares": 50,
            "leads_generated": 25
        }
        
        await engine.track_content_engagement(content_id, metrics)
        
        content = engine.content_library[content_id]
        assert content.engagement_metrics == metrics
        assert content.effectiveness_score > 0
    
    @pytest.mark.asyncio
    async def test_adapt_campaign_strategy(self, engine):
        """Test campaign strategy adaptation"""
        campaign = await engine.create_campaign("year_1_awareness")
        initial_content_count = len(campaign.content_pieces)
        
        feedback = {
            "resistance_high": True,
            "engagement_low": False,
            "adoption_accelerating": False
        }
        
        await engine.adapt_campaign_strategy(campaign.id, feedback)
        
        # Should have added educational content
        assert len(campaign.content_pieces) > initial_content_count
    
    @pytest.mark.asyncio
    async def test_generate_readiness_report(self, engine):
        """Test readiness report generation"""
        report = await engine.generate_readiness_report()
        
        assert "overall_readiness" in report
        assert "segment_readiness" in report
        assert "campaign_effectiveness" in report
        assert "recommendations" in report
        assert "generated_at" in report
        assert 0 <= report["overall_readiness"] <= 100
    
    @pytest.mark.asyncio
    async def test_execute_five_year_plan(self, engine):
        """Test 5-year plan execution"""
        execution_plan = await engine.execute_five_year_plan()
        
        assert execution_plan["plan_status"] == "initiated"
        assert len(execution_plan["campaigns_created"]) == 5
        assert execution_plan["target_readiness"] == 95.0
        assert "execution_start" in execution_plan
        assert "estimated_completion" in execution_plan
        
        # Verify campaigns were created
        assert len(engine.campaigns) == 5

class TestContentDeliverySystem:
    """Test cases for Content Delivery System"""
    
    @pytest.fixture
    def delivery_system(self):
        return ContentDeliverySystem()
    
    @pytest.mark.asyncio
    async def test_register_target(self, delivery_system):
        """Test registering delivery target"""
        target_data = {
            "name": "John Doe",
            "segment": "enterprise_ctos",
            "contact_info": {"email": "john@example.com"},
            "preferred_channels": ["email", "webinar"]
        }
        
        target = await delivery_system.register_target(target_data)
        
        assert target.name == "John Doe"
        assert target.segment == "enterprise_ctos"
        assert len(target.preferred_channels) == 2
        assert target.conversion_probability > 0
        assert target.id in delivery_system.delivery_targets
    
    @pytest.mark.asyncio
    async def test_schedule_content_delivery(self, delivery_system):
        """Test scheduling content delivery"""
        # Register target first
        target_data = {
            "name": "Jane Smith",
            "segment": "tech_leaders",
            "contact_info": {"email": "jane@example.com"},
            "preferred_channels": ["email"]
        }
        target = await delivery_system.register_target(target_data)
        
        deliveries = await delivery_system.schedule_content_delivery(
            "content_123",
            [target.id],
            DeliveryChannel.EMAIL
        )
        
        assert len(deliveries) == 1
        assert deliveries[0].content_id == "content_123"
        assert deliveries[0].target_id == target.id
        assert deliveries[0].channel == DeliveryChannel.EMAIL
        assert deliveries[0].status == "scheduled"
    
    @pytest.mark.asyncio
    async def test_execute_delivery(self, delivery_system):
        """Test executing delivery"""
        # Setup target and delivery
        target_data = {
            "name": "Bob Johnson",
            "segment": "investors",
            "contact_info": {"email": "bob@example.com"},
            "preferred_channels": ["email"]
        }
        target = await delivery_system.register_target(target_data)
        
        deliveries = await delivery_system.schedule_content_delivery(
            "content_456",
            [target.id],
            DeliveryChannel.EMAIL
        )
        
        delivery_id = deliveries[0].id
        success = await delivery_system.execute_delivery(delivery_id)
        
        # Success depends on random simulation, but should be boolean
        assert isinstance(success, bool)
        
        delivery = delivery_system.scheduled_deliveries[delivery_id]
        if success:
            assert delivery.status == "delivered"
            assert delivery.delivered_time is not None
        else:
            assert delivery.status == "failed"
            assert delivery.follow_up_required is True
    
    @pytest.mark.asyncio
    async def test_track_engagement(self, delivery_system):
        """Test engagement tracking"""
        # Setup target and delivery
        target_data = {
            "name": "Alice Brown",
            "segment": "board_members",
            "contact_info": {"email": "alice@example.com"},
            "preferred_channels": ["email"]
        }
        target = await delivery_system.register_target(target_data)
        
        deliveries = await delivery_system.schedule_content_delivery(
            "content_789",
            [target.id],
            DeliveryChannel.EMAIL
        )
        
        delivery_id = deliveries[0].id
        
        engagement_data = {
            "opened": True,
            "clicked": True,
            "shared": False,
            "responded": False,
            "converted": False
        }
        
        await delivery_system.track_engagement(delivery_id, engagement_data)
        
        delivery = delivery_system.scheduled_deliveries[delivery_id]
        assert delivery.engagement_metrics == engagement_data
        
        # Target engagement level should be updated
        updated_target = delivery_system.delivery_targets[target.id]
        assert updated_target.current_engagement_level != EngagementLevel.LOW
    
    @pytest.mark.asyncio
    async def test_optimize_delivery_strategy(self, delivery_system):
        """Test delivery strategy optimization"""
        # Setup some delivery data
        target_data = {
            "name": "Charlie Wilson",
            "segment": "developers",
            "contact_info": {"email": "charlie@example.com"},
            "preferred_channels": ["email"]
        }
        target = await delivery_system.register_target(target_data)
        
        deliveries = await delivery_system.schedule_content_delivery(
            "content_opt",
            [target.id],
            DeliveryChannel.EMAIL
        )
        
        # Track some engagement
        await delivery_system.track_engagement(deliveries[0].id, {
            "opened": True,
            "clicked": True,
            "converted": False
        })
        
        optimization = await delivery_system.optimize_delivery_strategy("campaign_123")
        
        assert "best_channels" in optimization
        assert "segment_performance" in optimization
        assert "recommendations" in optimization
        assert "generated_at" in optimization
    
    @pytest.mark.asyncio
    async def test_generate_delivery_report(self, delivery_system):
        """Test delivery report generation"""
        report = await delivery_system.generate_delivery_report()
        
        assert "summary" in report
        assert "channel_performance" in report
        assert "content_performance" in report
        assert "target_engagement" in report
        assert "generated_at" in report
        
        summary = report["summary"]
        assert "total_deliveries" in summary
        assert "successful_deliveries" in summary
        assert "delivery_rate" in summary

class TestMarketReadinessAssessment:
    """Test cases for Market Readiness Assessment"""
    
    @pytest.fixture
    def assessment(self):
        return MarketReadinessAssessment()
    
    @pytest.mark.asyncio
    async def test_assess_segment_readiness(self, assessment):
        """Test segment readiness assessment"""
        segment_assessment = await assessment.assess_segment_readiness(
            MarketSegment.ENTERPRISE_CTOS
        )
        
        assert segment_assessment.segment == MarketSegment.ENTERPRISE_CTOS
        assert isinstance(segment_assessment.readiness_level, ReadinessLevel)
        assert 0 <= segment_assessment.readiness_score <= 100
        assert 0 <= segment_assessment.confidence_level <= 1
        assert len(segment_assessment.indicators) > 0
        assert len(segment_assessment.barriers) > 0
        assert len(segment_assessment.accelerators) > 0
        assert len(segment_assessment.recommended_actions) > 0
    
    @pytest.mark.asyncio
    async def test_monitor_market_conditions(self, assessment):
        """Test market conditions monitoring"""
        conditions = await assessment.monitor_market_conditions()
        
        assert len(conditions) > 0
        
        for condition in conditions:
            assert condition.factor is not None
            assert condition.impact_level in ["high", "medium", "low"]
            assert condition.trend in ["positive", "negative", "neutral"]
            assert condition.description is not None
            assert condition.recommended_response is not None
    
    @pytest.mark.asyncio
    async def test_adapt_strategy_based_on_conditions(self, assessment):
        """Test strategy adaptation based on conditions"""
        # Monitor conditions first
        await assessment.monitor_market_conditions()
        
        adaptations = await assessment.adapt_strategy_based_on_conditions()
        
        assert len(adaptations) > 0
        
        for adaptation in adaptations:
            assert adaptation.id is not None
            assert len(adaptation.trigger_conditions) > 0
            assert len(adaptation.actions) > 0
            assert adaptation.expected_impact >= 0
            assert adaptation.implementation_timeline > 0
            assert len(adaptation.success_metrics) > 0
    
    @pytest.mark.asyncio
    async def test_generate_comprehensive_report(self, assessment):
        """Test comprehensive report generation"""
        report = await assessment.generate_comprehensive_report()
        
        assert "executive_summary" in report
        assert "segment_assessments" in report
        assert "market_conditions" in report
        assert "adaptation_strategies" in report
        assert "recommendations" in report
        assert "generated_at" in report
        
        executive_summary = report["executive_summary"]
        assert "overall_readiness_score" in executive_summary
        assert "market_readiness_level" in executive_summary
        assert "total_segments_assessed" in executive_summary
        assert "key_opportunities" in executive_summary
        assert "critical_barriers" in executive_summary
        
        # Verify all segments were assessed
        assert len(report["segment_assessments"]) == len(MarketSegment)

class TestIntegration:
    """Integration tests for market education system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_campaign_execution(self):
        """Test complete campaign execution flow"""
        # Initialize systems
        education_engine = MarketEducationEngine()
        delivery_system = ContentDeliverySystem()
        readiness_assessment = MarketReadinessAssessment()
        
        # 1. Create campaign
        campaign = await education_engine.create_campaign("year_1_awareness")
        assert campaign is not None
        
        # 2. Register delivery targets
        target_data = {
            "name": "Test Executive",
            "segment": "enterprise_ctos",
            "contact_info": {"email": "exec@example.com"},
            "preferred_channels": ["email", "webinar"]
        }
        target = await delivery_system.register_target(target_data)
        
        # 3. Schedule content delivery
        content_id = campaign.content_pieces[0].id
        deliveries = await delivery_system.schedule_content_delivery(
            content_id,
            [target.id],
            DeliveryChannel.EMAIL
        )
        
        # 4. Execute delivery
        delivery_id = deliveries[0].id
        success = await delivery_system.execute_delivery(delivery_id)
        
        # 5. Track engagement
        if success:
            engagement_data = {
                "opened": True,
                "clicked": True,
                "shared": False,
                "converted": False
            }
            await delivery_system.track_engagement(delivery_id, engagement_data)
        
        # 6. Assess market readiness
        assessment = await readiness_assessment.assess_segment_readiness(
            MarketSegment.ENTERPRISE_CTOS
        )
        
        # 7. Generate reports
        education_report = await education_engine.generate_readiness_report()
        delivery_report = await delivery_system.generate_delivery_report()
        comprehensive_report = await readiness_assessment.generate_comprehensive_report()
        
        # Verify integration
        assert len(education_engine.campaigns) > 0
        assert len(delivery_system.delivery_targets) > 0
        assert len(delivery_system.scheduled_deliveries) > 0
        assert assessment.readiness_score > 0
        assert education_report["overall_readiness"] > 0
        assert delivery_report["summary"]["total_deliveries"] > 0
        assert comprehensive_report["executive_summary"]["overall_readiness_score"] > 0
    
    @pytest.mark.asyncio
    async def test_five_year_plan_integration(self):
        """Test 5-year plan integration with all systems"""
        education_engine = MarketEducationEngine()
        delivery_system = ContentDeliverySystem()
        readiness_assessment = MarketReadinessAssessment()
        
        # Execute 5-year plan
        execution_plan = await education_engine.execute_five_year_plan()
        
        # Verify plan creation
        assert execution_plan["plan_status"] == "initiated"
        assert len(execution_plan["campaigns_created"]) == 5
        
        # Register targets for all segments
        for segment in ["enterprise_ctos", "tech_leaders", "board_members", "investors"]:
            target_data = {
                "name": f"Test {segment.title()}",
                "segment": segment,
                "contact_info": {"email": f"{segment}@example.com"},
                "preferred_channels": ["email"]
            }
            await delivery_system.register_target(target_data)
        
        # Assess readiness for all segments
        assessments = {}
        for segment in MarketSegment:
            assessment = await readiness_assessment.assess_segment_readiness(segment)
            assessments[segment] = assessment
        
        # Generate comprehensive report
        comprehensive_report = await readiness_assessment.generate_comprehensive_report()
        
        # Verify integration
        assert len(education_engine.campaigns) == 5
        assert len(delivery_system.delivery_targets) == 4
        assert len(assessments) == len(MarketSegment)
        assert comprehensive_report["executive_summary"]["total_segments_assessed"] == len(MarketSegment)
        
        # Verify campaign types cover all phases
        campaign_types = [c.campaign_type for c in education_engine.campaigns.values()]
        expected_types = [
            CampaignType.AWARENESS,
            CampaignType.EDUCATION,
            CampaignType.DEMONSTRATION,
            CampaignType.VALIDATION,
            CampaignType.ADOPTION
        ]
        
        for expected_type in expected_types:
            assert expected_type in campaign_types

if __name__ == "__main__":
    pytest.main([__file__])