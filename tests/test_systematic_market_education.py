"""
Tests for Systematic Market Education System
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.core.campaign_management_platform import (
    CampaignManagementPlatform,
    CampaignPhase,
    CampaignStatus,
    CampaignPriority
)
from scrollintel.core.education_content_tracker import (
    EducationContentTracker,
    ContentFormat,
    LearningObjective,
    PersonalizationLevel
)
from scrollintel.core.market_readiness_assessment import (
    MarketReadinessAssessment,
    MarketSegment,
    ReadinessLevel
)

class TestCampaignManagementPlatform:
    """Test the campaign management platform"""
    
    @pytest.fixture
    def platform(self):
        return CampaignManagementPlatform()
    
    @pytest.mark.asyncio
    async def test_create_five_year_master_plan(self, platform):
        """Test creating the 5-year master plan"""
        master_plan = await platform.create_five_year_master_plan()
        
        assert master_plan is not None
        assert master_plan["name"] == "ScrollIntel AI CTO Market Conditioning Master Plan"
        assert len(master_plan["yearly_campaigns"]) == 5
        assert master_plan["total_budget"] > 0
        assert "success_metrics" in master_plan
        assert "risk_mitigation" in master_plan
        
        # Verify yearly campaigns
        for i, campaign in enumerate(master_plan["yearly_campaigns"], 1):
            assert campaign["year"] == i
            assert campaign["budget"] > 0
            assert len(campaign["key_objectives"]) > 0
    
    @pytest.mark.asyncio
    async def test_execute_campaign(self, platform):
        """Test executing a campaign"""
        # First create master plan
        master_plan = await platform.create_five_year_master_plan()
        campaign_id = master_plan["yearly_campaigns"][0]["campaign_id"]
        
        # Execute the campaign
        execution_result = await platform.execute_campaign(campaign_id)
        
        assert execution_result["campaign_id"] == campaign_id
        assert execution_result["status"] == "executing"
        assert execution_result["allocated_budget"] > 0
        assert execution_result["team_assigned"] > 0
        assert execution_result["channels_activated"] > 0
        
        # Verify campaign status updated
        campaign = platform.campaigns[campaign_id]
        assert campaign.status == CampaignStatus.ACTIVE
        assert campaign.phase == CampaignPhase.EXECUTION
    
    @pytest.mark.asyncio
    async def test_monitor_campaign_performance(self, platform):
        """Test monitoring campaign performance"""
        # Create and execute campaign
        master_plan = await platform.create_five_year_master_plan()
        campaign_id = master_plan["yearly_campaigns"][0]["campaign_id"]
        await platform.execute_campaign(campaign_id)
        
        # Monitor performance
        performance_report = await platform.monitor_campaign_performance(campaign_id)
        
        assert performance_report["campaign_id"] == campaign_id
        assert "performance_data" in performance_report
        assert "benchmark_analysis" in performance_report
        assert "recommendations" in performance_report
        assert "overall_health_score" in performance_report
        
        # Verify performance data structure
        perf_data = performance_report["performance_data"]
        assert "awareness_metrics" in perf_data
        assert "engagement_metrics" in perf_data
        assert "conversion_metrics" in perf_data
        assert "roi_metrics" in perf_data
    
    @pytest.mark.asyncio
    async def test_optimize_campaign(self, platform):
        """Test campaign optimization"""
        # Create and execute campaign
        master_plan = await platform.create_five_year_master_plan()
        campaign_id = master_plan["yearly_campaigns"][0]["campaign_id"]
        await platform.execute_campaign(campaign_id)
        
        # Optimize campaign
        optimization_data = {
            "reallocate_budget": True,
            "optimize_channels": True,
            "optimize_content": True,
            "high_performing_channels": ["digital_marketing", "webinars"],
            "low_performing_channels": ["print_advertising"]
        }
        
        optimization_result = await platform.optimize_campaign(campaign_id, optimization_data)
        
        assert optimization_result["campaign_id"] == campaign_id
        assert len(optimization_result["optimizations_applied"]) > 0
        assert "expected_impact" in optimization_result
        
        # Verify campaign phase updated
        campaign = platform.campaigns[campaign_id]
        assert campaign.phase == CampaignPhase.OPTIMIZATION
    
    @pytest.mark.asyncio
    async def test_generate_master_plan_report(self, platform):
        """Test generating master plan report"""
        # Create master plan with some campaigns
        await platform.create_five_year_master_plan()
        
        report = await platform.generate_master_plan_report()
        
        assert "master_plan_id" in report
        assert "overall_progress" in report
        assert "budget_summary" in report
        assert "campaign_summaries" in report
        assert "key_achievements" in report
        assert "upcoming_milestones" in report
        assert "risks_and_mitigation" in report
        assert "recommendations" in report
        
        # Verify budget summary
        budget_summary = report["budget_summary"]
        assert budget_summary["total_budget"] > 0
        assert budget_summary["budget_utilization"] >= 0
    
    @pytest.mark.asyncio
    async def test_adapt_master_plan(self, platform):
        """Test adapting master plan"""
        # Create master plan
        await platform.create_five_year_master_plan()
        
        adaptation_data = {
            "accelerate_timeline": True,
            "reallocate_budget": True,
            "adjust_strategy": True,
            "reason": "Market opportunity acceleration",
            "acceleration_factor": 1.2,
            "reallocation_amount": 1000000
        }
        
        adaptation_result = await platform.adapt_master_plan(adaptation_data)
        
        assert "master_plan_id" in adaptation_result
        assert len(adaptation_result["adaptations_made"]) > 0
        assert "expected_impact" in adaptation_result
        
        # Verify master plan updated
        assert "last_adapted" in platform.five_year_master_plan
        assert "adaptations_history" in platform.five_year_master_plan

class TestEducationContentTracker:
    """Test the education content tracking system"""
    
    @pytest.fixture
    def tracker(self):
        return EducationContentTracker()
    
    @pytest.mark.asyncio
    async def test_track_content_engagement(self, tracker):
        """Test tracking content engagement"""
        engagement_data = {
            "start_time": datetime.now().isoformat(),
            "end_time": (datetime.now() + timedelta(minutes=15)).isoformat(),
            "duration_seconds": 900,
            "completion_percentage": 85.0,
            "interactions": [
                {"type": "click", "timestamp": datetime.now().isoformat()},
                {"type": "scroll", "timestamp": datetime.now().isoformat()}
            ],
            "feedback_rating": 4.5
        }
        
        engagement = await tracker.track_content_engagement(
            "test_content", "test_user", engagement_data
        )
        
        assert engagement.content_id == "test_content"
        assert engagement.user_id == "test_user"
        assert engagement.duration_seconds == 900
        assert engagement.completion_percentage == 85.0
        assert engagement.engagement_score > 0
        assert engagement.feedback_rating == 4.5
        
        # Verify engagement stored
        assert "test_content" in tracker.content_engagements
        assert len(tracker.content_engagements["test_content"]) == 1
    
    @pytest.mark.asyncio
    async def test_generate_personalized_recommendations(self, tracker):
        """Test generating personalized recommendations"""
        # First track some engagement to build profile
        engagement_data = {
            "start_time": datetime.now().isoformat(),
            "end_time": (datetime.now() + timedelta(minutes=10)).isoformat(),
            "duration_seconds": 600,
            "completion_percentage": 90.0,
            "interactions": [{"type": "click", "timestamp": datetime.now().isoformat()}],
            "feedback_rating": 4.0
        }
        
        await tracker.track_content_engagement("content_1", "user_1", engagement_data)
        
        # Generate recommendations
        recommendations = await tracker.generate_personalized_recommendations("user_1", 3)
        
        assert len(recommendations) <= 3
        for rec in recommendations:
            assert rec.user_id == "user_1"
            assert rec.recommendation_score > 0
            assert len(rec.reasoning) > 0
            assert rec.estimated_engagement_probability > 0
    
    @pytest.mark.asyncio
    async def test_analyze_learning_effectiveness(self, tracker):
        """Test analyzing learning effectiveness"""
        # Track some engagements first
        users = ["user_1", "user_2", "user_3"]
        contents = ["content_1", "content_2"]
        
        for user in users:
            for content in contents:
                engagement_data = {
                    "start_time": datetime.now().isoformat(),
                    "end_time": (datetime.now() + timedelta(minutes=12)).isoformat(),
                    "duration_seconds": 720,
                    "completion_percentage": 80.0,
                    "interactions": [{"type": "click", "timestamp": datetime.now().isoformat()}],
                    "feedback_rating": 4.2
                }
                
                await tracker.track_content_engagement(content, user, engagement_data)
        
        # Analyze effectiveness
        analysis = await tracker.analyze_learning_effectiveness()
        
        assert "total_engagements" in analysis
        assert "unique_users" in analysis
        assert "average_engagement_score" in analysis
        assert "overall_completion_rate" in analysis
        assert "content_performance_ranking" in analysis
        assert "user_learning_progress" in analysis
        
        assert analysis["total_engagements"] == 6  # 3 users × 2 contents
        assert analysis["unique_users"] == 3
    
    @pytest.mark.asyncio
    async def test_export_analytics_report(self, tracker):
        """Test exporting analytics report"""
        # Track some engagement first
        engagement_data = {
            "start_time": datetime.now().isoformat(),
            "end_time": (datetime.now() + timedelta(minutes=10)).isoformat(),
            "duration_seconds": 600,
            "completion_percentage": 75.0,
            "interactions": [{"type": "click", "timestamp": datetime.now().isoformat()}],
            "feedback_rating": 4.0
        }
        
        await tracker.track_content_engagement("test_content", "test_user", engagement_data)
        
        # Export report
        report = await tracker.export_analytics_report()
        
        assert "report_metadata" in report
        assert "executive_summary" in report
        assert "content_performance" in report
        assert "user_analytics" in report
        assert "learning_path_effectiveness" in report
        assert "engagement_trends" in report
        assert "recommendations" in report
        assert "detailed_metrics" in report
        
        # Verify metadata
        metadata = report["report_metadata"]
        assert metadata["report_type"] == "comprehensive_learning_analytics"
        assert metadata["format"] == "json"
    
    def test_learning_paths_initialization(self, tracker):
        """Test that learning paths are properly initialized"""
        assert len(tracker.learning_paths) > 0
        
        # Check CTO learning path
        cto_path = tracker.learning_paths.get("cto_ai_leadership_path")
        assert cto_path is not None
        assert cto_path.target_segment == "enterprise_ctos"
        assert len(cto_path.content_sequence) > 0
        assert cto_path.estimated_duration_hours > 0
        
        # Check tech leader learning path
        tech_path = tracker.learning_paths.get("tech_leader_implementation_path")
        assert tech_path is not None
        assert tech_path.target_segment == "tech_leaders"
        assert tech_path.difficulty_level == "advanced"

class TestMarketReadinessAssessment:
    """Test the market readiness assessment system"""
    
    @pytest.fixture
    def assessment(self):
        return MarketReadinessAssessment()
    
    @pytest.mark.asyncio
    async def test_assess_segment_readiness(self, assessment):
        """Test assessing segment readiness"""
        segment = MarketSegment.ENTERPRISE_CTOS
        
        segment_assessment = await assessment.assess_segment_readiness(segment)
        
        assert segment_assessment.segment == segment
        assert isinstance(segment_assessment.readiness_level, ReadinessLevel)
        assert 0 <= segment_assessment.readiness_score <= 100
        assert 0 <= segment_assessment.confidence_level <= 1
        assert len(segment_assessment.indicators) > 0
        assert len(segment_assessment.barriers) > 0
        assert len(segment_assessment.accelerators) > 0
        assert len(segment_assessment.recommended_actions) > 0
        
        # Verify indicators have proper structure
        for indicator in segment_assessment.indicators.values():
            assert indicator.weight > 0
            assert indicator.current_value >= 0
            assert indicator.target_value > 0
            assert indicator.trend in ["increasing", "decreasing", "stable"]
    
    @pytest.mark.asyncio
    async def test_monitor_market_conditions(self, assessment):
        """Test monitoring market conditions"""
        conditions = await assessment.monitor_market_conditions()
        
        assert len(conditions) > 0
        
        for condition in conditions:
            assert condition.factor
            assert condition.impact_level in ["high", "medium", "low"]
            assert condition.trend in ["positive", "negative", "neutral"]
            assert condition.description
            assert condition.recommended_response
    
    @pytest.mark.asyncio
    async def test_adapt_strategy_based_on_conditions(self, assessment):
        """Test adapting strategy based on conditions"""
        # First monitor conditions
        await assessment.monitor_market_conditions()
        
        # Adapt strategy
        adaptations = await assessment.adapt_strategy_based_on_conditions()
        
        assert len(adaptations) > 0
        
        for adaptation in adaptations:
            assert adaptation.id
            assert len(adaptation.trigger_conditions) > 0
            assert len(adaptation.actions) > 0
            assert adaptation.expected_impact > 0
            assert adaptation.implementation_timeline > 0
            assert len(adaptation.success_metrics) > 0
    
    @pytest.mark.asyncio
    async def test_generate_comprehensive_report(self, assessment):
        """Test generating comprehensive report"""
        report = await assessment.generate_comprehensive_report()
        
        assert "executive_summary" in report
        assert "segment_assessments" in report
        assert "market_conditions" in report
        assert "adaptation_strategies" in report
        assert "recommendations" in report
        assert "generated_at" in report
        
        # Verify executive summary
        exec_summary = report["executive_summary"]
        assert "overall_readiness_score" in exec_summary
        assert "market_readiness_level" in exec_summary
        assert "total_segments_assessed" in exec_summary
        assert "key_opportunities" in exec_summary
        assert "critical_barriers" in exec_summary
        
        # Verify segment assessments
        assert len(report["segment_assessments"]) > 0
        for segment_data in report["segment_assessments"].values():
            assert "readiness_score" in segment_data
            assert "confidence_level" in segment_data
            assert "barriers" in segment_data
            assert "accelerators" in segment_data
            assert "recommendations" in segment_data
    
    @pytest.mark.asyncio
    async def test_create_adaptive_strategy_framework(self, assessment):
        """Test creating adaptive strategy framework"""
        framework = await assessment.create_adaptive_strategy_framework()
        
        assert framework["name"] == "Market Readiness Adaptive Strategy Framework"
        assert "components" in framework
        assert "adaptation_rules" in framework
        assert "success_metrics" in framework
        assert "escalation_procedures" in framework
        
        # Verify components
        components = framework["components"]
        assert "monitoring_system" in components
        assert "trigger_conditions" in components
        assert "response_strategies" in components
        assert "feedback_loops" in components
        assert "optimization_engine" in components
        
        # Verify monitoring system
        monitoring = components["monitoring_system"]
        assert len(monitoring["real_time_indicators"]) > 0
        assert len(monitoring["data_sources"]) > 0
        assert len(monitoring["alert_thresholds"]) > 0
        
        # Verify trigger conditions
        triggers = components["trigger_conditions"]
        assert len(triggers) > 0
        for trigger in triggers:
            assert "condition_id" in trigger
            assert "description" in trigger
            assert "trigger_criteria" in trigger
            assert "severity" in trigger
            assert "response_time_hours" in trigger
        
        # Verify response strategies
        responses = components["response_strategies"]
        assert len(responses) > 0
        for strategy in responses.values():
            assert "immediate_actions" in strategy
            assert "medium_term_actions" in strategy
            assert "resource_allocation" in strategy
            assert "success_metrics" in strategy
        
        # Verify escalation procedures
        escalation = framework["escalation_procedures"]
        assert "escalation_levels" in escalation
        assert len(escalation["escalation_levels"]) == 4  # 4 levels defined
        
        for level in escalation["escalation_levels"]:
            assert "level" in level
            assert "name" in level
            assert "triggers" in level
            assert "response_team" in level
            assert "response_time_hours" in level
            assert "authority_level" in level

class TestSystemIntegration:
    """Test integration between all systems"""
    
    @pytest.fixture
    def systems(self):
        return {
            "platform": CampaignManagementPlatform(),
            "tracker": EducationContentTracker(),
            "assessment": MarketReadinessAssessment()
        }
    
    @pytest.mark.asyncio
    async def test_integrated_workflow(self, systems):
        """Test integrated workflow across all systems"""
        platform = systems["platform"]
        tracker = systems["tracker"]
        assessment = systems["assessment"]
        
        # 1. Create master plan
        master_plan = await platform.create_five_year_master_plan()
        assert master_plan is not None
        
        # 2. Generate readiness report
        readiness_report = await assessment.generate_comprehensive_report()
        assert readiness_report["overall_readiness"] > 0
        
        # 3. Track some content engagement
        engagement_data = {
            "start_time": datetime.now().isoformat(),
            "end_time": (datetime.now() + timedelta(minutes=10)).isoformat(),
            "duration_seconds": 600,
            "completion_percentage": 80.0,
            "interactions": [{"type": "click", "timestamp": datetime.now().isoformat()}],
            "feedback_rating": 4.0
        }
        
        engagement = await tracker.track_content_engagement(
            "integrated_content", "integrated_user", engagement_data
        )
        assert engagement.engagement_score > 0
        
        # 4. Execute campaign
        campaign_id = master_plan["yearly_campaigns"][0]["campaign_id"]
        execution_result = await platform.execute_campaign(campaign_id)
        assert execution_result["status"] == "executing"
        
        # 5. Monitor performance
        performance_report = await platform.monitor_campaign_performance(campaign_id)
        assert performance_report["overall_health_score"] > 0
        
        # 6. Generate analytics
        analytics_report = await tracker.export_analytics_report()
        assert "executive_summary" in analytics_report
        
        # Verify integration points
        assert len(platform.campaigns) > 0
        assert len(tracker.content_engagements) > 0
        assert len(assessment.segment_assessments) > 0
    
    @pytest.mark.asyncio
    async def test_cross_system_data_flow(self, systems):
        """Test data flow between systems"""
        platform = systems["platform"]
        tracker = systems["tracker"]
        assessment = systems["assessment"]
        
        # Create master plan
        master_plan = await platform.create_five_year_master_plan()
        
        # Assess market readiness
        readiness_report = await assessment.generate_comprehensive_report()
        
        # Verify data can be correlated
        campaign_count = len(master_plan["yearly_campaigns"])
        segment_count = len(readiness_report["segment_assessments"])
        learning_path_count = len(tracker.learning_paths)
        
        assert campaign_count > 0
        assert segment_count > 0
        assert learning_path_count > 0
        
        # Verify systems can work with shared data
        for campaign in master_plan["yearly_campaigns"]:
            assert "budget" in campaign
            assert campaign["budget"] > 0
        
        for segment_data in readiness_report["segment_assessments"].values():
            assert "readiness_score" in segment_data
            assert segment_data["readiness_score"] >= 0
        
        for path in tracker.learning_paths.values():
            assert len(path.content_sequence) > 0
            assert path.estimated_duration_hours > 0

@pytest.mark.asyncio
async def test_system_performance():
    """Test system performance under load"""
    platform = CampaignManagementPlatform()
    tracker = EducationContentTracker()
    
    # Test multiple concurrent operations
    tasks = []
    
    # Create multiple engagement tracking tasks
    for i in range(10):
        engagement_data = {
            "start_time": datetime.now().isoformat(),
            "end_time": (datetime.now() + timedelta(minutes=5)).isoformat(),
            "duration_seconds": 300,
            "completion_percentage": 70.0 + i,
            "interactions": [{"type": "click", "timestamp": datetime.now().isoformat()}],
            "feedback_rating": 3.5 + (i * 0.1)
        }
        
        task = tracker.track_content_engagement(f"content_{i}", f"user_{i}", engagement_data)
        tasks.append(task)
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Verify all completed successfully
    assert len(results) == 10
    for result in results:
        assert result.engagement_score > 0
    
    # Verify data integrity
    assert len(tracker.content_engagements) == 10
    assert len(tracker.user_profiles) == 10

if __name__ == "__main__":
    pytest.main([__file__, "-v"])