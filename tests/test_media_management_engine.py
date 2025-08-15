"""
Tests for Media Management Engine

This module contains comprehensive tests for the media management system
used in crisis leadership excellence.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.media_management_engine import MediaManagementEngine
from scrollintel.models.media_management_models import (
    MediaInquiry, MediaOutlet, PRStrategy, MediaMention,
    MediaInquiryType, InquiryPriority, MediaOutletType, SentimentScore
)


class TestMediaManagementEngine:
    """Test cases for MediaManagementEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create a MediaManagementEngine instance for testing"""
        return MediaManagementEngine()
    
    @pytest.fixture
    def sample_media_outlet(self):
        """Create a sample media outlet for testing"""
        return MediaOutlet(
            name="Tech News Daily",
            outlet_type=MediaOutletType.ONLINE_NEWS,
            reach=500000,
            influence_score=75.0,
            contact_info={"email": "news@technews.com"},
            typical_response_time=60
        )
    
    @pytest.fixture
    def sample_media_inquiry(self, sample_media_outlet):
        """Create a sample media inquiry for testing"""
        return MediaInquiry(
            crisis_id="crisis_123",
            outlet=sample_media_outlet,
            reporter_name="Jane Reporter",
            reporter_contact="jane@technews.com",
            inquiry_type=MediaInquiryType.STATEMENT_REQUEST,
            subject="Company Response to Recent Incident",
            questions=["What is the company's official response?", "What steps are being taken?"],
            deadline=datetime.now() + timedelta(hours=4)
        )
    
    @pytest.fixture
    def sample_pr_strategy(self):
        """Create a sample PR strategy for testing"""
        return PRStrategy(
            crisis_id="crisis_123",
            strategy_name="Crisis Response Strategy",
            objectives=["Maintain stakeholder confidence", "Provide transparent communication"],
            target_audiences=["Customers", "Investors", "Media", "Employees"],
            key_messages=[
                "We are taking immediate action to address the situation",
                "Customer safety is our top priority",
                "We are committed to full transparency"
            ],
            communication_channels=["Press releases", "Social media", "Direct stakeholder communication"]
        )
    
    @pytest.mark.asyncio
    async def test_handle_media_inquiry_basic(self, engine, sample_media_inquiry):
        """Test basic media inquiry handling"""
        response = await engine.handle_media_inquiry(sample_media_inquiry)
        
        assert response is not None
        assert response.inquiry_id == sample_media_inquiry.id
        assert response.response_type in ["statement", "interview", "factual_response"]
        assert len(response.key_messages) > 0
        assert response.content != ""
        assert sample_media_inquiry.id in engine.active_inquiries
    
    @pytest.mark.asyncio
    async def test_assess_inquiry_priority_high_influence(self, engine, sample_media_inquiry):
        """Test priority assessment for high-influence outlet"""
        sample_media_inquiry.outlet.influence_score = 90.0
        sample_media_inquiry.outlet.reach = 2000000
        sample_media_inquiry.inquiry_type = MediaInquiryType.BREAKING_NEWS
        sample_media_inquiry.deadline = datetime.now() + timedelta(hours=1)
        
        priority = await engine._assess_inquiry_priority(sample_media_inquiry)
        
        assert priority in [InquiryPriority.CRITICAL, InquiryPriority.HIGH]
    
    @pytest.mark.asyncio
    async def test_assess_inquiry_priority_low_influence(self, engine, sample_media_inquiry):
        """Test priority assessment for low-influence outlet"""
        sample_media_inquiry.outlet.influence_score = 30.0
        sample_media_inquiry.outlet.reach = 10000
        sample_media_inquiry.inquiry_type = MediaInquiryType.FOLLOW_UP
        sample_media_inquiry.deadline = datetime.now() + timedelta(days=1)
        
        priority = await engine._assess_inquiry_priority(sample_media_inquiry)
        
        assert priority in [InquiryPriority.LOW, InquiryPriority.MEDIUM]
    
    @pytest.mark.asyncio
    async def test_generate_response_strategy_statement(self, engine, sample_media_inquiry):
        """Test response strategy generation for statement request"""
        strategy = await engine._generate_response_strategy(sample_media_inquiry)
        
        assert strategy['type'] == 'statement'
        assert len(strategy['key_messages']) > 0
        assert strategy['draft_content'] != ''
        assert strategy['tone'] == 'professional'
        assert strategy['approval_required'] is True
    
    @pytest.mark.asyncio
    async def test_generate_response_strategy_interview(self, engine, sample_media_inquiry):
        """Test response strategy generation for interview request"""
        sample_media_inquiry.inquiry_type = MediaInquiryType.INTERVIEW_REQUEST
        sample_media_inquiry.priority = InquiryPriority.CRITICAL
        
        strategy = await engine._generate_response_strategy(sample_media_inquiry)
        
        assert strategy['type'] == 'interview'
        assert len(strategy['key_messages']) > 0
    
    @pytest.mark.asyncio
    async def test_assign_spokesperson_ceo_critical(self, engine, sample_media_inquiry):
        """Test spokesperson assignment for critical inquiry"""
        sample_media_inquiry.priority = InquiryPriority.CRITICAL
        
        spokesperson = await engine._assign_spokesperson(sample_media_inquiry)
        
        assert spokesperson == "CEO"
    
    @pytest.mark.asyncio
    async def test_assign_spokesperson_legal_investigative(self, engine, sample_media_inquiry):
        """Test spokesperson assignment for investigative inquiry"""
        sample_media_inquiry.inquiry_type = MediaInquiryType.INVESTIGATIVE
        
        spokesperson = await engine._assign_spokesperson(sample_media_inquiry)
        
        assert spokesperson == "Legal Counsel"
    
    @pytest.mark.asyncio
    async def test_coordinate_pr_strategy(self, engine, sample_pr_strategy):
        """Test PR strategy coordination"""
        result = await engine.coordinate_pr_strategy(sample_pr_strategy.crisis_id, sample_pr_strategy)
        
        assert result is not None
        assert result['strategy_id'] == sample_pr_strategy.id
        assert 'consistency_score' in result
        assert 'timeline' in result
        assert 'assignments' in result
        assert 'monitoring_plan' in result
        assert result['status'] == 'active'
        assert sample_pr_strategy.id in engine.pr_strategies
    
    @pytest.mark.asyncio
    async def test_validate_message_consistency_good(self, engine, sample_pr_strategy):
        """Test message consistency validation with good messages"""
        result = await engine._validate_message_consistency(sample_pr_strategy)
        
        assert result['score'] >= 0
        assert result['score'] <= 100
        assert isinstance(result['issues'], list)
        assert isinstance(result['recommendations'], list)
    
    @pytest.mark.asyncio
    async def test_validate_message_consistency_conflicts(self, engine, sample_pr_strategy):
        """Test message consistency validation with conflicting messages"""
        sample_pr_strategy.key_messages = [
            "We deny any wrongdoing in this matter",
            "We confirm that mistakes were made and we take responsibility"
        ]
        
        result = await engine._validate_message_consistency(sample_pr_strategy)
        
        assert result['score'] < 100
        assert any("conflict" in issue.lower() for issue in result['issues'])
    
    @pytest.mark.asyncio
    async def test_messages_conflict_detection(self, engine):
        """Test conflict detection between messages"""
        msg1 = "We deny any involvement in this issue"
        msg2 = "We confirm our role in the situation"
        
        conflict = await engine._messages_conflict(msg1, msg2)
        
        assert conflict is True
    
    @pytest.mark.asyncio
    async def test_messages_no_conflict(self, engine):
        """Test no conflict detection between compatible messages"""
        msg1 = "We are committed to transparency"
        msg2 = "We will provide regular updates"
        
        conflict = await engine._messages_conflict(msg1, msg2)
        
        assert conflict is False
    
    @pytest.mark.asyncio
    async def test_create_communication_timeline(self, engine, sample_pr_strategy):
        """Test communication timeline creation"""
        timeline = await engine._create_communication_timeline(sample_pr_strategy)
        
        assert isinstance(timeline, dict)
        assert 'immediate_response' in timeline
        assert 'first_update' in timeline
        assert 'stakeholder_briefing' in timeline
        assert 'media_availability' in timeline
        
        # Verify timeline ordering
        times = list(timeline.values())
        assert all(times[i] <= times[i+1] for i in range(len(times)-1))
    
    @pytest.mark.asyncio
    async def test_assign_spokesperson_responsibilities(self, engine, sample_pr_strategy):
        """Test spokesperson responsibility assignment"""
        assignments = await engine._assign_spokesperson_responsibilities(sample_pr_strategy)
        
        assert isinstance(assignments, dict)
        assert 'CEO' in assignments
        assert 'Communications Director' in assignments
        assert 'Legal Counsel' in assignments
        assert 'Technical Lead' in assignments
        
        # Verify each has responsibilities
        for spokesperson, responsibilities in assignments.items():
            assert isinstance(responsibilities, list)
            assert len(responsibilities) > 0
    
    @pytest.mark.asyncio
    async def test_monitor_media_sentiment(self, engine):
        """Test media sentiment monitoring"""
        crisis_id = "crisis_123"
        time_period = timedelta(hours=24)
        
        analysis = await engine.monitor_media_sentiment(crisis_id, time_period)
        
        assert analysis is not None
        assert analysis.crisis_id == crisis_id
        assert analysis.overall_sentiment in [s for s in SentimentScore]
        assert analysis.mention_volume >= 0
        assert analysis.positive_mentions >= 0
        assert analysis.negative_mentions >= 0
        assert analysis.neutral_mentions >= 0
        assert isinstance(analysis.key_sentiment_drivers, list)
        assert isinstance(analysis.recommendations, list)
        assert analysis.id in engine.sentiment_history
    
    @pytest.mark.asyncio
    async def test_collect_media_mentions(self, engine):
        """Test media mention collection"""
        crisis_id = "crisis_123"
        time_period = timedelta(hours=24)
        
        mentions = await engine._collect_media_mentions(crisis_id, time_period)
        
        assert isinstance(mentions, list)
        # Should have sample mentions for testing
        assert len(mentions) >= 0
        
        for mention in mentions:
            assert isinstance(mention, MediaMention)
            assert mention.crisis_id == crisis_id
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_batch(self, engine):
        """Test batch sentiment analysis"""
        mentions = [
            MediaMention(
                crisis_id="crisis_123",
                headline="Company Handles Crisis Well",
                content="The company's response has been professional and effective"
            ),
            MediaMention(
                crisis_id="crisis_123",
                headline="Problems Continue to Mount",
                content="The situation continues to deteriorate with no clear solution"
            )
        ]
        
        results = await engine._analyze_sentiment_batch(mentions)
        
        assert len(results) == len(mentions)
        for result in results:
            assert 'mention_id' in result
            assert 'sentiment' in result
            assert 'confidence' in result
            assert 'key_phrases' in result
            assert result['sentiment'] in ['positive', 'negative', 'neutral']
            assert 0 <= result['confidence'] <= 1
    
    @pytest.mark.asyncio
    async def test_analyze_single_mention_sentiment_positive(self, engine):
        """Test single mention sentiment analysis - positive"""
        mention = MediaMention(
            headline="Company Achieves Success",
            content="The company has shown great improvement and achieved significant success"
        )
        
        result = await engine._analyze_single_mention_sentiment(mention)
        
        assert result['sentiment'] == 'positive'
        assert result['confidence'] > 0.5
        assert isinstance(result['key_phrases'], list)
    
    @pytest.mark.asyncio
    async def test_analyze_single_mention_sentiment_negative(self, engine):
        """Test single mention sentiment analysis - negative"""
        mention = MediaMention(
            headline="Company Faces Crisis",
            content="The company is dealing with a major crisis and significant problems"
        )
        
        result = await engine._analyze_single_mention_sentiment(mention)
        
        assert result['sentiment'] == 'negative'
        assert result['confidence'] > 0.5
        assert isinstance(result['key_phrases'], list)
    
    @pytest.mark.asyncio
    async def test_calculate_overall_sentiment_positive(self, engine):
        """Test overall sentiment calculation - positive"""
        sentiment_results = [
            {'sentiment': 'positive'},
            {'sentiment': 'positive'},
            {'sentiment': 'neutral'}
        ]
        
        overall = await engine._calculate_overall_sentiment(sentiment_results)
        
        assert overall in [SentimentScore.POSITIVE, SentimentScore.NEUTRAL]
    
    @pytest.mark.asyncio
    async def test_calculate_overall_sentiment_negative(self, engine):
        """Test overall sentiment calculation - negative"""
        sentiment_results = [
            {'sentiment': 'negative'},
            {'sentiment': 'negative'},
            {'sentiment': 'negative'},
            {'sentiment': 'neutral'}
        ]
        
        overall = await engine._calculate_overall_sentiment(sentiment_results)
        
        assert overall == SentimentScore.NEGATIVE
    
    @pytest.mark.asyncio
    async def test_identify_sentiment_drivers(self, engine):
        """Test sentiment driver identification"""
        sentiment_results = [
            {'key_phrases': ['success', 'achievement', 'improvement']},
            {'key_phrases': ['success', 'solution', 'recovery']},
            {'key_phrases': ['problem', 'crisis', 'failure']}
        ]
        
        drivers = await engine._identify_sentiment_drivers(sentiment_results)
        
        assert isinstance(drivers, list)
        assert len(drivers) <= 5
        # 'success' should be top driver (appears twice)
        if drivers:
            assert 'success' in drivers
    
    @pytest.mark.asyncio
    async def test_generate_sentiment_recommendations_negative(self, engine):
        """Test sentiment recommendation generation for negative sentiment"""
        sentiment_results = [
            {'sentiment': 'negative'} for _ in range(8)
        ] + [
            {'sentiment': 'positive'} for _ in range(2)
        ]
        
        recommendations = await engine._generate_sentiment_recommendations(sentiment_results)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("negative sentiment" in rec.lower() for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_check_sentiment_alerts(self, engine):
        """Test sentiment alert checking"""
        from scrollintel.models.media_management_models import SentimentAnalysis
        
        analysis = SentimentAnalysis(
            crisis_id="crisis_123",
            negative_mentions=8,
            positive_mentions=2,
            mention_volume=10
        )
        
        await engine._check_sentiment_alerts(analysis)
        
        # Should create alert for negative sentiment spike
        assert len(engine.monitoring_alerts) > 0
        
        # Find the sentiment alert
        sentiment_alerts = [
            alert for alert in engine.monitoring_alerts.values()
            if alert.alert_type == "sentiment_drop"
        ]
        assert len(sentiment_alerts) > 0
        
        alert = sentiment_alerts[0]
        assert alert.crisis_id == "crisis_123"
        assert alert.severity == "high"
    
    @pytest.mark.asyncio
    async def test_get_media_management_metrics(self, engine, sample_media_inquiry):
        """Test media management metrics calculation"""
        # Add some test data
        await engine.handle_media_inquiry(sample_media_inquiry)
        
        metrics = await engine.get_media_management_metrics("crisis_123")
        
        assert metrics is not None
        assert metrics.crisis_id == "crisis_123"
        assert metrics.total_inquiries >= 0
        assert 0 <= metrics.response_rate <= 1
        assert metrics.average_response_time >= 0
        assert 0 <= metrics.positive_coverage_percentage <= 100
        assert metrics.media_reach >= 0
        assert 0 <= metrics.message_consistency_score <= 100
        assert isinstance(metrics.spokesperson_effectiveness, dict)
        assert 0 <= metrics.crisis_narrative_control <= 100
        assert 0 <= metrics.reputation_impact_score <= 100
    
    @pytest.mark.asyncio
    async def test_setup_strategy_monitoring(self, engine, sample_pr_strategy):
        """Test PR strategy monitoring setup"""
        monitoring_plan = await engine._setup_strategy_monitoring(sample_pr_strategy)
        
        assert isinstance(monitoring_plan, dict)
        assert 'metrics_to_track' in monitoring_plan
        assert 'monitoring_frequency' in monitoring_plan
        assert 'alert_thresholds' in monitoring_plan
        assert 'reporting_schedule' in monitoring_plan
        
        assert isinstance(monitoring_plan['metrics_to_track'], list)
        assert len(monitoring_plan['metrics_to_track']) > 0
        assert isinstance(monitoring_plan['alert_thresholds'], dict)
    
    @pytest.mark.asyncio
    async def test_generate_key_messages_investigative(self, engine, sample_media_inquiry):
        """Test key message generation for investigative inquiry"""
        sample_media_inquiry.inquiry_type = MediaInquiryType.INVESTIGATIVE
        
        messages = await engine._generate_key_messages(sample_media_inquiry)
        
        assert isinstance(messages, list)
        assert len(messages) > 0
        assert any("accurate information" in msg.lower() for msg in messages)
    
    @pytest.mark.asyncio
    async def test_generate_key_messages_breaking_news(self, engine, sample_media_inquiry):
        """Test key message generation for breaking news inquiry"""
        sample_media_inquiry.inquiry_type = MediaInquiryType.BREAKING_NEWS
        
        messages = await engine._generate_key_messages(sample_media_inquiry)
        
        assert isinstance(messages, list)
        assert len(messages) > 0
        assert any("developing situation" in msg.lower() for msg in messages)
    
    @pytest.mark.asyncio
    async def test_draft_response_content(self, engine, sample_media_inquiry):
        """Test response content drafting"""
        strategy = {
            'key_messages': [
                "We are taking this seriously",
                "Safety is our priority"
            ]
        }
        
        content = await engine._draft_response_content(sample_media_inquiry, strategy)
        
        assert isinstance(content, str)
        assert len(content) > 0
        assert sample_media_inquiry.subject in content
        assert "Thank you for reaching out" in content
    
    def test_response_time_targets_configuration(self, engine):
        """Test response time targets are properly configured"""
        assert InquiryPriority.CRITICAL in engine.response_time_targets
        assert InquiryPriority.HIGH in engine.response_time_targets
        assert InquiryPriority.MEDIUM in engine.response_time_targets
        assert InquiryPriority.LOW in engine.response_time_targets
        
        # Critical should have shortest response time
        assert engine.response_time_targets[InquiryPriority.CRITICAL] < engine.response_time_targets[InquiryPriority.HIGH]
        assert engine.response_time_targets[InquiryPriority.HIGH] < engine.response_time_targets[InquiryPriority.MEDIUM]
        assert engine.response_time_targets[InquiryPriority.MEDIUM] < engine.response_time_targets[InquiryPriority.LOW]
    
    def test_sentiment_keywords_configuration(self, engine):
        """Test sentiment keywords are properly configured"""
        assert 'positive' in engine.sentiment_keywords
        assert 'negative' in engine.sentiment_keywords
        
        assert isinstance(engine.sentiment_keywords['positive'], list)
        assert isinstance(engine.sentiment_keywords['negative'], list)
        assert len(engine.sentiment_keywords['positive']) > 0
        assert len(engine.sentiment_keywords['negative']) > 0
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_inquiry(self, engine):
        """Test error handling for invalid inquiry"""
        invalid_inquiry = None
        
        with pytest.raises(Exception):
            await engine.handle_media_inquiry(invalid_inquiry)
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_strategy(self, engine):
        """Test error handling for invalid PR strategy"""
        invalid_strategy = None
        
        with pytest.raises(Exception):
            await engine.coordinate_pr_strategy("crisis_123", invalid_strategy)