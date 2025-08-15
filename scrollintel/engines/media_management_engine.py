"""
Media Management Engine for Crisis Leadership Excellence

This engine handles professional media inquiry processing, public relations strategy
coordination, and media monitoring with sentiment analysis during crisis situations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
import re
from collections import defaultdict

from ..models.media_management_models import (
    MediaInquiry, MediaResponse, MediaOutlet, PRStrategy, MediaMention,
    SentimentAnalysis, MediaMonitoringAlert, MediaManagementMetrics,
    MediaInquiryType, InquiryPriority, ResponseStatus, SentimentScore,
    MediaOutletType
)
from ..models.crisis_models_simple import Crisis


class MediaManagementEngine:
    """
    Advanced media management system for crisis situations
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.media_outlets = {}
        self.active_inquiries = {}
        self.pr_strategies = {}
        self.media_mentions = {}
        self.sentiment_history = {}
        self.monitoring_alerts = {}
        
        # Configuration
        self.response_time_targets = {
            InquiryPriority.CRITICAL: 15,  # minutes
            InquiryPriority.HIGH: 30,
            InquiryPriority.MEDIUM: 60,
            InquiryPriority.LOW: 120
        }
        
        self.sentiment_keywords = {
            'positive': ['success', 'achievement', 'improvement', 'solution', 'recovery'],
            'negative': ['failure', 'crisis', 'problem', 'scandal', 'controversy', 'damage']
        }
    
    async def handle_media_inquiry(self, inquiry: MediaInquiry) -> MediaResponse:
        """
        Process incoming media inquiry with professional handling
        """
        try:
            # Store inquiry
            self.active_inquiries[inquiry.id] = inquiry
            
            # Assess inquiry priority and urgency
            priority = await self._assess_inquiry_priority(inquiry)
            inquiry.priority = priority
            
            # Generate recommended response strategy
            response_strategy = await self._generate_response_strategy(inquiry)
            
            # Create initial response framework
            response = MediaResponse(
                inquiry_id=inquiry.id,
                response_type=response_strategy['type'],
                key_messages=response_strategy['key_messages'],
                content=response_strategy['draft_content']
            )
            
            # Route to appropriate spokesperson
            spokesperson = await self._assign_spokesperson(inquiry)
            inquiry.assigned_spokesperson = spokesperson
            
            # Set response timeline
            deadline_buffer = self.response_time_targets.get(priority, 60)
            target_response_time = inquiry.received_at + timedelta(minutes=deadline_buffer)
            
            self.logger.info(f"Media inquiry {inquiry.id} processed, assigned to {spokesperson}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error handling media inquiry: {str(e)}")
            raise
    
    async def _assess_inquiry_priority(self, inquiry: MediaInquiry) -> InquiryPriority:
        """
        Assess the priority level of a media inquiry
        """
        priority_score = 0
        
        # Outlet influence factor
        if inquiry.outlet.influence_score > 80:
            priority_score += 3
        elif inquiry.outlet.influence_score > 60:
            priority_score += 2
        elif inquiry.outlet.influence_score > 40:
            priority_score += 1
        
        # Outlet reach factor
        if inquiry.outlet.reach > 1000000:
            priority_score += 2
        elif inquiry.outlet.reach > 100000:
            priority_score += 1
        
        # Inquiry type factor
        if inquiry.inquiry_type in [MediaInquiryType.BREAKING_NEWS, MediaInquiryType.INVESTIGATIVE]:
            priority_score += 3
        elif inquiry.inquiry_type == MediaInquiryType.INTERVIEW_REQUEST:
            priority_score += 2
        
        # Deadline urgency
        time_to_deadline = (inquiry.deadline - datetime.now()).total_seconds() / 3600
        if time_to_deadline < 2:
            priority_score += 3
        elif time_to_deadline < 6:
            priority_score += 2
        elif time_to_deadline < 24:
            priority_score += 1
        
        # Convert score to priority
        if priority_score >= 8:
            return InquiryPriority.CRITICAL
        elif priority_score >= 5:
            return InquiryPriority.HIGH
        elif priority_score >= 3:
            return InquiryPriority.MEDIUM
        else:
            return InquiryPriority.LOW
    
    async def _generate_response_strategy(self, inquiry: MediaInquiry) -> Dict[str, Any]:
        """
        Generate appropriate response strategy for media inquiry
        """
        strategy = {
            'type': 'statement',
            'key_messages': [],
            'draft_content': '',
            'tone': 'professional',
            'approval_required': True
        }
        
        # Determine response type based on inquiry
        if inquiry.inquiry_type == MediaInquiryType.INTERVIEW_REQUEST:
            if inquiry.priority in [InquiryPriority.CRITICAL, InquiryPriority.HIGH]:
                strategy['type'] = 'interview'
            else:
                strategy['type'] = 'statement'
        elif inquiry.inquiry_type == MediaInquiryType.FACT_CHECK:
            strategy['type'] = 'factual_response'
        
        # Generate key messages based on crisis context
        strategy['key_messages'] = await self._generate_key_messages(inquiry)
        
        # Create draft response content
        strategy['draft_content'] = await self._draft_response_content(inquiry, strategy)
        
        return strategy
    
    async def _generate_key_messages(self, inquiry: MediaInquiry) -> List[str]:
        """
        Generate key messages for media response
        """
        key_messages = [
            "We are taking this situation very seriously and are committed to transparency",
            "The safety and well-being of our stakeholders is our top priority",
            "We are working diligently to address the situation and prevent recurrence",
            "We will continue to provide updates as more information becomes available"
        ]
        
        # Customize based on inquiry type
        if inquiry.inquiry_type == MediaInquiryType.INVESTIGATIVE:
            key_messages.append("We welcome the opportunity to provide accurate information")
        elif inquiry.inquiry_type == MediaInquiryType.BREAKING_NEWS:
            key_messages.insert(0, "We are aware of the developing situation and are responding immediately")
        
        return key_messages
    
    async def _draft_response_content(self, inquiry: MediaInquiry, strategy: Dict) -> str:
        """
        Draft initial response content
        """
        content_template = f"""
Thank you for reaching out regarding {inquiry.subject}.

{' '.join(strategy['key_messages'])}

We understand the importance of providing timely and accurate information to the public and media. 
Our team is currently reviewing the details of your inquiry and will provide a comprehensive response 
within the appropriate timeframe.

For immediate questions, please contact our media relations team.

Best regards,
[Spokesperson Name]
[Company] Communications Team
        """.strip()
        
        return content_template
    
    async def _assign_spokesperson(self, inquiry: MediaInquiry) -> str:
        """
        Assign appropriate spokesperson based on inquiry characteristics
        """
        # Default spokesperson assignment logic
        if inquiry.priority == InquiryPriority.CRITICAL:
            return "CEO"
        elif inquiry.inquiry_type == MediaInquiryType.INVESTIGATIVE:
            return "Legal Counsel"
        elif inquiry.outlet.outlet_type == MediaOutletType.TRADE_PUBLICATION:
            return "Technical Lead"
        else:
            return "Communications Director"
    
    async def coordinate_pr_strategy(self, crisis_id: str, strategy: PRStrategy) -> Dict[str, Any]:
        """
        Coordinate public relations strategy and message consistency
        """
        try:
            # Store PR strategy
            self.pr_strategies[strategy.id] = strategy
            
            # Ensure message consistency across all communications
            consistency_check = await self._validate_message_consistency(strategy)
            
            # Create communication timeline
            timeline = await self._create_communication_timeline(strategy)
            
            # Assign spokesperson responsibilities
            assignments = await self._assign_spokesperson_responsibilities(strategy)
            
            # Set up monitoring and tracking
            monitoring_plan = await self._setup_strategy_monitoring(strategy)
            
            coordination_result = {
                'strategy_id': strategy.id,
                'consistency_score': consistency_check['score'],
                'timeline': timeline,
                'assignments': assignments,
                'monitoring_plan': monitoring_plan,
                'status': 'active',
                'next_review': datetime.now() + timedelta(hours=4)
            }
            
            self.logger.info(f"PR strategy {strategy.id} coordinated for crisis {crisis_id}")
            
            return coordination_result
            
        except Exception as e:
            self.logger.error(f"Error coordinating PR strategy: {str(e)}")
            raise
    
    async def _validate_message_consistency(self, strategy: PRStrategy) -> Dict[str, Any]:
        """
        Validate consistency of key messages across strategy
        """
        consistency_issues = []
        score = 100.0
        
        # Check for conflicting messages
        messages = strategy.key_messages
        for i, msg1 in enumerate(messages):
            for j, msg2 in enumerate(messages[i+1:], i+1):
                if await self._messages_conflict(msg1, msg2):
                    consistency_issues.append(f"Potential conflict between messages {i+1} and {j+1}")
                    score -= 10
        
        # Check message clarity and specificity
        for i, message in enumerate(messages):
            if len(message.split()) < 5:
                consistency_issues.append(f"Message {i+1} may be too brief")
                score -= 5
            elif len(message.split()) > 30:
                consistency_issues.append(f"Message {i+1} may be too complex")
                score -= 5
        
        return {
            'score': max(0, score),
            'issues': consistency_issues,
            'recommendations': await self._generate_consistency_recommendations(consistency_issues)
        }
    
    async def _messages_conflict(self, msg1: str, msg2: str) -> bool:
        """
        Check if two messages potentially conflict
        """
        # Simple conflict detection based on contradictory keywords
        contradictions = [
            (['no', 'not', 'never'], ['yes', 'will', 'always']),
            (['deny', 'false'], ['confirm', 'true']),
            (['minimal', 'small'], ['significant', 'major'])
        ]
        
        msg1_lower = msg1.lower()
        msg2_lower = msg2.lower()
        
        for negative_words, positive_words in contradictions:
            has_negative = any(word in msg1_lower for word in negative_words)
            has_positive = any(word in msg2_lower for word in positive_words)
            if has_negative and has_positive:
                return True
        
        return False
    
    async def _create_communication_timeline(self, strategy: PRStrategy) -> Dict[str, datetime]:
        """
        Create detailed communication timeline
        """
        now = datetime.now()
        timeline = {
            'immediate_response': now + timedelta(minutes=30),
            'first_update': now + timedelta(hours=2),
            'stakeholder_briefing': now + timedelta(hours=4),
            'media_availability': now + timedelta(hours=6),
            'progress_update': now + timedelta(hours=12),
            'resolution_communication': now + timedelta(days=1)
        }
        
        return timeline
    
    async def _assign_spokesperson_responsibilities(self, strategy: PRStrategy) -> Dict[str, List[str]]:
        """
        Assign specific responsibilities to spokespersons
        """
        assignments = {
            'CEO': ['major_announcements', 'stakeholder_communications', 'crisis_resolution'],
            'Communications Director': ['media_inquiries', 'social_media', 'internal_communications'],
            'Legal Counsel': ['regulatory_matters', 'compliance_issues', 'legal_implications'],
            'Technical Lead': ['technical_explanations', 'solution_details', 'implementation_updates']
        }
        
        return assignments
    
    async def _setup_strategy_monitoring(self, strategy: PRStrategy) -> Dict[str, Any]:
        """
        Set up monitoring for PR strategy effectiveness
        """
        monitoring_plan = {
            'metrics_to_track': [
                'media_sentiment',
                'message_pickup',
                'stakeholder_response',
                'social_media_engagement'
            ],
            'monitoring_frequency': 'hourly',
            'alert_thresholds': {
                'sentiment_drop': -20,
                'negative_coverage_spike': 50,
                'message_distortion': 30
            },
            'reporting_schedule': {
                'real_time_dashboard': True,
                'hourly_updates': True,
                'daily_summary': True
            }
        }
        
        return monitoring_plan
    
    async def monitor_media_sentiment(self, crisis_id: str, time_period: timedelta = None) -> SentimentAnalysis:
        """
        Monitor and analyze media sentiment with real-time tracking
        """
        try:
            if time_period is None:
                time_period = timedelta(hours=24)
            
            # Collect media mentions for analysis period
            mentions = await self._collect_media_mentions(crisis_id, time_period)
            
            # Perform sentiment analysis on mentions
            sentiment_results = await self._analyze_sentiment_batch(mentions)
            
            # Calculate overall sentiment metrics
            overall_sentiment = await self._calculate_overall_sentiment(sentiment_results)
            
            # Identify sentiment trends
            sentiment_trend = await self._identify_sentiment_trends(crisis_id, sentiment_results)
            
            # Generate sentiment analysis report
            analysis = SentimentAnalysis(
                crisis_id=crisis_id,
                analysis_period={
                    'start': datetime.now() - time_period,
                    'end': datetime.now()
                },
                overall_sentiment=overall_sentiment,
                sentiment_trend=sentiment_trend,
                mention_volume=len(mentions),
                positive_mentions=len([m for m in sentiment_results if m['sentiment'] in ['positive', 'very_positive']]),
                negative_mentions=len([m for m in sentiment_results if m['sentiment'] in ['negative', 'very_negative']]),
                neutral_mentions=len([m for m in sentiment_results if m['sentiment'] == 'neutral']),
                key_sentiment_drivers=await self._identify_sentiment_drivers(sentiment_results),
                outlet_breakdown=await self._analyze_outlet_sentiment(sentiment_results),
                recommendations=await self._generate_sentiment_recommendations(sentiment_results)
            )
            
            # Store analysis results
            self.sentiment_history[analysis.id] = analysis
            
            # Check for alerts
            await self._check_sentiment_alerts(analysis)
            
            self.logger.info(f"Sentiment analysis completed for crisis {crisis_id}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error monitoring media sentiment: {str(e)}")
            raise
    
    async def _collect_media_mentions(self, crisis_id: str, time_period: timedelta) -> List[MediaMention]:
        """
        Collect media mentions for analysis
        """
        # Simulate media mention collection
        mentions = []
        
        # In a real implementation, this would integrate with:
        # - News APIs (Google News, Bing News)
        # - Social media monitoring tools
        # - RSS feeds
        # - Media monitoring services
        
        sample_mentions = [
            {
                'headline': 'Company Responds to Crisis Situation',
                'content': 'The company has issued a statement addressing the recent situation...',
                'outlet_name': 'Tech News Daily',
                'published_at': datetime.now() - timedelta(hours=2)
            },
            {
                'headline': 'Industry Experts Weigh In on Recent Developments',
                'content': 'Experts believe the company is handling the situation appropriately...',
                'outlet_name': 'Business Weekly',
                'published_at': datetime.now() - timedelta(hours=4)
            }
        ]
        
        for mention_data in sample_mentions:
            mention = MediaMention(
                crisis_id=crisis_id,
                headline=mention_data['headline'],
                content=mention_data['content'],
                published_at=mention_data['published_at']
            )
            mentions.append(mention)
        
        return mentions
    
    async def _analyze_sentiment_batch(self, mentions: List[MediaMention]) -> List[Dict[str, Any]]:
        """
        Perform sentiment analysis on batch of mentions
        """
        results = []
        
        for mention in mentions:
            sentiment_result = await self._analyze_single_mention_sentiment(mention)
            results.append({
                'mention_id': mention.id,
                'sentiment': sentiment_result['sentiment'],
                'confidence': sentiment_result['confidence'],
                'key_phrases': sentiment_result['key_phrases']
            })
        
        return results
    
    async def _analyze_single_mention_sentiment(self, mention: MediaMention) -> Dict[str, Any]:
        """
        Analyze sentiment of a single media mention
        """
        # Simple keyword-based sentiment analysis
        # In production, would use advanced NLP models
        
        text = f"{mention.headline} {mention.content}".lower()
        
        positive_score = sum(1 for word in self.sentiment_keywords['positive'] if word in text)
        negative_score = sum(1 for word in self.sentiment_keywords['negative'] if word in text)
        
        if positive_score > negative_score:
            sentiment = 'positive'
            confidence = min(0.9, 0.5 + (positive_score - negative_score) * 0.1)
        elif negative_score > positive_score:
            sentiment = 'negative'
            confidence = min(0.9, 0.5 + (negative_score - positive_score) * 0.1)
        else:
            sentiment = 'neutral'
            confidence = 0.6
        
        # Extract key phrases
        key_phrases = []
        for category, words in self.sentiment_keywords.items():
            found_words = [word for word in words if word in text]
            key_phrases.extend(found_words)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'key_phrases': key_phrases[:5]  # Top 5 key phrases
        }
    
    async def _calculate_overall_sentiment(self, sentiment_results: List[Dict]) -> SentimentScore:
        """
        Calculate overall sentiment from individual results
        """
        if not sentiment_results:
            return SentimentScore.NEUTRAL
        
        sentiment_counts = defaultdict(int)
        for result in sentiment_results:
            sentiment_counts[result['sentiment']] += 1
        
        total = len(sentiment_results)
        positive_ratio = (sentiment_counts['positive'] + sentiment_counts['very_positive']) / total
        negative_ratio = (sentiment_counts['negative'] + sentiment_counts['very_negative']) / total
        
        if positive_ratio > 0.6:
            return SentimentScore.POSITIVE
        elif negative_ratio > 0.6:
            return SentimentScore.NEGATIVE
        elif positive_ratio > 0.4:
            return SentimentScore.POSITIVE if positive_ratio > negative_ratio else SentimentScore.NEUTRAL
        elif negative_ratio > 0.4:
            return SentimentScore.NEGATIVE
        else:
            return SentimentScore.NEUTRAL
    
    async def _identify_sentiment_trends(self, crisis_id: str, current_results: List[Dict]) -> str:
        """
        Identify sentiment trends over time
        """
        # Compare with historical data
        if crisis_id in self.sentiment_history:
            # Simple trend analysis
            return "stable"  # Would implement proper trend analysis
        
        return "stable"
    
    async def _identify_sentiment_drivers(self, sentiment_results: List[Dict]) -> List[str]:
        """
        Identify key drivers of sentiment
        """
        all_phrases = []
        for result in sentiment_results:
            all_phrases.extend(result.get('key_phrases', []))
        
        # Count phrase frequency
        phrase_counts = defaultdict(int)
        for phrase in all_phrases:
            phrase_counts[phrase] += 1
        
        # Return top drivers
        top_drivers = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)
        return [phrase for phrase, count in top_drivers[:5]]
    
    async def _analyze_outlet_sentiment(self, sentiment_results: List[Dict]) -> Dict[str, Dict]:
        """
        Analyze sentiment breakdown by outlet
        """
        # Simplified outlet analysis
        return {
            'major_outlets': {'positive': 60, 'neutral': 30, 'negative': 10},
            'trade_publications': {'positive': 40, 'neutral': 40, 'negative': 20},
            'social_media': {'positive': 30, 'neutral': 20, 'negative': 50}
        }
    
    async def _generate_sentiment_recommendations(self, sentiment_results: List[Dict]) -> List[str]:
        """
        Generate recommendations based on sentiment analysis
        """
        recommendations = []
        
        negative_count = len([r for r in sentiment_results if r['sentiment'] in ['negative', 'very_negative']])
        total_count = len(sentiment_results)
        
        if negative_count / total_count > 0.4:
            recommendations.append("Consider proactive outreach to address negative sentiment")
            recommendations.append("Review and strengthen key messaging")
        
        if total_count < 10:
            recommendations.append("Increase media engagement to improve coverage volume")
        
        recommendations.append("Continue monitoring sentiment trends closely")
        
        return recommendations
    
    async def _check_sentiment_alerts(self, analysis: SentimentAnalysis):
        """
        Check if sentiment analysis triggers any alerts
        """
        alerts = []
        
        # Check for negative sentiment spike
        if analysis.negative_mentions > analysis.positive_mentions * 2:
            alert = MediaMonitoringAlert(
                crisis_id=analysis.crisis_id,
                alert_type="sentiment_drop",
                severity="high",
                description="Significant increase in negative media coverage detected",
                recommended_actions=[
                    "Review and adjust messaging strategy",
                    "Consider proactive media outreach",
                    "Prepare additional spokesperson statements"
                ]
            )
            alerts.append(alert)
        
        # Store alerts
        for alert in alerts:
            self.monitoring_alerts[alert.id] = alert
    
    async def _generate_consistency_recommendations(self, issues: List[str]) -> List[str]:
        """
        Generate recommendations for message consistency
        """
        recommendations = []
        
        if any("conflict" in issue.lower() for issue in issues):
            recommendations.append("Review conflicting messages and align positioning")
        
        if any("brief" in issue.lower() for issue in issues):
            recommendations.append("Expand brief messages with more context")
        
        if any("complex" in issue.lower() for issue in issues):
            recommendations.append("Simplify complex messages for better clarity")
        
        recommendations.append("Conduct message testing with key stakeholders")
        
        return recommendations
    
    async def get_media_management_metrics(self, crisis_id: str) -> MediaManagementMetrics:
        """
        Calculate comprehensive media management effectiveness metrics
        """
        try:
            # Collect relevant data
            crisis_inquiries = [inq for inq in self.active_inquiries.values() if inq.crisis_id == crisis_id]
            
            # Calculate metrics
            total_inquiries = len(crisis_inquiries)
            responded_inquiries = len([inq for inq in crisis_inquiries if inq.response_status != ResponseStatus.PENDING])
            response_rate = responded_inquiries / total_inquiries if total_inquiries > 0 else 0.0
            
            # Calculate average response time
            response_times = []
            for inquiry in crisis_inquiries:
                if inquiry.response_status == ResponseStatus.SENT:
                    # Simulate response time calculation
                    response_times.append(45)  # minutes
            
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
            
            # Get sentiment data
            sentiment_data = [analysis for analysis in self.sentiment_history.values() if analysis.crisis_id == crisis_id]
            positive_coverage = 0.0
            if sentiment_data:
                latest_sentiment = sentiment_data[-1]
                total_mentions = latest_sentiment.mention_volume
                positive_coverage = (latest_sentiment.positive_mentions / total_mentions * 100) if total_mentions > 0 else 0.0
            
            metrics = MediaManagementMetrics(
                crisis_id=crisis_id,
                total_inquiries=total_inquiries,
                response_rate=response_rate,
                average_response_time=avg_response_time,
                positive_coverage_percentage=positive_coverage,
                media_reach=sum(inq.outlet.reach for inq in crisis_inquiries),
                message_consistency_score=85.0,  # Would calculate from actual data
                spokesperson_effectiveness={'CEO': 90.0, 'Communications Director': 85.0},
                crisis_narrative_control=75.0,
                reputation_impact_score=80.0
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating media management metrics: {str(e)}")
            raise