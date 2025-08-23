"""
Media Influence Engine for Global Influence Network System

This engine provides media relationship management, automated story pitching,
and reputation management capabilities.
"""

import logging
import asyncio
from typing import List, Dict, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json

from ..models.influence_network_models import (
    MediaOutlet, MediaRelationship, ReputationMonitor, CrisisResponse,
    MediaOutletType
)


class PitchStatus(Enum):
    """Status of media pitches"""
    DRAFT = "draft"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    PUBLISHED = "published"


class SentimentType(Enum):
    """Types of sentiment"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"


@dataclass
class MediaPitch:
    """Represents a media pitch"""
    pitch_id: str
    outlet_id: str
    journalist_email: str
    subject_line: str
    pitch_content: str
    story_angle: str
    supporting_materials: List[str]
    follow_up_schedule: List[datetime]
    status: PitchStatus
    response_received: Optional[str]
    created_at: datetime
    sent_at: Optional[datetime]
    response_at: Optional[datetime]


@dataclass
class MediaCoverage:
    """Represents media coverage received"""
    coverage_id: str
    outlet_id: str
    headline: str
    url: str
    publication_date: datetime
    journalist: str
    coverage_type: str  # 'news', 'feature', 'opinion', 'interview'
    sentiment: SentimentType
    reach_estimate: int
    key_messages_included: List[str]
    share_count: int
    engagement_metrics: Dict[str, int]


@dataclass
class ReputationAlert:
    """Represents a reputation monitoring alert"""
    alert_id: str
    entity_name: str
    alert_type: str  # 'negative_mention', 'crisis_indicator', 'opportunity'
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    source_url: str
    sentiment_score: float
    reach_estimate: int
    recommended_response: str
    created_at: datetime
    acknowledged: bool


class MediaInfluenceEngine:
    """Engine for managing media relationships and influence"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.media_outlets = {}
        self.media_relationships = {}
        self.active_pitches = {}
        self.coverage_tracking = {}
        self.reputation_monitors = {}
        self.crisis_responses = {}
        
        # Configuration
        self.config = {
            'pitch_follow_up_days': [3, 7, 14],
            'sentiment_alert_threshold': -0.3,
            'crisis_mention_threshold': 10,
            'response_time_target_hours': 2
        }
    
    async def build_media_database(
        self,
        industry_focus: List[str],
        geographic_regions: List[str]
    ) -> Dict[str, Any]:
        """Build comprehensive media outlet and journalist database"""
        try:
            database_result = {
                'total_outlets': 0,
                'outlets_by_type': {},
                'journalists_added': 0,
                'coverage_areas': industry_focus,
                'geographic_reach': geographic_regions
            }
            
            # Simulate building media database (would integrate with real data sources)
            outlet_types = [
                MediaOutletType.TRADITIONAL_MEDIA,
                MediaOutletType.DIGITAL_MEDIA,
                MediaOutletType.INDUSTRY_PUBLICATION,
                MediaOutletType.PODCAST,
                MediaOutletType.BLOG
            ]
            
            for outlet_type in outlet_types:
                outlets = await self._discover_outlets_by_type(
                    outlet_type, industry_focus, geographic_regions
                )
                
                for outlet_data in outlets:
                    outlet = MediaOutlet(
                        outlet_id=outlet_data['id'],
                        name=outlet_data['name'],
                        outlet_type=outlet_type,
                        industry_focus=outlet_data.get('industry_focus', industry_focus),
                        geographic_reach=outlet_data.get('geographic_reach', geographic_regions),
                        audience_size=outlet_data.get('audience_size', 10000),
                        influence_score=outlet_data.get('influence_score', 0.5),
                        editorial_stance=outlet_data.get('editorial_stance', 'neutral'),
                        key_journalists=outlet_data.get('journalists', []),
                        contact_information=outlet_data.get('contact_info', {}),
                        pitch_preferences=outlet_data.get('pitch_preferences', {}),
                        response_rate=outlet_data.get('response_rate', 0.2),
                        last_interaction=None
                    )
                    
                    self.media_outlets[outlet.outlet_id] = outlet
                    database_result['total_outlets'] += 1
                    database_result['journalists_added'] += len(outlet.key_journalists)
                
                database_result['outlets_by_type'][outlet_type.value] = len(outlets)
            
            self.logger.info(f"Built media database with {database_result['total_outlets']} outlets")
            return database_result
            
        except Exception as e:
            self.logger.error(f"Error building media database: {str(e)}")
            raise
    
    async def create_automated_pitch_system(
        self,
        story_angles: List[Dict[str, Any]],
        target_outlets: List[str],
        personalization_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create automated system for story pitching"""
        try:
            pitch_system_result = {
                'total_pitches_created': 0,
                'pitches_by_outlet': {},
                'personalization_applied': 0,
                'follow_up_scheduled': 0,
                'estimated_response_rate': 0.0
            }
            
            for story_angle in story_angles:
                for outlet_id in target_outlets:
                    if outlet_id not in self.media_outlets:
                        continue
                    
                    outlet = self.media_outlets[outlet_id]
                    
                    # Create personalized pitch for each journalist
                    for journalist in outlet.key_journalists:
                        pitch = await self._create_personalized_pitch(
                            story_angle, outlet, journalist, personalization_data
                        )
                        
                        self.active_pitches[pitch.pitch_id] = pitch
                        pitch_system_result['total_pitches_created'] += 1
                        
                        if outlet_id not in pitch_system_result['pitches_by_outlet']:
                            pitch_system_result['pitches_by_outlet'][outlet_id] = 0
                        pitch_system_result['pitches_by_outlet'][outlet_id] += 1
                        
                        # Schedule follow-ups
                        await self._schedule_pitch_follow_ups(pitch)
                        pitch_system_result['follow_up_scheduled'] += len(pitch.follow_up_schedule)
            
            # Calculate estimated response rate
            total_outlets = len(target_outlets)
            avg_response_rate = sum(
                self.media_outlets[oid].response_rate 
                for oid in target_outlets 
                if oid in self.media_outlets
            ) / max(total_outlets, 1)
            
            pitch_system_result['estimated_response_rate'] = avg_response_rate
            
            self.logger.info(f"Created automated pitch system with {pitch_system_result['total_pitches_created']} pitches")
            return pitch_system_result
            
        except Exception as e:
            self.logger.error(f"Error creating automated pitch system: {str(e)}")
            raise
    
    async def track_media_coverage(
        self,
        monitoring_keywords: List[str],
        date_range: Dict[str, datetime]
    ) -> Dict[str, Any]:
        """Track and analyze media coverage"""
        try:
            coverage_result = {
                'total_mentions': 0,
                'coverage_by_outlet_type': {},
                'sentiment_distribution': {},
                'reach_analysis': {},
                'key_coverage_pieces': [],
                'trending_topics': []
            }
            
            # Simulate coverage tracking (would integrate with media monitoring APIs)
            coverage_pieces = await self._discover_media_coverage(
                monitoring_keywords, date_range
            )
            
            for coverage_data in coverage_pieces:
                coverage = MediaCoverage(
                    coverage_id=coverage_data['id'],
                    outlet_id=coverage_data['outlet_id'],
                    headline=coverage_data['headline'],
                    url=coverage_data['url'],
                    publication_date=coverage_data['publication_date'],
                    journalist=coverage_data['journalist'],
                    coverage_type=coverage_data['type'],
                    sentiment=SentimentType(coverage_data['sentiment']),
                    reach_estimate=coverage_data['reach'],
                    key_messages_included=coverage_data.get('key_messages', []),
                    share_count=coverage_data.get('shares', 0),
                    engagement_metrics=coverage_data.get('engagement', {})
                )
                
                self.coverage_tracking[coverage.coverage_id] = coverage
                coverage_result['total_mentions'] += 1
                
                # Analyze by outlet type
                if coverage.outlet_id in self.media_outlets:
                    outlet_type = self.media_outlets[coverage.outlet_id].outlet_type.value
                    if outlet_type not in coverage_result['coverage_by_outlet_type']:
                        coverage_result['coverage_by_outlet_type'][outlet_type] = 0
                    coverage_result['coverage_by_outlet_type'][outlet_type] += 1
                
                # Analyze sentiment
                sentiment_key = coverage.sentiment.value
                if sentiment_key not in coverage_result['sentiment_distribution']:
                    coverage_result['sentiment_distribution'][sentiment_key] = 0
                coverage_result['sentiment_distribution'][sentiment_key] += 1
                
                # Track high-impact coverage
                if coverage.reach_estimate > 50000 or coverage.share_count > 100:
                    coverage_result['key_coverage_pieces'].append({
                        'headline': coverage.headline,
                        'outlet': coverage.outlet_id,
                        'reach': coverage.reach_estimate,
                        'sentiment': coverage.sentiment.value,
                        'url': coverage.url
                    })
            
            # Calculate reach analysis
            total_reach = sum(c.reach_estimate for c in self.coverage_tracking.values())
            avg_reach = total_reach / max(len(self.coverage_tracking), 1)
            
            coverage_result['reach_analysis'] = {
                'total_reach': total_reach,
                'average_reach': avg_reach,
                'high_impact_pieces': len(coverage_result['key_coverage_pieces'])
            }
            
            self.logger.info(f"Tracked {coverage_result['total_mentions']} media mentions")
            return coverage_result
            
        except Exception as e:
            self.logger.error(f"Error tracking media coverage: {str(e)}")
            raise    
    a
sync def implement_reputation_management(
        self,
        entities_to_monitor: List[str],
        monitoring_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Implement comprehensive reputation management system"""
        try:
            reputation_result = {
                'entities_monitored': len(entities_to_monitor),
                'monitoring_sources': monitoring_config.get('sources', []),
                'alerts_configured': 0,
                'baseline_sentiment': {},
                'crisis_thresholds_set': True
            }
            
            for entity in entities_to_monitor:
                # Create reputation monitor
                monitor = ReputationMonitor(
                    monitor_id=f"monitor_{entity}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    entity_name=entity,
                    monitoring_keywords=monitoring_config.get('keywords', [entity]),
                    sentiment_score=0.0,  # Will be updated with real data
                    mention_volume=0,
                    reach_estimate=0,
                    top_mentions=[],
                    sentiment_trend=[],
                    crisis_indicators=[],
                    response_recommendations=[],
                    monitoring_date=datetime.now(),
                    data_sources=monitoring_config.get('sources', ['social_media', 'news', 'blogs'])
                )
                
                # Get baseline sentiment
                baseline = await self._calculate_baseline_sentiment(entity, monitoring_config)
                monitor.sentiment_score = baseline['sentiment_score']
                monitor.mention_volume = baseline['mention_volume']
                monitor.reach_estimate = baseline['reach_estimate']
                
                self.reputation_monitors[monitor.monitor_id] = monitor
                reputation_result['baseline_sentiment'][entity] = baseline
                
                # Configure alerts
                alerts_configured = await self._configure_reputation_alerts(entity, monitor)
                reputation_result['alerts_configured'] += alerts_configured
            
            self.logger.info(f"Implemented reputation management for {len(entities_to_monitor)} entities")
            return reputation_result
            
        except Exception as e:
            self.logger.error(f"Error implementing reputation management: {str(e)}")
            raise
    
    async def create_crisis_response_system(
        self,
        crisis_scenarios: List[Dict[str, Any]],
        response_team: List[str],
        escalation_rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create automated crisis response system"""
        try:
            crisis_system_result = {
                'scenarios_prepared': len(crisis_scenarios),
                'response_templates_created': 0,
                'escalation_rules_configured': len(escalation_rules),
                'team_members_assigned': len(response_team),
                'response_time_target': self.config['response_time_target_hours']
            }
            
            for scenario in crisis_scenarios:
                # Create crisis response plan
                response = CrisisResponse(
                    response_id=f"crisis_response_{scenario['type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    crisis_type=scenario['type'],
                    severity_level=scenario.get('severity', 'medium'),
                    affected_entities=scenario.get('affected_entities', []),
                    response_strategy=scenario.get('strategy', 'transparent_communication'),
                    key_messages=scenario.get('key_messages', []),
                    communication_channels=scenario.get('channels', ['social_media', 'press_release']),
                    stakeholder_communications=[],
                    media_responses=[],
                    timeline={
                        'detection': datetime.now(),
                        'initial_response': datetime.now() + timedelta(hours=1),
                        'full_response': datetime.now() + timedelta(hours=4),
                        'follow_up': datetime.now() + timedelta(days=1)
                    },
                    team_assignments={member: 'responder' for member in response_team},
                    success_criteria=scenario.get('success_criteria', []),
                    created_at=datetime.now(),
                    activated_at=None,
                    resolved_at=None
                )
                
                self.crisis_responses[response.response_id] = response
                crisis_system_result['response_templates_created'] += 1
            
            self.logger.info(f"Created crisis response system with {len(crisis_scenarios)} scenarios")
            return crisis_system_result
            
        except Exception as e:
            self.logger.error(f"Error creating crisis response system: {str(e)}")
            raise
    
    async def execute_narrative_correction(
        self,
        incorrect_narrative: str,
        correct_information: str,
        target_outlets: List[str],
        urgency_level: str = "medium"
    ) -> Dict[str, Any]:
        """Execute narrative correction campaign"""
        try:
            correction_result = {
                'correction_campaign_id': f"correction_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'target_outlets': len(target_outlets),
                'correction_messages_sent': 0,
                'follow_up_scheduled': 0,
                'estimated_reach': 0,
                'urgency_level': urgency_level
            }
            
            # Create correction message
            correction_message = await self._create_correction_message(
                incorrect_narrative, correct_information, urgency_level
            )
            
            # Send to target outlets
            for outlet_id in target_outlets:
                if outlet_id not in self.media_outlets:
                    continue
                
                outlet = self.media_outlets[outlet_id]
                
                # Create personalized correction pitch
                correction_pitch = await self._create_correction_pitch(
                    correction_message, outlet, urgency_level
                )
                
                # Send correction
                await self._send_correction_pitch(correction_pitch, outlet)
                correction_result['correction_messages_sent'] += 1
                correction_result['estimated_reach'] += outlet.audience_size
                
                # Schedule follow-up if high urgency
                if urgency_level == "high":
                    await self._schedule_urgent_follow_up(correction_pitch, outlet)
                    correction_result['follow_up_scheduled'] += 1
            
            self.logger.info(f"Executed narrative correction campaign reaching {correction_result['estimated_reach']} audience")
            return correction_result
            
        except Exception as e:
            self.logger.error(f"Error executing narrative correction: {str(e)}")
            raise
    
    # Helper methods
    
    async def _discover_outlets_by_type(
        self,
        outlet_type: MediaOutletType,
        industry_focus: List[str],
        regions: List[str]
    ) -> List[Dict[str, Any]]:
        """Discover media outlets by type (simulated)"""
        # This would integrate with real media databases
        outlets = []
        
        base_outlets = {
            MediaOutletType.TRADITIONAL_MEDIA: [
                {'name': 'Tech Times', 'audience_size': 500000, 'influence_score': 0.8},
                {'name': 'Business Weekly', 'audience_size': 300000, 'influence_score': 0.7},
                {'name': 'Innovation Daily', 'audience_size': 200000, 'influence_score': 0.6}
            ],
            MediaOutletType.DIGITAL_MEDIA: [
                {'name': 'TechCrunch', 'audience_size': 1000000, 'influence_score': 0.9},
                {'name': 'VentureBeat', 'audience_size': 800000, 'influence_score': 0.8},
                {'name': 'The Verge', 'audience_size': 1200000, 'influence_score': 0.85}
            ],
            MediaOutletType.INDUSTRY_PUBLICATION: [
                {'name': 'AI Research Journal', 'audience_size': 50000, 'influence_score': 0.9},
                {'name': 'Enterprise Tech Review', 'audience_size': 75000, 'influence_score': 0.7}
            ],
            MediaOutletType.PODCAST: [
                {'name': 'Tech Talk Podcast', 'audience_size': 100000, 'influence_score': 0.6},
                {'name': 'Innovation Insights', 'audience_size': 80000, 'influence_score': 0.5}
            ],
            MediaOutletType.BLOG: [
                {'name': 'Tech Blogger Network', 'audience_size': 25000, 'influence_score': 0.4},
                {'name': 'Startup Stories', 'audience_size': 30000, 'influence_score': 0.5}
            ]
        }
        
        base_list = base_outlets.get(outlet_type, [])
        
        for i, outlet_data in enumerate(base_list):
            outlet_info = {
                'id': f"{outlet_type.value}_{i}",
                'name': outlet_data['name'],
                'audience_size': outlet_data['audience_size'],
                'influence_score': outlet_data['influence_score'],
                'industry_focus': industry_focus,
                'geographic_reach': regions,
                'journalists': [f"journalist_{i}_1@{outlet_data['name'].lower().replace(' ', '')}.com"],
                'contact_info': {'email': f"news@{outlet_data['name'].lower().replace(' ', '')}.com"},
                'pitch_preferences': {'format': 'email', 'length': 'brief'},
                'response_rate': 0.15 + (outlet_data['influence_score'] * 0.2)
            }
            outlets.append(outlet_info)
        
        return outlets
    
    async def _create_personalized_pitch(
        self,
        story_angle: Dict[str, Any],
        outlet: MediaOutlet,
        journalist: str,
        personalization_data: Dict[str, Any]
    ) -> MediaPitch:
        """Create personalized pitch for journalist"""
        pitch_id = f"pitch_{outlet.outlet_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Personalize subject line
        subject_line = f"{story_angle['headline']} - Exclusive for {outlet.name}"
        
        # Create personalized content
        pitch_content = f"""
        Dear {journalist.split('@')[0].title()},
        
        I hope this email finds you well. I'm reaching out with an exclusive story opportunity 
        that aligns perfectly with {outlet.name}'s coverage of {', '.join(outlet.industry_focus)}.
        
        {story_angle['description']}
        
        Key angles for your audience:
        {chr(10).join(f"• {angle}" for angle in story_angle.get('key_angles', []))}
        
        I have exclusive access to:
        {chr(10).join(f"• {resource}" for resource in story_angle.get('resources', []))}
        
        This story would be perfect for {outlet.name} because:
        {chr(10).join(f"• {reason}" for reason in story_angle.get('outlet_fit', []))}
        
        I'm happy to provide additional information, arrange interviews, or customize 
        the angle to best fit your editorial calendar.
        
        Best regards,
        [Your Name]
        """
        
        pitch = MediaPitch(
            pitch_id=pitch_id,
            outlet_id=outlet.outlet_id,
            journalist_email=journalist,
            subject_line=subject_line,
            pitch_content=pitch_content,
            story_angle=story_angle['headline'],
            supporting_materials=story_angle.get('materials', []),
            follow_up_schedule=[],
            status=PitchStatus.DRAFT,
            response_received=None,
            created_at=datetime.now(),
            sent_at=None,
            response_at=None
        )
        
        return pitch
    
    async def _schedule_pitch_follow_ups(self, pitch: MediaPitch):
        """Schedule follow-up communications for pitch"""
        base_time = datetime.now()
        
        for days in self.config['pitch_follow_up_days']:
            follow_up_time = base_time + timedelta(days=days)
            pitch.follow_up_schedule.append(follow_up_time)
    
    async def _discover_media_coverage(
        self,
        keywords: List[str],
        date_range: Dict[str, datetime]
    ) -> List[Dict[str, Any]]:
        """Discover media coverage (simulated)"""
        # This would integrate with media monitoring APIs
        coverage_pieces = []
        
        # Simulate some coverage
        sample_coverage = [
            {
                'id': 'coverage_1',
                'outlet_id': 'digital_media_0',
                'headline': 'Tech Innovation Breakthrough Announced',
                'url': 'https://techcrunch.com/sample-article',
                'publication_date': datetime.now() - timedelta(days=1),
                'journalist': 'Tech Reporter',
                'type': 'news',
                'sentiment': 'positive',
                'reach': 100000,
                'shares': 250,
                'engagement': {'likes': 500, 'comments': 50}
            },
            {
                'id': 'coverage_2',
                'outlet_id': 'traditional_media_0',
                'headline': 'Industry Analysis: Market Trends',
                'url': 'https://techtimes.com/analysis',
                'publication_date': datetime.now() - timedelta(days=2),
                'journalist': 'Senior Editor',
                'type': 'feature',
                'sentiment': 'neutral',
                'reach': 75000,
                'shares': 100,
                'engagement': {'likes': 200, 'comments': 25}
            }
        ]
        
        return sample_coverage
    
    async def _calculate_baseline_sentiment(
        self,
        entity: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate baseline sentiment for entity"""
        # Simulate baseline calculation
        return {
            'sentiment_score': 0.2,  # Slightly positive
            'mention_volume': 50,
            'reach_estimate': 500000,
            'confidence': 0.8
        }
    
    async def _configure_reputation_alerts(
        self,
        entity: str,
        monitor: ReputationMonitor
    ) -> int:
        """Configure reputation alerts for entity"""
        # Configure different types of alerts
        alert_types = [
            'negative_sentiment_spike',
            'mention_volume_increase',
            'crisis_keywords_detected',
            'competitor_comparison'
        ]
        
        return len(alert_types)
    
    async def _create_correction_message(
        self,
        incorrect_narrative: str,
        correct_information: str,
        urgency: str
    ) -> Dict[str, Any]:
        """Create correction message"""
        return {
            'subject': f"{'URGENT: ' if urgency == 'high' else ''}Correction Needed - {incorrect_narrative[:50]}...",
            'body': f"""
            We need to address an inaccuracy in recent coverage:
            
            Incorrect Information: {incorrect_narrative}
            
            Correct Information: {correct_information}
            
            We have supporting documentation and are available for immediate clarification.
            """,
            'urgency': urgency,
            'supporting_docs': ['fact_sheet.pdf', 'expert_statement.pdf']
        }
    
    async def _create_correction_pitch(
        self,
        correction_message: Dict[str, Any],
        outlet: MediaOutlet,
        urgency: str
    ) -> MediaPitch:
        """Create correction pitch for outlet"""
        pitch_id = f"correction_{outlet.outlet_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        pitch = MediaPitch(
            pitch_id=pitch_id,
            outlet_id=outlet.outlet_id,
            journalist_email=outlet.key_journalists[0] if outlet.key_journalists else "news@outlet.com",
            subject_line=correction_message['subject'],
            pitch_content=correction_message['body'],
            story_angle="correction",
            supporting_materials=correction_message['supporting_docs'],
            follow_up_schedule=[],
            status=PitchStatus.DRAFT,
            response_received=None,
            created_at=datetime.now(),
            sent_at=None,
            response_at=None
        )
        
        return pitch
    
    async def _send_correction_pitch(self, pitch: MediaPitch, outlet: MediaOutlet):
        """Send correction pitch to outlet"""
        # Simulate sending pitch
        pitch.status = PitchStatus.SENT
        pitch.sent_at = datetime.now()
        self.active_pitches[pitch.pitch_id] = pitch
    
    async def _schedule_urgent_follow_up(self, pitch: MediaPitch, outlet: MediaOutlet):
        """Schedule urgent follow-up for correction"""
        # Schedule follow-up in 2 hours for urgent corrections
        follow_up_time = datetime.now() + timedelta(hours=2)
        pitch.follow_up_schedule.append(follow_up_time)
    
    def get_media_relationship_status(self, outlet_id: str) -> Optional[Dict[str, Any]]:
        """Get status of relationship with media outlet"""
        if outlet_id not in self.media_outlets:
            return None
        
        outlet = self.media_outlets[outlet_id]
        
        # Find related pitches
        outlet_pitches = [
            pitch for pitch in self.active_pitches.values()
            if pitch.outlet_id == outlet_id
        ]
        
        return {
            'outlet_id': outlet_id,
            'outlet_name': outlet.name,
            'outlet_type': outlet.outlet_type.value,
            'influence_score': outlet.influence_score,
            'response_rate': outlet.response_rate,
            'total_pitches': len(outlet_pitches),
            'successful_pitches': len([p for p in outlet_pitches if p.status == PitchStatus.ACCEPTED]),
            'last_interaction': outlet.last_interaction.isoformat() if outlet.last_interaction else None
        }