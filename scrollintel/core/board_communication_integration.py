"""
Board Executive Mastery - Communication Systems Integration

This module provides seamless integration with all executive communication channels,
building board context awareness and implementing board-appropriate response generation.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from ..engines.executive_communication_engine import LanguageAdapter, ExecutiveCommunicationSystem
from ..engines.board_dynamics_engine import BoardDynamicsAnalysisEngine, BoardMember
from ..engines.strategic_narrative_engine import StrategicNarrativeSystem
from ..engines.information_synthesis_engine import InformationSynthesisSystem
from ..models.executive_communication_models import (
    ExecutiveAudience, Message, AdaptedMessage, CommunicationEffectiveness,
    ExecutiveLevel, CommunicationStyle, MessageType
)
from ..models.board_dynamics_models import Board

logger = logging.getLogger(__name__)


class CommunicationChannel(Enum):
    EMAIL = "email"
    BOARD_MEETING = "board_meeting"
    EXECUTIVE_BRIEFING = "executive_briefing"
    PRESENTATION = "presentation"
    REPORT = "report"
    MEMO = "memo"
    VIDEO_CONFERENCE = "video_conference"
    PHONE_CALL = "phone_call"


@dataclass
class BoardContextualMessage:
    """Message with board context awareness"""
    message_id: str
    original_content: str
    board_context: Dict[str, Any]
    board_member_profiles: List[Dict[str, Any]]
    adapted_versions: Dict[str, AdaptedMessage]
    channel: CommunicationChannel
    urgency_level: str
    board_relevance_score: float
    created_at: datetime


@dataclass
class CommunicationIntegrationConfig:
    """Configuration for communication integration"""
    board_id: str
    enabled_channels: List[CommunicationChannel]
    auto_adaptation: bool
    context_awareness_level: str
    response_generation_mode: str
    escalation_rules: Dict[str, Any]


@dataclass
class BoardResponseGeneration:
    """Board-appropriate response generation"""
    response_id: str
    original_message_id: str
    board_member_id: str
    generated_response: str
    response_tone: str
    board_appropriateness_score: float
    key_messages: List[str]
    follow_up_actions: List[str]
    created_at: datetime


class BoardCommunicationIntegration:
    """
    Integration system for board executive mastery and communication systems
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.language_adapter = LanguageAdapter()
        self.communication_engine = ExecutiveCommunicationSystem()
        self.board_dynamics = BoardDynamicsAnalysisEngine()
        self.narrative_engine = StrategicNarrativeSystem()
        self.synthesis_engine = InformationSynthesisSystem()
        
    async def create_board_contextual_communication(
        self,
        message: Message,
        board: Board,
        channel: CommunicationChannel,
        config: CommunicationIntegrationConfig
    ) -> BoardContextualMessage:
        """
        Create seamless integration with all executive communication channels
        
        Args:
            message: Original message to be adapted
            board: Board composition and dynamics
            channel: Communication channel being used
            config: Integration configuration
            
        Returns:
            Board-contextual message with adaptations
        """
        try:
            self.logger.info(f"Creating board contextual communication for {channel.value}")
            
            # Analyze board context
            board_context = await self._analyze_board_context(board, message)
            
            # Create board member profiles for communication
            member_profiles = await self._create_member_communication_profiles(board.members)
            
            # Generate adapted versions for each board member
            adapted_versions = {}
            for member in board.members:
                audience = await self._create_executive_audience(member)
                adapted_message = await self.language_adapter.adapt_communication_style(
                    message, audience
                )
                adapted_versions[member.id] = adapted_message
            
            # Calculate board relevance score
            relevance_score = await self._calculate_board_relevance(message, board)
            
            # Determine urgency level based on board context
            urgency_level = await self._determine_urgency_level(message, board_context)
            
            # Create board contextual message
            contextual_message = BoardContextualMessage(
                message_id=f"board_ctx_{message.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                original_content=message.content,
                board_context=board_context,
                board_member_profiles=member_profiles,
                adapted_versions=adapted_versions,
                channel=channel,
                urgency_level=urgency_level,
                board_relevance_score=relevance_score,
                created_at=datetime.now()
            )
            
            self.logger.info(f"Board contextual communication created with {relevance_score:.2f} relevance score")
            return contextual_message
            
        except Exception as e:
            self.logger.error(f"Error creating board contextual communication: {str(e)}")
            raise
    
    async def _analyze_board_context(self, board: Board, message: Message) -> Dict[str, Any]:
        """Analyze board context for communication adaptation"""
        
        # Analyze board composition
        composition_analysis = self.board_dynamics.analyze_board_composition(board.members)
        
        # Identify relevant board priorities
        relevant_priorities = []
        for member in board.members:
            for priority in member.priorities:
                if any(keyword.lower() in message.content.lower() 
                      for keyword in priority.area.split()):
                    relevant_priorities.append({
                        'member_id': member.id,
                        'priority': priority.area,
                        'importance': priority.importance
                    })
        
        # Assess board sentiment and dynamics
        board_sentiment = await self._assess_board_sentiment(board, message)
        
        # Identify key stakeholders for this message
        key_stakeholders = await self._identify_key_stakeholders(board, message)
        
        return {
            'composition_analysis': composition_analysis,
            'relevant_priorities': relevant_priorities,
            'board_sentiment': board_sentiment,
            'key_stakeholders': key_stakeholders,
            'total_members': len(board.members),
            'governance_structure': board.governance_structure,
            'meeting_context': self._get_meeting_context(board)
        }
    
    async def _create_member_communication_profiles(
        self, 
        members: List[BoardMember]
    ) -> List[Dict[str, Any]]:
        """Create communication profiles for board members"""
        profiles = []
        
        for member in members:
            profile = {
                'member_id': member.id,
                'name': member.name,
                'communication_style': member.communication_style.value,
                'decision_pattern': member.decision_making_pattern.value,
                'influence_level': member.influence_level.value,
                'expertise_areas': member.expertise_areas,
                'preferred_detail_level': self._determine_detail_preference(member),
                'attention_span': self._estimate_attention_span(member),
                'response_style': self._determine_response_style(member),
                'key_concerns': [p.area for p in member.priorities[:3]]
            }
            profiles.append(profile)
        
        return profiles
    
    async def _create_executive_audience(self, member: BoardMember) -> ExecutiveAudience:
        """Create executive audience profile from board member"""
        
        # Map board member to executive level
        executive_level = ExecutiveLevel.BOARD_MEMBER
        if 'chair' in member.name.lower() or 'chairman' in member.name.lower():
            executive_level = ExecutiveLevel.BOARD_CHAIR
        elif 'ceo' in [role.lower() for role in member.background.previous_roles]:
            executive_level = ExecutiveLevel.CEO
        elif 'cto' in [role.lower() for role in member.background.previous_roles]:
            executive_level = ExecutiveLevel.CTO
        elif 'cfo' in [role.lower() for role in member.background.previous_roles]:
            executive_level = ExecutiveLevel.CFO
        
        # Map communication style
        comm_style = CommunicationStyle.BALANCED
        if member.communication_style.value == 'analytical':
            comm_style = CommunicationStyle.ANALYTICAL
        elif member.communication_style.value == 'results_oriented':
            comm_style = CommunicationStyle.DIRECT
        elif member.communication_style.value == 'relationship_focused':
            comm_style = CommunicationStyle.DIPLOMATIC
        
        return ExecutiveAudience(
            id=f"audience_{member.id}",
            name=member.name,
            executive_level=executive_level,
            communication_style=comm_style,
            detail_preference=self._determine_detail_preference(member),
            attention_span=self._estimate_attention_span(member),
            decision_making_style=member.decision_making_pattern.value,
            expertise_areas=member.expertise_areas,
            priorities=[p.area for p in member.priorities],
            influence_network=[r.member_id for r in member.relationships],
            created_at=datetime.now()
        )
    
    def _determine_detail_preference(self, member: BoardMember) -> str:
        """Determine detail preference based on member profile"""
        if member.communication_style.value == 'analytical':
            return "high"
        elif member.communication_style.value == 'results_oriented':
            return "low"
        else:
            return "medium"
    
    def _estimate_attention_span(self, member: BoardMember) -> int:
        """Estimate attention span in minutes"""
        base_span = 10  # Base 10 minutes for board members
        
        if member.decision_making_pattern.value == 'quick_decider':
            return base_span - 3
        elif member.decision_making_pattern.value == 'data_driven':
            return base_span + 5
        elif member.communication_style.value == 'detail_oriented':
            return base_span + 7
        
        return base_span
    
    def _determine_response_style(self, member: BoardMember) -> str:
        """Determine preferred response style"""
        if member.communication_style.value == 'analytical':
            return "data_driven"
        elif member.communication_style.value == 'results_oriented':
            return "action_oriented"
        elif member.communication_style.value == 'relationship_focused':
            return "collaborative"
        else:
            return "balanced"
    
    async def _calculate_board_relevance(self, message: Message, board: Board) -> float:
        """Calculate relevance score for board members"""
        relevance_factors = []
        
        # Check against board member priorities
        priority_matches = 0
        total_priorities = 0
        
        for member in board.members:
            for priority in member.priorities:
                total_priorities += 1
                if any(keyword.lower() in message.content.lower() 
                      for keyword in priority.area.split()):
                    priority_matches += priority.importance
        
        priority_relevance = priority_matches / max(total_priorities, 1)
        relevance_factors.append(priority_relevance)
        
        # Check message type relevance
        type_relevance = 0.5  # Default relevance
        if message.message_type == MessageType.STRATEGIC_UPDATE:
            type_relevance = 0.9
        elif message.message_type == MessageType.FINANCIAL_REPORT:
            type_relevance = 0.8
        elif message.message_type == MessageType.RISK_ALERT:
            type_relevance = 0.95
        
        relevance_factors.append(type_relevance)
        
        # Check urgency relevance
        urgency_relevance = 0.6  # Default
        if message.urgency == "high":
            urgency_relevance = 0.9
        elif message.urgency == "critical":
            urgency_relevance = 1.0
        
        relevance_factors.append(urgency_relevance)
        
        # Calculate weighted average
        return sum(relevance_factors) / len(relevance_factors)
    
    async def _determine_urgency_level(self, message: Message, board_context: Dict[str, Any]) -> str:
        """Determine urgency level based on board context"""
        base_urgency = message.urgency
        
        # Escalate urgency if high board relevance
        if len(board_context['relevant_priorities']) > 2:
            if base_urgency == "medium":
                return "high"
            elif base_urgency == "low":
                return "medium"
        
        # Escalate if affects key stakeholders
        if len(board_context['key_stakeholders']) > 1:
            if base_urgency == "low":
                return "medium"
        
        return base_urgency
    
    async def _assess_board_sentiment(self, board: Board, message: Message) -> Dict[str, Any]:
        """Assess board sentiment regarding the message"""
        sentiment_analysis = {
            'overall_sentiment': 'neutral',
            'member_sentiments': {},
            'concern_level': 0.5,
            'support_level': 0.5
        }
        
        # Analyze sentiment based on member priorities and message content
        positive_indicators = ['success', 'achievement', 'growth', 'opportunity']
        negative_indicators = ['risk', 'challenge', 'problem', 'decline']
        
        positive_count = sum(1 for indicator in positive_indicators 
                           if indicator in message.content.lower())
        negative_count = sum(1 for indicator in negative_indicators 
                           if indicator in message.content.lower())
        
        if positive_count > negative_count:
            sentiment_analysis['overall_sentiment'] = 'positive'
            sentiment_analysis['support_level'] = 0.7
        elif negative_count > positive_count:
            sentiment_analysis['overall_sentiment'] = 'negative'
            sentiment_analysis['concern_level'] = 0.8
        
        return sentiment_analysis
    
    async def _identify_key_stakeholders(self, board: Board, message: Message) -> List[str]:
        """Identify key stakeholders for the message"""
        key_stakeholders = []
        
        for member in board.members:
            # Check if member's expertise aligns with message content
            expertise_match = any(
                expertise.lower() in message.content.lower() 
                for expertise in member.expertise_areas
            )
            
            # Check if member's priorities align with message
            priority_match = any(
                priority.area.lower() in message.content.lower()
                for priority in member.priorities
            )
            
            if expertise_match or priority_match:
                key_stakeholders.append(member.id)
        
        return key_stakeholders
    
    def _get_meeting_context(self, board: Board) -> Dict[str, Any]:
        """Get current meeting context"""
        return {
            'next_meeting': board.meeting_schedule[0] if board.meeting_schedule else None,
            'committees': board.committees,
            'governance_type': board.governance_structure.get('type', 'traditional')
        }
    
    async def generate_board_appropriate_response(
        self,
        original_message: Message,
        board_member: BoardMember,
        response_context: Dict[str, Any]
    ) -> BoardResponseGeneration:
        """
        Implement board-appropriate response generation and messaging
        
        Args:
            original_message: Message being responded to
            board_member: Board member generating response
            response_context: Context for response generation
            
        Returns:
            Board-appropriate response
        """
        try:
            self.logger.info(f"Generating board-appropriate response for {board_member.name}")
            
            # Create executive audience for the board member
            audience = await self._create_executive_audience(board_member)
            
            # Generate response content based on member profile
            response_content = await self._generate_response_content(
                original_message, board_member, response_context
            )
            
            # Adapt response tone for board appropriateness
            response_tone = await self._determine_response_tone(board_member, original_message)
            
            # Extract key messages for the response
            key_messages = await self._extract_response_key_messages(
                response_content, board_member
            )
            
            # Generate follow-up actions
            follow_up_actions = await self._generate_follow_up_actions(
                original_message, board_member, response_context
            )
            
            # Calculate board appropriateness score
            appropriateness_score = await self._calculate_appropriateness_score(
                response_content, board_member, original_message
            )
            
            # Create response generation record
            response_generation = BoardResponseGeneration(
                response_id=f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                original_message_id=original_message.id,
                board_member_id=board_member.id,
                generated_response=response_content,
                response_tone=response_tone,
                board_appropriateness_score=appropriateness_score,
                key_messages=key_messages,
                follow_up_actions=follow_up_actions,
                created_at=datetime.now()
            )
            
            self.logger.info(f"Board-appropriate response generated with {appropriateness_score:.2f} appropriateness score")
            return response_generation
            
        except Exception as e:
            self.logger.error(f"Error generating board-appropriate response: {str(e)}")
            raise
    
    async def _generate_response_content(
        self,
        original_message: Message,
        board_member: BoardMember,
        context: Dict[str, Any]
    ) -> str:
        """Generate response content based on board member profile"""
        
        # Base response structure
        response_parts = []
        
        # Opening based on communication style
        if board_member.communication_style.value == 'relationship_focused':
            response_parts.append("Thank you for bringing this to our attention.")
        elif board_member.communication_style.value == 'results_oriented':
            response_parts.append("I've reviewed the information and have the following perspective:")
        elif board_member.communication_style.value == 'analytical':
            response_parts.append("Based on my analysis of the presented information:")
        else:
            response_parts.append("I appreciate the update and would like to share my thoughts:")
        
        # Main content based on member priorities
        main_content = []
        for priority in board_member.priorities[:2]:  # Top 2 priorities
            if any(keyword.lower() in original_message.content.lower() 
                  for keyword in priority.area.split()):
                main_content.append(
                    f"From a {priority.area.lower()} perspective, {priority.description.lower()}."
                )
        
        if main_content:
            response_parts.extend(main_content)
        else:
            response_parts.append("This aligns with our strategic objectives and warrants careful consideration.")
        
        # Closing based on decision pattern
        if board_member.decision_making_pattern.value == 'quick_decider':
            response_parts.append("I recommend we move forward with clear next steps.")
        elif board_member.decision_making_pattern.value == 'data_driven':
            response_parts.append("I'd like to see additional data before making a final decision.")
        elif board_member.decision_making_pattern.value == 'collaborative':
            response_parts.append("I believe we should discuss this further as a board.")
        else:
            response_parts.append("I look forward to our continued discussion on this matter.")
        
        return " ".join(response_parts)
    
    async def _determine_response_tone(self, board_member: BoardMember, original_message: Message) -> str:
        """Determine appropriate response tone"""
        
        # Base tone on communication style
        if board_member.communication_style.value == 'analytical':
            return "professional_analytical"
        elif board_member.communication_style.value == 'relationship_focused':
            return "warm_collaborative"
        elif board_member.communication_style.value == 'results_oriented':
            return "direct_decisive"
        elif board_member.communication_style.value == 'visionary':
            return "inspiring_strategic"
        else:
            return "balanced_professional"
    
    async def _extract_response_key_messages(
        self, 
        response_content: str, 
        board_member: BoardMember
    ) -> List[str]:
        """Extract key messages from response"""
        
        # Simple extraction based on sentences
        sentences = response_content.split('.')
        key_messages = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Meaningful sentences
                key_messages.append(sentence)
        
        # Limit to top 3 key messages
        return key_messages[:3]
    
    async def _generate_follow_up_actions(
        self,
        original_message: Message,
        board_member: BoardMember,
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate follow-up actions based on response"""
        
        actions = []
        
        # Actions based on decision pattern
        if board_member.decision_making_pattern.value == 'data_driven':
            actions.append("Request additional data and analysis")
            actions.append("Schedule follow-up review meeting")
        elif board_member.decision_making_pattern.value == 'collaborative':
            actions.append("Schedule board discussion session")
            actions.append("Gather input from other board members")
        elif board_member.decision_making_pattern.value == 'quick_decider':
            actions.append("Prepare decision recommendation")
            actions.append("Identify implementation timeline")
        
        # Actions based on message urgency
        if original_message.urgency == "high":
            actions.append("Expedite review process")
        elif original_message.urgency == "critical":
            actions.append("Schedule emergency board session")
        
        return actions[:3]  # Limit to 3 actions
    
    async def _calculate_appropriateness_score(
        self,
        response_content: str,
        board_member: BoardMember,
        original_message: Message
    ) -> float:
        """Calculate board appropriateness score for response"""
        
        score_factors = []
        
        # Professional language score
        professional_terms = ['strategic', 'analysis', 'recommend', 'consider', 'perspective']
        professional_count = sum(1 for term in professional_terms 
                                if term in response_content.lower())
        professional_score = min(professional_count / 3, 1.0)
        score_factors.append(professional_score)
        
        # Length appropriateness (not too long, not too short)
        word_count = len(response_content.split())
        if 50 <= word_count <= 150:  # Appropriate length for board communication
            length_score = 1.0
        elif word_count < 50:
            length_score = word_count / 50
        else:
            length_score = max(0.5, 150 / word_count)
        score_factors.append(length_score)
        
        # Tone consistency score
        tone_score = 0.8  # Default good tone score
        score_factors.append(tone_score)
        
        # Calculate weighted average
        return sum(score_factors) / len(score_factors)
    
    async def build_board_context_awareness(
        self,
        communication_history: List[Message],
        board: Board,
        time_window_days: int = 30
    ) -> Dict[str, Any]:
        """
        Build board context awareness in all communications
        
        Args:
            communication_history: Recent communication history
            board: Board composition and dynamics
            time_window_days: Time window for context analysis
            
        Returns:
            Board context awareness data
        """
        try:
            self.logger.info(f"Building board context awareness for {time_window_days} days")
            
            # Analyze communication patterns
            communication_patterns = await self._analyze_communication_patterns(
                communication_history, board
            )
            
            # Identify recurring themes
            recurring_themes = await self._identify_recurring_themes(communication_history)
            
            # Track board member engagement
            member_engagement = await self._track_member_engagement(
                communication_history, board
            )
            
            # Assess communication effectiveness
            effectiveness_metrics = await self._assess_communication_effectiveness(
                communication_history, board
            )
            
            # Generate context insights
            context_insights = await self._generate_context_insights(
                communication_patterns, recurring_themes, member_engagement
            )
            
            context_awareness = {
                'board_id': board.id,
                'analysis_period': f"{time_window_days} days",
                'communication_patterns': communication_patterns,
                'recurring_themes': recurring_themes,
                'member_engagement': member_engagement,
                'effectiveness_metrics': effectiveness_metrics,
                'context_insights': context_insights,
                'last_updated': datetime.now().isoformat()
            }
            
            self.logger.info("Board context awareness built successfully")
            return context_awareness
            
        except Exception as e:
            self.logger.error(f"Error building board context awareness: {str(e)}")
            raise
    
    async def _analyze_communication_patterns(
        self, 
        history: List[Message], 
        board: Board
    ) -> Dict[str, Any]:
        """Analyze communication patterns"""
        
        patterns = {
            'message_frequency': len(history),
            'channel_distribution': {},
            'urgency_distribution': {},
            'response_times': [],
            'member_participation': {}
        }
        
        # Analyze channel usage
        for message in history:
            channel = getattr(message, 'channel', 'unknown')
            patterns['channel_distribution'][channel] = patterns['channel_distribution'].get(channel, 0) + 1
        
        # Analyze urgency levels
        for message in history:
            urgency = getattr(message, 'urgency', 'medium')
            patterns['urgency_distribution'][urgency] = patterns['urgency_distribution'].get(urgency, 0) + 1
        
        # Analyze member participation
        for member in board.members:
            patterns['member_participation'][member.id] = {
                'messages_sent': 0,
                'messages_received': 0,
                'response_rate': 0.0
            }
        
        return patterns
    
    async def _identify_recurring_themes(self, history: List[Message]) -> List[Dict[str, Any]]:
        """Identify recurring themes in communications"""
        
        themes = {}
        common_keywords = ['strategy', 'financial', 'risk', 'growth', 'innovation', 'market']
        
        for message in history:
            for keyword in common_keywords:
                if keyword in message.content.lower():
                    if keyword not in themes:
                        themes[keyword] = {'count': 0, 'messages': []}
                    themes[keyword]['count'] += 1
                    themes[keyword]['messages'].append(message.id)
        
        # Convert to sorted list
        recurring_themes = []
        for theme, data in sorted(themes.items(), key=lambda x: x[1]['count'], reverse=True):
            recurring_themes.append({
                'theme': theme,
                'frequency': data['count'],
                'message_ids': data['messages'][:5]  # Top 5 messages
            })
        
        return recurring_themes[:10]  # Top 10 themes
    
    async def _track_member_engagement(
        self, 
        history: List[Message], 
        board: Board
    ) -> Dict[str, Any]:
        """Track board member engagement in communications"""
        
        engagement = {}
        
        for member in board.members:
            engagement[member.id] = {
                'name': member.name,
                'messages_initiated': 0,
                'responses_provided': 0,
                'avg_response_time': 0,
                'engagement_score': 0.5
            }
        
        # Calculate engagement metrics
        for member_id in engagement:
            # Simple engagement score calculation
            initiated = engagement[member_id]['messages_initiated']
            responded = engagement[member_id]['responses_provided']
            total_messages = len(history)
            
            if total_messages > 0:
                engagement_score = (initiated + responded) / (total_messages * 2)
                engagement[member_id]['engagement_score'] = min(engagement_score, 1.0)
        
        return engagement
    
    async def _assess_communication_effectiveness(
        self, 
        history: List[Message], 
        board: Board
    ) -> Dict[str, Any]:
        """Assess communication effectiveness"""
        
        effectiveness = {
            'overall_score': 0.7,  # Default score
            'response_rate': 0.0,
            'clarity_score': 0.0,
            'engagement_level': 0.0,
            'action_completion_rate': 0.0
        }
        
        if history:
            # Calculate response rate
            messages_with_responses = sum(1 for msg in history if hasattr(msg, 'responses') and msg.responses)
            effectiveness['response_rate'] = messages_with_responses / len(history)
            
            # Estimate other metrics (in real implementation, these would be calculated from actual data)
            effectiveness['clarity_score'] = 0.75
            effectiveness['engagement_level'] = 0.65
            effectiveness['action_completion_rate'] = 0.70
            
            # Calculate overall score
            effectiveness['overall_score'] = (
                effectiveness['response_rate'] * 0.3 +
                effectiveness['clarity_score'] * 0.25 +
                effectiveness['engagement_level'] * 0.25 +
                effectiveness['action_completion_rate'] * 0.20
            )
        
        return effectiveness
    
    async def _generate_context_insights(
        self,
        patterns: Dict[str, Any],
        themes: List[Dict[str, Any]],
        engagement: Dict[str, Any]
    ) -> List[str]:
        """Generate context insights from analysis"""
        
        insights = []
        
        # Communication frequency insights
        if patterns['message_frequency'] > 50:
            insights.append("High communication volume may indicate active board engagement")
        elif patterns['message_frequency'] < 10:
            insights.append("Low communication volume may suggest need for increased engagement")
        
        # Theme insights
        if themes:
            top_theme = themes[0]['theme']
            insights.append(f"'{top_theme}' is the most discussed topic in recent communications")
        
        # Engagement insights
        avg_engagement = sum(member['engagement_score'] for member in engagement.values()) / len(engagement)
        if avg_engagement > 0.7:
            insights.append("Board members show high engagement in communications")
        elif avg_engagement < 0.4:
            insights.append("Board member engagement could be improved")
        
        return insights