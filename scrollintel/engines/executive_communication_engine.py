"""
Executive Communication Engine for Board Executive Mastery System
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import re
import json

from ..models.executive_communication_models import (
    ExecutiveAudience, Message, AdaptedMessage, CommunicationEffectiveness,
    LanguageAdaptationRule, ExecutiveLevel, CommunicationStyle, MessageType
)


class LanguageAdapter:
    """Adapts language and communication style for executive audiences"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.adaptation_rules = self._load_adaptation_rules()
        self.executive_vocabulary = self._load_executive_vocabulary()
        
    def adapt_communication_style(self, message: Message, audience: ExecutiveAudience) -> AdaptedMessage:
        """Optimize communication style for C-level and board interactions"""
        try:
            # Analyze message complexity and audience preferences
            complexity_analysis = self._analyze_message_complexity(message)
            audience_preferences = self._analyze_audience_preferences(audience)
            
            # Apply language adaptations
            adapted_content = self._adapt_language(message.content, audience)
            executive_summary = self._create_executive_summary(message, audience)
            key_recommendations = self._extract_key_recommendations(message, audience)
            
            # Determine optimal tone and structure
            tone = self._determine_optimal_tone(audience, message.message_type)
            language_complexity = self._adjust_language_complexity(audience)
            
            # Calculate effectiveness prediction
            effectiveness_score = self._predict_effectiveness(
                adapted_content, audience, message.message_type
            )
            
            adapted_message = AdaptedMessage(
                id=f"adapted_{message.id}_{audience.id}",
                original_message_id=message.id,
                audience_id=audience.id,
                adapted_content=adapted_content,
                executive_summary=executive_summary,
                key_recommendations=key_recommendations,
                tone=tone,
                language_complexity=language_complexity,
                estimated_reading_time=self._estimate_reading_time(adapted_content),
                effectiveness_score=effectiveness_score,
                adaptation_rationale=self._generate_adaptation_rationale(audience, message),
                created_at=datetime.now()
            )
            
            self.logger.info(f"Successfully adapted message for {audience.executive_level.value}")
            return adapted_message
            
        except Exception as e:
            self.logger.error(f"Error adapting communication style: {str(e)}")
            raise
    
    def _adapt_language(self, content: str, audience: ExecutiveAudience) -> str:
        """Apply language adaptations based on audience profile"""
        adapted_content = content
        
        # Apply executive-level vocabulary
        for technical_term, executive_term in self.executive_vocabulary.items():
            if audience.detail_preference == "low":
                adapted_content = adapted_content.replace(technical_term, executive_term)
        
        # Adjust sentence structure for executive consumption
        if audience.executive_level in [ExecutiveLevel.CEO, ExecutiveLevel.BOARD_CHAIR]:
            adapted_content = self._simplify_sentence_structure(adapted_content)
        
        # Apply communication style preferences
        if audience.communication_style == CommunicationStyle.DIRECT:
            adapted_content = self._make_more_direct(adapted_content)
        elif audience.communication_style == CommunicationStyle.DIPLOMATIC:
            adapted_content = self._make_more_diplomatic(adapted_content)
        elif audience.communication_style == CommunicationStyle.ANALYTICAL:
            adapted_content = self._add_analytical_structure(adapted_content)
        
        return adapted_content
    
    def _create_executive_summary(self, message: Message, audience: ExecutiveAudience) -> str:
        """Create concise executive summary"""
        key_points = message.key_points[:3]  # Top 3 points for executives
        
        if audience.attention_span <= 5:  # Very short attention span
            summary = f"Key Decision: {key_points[0] if key_points else 'Action required'}"
        else:
            summary = "Executive Summary:\n"
            for i, point in enumerate(key_points, 1):
                summary += f"{i}. {point}\n"
        
        return summary.strip()
    
    def _extract_key_recommendations(self, message: Message, audience: ExecutiveAudience) -> List[str]:
        """Extract and prioritize key recommendations"""
        recommendations = []
        
        # Extract action items and recommendations from content
        content_lines = message.content.split('\n')
        for line in content_lines:
            if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'should', 'must', 'action', 'requires', 'need']):
                recommendations.append(line.strip())
        
        # If no explicit recommendations found, generate from key points
        if not recommendations and message.key_points:
            for point in message.key_points:
                if any(keyword in point.lower() for keyword in ['implement', 'address', 'improve', 'fix', 'upgrade']):
                    recommendations.append(f"Address: {point}")
        
        # If still no recommendations, create from message type and content
        if not recommendations:
            if message.message_type == MessageType.RISK_ASSESSMENT:
                recommendations.append("Immediate risk mitigation required")
            elif message.message_type == MessageType.STRATEGIC_UPDATE:
                recommendations.append("Strategic implementation needed")
            elif "approval" in message.content.lower():
                recommendations.append("Board approval required")
        
        # Prioritize based on audience level
        if audience.executive_level in [ExecutiveLevel.CEO, ExecutiveLevel.BOARD_CHAIR]:
            # Focus on strategic recommendations
            strategic_recs = [rec for rec in recommendations if any(
                word in rec.lower() for word in ['strategic', 'market', 'competitive', 'growth', 'approval', 'budget']
            )]
            if strategic_recs:
                return strategic_recs[:3]
        
        return recommendations[:5]
    
    def _determine_optimal_tone(self, audience: ExecutiveAudience, message_type: MessageType) -> str:
        """Determine optimal communication tone"""
        if message_type == MessageType.CRISIS_COMMUNICATION:
            return "urgent_but_controlled"
        elif audience.executive_level == ExecutiveLevel.BOARD_CHAIR:
            return "respectful_strategic"
        elif audience.communication_style == CommunicationStyle.DIRECT:
            return "direct_confident"
        elif audience.communication_style == CommunicationStyle.DIPLOMATIC:
            return "diplomatic_collaborative"
        else:
            return "professional_strategic"
    
    def _adjust_language_complexity(self, audience: ExecutiveAudience) -> str:
        """Adjust language complexity based on audience"""
        if audience.detail_preference == "low":
            return "simplified"
        elif audience.detail_preference == "high" and "technical" in audience.expertise_areas:
            return "detailed"
        else:
            return "balanced"
    
    def _predict_effectiveness(self, content: str, audience: ExecutiveAudience, message_type: MessageType) -> float:
        """Predict communication effectiveness"""
        score = 0.7  # Base score
        
        # Adjust based on content length vs attention span
        reading_time = self._estimate_reading_time(content)
        if reading_time <= audience.attention_span:
            score += 0.2
        elif reading_time > audience.attention_span * 2:
            score -= 0.3
        
        # Adjust based on message type alignment
        if message_type == MessageType.STRATEGIC_UPDATE and audience.executive_level == ExecutiveLevel.CEO:
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _estimate_reading_time(self, content: str) -> int:
        """Estimate reading time in minutes"""
        word_count = len(content.split())
        return max(1, word_count // 200)  # Assume 200 words per minute
    
    def _generate_adaptation_rationale(self, audience: ExecutiveAudience, message: Message) -> str:
        """Generate rationale for adaptation choices"""
        return f"Adapted for {audience.executive_level.value} with {audience.communication_style.value} style, " \
               f"{audience.detail_preference} detail preference, {audience.attention_span}min attention span"
    
    def _analyze_message_complexity(self, message: Message) -> Dict[str, Any]:
        """Analyze message complexity"""
        return {
            "technical_terms": len(re.findall(r'\b[A-Z]{2,}\b', message.content)),
            "sentence_length": len(message.content.split('.')) / len(message.content.split()),
            "complexity_score": message.technical_complexity
        }
    
    def _analyze_audience_preferences(self, audience: ExecutiveAudience) -> Dict[str, Any]:
        """Analyze audience communication preferences"""
        return {
            "prefers_brevity": audience.attention_span <= 10,
            "technical_background": "technical" in audience.expertise_areas,
            "decision_maker": audience.influence_level > 0.8
        }
    
    def _simplify_sentence_structure(self, content: str) -> str:
        """Simplify sentence structure for executive consumption"""
        # Split long sentences and use bullet points
        sentences = content.split('.')
        simplified = []
        
        for sentence in sentences:
            if len(sentence.split()) > 20:  # Long sentence
                # Try to break into bullet points
                if ',' in sentence:
                    parts = sentence.split(',')
                    simplified.append(parts[0] + ':')
                    for part in parts[1:]:
                        simplified.append(f"• {part.strip()}")
                else:
                    simplified.append(sentence)
            else:
                simplified.append(sentence)
        
        return '. '.join(simplified)
    
    def _make_more_direct(self, content: str) -> str:
        """Make communication more direct"""
        # Remove hedging language
        hedging_words = ['perhaps', 'maybe', 'possibly', 'might', 'could potentially']
        for word in hedging_words:
            content = content.replace(word, '')
        
        # Add direct action language
        content = re.sub(r'we should consider', 'we must', content, flags=re.IGNORECASE)
        content = re.sub(r'it might be beneficial', 'we recommend', content, flags=re.IGNORECASE)
        
        return content
    
    def _make_more_diplomatic(self, content: str) -> str:
        """Make communication more diplomatic"""
        # Soften direct statements
        content = re.sub(r'we must', 'we should consider', content, flags=re.IGNORECASE)
        content = re.sub(r'this is wrong', 'this presents challenges', content, flags=re.IGNORECASE)
        
        return content
    
    def _add_analytical_structure(self, content: str) -> str:
        """Add analytical structure to content"""
        if not content.startswith('Analysis:'):
            content = f"Analysis:\n{content}\n\nConclusion:\nBased on the above analysis, we recommend proceeding with the outlined approach."
        
        return content
    
    def _load_adaptation_rules(self) -> List[LanguageAdaptationRule]:
        """Load language adaptation rules"""
        # In a real implementation, this would load from database
        return []
    
    def _load_executive_vocabulary(self) -> Dict[str, str]:
        """Load executive vocabulary mappings"""
        return {
            "API": "system interface",
            "microservices": "modular architecture",
            "containerization": "deployment optimization",
            "CI/CD": "automated deployment",
            "scalability": "growth capacity",
            "latency": "response time",
            "throughput": "processing capacity",
            "redundancy": "backup systems",
            "load balancing": "traffic distribution",
            "caching": "performance optimization"
        }


class CommunicationEffectivenessTracker:
    """Tracks and measures communication effectiveness"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.effectiveness_history = []
    
    def measure_effectiveness(self, message_id: str, audience_id: str, 
                            engagement_data: Dict[str, Any]) -> CommunicationEffectiveness:
        """Measure communication effectiveness"""
        try:
            effectiveness = CommunicationEffectiveness(
                id=f"eff_{message_id}_{audience_id}",
                message_id=message_id,
                audience_id=audience_id,
                engagement_score=engagement_data.get('engagement_score', 0.0),
                comprehension_score=engagement_data.get('comprehension_score', 0.0),
                action_taken=engagement_data.get('action_taken', False),
                feedback_received=engagement_data.get('feedback'),
                response_time=engagement_data.get('response_time'),
                follow_up_questions=engagement_data.get('follow_up_questions', 0),
                decision_influenced=engagement_data.get('decision_influenced', False),
                measured_at=datetime.now()
            )
            
            self.effectiveness_history.append(effectiveness)
            self.logger.info(f"Measured communication effectiveness: {effectiveness.engagement_score}")
            
            return effectiveness
            
        except Exception as e:
            self.logger.error(f"Error measuring effectiveness: {str(e)}")
            raise
    
    def optimize_based_on_feedback(self, effectiveness: CommunicationEffectiveness) -> Dict[str, str]:
        """Generate optimization recommendations based on effectiveness data"""
        recommendations = {}
        
        if effectiveness.engagement_score < 0.6:
            recommendations['engagement'] = "Consider shorter, more direct communication"
        
        if effectiveness.comprehension_score < 0.7:
            recommendations['clarity'] = "Simplify language and add more context"
        
        if effectiveness.follow_up_questions > 3:
            recommendations['completeness'] = "Provide more comprehensive initial information"
        
        if not effectiveness.action_taken:
            recommendations['actionability'] = "Include clearer call-to-action and next steps"
        
        return recommendations


class ExecutiveCommunicationSystem:
    """Main executive communication system"""
    
    def __init__(self):
        self.language_adapter = LanguageAdapter()
        self.effectiveness_tracker = CommunicationEffectivenessTracker()
        self.logger = logging.getLogger(__name__)
    
    def process_executive_communication(self, message: Message, 
                                      audience: ExecutiveAudience) -> AdaptedMessage:
        """Process and adapt communication for executive audience"""
        try:
            # Adapt the message
            adapted_message = self.language_adapter.adapt_communication_style(message, audience)
            
            self.logger.info(f"Successfully processed executive communication for {audience.name}")
            return adapted_message
            
        except Exception as e:
            self.logger.error(f"Error processing executive communication: {str(e)}")
            raise
    
    def track_communication_effectiveness(self, message_id: str, audience_id: str,
                                        engagement_data: Dict[str, Any]) -> CommunicationEffectiveness:
        """Track and measure communication effectiveness"""
        return self.effectiveness_tracker.measure_effectiveness(message_id, audience_id, engagement_data)
    
    def get_optimization_recommendations(self, effectiveness: CommunicationEffectiveness) -> Dict[str, str]:
        """Get recommendations for improving communication effectiveness"""
        return self.effectiveness_tracker.optimize_based_on_feedback(effectiveness)