"""
Intent Classification Engine for the Automated Code Generation System.
Classifies user intents from natural language requirements.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import re
from enum import Enum
import openai
from sqlalchemy.orm import Session

from scrollintel.models.nlp_models import IntentType, ConfidenceLevel
from scrollintel.core.config import get_settings

logger = logging.getLogger(__name__)


class IntentClassificationResult:
    """Result of intent classification."""
    
    def __init__(
        self,
        primary_intent: IntentType,
        confidence: float,
        secondary_intents: List[Tuple[IntentType, float]] = None,
        reasoning: str = ""
    ):
        self.primary_intent = primary_intent
        self.confidence = confidence
        self.secondary_intents = secondary_intents or []
        self.reasoning = reasoning
        self.confidence_level = self._get_confidence_level(confidence)
    
    def _get_confidence_level(self, score: float) -> ConfidenceLevel:
        """Convert confidence score to confidence level."""
        if score >= 0.8:
            return ConfidenceLevel.HIGH
        elif score >= 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW


class IntentClassifier:
    """
    Advanced Intent Classifier using both rule-based and AI-powered approaches.
    Identifies user intentions from natural language requirements.
    """
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.settings = get_settings()
        self.client = openai.OpenAI(api_key=self.settings.openai_api_key)
        
        # Intent patterns for rule-based classification
        self.intent_patterns = {
            IntentType.CREATE_APPLICATION: [
                r'\b(create|build|develop|make)\s+(a|an|new)?\s*(application|app|system|platform)',
                r'\bi\s+want\s+to\s+(create|build|develop)',
                r'\bneed\s+(a|an)\s+(new|custom)?\s*(application|system|platform)',
                r'\bfrom\s+scratch\b',
                r'\bstart\s+(a|with)\s+new\b'
            ],
            IntentType.MODIFY_FEATURE: [
                r'\b(modify|change|update|alter|edit)\b.*\b(feature|functionality|existing)',
                r'\bchange\s+how\b',
                r'\bupdate\s+the\s+way\b',
                r'\bmodify\s+existing\b',
                r'\balter\s+the\s+current\b'
            ],
            IntentType.ADD_FUNCTIONALITY: [
                r'\b(add|include|implement|introduce)\b.*\b(feature|functionality|capability)',
                r'\bi\s+also\s+need\b',
                r'\badditionally\b',
                r'\bextend\s+with\b',
                r'\benhance\s+by\s+adding\b'
            ],
            IntentType.INTEGRATE_SYSTEM: [
                r'\b(integrate|connect|link)\s+with\b',
                r'\bconnect\s+to\s+(external|third.?party)\b',
                r'\bapi\s+integration\b',
                r'\bsync\s+with\b',
                r'\bimport\s+from\b'
            ],
            IntentType.OPTIMIZE_PERFORMANCE: [
                r'\b(optimize|improve|enhance)\b.*\b(performance|speed|efficiency)',
                r'\bmake\s+it\s+(faster|quicker|more\s+efficient)',
                r'\bperformance\s+(issues|problems|optimization)',
                r'\bspeed\s+up\b',
                r'\breduce\s+(latency|response\s+time)\b',
                r'\btoo\s+slow\b'
            ],
            IntentType.ENHANCE_SECURITY: [
                r'\b(secure|protect|safeguard)\b',
                r'\bsecurity\s+(features|measures|enhancements)',
                r'\bauthentication\b',
                r'\bauthorization\b',
                r'\bencryption\b',
                r'\baccess\s+control\b'
            ],
            IntentType.IMPROVE_UI: [
                r'\b(ui|user\s+interface|frontend|design)\b',
                r'\bmake\s+it\s+(prettier|more\s+attractive|user.?friendly)',
                r'\bimprove\s+(the\s+)?design\b',
                r'\buser\s+experience\b',
                r'\bresponsive\s+design\b'
            ],
            IntentType.MANAGE_DATA: [
                r'\b(data|database|storage)\b.*\b(management|handling|processing|design|schema)',
                r'\bstore\s+(data|information)\b',
                r'\bdatabase\s+(design|schema|structure)',
                r'\bdata\s+(migration|import|export)',
                r'\bmanage\s+(records|entries|information)\b',
                r'\bdesign.*database\b'
            ],
            IntentType.DEPLOY_APPLICATION: [
                r'\b(deploy|deployment|hosting|launch)\b',
                r'\bput\s+it\s+(online|live|in\s+production)',
                r'\bcloud\s+(deployment|hosting)',
                r'\bserver\s+setup\b',
                r'\bgo\s+live\b'
            ],
            IntentType.CLARIFY_REQUIREMENT: [
                r'\bwhat\s+(do\s+you\s+mean|does\s+that\s+mean)\b',
                r'\bcan\s+you\s+(explain|clarify|elaborate)\b',
                r'\bi\s+don\'?t\s+understand\b',
                r'\bnot\s+sure\s+(about|what)\b',
                r'\bconfused\s+about\b'
            ]
        }
        
        # System prompt for AI-powered classification
        self.classification_prompt = """
        You are an expert software requirements analyst. Classify the user's intent from their natural language requirement.
        
        Available intent types:
        - CREATE_APPLICATION: User wants to create a new application from scratch
        - MODIFY_FEATURE: User wants to modify existing functionality
        - ADD_FUNCTIONALITY: User wants to add new features to existing system
        - INTEGRATE_SYSTEM: User wants to integrate with external systems
        - OPTIMIZE_PERFORMANCE: User wants to improve system performance
        - ENHANCE_SECURITY: User wants to add or improve security features
        - IMPROVE_UI: User wants to improve user interface or user experience
        - MANAGE_DATA: User wants to handle data storage, processing, or management
        - DEPLOY_APPLICATION: User wants to deploy or host the application
        - CLARIFY_REQUIREMENT: User is asking for clarification or has questions
        
        Analyze the text and return:
        1. Primary intent (most likely)
        2. Confidence score (0.0 to 1.0)
        3. Secondary intents (if any) with their confidence scores
        4. Brief reasoning for the classification
        
        Return as JSON format.
        """

    async def classify_intent(self, text: str) -> IntentClassificationResult:
        """
        Classify the intent of the given text.
        
        Args:
            text: Natural language text to classify
            
        Returns:
            IntentClassificationResult with classification details
        """
        try:
            # First try rule-based classification
            rule_based_result = self._classify_with_rules(text)
            
            # If rule-based classification has high confidence, use it
            if rule_based_result.confidence >= 0.8:
                return rule_based_result
            
            # Otherwise, use AI-powered classification
            ai_result = await self._classify_with_ai(text)
            
            # Combine results if both have reasonable confidence
            if rule_based_result.confidence >= 0.6 and ai_result.confidence >= 0.6:
                return self._combine_results(rule_based_result, ai_result)
            
            # Return the result with higher confidence
            return ai_result if ai_result.confidence > rule_based_result.confidence else rule_based_result
            
        except Exception as e:
            logger.error(f"Error classifying intent: {str(e)}")
            return IntentClassificationResult(
                primary_intent=IntentType.CREATE_APPLICATION,
                confidence=0.1,
                reasoning=f"Classification failed: {str(e)}"
            )

    def _classify_with_rules(self, text: str) -> IntentClassificationResult:
        """Classify intent using rule-based patterns."""
        text_lower = text.lower()
        intent_scores = {}
        
        # Score each intent based on pattern matches
        for intent, patterns in self.intent_patterns.items():
            score = 0.0
            matches = []
            
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    score += 1.0
                    matches.append(pattern)
            
            # Normalize score based on number of patterns
            if patterns:
                intent_scores[intent] = min(score / len(patterns), 1.0)
        
        if not intent_scores:
            return IntentClassificationResult(
                primary_intent=IntentType.CREATE_APPLICATION,
                confidence=0.0,
                reasoning="No patterns matched - defaulting to CREATE_APPLICATION"
            )
        
        # Find primary intent (highest score)
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])
        
        # Find secondary intents (scores > 0.3 and not primary)
        secondary_intents = [
            (intent, score) for intent, score in intent_scores.items()
            if score > 0.3 and intent != primary_intent[0]
        ]
        secondary_intents.sort(key=lambda x: x[1], reverse=True)
        
        return IntentClassificationResult(
            primary_intent=primary_intent[0],
            confidence=primary_intent[1],
            secondary_intents=secondary_intents[:3],  # Top 3 secondary intents
            reasoning=f"Rule-based classification based on pattern matching"
        )

    async def _classify_with_ai(self, text: str) -> IntentClassificationResult:
        """Classify intent using AI (GPT-4)."""
        try:
            response = await self.client.chat.completions.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.classification_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            result_data = self._parse_ai_response(content)
            
            primary_intent = IntentType(result_data.get('primary_intent', 'CREATE_APPLICATION'))
            confidence = float(result_data.get('confidence', 0.5))
            
            secondary_intents = []
            for secondary in result_data.get('secondary_intents', []):
                try:
                    intent = IntentType(secondary.get('intent'))
                    score = float(secondary.get('confidence', 0.0))
                    secondary_intents.append((intent, score))
                except (ValueError, KeyError):
                    continue
            
            reasoning = result_data.get('reasoning', 'AI-powered classification')
            
            return IntentClassificationResult(
                primary_intent=primary_intent,
                confidence=confidence,
                secondary_intents=secondary_intents,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error in AI classification: {str(e)}")
            return IntentClassificationResult(
                primary_intent=IntentType.CREATE_APPLICATION,
                confidence=0.1,
                reasoning=f"AI classification failed: {str(e)}"
            )

    def _parse_ai_response(self, content: str) -> Dict[str, Any]:
        """Parse AI response content into structured data."""
        try:
            import json
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from text
            json_pattern = r'\{.*\}'
            matches = re.findall(json_pattern, content, re.DOTALL)
            
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
            
            # Fallback: parse manually
            return self._manual_parse_response(content)

    def _manual_parse_response(self, content: str) -> Dict[str, Any]:
        """Manually parse AI response when JSON parsing fails."""
        result = {
            'primary_intent': 'CREATE_APPLICATION',
            'confidence': 0.5,
            'secondary_intents': [],
            'reasoning': 'Manual parsing of AI response'
        }
        
        # Look for intent mentions
        content_upper = content.upper()
        for intent in IntentType:
            if intent.value.upper() in content_upper:
                result['primary_intent'] = intent.value
                break
        
        # Look for confidence scores
        confidence_pattern = r'confidence[:\s]+([0-9.]+)'
        confidence_match = re.search(confidence_pattern, content.lower())
        if confidence_match:
            try:
                result['confidence'] = float(confidence_match.group(1))
            except ValueError:
                pass
        
        return result

    def _combine_results(
        self, 
        rule_result: IntentClassificationResult, 
        ai_result: IntentClassificationResult
    ) -> IntentClassificationResult:
        """Combine rule-based and AI classification results."""
        # If both agree on primary intent, increase confidence
        if rule_result.primary_intent == ai_result.primary_intent:
            combined_confidence = min((rule_result.confidence + ai_result.confidence) / 2 * 1.2, 1.0)
            return IntentClassificationResult(
                primary_intent=rule_result.primary_intent,
                confidence=combined_confidence,
                secondary_intents=rule_result.secondary_intents + ai_result.secondary_intents,
                reasoning=f"Combined rule-based and AI classification (agreement)"
            )
        
        # If they disagree, use the one with higher confidence
        if ai_result.confidence > rule_result.confidence:
            return IntentClassificationResult(
                primary_intent=ai_result.primary_intent,
                confidence=ai_result.confidence * 0.9,  # Slight penalty for disagreement
                secondary_intents=[(rule_result.primary_intent, rule_result.confidence)] + ai_result.secondary_intents,
                reasoning=f"AI classification chosen over rule-based (disagreement)"
            )
        else:
            return IntentClassificationResult(
                primary_intent=rule_result.primary_intent,
                confidence=rule_result.confidence * 0.9,  # Slight penalty for disagreement
                secondary_intents=[(ai_result.primary_intent, ai_result.confidence)] + rule_result.secondary_intents,
                reasoning=f"Rule-based classification chosen over AI (disagreement)"
            )

    async def classify_multiple_intents(self, texts: List[str]) -> List[IntentClassificationResult]:
        """Classify intents for multiple texts."""
        results = []
        for text in texts:
            result = await self.classify_intent(text)
            results.append(result)
        return results

    def get_intent_statistics(self, results: List[IntentClassificationResult]) -> Dict[str, Any]:
        """Get statistics about intent classification results."""
        if not results:
            return {}
        
        intent_counts = {}
        confidence_scores = []
        
        for result in results:
            intent = result.primary_intent.value
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
            confidence_scores.append(result.confidence)
        
        return {
            'total_classifications': len(results),
            'intent_distribution': intent_counts,
            'average_confidence': sum(confidence_scores) / len(confidence_scores),
            'high_confidence_count': len([s for s in confidence_scores if s >= 0.8]),
            'low_confidence_count': len([s for s in confidence_scores if s < 0.6])
        }