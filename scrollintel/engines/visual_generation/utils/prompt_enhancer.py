"""
Intelligent Prompt Enhancement Engine for visual content generation.
Provides ML-based prompt improvement, context-aware suggestions, and learning from feedback.
"""

import re
import json
import random
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import Counter, defaultdict
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_

from .prompt_analyzer import PromptAnalyzer, PromptAnalysisResult, ArtisticStyle, ContentType
from .prompt_template_manager import PromptTemplateManager
from scrollintel.models.prompt_enhancement_models import (
    VisualPromptTemplate, VisualPromptPattern, VisualPromptOptimizationSuggestion,
    VisualPromptUsageLog, VisualABTestResult
)
from ..exceptions import PromptEnhancementError
from ..base import GenerationRequest


class SuggestionType(Enum):
    """Types of prompt enhancement suggestions."""
    QUALITY_IMPROVEMENT = "quality_improvement"
    STYLE_ENHANCEMENT = "style_enhancement"
    TECHNICAL_OPTIMIZATION = "technical_optimization"
    COMPOSITION_IMPROVEMENT = "composition_improvement"
    CONTEXT_ENRICHMENT = "context_enrichment"
    SPECIFICITY_BOOST = "specificity_boost"


class EnhancementStrategy(Enum):
    """Enhancement strategies based on user intent."""
    CONSERVATIVE = "conservative"  # Minimal changes, preserve original intent
    MODERATE = "moderate"  # Balanced improvements
    AGGRESSIVE = "aggressive"  # Maximum enhancement for best results


@dataclass
class EnhancementSuggestion:
    """Individual enhancement suggestion."""
    suggestion_type: SuggestionType
    original_text: str
    enhanced_text: str
    confidence: float
    reasoning: str
    impact_score: float
    pattern_source: Optional[str] = None


@dataclass
class PromptEnhancementResult:
    """Complete result of prompt enhancement."""
    original_prompt: str
    enhanced_prompt: str
    suggestions: List[EnhancementSuggestion]
    overall_confidence: float
    improvement_score: float
    strategy_used: EnhancementStrategy
    analysis_result: PromptAnalysisResult
    enhancement_metadata: Dict[str, Any]
    timestamp: str


class PromptEnhancer:
    """Intelligent prompt enhancement engine with ML-based improvements."""
    
    def __init__(self, db_session: Optional[Session] = None):
        self.analyzer = PromptAnalyzer()
        self.template_manager = PromptTemplateManager(db_session)
        
        # Enhancement patterns learned from successful generations
        self.quality_enhancers = self._load_quality_enhancers()
        self.style_enhancers = self._load_style_enhancers()
        self.technical_enhancers = self._load_technical_enhancers()
        self.composition_enhancers = self._load_composition_enhancers()
        
        # Context-aware enhancement rules
        self.context_rules = self._load_context_rules()
        
        # Feedback learning system
        self.feedback_weights = self._load_feedback_weights()
        
        # Performance tracking
        self.enhancement_history = []
        
    def _load_quality_enhancers(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load quality enhancement patterns from successful generations."""
        return {
            'high_quality_modifiers': [
                {'pattern': 'high quality', 'weight': 0.9, 'context': 'general'},
                {'pattern': 'masterpiece', 'weight': 0.95, 'context': 'artistic'},
                {'pattern': 'award winning', 'weight': 0.92, 'context': 'professional'},
                {'pattern': 'ultra detailed', 'weight': 0.88, 'context': 'technical'},
                {'pattern': 'photorealistic', 'weight': 0.9, 'context': 'realistic'},
                {'pattern': 'sharp focus', 'weight': 0.85, 'context': 'photography'},
                {'pattern': 'professional photography', 'weight': 0.93, 'context': 'photo'},
                {'pattern': 'studio lighting', 'weight': 0.87, 'context': 'portrait'},
                {'pattern': 'cinematic', 'weight': 0.89, 'context': 'dramatic'},
                {'pattern': 'trending on artstation', 'weight': 0.86, 'context': 'digital_art'}
            ],
            'resolution_enhancers': [
                {'pattern': '8k', 'weight': 0.95, 'context': 'high_res'},
                {'pattern': '4k', 'weight': 0.9, 'context': 'standard_high'},
                {'pattern': 'ultra hd', 'weight': 0.88, 'context': 'video'},
                {'pattern': 'high resolution', 'weight': 0.85, 'context': 'general'}
            ],
            'detail_enhancers': [
                {'pattern': 'intricate details', 'weight': 0.87, 'context': 'complex'},
                {'pattern': 'fine details', 'weight': 0.84, 'context': 'subtle'},
                {'pattern': 'hyperdetailed', 'weight': 0.89, 'context': 'maximum'},
                {'pattern': 'extremely detailed', 'weight': 0.86, 'context': 'emphasis'}
            ]
        }
    
    def _load_style_enhancers(self) -> Dict[ArtisticStyle, List[Dict[str, Any]]]:
        """Load style-specific enhancement patterns."""
        return {
            ArtisticStyle.PHOTOREALISTIC: [
                {'pattern': 'natural lighting', 'weight': 0.9, 'impact': 'lighting'},
                {'pattern': 'depth of field', 'weight': 0.85, 'impact': 'composition'},
                {'pattern': 'bokeh', 'weight': 0.8, 'impact': 'background'},
                {'pattern': 'professional camera', 'weight': 0.87, 'impact': 'technical'}
            ],
            ArtisticStyle.ARTISTIC: [
                {'pattern': 'oil painting style', 'weight': 0.92, 'impact': 'medium'},
                {'pattern': 'brush strokes', 'weight': 0.88, 'impact': 'texture'},
                {'pattern': 'canvas texture', 'weight': 0.85, 'impact': 'surface'},
                {'pattern': 'artistic composition', 'weight': 0.9, 'impact': 'layout'}
            ],
            ArtisticStyle.CINEMATIC: [
                {'pattern': 'dramatic lighting', 'weight': 0.93, 'impact': 'mood'},
                {'pattern': 'wide shot', 'weight': 0.87, 'impact': 'framing'},
                {'pattern': 'film grain', 'weight': 0.82, 'impact': 'texture'},
                {'pattern': 'color grading', 'weight': 0.89, 'impact': 'color'}
            ],
            ArtisticStyle.DIGITAL_ART: [
                {'pattern': 'concept art style', 'weight': 0.91, 'impact': 'style'},
                {'pattern': 'matte painting', 'weight': 0.88, 'impact': 'technique'},
                {'pattern': 'digital painting', 'weight': 0.86, 'impact': 'medium'},
                {'pattern': 'artstation quality', 'weight': 0.89, 'impact': 'quality'}
            ]
        }
    
    def _load_technical_enhancers(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load technical enhancement patterns."""
        return {
            'lighting': [
                {'pattern': 'golden hour lighting', 'weight': 0.9, 'context': 'outdoor'},
                {'pattern': 'soft lighting', 'weight': 0.87, 'context': 'portrait'},
                {'pattern': 'rim lighting', 'weight': 0.85, 'context': 'dramatic'},
                {'pattern': 'volumetric lighting', 'weight': 0.88, 'context': 'atmospheric'}
            ],
            'camera': [
                {'pattern': 'shallow depth of field', 'weight': 0.86, 'context': 'focus'},
                {'pattern': 'wide angle lens', 'weight': 0.83, 'context': 'landscape'},
                {'pattern': 'macro lens', 'weight': 0.85, 'context': 'close_up'},
                {'pattern': 'telephoto lens', 'weight': 0.82, 'context': 'distant'}
            ],
            'composition': [
                {'pattern': 'rule of thirds', 'weight': 0.84, 'context': 'layout'},
                {'pattern': 'leading lines', 'weight': 0.82, 'context': 'guidance'},
                {'pattern': 'symmetrical composition', 'weight': 0.81, 'context': 'balance'},
                {'pattern': 'dynamic composition', 'weight': 0.85, 'context': 'energy'}
            ]
        }
    
    def _load_composition_enhancers(self) -> Dict[ContentType, List[Dict[str, Any]]]:
        """Load composition enhancers based on content type."""
        return {
            ContentType.PORTRAIT: [
                {'pattern': 'eye contact', 'weight': 0.88, 'impact': 'connection'},
                {'pattern': 'natural expression', 'weight': 0.86, 'impact': 'emotion'},
                {'pattern': 'perfect skin', 'weight': 0.84, 'impact': 'quality'},
                {'pattern': 'professional headshot', 'weight': 0.89, 'impact': 'style'}
            ],
            ContentType.LANDSCAPE: [
                {'pattern': 'vast landscape', 'weight': 0.87, 'impact': 'scale'},
                {'pattern': 'dramatic sky', 'weight': 0.89, 'impact': 'atmosphere'},
                {'pattern': 'foreground interest', 'weight': 0.85, 'impact': 'depth'},
                {'pattern': 'natural colors', 'weight': 0.83, 'impact': 'realism'}
            ],
            ContentType.ARCHITECTURE: [
                {'pattern': 'architectural details', 'weight': 0.86, 'impact': 'precision'},
                {'pattern': 'geometric patterns', 'weight': 0.84, 'impact': 'structure'},
                {'pattern': 'urban environment', 'weight': 0.82, 'impact': 'context'},
                {'pattern': 'modern design', 'weight': 0.85, 'impact': 'style'}
            ]
        }
    
    def _load_context_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load context-aware enhancement rules."""
        return {
            'time_of_day': {
                'morning': ['golden hour', 'soft morning light', 'fresh atmosphere'],
                'evening': ['sunset lighting', 'warm colors', 'dramatic sky'],
                'night': ['moonlight', 'artificial lighting', 'dark atmosphere']
            },
            'mood': {
                'happy': ['bright colors', 'cheerful', 'positive energy'],
                'dramatic': ['high contrast', 'dramatic lighting', 'intense mood'],
                'peaceful': ['soft colors', 'calm atmosphere', 'serene']
            },
            'purpose': {
                'commercial': ['professional', 'clean', 'marketing quality'],
                'artistic': ['creative', 'expressive', 'unique perspective'],
                'documentary': ['realistic', 'authentic', 'natural']
            }
        }
    
    def _load_feedback_weights(self) -> Dict[str, float]:
        """Load weights based on user feedback learning."""
        # These would be learned from actual user feedback over time
        return {
            'quality_modifiers': 0.9,
            'style_consistency': 0.85,
            'technical_accuracy': 0.88,
            'user_preference': 0.92,
            'generation_success': 0.95
        }
    
    async def enhance_prompt(self, prompt: str, user_id: Optional[str] = None,
                           strategy: EnhancementStrategy = EnhancementStrategy.MODERATE,
                           context: Optional[Dict[str, Any]] = None) -> PromptEnhancementResult:
        """
        Enhance a prompt using ML-based improvements and context-aware suggestions.
        
        Args:
            prompt: Original user prompt
            user_id: User identifier for personalized suggestions
            strategy: Enhancement strategy (conservative, moderate, aggressive)
            context: Additional context for enhancement (purpose, mood, etc.)
            
        Returns:
            PromptEnhancementResult with enhanced prompt and suggestions
        """
        try:
            # Analyze the original prompt
            analysis = await self.analyzer.analyze_prompt(prompt)
            
            # Get user preferences if available
            user_preferences = await self._get_user_preferences(user_id) if user_id else {}
            
            # Generate enhancement suggestions
            suggestions = await self._generate_suggestions(
                prompt, analysis, strategy, context, user_preferences
            )
            
            # Apply suggestions to create enhanced prompt
            enhanced_prompt = await self._apply_suggestions(prompt, suggestions, strategy)
            
            # Calculate improvement metrics
            improvement_score = await self._calculate_improvement_score(
                analysis, suggestions
            )
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(suggestions)
            
            # Create result
            result = PromptEnhancementResult(
                original_prompt=prompt,
                enhanced_prompt=enhanced_prompt,
                suggestions=suggestions,
                overall_confidence=overall_confidence,
                improvement_score=improvement_score,
                strategy_used=strategy,
                analysis_result=analysis,
                enhancement_metadata={
                    'user_id': user_id,
                    'context': context,
                    'suggestions_count': len(suggestions),
                    'enhancement_types': [s.suggestion_type.value for s in suggestions]
                },
                timestamp=datetime.now().isoformat()
            )
            
            # Log enhancement for learning
            await self._log_enhancement(result, user_id)
            
            return result
            
        except Exception as e:
            raise PromptEnhancementError(f"Failed to enhance prompt: {str(e)}")
    
    async def _generate_suggestions(self, prompt: str, analysis: PromptAnalysisResult,
                                  strategy: EnhancementStrategy, 
                                  context: Optional[Dict[str, Any]],
                                  user_preferences: Dict[str, Any]) -> List[EnhancementSuggestion]:
        """Generate context-aware enhancement suggestions."""
        suggestions = []
        
        # Quality improvement suggestions
        quality_suggestions = await self._generate_quality_suggestions(
            prompt, analysis, strategy
        )
        suggestions.extend(quality_suggestions)
        
        # Style enhancement suggestions
        style_suggestions = await self._generate_style_suggestions(
            prompt, analysis, strategy
        )
        suggestions.extend(style_suggestions)
        
        # Technical optimization suggestions
        technical_suggestions = await self._generate_technical_suggestions(
            prompt, analysis, strategy
        )
        suggestions.extend(technical_suggestions)
        
        # Context-aware suggestions
        if context:
            context_suggestions = await self._generate_context_suggestions(
                prompt, analysis, context, strategy
            )
            suggestions.extend(context_suggestions)
        
        # Personalized suggestions based on user preferences
        if user_preferences:
            personal_suggestions = await self._generate_personal_suggestions(
                prompt, analysis, user_preferences, strategy
            )
            suggestions.extend(personal_suggestions)
        
        # Template-based suggestions
        template_suggestions = await self._generate_template_suggestions(
            prompt, analysis, strategy
        )
        suggestions.extend(template_suggestions)
        
        # Filter and rank suggestions
        filtered_suggestions = self._filter_and_rank_suggestions(
            suggestions, strategy, analysis
        )
        
        return filtered_suggestions
    
    async def _generate_quality_suggestions(self, prompt: str, 
                                          analysis: PromptAnalysisResult,
                                          strategy: EnhancementStrategy) -> List[EnhancementSuggestion]:
        """Generate quality improvement suggestions."""
        suggestions = []
        
        # Check if quality modifiers are missing
        if analysis.quality_metrics.technical_completeness < 0.7:
            quality_enhancers = self.quality_enhancers['high_quality_modifiers']
            
            # Select appropriate quality enhancers
            for enhancer in quality_enhancers[:3]:  # Limit to top 3
                if enhancer['pattern'].lower() not in prompt.lower():
                    confidence = enhancer['weight'] * self.feedback_weights['quality_modifiers']
                    
                    suggestion = EnhancementSuggestion(
                        suggestion_type=SuggestionType.QUALITY_IMPROVEMENT,
                        original_text=prompt,
                        enhanced_text=f"{prompt}, {enhancer['pattern']}",
                        confidence=confidence,
                        reasoning=f"Adding '{enhancer['pattern']}' to improve overall quality",
                        impact_score=0.8,
                        pattern_source="quality_enhancers"
                    )
                    suggestions.append(suggestion)
        
        # Resolution enhancement
        if not analysis.technical_parameters.resolution:
            resolution_enhancers = self.quality_enhancers['resolution_enhancers']
            best_resolution = max(resolution_enhancers, key=lambda x: x['weight'])
            
            suggestion = EnhancementSuggestion(
                suggestion_type=SuggestionType.TECHNICAL_OPTIMIZATION,
                original_text=prompt,
                enhanced_text=f"{prompt}, {best_resolution['pattern']}",
                confidence=best_resolution['weight'],
                reasoning=f"Adding resolution specification for better quality",
                impact_score=0.7,
                pattern_source="resolution_enhancers"
            )
            suggestions.append(suggestion)
        
        return suggestions
    
    async def _generate_style_suggestions(self, prompt: str,
                                        analysis: PromptAnalysisResult,
                                        strategy: EnhancementStrategy) -> List[EnhancementSuggestion]:
        """Generate style enhancement suggestions."""
        suggestions = []
        
        if analysis.artistic_style in self.style_enhancers:
            style_patterns = self.style_enhancers[analysis.artistic_style]
            
            for pattern in style_patterns[:2]:  # Limit to top 2
                if pattern['pattern'].lower() not in prompt.lower():
                    confidence = pattern['weight'] * self.feedback_weights['style_consistency']
                    
                    suggestion = EnhancementSuggestion(
                        suggestion_type=SuggestionType.STYLE_ENHANCEMENT,
                        original_text=prompt,
                        enhanced_text=f"{prompt}, {pattern['pattern']}",
                        confidence=confidence,
                        reasoning=f"Enhancing {analysis.artistic_style.value} style with {pattern['impact']} improvement",
                        impact_score=0.75,
                        pattern_source=f"style_enhancers_{analysis.artistic_style.value}"
                    )
                    suggestions.append(suggestion)
        
        return suggestions
    
    async def _generate_technical_suggestions(self, prompt: str,
                                            analysis: PromptAnalysisResult,
                                            strategy: EnhancementStrategy) -> List[EnhancementSuggestion]:
        """Generate technical optimization suggestions."""
        suggestions = []
        
        # Lighting suggestions
        if not analysis.technical_parameters.lighting:
            lighting_patterns = self.technical_enhancers['lighting']
            best_lighting = max(lighting_patterns, key=lambda x: x['weight'])
            
            suggestion = EnhancementSuggestion(
                suggestion_type=SuggestionType.TECHNICAL_OPTIMIZATION,
                original_text=prompt,
                enhanced_text=f"{prompt}, {best_lighting['pattern']}",
                confidence=best_lighting['weight'],
                reasoning="Adding lighting specification for better visual quality",
                impact_score=0.8,
                pattern_source="technical_lighting"
            )
            suggestions.append(suggestion)
        
        # Camera settings suggestions
        if not analysis.technical_parameters.camera_settings:
            camera_patterns = self.technical_enhancers['camera']
            
            # Select based on content type
            if analysis.content_type == ContentType.PORTRAIT:
                suitable_camera = next((p for p in camera_patterns if 'shallow' in p['pattern']), camera_patterns[0])
            elif analysis.content_type == ContentType.LANDSCAPE:
                suitable_camera = next((p for p in camera_patterns if 'wide' in p['pattern']), camera_patterns[0])
            else:
                suitable_camera = camera_patterns[0]
            
            suggestion = EnhancementSuggestion(
                suggestion_type=SuggestionType.TECHNICAL_OPTIMIZATION,
                original_text=prompt,
                enhanced_text=f"{prompt}, {suitable_camera['pattern']}",
                confidence=suitable_camera['weight'],
                reasoning=f"Adding camera setting suitable for {analysis.content_type.value} content",
                impact_score=0.7,
                pattern_source="technical_camera"
            )
            suggestions.append(suggestion)
        
        return suggestions
    
    async def _generate_context_suggestions(self, prompt: str,
                                          analysis: PromptAnalysisResult,
                                          context: Dict[str, Any],
                                          strategy: EnhancementStrategy) -> List[EnhancementSuggestion]:
        """Generate context-aware suggestions."""
        suggestions = []
        
        for context_key, context_value in context.items():
            if context_key in self.context_rules and context_value in self.context_rules[context_key]:
                context_patterns = self.context_rules[context_key][context_value]
                
                for pattern in context_patterns[:2]:  # Limit to 2 per context
                    if pattern.lower() not in prompt.lower():
                        suggestion = EnhancementSuggestion(
                            suggestion_type=SuggestionType.CONTEXT_ENRICHMENT,
                            original_text=prompt,
                            enhanced_text=f"{prompt}, {pattern}",
                            confidence=0.85,
                            reasoning=f"Adding context-appropriate element for {context_key}: {context_value}",
                            impact_score=0.7,
                            pattern_source=f"context_{context_key}"
                        )
                        suggestions.append(suggestion)
        
        return suggestions
    
    async def _generate_personal_suggestions(self, prompt: str,
                                           analysis: PromptAnalysisResult,
                                           user_preferences: Dict[str, Any],
                                           strategy: EnhancementStrategy) -> List[EnhancementSuggestion]:
        """Generate personalized suggestions based on user preferences."""
        suggestions = []
        
        # This would use learned user preferences from their generation history
        preferred_styles = user_preferences.get('preferred_styles', [])
        preferred_quality_level = user_preferences.get('quality_level', 'high')
        
        if preferred_styles and analysis.artistic_style.value not in preferred_styles:
            # Suggest incorporating user's preferred style
            preferred_style = preferred_styles[0]
            suggestion = EnhancementSuggestion(
                suggestion_type=SuggestionType.STYLE_ENHANCEMENT,
                original_text=prompt,
                enhanced_text=f"{prompt}, in {preferred_style} style",
                confidence=0.8,
                reasoning=f"Incorporating your preferred {preferred_style} style",
                impact_score=0.75,
                pattern_source="user_preferences"
            )
            suggestions.append(suggestion)
        
        return suggestions
    
    async def _generate_template_suggestions(self, prompt: str,
                                           analysis: PromptAnalysisResult,
                                           strategy: EnhancementStrategy) -> List[EnhancementSuggestion]:
        """Generate suggestions based on successful templates."""
        suggestions = []
        
        # Find similar successful templates
        similar_templates = self.template_manager.search_templates(
            prompt, limit=3
        )
        
        for template in similar_templates:
            if template.success_rate > 0.8:  # Only high-performing templates
                # Extract patterns from successful template
                template_patterns = self._extract_patterns_from_template(template.template)
                
                for pattern in template_patterns[:2]:  # Limit to 2 patterns per template
                    if pattern.lower() not in prompt.lower():
                        suggestion = EnhancementSuggestion(
                            suggestion_type=SuggestionType.QUALITY_IMPROVEMENT,
                            original_text=prompt,
                            enhanced_text=f"{prompt}, {pattern}",
                            confidence=template.success_rate * 0.9,
                            reasoning=f"Pattern from high-performing template '{template.name}'",
                            impact_score=0.8,
                            pattern_source=f"template_{template.id}"
                        )
                        suggestions.append(suggestion)
        
        return suggestions
    
    def _extract_patterns_from_template(self, template_text: str) -> List[str]:
        """Extract useful patterns from a successful template."""
        # Simple pattern extraction - could be more sophisticated
        patterns = []
        
        # Split by commas and extract quality/style modifiers
        parts = [part.strip() for part in template_text.split(',')]
        
        quality_keywords = ['high quality', 'detailed', 'professional', 'masterpiece', 'award winning']
        style_keywords = ['cinematic', 'artistic', 'photorealistic', 'dramatic']
        
        for part in parts:
            if any(keyword in part.lower() for keyword in quality_keywords + style_keywords):
                patterns.append(part)
        
        return patterns[:3]  # Limit to 3 patterns
    
    def _filter_and_rank_suggestions(self, suggestions: List[EnhancementSuggestion],
                                   strategy: EnhancementStrategy,
                                   analysis: PromptAnalysisResult) -> List[EnhancementSuggestion]:
        """Filter and rank suggestions based on strategy and analysis."""
        # Remove duplicates
        unique_suggestions = []
        seen_patterns = set()
        
        for suggestion in suggestions:
            pattern_key = suggestion.enhanced_text.split(', ')[-1].lower()
            if pattern_key not in seen_patterns:
                unique_suggestions.append(suggestion)
                seen_patterns.add(pattern_key)
        
        # Sort by confidence and impact
        unique_suggestions.sort(key=lambda x: (x.confidence * x.impact_score), reverse=True)
        
        # Apply strategy-based filtering
        if strategy == EnhancementStrategy.CONSERVATIVE:
            # Only high-confidence, low-impact suggestions
            filtered = [s for s in unique_suggestions if s.confidence > 0.8 and s.impact_score < 0.8]
            return filtered[:3]
        elif strategy == EnhancementStrategy.MODERATE:
            # Balanced approach
            return unique_suggestions[:5]
        else:  # AGGRESSIVE
            # All suggestions, prioritize high impact
            unique_suggestions.sort(key=lambda x: x.impact_score, reverse=True)
            return unique_suggestions[:7]
    
    async def _apply_suggestions(self, original_prompt: str,
                               suggestions: List[EnhancementSuggestion],
                               strategy: EnhancementStrategy) -> str:
        """Apply suggestions to create enhanced prompt."""
        enhanced_prompt = original_prompt
        
        # Apply suggestions in order of confidence
        applied_count = 0
        max_applications = {
            EnhancementStrategy.CONSERVATIVE: 2,
            EnhancementStrategy.MODERATE: 3,
            EnhancementStrategy.AGGRESSIVE: 5
        }
        
        for suggestion in suggestions:
            if applied_count >= max_applications[strategy]:
                break
            
            # Extract the enhancement part
            enhancement = suggestion.enhanced_text.replace(original_prompt, '').strip(', ')
            
            if enhancement and enhancement.lower() not in enhanced_prompt.lower():
                enhanced_prompt += f", {enhancement}"
                applied_count += 1
        
        return enhanced_prompt
    
    async def _calculate_improvement_score(self, analysis: PromptAnalysisResult,
                                         suggestions: List[EnhancementSuggestion]) -> float:
        """Calculate expected improvement score."""
        base_quality = analysis.quality_metrics.overall_score
        
        # Calculate potential improvement from suggestions
        improvement_potential = sum(s.impact_score * s.confidence for s in suggestions)
        improvement_potential = min(improvement_potential, 1.0)  # Cap at 1.0
        
        # Expected improvement
        expected_improvement = improvement_potential * (1 - base_quality)
        
        return min(base_quality + expected_improvement, 1.0)
    
    def _calculate_overall_confidence(self, suggestions: List[EnhancementSuggestion]) -> float:
        """Calculate overall confidence in the enhancement."""
        if not suggestions:
            return 0.5
        
        # Weighted average of suggestion confidences
        total_weight = sum(s.impact_score for s in suggestions)
        if total_weight == 0:
            return 0.5
        
        weighted_confidence = sum(s.confidence * s.impact_score for s in suggestions) / total_weight
        return min(weighted_confidence, 1.0)
    
    async def _get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences from usage history."""
        # Query user's generation history to learn preferences
        recent_usage = self.template_manager.db.query(VisualPromptUsageLog).filter(
            VisualPromptUsageLog.user_id == user_id,
            VisualPromptUsageLog.created_at >= datetime.utcnow() - timedelta(days=30),
            VisualPromptUsageLog.success == True
        ).limit(50).all()
        
        if not recent_usage:
            return {}
        
        # Analyze patterns in successful generations
        style_counter = Counter()
        quality_preferences = []
        
        for usage in recent_usage:
            # Extract style preferences
            prompt_lower = usage.prompt_text.lower()
            for style in ArtisticStyle:
                if style.value in prompt_lower:
                    style_counter[style.value] += 1
            
            # Track quality preferences
            if usage.quality_score:
                quality_preferences.append(usage.quality_score)
        
        preferred_styles = [style for style, count in style_counter.most_common(3)]
        avg_quality = np.mean(quality_preferences) if quality_preferences else 0.7
        
        return {
            'preferred_styles': preferred_styles,
            'quality_level': 'high' if avg_quality > 0.8 else 'medium',
            'usage_count': len(recent_usage)
        }
    
    async def _log_enhancement(self, result: PromptEnhancementResult, user_id: Optional[str]):
        """Log enhancement for learning and analytics."""
        try:
            # Store enhancement suggestion for future learning
            suggestion_record = VisualPromptOptimizationSuggestion(
                original_prompt=result.original_prompt,
                suggested_prompt=result.enhanced_prompt,
                suggestion_type=result.strategy_used.value,
                confidence_score=result.overall_confidence,
                reasoning=f"Applied {len(result.suggestions)} suggestions with {result.strategy_used.value} strategy"
            )
            
            self.template_manager.db.add(suggestion_record)
            self.template_manager.db.commit()
            
            # Add to enhancement history for performance tracking
            self.enhancement_history.append({
                'timestamp': result.timestamp,
                'improvement_score': result.improvement_score,
                'confidence': result.overall_confidence,
                'suggestions_count': len(result.suggestions),
                'strategy': result.strategy_used.value
            })
            
            # Keep only recent history
            if len(self.enhancement_history) > 1000:
                self.enhancement_history = self.enhancement_history[-500:]
                
        except Exception as e:
            # Don't fail enhancement if logging fails
            print(f"Warning: Failed to log enhancement: {e}")
    
    async def learn_from_feedback(self, enhancement_id: int, feedback: str,
                                quality_score: Optional[float] = None,
                                user_rating: Optional[int] = None) -> None:
        """Learn from user feedback to improve future suggestions."""
        try:
            # Update the suggestion record with feedback
            suggestion = self.template_manager.db.query(VisualPromptOptimizationSuggestion).filter(
                VisualPromptOptimizationSuggestion.id == enhancement_id
            ).first()
            
            if suggestion:
                suggestion.user_feedback = feedback
                suggestion.applied = feedback in ['accepted', 'modified']
                
                # Adjust feedback weights based on user response
                if feedback == 'accepted' and quality_score and quality_score > 0.8:
                    # Positive feedback - increase weights for similar patterns
                    self._adjust_feedback_weights(suggestion.suggestion_type, 0.05)
                elif feedback == 'rejected':
                    # Negative feedback - decrease weights
                    self._adjust_feedback_weights(suggestion.suggestion_type, -0.05)
                
                self.template_manager.db.commit()
                
        except Exception as e:
            print(f"Warning: Failed to process feedback: {e}")
    
    def _adjust_feedback_weights(self, suggestion_type: str, adjustment: float):
        """Adjust feedback weights based on user response."""
        # Map suggestion types to feedback weight keys
        type_mapping = {
            'quality_improvement': 'quality_modifiers',
            'style_enhancement': 'style_consistency',
            'technical_optimization': 'technical_accuracy',
            'context_enrichment': 'user_preference',
            'composition_improvement': 'technical_accuracy',
            'specificity_boost': 'quality_modifiers'
        }
        
        weight_key = type_mapping.get(suggestion_type, 'quality_modifiers')
        if weight_key in self.feedback_weights:
            new_weight = self.feedback_weights[weight_key] + adjustment
            self.feedback_weights[weight_key] = max(0.1, min(1.0, new_weight))
    
    async def get_enhancement_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get analytics on enhancement performance."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get recent enhancements
        recent_enhancements = [
            h for h in self.enhancement_history 
            if datetime.fromisoformat(h['timestamp']) >= cutoff_date
        ]
        
        if not recent_enhancements:
            return {'message': 'No enhancement data available for the specified period'}
        
        # Calculate metrics
        avg_improvement = np.mean([h['improvement_score'] for h in recent_enhancements])
        avg_confidence = np.mean([h['confidence'] for h in recent_enhancements])
        avg_suggestions = np.mean([h['suggestions_count'] for h in recent_enhancements])
        
        # Strategy distribution
        strategy_counts = Counter(h['strategy'] for h in recent_enhancements)
        
        # Get feedback statistics
        feedback_stats = self.template_manager.db.query(
            VisualPromptOptimizationSuggestion.user_feedback,
            func.count(VisualPromptOptimizationSuggestion.id).label('count')
        ).filter(
            VisualPromptOptimizationSuggestion.created_at >= cutoff_date,
            VisualPromptOptimizationSuggestion.user_feedback.isnot(None)
        ).group_by(VisualPromptOptimizationSuggestion.user_feedback).all()
        
        return {
            'period_days': days,
            'total_enhancements': len(recent_enhancements),
            'average_improvement_score': avg_improvement,
            'average_confidence': avg_confidence,
            'average_suggestions_per_enhancement': avg_suggestions,
            'strategy_distribution': dict(strategy_counts),
            'feedback_statistics': {stat.user_feedback: stat.count for stat in feedback_stats},
            'current_feedback_weights': self.feedback_weights
        }
    
    async def batch_enhance_prompts(self, prompts: List[str], 
                                  strategy: EnhancementStrategy = EnhancementStrategy.MODERATE) -> List[PromptEnhancementResult]:
        """Enhance multiple prompts in batch for efficiency."""
        tasks = [self.enhance_prompt(prompt, strategy=strategy) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
    def export_enhancement_result(self, result: PromptEnhancementResult) -> Dict[str, Any]:
        """Export enhancement result to dictionary format."""
        return asdict(result)
    
    def get_enhancement_summary(self, result: PromptEnhancementResult) -> str:
        """Get a human-readable summary of the enhancement."""
        summary_parts = [
            f"Prompt Enhancement Summary:",
            f"- Strategy: {result.strategy_used.value}",
            f"- Suggestions Applied: {len(result.suggestions)}",
            f"- Improvement Score: {result.improvement_score:.2f}",
            f"- Confidence: {result.overall_confidence:.2f}",
            f"",
            f"Original: {result.original_prompt}",
            f"Enhanced: {result.enhanced_prompt}",
            f"",
            f"Key Improvements:"
        ]
        
        for i, suggestion in enumerate(result.suggestions[:3], 1):
            summary_parts.append(f"  {i}. {suggestion.reasoning} (confidence: {suggestion.confidence:.2f})")
        
        return "\n".join(summary_parts)