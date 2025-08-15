"""
Presentation Design Engine for Board Presentations

This engine creates board-appropriate presentation materials with format optimization
and quality assessment capabilities.
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime

from ..models.board_presentation_models import (
    BoardPresentation, PresentationSlide, ContentSection, PresentationTemplate,
    BoardMemberProfile, PresentationFormat, DesignPreferences, QualityMetrics
)
from ..models.board_dynamics_models import Board


class PresentationDesigner:
    """Core presentation design component"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.design_templates = self._load_design_templates()
        self.content_guidelines = self._load_content_guidelines()
    
    def create_board_presentation(
        self, 
        content: Dict[str, Any], 
        board: Board,
        format_type: PresentationFormat = PresentationFormat.STRATEGIC_OVERVIEW
    ) -> BoardPresentation:
        """Create board-appropriate presentation materials"""
        try:
            # Select optimal template
            template = self._select_optimal_template(board, format_type)
            
            # Generate slides based on content and template
            slides = self._generate_slides(content, template, board)
            
            # Create executive summary
            executive_summary = self._create_executive_summary(content, board)
            
            # Extract key messages
            key_messages = self._extract_key_messages(content, board)
            
            # Define success metrics
            success_metrics = self._define_success_metrics(format_type)
            
            presentation = BoardPresentation(
                id=f"pres_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                title=content.get('title', 'Board Presentation'),
                board_id=board.id,
                presenter_id=content.get('presenter_id', 'scrollintel'),
                presentation_date=content.get('date', datetime.now()),
                format_type=format_type,
                slides=slides,
                executive_summary=executive_summary,
                key_messages=key_messages,
                success_metrics=success_metrics
            )
            
            self.logger.info(f"Created board presentation: {presentation.id}")
            return presentation
            
        except Exception as e:
            self.logger.error(f"Error creating board presentation: {str(e)}")
            raise
    
    def _select_optimal_template(self, board: Board, format_type: PresentationFormat) -> PresentationTemplate:
        """Select the most appropriate template for the board"""
        # Analyze board composition and preferences
        board_preferences = self._analyze_board_preferences(board)
        
        # Find matching template
        for template in self.design_templates:
            if (template.format_type == format_type and 
                self._template_matches_preferences(template, board_preferences)):
                return template
        
        # Return default template if no match
        return self._get_default_template(format_type)
    
    def _generate_slides(
        self, 
        content: Dict[str, Any], 
        template: PresentationTemplate, 
        board: Board
    ) -> List[PresentationSlide]:
        """Generate slides based on content and template"""
        slides = []
        
        for i, section_name in enumerate(template.slide_structure):
            if section_name in content:
                slide = self._create_slide(
                    slide_number=i + 1,
                    section_name=section_name,
                    content=content[section_name],
                    template=template,
                    board=board
                )
                slides.append(slide)
        
        return slides
    
    def _create_slide(
        self,
        slide_number: int,
        section_name: str,
        content: Any,
        template: PresentationTemplate,
        board: Board
    ) -> PresentationSlide:
        """Create individual presentation slide"""
        # Create content sections
        sections = self._create_content_sections(content, board)
        
        # Apply design elements
        design_elements = self._apply_design_elements(template, section_name)
        
        # Generate speaker notes
        speaker_notes = self._generate_speaker_notes(content, board)
        
        # Estimate duration
        duration = self._estimate_slide_duration(sections)
        
        # Identify interaction points
        interaction_points = self._identify_interaction_points(content, board)
        
        return PresentationSlide(
            id=f"slide_{slide_number}",
            slide_number=slide_number,
            title=section_name.replace('_', ' ').title(),
            sections=sections,
            design_elements=design_elements,
            speaker_notes=speaker_notes,
            estimated_duration=duration,
            interaction_points=interaction_points
        )
    
    def _create_content_sections(self, content: Any, board: Board) -> List[ContentSection]:
        """Create content sections for slide"""
        sections = []
        
        if isinstance(content, dict):
            for key, value in content.items():
                section = ContentSection(
                    id=f"section_{key}",
                    title=key.replace('_', ' ').title(),
                    content_type=self._determine_content_type(value),
                    content=value,
                    importance_level=self._assess_importance(key, value),
                    estimated_time=self._estimate_section_time(value),
                    board_member_relevance=self._calculate_relevance(key, value, board)
                )
                sections.append(section)
        else:
            section = ContentSection(
                id="main_content",
                title="Main Content",
                content_type=self._determine_content_type(content),
                content=content,
                importance_level="important",
                estimated_time=30,
                board_member_relevance={}
            )
            sections.append(section)
        
        return sections
    
    def _load_design_templates(self) -> List[PresentationTemplate]:
        """Load presentation design templates"""
        return [
            PresentationTemplate(
                id="executive_summary_template",
                name="Executive Summary Template",
                format_type=PresentationFormat.EXECUTIVE_SUMMARY,
                target_audience=[],
                slide_structure=["executive_summary", "key_points", "recommendations", "next_steps"],
                design_guidelines={
                    "color_scheme": "professional_blue",
                    "font_size": "large",
                    "layout": "minimal"
                },
                content_guidelines={
                    "max_bullets": "5",
                    "max_words_per_slide": "50"
                },
                timing_recommendations={
                    "total_time": 900,  # 15 minutes
                    "slide_time": 180   # 3 minutes per slide
                }
            ),
            PresentationTemplate(
                id="strategic_overview_template",
                name="Strategic Overview Template",
                format_type=PresentationFormat.STRATEGIC_OVERVIEW,
                target_audience=[],
                slide_structure=["overview", "strategic_context", "key_initiatives", "performance", "outlook"],
                design_guidelines={
                    "color_scheme": "corporate_gray",
                    "font_size": "medium",
                    "layout": "structured"
                },
                content_guidelines={
                    "max_bullets": "7",
                    "max_words_per_slide": "75"
                },
                timing_recommendations={
                    "total_time": 1800,  # 30 minutes
                    "slide_time": 360    # 6 minutes per slide
                }
            )
        ]
    
    def _load_content_guidelines(self) -> Dict[str, Any]:
        """Load content creation guidelines"""
        return {
            "executive_language": {
                "tone": "confident_professional",
                "complexity": "high_level",
                "jargon": "minimal"
            },
            "visual_hierarchy": {
                "title_prominence": "high",
                "key_points_emphasis": "medium",
                "supporting_details": "low"
            },
            "engagement_principles": {
                "storytelling": True,
                "data_visualization": True,
                "interaction_points": True
            }
        }
    
    def _analyze_board_preferences(self, board: Board) -> Dict[str, Any]:
        """Analyze board member preferences for presentation customization"""
        preferences = {
            "detail_level": "medium",
            "visual_preference": True,
            "interaction_style": "formal",
            "time_constraints": 30  # minutes
        }
        
        # Analyze board composition if available
        if hasattr(board, 'members'):
            detail_levels = []
            visual_prefs = []
            
            for member in board.members:
                if hasattr(member, 'detail_preference'):
                    detail_levels.append(member.detail_preference)
                if hasattr(member, 'visual_preference'):
                    visual_prefs.append(member.visual_preference)
            
            if detail_levels:
                preferences["detail_level"] = max(set(detail_levels), key=detail_levels.count)
            if visual_prefs:
                preferences["visual_preference"] = sum(visual_prefs) / len(visual_prefs) > 0.5
        
        return preferences
    
    def _template_matches_preferences(self, template: PresentationTemplate, preferences: Dict[str, Any]) -> bool:
        """Check if template matches board preferences"""
        # Simple matching logic - can be enhanced
        return True
    
    def _get_default_template(self, format_type: PresentationFormat) -> PresentationTemplate:
        """Get default template for format type"""
        for template in self.design_templates:
            if template.format_type == format_type:
                return template
        
        # Return first template as fallback
        return self.design_templates[0] if self.design_templates else None
    
    def _create_executive_summary(self, content: Dict[str, Any], board: Board) -> str:
        """Create executive summary for presentation"""
        summary_points = []
        
        # Extract key information
        if 'overview' in content:
            summary_points.append(f"Overview: {str(content['overview'])[:100]}...")
        
        if 'key_metrics' in content:
            summary_points.append(f"Key Metrics: {str(content['key_metrics'])[:100]}...")
        
        if 'recommendations' in content:
            summary_points.append(f"Recommendations: {str(content['recommendations'])[:100]}...")
        
        return " | ".join(summary_points)
    
    def _extract_key_messages(self, content: Dict[str, Any], board: Board) -> List[str]:
        """Extract key messages from content"""
        messages = []
        
        # Look for explicit key messages
        if 'key_messages' in content:
            if isinstance(content['key_messages'], list):
                messages.extend(content['key_messages'])
            else:
                messages.append(str(content['key_messages']))
        
        # Extract from other sections
        priority_sections = ['recommendations', 'conclusions', 'next_steps']
        for section in priority_sections:
            if section in content:
                messages.append(f"{section.title()}: {str(content[section])[:50]}...")
        
        return messages[:5]  # Limit to 5 key messages
    
    def _define_success_metrics(self, format_type: PresentationFormat) -> List[str]:
        """Define success metrics for presentation"""
        base_metrics = [
            "Board engagement level",
            "Question quality and depth",
            "Decision clarity achieved",
            "Time efficiency maintained"
        ]
        
        format_specific = {
            PresentationFormat.EXECUTIVE_SUMMARY: ["Key points comprehension"],
            PresentationFormat.STRATEGIC_OVERVIEW: ["Strategic alignment achieved"],
            PresentationFormat.FINANCIAL_REPORT: ["Financial clarity provided"],
            PresentationFormat.RISK_ASSESSMENT: ["Risk understanding demonstrated"]
        }
        
        return base_metrics + format_specific.get(format_type, [])
    
    def _determine_content_type(self, content: Any) -> str:
        """Determine content type for visualization"""
        if isinstance(content, (int, float)):
            return "metric"
        elif isinstance(content, list) and len(content) > 0:
            if isinstance(content[0], (int, float)):
                return "chart"
            else:
                return "list"
        elif isinstance(content, dict):
            return "table"
        else:
            return "text"
    
    def _assess_importance(self, key: str, value: Any) -> str:
        """Assess importance level of content"""
        critical_keywords = ['revenue', 'profit', 'risk', 'strategic', 'critical']
        important_keywords = ['performance', 'growth', 'market', 'competitive']
        
        key_lower = key.lower()
        
        if any(keyword in key_lower for keyword in critical_keywords):
            return "critical"
        elif any(keyword in key_lower for keyword in important_keywords):
            return "important"
        else:
            return "supporting"
    
    def _estimate_section_time(self, content: Any) -> int:
        """Estimate time needed for content section"""
        if isinstance(content, str):
            words = len(content.split())
            return max(15, words * 2)  # 2 seconds per word, minimum 15 seconds
        elif isinstance(content, (list, dict)):
            return 30  # 30 seconds for structured content
        else:
            return 20  # 20 seconds for simple content
    
    def _calculate_relevance(self, key: str, value: Any, board: Board) -> Dict[str, float]:
        """Calculate content relevance for board members"""
        # Simplified relevance calculation
        base_relevance = 0.7
        
        relevance_map = {}
        if hasattr(board, 'members'):
            for member in board.members:
                member_id = getattr(member, 'id', 'unknown')
                relevance_map[member_id] = base_relevance
        
        return relevance_map
    
    def _apply_design_elements(self, template: PresentationTemplate, section_name: str) -> Dict[str, Any]:
        """Apply design elements based on template"""
        return {
            "color_scheme": template.design_guidelines.get("color_scheme", "default"),
            "font_family": template.design_guidelines.get("font_family", "Arial"),
            "layout": template.design_guidelines.get("layout", "standard"),
            "emphasis": "high" if section_name in ["overview", "recommendations"] else "medium"
        }
    
    def _generate_speaker_notes(self, content: Any, board: Board) -> str:
        """Generate speaker notes for slide"""
        notes = []
        
        notes.append("Key talking points:")
        if isinstance(content, dict):
            for key, value in list(content.items())[:3]:  # Top 3 points
                notes.append(f"- {key}: {str(value)[:50]}...")
        
        notes.append("\nBoard considerations:")
        notes.append("- Maintain eye contact with all members")
        notes.append("- Pause for questions after key points")
        notes.append("- Be prepared to dive deeper into details")
        
        return "\n".join(notes)
    
    def _estimate_slide_duration(self, sections: List[ContentSection]) -> int:
        """Estimate slide presentation duration"""
        total_time = 0
        for section in sections:
            total_time += section.estimated_time
        
        # Add buffer time for transitions and questions
        return int(total_time * 1.2)
    
    def _identify_interaction_points(self, content: Any, board: Board) -> List[str]:
        """Identify potential interaction points in presentation"""
        interaction_points = []
        
        if isinstance(content, dict):
            if 'questions' in content:
                interaction_points.append("Q&A opportunity")
            if 'recommendations' in content:
                interaction_points.append("Decision point")
            if 'risks' in content:
                interaction_points.append("Risk discussion")
        
        # Always include standard interaction points
        interaction_points.extend([
            "Pause for clarification",
            "Board member input welcome"
        ])
        
        return interaction_points


class PresentationFormatOptimizer:
    """Optimizes presentation format for board preferences"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def optimize_for_board(self, presentation: BoardPresentation, board: Board) -> BoardPresentation:
        """Optimize presentation format for specific board preferences"""
        try:
            # Analyze board preferences
            preferences = self._analyze_board_preferences(board)
            
            # Optimize slide order
            presentation.slides = self._optimize_slide_order(presentation.slides, preferences)
            
            # Adjust content density
            presentation.slides = self._adjust_content_density(presentation.slides, preferences)
            
            # Optimize timing
            presentation.slides = self._optimize_timing(presentation.slides, preferences)
            
            self.logger.info(f"Optimized presentation format for board: {board.id}")
            return presentation
            
        except Exception as e:
            self.logger.error(f"Error optimizing presentation format: {str(e)}")
            return presentation
    
    def _analyze_board_preferences(self, board: Board) -> Dict[str, Any]:
        """Analyze board preferences for optimization"""
        return {
            "attention_span": 20,  # minutes
            "detail_preference": "medium",
            "interaction_frequency": "moderate",
            "visual_emphasis": True
        }
    
    def _optimize_slide_order(self, slides: List[PresentationSlide], preferences: Dict[str, Any]) -> List[PresentationSlide]:
        """Optimize slide order based on preferences"""
        # Sort by importance and engagement potential
        return sorted(slides, key=lambda s: self._calculate_slide_priority(s, preferences), reverse=True)
    
    def _adjust_content_density(self, slides: List[PresentationSlide], preferences: Dict[str, Any]) -> List[PresentationSlide]:
        """Adjust content density based on preferences"""
        target_density = preferences.get("detail_preference", "medium")
        
        for slide in slides:
            if target_density == "low":
                # Reduce content sections
                slide.sections = slide.sections[:3]
            elif target_density == "high":
                # Keep all sections
                pass
            # Medium is default
        
        return slides
    
    def _optimize_timing(self, slides: List[PresentationSlide], preferences: Dict[str, Any]) -> List[PresentationSlide]:
        """Optimize slide timing based on preferences"""
        attention_span = preferences.get("attention_span", 20) * 60  # Convert to seconds
        total_time = sum(slide.estimated_duration for slide in slides)
        
        if total_time > attention_span:
            # Reduce time per slide proportionally
            reduction_factor = attention_span / total_time
            for slide in slides:
                slide.estimated_duration = int(slide.estimated_duration * reduction_factor)
        
        return slides
    
    def _calculate_slide_priority(self, slide: PresentationSlide, preferences: Dict[str, Any]) -> float:
        """Calculate slide priority for ordering"""
        priority = 0.0
        
        # Importance-based priority
        critical_sections = sum(1 for s in slide.sections if s.importance_level == "critical")
        priority += critical_sections * 3
        
        important_sections = sum(1 for s in slide.sections if s.importance_level == "important")
        priority += important_sections * 2
        
        # Interaction potential
        priority += len(slide.interaction_points) * 0.5
        
        return priority


class PresentationQualityAssessor:
    """Assesses and enhances presentation quality"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def assess_quality(self, presentation: BoardPresentation, board: Board) -> QualityMetrics:
        """Assess presentation quality and provide metrics"""
        try:
            clarity_score = self._assess_clarity(presentation)
            relevance_score = self._assess_relevance(presentation, board)
            engagement_score = self._assess_engagement_potential(presentation)
            professional_score = self._assess_professionalism(presentation)
            time_efficiency_score = self._assess_time_efficiency(presentation)
            
            overall_score = (
                clarity_score * 0.25 +
                relevance_score * 0.25 +
                engagement_score * 0.2 +
                professional_score * 0.15 +
                time_efficiency_score * 0.15
            )
            
            improvement_suggestions = self._generate_improvement_suggestions(
                clarity_score, relevance_score, engagement_score, 
                professional_score, time_efficiency_score
            )
            
            metrics = QualityMetrics(
                clarity_score=clarity_score,
                relevance_score=relevance_score,
                engagement_score=engagement_score,
                professional_score=professional_score,
                time_efficiency_score=time_efficiency_score,
                overall_score=overall_score,
                improvement_suggestions=improvement_suggestions
            )
            
            self.logger.info(f"Assessed presentation quality: {overall_score:.2f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error assessing presentation quality: {str(e)}")
            return QualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [])
    
    def enhance_presentation(self, presentation: BoardPresentation, metrics: QualityMetrics) -> BoardPresentation:
        """Enhance presentation based on quality assessment"""
        try:
            if metrics.clarity_score < 0.7:
                presentation = self._improve_clarity(presentation)
            
            if metrics.engagement_score < 0.7:
                presentation = self._improve_engagement(presentation)
            
            if metrics.time_efficiency_score < 0.7:
                presentation = self._improve_time_efficiency(presentation)
            
            self.logger.info(f"Enhanced presentation: {presentation.id}")
            return presentation
            
        except Exception as e:
            self.logger.error(f"Error enhancing presentation: {str(e)}")
            return presentation
    
    def _assess_clarity(self, presentation: BoardPresentation) -> float:
        """Assess presentation clarity"""
        score = 0.8  # Base score
        
        # Check for clear structure
        if len(presentation.slides) > 0:
            score += 0.1
        
        # Check for executive summary
        if presentation.executive_summary:
            score += 0.1
        
        return min(1.0, score)
    
    def _assess_relevance(self, presentation: BoardPresentation, board: Board) -> float:
        """Assess content relevance to board"""
        score = 0.7  # Base score
        
        # Check for key messages
        if presentation.key_messages:
            score += 0.2
        
        # Check for strategic content
        strategic_keywords = ['strategy', 'growth', 'performance', 'risk']
        content_text = presentation.executive_summary.lower()
        
        keyword_matches = sum(1 for keyword in strategic_keywords if keyword in content_text)
        score += (keyword_matches / len(strategic_keywords)) * 0.1
        
        return min(1.0, score)
    
    def _assess_engagement_potential(self, presentation: BoardPresentation) -> float:
        """Assess presentation engagement potential"""
        score = 0.6  # Base score
        
        # Check for interaction points
        total_interactions = sum(len(slide.interaction_points) for slide in presentation.slides)
        if total_interactions > 0:
            score += min(0.3, total_interactions * 0.05)
        
        # Check for visual elements
        visual_slides = sum(1 for slide in presentation.slides 
                          if any(section.content_type in ['chart', 'image'] for section in slide.sections))
        if visual_slides > 0:
            score += min(0.1, visual_slides * 0.02)
        
        return min(1.0, score)
    
    def _assess_professionalism(self, presentation: BoardPresentation) -> float:
        """Assess presentation professionalism"""
        score = 0.8  # Base score
        
        # Check for proper structure
        if len(presentation.slides) >= 3:
            score += 0.1
        
        # Check for success metrics
        if presentation.success_metrics:
            score += 0.1
        
        return min(1.0, score)
    
    def _assess_time_efficiency(self, presentation: BoardPresentation) -> float:
        """Assess time efficiency of presentation"""
        total_time = sum(slide.estimated_duration for slide in presentation.slides)
        optimal_time = 1800  # 30 minutes
        
        if total_time <= optimal_time:
            return 1.0
        elif total_time <= optimal_time * 1.5:
            return 0.8
        else:
            return 0.5
    
    def _generate_improvement_suggestions(self, *scores) -> List[str]:
        """Generate improvement suggestions based on scores"""
        suggestions = []
        
        clarity, relevance, engagement, professional, time_efficiency = scores
        
        if clarity < 0.7:
            suggestions.append("Improve content clarity and structure")
        if relevance < 0.7:
            suggestions.append("Increase relevance to board priorities")
        if engagement < 0.7:
            suggestions.append("Add more interaction points and visual elements")
        if professional < 0.7:
            suggestions.append("Enhance professional presentation standards")
        if time_efficiency < 0.7:
            suggestions.append("Optimize presentation timing and content density")
        
        return suggestions
    
    def _improve_clarity(self, presentation: BoardPresentation) -> BoardPresentation:
        """Improve presentation clarity"""
        # Simplify slide titles
        for slide in presentation.slides:
            if len(slide.title.split()) > 5:
                slide.title = " ".join(slide.title.split()[:5]) + "..."
        
        return presentation
    
    def _improve_engagement(self, presentation: BoardPresentation) -> BoardPresentation:
        """Improve presentation engagement"""
        # Add interaction points to slides without them
        for slide in presentation.slides:
            if not slide.interaction_points:
                slide.interaction_points = ["Pause for questions", "Board input welcome"]
        
        return presentation
    
    def _improve_time_efficiency(self, presentation: BoardPresentation) -> BoardPresentation:
        """Improve presentation time efficiency"""
        # Reduce estimated duration for long slides
        for slide in presentation.slides:
            if slide.estimated_duration > 300:  # 5 minutes
                slide.estimated_duration = 300
        
        return presentation


class BoardPresentationFramework:
    """Main framework for board presentation creation and management"""
    
    def __init__(self):
        self.designer = PresentationDesigner()
        self.optimizer = PresentationFormatOptimizer()
        self.assessor = PresentationQualityAssessor()
        self.logger = logging.getLogger(__name__)
    
    def create_optimized_presentation(
        self, 
        content: Dict[str, Any], 
        board: Board,
        format_type: PresentationFormat = PresentationFormat.STRATEGIC_OVERVIEW
    ) -> BoardPresentation:
        """Create and optimize board presentation"""
        try:
            # Create initial presentation
            presentation = self.designer.create_board_presentation(content, board, format_type)
            
            # Optimize for board preferences
            presentation = self.optimizer.optimize_for_board(presentation, board)
            
            # Assess quality
            quality_metrics = self.assessor.assess_quality(presentation, board)
            presentation.quality_score = quality_metrics.overall_score
            
            # Enhance if needed
            if quality_metrics.overall_score < 0.8:
                presentation = self.assessor.enhance_presentation(presentation, quality_metrics)
            
            self.logger.info(f"Created optimized board presentation: {presentation.id}")
            return presentation
            
        except Exception as e:
            self.logger.error(f"Error creating optimized presentation: {str(e)}")
            raise