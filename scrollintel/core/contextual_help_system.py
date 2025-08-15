"""
Contextual Help System - Smart User Guidance

This module provides contextual help and guidance based on user actions:
- Context-aware help suggestions
- Interactive tutorials and walkthroughs
- Smart error recovery guidance
- Proactive assistance based on user behavior
- Learning from user interactions

Requirements: 6.3, 6.4, 6.6
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from dataclasses import dataclass, field
import json
import re
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class HelpTrigger(Enum):
    """Help trigger types"""
    USER_REQUEST = "user_request"
    ERROR_OCCURRED = "error_occurred"
    STUCK_DETECTION = "stuck_detection"
    NEW_FEATURE = "new_feature"
    WORKFLOW_START = "workflow_start"
    CONFUSION_DETECTED = "confusion_detected"
    PROACTIVE = "proactive"


class HelpFormat(Enum):
    """Help content formats"""
    TEXT = "text"
    HTML = "html"
    MARKDOWN = "markdown"
    VIDEO = "video"
    INTERACTIVE = "interactive"
    STEP_BY_STEP = "step_by_step"


class UserExpertiseLevel(Enum):
    """User expertise levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class HelpContent:
    """Help content structure"""
    content_id: str
    title: str
    content: str
    format: HelpFormat
    expertise_level: UserExpertiseLevel
    tags: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    related_content: List[str] = field(default_factory=list)
    estimated_time: Optional[int] = None  # minutes
    interactive_elements: List[Dict[str, Any]] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class UserContext:
    """User context for help personalization"""
    user_id: str
    current_page: str
    current_action: Optional[str] = None
    workflow_step: Optional[str] = None
    recent_actions: List[str] = field(default_factory=list)
    error_history: List[Dict[str, Any]] = field(default_factory=list)
    expertise_level: UserExpertiseLevel = UserExpertiseLevel.BEGINNER
    preferences: Dict[str, Any] = field(default_factory=dict)
    completed_tutorials: Set[str] = field(default_factory=set)
    help_history: List[str] = field(default_factory=list)
    session_start: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HelpSuggestion:
    """Help suggestion with relevance scoring"""
    suggestion_id: str
    content: HelpContent
    relevance_score: float
    trigger: HelpTrigger
    context_match: Dict[str, Any]
    personalization_factors: List[str] = field(default_factory=list)
    estimated_helpfulness: float = 0.8
    priority: int = 1  # 1=high, 2=medium, 3=low


@dataclass
class InteractionPattern:
    """User interaction pattern for learning"""
    pattern_id: str
    user_actions: List[str]
    context: Dict[str, Any]
    outcome: str  # success, failure, abandoned
    help_requested: bool
    help_content_used: List[str] = field(default_factory=list)
    duration: Optional[float] = None  # seconds
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ContextualHelpSystem:
    """
    Provides intelligent, contextual help and guidance based on user behavior,
    current context, and learned patterns
    """
    
    def __init__(self):
        self.help_content: Dict[str, HelpContent] = {}
        self.user_contexts: Dict[str, UserContext] = {}
        self.interaction_patterns: List[InteractionPattern] = []
        self.help_triggers: Dict[str, List[Callable]] = defaultdict(list)
        self.content_effectiveness: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.user_learning_progress: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Pattern recognition
        self.common_patterns: Dict[str, List[str]] = {}
        self.confusion_indicators: List[str] = [
            "repeated_same_action", "rapid_navigation", "error_sequence",
            "long_idle_time", "help_search", "undo_sequence"
        ]
        
        # Initialize default help content
        self._initialize_default_content()
    
    async def get_contextual_help(
        self,
        user_id: str,
        trigger: HelpTrigger = HelpTrigger.USER_REQUEST,
        context: Optional[Dict[str, Any]] = None,
        max_suggestions: int = 5
    ) -> List[HelpSuggestion]:
        """Get contextual help suggestions for user"""
        try:
            # Update user context
            user_context = await self._update_user_context(user_id, context)
            
            # Get relevant help content
            relevant_content = await self._find_relevant_content(user_context, trigger)
            
            # Score and rank suggestions
            suggestions = await self._score_and_rank_suggestions(
                user_context, relevant_content, trigger, max_suggestions
            )
            
            # Log help request for learning
            await self._log_help_request(user_id, trigger, context, suggestions)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error getting contextual help: {e}")
            return []
    
    async def provide_proactive_help(
        self,
        user_id: str,
        current_context: Dict[str, Any]
    ) -> Optional[HelpSuggestion]:
        """Provide proactive help based on user behavior analysis"""
        try:
            user_context = self.user_contexts.get(user_id)
            if not user_context:
                return None
            
            # Detect if user might need help
            confusion_signals = await self._detect_confusion_signals(user_context, current_context)
            
            if not confusion_signals:
                return None
            
            # Get proactive help suggestions
            suggestions = await self.get_contextual_help(
                user_id, HelpTrigger.PROACTIVE, current_context, max_suggestions=1
            )
            
            if suggestions:
                suggestion = suggestions[0]
                suggestion.personalization_factors.extend(confusion_signals)
                return suggestion
            
            return None
            
        except Exception as e:
            logger.error(f"Error providing proactive help: {e}")
            return None
    
    async def handle_error_help(
        self,
        user_id: str,
        error: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[HelpSuggestion]:
        """Provide help for error recovery"""
        try:
            # Update user context with error
            user_context = await self._update_user_context(user_id, context)
            user_context.error_history.append({
                "error": error,
                "context": context,
                "timestamp": datetime.utcnow()
            })
            
            # Get error-specific help
            error_help = await self._get_error_specific_help(error, user_context)
            
            # Get general recovery help
            recovery_help = await self.get_contextual_help(
                user_id, HelpTrigger.ERROR_OCCURRED, context
            )
            
            # Combine and prioritize
            all_suggestions = error_help + recovery_help
            return await self._deduplicate_and_prioritize(all_suggestions)
            
        except Exception as e:
            logger.error(f"Error handling error help: {e}")
            return []
    
    async def start_interactive_tutorial(
        self,
        user_id: str,
        tutorial_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Start an interactive tutorial for the user"""
        try:
            if tutorial_id not in self.help_content:
                raise ValueError(f"Tutorial {tutorial_id} not found")
            
            tutorial = self.help_content[tutorial_id]
            user_context = await self._update_user_context(user_id, context)
            
            # Create tutorial session
            session = {
                "tutorial_id": tutorial_id,
                "user_id": user_id,
                "current_step": 0,
                "total_steps": len(tutorial.interactive_elements),
                "started_at": datetime.utcnow(),
                "context": context or {},
                "progress": {}
            }
            
            # Store session (in production, use database)
            if not hasattr(self, 'tutorial_sessions'):
                self.tutorial_sessions = {}
            self.tutorial_sessions[f"{user_id}_{tutorial_id}"] = session
            
            # Get first step
            first_step = await self._get_tutorial_step(tutorial, 0, user_context)
            
            return {
                "session_id": f"{user_id}_{tutorial_id}",
                "tutorial": {
                    "id": tutorial_id,
                    "title": tutorial.title,
                    "estimated_time": tutorial.estimated_time,
                    "total_steps": len(tutorial.interactive_elements)
                },
                "current_step": first_step,
                "progress": 0
            }
            
        except Exception as e:
            logger.error(f"Error starting tutorial: {e}")
            raise
    
    async def advance_tutorial(
        self,
        session_id: str,
        step_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Advance tutorial to next step"""
        try:
            if not hasattr(self, 'tutorial_sessions') or session_id not in self.tutorial_sessions:
                raise ValueError(f"Tutorial session {session_id} not found")
            
            session = self.tutorial_sessions[session_id]
            tutorial = self.help_content[session["tutorial_id"]]
            user_context = self.user_contexts.get(session["user_id"])
            
            # Record step result
            session["progress"][session["current_step"]] = step_result
            
            # Check if step was successful
            if step_result.get("success", False):
                session["current_step"] += 1
            
            # Check if tutorial is complete
            if session["current_step"] >= len(tutorial.interactive_elements):
                return await self._complete_tutorial(session_id)
            
            # Get next step
            next_step = await self._get_tutorial_step(
                tutorial, session["current_step"], user_context
            )
            
            progress_percentage = (session["current_step"] / len(tutorial.interactive_elements)) * 100
            
            return {
                "session_id": session_id,
                "current_step": next_step,
                "progress": progress_percentage,
                "completed": False
            }
            
        except Exception as e:
            logger.error(f"Error advancing tutorial: {e}")
            raise
    
    async def add_help_content(
        self,
        content_id: str,
        title: str,
        content: str,
        format: HelpFormat,
        expertise_level: UserExpertiseLevel,
        tags: List[str] = None,
        prerequisites: List[str] = None,
        interactive_elements: List[Dict[str, Any]] = None
    ) -> None:
        """Add new help content"""
        help_content = HelpContent(
            content_id=content_id,
            title=title,
            content=content,
            format=format,
            expertise_level=expertise_level,
            tags=tags or [],
            prerequisites=prerequisites or [],
            interactive_elements=interactive_elements or []
        )
        
        self.help_content[content_id] = help_content
        logger.info(f"Added help content: {title}")
    
    async def update_user_expertise(
        self,
        user_id: str,
        expertise_level: UserExpertiseLevel
    ) -> None:
        """Update user expertise level"""
        if user_id in self.user_contexts:
            self.user_contexts[user_id].expertise_level = expertise_level
        else:
            self.user_contexts[user_id] = UserContext(
                user_id=user_id,
                current_page="",
                expertise_level=expertise_level
            )
    
    async def record_help_feedback(
        self,
        user_id: str,
        content_id: str,
        helpful: bool,
        feedback: Optional[str] = None
    ) -> None:
        """Record user feedback on help content"""
        try:
            # Update content effectiveness
            if content_id not in self.content_effectiveness:
                self.content_effectiveness[content_id] = {"helpful": 0, "total": 0}
            
            self.content_effectiveness[content_id]["total"] += 1
            if helpful:
                self.content_effectiveness[content_id]["helpful"] += 1
            
            # Store detailed feedback
            feedback_record = {
                "user_id": user_id,
                "content_id": content_id,
                "helpful": helpful,
                "feedback": feedback,
                "timestamp": datetime.utcnow()
            }
            
            # In production, store in database
            if not hasattr(self, 'feedback_records'):
                self.feedback_records = []
            self.feedback_records.append(feedback_record)
            
            logger.info(f"Recorded help feedback for content {content_id}: {'helpful' if helpful else 'not helpful'}")
            
        except Exception as e:
            logger.error(f"Error recording help feedback: {e}")
    
    async def get_help_analytics(self) -> Dict[str, Any]:
        """Get help system analytics"""
        try:
            total_content = len(self.help_content)
            total_users = len(self.user_contexts)
            
            # Content effectiveness
            content_stats = {}
            for content_id, stats in self.content_effectiveness.items():
                if stats["total"] > 0:
                    effectiveness = stats["helpful"] / stats["total"]
                    content_stats[content_id] = {
                        "effectiveness": effectiveness,
                        "total_views": stats["total"],
                        "helpful_votes": stats["helpful"]
                    }
            
            # Most requested help topics
            help_requests = Counter()
            for pattern in self.interaction_patterns:
                if pattern.help_requested:
                    for content_id in pattern.help_content_used:
                        help_requests[content_id] += 1
            
            return {
                "total_content": total_content,
                "total_users": total_users,
                "content_effectiveness": content_stats,
                "most_requested": dict(help_requests.most_common(10)),
                "average_effectiveness": sum(
                    stats["effectiveness"] for stats in content_stats.values()
                ) / len(content_stats) if content_stats else 0,
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting help analytics: {e}")
            return {}
    
    # Private methods
    
    async def _update_user_context(
        self,
        user_id: str,
        context: Optional[Dict[str, Any]]
    ) -> UserContext:
        """Update user context with new information"""
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = UserContext(
                user_id=user_id,
                current_page=context.get("page", "") if context else ""
            )
        
        user_context = self.user_contexts[user_id]
        user_context.last_activity = datetime.utcnow()
        
        if context:
            if "page" in context:
                user_context.current_page = context["page"]
            if "action" in context:
                user_context.current_action = context["action"]
                user_context.recent_actions.append(context["action"])
                # Keep only last 10 actions
                user_context.recent_actions = user_context.recent_actions[-10:]
            if "workflow_step" in context:
                user_context.workflow_step = context["workflow_step"]
        
        return user_context
    
    async def _find_relevant_content(
        self,
        user_context: UserContext,
        trigger: HelpTrigger
    ) -> List[HelpContent]:
        """Find relevant help content based on context"""
        relevant_content = []
        
        for content in self.help_content.values():
            relevance_score = await self._calculate_content_relevance(
                content, user_context, trigger
            )
            
            if relevance_score > 0.3:  # Threshold for relevance
                relevant_content.append(content)
        
        return relevant_content
    
    async def _calculate_content_relevance(
        self,
        content: HelpContent,
        user_context: UserContext,
        trigger: HelpTrigger
    ) -> float:
        """Calculate relevance score for help content"""
        score = 0.0
        
        # Page/context matching
        if user_context.current_page:
            if user_context.current_page.lower() in content.content.lower():
                score += 0.3
            if any(tag.lower() in user_context.current_page.lower() for tag in content.tags):
                score += 0.2
        
        # Action matching
        if user_context.current_action:
            if user_context.current_action.lower() in content.content.lower():
                score += 0.3
            if any(tag.lower() in user_context.current_action.lower() for tag in content.tags):
                score += 0.2
        
        # Expertise level matching
        expertise_levels = {
            UserExpertiseLevel.BEGINNER: 1,
            UserExpertiseLevel.INTERMEDIATE: 2,
            UserExpertiseLevel.ADVANCED: 3,
            UserExpertiseLevel.EXPERT: 4
        }
        
        user_level = expertise_levels[user_context.expertise_level]
        content_level = expertise_levels[content.expertise_level]
        
        if abs(user_level - content_level) <= 1:
            score += 0.2
        
        # Trigger-specific scoring
        if trigger == HelpTrigger.ERROR_OCCURRED:
            if "error" in content.tags or "troubleshoot" in content.tags:
                score += 0.3
        elif trigger == HelpTrigger.NEW_FEATURE:
            if "tutorial" in content.tags or "getting-started" in content.tags:
                score += 0.3
        
        # Content effectiveness
        if content.content_id in self.content_effectiveness:
            effectiveness = self.content_effectiveness[content.content_id]
            if effectiveness["total"] > 0:
                effectiveness_score = effectiveness["helpful"] / effectiveness["total"]
                score += effectiveness_score * 0.2
        
        return min(score, 1.0)
    
    async def _score_and_rank_suggestions(
        self,
        user_context: UserContext,
        relevant_content: List[HelpContent],
        trigger: HelpTrigger,
        max_suggestions: int
    ) -> List[HelpSuggestion]:
        """Score and rank help suggestions"""
        suggestions = []
        
        for content in relevant_content:
            relevance_score = await self._calculate_content_relevance(
                content, user_context, trigger
            )
            
            # Calculate personalization factors
            personalization_factors = []
            if content.expertise_level == user_context.expertise_level:
                personalization_factors.append("expertise_match")
            if content.content_id not in user_context.completed_tutorials:
                personalization_factors.append("not_completed")
            if any(action in content.content.lower() for action in user_context.recent_actions):
                personalization_factors.append("recent_action_match")
            
            suggestion = HelpSuggestion(
                suggestion_id=f"sugg_{content.content_id}_{int(datetime.utcnow().timestamp())}",
                content=content,
                relevance_score=relevance_score,
                trigger=trigger,
                context_match={
                    "page": user_context.current_page,
                    "action": user_context.current_action,
                    "workflow_step": user_context.workflow_step
                },
                personalization_factors=personalization_factors
            )
            
            suggestions.append(suggestion)
        
        # Sort by relevance score
        suggestions.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return suggestions[:max_suggestions]
    
    async def _detect_confusion_signals(
        self,
        user_context: UserContext,
        current_context: Dict[str, Any]
    ) -> List[str]:
        """Detect signals that user might be confused or stuck"""
        signals = []
        
        # Check for repeated actions
        if len(user_context.recent_actions) >= 3:
            last_three = user_context.recent_actions[-3:]
            if len(set(last_three)) == 1:
                signals.append("repeated_same_action")
        
        # Check for rapid navigation
        if len(user_context.recent_actions) >= 5:
            recent_time = datetime.utcnow() - timedelta(minutes=2)
            if user_context.last_activity > recent_time:
                signals.append("rapid_navigation")
        
        # Check for error sequence
        if len(user_context.error_history) >= 2:
            recent_errors = [
                err for err in user_context.error_history
                if err["timestamp"] > datetime.utcnow() - timedelta(minutes=5)
            ]
            if len(recent_errors) >= 2:
                signals.append("error_sequence")
        
        # Check for long idle time followed by activity
        if current_context.get("idle_time", 0) > 300:  # 5 minutes
            signals.append("long_idle_time")
        
        return signals
    
    async def _get_error_specific_help(
        self,
        error: Dict[str, Any],
        user_context: UserContext
    ) -> List[HelpSuggestion]:
        """Get help specific to the error that occurred"""
        error_type = error.get("type", "")
        error_message = error.get("message", "")
        
        # Find content that matches the error
        matching_content = []
        for content in self.help_content.values():
            if (error_type.lower() in content.content.lower() or
                any(word in content.content.lower() for word in error_message.lower().split()[:5])):
                matching_content.append(content)
        
        # Create suggestions
        suggestions = []
        for content in matching_content:
            suggestion = HelpSuggestion(
                suggestion_id=f"error_sugg_{content.content_id}_{int(datetime.utcnow().timestamp())}",
                content=content,
                relevance_score=0.9,  # High relevance for error-specific help
                trigger=HelpTrigger.ERROR_OCCURRED,
                context_match={"error": error},
                personalization_factors=["error_specific"]
            )
            suggestions.append(suggestion)
        
        return suggestions
    
    async def _deduplicate_and_prioritize(
        self,
        suggestions: List[HelpSuggestion]
    ) -> List[HelpSuggestion]:
        """Remove duplicates and prioritize suggestions"""
        seen_content = set()
        unique_suggestions = []
        
        # Sort by relevance score first
        suggestions.sort(key=lambda x: x.relevance_score, reverse=True)
        
        for suggestion in suggestions:
            if suggestion.content.content_id not in seen_content:
                seen_content.add(suggestion.content.content_id)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions
    
    async def _get_tutorial_step(
        self,
        tutorial: HelpContent,
        step_index: int,
        user_context: Optional[UserContext]
    ) -> Dict[str, Any]:
        """Get specific tutorial step"""
        if step_index >= len(tutorial.interactive_elements):
            return {}
        
        step = tutorial.interactive_elements[step_index]
        
        # Personalize step based on user context
        personalized_step = step.copy()
        if user_context and user_context.expertise_level == UserExpertiseLevel.BEGINNER:
            # Add more detailed instructions for beginners
            if "instructions" in personalized_step:
                personalized_step["instructions"] = f"[Detailed] {personalized_step['instructions']}"
        
        return personalized_step
    
    async def _complete_tutorial(self, session_id: str) -> Dict[str, Any]:
        """Complete tutorial session"""
        session = self.tutorial_sessions[session_id]
        user_id = session["user_id"]
        tutorial_id = session["tutorial_id"]
        
        # Mark tutorial as completed
        if user_id in self.user_contexts:
            self.user_contexts[user_id].completed_tutorials.add(tutorial_id)
        
        # Record completion
        completion_time = datetime.utcnow() - session["started_at"]
        
        # Clean up session
        del self.tutorial_sessions[session_id]
        
        return {
            "session_id": session_id,
            "completed": True,
            "completion_time": completion_time.total_seconds(),
            "progress": 100
        }
    
    async def _log_help_request(
        self,
        user_id: str,
        trigger: HelpTrigger,
        context: Optional[Dict[str, Any]],
        suggestions: List[HelpSuggestion]
    ) -> None:
        """Log help request for learning and analytics"""
        try:
            # Create interaction pattern
            pattern = InteractionPattern(
                pattern_id=f"help_{user_id}_{int(datetime.utcnow().timestamp())}",
                user_actions=[trigger.value],
                context=context or {},
                outcome="help_requested",
                help_requested=True,
                help_content_used=[s.content.content_id for s in suggestions]
            )
            
            self.interaction_patterns.append(pattern)
            
            # Keep only recent patterns (last 1000)
            if len(self.interaction_patterns) > 1000:
                self.interaction_patterns = self.interaction_patterns[-1000:]
                
        except Exception as e:
            logger.error(f"Error logging help request: {e}")
    
    def _initialize_default_content(self) -> None:
        """Initialize default help content"""
        default_content = [
            {
                "content_id": "getting_started",
                "title": "Getting Started with ScrollIntel",
                "content": "Welcome to ScrollIntel! This guide will help you get started with the platform. Learn how to upload data, create visualizations, and run analyses.",
                "format": HelpFormat.STEP_BY_STEP,
                "expertise_level": UserExpertiseLevel.BEGINNER,
                "tags": ["getting-started", "tutorial", "basics"],
                "interactive_elements": [
                    {"type": "highlight", "target": "#upload-button", "instructions": "Click here to upload your first dataset"},
                    {"type": "form", "target": "#file-input", "instructions": "Select a CSV or Excel file from your computer"},
                    {"type": "wait", "target": "#upload-progress", "instructions": "Wait for the upload to complete"}
                ]
            },
            {
                "content_id": "file_upload_help",
                "title": "File Upload Troubleshooting",
                "content": "Having trouble uploading files? Here are common solutions: Check file size (max 100MB), ensure supported format (CSV, Excel, JSON), verify file permissions.",
                "format": HelpFormat.TEXT,
                "expertise_level": UserExpertiseLevel.BEGINNER,
                "tags": ["upload", "troubleshooting", "files", "error"]
            },
            {
                "content_id": "visualization_guide",
                "title": "Creating Effective Visualizations",
                "content": "Learn how to create compelling visualizations. Choose the right chart type for your data, customize colors and labels, and add interactive elements.",
                "format": HelpFormat.HTML,
                "expertise_level": UserExpertiseLevel.INTERMEDIATE,
                "tags": ["visualization", "charts", "design"]
            },
            {
                "content_id": "analysis_basics",
                "title": "Data Analysis Fundamentals",
                "content": "Master the basics of data analysis in ScrollIntel. Learn about descriptive statistics, correlation analysis, and trend identification.",
                "format": HelpFormat.MARKDOWN,
                "expertise_level": UserExpertiseLevel.INTERMEDIATE,
                "tags": ["analysis", "statistics", "data-science"]
            }
        ]
        
        for content_data in default_content:
            content = HelpContent(**content_data)
            self.help_content[content.content_id] = content


# Global instance
contextual_help_system = ContextualHelpSystem()