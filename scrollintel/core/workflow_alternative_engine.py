"""
Alternative Workflow Suggestion Engine for ScrollIntel.
Provides intelligent workflow alternatives when primary paths fail,
with context-aware suggestions and success probability estimation.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import hashlib
import random

logger = logging.getLogger(__name__)


class WorkflowCategory(Enum):
    """Categories of workflows that can have alternatives."""
    DATA_ANALYSIS = "data_analysis"
    VISUALIZATION = "visualization"
    REPORT_GENERATION = "report_generation"
    FILE_PROCESSING = "file_processing"
    AI_INTERACTION = "ai_interaction"
    DASHBOARD_CREATION = "dashboard_creation"
    DATA_EXPORT = "data_export"
    COLLABORATION = "collaboration"
    SYSTEM_ADMINISTRATION = "system_administration"
    TROUBLESHOOTING = "troubleshooting"


class DifficultyLevel(Enum):
    """Difficulty levels for workflow alternatives."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class AlternativeType(Enum):
    """Types of workflow alternatives."""
    SIMPLIFIED = "simplified"        # Simpler version of original workflow
    MANUAL = "manual"               # Manual alternative to automated process
    WORKAROUND = "workaround"       # Work around a specific issue
    FALLBACK = "fallback"           # Fallback when primary method fails
    ENHANCED = "enhanced"           # Enhanced version with additional features
    PARALLEL = "parallel"           # Alternative parallel approach


@dataclass
class WorkflowStep:
    """Individual step in a workflow."""
    step_id: str
    title: str
    description: str
    estimated_time_minutes: int
    required_skills: List[str] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    optional: bool = False
    automation_available: bool = False
    help_resources: List[str] = field(default_factory=list)


@dataclass
class WorkflowAlternative:
    """Complete workflow alternative."""
    alternative_id: str
    name: str
    description: str
    category: WorkflowCategory
    alternative_type: AlternativeType
    difficulty: DifficultyLevel
    estimated_total_time_minutes: int
    success_probability: float  # 0.0 to 1.0
    steps: List[WorkflowStep]
    prerequisites: List[str] = field(default_factory=list)
    benefits: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    when_to_use: List[str] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)
    related_alternatives: List[str] = field(default_factory=list)
    user_feedback_score: Optional[float] = None
    usage_count: int = 0
    success_count: int = 0


@dataclass
class WorkflowContext:
    """Context for workflow alternative suggestions."""
    user_id: Optional[str] = None
    original_workflow: str = ""
    failure_reason: Optional[str] = None
    user_skill_level: DifficultyLevel = DifficultyLevel.INTERMEDIATE
    available_tools: List[str] = field(default_factory=list)
    time_constraints: Optional[int] = None  # minutes
    priority_level: str = "normal"  # low, normal, high, critical
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    system_capabilities: Dict[str, Any] = field(default_factory=dict)
    historical_successes: List[str] = field(default_factory=list)
    current_data_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SuggestionResult:
    """Result of workflow alternative suggestion."""
    alternatives: List[WorkflowAlternative]
    confidence_score: float
    reasoning: str
    personalization_applied: bool
    context_factors: List[str]
    estimated_selection_time: int  # seconds for user to choose


class WorkflowAlternativeEngine:
    """Engine for generating and managing workflow alternatives."""
    
    def __init__(self):
        self.alternatives_db: Dict[str, List[WorkflowAlternative]] = {}
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        self.success_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.context_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Learning and adaptation
        self.alternative_performance: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.user_skill_assessments: Dict[str, DifficultyLevel] = {}
        self.contextual_success_rates: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Configuration
        self.max_alternatives_per_request = 5
        self.min_confidence_threshold = 0.3
        self.personalization_weight = 0.4
        self.context_similarity_threshold = 0.7
        
        # Initialize with default alternatives
        self._initialize_default_alternatives()
    
    def _initialize_default_alternatives(self):
        """Initialize the system with default workflow alternatives."""
        
        # Data Analysis Alternatives
        data_analysis_alternatives = [
            WorkflowAlternative(
                alternative_id="manual_data_review",
                name="Manual Data Review",
                description="Review and analyze data manually using basic tools",
                category=WorkflowCategory.DATA_ANALYSIS,
                alternative_type=AlternativeType.MANUAL,
                difficulty=DifficultyLevel.BEGINNER,
                estimated_total_time_minutes=30,
                success_probability=0.8,
                steps=[
                    WorkflowStep(
                        step_id="export_data",
                        title="Export Data",
                        description="Export your data to CSV or Excel format",
                        estimated_time_minutes=5,
                        required_tools=["data_export"]
                    ),
                    WorkflowStep(
                        step_id="open_spreadsheet",
                        title="Open in Spreadsheet",
                        description="Open the exported data in Excel or Google Sheets",
                        estimated_time_minutes=2,
                        required_tools=["spreadsheet_software"]
                    ),
                    WorkflowStep(
                        step_id="basic_analysis",
                        title="Perform Basic Analysis",
                        description="Use spreadsheet functions to calculate sums, averages, and create simple charts",
                        estimated_time_minutes=20,
                        required_skills=["basic_spreadsheet_skills"]
                    ),
                    WorkflowStep(
                        step_id="document_findings",
                        title="Document Findings",
                        description="Create a summary of your findings and insights",
                        estimated_time_minutes=3,
                        optional=True
                    )
                ],
                benefits=[
                    "Works with any data size",
                    "No dependency on automated systems",
                    "Full control over analysis process",
                    "Can be done offline"
                ],
                limitations=[
                    "Time-consuming for large datasets",
                    "Limited to basic statistical analysis",
                    "Requires manual effort"
                ],
                when_to_use=[
                    "Automated analysis is unavailable",
                    "Small to medium datasets",
                    "Need quick insights",
                    "System performance issues"
                ]
            ),
            
            WorkflowAlternative(
                alternative_id="statistical_sampling",
                name="Statistical Sampling Analysis",
                description="Analyze a representative sample of your data for quick insights",
                category=WorkflowCategory.DATA_ANALYSIS,
                alternative_type=AlternativeType.SIMPLIFIED,
                difficulty=DifficultyLevel.INTERMEDIATE,
                estimated_total_time_minutes=15,
                success_probability=0.7,
                steps=[
                    WorkflowStep(
                        step_id="select_sample",
                        title="Select Representative Sample",
                        description="Choose a random sample of 1000-5000 records from your dataset",
                        estimated_time_minutes=5,
                        required_skills=["sampling_techniques"]
                    ),
                    WorkflowStep(
                        step_id="quick_analysis",
                        title="Quick Statistical Analysis",
                        description="Run basic statistics on the sample data",
                        estimated_time_minutes=8,
                        automation_available=True
                    ),
                    WorkflowStep(
                        step_id="extrapolate_results",
                        title="Extrapolate to Full Dataset",
                        description="Apply findings to the complete dataset with confidence intervals",
                        estimated_time_minutes=2,
                        required_skills=["statistical_inference"]
                    )
                ],
                benefits=[
                    "Much faster than full analysis",
                    "Statistically valid results",
                    "Works with very large datasets",
                    "Lower computational requirements"
                ],
                limitations=[
                    "Results are estimates with confidence intervals",
                    "May miss rare patterns",
                    "Requires understanding of sampling"
                ],
                when_to_use=[
                    "Very large datasets",
                    "Time constraints",
                    "System resource limitations",
                    "Need quick preliminary insights"
                ]
            )
        ]
        
        # Visualization Alternatives
        visualization_alternatives = [
            WorkflowAlternative(
                alternative_id="simple_table_view",
                name="Simple Table View",
                description="Display data in a clean, sortable table format",
                category=WorkflowCategory.VISUALIZATION,
                alternative_type=AlternativeType.SIMPLIFIED,
                difficulty=DifficultyLevel.BEGINNER,
                estimated_total_time_minutes=5,
                success_probability=0.95,
                steps=[
                    WorkflowStep(
                        step_id="format_data",
                        title="Format Data for Table",
                        description="Ensure data is in tabular format with clear column headers",
                        estimated_time_minutes=2
                    ),
                    WorkflowStep(
                        step_id="apply_sorting",
                        title="Apply Sorting and Filtering",
                        description="Add sorting and basic filtering capabilities",
                        estimated_time_minutes=2,
                        automation_available=True
                    ),
                    WorkflowStep(
                        step_id="style_table",
                        title="Apply Basic Styling",
                        description="Apply clean styling for better readability",
                        estimated_time_minutes=1,
                        optional=True
                    )
                ],
                benefits=[
                    "Always works regardless of data complexity",
                    "Fast to implement",
                    "Universally understood format",
                    "Good for detailed data inspection"
                ],
                limitations=[
                    "Not visually engaging",
                    "Difficult to spot trends",
                    "Limited for large datasets"
                ],
                when_to_use=[
                    "Chart generation fails",
                    "Need detailed data view",
                    "Simple data presentation required",
                    "Accessibility concerns"
                ]
            ),
            
            WorkflowAlternative(
                alternative_id="text_based_summary",
                name="Text-Based Data Summary",
                description="Create a narrative summary of key data insights",
                category=WorkflowCategory.VISUALIZATION,
                alternative_type=AlternativeType.WORKAROUND,
                difficulty=DifficultyLevel.INTERMEDIATE,
                estimated_total_time_minutes=10,
                success_probability=0.8,
                steps=[
                    WorkflowStep(
                        step_id="calculate_key_metrics",
                        title="Calculate Key Metrics",
                        description="Compute essential statistics like totals, averages, min/max values",
                        estimated_time_minutes=3
                    ),
                    WorkflowStep(
                        step_id="identify_trends",
                        title="Identify Key Trends",
                        description="Look for patterns, outliers, and significant changes",
                        estimated_time_minutes=4,
                        required_skills=["data_interpretation"]
                    ),
                    WorkflowStep(
                        step_id="write_summary",
                        title="Write Narrative Summary",
                        description="Create a clear, concise summary of findings in plain language",
                        estimated_time_minutes=3,
                        required_skills=["technical_writing"]
                    )
                ],
                benefits=[
                    "Accessible to all users",
                    "Focuses on key insights",
                    "No technical dependencies",
                    "Easy to share and discuss"
                ],
                limitations=[
                    "Time-consuming to create",
                    "Subjective interpretation",
                    "May miss subtle patterns"
                ],
                when_to_use=[
                    "Visualization tools unavailable",
                    "Audience prefers text",
                    "Need executive summary",
                    "Accessibility requirements"
                ]
            )
        ]
        
        # Report Generation Alternatives
        report_alternatives = [
            WorkflowAlternative(
                alternative_id="template_based_report",
                name="Template-Based Report Creation",
                description="Use pre-built templates to create reports manually",
                category=WorkflowCategory.REPORT_GENERATION,
                alternative_type=AlternativeType.MANUAL,
                difficulty=DifficultyLevel.BEGINNER,
                estimated_total_time_minutes=45,
                success_probability=0.9,
                steps=[
                    WorkflowStep(
                        step_id="select_template",
                        title="Choose Report Template",
                        description="Select an appropriate template based on report type and audience",
                        estimated_time_minutes=5,
                        required_tools=["document_templates"]
                    ),
                    WorkflowStep(
                        step_id="gather_data",
                        title="Gather Required Data",
                        description="Collect all necessary data points and metrics",
                        estimated_time_minutes=15
                    ),
                    WorkflowStep(
                        step_id="populate_template",
                        title="Populate Template",
                        description="Fill in the template with your data and analysis",
                        estimated_time_minutes=20,
                        required_skills=["document_editing"]
                    ),
                    WorkflowStep(
                        step_id="review_format",
                        title="Review and Format",
                        description="Review content and apply final formatting",
                        estimated_time_minutes=5,
                        optional=True
                    )
                ],
                benefits=[
                    "Consistent professional format",
                    "Faster than starting from scratch",
                    "Ensures all key sections included",
                    "No dependency on automated systems"
                ],
                limitations=[
                    "Limited customization",
                    "Manual data entry required",
                    "Time-consuming for complex reports"
                ],
                when_to_use=[
                    "Automated report generation fails",
                    "Standard report format needed",
                    "Limited time for custom design",
                    "Consistent branding required"
                ]
            )
        ]
        
        # Store alternatives by category
        self.alternatives_db[WorkflowCategory.DATA_ANALYSIS.value] = data_analysis_alternatives
        self.alternatives_db[WorkflowCategory.VISUALIZATION.value] = visualization_alternatives
        self.alternatives_db[WorkflowCategory.REPORT_GENERATION.value] = report_alternatives
    
    async def suggest_alternatives(self, context: WorkflowContext) -> SuggestionResult:
        """Suggest workflow alternatives based on context."""
        start_time = time.time()
        
        # Determine workflow category
        category = self._determine_workflow_category(context)
        
        # Get candidate alternatives
        candidates = self._get_candidate_alternatives(category, context)
        
        # Score and rank alternatives
        scored_alternatives = await self._score_alternatives(candidates, context)
        
        # Apply personalization
        personalized_alternatives = self._apply_personalization(scored_alternatives, context)
        
        # Select top alternatives
        selected_alternatives = self._select_top_alternatives(personalized_alternatives, context)
        
        # Calculate confidence and reasoning
        confidence_score = self._calculate_confidence(selected_alternatives, context)
        reasoning = self._generate_reasoning(selected_alternatives, context)
        
        # Record context pattern for learning
        self._record_context_pattern(context, selected_alternatives)
        
        processing_time = time.time() - start_time
        
        return SuggestionResult(
            alternatives=selected_alternatives,
            confidence_score=confidence_score,
            reasoning=reasoning,
            personalization_applied=context.user_id is not None,
            context_factors=self._extract_context_factors(context),
            estimated_selection_time=max(10, len(selected_alternatives) * 5)  # 5 seconds per alternative
        )
    
    def _determine_workflow_category(self, context: WorkflowContext) -> WorkflowCategory:
        """Determine the workflow category from context."""
        workflow_lower = context.original_workflow.lower()
        
        # Keyword-based category detection
        category_keywords = {
            WorkflowCategory.DATA_ANALYSIS: ["analyze", "analysis", "statistics", "calculate", "compute"],
            WorkflowCategory.VISUALIZATION: ["chart", "graph", "plot", "visualize", "display"],
            WorkflowCategory.REPORT_GENERATION: ["report", "summary", "document", "generate"],
            WorkflowCategory.FILE_PROCESSING: ["upload", "import", "export", "file", "process"],
            WorkflowCategory.AI_INTERACTION: ["ai", "assistant", "chat", "query", "ask"],
            WorkflowCategory.DASHBOARD_CREATION: ["dashboard", "widget", "panel", "overview"],
            WorkflowCategory.DATA_EXPORT: ["export", "download", "save", "extract"],
            WorkflowCategory.COLLABORATION: ["share", "collaborate", "team", "comment"],
            WorkflowCategory.TROUBLESHOOTING: ["error", "fix", "debug", "troubleshoot", "issue"]
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in workflow_lower for keyword in keywords):
                return category
        
        # Default fallback
        return WorkflowCategory.DATA_ANALYSIS
    
    def _get_candidate_alternatives(self, category: WorkflowCategory, context: WorkflowContext) -> List[WorkflowAlternative]:
        """Get candidate alternatives for the given category."""
        candidates = self.alternatives_db.get(category.value, []).copy()
        
        # Filter by user skill level
        skill_filtered = []
        for alt in candidates:
            if alt.difficulty.value <= context.user_skill_level.value or context.user_skill_level == DifficultyLevel.EXPERT:
                skill_filtered.append(alt)
        
        # Filter by available tools
        if context.available_tools:
            tool_filtered = []
            for alt in skill_filtered:
                required_tools = set()
                for step in alt.steps:
                    required_tools.update(step.required_tools)
                
                if not required_tools or required_tools.issubset(set(context.available_tools)):
                    tool_filtered.append(alt)
            
            return tool_filtered if tool_filtered else skill_filtered
        
        return skill_filtered
    
    async def _score_alternatives(self, alternatives: List[WorkflowAlternative], 
                                context: WorkflowContext) -> List[Tuple[WorkflowAlternative, float]]:
        """Score alternatives based on context and historical performance."""
        scored_alternatives = []
        
        for alt in alternatives:
            score = 0.0
            
            # Base success probability
            score += alt.success_probability * 0.4
            
            # Time constraint factor
            if context.time_constraints:
                if alt.estimated_total_time_minutes <= context.time_constraints:
                    score += 0.2
                else:
                    # Penalize alternatives that take too long
                    time_penalty = min(0.2, (alt.estimated_total_time_minutes - context.time_constraints) / context.time_constraints)
                    score -= time_penalty
            
            # Historical success rate
            if context.user_id and alt.alternative_id in self.alternative_performance.get(context.user_id, {}):
                historical_success = self.alternative_performance[context.user_id][alt.alternative_id]
                score += historical_success * 0.2
            
            # Priority level adjustment
            if context.priority_level == "critical":
                # Prefer simpler, more reliable alternatives for critical tasks
                if alt.difficulty == DifficultyLevel.BEGINNER:
                    score += 0.1
                if alt.success_probability > 0.8:
                    score += 0.1
            elif context.priority_level == "low":
                # Allow more experimental alternatives for low priority
                if alt.alternative_type == AlternativeType.ENHANCED:
                    score += 0.05
            
            # Failure reason specific scoring
            if context.failure_reason:
                failure_lower = context.failure_reason.lower()
                if "timeout" in failure_lower and alt.estimated_total_time_minutes < 15:
                    score += 0.15
                elif "resource" in failure_lower and alt.alternative_type == AlternativeType.SIMPLIFIED:
                    score += 0.15
                elif "network" in failure_lower and "offline" in " ".join(alt.benefits).lower():
                    score += 0.2
            
            # User preference alignment
            if context.user_preferences:
                if context.user_preferences.get("prefer_manual", False) and alt.alternative_type == AlternativeType.MANUAL:
                    score += 0.1
                if context.user_preferences.get("prefer_simple", False) and alt.difficulty == DifficultyLevel.BEGINNER:
                    score += 0.1
            
            scored_alternatives.append((alt, max(0.0, min(1.0, score))))
        
        return scored_alternatives
    
    def _apply_personalization(self, scored_alternatives: List[Tuple[WorkflowAlternative, float]], 
                             context: WorkflowContext) -> List[Tuple[WorkflowAlternative, float]]:
        """Apply personalization based on user history and preferences."""
        if not context.user_id:
            return scored_alternatives
        
        personalized = []
        user_history = self.success_history.get(context.user_id, [])
        
        for alt, score in scored_alternatives:
            personalized_score = score
            
            # Boost alternatives the user has succeeded with before
            successful_alternatives = [h['alternative_id'] for h in user_history if h.get('success', False)]
            if alt.alternative_id in successful_alternatives:
                success_count = successful_alternatives.count(alt.alternative_id)
                boost = min(0.2, success_count * 0.05)
                personalized_score += boost
            
            # Penalize alternatives the user has failed with recently
            recent_failures = [
                h for h in user_history 
                if h.get('alternative_id') == alt.alternative_id and 
                not h.get('success', False) and
                (datetime.utcnow() - datetime.fromisoformat(h.get('timestamp', '2000-01-01'))).days < 7
            ]
            if recent_failures:
                penalty = min(0.15, len(recent_failures) * 0.05)
                personalized_score -= penalty
            
            # Consider user skill progression
            if context.user_id in self.user_skill_assessments:
                assessed_skill = self.user_skill_assessments[context.user_id]
                if alt.difficulty.value < assessed_skill.value:
                    # User might find this too simple
                    personalized_score -= 0.05
                elif alt.difficulty.value > assessed_skill.value + 1:
                    # Might be too challenging
                    personalized_score -= 0.1
            
            personalized.append((alt, max(0.0, min(1.0, personalized_score))))
        
        return personalized
    
    def _select_top_alternatives(self, scored_alternatives: List[Tuple[WorkflowAlternative, float]], 
                               context: WorkflowContext) -> List[WorkflowAlternative]:
        """Select the top alternatives based on scores."""
        # Sort by score (descending)
        sorted_alternatives = sorted(scored_alternatives, key=lambda x: x[1], reverse=True)
        
        # Filter by minimum confidence threshold
        filtered = [(alt, score) for alt, score in sorted_alternatives if score >= self.min_confidence_threshold]
        
        if not filtered:
            # If no alternatives meet threshold, take the best one anyway
            filtered = sorted_alternatives[:1]
        
        # Select top N alternatives
        selected = filtered[:self.max_alternatives_per_request]
        
        # Ensure diversity in alternative types
        diverse_alternatives = []
        seen_types = set()
        
        for alt, score in selected:
            if alt.alternative_type not in seen_types or len(diverse_alternatives) < 2:
                diverse_alternatives.append(alt)
                seen_types.add(alt.alternative_type)
        
        # Fill remaining slots if needed
        remaining_slots = self.max_alternatives_per_request - len(diverse_alternatives)
        for alt, score in selected:
            if alt not in diverse_alternatives and remaining_slots > 0:
                diverse_alternatives.append(alt)
                remaining_slots -= 1
        
        return diverse_alternatives
    
    def _calculate_confidence(self, alternatives: List[WorkflowAlternative], 
                            context: WorkflowContext) -> float:
        """Calculate overall confidence in the suggestions."""
        if not alternatives:
            return 0.0
        
        # Base confidence from alternative success probabilities
        avg_success_prob = sum(alt.success_probability for alt in alternatives) / len(alternatives)
        
        # Adjust based on context completeness
        context_completeness = 0.0
        if context.user_id:
            context_completeness += 0.2
        if context.failure_reason:
            context_completeness += 0.2
        if context.available_tools:
            context_completeness += 0.2
        if context.user_preferences:
            context_completeness += 0.2
        if context.historical_successes:
            context_completeness += 0.2
        
        # Combine factors
        confidence = (avg_success_prob * 0.7) + (context_completeness * 0.3)
        
        return max(0.0, min(1.0, confidence))
    
    def _generate_reasoning(self, alternatives: List[WorkflowAlternative], 
                          context: WorkflowContext) -> str:
        """Generate human-readable reasoning for the suggestions."""
        if not alternatives:
            return "No suitable alternatives found for the current context."
        
        reasoning_parts = []
        
        # Context-based reasoning
        if context.failure_reason:
            reasoning_parts.append(f"Based on the failure reason '{context.failure_reason}', ")
        
        if context.time_constraints:
            reasoning_parts.append(f"considering your time constraint of {context.time_constraints} minutes, ")
        
        # Alternative-specific reasoning
        top_alt = alternatives[0]
        reasoning_parts.append(f"I recommend starting with '{top_alt.name}' ")
        
        if top_alt.success_probability > 0.8:
            reasoning_parts.append("due to its high success rate ")
        
        if top_alt.difficulty == DifficultyLevel.BEGINNER:
            reasoning_parts.append("and beginner-friendly approach")
        elif top_alt.difficulty == DifficultyLevel.EXPERT:
            reasoning_parts.append("which leverages advanced techniques")
        
        # Additional alternatives
        if len(alternatives) > 1:
            reasoning_parts.append(f". Alternative options include {alternatives[1].name}")
            if len(alternatives) > 2:
                reasoning_parts.append(f" and {alternatives[2].name}")
        
        reasoning_parts.append(".")
        
        return "".join(reasoning_parts)
    
    def _extract_context_factors(self, context: WorkflowContext) -> List[str]:
        """Extract key context factors that influenced the suggestions."""
        factors = []
        
        if context.failure_reason:
            factors.append(f"failure_reason: {context.failure_reason}")
        
        if context.time_constraints:
            factors.append(f"time_constraint: {context.time_constraints}min")
        
        if context.user_skill_level != DifficultyLevel.INTERMEDIATE:
            factors.append(f"skill_level: {context.user_skill_level.value}")
        
        if context.priority_level != "normal":
            factors.append(f"priority: {context.priority_level}")
        
        if context.available_tools:
            factors.append(f"available_tools: {len(context.available_tools)}")
        
        if context.user_preferences:
            factors.append("user_preferences_applied")
        
        if context.historical_successes:
            factors.append("historical_data_used")
        
        return factors
    
    def _record_context_pattern(self, context: WorkflowContext, alternatives: List[WorkflowAlternative]):
        """Record context pattern for machine learning."""
        if not context.user_id:
            return
        
        pattern = {
            "timestamp": datetime.utcnow().isoformat(),
            "original_workflow": context.original_workflow,
            "failure_reason": context.failure_reason,
            "user_skill_level": context.user_skill_level.value,
            "time_constraints": context.time_constraints,
            "priority_level": context.priority_level,
            "suggested_alternatives": [alt.alternative_id for alt in alternatives],
            "context_hash": hashlib.md5(
                json.dumps({
                    "workflow": context.original_workflow,
                    "failure": context.failure_reason,
                    "skill": context.user_skill_level.value,
                    "priority": context.priority_level
                }, sort_keys=True).encode()
            ).hexdigest()
        }
        
        self.context_patterns[context.user_id].append(pattern)
        
        # Keep only recent patterns
        cutoff_time = datetime.utcnow() - timedelta(days=30)
        self.context_patterns[context.user_id] = [
            p for p in self.context_patterns[context.user_id]
            if datetime.fromisoformat(p['timestamp']) > cutoff_time
        ]
    
    async def record_alternative_outcome(self, user_id: str, alternative_id: str, 
                                       success: bool, feedback_score: Optional[float] = None,
                                       completion_time_minutes: Optional[int] = None):
        """Record the outcome of using a workflow alternative."""
        # Update success history
        outcome = {
            "timestamp": datetime.utcnow().isoformat(),
            "alternative_id": alternative_id,
            "success": success,
            "feedback_score": feedback_score,
            "completion_time_minutes": completion_time_minutes
        }
        
        self.success_history[user_id].append(outcome)
        
        # Update alternative performance metrics
        if user_id not in self.alternative_performance:
            self.alternative_performance[user_id] = {}
        
        if alternative_id not in self.alternative_performance[user_id]:
            self.alternative_performance[user_id][alternative_id] = 0.5  # Start with neutral
        
        # Update performance using exponential moving average
        current_performance = self.alternative_performance[user_id][alternative_id]
        new_performance = 1.0 if success else 0.0
        
        # Weight recent outcomes more heavily
        alpha = 0.3
        self.alternative_performance[user_id][alternative_id] = (
            alpha * new_performance + (1 - alpha) * current_performance
        )
        
        # Update alternative usage statistics
        for category_alternatives in self.alternatives_db.values():
            for alt in category_alternatives:
                if alt.alternative_id == alternative_id:
                    alt.usage_count += 1
                    if success:
                        alt.success_count += 1
                    
                    if feedback_score is not None:
                        if alt.user_feedback_score is None:
                            alt.user_feedback_score = feedback_score
                        else:
                            # Moving average of feedback scores
                            alt.user_feedback_score = (
                                0.7 * alt.user_feedback_score + 0.3 * feedback_score
                            )
                    
                    # Update success probability based on actual outcomes
                    if alt.usage_count > 5:  # Only after sufficient usage
                        actual_success_rate = alt.success_count / alt.usage_count
                        # Blend with original estimate
                        alt.success_probability = (
                            0.6 * actual_success_rate + 0.4 * alt.success_probability
                        )
                    
                    break
        
        logger.info(f"Recorded outcome for alternative {alternative_id}: success={success}")
    
    def get_alternative_by_id(self, alternative_id: str) -> Optional[WorkflowAlternative]:
        """Get a specific alternative by its ID."""
        for category_alternatives in self.alternatives_db.values():
            for alt in category_alternatives:
                if alt.alternative_id == alternative_id:
                    return alt
        return None
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get statistics for a specific user."""
        if user_id not in self.success_history:
            return {"total_attempts": 0, "success_rate": 0.0}
        
        history = self.success_history[user_id]
        total_attempts = len(history)
        successful_attempts = sum(1 for h in history if h.get('success', False))
        success_rate = successful_attempts / total_attempts if total_attempts > 0 else 0.0
        
        # Most used alternatives
        alternative_usage = defaultdict(int)
        for h in history:
            alternative_usage[h.get('alternative_id', 'unknown')] += 1
        
        most_used = sorted(alternative_usage.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_attempts": total_attempts,
            "successful_attempts": successful_attempts,
            "success_rate": success_rate,
            "most_used_alternatives": most_used,
            "skill_level": self.user_skill_assessments.get(user_id, DifficultyLevel.INTERMEDIATE).value
        }
    
    def add_custom_alternative(self, alternative: WorkflowAlternative, category: WorkflowCategory):
        """Add a custom workflow alternative."""
        if category.value not in self.alternatives_db:
            self.alternatives_db[category.value] = []
        
        self.alternatives_db[category.value].append(alternative)
        logger.info(f"Added custom alternative {alternative.alternative_id} to category {category.value}")


# Global instance
workflow_alternative_engine = WorkflowAlternativeEngine()


# Convenience functions
async def get_workflow_alternatives(original_workflow: str, user_id: str = None,
                                  failure_reason: str = None, time_constraints: int = None) -> SuggestionResult:
    """Get workflow alternatives for a failed or problematic workflow."""
    context = WorkflowContext(
        user_id=user_id,
        original_workflow=original_workflow,
        failure_reason=failure_reason,
        time_constraints=time_constraints
    )
    
    return await workflow_alternative_engine.suggest_alternatives(context)


async def record_workflow_outcome(user_id: str, alternative_id: str, success: bool,
                                feedback_score: float = None) -> None:
    """Record the outcome of using a workflow alternative."""
    await workflow_alternative_engine.record_alternative_outcome(
        user_id, alternative_id, success, feedback_score
    )