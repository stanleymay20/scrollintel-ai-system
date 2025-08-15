"""
Intelligent Fallback Content Generation System for ScrollIntel.
Provides context-aware fallback content, smart caching, alternative workflows,
and progressive content loading to ensure users never see empty or broken states.
"""

import asyncio
import logging
import json
import hashlib
import time
import random
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import pickle
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Types of content that can have fallbacks."""
    CHART = "chart"
    TABLE = "table"
    TEXT = "text"
    IMAGE = "image"
    LIST = "list"
    FORM = "form"
    DASHBOARD = "dashboard"
    REPORT = "report"
    ANALYSIS = "analysis"
    RECOMMENDATION = "recommendation"


class FallbackQuality(Enum):
    """Quality levels of fallback content."""
    HIGH = "high"           # Near-identical to original
    MEDIUM = "medium"       # Reduced functionality but useful
    LOW = "low"            # Basic placeholder
    EMERGENCY = "emergency" # Minimal viable content


@dataclass
class ContentContext:
    """Context information for generating appropriate fallbacks."""
    user_id: Optional[str] = None
    content_type: ContentType = ContentType.TEXT
    original_request: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)
    historical_usage: List[Dict[str, Any]] = field(default_factory=list)
    error_context: Optional[Exception] = None
    urgency_level: str = "normal"  # low, normal, high, critical
    device_capabilities: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FallbackContent:
    """Generated fallback content with metadata."""
    content: Any
    content_type: ContentType
    quality: FallbackQuality
    confidence: float  # 0.0 to 1.0
    staleness: Optional[timedelta] = None
    generation_time: datetime = field(default_factory=datetime.utcnow)
    cache_key: Optional[str] = None
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    user_message: Optional[str] = None
    suggested_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheEntry:
    """Smart cache entry with staleness tracking."""
    content: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    staleness_indicators: Dict[str, Any]
    quality_score: float
    user_feedback: List[float] = field(default_factory=list)
    context_hash: str = ""
    expiry_time: Optional[datetime] = None


@dataclass
class WorkflowAlternative:
    """Alternative workflow suggestion."""
    name: str
    description: str
    steps: List[str]
    estimated_time: str
    difficulty: str  # easy, medium, hard
    success_probability: float
    required_capabilities: List[str] = field(default_factory=list)
    fallback_data: Optional[Dict[str, Any]] = None


class IntelligentFallbackManager:
    """Main manager for intelligent fallback content generation."""
    
    def __init__(self):
        self.content_generators: Dict[ContentType, Callable] = {}
        self.cache: Dict[str, CacheEntry] = {}
        self.workflow_alternatives: Dict[str, List[WorkflowAlternative]] = {}
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        self.content_templates: Dict[ContentType, Dict[str, Any]] = {}
        self.progressive_loaders: Dict[str, Callable] = {}
        
        # Learning and adaptation
        self.usage_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.feedback_history: Dict[str, List[float]] = defaultdict(list)
        self.context_similarity_cache: Dict[str, List[Tuple[str, float]]] = {}
        
        # Configuration
        self.cache_max_size = 1000
        self.cache_ttl_hours = 24
        self.staleness_threshold_hours = 1
        self.min_confidence_threshold = 0.3
        
        # Initialize system
        self._initialize_content_generators()
        self._initialize_content_templates()
        self._initialize_workflow_alternatives()
        self._load_cached_data()
    
    def _initialize_content_generators(self):
        """Initialize content generators for different types."""
        self.content_generators = {
            ContentType.CHART: self._generate_chart_fallback,
            ContentType.TABLE: self._generate_table_fallback,
            ContentType.TEXT: self._generate_text_fallback,
            ContentType.IMAGE: self._generate_image_fallback,
            ContentType.LIST: self._generate_list_fallback,
            ContentType.FORM: self._generate_form_fallback,
            ContentType.DASHBOARD: self._generate_dashboard_fallback,
            ContentType.REPORT: self._generate_report_fallback,
            ContentType.ANALYSIS: self._generate_analysis_fallback,
            ContentType.RECOMMENDATION: self._generate_recommendation_fallback,
        }
    
    def _initialize_content_templates(self):
        """Initialize content templates for fallback generation."""
        self.content_templates = {
            ContentType.CHART: {
                "bar_chart": {
                    "type": "bar",
                    "data": [
                        {"category": "Sample A", "value": 100},
                        {"category": "Sample B", "value": 150},
                        {"category": "Sample C", "value": 120}
                    ],
                    "title": "Sample Data",
                    "message": "Actual data temporarily unavailable"
                },
                "line_chart": {
                    "type": "line",
                    "data": [
                        {"x": "Jan", "y": 85},
                        {"x": "Feb", "y": 92},
                        {"x": "Mar", "y": 88},
                        {"x": "Apr", "y": 95}
                    ],
                    "title": "Trend Analysis",
                    "message": "Using sample trend data"
                },
                "pie_chart": {
                    "type": "pie",
                    "data": [
                        {"label": "Category A", "value": 40},
                        {"label": "Category B", "value": 35},
                        {"label": "Category C", "value": 25}
                    ],
                    "title": "Distribution",
                    "message": "Sample distribution shown"
                }
            },
            ContentType.TABLE: {
                "data_table": {
                    "columns": ["Name", "Value", "Status"],
                    "rows": [
                        ["Sample Item 1", "100", "Active"],
                        ["Sample Item 2", "150", "Pending"],
                        ["Sample Item 3", "120", "Complete"]
                    ],
                    "message": "Sample data shown - actual data loading"
                },
                "summary_table": {
                    "columns": ["Metric", "Current", "Previous"],
                    "rows": [
                        ["Total Users", "---", "---"],
                        ["Revenue", "---", "---"],
                        ["Growth Rate", "---", "---"]
                    ],
                    "message": "Data temporarily unavailable"
                }
            },
            ContentType.TEXT: {
                "loading_message": "Content is being prepared. Please wait a moment...",
                "error_message": "Content temporarily unavailable. Please try refreshing the page.",
                "placeholder": "This section will display your content once it's ready.",
                "help_text": "While we prepare your content, you can explore other features or check back in a moment."
            },
            ContentType.ANALYSIS: {
                "summary": {
                    "title": "Analysis Summary",
                    "insights": [
                        "Analysis is being processed with your data",
                        "Results will be available shortly",
                        "You can continue with other tasks while waiting"
                    ],
                    "confidence": 0.0,
                    "status": "processing"
                }
            }
        }
    
    def _initialize_workflow_alternatives(self):
        """Initialize alternative workflow suggestions."""
        self.workflow_alternatives = {
            "data_analysis": [
                WorkflowAlternative(
                    name="Manual Data Review",
                    description="Review your data manually while automated analysis loads",
                    steps=[
                        "Navigate to the data table view",
                        "Sort and filter data as needed",
                        "Look for obvious patterns or outliers",
                        "Take notes for later comparison with automated analysis"
                    ],
                    estimated_time="5-10 minutes",
                    difficulty="easy",
                    success_probability=0.8
                ),
                WorkflowAlternative(
                    name="Export and Analyze Offline",
                    description="Download data for analysis in external tools",
                    steps=[
                        "Export data to CSV or Excel format",
                        "Open in your preferred analysis tool",
                        "Perform basic statistical analysis",
                        "Import results back when system is ready"
                    ],
                    estimated_time="15-30 minutes",
                    difficulty="medium",
                    success_probability=0.9
                )
            ],
            "visualization": [
                WorkflowAlternative(
                    name="Simple Table View",
                    description="View data in table format while charts load",
                    steps=[
                        "Switch to table view",
                        "Use sorting and filtering",
                        "Export table if needed"
                    ],
                    estimated_time="2-5 minutes",
                    difficulty="easy",
                    success_probability=0.95
                ),
                WorkflowAlternative(
                    name="Basic Chart Creation",
                    description="Create simple charts with available data",
                    steps=[
                        "Select chart type manually",
                        "Choose data columns",
                        "Apply basic formatting",
                        "Save for later enhancement"
                    ],
                    estimated_time="10-15 minutes",
                    difficulty="medium",
                    success_probability=0.7
                )
            ],
            "report_generation": [
                WorkflowAlternative(
                    name="Manual Report Draft",
                    description="Create report outline while automated generation processes",
                    steps=[
                        "Create document structure",
                        "Add section headers",
                        "Insert placeholder content",
                        "Add available data manually"
                    ],
                    estimated_time="20-30 minutes",
                    difficulty="medium",
                    success_probability=0.8
                )
            ]
        }
    
    async def generate_fallback_content(self, context: ContentContext) -> FallbackContent:
        """Generate context-aware fallback content."""
        # Check cache first
        cache_key = self._generate_cache_key(context)
        cached_content = await self._get_cached_content(cache_key, context)
        
        if cached_content and not self._is_stale(cached_content, context):
            logger.info(f"Using cached fallback content for {context.content_type.value}")
            cached_content.last_accessed = datetime.utcnow()
            cached_content.access_count += 1
            return self._cache_entry_to_fallback_content(cached_content, context)
        
        # Generate new fallback content
        generator = self.content_generators.get(context.content_type)
        if not generator:
            logger.warning(f"No generator found for content type: {context.content_type.value}")
            return await self._generate_generic_fallback(context)
        
        try:
            fallback_content = await generator(context)
            
            # Cache the generated content
            await self._cache_content(cache_key, fallback_content, context)
            
            # Record usage pattern
            self._record_usage_pattern(context, fallback_content)
            
            logger.info(f"Generated {fallback_content.quality.value} quality fallback for {context.content_type.value}")
            return fallback_content
            
        except Exception as e:
            logger.error(f"Failed to generate fallback content: {e}")
            return await self._generate_emergency_fallback(context)
    
    async def _generate_chart_fallback(self, context: ContentContext) -> FallbackContent:
        """Generate chart fallback content."""
        # Determine chart type from context
        chart_type = context.original_request.get('chart_type', 'bar')
        
        # Get appropriate template
        template_key = f"{chart_type}_chart"
        if template_key not in self.content_templates[ContentType.CHART]:
            template_key = "bar_chart"  # Default fallback
        
        template = self.content_templates[ContentType.CHART][template_key].copy()
        
        # Customize based on context
        if context.original_request.get('title'):
            template['title'] = f"{context.original_request['title']} (Sample Data)"
        
        # Try to use historical data if available
        historical_data = self._get_historical_data(context)
        if historical_data and 'chart_data' in historical_data:
            template['data'] = historical_data['chart_data'][:5]  # Limit to 5 items
            template['message'] = "Using recent historical data"
            quality = FallbackQuality.MEDIUM
            confidence = 0.7
        else:
            quality = FallbackQuality.LOW
            confidence = 0.4
        
        # Add user-friendly message
        user_message = self._generate_user_message(context, "chart")
        
        # Suggest alternatives
        alternatives = self._get_workflow_alternatives("visualization", context)
        
        return FallbackContent(
            content=template,
            content_type=ContentType.CHART,
            quality=quality,
            confidence=confidence,
            user_message=user_message,
            alternatives=alternatives,
            suggested_actions=[
                "Try refreshing the page",
                "Check your data source connection",
                "View data in table format instead"
            ]
        )
    
    async def _generate_table_fallback(self, context: ContentContext) -> FallbackContent:
        """Generate table fallback content."""
        template = self.content_templates[ContentType.TABLE]["data_table"].copy()
        
        # Try to get column information from context
        if context.original_request.get('columns'):
            template['columns'] = context.original_request['columns'][:10]  # Limit columns
        
        # Use historical data if available
        historical_data = self._get_historical_data(context)
        if historical_data and 'table_data' in historical_data:
            template['rows'] = historical_data['table_data'][:10]  # Limit rows
            template['message'] = "Showing recent data while current data loads"
            quality = FallbackQuality.MEDIUM
            confidence = 0.8
        else:
            # Generate sample rows based on columns
            sample_rows = []
            for i in range(3):
                row = []
                for col in template['columns']:
                    if 'id' in col.lower():
                        row.append(f"ID{i+1}")
                    elif 'name' in col.lower():
                        row.append(f"Sample {i+1}")
                    elif 'value' in col.lower() or 'amount' in col.lower():
                        row.append(f"{random.randint(100, 999)}")
                    elif 'status' in col.lower():
                        row.append(random.choice(["Active", "Pending", "Complete"]))
                    else:
                        row.append("---")
                sample_rows.append(row)
            
            template['rows'] = sample_rows
            quality = FallbackQuality.LOW
            confidence = 0.3
        
        user_message = self._generate_user_message(context, "table")
        alternatives = self._get_workflow_alternatives("data_analysis", context)
        
        return FallbackContent(
            content=template,
            content_type=ContentType.TABLE,
            quality=quality,
            confidence=confidence,
            user_message=user_message,
            alternatives=alternatives,
            suggested_actions=[
                "Export available data",
                "Try filtering or sorting",
                "Refresh the data source"
            ]
        )
    
    async def _generate_text_fallback(self, context: ContentContext) -> FallbackContent:
        """Generate text fallback content."""
        # Choose appropriate message based on context
        if context.error_context:
            message = self.content_templates[ContentType.TEXT]["error_message"]
            quality = FallbackQuality.LOW
        elif context.urgency_level == "critical":
            message = "Critical content is being prepared. Please wait..."
            quality = FallbackQuality.MEDIUM
        else:
            message = self.content_templates[ContentType.TEXT]["loading_message"]
            quality = FallbackQuality.MEDIUM
        
        # Customize message based on user preferences
        if context.user_preferences.get('verbose_messages', False):
            message += " " + self.content_templates[ContentType.TEXT]["help_text"]
        
        user_message = self._generate_user_message(context, "text")
        
        return FallbackContent(
            content={"text": message, "type": "fallback_text"},
            content_type=ContentType.TEXT,
            quality=quality,
            confidence=0.9,
            user_message=user_message,
            suggested_actions=[
                "Wait a moment and try again",
                "Check your internet connection",
                "Contact support if the issue persists"
            ]
        )
    
    async def _generate_analysis_fallback(self, context: ContentContext) -> FallbackContent:
        """Generate analysis fallback content."""
        template = self.content_templates[ContentType.ANALYSIS]["summary"].copy()
        
        # Try to provide some basic analysis from historical data
        historical_data = self._get_historical_data(context)
        if historical_data and 'analysis_results' in historical_data:
            template['insights'] = historical_data['analysis_results']['insights'][:3]
            template['title'] = "Previous Analysis Results"
            template['status'] = "historical"
            quality = FallbackQuality.MEDIUM
            confidence = 0.6
        else:
            # Generate generic insights based on context
            insights = [
                "Analysis is processing your data",
                "Results will include key metrics and trends",
                "Recommendations will be provided based on findings"
            ]
            
            if context.original_request.get('analysis_type') == 'trend':
                insights.append("Trend analysis will show patterns over time")
            elif context.original_request.get('analysis_type') == 'comparison':
                insights.append("Comparison analysis will highlight differences")
            
            template['insights'] = insights
            quality = FallbackQuality.LOW
            confidence = 0.3
        
        user_message = self._generate_user_message(context, "analysis")
        alternatives = self._get_workflow_alternatives("data_analysis", context)
        
        return FallbackContent(
            content=template,
            content_type=ContentType.ANALYSIS,
            quality=quality,
            confidence=confidence,
            user_message=user_message,
            alternatives=alternatives,
            suggested_actions=[
                "Review raw data while analysis completes",
                "Check analysis parameters",
                "Try a simpler analysis first"
            ]
        )
    
    async def _generate_recommendation_fallback(self, context: ContentContext) -> FallbackContent:
        """Generate recommendation fallback content."""
        # Generate generic recommendations based on context
        recommendations = []
        
        if context.original_request.get('domain') == 'business':
            recommendations = [
                "Review your key performance indicators",
                "Consider market trends in your decision making",
                "Analyze competitor strategies",
                "Focus on customer satisfaction metrics"
            ]
        elif context.original_request.get('domain') == 'technical':
            recommendations = [
                "Monitor system performance regularly",
                "Keep software dependencies updated",
                "Implement proper error handling",
                "Document your processes thoroughly"
            ]
        else:
            recommendations = [
                "Review available data sources",
                "Consider multiple perspectives",
                "Test different approaches",
                "Monitor results and adjust as needed"
            ]
        
        # Try to get personalized recommendations from history
        historical_data = self._get_historical_data(context)
        if historical_data and 'recommendations' in historical_data:
            recommendations = historical_data['recommendations'][:5]
            quality = FallbackQuality.MEDIUM
            confidence = 0.7
        else:
            quality = FallbackQuality.LOW
            confidence = 0.4
        
        content = {
            "recommendations": recommendations,
            "confidence": confidence,
            "source": "fallback_generator",
            "message": "General recommendations while personalized suggestions load"
        }
        
        user_message = self._generate_user_message(context, "recommendations")
        
        return FallbackContent(
            content=content,
            content_type=ContentType.RECOMMENDATION,
            quality=quality,
            confidence=confidence,
            user_message=user_message,
            suggested_actions=[
                "Review your historical data",
                "Update your preferences",
                "Try different filter criteria"
            ]
        )
    
    async def _generate_image_fallback(self, context: ContentContext) -> FallbackContent:
        """Generate image fallback content."""
        # Create placeholder image data
        placeholder = {
            "type": "placeholder",
            "width": context.original_request.get('width', 400),
            "height": context.original_request.get('height', 300),
            "background_color": "#f0f0f0",
            "text": "Image Loading...",
            "alt_text": "Placeholder while image loads"
        }
        
        user_message = "Image is being prepared. A placeholder is shown temporarily."
        
        return FallbackContent(
            content=placeholder,
            content_type=ContentType.IMAGE,
            quality=FallbackQuality.LOW,
            confidence=0.5,
            user_message=user_message,
            suggested_actions=[
                "Wait for image to load",
                "Check image source",
                "Try refreshing the page"
            ]
        )
    
    async def _generate_list_fallback(self, context: ContentContext) -> FallbackContent:
        """Generate list fallback content."""
        # Create sample list items
        sample_items = [
            {"id": 1, "title": "Sample Item 1", "description": "This is a sample item"},
            {"id": 2, "title": "Sample Item 2", "description": "This is another sample item"},
            {"id": 3, "title": "Sample Item 3", "description": "This is a third sample item"}
        ]
        
        # Try to use historical data
        historical_data = self._get_historical_data(context)
        if historical_data and 'list_items' in historical_data:
            sample_items = historical_data['list_items'][:10]
            quality = FallbackQuality.MEDIUM
            confidence = 0.7
        else:
            quality = FallbackQuality.LOW
            confidence = 0.3
        
        content = {
            "items": sample_items,
            "total_count": len(sample_items),
            "message": "Sample items shown while actual data loads"
        }
        
        user_message = self._generate_user_message(context, "list")
        
        return FallbackContent(
            content=content,
            content_type=ContentType.LIST,
            quality=quality,
            confidence=confidence,
            user_message=user_message,
            suggested_actions=[
                "Refresh the list",
                "Check filter settings",
                "Try different search terms"
            ]
        )
    
    async def _generate_form_fallback(self, context: ContentContext) -> FallbackContent:
        """Generate form fallback content."""
        # Create basic form structure
        form_fields = [
            {"name": "name", "type": "text", "label": "Name", "required": True},
            {"name": "email", "type": "email", "label": "Email", "required": True},
            {"name": "message", "type": "textarea", "label": "Message", "required": False}
        ]
        
        # Customize based on context
        if context.original_request.get('form_type') == 'contact':
            form_fields.append({"name": "phone", "type": "tel", "label": "Phone", "required": False})
        elif context.original_request.get('form_type') == 'feedback':
            form_fields.append({"name": "rating", "type": "select", "label": "Rating", "options": ["1", "2", "3", "4", "5"]})
        
        content = {
            "fields": form_fields,
            "title": "Form",
            "message": "Basic form available while full form loads"
        }
        
        user_message = "A simplified form is available while the full form loads."
        
        return FallbackContent(
            content=content,
            content_type=ContentType.FORM,
            quality=FallbackQuality.MEDIUM,
            confidence=0.6,
            user_message=user_message,
            suggested_actions=[
                "Fill out available fields",
                "Save as draft",
                "Try refreshing for full form"
            ]
        )
    
    async def _generate_dashboard_fallback(self, context: ContentContext) -> FallbackContent:
        """Generate dashboard fallback content."""
        # Create basic dashboard layout
        widgets = [
            {
                "type": "metric",
                "title": "Key Metric 1",
                "value": "---",
                "status": "loading"
            },
            {
                "type": "chart",
                "title": "Trend Chart",
                "data": "loading",
                "status": "loading"
            },
            {
                "type": "list",
                "title": "Recent Items",
                "items": ["Loading..."],
                "status": "loading"
            }
        ]
        
        content = {
            "widgets": widgets,
            "layout": "grid",
            "message": "Dashboard components are loading"
        }
        
        user_message = "Dashboard is loading. Individual components will appear as they become available."
        
        return FallbackContent(
            content=content,
            content_type=ContentType.DASHBOARD,
            quality=FallbackQuality.MEDIUM,
            confidence=0.5,
            user_message=user_message,
            suggested_actions=[
                "Wait for components to load",
                "Refresh individual widgets",
                "Check data connections"
            ]
        )
    
    async def _generate_report_fallback(self, context: ContentContext) -> FallbackContent:
        """Generate report fallback content."""
        # Create basic report structure
        sections = [
            {
                "title": "Executive Summary",
                "content": "Report is being generated with your data...",
                "status": "loading"
            },
            {
                "title": "Key Findings",
                "content": "Analysis in progress...",
                "status": "loading"
            },
            {
                "title": "Recommendations",
                "content": "Recommendations will be provided based on analysis...",
                "status": "loading"
            }
        ]
        
        content = {
            "title": context.original_request.get('title', 'Report'),
            "sections": sections,
            "generated_at": datetime.utcnow().isoformat(),
            "status": "generating"
        }
        
        user_message = "Report is being generated. Sections will be populated as analysis completes."
        alternatives = self._get_workflow_alternatives("report_generation", context)
        
        return FallbackContent(
            content=content,
            content_type=ContentType.REPORT,
            quality=FallbackQuality.MEDIUM,
            confidence=0.4,
            user_message=user_message,
            alternatives=alternatives,
            suggested_actions=[
                "Wait for report generation",
                "Review data sources",
                "Create manual report outline"
            ]
        )
    
    async def _generate_generic_fallback(self, context: ContentContext) -> FallbackContent:
        """Generate generic fallback when no specific generator exists."""
        content = {
            "message": f"Content of type '{context.content_type.value}' is being prepared",
            "status": "loading",
            "type": context.content_type.value
        }
        
        user_message = f"Your {context.content_type.value} content is being prepared. Please wait a moment."
        
        return FallbackContent(
            content=content,
            content_type=context.content_type,
            quality=FallbackQuality.LOW,
            confidence=0.2,
            user_message=user_message,
            suggested_actions=[
                "Wait a moment and try again",
                "Refresh the page",
                "Check your connection"
            ]
        )
    
    async def _generate_emergency_fallback(self, context: ContentContext) -> FallbackContent:
        """Generate emergency fallback when all else fails."""
        content = {
            "error": "Content temporarily unavailable",
            "message": "We're working to restore this content. Please try again in a moment.",
            "type": "emergency_fallback",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        user_message = "Content is temporarily unavailable. Our team has been notified and is working to resolve the issue."
        
        return FallbackContent(
            content=content,
            content_type=context.content_type,
            quality=FallbackQuality.EMERGENCY,
            confidence=0.1,
            user_message=user_message,
            suggested_actions=[
                "Try again in a few minutes",
                "Contact support if the issue persists",
                "Use alternative features while we resolve this"
            ]
        )
    
    def _generate_cache_key(self, context: ContentContext) -> str:
        """Generate cache key for content context."""
        key_data = {
            "content_type": context.content_type.value,
            "user_id": context.user_id,
            "request_hash": hashlib.md5(
                json.dumps(context.original_request, sort_keys=True).encode()
            ).hexdigest()[:8]
        }
        
        return hashlib.md5(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()
    
    async def _get_cached_content(self, cache_key: str, context: ContentContext) -> Optional[CacheEntry]:
        """Get content from cache if available and valid."""
        if cache_key not in self.cache:
            return None
        
        entry = self.cache[cache_key]
        
        # Check if expired
        if entry.expiry_time and datetime.utcnow() > entry.expiry_time:
            del self.cache[cache_key]
            return None
        
        # Check if too old
        age = datetime.utcnow() - entry.created_at
        if age.total_seconds() > self.cache_ttl_hours * 3600:
            del self.cache[cache_key]
            return None
        
        return entry
    
    def _is_stale(self, cache_entry: CacheEntry, context: ContentContext) -> bool:
        """Check if cached content is stale."""
        age = datetime.utcnow() - cache_entry.created_at
        
        # Basic staleness check
        if age.total_seconds() > self.staleness_threshold_hours * 3600:
            return True
        
        # Context-specific staleness checks
        staleness_indicators = cache_entry.staleness_indicators
        
        # Check if user preferences changed
        if context.user_id and context.user_id in self.user_preferences:
            current_prefs_hash = hashlib.md5(
                json.dumps(self.user_preferences[context.user_id], sort_keys=True).encode()
            ).hexdigest()
            
            if staleness_indicators.get('user_prefs_hash') != current_prefs_hash:
                return True
        
        # Check if system state changed significantly
        if context.system_state:
            current_state_hash = hashlib.md5(
                json.dumps(context.system_state, sort_keys=True).encode()
            ).hexdigest()
            
            if staleness_indicators.get('system_state_hash') != current_state_hash:
                return True
        
        return False
    
    async def _cache_content(self, cache_key: str, content: FallbackContent, context: ContentContext):
        """Cache generated content."""
        # Clean cache if too large
        if len(self.cache) >= self.cache_max_size:
            self._clean_cache()
        
        # Create staleness indicators
        staleness_indicators = {}
        
        if context.user_id and context.user_id in self.user_preferences:
            staleness_indicators['user_prefs_hash'] = hashlib.md5(
                json.dumps(self.user_preferences[context.user_id], sort_keys=True).encode()
            ).hexdigest()
        
        if context.system_state:
            staleness_indicators['system_state_hash'] = hashlib.md5(
                json.dumps(context.system_state, sort_keys=True).encode()
            ).hexdigest()
        
        # Create cache entry
        entry = CacheEntry(
            content=content.content,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            access_count=1,
            staleness_indicators=staleness_indicators,
            quality_score=content.confidence,
            context_hash=cache_key
        )
        
        self.cache[cache_key] = entry
        logger.debug(f"Cached fallback content with key: {cache_key}")
    
    def _clean_cache(self):
        """Clean old entries from cache."""
        # Remove oldest entries first
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Remove oldest 20% of entries
        remove_count = max(1, len(sorted_entries) // 5)
        for i in range(remove_count):
            key = sorted_entries[i][0]
            del self.cache[key]
        
        logger.debug(f"Cleaned {remove_count} entries from fallback cache")
    
    def _cache_entry_to_fallback_content(self, entry: CacheEntry, context: ContentContext) -> FallbackContent:
        """Convert cache entry back to fallback content."""
        # Calculate staleness
        age = datetime.utcnow() - entry.created_at
        
        return FallbackContent(
            content=entry.content,
            content_type=context.content_type,
            quality=FallbackQuality.MEDIUM if entry.quality_score > 0.5 else FallbackQuality.LOW,
            confidence=entry.quality_score,
            staleness=age,
            cache_key=entry.context_hash,
            user_message=f"Showing cached content (from {age.total_seconds():.0f} seconds ago)",
            suggested_actions=[
                "Refresh for latest content",
                "This content may be outdated"
            ]
        )
    
    def _get_historical_data(self, context: ContentContext) -> Optional[Dict[str, Any]]:
        """Get historical data for the user/context."""
        if not context.user_id:
            return None
        
        user_patterns = self.usage_patterns.get(context.user_id, [])
        if not user_patterns:
            return None
        
        # Find similar contexts
        similar_patterns = []
        for pattern in user_patterns:
            if pattern.get('content_type') == context.content_type.value:
                similar_patterns.append(pattern)
        
        if not similar_patterns:
            return None
        
        # Return most recent similar data
        recent_pattern = max(similar_patterns, key=lambda x: x.get('timestamp', 0))
        return recent_pattern.get('data')
    
    def _generate_user_message(self, context: ContentContext, content_type: str) -> str:
        """Generate user-friendly message for fallback content."""
        messages = {
            "chart": "Your chart is being prepared. Sample data is shown temporarily.",
            "table": "Your data table is loading. Sample rows are displayed below.",
            "text": "Content is being prepared. Please wait a moment.",
            "analysis": "Analysis is processing your data. Results will appear shortly.",
            "recommendations": "Personalized recommendations are being generated.",
            "list": "Your list is loading. Sample items are shown temporarily.",
            "form": "A simplified form is available while the full form loads."
        }
        
        base_message = messages.get(content_type, "Content is being prepared.")
        
        # Add context-specific information
        if context.urgency_level == "high":
            base_message += " This is a priority request and will be processed quickly."
        elif context.error_context:
            base_message += " There was a temporary issue, but we're working to resolve it."
        
        return base_message
    
    def _get_workflow_alternatives(self, workflow_type: str, context: ContentContext) -> List[Dict[str, Any]]:
        """Get workflow alternatives for the given type."""
        alternatives = self.workflow_alternatives.get(workflow_type, [])
        
        # Convert to dict format and filter based on context
        result = []
        for alt in alternatives:
            # Check if user has required capabilities
            if alt.required_capabilities:
                user_capabilities = context.device_capabilities.get('features', [])
                if not all(cap in user_capabilities for cap in alt.required_capabilities):
                    continue
            
            result.append({
                "name": alt.name,
                "description": alt.description,
                "steps": alt.steps,
                "estimated_time": alt.estimated_time,
                "difficulty": alt.difficulty,
                "success_probability": alt.success_probability
            })
        
        return result[:3]  # Limit to top 3 alternatives
    
    def _record_usage_pattern(self, context: ContentContext, content: FallbackContent):
        """Record usage pattern for learning."""
        if not context.user_id:
            return
        
        pattern = {
            "timestamp": time.time(),
            "content_type": context.content_type.value,
            "quality": content.quality.value,
            "confidence": content.confidence,
            "context_hash": hashlib.md5(
                json.dumps(context.original_request, sort_keys=True).encode()
            ).hexdigest(),
            "data": content.content
        }
        
        self.usage_patterns[context.user_id].append(pattern)
        
        # Keep only recent patterns
        cutoff_time = time.time() - 86400  # 24 hours
        self.usage_patterns[context.user_id] = [
            p for p in self.usage_patterns[context.user_id]
            if p['timestamp'] > cutoff_time
        ]
    
    def _load_cached_data(self):
        """Load cached data from storage."""
        try:
            cache_file = Path("data/fallback_cache.pkl")
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.cache = data.get('cache', {})
                    self.usage_patterns = data.get('usage_patterns', defaultdict(list))
                    self.user_preferences = data.get('user_preferences', {})
                logger.info(f"Loaded {len(self.cache)} cached fallback entries")
        except Exception as e:
            logger.warning(f"Could not load fallback cache: {e}")
    
    def _save_cached_data(self):
        """Save cached data to storage."""
        try:
            os.makedirs("data", exist_ok=True)
            cache_file = Path("data/fallback_cache.pkl")
            data = {
                'cache': self.cache,
                'usage_patterns': dict(self.usage_patterns),
                'user_preferences': self.user_preferences
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Could not save fallback cache: {e}")
    
    async def record_user_feedback(self, cache_key: str, satisfaction_score: float):
        """Record user feedback on fallback content quality."""
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            entry.user_feedback.append(satisfaction_score)
            
            # Update quality score based on feedback
            if entry.user_feedback:
                entry.quality_score = (entry.quality_score + satisfaction_score) / 2
        
        # Save periodically
        if random.random() < 0.1:  # 10% chance
            self._save_cached_data()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.cache:
            return {"total_entries": 0}
        
        total_entries = len(self.cache)
        avg_quality = sum(entry.quality_score for entry in self.cache.values()) / total_entries
        avg_age = sum(
            (datetime.utcnow() - entry.created_at).total_seconds()
            for entry in self.cache.values()
        ) / total_entries
        
        content_type_distribution = defaultdict(int)
        for entry in self.cache.values():
            # This is a simplified approach - in practice you'd store content type in cache entry
            content_type_distribution["unknown"] += 1
        
        return {
            "total_entries": total_entries,
            "average_quality_score": avg_quality,
            "average_age_seconds": avg_age,
            "content_type_distribution": dict(content_type_distribution),
            "cache_hit_rate": "Not tracked yet"  # Would need request tracking
        }


# Global instance
intelligent_fallback_manager = IntelligentFallbackManager()


# Convenience functions
async def generate_fallback(content_type: ContentType, user_id: str = None, 
                          original_request: Dict[str, Any] = None, 
                          error_context: Exception = None) -> FallbackContent:
    """Convenience function to generate fallback content."""
    context = ContentContext(
        user_id=user_id,
        content_type=content_type,
        original_request=original_request or {},
        error_context=error_context
    )
    
    return await intelligent_fallback_manager.generate_fallback_content(context)


async def get_workflow_alternatives(workflow_type: str, user_id: str = None) -> List[WorkflowAlternative]:
    """Get workflow alternatives for a specific type."""
    context = ContentContext(user_id=user_id)
    alternatives = intelligent_fallback_manager._get_workflow_alternatives(workflow_type, context)
    
    return [
        WorkflowAlternative(
            name=alt["name"],
            description=alt["description"],
            steps=alt["steps"],
            estimated_time=alt["estimated_time"],
            difficulty=alt["difficulty"],
            success_probability=alt["success_probability"]
        )
        for alt in alternatives
    ]