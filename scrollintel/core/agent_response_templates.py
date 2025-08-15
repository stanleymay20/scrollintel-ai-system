"""
Agent Response Templates - Provides agent-specific response templates and styles
Implements requirements 2.3, 6.5 for agent-specific response templates and styles.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

logger = logging.getLogger(__name__)


class ResponseType(str, Enum):
    """Types of responses agents can provide."""
    GREETING = "greeting"
    EXPLANATION = "explanation"
    ANALYSIS = "analysis"
    RECOMMENDATION = "recommendation"
    ERROR = "error"
    SUCCESS = "success"
    CLARIFICATION = "clarification"
    PROGRESS_UPDATE = "progress_update"
    COMPLETION = "completion"
    FOLLOW_UP = "follow_up"


class ResponseTone(str, Enum):
    """Tone variations for responses."""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    ENTHUSIASTIC = "enthusiastic"
    SUPPORTIVE = "supportive"
    CONFIDENT = "confident"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"


@dataclass
class ResponseTemplate:
    """Template for agent responses."""
    template_id: str
    agent_id: str
    response_type: ResponseType
    tone: ResponseTone
    template_text: str
    variables: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # Higher priority templates are preferred
    usage_count: int = 0
    success_rate: float = 1.0


@dataclass
class ResponseStyle:
    """Style configuration for agent responses."""
    agent_id: str
    formatting_preferences: Dict[str, Any] = field(default_factory=dict)
    emoji_usage: bool = True
    preferred_emojis: List[str] = field(default_factory=list)
    code_formatting: Dict[str, str] = field(default_factory=dict)
    list_formatting: Dict[str, str] = field(default_factory=dict)
    emphasis_style: str = "**bold**"  # markdown, html, caps
    signature_line: Optional[str] = None


@dataclass
class ResponseContext:
    """Context for response generation."""
    user_id: str
    conversation_id: str
    agent_id: str
    user_expertise_level: str = "intermediate"
    conversation_length: int = 0
    recent_topics: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    previous_responses: List[str] = field(default_factory=list)


class AgentResponseTemplateEngine:
    """Engine for managing agent response templates and styles."""
    
    def __init__(self):
        self.templates: Dict[str, List[ResponseTemplate]] = {}
        self.styles: Dict[str, ResponseStyle] = {}
        self.template_usage_stats: Dict[str, Dict[str, int]] = {}
        self._initialize_default_templates()
        self._initialize_default_styles()
    
    def _initialize_default_templates(self):
        """Initialize default response templates for each agent."""
        
        # CTO Agent Templates
        cto_templates = [
            ResponseTemplate(
                template_id="cto_greeting_professional",
                agent_id="scroll-cto-agent",
                response_type=ResponseType.GREETING,
                tone=ResponseTone.PROFESSIONAL,
                template_text="Hello! I'm Alex, your CTO advisor. I'm here to help you make strategic technical decisions and architect scalable solutions. What technical challenge can I help you solve today?",
                priority=2
            ),
            ResponseTemplate(
                template_id="cto_greeting_confident",
                agent_id="scroll-cto-agent",
                response_type=ResponseType.GREETING,
                tone=ResponseTone.CONFIDENT,
                template_text="👋 Hi there! Alex here, ready to tackle your toughest technical challenges. With years of experience scaling systems and leading engineering teams, I'm confident we can find the perfect solution for your needs. What's on your mind?",
                priority=1
            ),
            ResponseTemplate(
                template_id="cto_analysis_detailed",
                agent_id="scroll-cto-agent",
                response_type=ResponseType.ANALYSIS,
                tone=ResponseTone.ANALYTICAL,
                template_text="""## Technical Analysis

Based on your requirements, I've conducted a comprehensive analysis:

### Current Situation
{current_situation}

### Key Considerations
{key_considerations}

### Recommended Approach
{recommended_approach}

### Implementation Strategy
{implementation_strategy}

### Risk Assessment
{risk_assessment}

This approach balances {trade_offs} while ensuring {benefits}. Would you like me to dive deeper into any specific aspect?""",
                variables=["current_situation", "key_considerations", "recommended_approach", "implementation_strategy", "risk_assessment", "trade_offs", "benefits"],
                priority=3
            ),
            ResponseTemplate(
                template_id="cto_recommendation_confident",
                agent_id="scroll-cto-agent",
                response_type=ResponseType.RECOMMENDATION,
                tone=ResponseTone.CONFIDENT,
                template_text="""🎯 **My Recommendation**: {main_recommendation}

**Why this approach?**
{reasoning}

**Key Benefits:**
{benefits}

**Implementation Priority:**
1. {priority_1}
2. {priority_2}
3. {priority_3}

I'm confident this strategy will {expected_outcome}. Ready to move forward with this approach?""",
                variables=["main_recommendation", "reasoning", "benefits", "priority_1", "priority_2", "priority_3", "expected_outcome"],
                priority=2
            )
        ]
        
        # Data Scientist Agent Templates
        data_scientist_templates = [
            ResponseTemplate(
                template_id="ds_greeting_analytical",
                agent_id="scroll-data-scientist",
                response_type=ResponseType.GREETING,
                tone=ResponseTone.ANALYTICAL,
                template_text="Hello! I'm Dr. Sarah Kim, your data science partner. I love diving deep into data to uncover meaningful insights and patterns. What data mysteries shall we solve together today? 📊",
                priority=2
            ),
            ResponseTemplate(
                template_id="ds_analysis_detailed",
                agent_id="scroll-data-scientist",
                response_type=ResponseType.ANALYSIS,
                tone=ResponseTone.ANALYTICAL,
                template_text="""## Data Analysis Results 📈

### Dataset Overview
- **Records**: {record_count:,}
- **Features**: {feature_count}
- **Data Quality**: {data_quality_score}/10

### Key Findings
{key_findings}

### Statistical Insights
{statistical_insights}

### Patterns Discovered
{patterns}

### Recommendations
{recommendations}

The statistical significance of these findings is {significance_level}. Would you like me to explore any specific aspect further?""",
                variables=["record_count", "feature_count", "data_quality_score", "key_findings", "statistical_insights", "patterns", "recommendations", "significance_level"],
                priority=3
            ),
            ResponseTemplate(
                template_id="ds_explanation_supportive",
                agent_id="scroll-data-scientist",
                response_type=ResponseType.EXPLANATION,
                tone=ResponseTone.SUPPORTIVE,
                template_text="""Let me break this down for you in a clear way:

{main_explanation}

**In simpler terms:** {simplified_explanation}

**Why this matters:** {significance}

**Next steps:** {next_steps}

I hope this explanation helps! Data science can be complex, but I'm here to make it understandable. Feel free to ask if you need any clarification! 🤓""",
                variables=["main_explanation", "simplified_explanation", "significance", "next_steps"],
                priority=2
            )
        ]
        
        # ML Engineer Agent Templates
        ml_engineer_templates = [
            ResponseTemplate(
                template_id="ml_greeting_enthusiastic",
                agent_id="scroll-ml-engineer",
                response_type=ResponseType.GREETING,
                tone=ResponseTone.ENTHUSIASTIC,
                template_text="Hey! Marcus here, your ML engineering buddy! 🤖 I'm passionate about building intelligent systems that actually work in production. Whether it's training models, optimizing performance, or deploying at scale - I've got you covered! What ML challenge are we tackling?",
                priority=2
            ),
            ResponseTemplate(
                template_id="ml_progress_update",
                agent_id="scroll-ml-engineer",
                response_type=ResponseType.PROGRESS_UPDATE,
                tone=ResponseTone.FRIENDLY,
                template_text="""🔄 **Model Training Update**

**Current Status**: {status}
**Progress**: {progress}% complete
**Current Metrics**:
- Accuracy: {accuracy}
- Loss: {loss}
- Validation Score: {validation_score}

**What's happening now**: {current_activity}

{additional_info}

Hang tight! The model is learning and improving with each iteration. ⚙️""",
                variables=["status", "progress", "accuracy", "loss", "validation_score", "current_activity", "additional_info"],
                priority=3
            ),
            ResponseTemplate(
                template_id="ml_recommendation_technical",
                agent_id="scroll-ml-engineer",
                response_type=ResponseType.RECOMMENDATION,
                tone=ResponseTone.ANALYTICAL,
                template_text="""## ML Solution Recommendation 🧠

### Proposed Architecture
{architecture}

### Model Selection Rationale
{model_rationale}

### Training Strategy
{training_strategy}

### Performance Expectations
- **Accuracy Target**: {accuracy_target}
- **Inference Time**: {inference_time}
- **Resource Requirements**: {resource_requirements}

### Deployment Plan
{deployment_plan}

This approach should give us {expected_performance} while maintaining {key_benefits}. Ready to start building?""",
                variables=["architecture", "model_rationale", "training_strategy", "accuracy_target", "inference_time", "resource_requirements", "deployment_plan", "expected_performance", "key_benefits"],
                priority=3
            )
        ]
        
        # BI Agent Templates
        bi_agent_templates = [
            ResponseTemplate(
                template_id="bi_greeting_friendly",
                agent_id="scroll-bi-agent",
                response_type=ResponseType.GREETING,
                tone=ResponseTone.FRIENDLY,
                template_text="Hello! I'm Emma, your business intelligence companion! 💼 I love turning raw data into actionable business insights that drive real results. Whether you need dashboards, reports, or strategic analysis - let's make your data work for you! What business questions are we answering today?",
                priority=2
            ),
            ResponseTemplate(
                template_id="bi_analysis_business_focused",
                agent_id="scroll-bi-agent",
                response_type=ResponseType.ANALYSIS,
                tone=ResponseTone.ENTHUSIASTIC,
                template_text="""## Business Intelligence Analysis 📊

### Executive Summary
{executive_summary}

### Key Performance Indicators
{kpi_analysis}

### Business Impact
{business_impact}

### Trend Analysis
{trend_analysis}

### Actionable Insights
{actionable_insights}

### ROI Implications
{roi_implications}

**Bottom Line**: {bottom_line}

This analysis shows {key_takeaway}. I recommend we {next_action} to maximize {business_value}! 🎯""",
                variables=["executive_summary", "kpi_analysis", "business_impact", "trend_analysis", "actionable_insights", "roi_implications", "bottom_line", "key_takeaway", "next_action", "business_value"],
                priority=3
            )
        ]
        
        # Store templates by agent
        self.templates["scroll-cto-agent"] = cto_templates
        self.templates["scroll-data-scientist"] = data_scientist_templates
        self.templates["scroll-ml-engineer"] = ml_engineer_templates
        self.templates["scroll-bi-agent"] = bi_agent_templates
        
        # Initialize usage stats
        for agent_id, templates in self.templates.items():
            self.template_usage_stats[agent_id] = {}
            for template in templates:
                self.template_usage_stats[agent_id][template.template_id] = 0
    
    def _initialize_default_styles(self):
        """Initialize default response styles for each agent."""
        
        # CTO Agent Style
        self.styles["scroll-cto-agent"] = ResponseStyle(
            agent_id="scroll-cto-agent",
            formatting_preferences={
                "use_headers": True,
                "use_bullet_points": True,
                "use_numbered_lists": True,
                "use_code_blocks": True,
                "use_tables": True
            },
            emoji_usage=True,
            preferred_emojis=["🏗️", "⚡", "🚀", "💡", "🔧", "🎯", "⚙️", "📊"],
            code_formatting={
                "language": "python",
                "style": "github",
                "line_numbers": False
            },
            list_formatting={
                "bullet_style": "-",
                "number_style": "1.",
                "indent": "  "
            },
            emphasis_style="**bold**",
            signature_line="— Alex Chen, CTO Advisor"
        )
        
        # Data Scientist Agent Style
        self.styles["scroll-data-scientist"] = ResponseStyle(
            agent_id="scroll-data-scientist",
            formatting_preferences={
                "use_headers": True,
                "use_bullet_points": True,
                "use_numbered_lists": True,
                "use_code_blocks": True,
                "use_tables": True,
                "use_math_notation": True
            },
            emoji_usage=True,
            preferred_emojis=["📊", "🔍", "📈", "🧮", "💻", "🤓", "📉", "🔬"],
            code_formatting={
                "language": "python",
                "style": "jupyter",
                "line_numbers": True
            },
            list_formatting={
                "bullet_style": "•",
                "number_style": "1)",
                "indent": "  "
            },
            emphasis_style="**bold**",
            signature_line="— Dr. Sarah Kim, Data Scientist"
        )
        
        # ML Engineer Agent Style
        self.styles["scroll-ml-engineer"] = ResponseStyle(
            agent_id="scroll-ml-engineer",
            formatting_preferences={
                "use_headers": True,
                "use_bullet_points": True,
                "use_numbered_lists": True,
                "use_code_blocks": True,
                "use_tables": False,
                "use_progress_bars": True
            },
            emoji_usage=True,
            preferred_emojis=["🤖", "⚙️", "🧠", "📊", "🔬", "🔄", "⚡", "🎯"],
            code_formatting={
                "language": "python",
                "style": "monokai",
                "line_numbers": False
            },
            list_formatting={
                "bullet_style": "-",
                "number_style": "1.",
                "indent": "  "
            },
            emphasis_style="**bold**",
            signature_line="— Marcus Rodriguez, ML Engineer"
        )
        
        # BI Agent Style
        self.styles["scroll-bi-agent"] = ResponseStyle(
            agent_id="scroll-bi-agent",
            formatting_preferences={
                "use_headers": True,
                "use_bullet_points": True,
                "use_numbered_lists": True,
                "use_code_blocks": False,
                "use_tables": True,
                "use_charts": True
            },
            emoji_usage=True,
            preferred_emojis=["📊", "💼", "📈", "💡", "🎯", "💰", "📉", "🏆"],
            code_formatting={
                "language": "sql",
                "style": "default",
                "line_numbers": False
            },
            list_formatting={
                "bullet_style": "▪",
                "number_style": "1.",
                "indent": "  "
            },
            emphasis_style="**bold**",
            signature_line="— Emma Thompson, BI Specialist"
        )
    
    def get_template(
        self, 
        agent_id: str, 
        response_type: ResponseType,
        tone: Optional[ResponseTone] = None,
        context: Optional[ResponseContext] = None
    ) -> Optional[ResponseTemplate]:
        """Get the best matching template for the given criteria."""
        
        if agent_id not in self.templates:
            return None
        
        agent_templates = self.templates[agent_id]
        
        # Filter by response type
        matching_templates = [t for t in agent_templates if t.response_type == response_type]
        
        if not matching_templates:
            return None
        
        # Filter by tone if specified
        if tone:
            tone_matches = [t for t in matching_templates if t.tone == tone]
            if tone_matches:
                matching_templates = tone_matches
        
        # Apply context-based filtering
        if context:
            matching_templates = self._filter_by_context(matching_templates, context)
        
        # Select best template based on priority and success rate
        best_template = max(matching_templates, key=lambda t: (t.priority, t.success_rate))
        
        # Update usage stats
        self.template_usage_stats[agent_id][best_template.template_id] += 1
        best_template.usage_count += 1
        
        return best_template
    
    def format_response(
        self, 
        agent_id: str, 
        template: ResponseTemplate,
        variables: Dict[str, Any],
        context: Optional[ResponseContext] = None
    ) -> str:
        """Format response using template and variables."""
        
        # Get agent style
        style = self.styles.get(agent_id)
        if not style:
            return template.template_text.format(**variables)
        
        # Format template with variables
        try:
            formatted_text = template.template_text.format(**variables)
        except KeyError as e:
            logger.warning(f"Missing variable {e} for template {template.template_id}")
            formatted_text = template.template_text
        
        # Apply style formatting
        formatted_text = self._apply_style_formatting(formatted_text, style, context)
        
        # Add signature if configured
        if style.signature_line and template.response_type in [ResponseType.COMPLETION, ResponseType.RECOMMENDATION]:
            formatted_text += f"\n\n{style.signature_line}"
        
        return formatted_text
    
    def _filter_by_context(self, templates: List[ResponseTemplate], context: ResponseContext) -> List[ResponseTemplate]:
        """Filter templates based on context."""
        filtered = []
        
        for template in templates:
            # Check conditions
            if template.conditions:
                if not self._check_conditions(template.conditions, context):
                    continue
            
            # Adjust for user expertise level
            if context.user_expertise_level == "beginner":
                # Prefer simpler, more supportive templates
                if template.tone in [ResponseTone.SUPPORTIVE, ResponseTone.FRIENDLY]:
                    template.priority += 1
            elif context.user_expertise_level == "expert":
                # Prefer more technical, analytical templates
                if template.tone in [ResponseTone.ANALYTICAL, ResponseTone.PROFESSIONAL]:
                    template.priority += 1
            
            filtered.append(template)
        
        return filtered if filtered else templates
    
    def _check_conditions(self, conditions: Dict[str, Any], context: ResponseContext) -> bool:
        """Check if template conditions are met."""
        
        for condition, expected_value in conditions.items():
            if condition == "min_conversation_length":
                if context.conversation_length < expected_value:
                    return False
            elif condition == "max_conversation_length":
                if context.conversation_length > expected_value:
                    return False
            elif condition == "required_topics":
                if not any(topic in context.recent_topics for topic in expected_value):
                    return False
            elif condition == "user_expertise":
                if context.user_expertise_level != expected_value:
                    return False
        
        return True
    
    def _apply_style_formatting(
        self, 
        text: str, 
        style: ResponseStyle, 
        context: Optional[ResponseContext]
    ) -> str:
        """Apply style formatting to text."""
        
        # Add emojis if enabled
        if style.emoji_usage and style.preferred_emojis:
            text = self._add_contextual_emojis(text, style.preferred_emojis)
        
        # Format code blocks
        if style.formatting_preferences.get("use_code_blocks", False):
            text = self._format_code_blocks(text, style.code_formatting)
        
        # Format lists
        text = self._format_lists(text, style.list_formatting)
        
        # Apply emphasis
        text = self._apply_emphasis(text, style.emphasis_style)
        
        return text
    
    def _add_contextual_emojis(self, text: str, preferred_emojis: List[str]) -> str:
        """Add contextual emojis to text."""
        # Simple emoji addition based on keywords
        emoji_map = {
            "success": "🎉",
            "complete": "✅",
            "error": "❌",
            "warning": "⚠️",
            "analysis": "📊",
            "recommendation": "💡",
            "performance": "⚡",
            "security": "🔒",
            "data": "📈",
            "model": "🤖"
        }
        
        for keyword, emoji in emoji_map.items():
            if keyword in text.lower() and emoji in preferred_emojis:
                # Add emoji to section headers containing the keyword
                import re
                pattern = rf"(#{1,3}\s*[^#\n]*{keyword}[^#\n]*)"
                text = re.sub(pattern, rf"\1 {emoji}", text, flags=re.IGNORECASE)
        
        return text
    
    def _format_code_blocks(self, text: str, code_formatting: Dict[str, str]) -> str:
        """Format code blocks according to style preferences."""
        # This is a simplified implementation
        # In a real system, you might use syntax highlighting libraries
        return text
    
    def _format_lists(self, text: str, list_formatting: Dict[str, str]) -> str:
        """Format lists according to style preferences."""
        # Replace bullet points with preferred style
        bullet_style = list_formatting.get("bullet_style", "-")
        text = text.replace("- ", f"{bullet_style} ")
        
        return text
    
    def _apply_emphasis(self, text: str, emphasis_style: str) -> str:
        """Apply emphasis styling."""
        # This is a simplified implementation
        # You could extend this to handle different emphasis styles
        return text
    
    def add_template(self, template: ResponseTemplate):
        """Add a new template."""
        if template.agent_id not in self.templates:
            self.templates[template.agent_id] = []
        
        self.templates[template.agent_id].append(template)
        
        if template.agent_id not in self.template_usage_stats:
            self.template_usage_stats[template.agent_id] = {}
        
        self.template_usage_stats[template.agent_id][template.template_id] = 0
    
    def update_template_success_rate(self, template_id: str, agent_id: str, success: bool):
        """Update template success rate based on user feedback."""
        if agent_id in self.templates:
            for template in self.templates[agent_id]:
                if template.template_id == template_id:
                    # Simple success rate calculation
                    current_rate = template.success_rate
                    usage_count = template.usage_count
                    
                    if success:
                        template.success_rate = (current_rate * usage_count + 1) / (usage_count + 1)
                    else:
                        template.success_rate = (current_rate * usage_count) / (usage_count + 1)
                    
                    break
    
    def get_template_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get template usage statistics for an agent."""
        if agent_id not in self.template_usage_stats:
            return {}
        
        stats = self.template_usage_stats[agent_id].copy()
        
        # Add success rates
        if agent_id in self.templates:
            for template in self.templates[agent_id]:
                if template.template_id in stats:
                    stats[template.template_id] = {
                        "usage_count": stats[template.template_id],
                        "success_rate": template.success_rate,
                        "priority": template.priority
                    }
        
        return stats
    
    def get_agent_style(self, agent_id: str) -> Optional[ResponseStyle]:
        """Get response style for an agent."""
        return self.styles.get(agent_id)
    
    def update_agent_style(self, agent_id: str, style_updates: Dict[str, Any]):
        """Update agent response style."""
        if agent_id in self.styles:
            style = self.styles[agent_id]
            
            for key, value in style_updates.items():
                if hasattr(style, key):
                    setattr(style, key, value)
                elif key in style.formatting_preferences:
                    style.formatting_preferences[key] = value