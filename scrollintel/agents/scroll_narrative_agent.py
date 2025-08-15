"""
ScrollNarrativeAgent - Insight Storytelling and Policy Briefs
Transform data insights into compelling narratives, stories, and policy briefs.
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from uuid import uuid4
from dataclasses import dataclass
from enum import Enum
import logging

# AI libraries
try:
    import openai
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

from scrollintel.core.interfaces import BaseAgent, AgentType, AgentRequest, AgentResponse, AgentCapability, ResponseStatus

logger = logging.getLogger(__name__)


class NarrativeType(str, Enum):
    """Types of narratives that can be generated."""
    DATA_STORY = "data_story"
    EXECUTIVE_SUMMARY = "executive_summary"
    POLICY_BRIEF = "policy_brief"
    RESEARCH_REPORT = "research_report"
    BUSINESS_CASE = "business_case"
    TECHNICAL_BRIEF = "technical_brief"
    STAKEHOLDER_UPDATE = "stakeholder_update"
    PRESENTATION_SCRIPT = "presentation_script"
    PRESS_RELEASE = "press_release"
    WHITE_PAPER = "white_paper"


class AudienceType(str, Enum):
    """Target audience types."""
    EXECUTIVES = "executives"
    TECHNICAL_TEAM = "technical_team"
    BOARD_MEMBERS = "board_members"
    INVESTORS = "investors"
    CUSTOMERS = "customers"
    REGULATORS = "regulators"
    GENERAL_PUBLIC = "general_public"
    ACADEMIC = "academic"
    MEDIA = "media"
    INTERNAL_STAFF = "internal_staff"


class NarrativeStyle(str, Enum):
    """Narrative writing styles."""
    FORMAL = "formal"
    CONVERSATIONAL = "conversational"
    TECHNICAL = "technical"
    PERSUASIVE = "persuasive"
    ANALYTICAL = "analytical"
    STORYTELLING = "storytelling"
    JOURNALISTIC = "journalistic"
    ACADEMIC = "academic"


class ContentFormat(str, Enum):
    """Output content formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    PLAIN_TEXT = "plain_text"
    SLIDE_DECK = "slide_deck"
    PDF_REPORT = "pdf_report"
    WORD_DOCUMENT = "word_document"
    PRESENTATION = "presentation"


@dataclass
class NarrativeRequest:
    """Request for narrative generation."""
    id: str
    narrative_type: NarrativeType
    audience: AudienceType
    style: NarrativeStyle
    format: ContentFormat
    data_insights: Dict[str, Any]
    key_messages: List[str]
    context: Dict[str, Any]
    length_target: str  # "short", "medium", "long"
    tone: str  # "neutral", "positive", "urgent", "optimistic"
    branding: Dict[str, Any]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class GeneratedNarrative:
    """Generated narrative content."""
    id: str
    request_id: str
    title: str
    content: str
    executive_summary: str
    key_points: List[str]
    supporting_data: Dict[str, Any]
    recommendations: List[str]
    call_to_action: str
    metadata: Dict[str, Any]
    word_count: int
    readability_score: float
    engagement_score: float
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.word_count == 0:
            self.word_count = len(self.content.split())


@dataclass
class StoryStructure:
    """Narrative story structure."""
    hook: str
    context: str
    conflict_challenge: str
    resolution: str
    implications: str
    call_to_action: str


class ScrollNarrativeAgent(BaseAgent):
    """Advanced narrative generation agent for data storytelling."""
    
    def __init__(self):
        super().__init__(
            agent_id="scroll-narrative-agent",
            name="ScrollNarrative Agent",
            agent_type=AgentType.AI_ENGINEER
        )
        
        self.capabilities = [
            AgentCapability(
                name="data_storytelling",
                description="Transform data insights into compelling narratives and stories",
                input_types=["data_insights", "analysis_results", "metrics"],
                output_types=["data_story", "narrative", "story_structure"]
            ),
            AgentCapability(
                name="policy_brief_generation",
                description="Generate policy briefs and executive summaries",
                input_types=["policy_analysis", "recommendations", "stakeholder_context"],
                output_types=["policy_brief", "executive_summary", "action_plan"]
            ),
            AgentCapability(
                name="stakeholder_communication",
                description="Create audience-specific communications and updates",
                input_types=["audience_profile", "key_messages", "communication_goals"],
                output_types=["stakeholder_update", "presentation_script", "communication_plan"]
            ),
            AgentCapability(
                name="presentation_content",
                description="Generate presentation content and slide narratives",
                input_types=["presentation_outline", "data_points", "audience_info"],
                output_types=["slide_content", "speaker_notes", "presentation_flow"]
            )
        ]
        
        # Initialize AI components
        if AI_AVAILABLE:
            self.openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            self.openai_client = None
        
        # Narrative state
        self.active_narratives = {}
        self.narrative_templates = {}
        self.audience_profiles = {}
        
        # Initialize narrative templates and audience profiles
        self._initialize_narrative_templates()
        self._initialize_audience_profiles()
    
    def _initialize_narrative_templates(self):
        """Initialize narrative templates for different types."""
        self.narrative_templates = {
            NarrativeType.DATA_STORY: {
                "structure": ["hook", "context", "data_revelation", "implications", "action"],
                "key_elements": ["compelling opening", "data visualization", "human impact", "clear conclusion"],
                "length_guidelines": {"short": 500, "medium": 1000, "long": 2000}
            },
            NarrativeType.EXECUTIVE_SUMMARY: {
                "structure": ["situation", "key_findings", "recommendations", "next_steps"],
                "key_elements": ["executive focus", "actionable insights", "clear recommendations"],
                "length_guidelines": {"short": 300, "medium": 600, "long": 1000}
            },
            NarrativeType.POLICY_BRIEF: {
                "structure": ["issue_overview", "analysis", "policy_options", "recommendations"],
                "key_elements": ["policy context", "evidence base", "implementation guidance"],
                "length_guidelines": {"short": 800, "medium": 1500, "long": 3000}
            }
        }
    
    def _initialize_audience_profiles(self):
        """Initialize audience-specific communication profiles."""
        self.audience_profiles = {
            AudienceType.EXECUTIVES: {
                "communication_style": "concise and action-oriented",
                "key_interests": ["ROI", "strategic impact", "risk mitigation", "competitive advantage"],
                "preferred_format": "executive summary with key metrics",
                "attention_span": "short",
                "decision_factors": ["business impact", "resource requirements", "timeline"]
            },
            AudienceType.TECHNICAL_TEAM: {
                "communication_style": "detailed and technical",
                "key_interests": ["implementation details", "technical feasibility", "architecture"],
                "preferred_format": "technical documentation with specifications",
                "attention_span": "long",
                "decision_factors": ["technical merit", "implementation complexity", "maintainability"]
            },
            AudienceType.BOARD_MEMBERS: {
                "communication_style": "formal and strategic",
                "key_interests": ["governance", "risk management", "strategic alignment"],
                "preferred_format": "formal presentation with supporting data",
                "attention_span": "medium",
                "decision_factors": ["strategic fit", "risk profile", "stakeholder impact"]
            }
        }
    
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """Process narrative generation requests."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            prompt = request.prompt.lower()
            context = request.context or {}
            
            if "story" in prompt or "narrative" in prompt:
                content = await self._generate_data_story(request.prompt, context)
            elif "policy" in prompt or "brief" in prompt:
                content = await self._generate_policy_brief(request.prompt, context)
            elif "executive" in prompt or "summary" in prompt:
                content = await self._generate_executive_summary(request.prompt, context)
            elif "presentation" in prompt or "slides" in prompt:
                content = await self._generate_presentation_content(request.prompt, context)
            elif "stakeholder" in prompt or "communication" in prompt:
                content = await self._generate_stakeholder_communication(request.prompt, context)
            else:
                content = await self._generate_general_narrative(request.prompt, context)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return AgentResponse(
                id=f"narrative-{uuid4()}",
                request_id=request.id,
                content=content,
                artifacts=[],
                execution_time=execution_time,
                status=ResponseStatus.SUCCESS
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return AgentResponse(
                id=f"narrative-{uuid4()}",
                request_id=request.id,
                content=f"Error in narrative generation: {str(e)}",
                artifacts=[],
                execution_time=execution_time,
                status=ResponseStatus.ERROR,
                error_message=str(e)
            )
    
    async def _generate_data_story(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate compelling data story."""
        data_insights = context.get("data_insights", {})
        audience = AudienceType(context.get("audience", AudienceType.EXECUTIVES))
        style = NarrativeStyle(context.get("style", NarrativeStyle.STORYTELLING))
        
        # Create narrative request
        narrative_request = NarrativeRequest(
            id=f"story-req-{uuid4()}",
            narrative_type=NarrativeType.DATA_STORY,
            audience=audience,
            style=style,
            format=ContentFormat.MARKDOWN,
            data_insights=data_insights,
            key_messages=context.get("key_messages", []),
            context=context,
            length_target=context.get("length", "medium"),
            tone=context.get("tone", "engaging"),
            branding=context.get("branding", {})
        )
        
        # Generate story structure
        story_structure = await self._create_story_structure(data_insights, narrative_request)
        
        # Generate narrative content
        narrative = await self._generate_narrative_content(story_structure, narrative_request)
        
        # Store narrative
        self.active_narratives[narrative.id] = narrative
        
        return f"""
# {narrative.title}

## Executive Summary
{narrative.executive_summary}

## The Story

### Opening Hook
{story_structure.hook}

### Context and Background
{story_structure.context}

### The Challenge
{story_structure.conflict_challenge}

### Data Insights and Resolution
{story_structure.resolution}

### Implications and Impact
{story_structure.implications}

### Call to Action
{story_structure.call_to_action}

## Key Takeaways
{chr(10).join(f"- {point}" for point in narrative.key_points)}

## Supporting Data
{await self._format_supporting_data(narrative.supporting_data)}

## Recommendations
{chr(10).join(f"- {rec}" for rec in narrative.recommendations)}

## Narrative Metrics
- **Word Count**: {narrative.word_count}
- **Readability Score**: {narrative.readability_score:.1f}/10
- **Engagement Score**: {narrative.engagement_score:.1f}/10
- **Target Audience**: {audience.value}
- **Narrative Style**: {style.value}
"""
    
    async def _generate_policy_brief(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate policy brief."""
        policy_issue = context.get("policy_issue", prompt)
        stakeholders = context.get("stakeholders", [])
        policy_options = context.get("policy_options", [])
        
        # Create narrative request
        narrative_request = NarrativeRequest(
            id=f"policy-req-{uuid4()}",
            narrative_type=NarrativeType.POLICY_BRIEF,
            audience=AudienceType.REGULATORS,
            style=NarrativeStyle.FORMAL,
            format=ContentFormat.PDF_REPORT,
            data_insights=context.get("analysis_data", {}),
            key_messages=context.get("key_messages", []),
            context=context,
            length_target="long",
            tone="authoritative",
            branding=context.get("branding", {})
        )
        
        # Generate policy brief content
        brief_content = await self._create_policy_brief_content(policy_issue, policy_options, narrative_request)
        
        return f"""
# Policy Brief: {policy_issue}

## Executive Summary
{brief_content.get("executive_summary", "Policy brief executive summary")}

## Issue Overview
{brief_content.get("issue_overview", "Detailed issue analysis")}

## Stakeholder Analysis
{await self._format_stakeholder_analysis(stakeholders)}

## Policy Options Analysis
{await self._format_policy_options(policy_options)}

## Evidence Base
{brief_content.get("evidence_base", "Supporting evidence and data")}

## Recommendations
{brief_content.get("recommendations", "Policy recommendations")}

## Implementation Considerations
{brief_content.get("implementation", "Implementation guidance")}

## Risk Assessment
{brief_content.get("risk_assessment", "Risk analysis")}

## Next Steps
{brief_content.get("next_steps", "Recommended actions")}

## Appendices
{brief_content.get("appendices", "Supporting materials")}
"""
    
    async def _generate_executive_summary(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate executive summary."""
        source_content = context.get("source_content", prompt)
        key_findings = context.get("key_findings", [])
        recommendations = context.get("recommendations", [])
        
        # Create narrative request
        narrative_request = NarrativeRequest(
            id=f"exec-req-{uuid4()}",
            narrative_type=NarrativeType.EXECUTIVE_SUMMARY,
            audience=AudienceType.EXECUTIVES,
            style=NarrativeStyle.FORMAL,
            format=ContentFormat.MARKDOWN,
            data_insights=context.get("data_insights", {}),
            key_messages=key_findings,
            context=context,
            length_target="short",
            tone="professional",
            branding=context.get("branding", {})
        )
        
        # Generate executive summary
        summary = await self._create_executive_summary_content(source_content, narrative_request)
        
        return f"""
# Executive Summary

## Situation Overview
{summary.get("situation", "Current situation analysis")}

## Key Findings
{chr(10).join(f"- {finding}" for finding in key_findings)}

## Strategic Implications
{summary.get("implications", "Strategic implications for the organization")}

## Recommendations
{chr(10).join(f"- {rec}" for rec in recommendations)}

## Resource Requirements
{summary.get("resources", "Required resources and investment")}

## Timeline and Milestones
{summary.get("timeline", "Implementation timeline")}

## Risk Considerations
{summary.get("risks", "Key risks and mitigation strategies")}

## Expected Outcomes
{summary.get("outcomes", "Anticipated results and benefits")}

## Next Steps
{summary.get("next_steps", "Immediate actions required")}
"""
    
    async def _create_story_structure(self, data_insights: Dict[str, Any], request: NarrativeRequest) -> StoryStructure:
        """Create compelling story structure from data insights."""
        if self.openai_client:
            # Use AI to create story structure
            story_prompt = f"""
            Create a compelling story structure for a data narrative with the following insights:
            {json.dumps(data_insights, indent=2)}
            
            Audience: {request.audience.value}
            Style: {request.style.value}
            Tone: {request.tone}
            
            Create a story with:
            1. Hook - compelling opening that grabs attention
            2. Context - background and setting
            3. Conflict/Challenge - the problem or opportunity
            4. Resolution - how data provides insights/solutions
            5. Implications - what this means for the audience
            6. Call to Action - what should be done next
            """
            
            try:
                ai_response = await self._call_openai(story_prompt)
                # Parse AI response into story structure
                return await self._parse_story_structure(ai_response)
            except Exception as e:
                logger.warning(f"AI story generation failed: {e}")
        
        # Fallback to template-based story structure
        return StoryStructure(
            hook="Data reveals surprising insights that could transform our approach",
            context="In today's data-driven environment, understanding patterns is crucial",
            conflict_challenge="Traditional methods are missing key opportunities",
            resolution="Our analysis shows clear pathways to improvement",
            implications="These findings have significant strategic implications",
            call_to_action="We recommend immediate action to capitalize on these insights"
        )
    
    async def _generate_narrative_content(self, story_structure: StoryStructure, request: NarrativeRequest) -> GeneratedNarrative:
        """Generate full narrative content from story structure."""
        # Combine story structure into full narrative
        full_content = f"""
{story_structure.hook}

{story_structure.context}

{story_structure.conflict_challenge}

{story_structure.resolution}

{story_structure.implications}

{story_structure.call_to_action}
"""
        
        # Create generated narrative
        narrative = GeneratedNarrative(
            id=f"narrative-{uuid4()}",
            request_id=request.id,
            title=await self._generate_title(request),
            content=full_content,
            executive_summary=await self._create_executive_summary_from_content(full_content),
            key_points=await self._extract_key_points(full_content),
            supporting_data=request.data_insights,
            recommendations=await self._extract_recommendations(full_content),
            call_to_action=story_structure.call_to_action,
            metadata={"audience": request.audience.value, "style": request.style.value},
            word_count=0,  # Will be calculated in __post_init__
            readability_score=await self._calculate_readability_score(full_content),
            engagement_score=await self._calculate_engagement_score(full_content, request.audience)
        )
        
        return narrative
    
    async def _call_openai(self, prompt: str, max_tokens: int = 2000) -> str:
        """Call OpenAI API for narrative generation."""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert storyteller and communication specialist who creates compelling narratives from data and analysis."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API call failed: {str(e)}")
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Return agent capabilities."""
        return self.capabilities
    
    async def health_check(self) -> bool:
        """Check agent health."""
        return True
    
    # Helper methods (simplified implementations)
    async def _parse_story_structure(self, ai_response: str) -> StoryStructure:
        """Parse AI response into story structure."""
        # Mock parsing - in production would parse structured AI response
        return StoryStructure(
            hook="AI-generated compelling hook",
            context="AI-generated context",
            conflict_challenge="AI-identified challenge",
            resolution="AI-suggested resolution",
            implications="AI-analyzed implications",
            call_to_action="AI-recommended action"
        )
    
    async def _generate_title(self, request: NarrativeRequest) -> str:
        """Generate appropriate title for narrative."""
        titles = {
            NarrativeType.DATA_STORY: "Data-Driven Insights: Transforming Information into Action",
            NarrativeType.EXECUTIVE_SUMMARY: "Executive Summary: Strategic Analysis and Recommendations",
            NarrativeType.POLICY_BRIEF: "Policy Brief: Analysis and Recommendations"
        }
        return titles.get(request.narrative_type, "ScrollIntel Narrative Report")
    
    async def _calculate_readability_score(self, content: str) -> float:
        """Calculate readability score (simplified)."""
        # Mock readability calculation
        word_count = len(content.split())
        sentence_count = content.count('.') + content.count('!') + content.count('?')
        avg_words_per_sentence = word_count / max(sentence_count, 1)
        
        # Simple readability score (higher is more readable)
        if avg_words_per_sentence < 15:
            return 8.5
        elif avg_words_per_sentence < 20:
            return 7.0
        else:
            return 5.5
    
    async def _calculate_engagement_score(self, content: str, audience: AudienceType) -> float:
        """Calculate engagement score based on content and audience."""
        # Mock engagement calculation
        engagement_factors = {
            "questions": content.count('?') * 0.5,
            "action_words": len([w for w in content.lower().split() if w in ['action', 'implement', 'achieve', 'improve']]) * 0.3,
            "data_points": content.count('%') + content.count('$') * 0.2
        }
        
        base_score = sum(engagement_factors.values())
        
        # Adjust for audience
        if audience == AudienceType.EXECUTIVES:
            base_score += 1.0  # Executives prefer action-oriented content
        elif audience == AudienceType.TECHNICAL_TEAM:
            base_score += 0.5  # Technical teams prefer detailed content
        
        return min(10.0, max(1.0, base_score + 6.0))  # Normalize to 1-10 scale