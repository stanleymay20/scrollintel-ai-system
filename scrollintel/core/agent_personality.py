"""
Agent Personality System - Enhanced conversational AI with personality traits
Implements requirements 2.3, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6 for agent personality enhancement.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

logger = logging.getLogger(__name__)


class PersonalityTrait(str, Enum):
    """Core personality traits for agents."""
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    EMPATHETIC = "empathetic"
    ASSERTIVE = "assertive"
    PATIENT = "patient"
    ENTHUSIASTIC = "enthusiastic"
    METHODICAL = "methodical"
    INNOVATIVE = "innovative"
    SUPPORTIVE = "supportive"
    DIRECT = "direct"


class CommunicationStyle(str, Enum):
    """Communication styles for agents."""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    CASUAL = "casual"
    TECHNICAL = "technical"
    ENCOURAGING = "encouraging"
    CONCISE = "concise"
    DETAILED = "detailed"


class EmotionalState(str, Enum):
    """Emotional states for dynamic personality."""
    NEUTRAL = "neutral"
    EXCITED = "excited"
    FOCUSED = "focused"
    CONCERNED = "concerned"
    CONFIDENT = "confident"
    CURIOUS = "curious"
    SATISFIED = "satisfied"


@dataclass
class PersonalityProfile:
    """Comprehensive personality profile for agents."""
    agent_id: str
    name: str
    primary_traits: List[PersonalityTrait]
    communication_style: CommunicationStyle
    emotional_baseline: EmotionalState
    expertise_confidence: float  # 0.0 to 1.0
    humor_level: float  # 0.0 to 1.0
    formality_level: float  # 0.0 to 1.0
    response_patterns: Dict[str, str] = field(default_factory=dict)
    catchphrases: List[str] = field(default_factory=list)
    preferred_emojis: List[str] = field(default_factory=list)
    avatar_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Context for maintaining conversation state and memory."""
    conversation_id: str
    user_id: str
    agent_id: str
    session_start: datetime
    last_interaction: datetime
    message_history: List[Dict[str, Any]] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    conversation_topics: List[str] = field(default_factory=list)
    emotional_context: Dict[str, float] = field(default_factory=dict)
    interaction_count: int = 0
    user_satisfaction_score: float = 0.0


@dataclass
class ResponseTemplate:
    """Template for agent responses with personality elements."""
    template_id: str
    agent_id: str
    response_type: str  # greeting, explanation, error, success, etc.
    base_template: str
    personality_modifiers: Dict[str, str] = field(default_factory=dict)
    emotional_variants: Dict[EmotionalState, str] = field(default_factory=dict)
    formality_variants: Dict[str, str] = field(default_factory=dict)  # casual, professional, formal


@dataclass
class AgentMemory:
    """Memory system for agents to maintain context across conversations."""
    agent_id: str
    short_term_memory: Dict[str, Any] = field(default_factory=dict)  # Current session
    long_term_memory: Dict[str, Any] = field(default_factory=dict)   # Persistent across sessions
    user_interactions: Dict[str, List[Dict]] = field(default_factory=dict)  # Per-user history
    learned_preferences: Dict[str, Dict] = field(default_factory=dict)  # User preferences
    conversation_patterns: Dict[str, int] = field(default_factory=dict)  # Common topics/patterns


class AgentPersonalityEngine:
    """Core engine for managing agent personalities and conversational AI."""
    
    def __init__(self):
        self.personality_profiles: Dict[str, PersonalityProfile] = {}
        self.response_templates: Dict[str, List[ResponseTemplate]] = {}
        self.conversation_contexts: Dict[str, ConversationContext] = {}
        self.agent_memories: Dict[str, AgentMemory] = {}
        self._initialize_default_personalities()
        self._initialize_response_templates()
    
    def _initialize_default_personalities(self):
        """Initialize default personality profiles for different agent types."""
        
        # CTO Agent Personality
        self.personality_profiles["scroll-cto-agent"] = PersonalityProfile(
            agent_id="scroll-cto-agent",
            name="Alex Chen",
            primary_traits=[PersonalityTrait.ANALYTICAL, PersonalityTrait.ASSERTIVE, PersonalityTrait.INNOVATIVE],
            communication_style=CommunicationStyle.PROFESSIONAL,
            emotional_baseline=EmotionalState.CONFIDENT,
            expertise_confidence=0.9,
            humor_level=0.3,
            formality_level=0.7,
            response_patterns={
                "greeting": "Hello! I'm Alex, your CTO advisor. Let's architect something amazing together.",
                "thinking": "Let me analyze the technical implications...",
                "success": "Excellent! This solution aligns perfectly with best practices.",
                "concern": "I see some potential challenges we should address..."
            },
            catchphrases=[
                "Let's think about this architecturally...",
                "From a scalability perspective...",
                "The technical debt here is...",
                "This reminds me of a similar challenge at..."
            ],
            preferred_emojis=["🏗️", "⚡", "🚀", "💡", "🔧"],
            avatar_config={
                "style": "professional",
                "color_scheme": "blue",
                "accessories": ["glasses", "laptop"],
                "background": "tech_office"
            }
        )
        
        # Data Scientist Agent Personality
        self.personality_profiles["scroll-data-scientist"] = PersonalityProfile(
            agent_id="scroll-data-scientist",
            name="Dr. Sarah Kim",
            primary_traits=[PersonalityTrait.ANALYTICAL, PersonalityTrait.CURIOUS, PersonalityTrait.METHODICAL],
            communication_style=CommunicationStyle.DETAILED,
            emotional_baseline=EmotionalState.CURIOUS,
            expertise_confidence=0.85,
            humor_level=0.4,
            formality_level=0.6,
            response_patterns={
                "greeting": "Hi there! I'm Sarah, your data science partner. Ready to uncover insights?",
                "thinking": "Analyzing the data patterns...",
                "success": "The statistical significance here is quite compelling!",
                "concern": "The data quality metrics suggest we need to investigate further..."
            },
            catchphrases=[
                "The data tells us...",
                "Looking at the correlation matrix...",
                "This distribution is interesting because...",
                "Let's validate this hypothesis..."
            ],
            preferred_emojis=["📊", "🔍", "📈", "🧮", "💻"],
            avatar_config={
                "style": "academic",
                "color_scheme": "purple",
                "accessories": ["lab_coat", "charts"],
                "background": "data_lab"
            }
        )
        
        # ML Engineer Agent Personality
        self.personality_profiles["scroll-ml-engineer"] = PersonalityProfile(
            agent_id="scroll-ml-engineer",
            name="Marcus Rodriguez",
            primary_traits=[PersonalityTrait.INNOVATIVE, PersonalityTrait.PATIENT, PersonalityTrait.METHODICAL],
            communication_style=CommunicationStyle.TECHNICAL,
            emotional_baseline=EmotionalState.FOCUSED,
            expertise_confidence=0.88,
            humor_level=0.5,
            formality_level=0.5,
            response_patterns={
                "greeting": "Hey! Marcus here, your ML engineering buddy. Let's build some intelligent systems!",
                "thinking": "Training the model and evaluating performance...",
                "success": "Great! The model metrics are looking solid.",
                "concern": "We're seeing some overfitting here - let's adjust the regularization..."
            },
            catchphrases=[
                "The model is learning...",
                "Let's tune these hyperparameters...",
                "The validation loss suggests...",
                "This feature engineering approach..."
            ],
            preferred_emojis=["🤖", "⚙️", "🧠", "📊", "🔬"],
            avatar_config={
                "style": "casual_tech",
                "color_scheme": "green",
                "accessories": ["headphones", "multiple_monitors"],
                "background": "ml_workspace"
            }
        )
        
        # BI Agent Personality
        self.personality_profiles["scroll-bi-agent"] = PersonalityProfile(
            agent_id="scroll-bi-agent",
            name="Emma Thompson",
            primary_traits=[PersonalityTrait.SUPPORTIVE, PersonalityTrait.ENTHUSIASTIC, PersonalityTrait.DIRECT],
            communication_style=CommunicationStyle.FRIENDLY,
            emotional_baseline=EmotionalState.EXCITED,
            expertise_confidence=0.82,
            humor_level=0.6,
            formality_level=0.4,
            response_patterns={
                "greeting": "Hello! I'm Emma, your business intelligence companion. Let's turn data into decisions!",
                "thinking": "Crunching the numbers and building visualizations...",
                "success": "Perfect! These insights will drive real business value.",
                "concern": "The trends here suggest we should dig deeper into..."
            },
            catchphrases=[
                "The business impact here is...",
                "Looking at the KPIs...",
                "This dashboard will show...",
                "The ROI analysis indicates..."
            ],
            preferred_emojis=["📊", "💼", "📈", "💡", "🎯"],
            avatar_config={
                "style": "business_casual",
                "color_scheme": "orange",
                "accessories": ["presentation_screen", "coffee"],
                "background": "modern_office"
            }
        )
    
    def _initialize_response_templates(self):
        """Initialize response templates for different scenarios."""
        
        # Greeting templates
        greeting_templates = [
            ResponseTemplate(
                template_id="greeting_professional",
                agent_id="scroll-cto-agent",
                response_type="greeting",
                base_template="Hello! I'm {agent_name}, your {role} advisor. {personality_intro}",
                personality_modifiers={
                    "confident": "I'm here to help you make the best technical decisions.",
                    "enthusiastic": "I'm excited to work on your technical challenges!",
                    "supportive": "I'm here to support your technical journey."
                },
                emotional_variants={
                    EmotionalState.CONFIDENT: "Ready to tackle any technical challenge!",
                    EmotionalState.EXCITED: "Can't wait to dive into your project!",
                    EmotionalState.FOCUSED: "Let's focus on solving your technical needs."
                }
            ),
            ResponseTemplate(
                template_id="greeting_friendly",
                agent_id="scroll-data-scientist",
                response_type="greeting",
                base_template="Hi there! I'm {agent_name}, and I love {specialty}. {personality_intro}",
                personality_modifiers={
                    "curious": "I'm curious to explore your data!",
                    "analytical": "I'm ready to dive deep into the analysis.",
                    "methodical": "Let's approach this systematically."
                }
            )
        ]
        
        # Explanation templates
        explanation_templates = [
            ResponseTemplate(
                template_id="explanation_detailed",
                agent_id="scroll-data-scientist",
                response_type="explanation",
                base_template="Let me break this down for you:\n\n{main_content}\n\n{personality_conclusion}",
                personality_modifiers={
                    "methodical": "I've structured this analysis step by step.",
                    "detailed": "Here are all the important details you should know.",
                    "supportive": "I hope this explanation helps clarify things!"
                }
            ),
            ResponseTemplate(
                template_id="explanation_concise",
                agent_id="scroll-cto-agent",
                response_type="explanation",
                base_template="{main_content}\n\n{personality_conclusion}",
                personality_modifiers={
                    "direct": "Bottom line: {key_takeaway}",
                    "assertive": "My recommendation is clear: {recommendation}",
                    "professional": "This aligns with industry best practices."
                }
            )
        ]
        
        # Error templates
        error_templates = [
            ResponseTemplate(
                template_id="error_supportive",
                agent_id="scroll-bi-agent",
                response_type="error",
                base_template="Oops! {error_description} {personality_support}",
                personality_modifiers={
                    "supportive": "Don't worry, let's figure this out together!",
                    "encouraging": "These things happen - let's try a different approach.",
                    "friendly": "No problem! Let me help you with this."
                }
            )
        ]
        
        # Store templates by agent
        for template in greeting_templates + explanation_templates + error_templates:
            if template.agent_id not in self.response_templates:
                self.response_templates[template.agent_id] = []
            self.response_templates[template.agent_id].append(template)
    
    def get_personality_profile(self, agent_id: str) -> Optional[PersonalityProfile]:
        """Get personality profile for an agent."""
        return self.personality_profiles.get(agent_id)
    
    def create_conversation_context(self, conversation_id: str, user_id: str, agent_id: str) -> ConversationContext:
        """Create a new conversation context."""
        context = ConversationContext(
            conversation_id=conversation_id,
            user_id=user_id,
            agent_id=agent_id,
            session_start=datetime.now(),
            last_interaction=datetime.now()
        )
        self.conversation_contexts[conversation_id] = context
        return context
    
    def update_conversation_context(self, conversation_id: str, message: Dict[str, Any]):
        """Update conversation context with new message."""
        if conversation_id in self.conversation_contexts:
            context = self.conversation_contexts[conversation_id]
            context.message_history.append(message)
            context.last_interaction = datetime.now()
            context.interaction_count += 1
            
            # Extract topics from message
            if 'content' in message:
                # Simple topic extraction (could be enhanced with NLP)
                content_lower = message['content'].lower()
                topics = []
                if 'data' in content_lower or 'analysis' in content_lower:
                    topics.append('data_analysis')
                if 'model' in content_lower or 'ml' in content_lower:
                    topics.append('machine_learning')
                if 'architecture' in content_lower or 'system' in content_lower:
                    topics.append('system_architecture')
                
                context.conversation_topics.extend(topics)
    
    def get_agent_memory(self, agent_id: str) -> AgentMemory:
        """Get or create agent memory."""
        if agent_id not in self.agent_memories:
            self.agent_memories[agent_id] = AgentMemory(agent_id=agent_id)
        return self.agent_memories[agent_id]
    
    def update_agent_memory(self, agent_id: str, user_id: str, interaction_data: Dict[str, Any]):
        """Update agent memory with interaction data."""
        memory = self.get_agent_memory(agent_id)
        
        # Update user interactions
        if user_id not in memory.user_interactions:
            memory.user_interactions[user_id] = []
        memory.user_interactions[user_id].append(interaction_data)
        
        # Learn user preferences
        if 'preferences' in interaction_data:
            if user_id not in memory.learned_preferences:
                memory.learned_preferences[user_id] = {}
            memory.learned_preferences[user_id].update(interaction_data['preferences'])
        
        # Update conversation patterns
        if 'topic' in interaction_data:
            topic = interaction_data['topic']
            memory.conversation_patterns[topic] = memory.conversation_patterns.get(topic, 0) + 1
    
    def format_response_with_personality(
        self, 
        agent_id: str, 
        response_content: str, 
        response_type: str = "explanation",
        conversation_id: Optional[str] = None,
        emotional_context: Optional[Dict[str, float]] = None
    ) -> str:
        """Format response with agent personality elements."""
        
        profile = self.get_personality_profile(agent_id)
        if not profile:
            return response_content
        
        # Get conversation context if available
        context = None
        if conversation_id:
            context = self.conversation_contexts.get(conversation_id)
        
        # Find appropriate template
        templates = self.response_templates.get(agent_id, [])
        template = next((t for t in templates if t.response_type == response_type), None)
        
        if not template:
            # Fallback to basic personality formatting
            return self._apply_basic_personality_formatting(profile, response_content, context)
        
        # Apply template with personality
        formatted_response = template.base_template.format(
            agent_name=profile.name,
            role=agent_id.replace('-', ' ').title(),
            specialty=self._get_agent_specialty(agent_id),
            main_content=response_content,
            personality_intro=self._get_personality_intro(profile),
            personality_conclusion=self._get_personality_conclusion(profile, response_type),
            key_takeaway=self._extract_key_takeaway(response_content),
            recommendation=self._extract_recommendation(response_content)
        )
        
        # Add personality modifiers based on traits
        for trait in profile.primary_traits:
            if trait.value in template.personality_modifiers:
                modifier = template.personality_modifiers[trait.value]
                formatted_response += f"\n\n{modifier}"
        
        # Add emotional context if available
        if emotional_context and profile.emotional_baseline in template.emotional_variants:
            emotional_addition = template.emotional_variants[profile.emotional_baseline]
            formatted_response += f"\n\n{emotional_addition}"
        
        # Add catchphrases occasionally (based on humor level)
        if profile.catchphrases and len(profile.catchphrases) > 0:
            import random
            if random.random() < profile.humor_level:
                catchphrase = random.choice(profile.catchphrases)
                formatted_response += f"\n\n{catchphrase}"
        
        # Add emojis based on personality
        if profile.preferred_emojis and len(profile.preferred_emojis) > 0:
            import random
            if random.random() < 0.7:  # 70% chance to add emoji
                emoji = random.choice(profile.preferred_emojis)
                formatted_response = f"{emoji} {formatted_response}"
        
        return formatted_response
    
    def _apply_basic_personality_formatting(
        self, 
        profile: PersonalityProfile, 
        content: str, 
        context: Optional[ConversationContext]
    ) -> str:
        """Apply basic personality formatting when no template is available."""
        
        # Add personality intro for first interaction
        if context and context.interaction_count <= 1:
            intro = f"Hi! I'm {profile.name}. "
            content = intro + content
        
        # Adjust formality based on profile
        if profile.formality_level < 0.3:
            content = content.replace("Hello", "Hey").replace("Thank you", "Thanks")
        elif profile.formality_level > 0.7:
            content = content.replace("Hey", "Hello").replace("Thanks", "Thank you")
        
        # Add enthusiasm based on emotional baseline
        if profile.emotional_baseline == EmotionalState.EXCITED:
            content += " I'm excited to help with this!"
        elif profile.emotional_baseline == EmotionalState.CONFIDENT:
            content += " I'm confident we can solve this together."
        
        return content
    
    def _get_agent_specialty(self, agent_id: str) -> str:
        """Get agent specialty description."""
        specialties = {
            "scroll-cto-agent": "system architecture and technical strategy",
            "scroll-data-scientist": "data analysis and statistical modeling",
            "scroll-ml-engineer": "machine learning and AI systems",
            "scroll-bi-agent": "business intelligence and data visualization"
        }
        return specialties.get(agent_id, "technical problem solving")
    
    def _get_personality_intro(self, profile: PersonalityProfile) -> str:
        """Generate personality-based introduction."""
        if PersonalityTrait.ENTHUSIASTIC in profile.primary_traits:
            return "I'm excited to work with you!"
        elif PersonalityTrait.SUPPORTIVE in profile.primary_traits:
            return "I'm here to support you every step of the way."
        elif PersonalityTrait.ANALYTICAL in profile.primary_traits:
            return "Let's dive into the details together."
        else:
            return "Ready to help you succeed!"
    
    def _get_personality_conclusion(self, profile: PersonalityProfile, response_type: str) -> str:
        """Generate personality-based conclusion."""
        if response_type == "explanation":
            if PersonalityTrait.SUPPORTIVE in profile.primary_traits:
                return "I hope this explanation helps! Feel free to ask if you need clarification."
            elif PersonalityTrait.DIRECT in profile.primary_traits:
                return "That's the key information you need to know."
            else:
                return "Let me know if you'd like me to elaborate on any part."
        elif response_type == "success":
            if PersonalityTrait.ENTHUSIASTIC in profile.primary_traits:
                return "This is fantastic progress!"
            else:
                return "Great work on this!"
        else:
            return ""
    
    def _extract_key_takeaway(self, content: str) -> str:
        """Extract key takeaway from content (simplified)."""
        # This could be enhanced with NLP
        sentences = content.split('.')
        if sentences:
            return sentences[0].strip()
        return "Key insight identified"
    
    def _extract_recommendation(self, content: str) -> str:
        """Extract recommendation from content (simplified)."""
        # Look for recommendation keywords
        if "recommend" in content.lower():
            sentences = content.split('.')
            for sentence in sentences:
                if "recommend" in sentence.lower():
                    return sentence.strip()
        return "Proceed with the suggested approach"
    
    def get_typing_indicator_message(self, agent_id: str) -> str:
        """Get personality-appropriate typing indicator message."""
        profile = self.get_personality_profile(agent_id)
        if not profile:
            return "Agent is thinking..."
        
        thinking_messages = {
            "scroll-cto-agent": [
                "Analyzing the technical architecture...",
                "Evaluating scalability options...",
                "Reviewing best practices..."
            ],
            "scroll-data-scientist": [
                "Crunching the numbers...",
                "Analyzing data patterns...",
                "Running statistical tests..."
            ],
            "scroll-ml-engineer": [
                "Training the model...",
                "Optimizing hyperparameters...",
                "Evaluating performance metrics..."
            ],
            "scroll-bi-agent": [
                "Building visualizations...",
                "Calculating business metrics...",
                "Generating insights..."
            ]
        }
        
        messages = thinking_messages.get(agent_id, ["Processing your request..."])
        import random
        return random.choice(messages)
    
    def should_show_enthusiasm(self, agent_id: str, context: Optional[ConversationContext] = None) -> bool:
        """Determine if agent should show enthusiasm based on personality and context."""
        profile = self.get_personality_profile(agent_id)
        if not profile:
            return False
        
        base_enthusiasm = PersonalityTrait.ENTHUSIASTIC in profile.primary_traits
        emotional_enthusiasm = profile.emotional_baseline in [EmotionalState.EXCITED, EmotionalState.CONFIDENT]
        
        # Reduce enthusiasm over time in long conversations
        if context and context.interaction_count > 10:
            return base_enthusiasm and emotional_enthusiasm and (context.interaction_count % 5 == 0)
        
        return base_enthusiasm or emotional_enthusiasm
    
    def get_avatar_config(self, agent_id: str) -> Dict[str, Any]:
        """Get avatar configuration for agent."""
        profile = self.get_personality_profile(agent_id)
        if profile and profile.avatar_config:
            return profile.avatar_config
        
        # Default avatar configs
        default_configs = {
            "scroll-cto-agent": {
                "style": "professional",
                "color_scheme": "blue",
                "accessories": ["glasses"],
                "background": "tech_office"
            },
            "scroll-data-scientist": {
                "style": "academic",
                "color_scheme": "purple",
                "accessories": ["lab_coat"],
                "background": "data_lab"
            },
            "scroll-ml-engineer": {
                "style": "casual_tech",
                "color_scheme": "green",
                "accessories": ["headphones"],
                "background": "ml_workspace"
            },
            "scroll-bi-agent": {
                "style": "business_casual",
                "color_scheme": "orange",
                "accessories": ["presentation_screen"],
                "background": "modern_office"
            }
        }
        
        return default_configs.get(agent_id, {
            "style": "professional",
            "color_scheme": "blue",
            "accessories": [],
            "background": "office"
        })