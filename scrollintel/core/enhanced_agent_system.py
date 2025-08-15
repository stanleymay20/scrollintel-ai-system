"""
Enhanced Agent System - Integrates personality, memory, streaming, and templates
Implements requirements 2.3, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6 for enhanced agent capabilities.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from uuid import uuid4

from .agent_personality import AgentPersonalityEngine, PersonalityProfile
from .conversational_memory import ConversationalMemoryEngine, ConversationContext, ResponseContext
from .response_streaming import ResponseStreamingEngine, StreamingEventType
from .agent_response_templates import AgentResponseTemplateEngine, ResponseType, ResponseTone
from ..agents.scroll_cto_agent import ScrollCTOAgent
from ..core.interfaces import BaseAgent, AgentRequest, AgentResponse

logger = logging.getLogger(__name__)


@dataclass
class EnhancedAgentRequest:
    """Enhanced agent request with personality and context."""
    base_request: AgentRequest
    conversation_id: str
    user_id: str
    personality_context: Optional[Dict[str, Any]] = None
    streaming_enabled: bool = True
    response_tone: Optional[ResponseTone] = None
    custom_template: Optional[str] = None


@dataclass
class EnhancedAgentResponse:
    """Enhanced agent response with personality and streaming info."""
    base_response: AgentResponse
    conversation_id: str
    personality_applied: bool = False
    template_used: Optional[str] = None
    streaming_chunks: int = 0
    user_feedback_score: Optional[float] = None


class EnhancedAgentSystem:
    """Enhanced agent system with personality, memory, and streaming capabilities."""
    
    def __init__(self):
        self.personality_engine = AgentPersonalityEngine()
        self.memory_engine = ConversationalMemoryEngine()
        self.streaming_engine = ResponseStreamingEngine()
        self.template_engine = AgentResponseTemplateEngine()
        
        # Agent instances
        self.agents: Dict[str, BaseAgent] = {}
        self.active_conversations: Dict[str, str] = {}  # conversation_id -> agent_id
        
        # Initialize agents
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize agent instances."""
        self.agents["scroll-cto-agent"] = ScrollCTOAgent()
        # Add other agents as they're implemented
        logger.info(f"Initialized {len(self.agents)} agents")
    
    async def start_conversation(self, user_id: str, agent_id: str) -> str:
        """Start a new conversation with enhanced capabilities."""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Start conversation in memory engine
        conversation_id = await self.memory_engine.start_conversation(user_id, agent_id)
        self.active_conversations[conversation_id] = agent_id
        
        # Create personality context
        personality_profile = self.personality_engine.get_personality_profile(agent_id)
        if personality_profile:
            context = self.personality_engine.create_conversation_context(
                conversation_id, user_id, agent_id
            )
        
        logger.info(f"Started enhanced conversation {conversation_id}")
        return conversation_id
    
    async def process_message(
        self, 
        conversation_id: str,
        user_message: str,
        user_id: str,
        streaming_enabled: bool = True,
        response_tone: Optional[ResponseTone] = None
    ) -> EnhancedAgentResponse:
        """Process a message with full enhanced capabilities."""
        
        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        agent_id = self.active_conversations[conversation_id]
        agent = self.agents[agent_id]
        
        try:
            # Start typing indicator
            typing_indicator_id = None
            if streaming_enabled:
                typing_indicator_id = await self.streaming_engine.start_typing_indicator(
                    conversation_id, agent_id
                )
            
            # Get conversation context
            conversation_context = await self.memory_engine.get_conversation_context(conversation_id)
            
            # Create enhanced request
            base_request = AgentRequest(
                id=f"req_{uuid4()}",
                prompt=user_message,
                context=conversation_context,
                user_id=user_id
            )
            
            enhanced_request = EnhancedAgentRequest(
                base_request=base_request,
                conversation_id=conversation_id,
                user_id=user_id,
                streaming_enabled=streaming_enabled,
                response_tone=response_tone
            )
            
            # Process with agent
            base_response = await agent.process_request(base_request)
            
            # Stop typing indicator
            if typing_indicator_id:
                await self.streaming_engine.stop_typing_indicator(conversation_id, typing_indicator_id)
            
            # Apply personality and templates
            enhanced_response_content = await self._apply_personality_and_templates(
                agent_id, base_response.content, conversation_id, response_tone
            )
            
            # Stream response if enabled
            streaming_chunks = 0
            if streaming_enabled:
                response_generator = self.streaming_engine.create_response_generator(
                    enhanced_response_content
                )
                streamed_content = await self.streaming_engine.stream_response(
                    conversation_id, agent_id, response_generator
                )
                streaming_chunks = len(streamed_content.split())
            
            # Create enhanced response
            enhanced_response = EnhancedAgentResponse(
                base_response=base_response,
                conversation_id=conversation_id,
                personality_applied=True,
                streaming_chunks=streaming_chunks
            )
            
            # Update conversation memory
            await self.memory_engine.add_conversation_turn(
                conversation_id, user_message, enhanced_response_content
            )
            
            # Update personality context
            self.personality_engine.update_conversation_context(
                conversation_id, {
                    "role": "user",
                    "content": user_message,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            self.personality_engine.update_conversation_context(
                conversation_id, {
                    "role": "assistant",
                    "content": enhanced_response_content,
                    "timestamp": datetime.now().isoformat(),
                    "agent_id": agent_id
                }
            )
            
            return enhanced_response
            
        except Exception as e:
            # Handle errors gracefully
            if typing_indicator_id:
                await self.streaming_engine.stop_typing_indicator(conversation_id, typing_indicator_id)
            
            error_response = AgentResponse(
                id=f"resp_{uuid4()}",
                request_id=base_request.id if 'base_request' in locals() else "unknown",
                content=f"I apologize, but I encountered an error: {str(e)}",
                artifacts=[],
                execution_time=0.0,
                status="error",
                error_message=str(e)
            )
            
            return EnhancedAgentResponse(
                base_response=error_response,
                conversation_id=conversation_id,
                personality_applied=False
            )
    
    async def _apply_personality_and_templates(
        self,
        agent_id: str,
        base_content: str,
        conversation_id: str,
        response_tone: Optional[ResponseTone] = None
    ) -> str:
        """Apply personality and templates to response content."""
        
        # Get conversation context for template selection
        conversation_context = await self.memory_engine.get_conversation_context(conversation_id)
        
        # Create response context
        response_context = ResponseContext(
            user_id=conversation_context.get("user_id", ""),
            conversation_id=conversation_id,
            agent_id=agent_id,
            conversation_length=conversation_context.get("turn_count", 0),
            recent_topics=conversation_context.get("recent_topics", []),
            user_preferences=conversation_context.get("user_preferences", {})
        )
        
        # Determine response type based on content
        response_type = self._determine_response_type(base_content)
        
        # Get appropriate template
        template = self.template_engine.get_template(
            agent_id, response_type, response_tone, response_context
        )
        
        if template:
            # Extract variables from base content
            variables = self._extract_template_variables(base_content, template.variables)
            
            # Format with template
            formatted_content = self.template_engine.format_response(
                agent_id, template, variables, response_context
            )
            
            return formatted_content
        else:
            # Apply basic personality formatting
            return self.personality_engine.format_response_with_personality(
                agent_id, base_content, "explanation", conversation_id
            )
    
    def _determine_response_type(self, content: str) -> ResponseType:
        """Determine response type from content."""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["hello", "hi", "greetings"]):
            return ResponseType.GREETING
        elif any(word in content_lower for word in ["recommend", "suggest", "should"]):
            return ResponseType.RECOMMENDATION
        elif any(word in content_lower for word in ["analysis", "results", "findings"]):
            return ResponseType.ANALYSIS
        elif any(word in content_lower for word in ["error", "failed", "problem"]):
            return ResponseType.ERROR
        elif any(word in content_lower for word in ["complete", "finished", "done"]):
            return ResponseType.COMPLETION
        else:
            return ResponseType.EXPLANATION
    
    def _extract_template_variables(self, content: str, variable_names: List[str]) -> Dict[str, str]:
        """Extract template variables from content (simplified implementation)."""
        variables = {}
        
        # This is a simplified implementation
        # In a real system, you might use NLP to extract structured information
        for var_name in variable_names:
            if var_name == "main_content":
                variables[var_name] = content
            elif var_name == "key_takeaway":
                # Extract first sentence as key takeaway
                sentences = content.split('.')
                variables[var_name] = sentences[0].strip() if sentences else content[:100]
            elif var_name == "recommendation":
                # Look for recommendation in content
                if "recommend" in content.lower():
                    sentences = content.split('.')
                    for sentence in sentences:
                        if "recommend" in sentence.lower():
                            variables[var_name] = sentence.strip()
                            break
                else:
                    variables[var_name] = "Proceed with the suggested approach"
            else:
                # Default to content or placeholder
                variables[var_name] = content[:200] + "..." if len(content) > 200 else content
        
        return variables
    
    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get enhanced agent status."""
        if agent_id not in self.agents:
            return {"status": "not_found"}
        
        agent = self.agents[agent_id]
        personality_profile = self.personality_engine.get_personality_profile(agent_id)
        
        # Count active conversations for this agent
        active_conversations = sum(1 for aid in self.active_conversations.values() if aid == agent_id)
        
        return {
            "agent_id": agent_id,
            "status": "active" if await agent.health_check() else "inactive",
            "personality": {
                "name": personality_profile.name if personality_profile else "Unknown",
                "traits": [trait.value for trait in personality_profile.primary_traits] if personality_profile else [],
                "communication_style": personality_profile.communication_style.value if personality_profile else "professional"
            },
            "active_conversations": active_conversations,
            "capabilities": [cap.name for cap in agent.get_capabilities()],
            "avatar_config": self.personality_engine.get_avatar_config(agent_id)
        }
    
    async def end_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """End conversation with summary."""
        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        # End conversation in memory engine
        summary = await self.memory_engine.end_conversation(conversation_id)
        
        # Clean up active conversation
        del self.active_conversations[conversation_id]
        
        return {
            "conversation_id": conversation_id,
            "summary": {
                "total_turns": summary.total_turns,
                "main_topics": summary.main_topics,
                "user_satisfaction": summary.user_satisfaction_avg,
                "key_outcomes": summary.key_outcomes
            }
        }
    
    async def provide_feedback(
        self, 
        conversation_id: str, 
        message_id: str, 
        feedback_type: str,
        feedback_score: Optional[float] = None
    ):
        """Provide feedback on agent response."""
        if conversation_id not in self.active_conversations:
            return
        
        agent_id = self.active_conversations[conversation_id]
        
        # Update template success rate if applicable
        if feedback_type in ["positive", "negative"]:
            success = feedback_type == "positive"
            # This would need to track which template was used for the message
            # For now, we'll just log the feedback
            logger.info(f"Received {feedback_type} feedback for agent {agent_id}")
    
    async def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's conversation history with enhanced information."""
        summaries = await self.memory_engine.get_user_conversation_history(user_id, limit)
        
        enhanced_history = []
        for summary in summaries:
            agent_status = await self.get_agent_status(summary.agent_id)
            enhanced_history.append({
                "conversation_id": summary.conversation_id,
                "agent": {
                    "id": summary.agent_id,
                    "name": agent_status["personality"]["name"],
                    "avatar_config": agent_status["avatar_config"]
                },
                "start_time": summary.start_time.isoformat(),
                "end_time": summary.end_time.isoformat() if summary.end_time else None,
                "total_turns": summary.total_turns,
                "main_topics": summary.main_topics,
                "user_satisfaction": summary.user_satisfaction_avg,
                "key_outcomes": summary.key_outcomes,
                "follow_up_needed": summary.follow_up_needed
            })
        
        return enhanced_history
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get enhanced system statistics."""
        return {
            "agents": {
                "total": len(self.agents),
                "active": sum(1 for agent in self.agents.values() if await agent.health_check()),
                "personalities_loaded": len(self.personality_engine.personality_profiles)
            },
            "conversations": {
                "active": len(self.active_conversations),
                "streaming_enabled": len(self.streaming_engine.active_streams)
            },
            "templates": {
                "total": sum(len(templates) for templates in self.template_engine.templates.values()),
                "agents_with_templates": len(self.template_engine.templates)
            },
            "memory": {
                "active_contexts": len(self.memory_engine.conversation_contexts),
                "agent_memories": len(self.memory_engine.agent_memories)
            }
        }
    
    def get_available_agents(self) -> List[Dict[str, Any]]:
        """Get list of available agents with personality information."""
        agents_info = []
        
        for agent_id, agent in self.agents.items():
            personality_profile = self.personality_engine.get_personality_profile(agent_id)
            avatar_config = self.personality_engine.get_avatar_config(agent_id)
            
            agents_info.append({
                "id": agent_id,
                "name": personality_profile.name if personality_profile else agent_id,
                "description": avatar_config.get("description", "AI Agent"),
                "personality": {
                    "traits": [trait.value for trait in personality_profile.primary_traits] if personality_profile else [],
                    "communication_style": personality_profile.communication_style.value if personality_profile else "professional",
                    "expertise_confidence": personality_profile.expertise_confidence if personality_profile else 0.8
                },
                "avatar_config": avatar_config,
                "capabilities": [cap.name for cap in agent.get_capabilities()],
                "status": "active"  # This could be dynamic based on health check
            })
        
        return agents_info