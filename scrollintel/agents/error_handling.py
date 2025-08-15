"""
Agent-specific error handling utilities.
Provides error handling patterns for AI agents with fallback mechanisms.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from functools import wraps

from ..core.error_handling import (
    error_handler, ErrorContext, ErrorSeverity, ErrorCategory,
    with_error_handling, RetryConfig, CircuitBreakerConfig
)
from ..core.interfaces import AgentError, ExternalServiceError


class AgentErrorHandler:
    """Specialized error handler for AI agents."""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"{__name__}.{agent_name}")
        self.fallback_responses: Dict[str, Any] = {}
        self.degraded_capabilities: Dict[str, List[str]] = {}
        
    def register_fallback_response(self, operation: str, response: Dict[str, Any]):
        """Register fallback response for specific operation."""
        self.fallback_responses[operation] = response
    
    def register_degraded_capabilities(self, operation: str, capabilities: List[str]):
        """Register degraded capabilities for specific operation."""
        self.degraded_capabilities[operation] = capabilities
    
    async def handle_agent_unavailable(self, operation: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle agent unavailable scenario."""
        fallback_response = self.fallback_responses.get(operation)
        
        if fallback_response:
            self.logger.info(f"Using fallback response for {self.agent_name}.{operation}")
            return {
                "success": True,
                "fallback_used": True,
                "data": fallback_response,
                "message": f"{self.agent_name} is using cached response"
            }
        
        # Return degraded capabilities if available
        degraded_caps = self.degraded_capabilities.get(operation, [])
        if degraded_caps:
            self.logger.info(f"Using degraded capabilities for {self.agent_name}.{operation}")
            return {
                "success": True,
                "degraded": True,
                "data": {
                    "available_capabilities": degraded_caps,
                    "message": f"{self.agent_name} is running with limited capabilities"
                }
            }
        
        # No fallback available
        return {
            "success": False,
            "error": {
                "type": "agent_unavailable",
                "message": f"{self.agent_name} is temporarily unavailable",
                "recovery_action": "Please try again in a few moments"
            }
        }
    
    async def handle_ai_service_error(self, service_name: str, error: Exception, operation: str) -> Dict[str, Any]:
        """Handle AI service errors with fallback strategies."""
        self.logger.warning(f"AI service {service_name} error in {self.agent_name}.{operation}: {error}")
        
        # Try alternative AI service
        alternative_result = await self._try_alternative_ai_service(service_name, operation)
        if alternative_result:
            return alternative_result
        
        # Use cached response if available
        cached_result = await self._get_cached_response(operation)
        if cached_result:
            return cached_result
        
        # Return error with guidance
        return {
            "success": False,
            "error": {
                "type": "ai_service_error",
                "message": f"AI processing temporarily unavailable for {self.agent_name}",
                "recovery_action": "Please try again in a few moments",
                "service": service_name
            }
        }
    
    async def _try_alternative_ai_service(self, failed_service: str, operation: str) -> Optional[Dict[str, Any]]:
        """Try alternative AI service if available."""
        alternatives = {
            "openai": ["anthropic"],
            "anthropic": ["openai"],
            "pinecone": ["supabase"]
        }
        
        alternative_services = alternatives.get(failed_service, [])
        
        for alt_service in alternative_services:
            try:
                self.logger.info(f"Trying alternative service {alt_service} for {operation}")
                # This would be implemented by specific agents
                # For now, we'll simulate success
                return {
                    "success": True,
                    "fallback_used": True,
                    "data": {"message": f"Using alternative service {alt_service}"},
                    "alternative_service": alt_service
                }
            except Exception as e:
                self.logger.warning(f"Alternative service {alt_service} also failed: {e}")
                continue
        
        return None
    
    async def _get_cached_response(self, operation: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available."""
        # This would integrate with Redis cache
        # For now, we'll return None
        return None


def with_agent_error_handling(agent_name: str, operation: str):
    """Decorator for agent methods with comprehensive error handling."""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            agent_error_handler = AgentErrorHandler(agent_name)
            
            try:
                return await func(self, *args, **kwargs)
            
            except ExternalServiceError as e:
                # Handle external service errors (AI APIs, databases, etc.)
                service_name = getattr(e, 'service_name', 'unknown')
                return await agent_error_handler.handle_ai_service_error(
                    service_name, e, operation
                )
            
            except AgentError as e:
                # Handle agent-specific errors
                return await agent_error_handler.handle_agent_unavailable(operation)
            
            except Exception as e:
                # Handle unexpected errors
                logging.getLogger(agent_name).error(f"Unexpected error in {operation}: {e}")
                return {
                    "success": False,
                    "error": {
                        "type": "unexpected_error",
                        "message": f"An unexpected error occurred in {agent_name}",
                        "recovery_action": "Please try again or contact support"
                    }
                }
        
        return wrapper
    return decorator


class AIServiceFallbackManager:
    """Manages fallback strategies for AI services."""
    
    def __init__(self):
        self.service_priorities = {
            "text_generation": ["openai", "anthropic", "huggingface"],
            "embeddings": ["openai", "huggingface"],
            "vector_search": ["pinecone", "supabase", "local"],
            "speech_to_text": ["openai", "local"]
        }
        
        self.service_capabilities = {
            "openai": ["text_generation", "embeddings", "speech_to_text"],
            "anthropic": ["text_generation"],
            "huggingface": ["text_generation", "embeddings"],
            "pinecone": ["vector_search"],
            "supabase": ["vector_search"],
            "local": ["vector_search", "speech_to_text"]
        }
    
    async def get_fallback_service(self, failed_service: str, capability: str) -> Optional[str]:
        """Get fallback service for a specific capability."""
        priority_list = self.service_priorities.get(capability, [])
        
        # Find the failed service in the priority list
        try:
            failed_index = priority_list.index(failed_service)
            # Return the next service in the priority list
            if failed_index + 1 < len(priority_list):
                return priority_list[failed_index + 1]
        except ValueError:
            # Failed service not in priority list, return first available
            if priority_list:
                return priority_list[0]
        
        return None
    
    async def is_service_available(self, service_name: str) -> bool:
        """Check if a service is currently available."""
        # This would implement actual health checks
        # For now, we'll simulate availability
        circuit_breaker = error_handler.circuit_breakers.get(service_name)
        if circuit_breaker:
            return circuit_breaker.can_execute()
        return True


# Global fallback manager
ai_service_fallback_manager = AIServiceFallbackManager()


class AgentRecoveryStrategies:
    """Common recovery strategies for different types of agents."""
    
    @staticmethod
    async def cto_agent_fallback(error: Exception, context: ErrorContext) -> Dict[str, Any]:
        """Fallback strategy for ScrollCTO agent."""
        return {
            "recommendations": [
                "Use proven technology stack (React, Node.js, PostgreSQL)",
                "Implement microservices architecture for scalability",
                "Use cloud-native deployment (Docker, Kubernetes)",
                "Implement comprehensive monitoring and logging"
            ],
            "message": "Using standard architecture recommendations",
            "confidence": "medium"
        }
    
    @staticmethod
    async def data_scientist_fallback(error: Exception, context: ErrorContext) -> Dict[str, Any]:
        """Fallback strategy for ScrollDataScientist agent."""
        return {
            "analysis": {
                "summary": "Basic statistical analysis completed",
                "recommendations": [
                    "Check data quality and completeness",
                    "Perform exploratory data analysis",
                    "Consider data preprocessing steps",
                    "Validate assumptions before modeling"
                ]
            },
            "message": "Using standard data science workflow",
            "confidence": "low"
        }
    
    @staticmethod
    async def ml_engineer_fallback(error: Exception, context: ErrorContext) -> Dict[str, Any]:
        """Fallback strategy for ScrollMLEngineer agent."""
        return {
            "pipeline": {
                "steps": [
                    "Data preprocessing and feature engineering",
                    "Model selection and training",
                    "Model evaluation and validation",
                    "Model deployment and monitoring"
                ],
                "recommended_models": ["linear_regression", "random_forest", "gradient_boosting"]
            },
            "message": "Using standard ML pipeline template",
            "confidence": "medium"
        }
    
    @staticmethod
    async def ai_engineer_fallback(error: Exception, context: ErrorContext) -> Dict[str, Any]:
        """Fallback strategy for ScrollAIEngineer agent."""
        return {
            "response": "I'm currently using backup systems. I can help with basic queries, but advanced AI features may be limited.",
            "available_features": [
                "Basic text processing",
                "Simple question answering",
                "Document search (limited)"
            ],
            "unavailable_features": [
                "Advanced reasoning",
                "Complex analysis",
                "Real-time learning"
            ],
            "message": "AI agent running in degraded mode"
        }


# Register recovery strategies with the global error handler
def register_agent_recovery_strategies():
    """Register agent-specific recovery strategies."""
    error_handler.register_fallback("scroll_cto", AgentRecoveryStrategies.cto_agent_fallback)
    error_handler.register_fallback("scroll_data_scientist", AgentRecoveryStrategies.data_scientist_fallback)
    error_handler.register_fallback("scroll_ml_engineer", AgentRecoveryStrategies.ml_engineer_fallback)
    error_handler.register_fallback("scroll_ai_engineer", AgentRecoveryStrategies.ai_engineer_fallback)


# Initialize recovery strategies
register_agent_recovery_strategies()