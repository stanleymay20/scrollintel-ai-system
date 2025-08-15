"""
Agent Orchestrator - Routes requests to appropriate agents
Enhanced with Natural Language Interface capabilities
"""
import asyncio
import time
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import uuid

from .base import Agent, AgentRequest, AgentResponse
from .cto_agent import CTOAgent
from .data_scientist_agent import DataScientistAgent
from .ml_engineer_agent import MLEngineerAgent
from .bi_agent import BIAgent
from .ai_engineer_agent import AIEngineerAgent
from .qa_agent import QAAgent
from .forecast_agent import ForecastAgent
from ..nl_interface import NLProcessor

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Central orchestrator for routing requests to appropriate agents"""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.request_history: List[Dict[str, Any]] = []
        self.is_initialized = False
        self.agent_registry: Dict[str, Dict[str, Any]] = {}
        self.health_check_interval = 300  # 5 minutes
        self.last_health_check = None
        self.nl_processor = NLProcessor()  # Natural Language Interface
        
    async def initialize(self):
        """Initialize all agents"""
        try:
            logger.info("Initializing Agent Orchestrator")
            
            # Initialize all 7 core agents
            self.agents = {
                "cto": CTOAgent(),
                "data_scientist": DataScientistAgent(),
                "ml_engineer": MLEngineerAgent(),
                "bi": BIAgent(),
                "ai_engineer": AIEngineerAgent(),
                "qa": QAAgent(),
                "forecast": ForecastAgent()
            }
            
            # Build agent registry
            self._build_agent_registry()
            
            # Perform initial health checks
            health_results = await self.health_check()
            healthy_agents = sum(1 for result in health_results["agents"].values() if result.get("healthy", False))
            
            logger.info(f"Initialized {healthy_agents}/{len(self.agents)} agents successfully")
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown orchestrator and cleanup resources"""
        logger.info("Shutting down Agent Orchestrator")
        self.agents.clear()
        self.is_initialized = False
    
    async def route_request(self, user_input: str, context: Dict[str, Any] = None, 
                          session_id: str = None) -> AgentResponse:
        """Route user request to appropriate agent using NL processing"""
        if not self.is_initialized:
            raise RuntimeError("Orchestrator not initialized")
        
        context = context or {}
        session_id = session_id or str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Parse query using NL processor
            parsed_query = self.nl_processor.parse_query(user_input, session_id, context)
            agent_name = parsed_query.suggested_agent
            
            if agent_name not in self.agents:
                return AgentResponse(
                    agent_name="orchestrator",
                    success=False,
                    error=f"Agent '{agent_name}' not found",
                    error_code="AGENT_NOT_FOUND",
                    processing_time=time.time() - start_time,
                    suggestions=[
                        f"Available agents: {', '.join(self.agents.keys())}",
                        "Try rephrasing your query to match an agent's capabilities"
                    ]
                )
            
            # Merge extracted parameters with context
            merged_context = {**context}
            merged_context.update(parsed_query.parameters)
            merged_context["nl_entities"] = [
                {"type": e.type, "value": e.value, "confidence": e.confidence} 
                for e in parsed_query.entities
            ]
            merged_context["intent_confidence"] = parsed_query.confidence
            merged_context["context_needed"] = parsed_query.context_needed
            
            # Create request
            request = AgentRequest(
                query=user_input,
                context=merged_context,
                parameters=parsed_query.parameters,
                session_id=session_id,
                request_id=str(uuid.uuid4())
            )
            
            # Process request
            agent = self.agents[agent_name]
            response = await agent.process(request)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_agent_stats(agent_name, response.success, processing_time)
            
            # Log request with NL metadata
            self._log_request(user_input, agent_name, response.success, {
                "intent": parsed_query.intent.value,
                "confidence": parsed_query.confidence,
                "entities_count": len(parsed_query.entities),
                "session_id": session_id
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error routing request: {e}")
            return AgentResponse(
                agent_name="orchestrator",
                success=False,
                error=str(e),
                error_code="ORCHESTRATOR_ERROR",
                processing_time=time.time() - start_time,
                suggestions=[
                    "Please try again with a simpler query",
                    "Check if all required parameters are provided"
                ]
            )
    
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process request from API with NL processing"""
        user_input = request_data.get("query", "")
        context = request_data.get("context", {})
        session_id = request_data.get("session_id")
        
        response = await self.route_request(user_input, context, session_id)
        
        # Convert response to dict format
        response_dict = {
            "agent": response.agent_name,
            "success": response.success,
            "result": response.result,
            "error": response.error,
            "metadata": response.metadata,
            "processing_time": response.processing_time,
            "timestamp": response.timestamp.isoformat()
        }
        
        # Generate natural language response if successful
        if response.success and session_id:
            try:
                nl_response = self.nl_processor.process_conversation_turn(
                    user_input, response_dict, session_id, context
                )
                response_dict["nl_response"] = nl_response
            except Exception as e:
                logger.warning(f"Failed to generate NL response: {e}")
                response_dict["nl_response"] = str(response.result) if response.result else "Task completed successfully."
        
        return response_dict
    
    def _classify_intent(self, user_input: str) -> str:
        """Simple intent classification based on keywords"""
        user_input_lower = user_input.lower()
        
        # CTO Agent keywords
        if any(keyword in user_input_lower for keyword in [
            "architecture", "technology stack", "scaling", "infrastructure",
            "tech recommendation", "system design", "cto"
        ]):
            return "cto"
        
        # Data Scientist Agent keywords
        if any(keyword in user_input_lower for keyword in [
            "analyze data", "statistics", "correlation", "insights",
            "data analysis", "explore data", "data scientist"
        ]):
            return "data_scientist"
        
        # ML Engineer Agent keywords
        if any(keyword in user_input_lower for keyword in [
            "machine learning", "model", "predict", "train",
            "ml", "algorithm", "ml engineer"
        ]):
            return "ml_engineer"
        
        # BI Agent keywords
        if any(keyword in user_input_lower for keyword in [
            "dashboard", "report", "kpi", "business intelligence",
            "bi", "visualization", "metrics"
        ]):
            return "bi"
        
        # AI Engineer Agent keywords
        if any(keyword in user_input_lower for keyword in [
            "ai strategy", "artificial intelligence", "ai implementation",
            "ai engineer", "ai roadmap"
        ]):
            return "ai_engineer"
        
        # QA Agent keywords
        if any(keyword in user_input_lower for keyword in [
            "question", "query data", "what is", "how many",
            "show me", "find", "search"
        ]):
            return "qa"
        
        # Forecast Agent keywords
        if any(keyword in user_input_lower for keyword in [
            "forecast", "predict future", "trend", "time series",
            "projection", "future", "forecast agent"
        ]):
            return "forecast"
        
        # Default to QA agent for general questions
        return "qa"
    
    def _log_request(self, query: str, agent: str, success: bool, nl_metadata: Dict[str, Any] = None):
        """Log request for analytics with NL metadata"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query[:100],  # Truncate for privacy
            "agent": agent,
            "success": success
        }
        
        # Add NL processing metadata if available
        if nl_metadata:
            log_entry.update(nl_metadata)
        
        self.request_history.append(log_entry)
        
        # Keep only last 1000 requests
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]
    
    async def get_available_agents(self) -> List[Dict[str, Any]]:
        """Get list of available agents"""
        return [agent.get_info() for agent in self.agents.values()]
    
    def get_routing_info(self) -> Dict[str, Any]:
        """Get information about how requests are routed to agents"""
        return {
            "routing_strategy": "keyword_based",
            "agent_keywords": {
                "cto": ["architecture", "technology stack", "scaling", "infrastructure", "tech recommendation", "system design", "cto"],
                "data_scientist": ["analyze data", "statistics", "correlation", "insights", "data analysis", "explore data", "data scientist"],
                "ml_engineer": ["machine learning", "model", "predict", "train", "ml", "algorithm", "ml engineer"],
                "bi": ["dashboard", "report", "kpi", "business intelligence", "bi", "visualization", "metrics"],
                "ai_engineer": ["ai strategy", "artificial intelligence", "ai implementation", "ai engineer", "ai roadmap"],
                "qa": ["question", "query data", "what is", "how many", "show me", "find", "search"],
                "forecast": ["forecast", "predict future", "trend", "time series", "projection", "future", "forecast agent"]
            },
            "default_agent": "qa",
            "fallback_strategy": "route_to_qa_agent"
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all agents"""
        health_results = {}
        
        for name, agent in self.agents.items():
            health_results[name] = await agent.health_check()
        
        return {
            "orchestrator_healthy": self.is_initialized,
            "agents": health_results,
            "total_agents": len(self.agents),
            "healthy_agents": sum(1 for result in health_results.values() if result.get("healthy", False))
        }
    
    def _build_agent_registry(self):
        """Build agent registry with metadata"""
        for name, agent in self.agents.items():
            self.agent_registry[name] = {
                "name": agent.name,
                "description": agent.description,
                "capabilities": agent.capabilities,
                "status": "active",
                "last_used": None,
                "total_requests": 0,
                "successful_requests": 0,
                "average_response_time": 0.0,
                "error_count": 0
            }
    
    def _update_agent_stats(self, agent_name: str, success: bool, response_time: float):
        """Update agent statistics"""
        if agent_name in self.agent_registry:
            stats = self.agent_registry[agent_name]
            stats["total_requests"] += 1
            stats["last_used"] = datetime.utcnow().isoformat()
            
            if success:
                stats["successful_requests"] += 1
            else:
                stats["error_count"] += 1
            
            # Update average response time
            current_avg = stats["average_response_time"]
            total_requests = stats["total_requests"]
            stats["average_response_time"] = (
                (current_avg * (total_requests - 1) + response_time) / total_requests
            )
    
    async def get_agent_registry(self) -> Dict[str, Dict[str, Any]]:
        """Get complete agent registry with statistics"""
        return self.agent_registry.copy()
    
    async def get_agent_by_name(self, agent_name: str) -> Optional[Agent]:
        """Get specific agent by name"""
        return self.agents.get(agent_name)
    
    async def is_agent_healthy(self, agent_name: str) -> bool:
        """Check if specific agent is healthy"""
        if agent_name not in self.agents:
            return False
        
        agent = self.agents[agent_name]
        health_result = await agent.health_check()
        return health_result.get("healthy", False)
    
    async def periodic_health_check(self) -> Dict[str, Any]:
        """Perform periodic health check and update agent status"""
        current_time = datetime.utcnow()
        
        # Only perform if enough time has passed
        if (self.last_health_check and 
            (current_time - self.last_health_check).total_seconds() < self.health_check_interval):
            return await self.health_check()
        
        self.last_health_check = current_time
        health_results = await self.health_check()
        
        # Update agent registry status
        for agent_name, health_info in health_results["agents"].items():
            if agent_name in self.agent_registry:
                self.agent_registry[agent_name]["status"] = (
                    "active" if health_info.get("healthy", False) else "unhealthy"
                )
        
        logger.info(f"Periodic health check completed: {health_results['healthy_agents']}/{health_results['total_agents']} agents healthy")
        return health_results
    
    def get_request_stats(self) -> Dict[str, Any]:
        """Get request statistics"""
        if not self.request_history:
            return {"total_requests": 0}
        
        total = len(self.request_history)
        successful = sum(1 for req in self.request_history if req["success"])
        
        agent_counts = {}
        for req in self.request_history:
            agent = req["agent"]
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        return {
            "total_requests": total,
            "successful_requests": successful,
            "success_rate": successful / total if total > 0 else 0,
            "agent_usage": agent_counts,
            "registry_stats": {
                name: {
                    "total_requests": stats["total_requests"],
                    "success_rate": (
                        stats["successful_requests"] / stats["total_requests"] 
                        if stats["total_requests"] > 0 else 0
                    ),
                    "average_response_time": stats["average_response_time"],
                    "status": stats["status"]
                }
                for name, stats in self.agent_registry.items()
            }
        }
    
    def suggest_agent(self, user_input: str, session_id: str = None) -> Dict[str, Any]:
        """Suggest the best agent for a query using NL processing"""
        try:
            # Use NL processor for better suggestions
            suggestions = self.nl_processor.get_context_suggestions(user_input, session_id)
            return suggestions
        except Exception as e:
            logger.error(f"Error getting NL suggestions: {e}")
            # Fallback to simple keyword-based suggestion
            return self._simple_agent_suggestion(user_input)
    
    def _simple_agent_suggestion(self, user_input: str) -> Dict[str, Any]:
        """Fallback simple agent suggestion"""
        user_input_lower = user_input.lower()
        routing_info = self.get_routing_info()
        
        agent_scores = {}
        for agent_name, keywords in routing_info["agent_keywords"].items():
            score = sum(1 for keyword in keywords if keyword in user_input_lower)
            if score > 0:
                agent_scores[agent_name] = score / len(keywords)  # Normalize by keyword count
        
        if not agent_scores:
            return {
                "suggested_agent": routing_info["default_agent"],
                "confidence": 0.1,
                "reason": "No specific keywords found, using default agent",
                "alternatives": list(self.agents.keys())
            }
        
        # Get the agent with highest score
        best_agent = max(agent_scores.items(), key=lambda x: x[1])
        
        return {
            "suggested_agent": best_agent[0],
            "confidence": min(best_agent[1] * 2, 1.0),  # Scale confidence
            "reason": f"Query matches {best_agent[0]} keywords",
            "alternatives": [agent for agent, score in sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)[1:3]],
            "all_scores": agent_scores
        }
    
    # Natural Language Interface Methods
    
    async def parse_query(self, query: str, session_id: str = None, 
                         context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Parse user query and return structured information"""
        try:
            parsed = self.nl_processor.parse_query(query, session_id, context)
            return {
                "original_query": parsed.original_query,
                "intent": parsed.intent.value,
                "confidence": parsed.confidence,
                "suggested_agent": parsed.suggested_agent,
                "entities": [
                    {
                        "type": e.type,
                        "value": e.value,
                        "confidence": e.confidence,
                        "position": {"start": e.start_pos, "end": e.end_pos}
                    }
                    for e in parsed.entities
                ],
                "parameters": parsed.parameters,
                "context_needed": parsed.context_needed
            }
        except Exception as e:
            logger.error(f"Error parsing query: {e}")
            return {
                "original_query": query,
                "intent": "general",
                "confidence": 0.1,
                "suggested_agent": "qa",
                "entities": [],
                "parameters": {},
                "context_needed": [],
                "error": str(e)
            }
    
    async def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        try:
            return self.nl_processor.get_conversation_history(session_id)
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
    
    async def clear_conversation(self, session_id: str) -> bool:
        """Clear conversation history for a session"""
        try:
            self.nl_processor.clear_conversation(session_id)
            return True
        except Exception as e:
            logger.error(f"Error clearing conversation: {e}")
            return False
    
    async def process_conversational_request(self, query: str, session_id: str,
                                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a conversational request with full NL capabilities"""
        try:
            # Parse the query first
            parsed_info = await self.parse_query(query, session_id, context)
            
            # Process the request
            request_data = {
                "query": query,
                "context": context or {},
                "session_id": session_id
            }
            
            response = await self.process_request(request_data)
            
            # Add parsing information to response
            response["parsing"] = parsed_info
            response["session_id"] = session_id
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing conversational request: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent": "orchestrator",
                "session_id": session_id,
                "nl_response": f"I apologize, but I encountered an issue processing your request: {query}"
            }
    
    async def get_nl_suggestions(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """Get NL processing suggestions for improving a query"""
        try:
            return self.nl_processor.get_context_suggestions(query, session_id)
        except Exception as e:
            logger.error(f"Error getting NL suggestions: {e}")
            return {
                "confidence": 0.1,
                "suggested_agent": "qa",
                "missing_context": [],
                "extracted_entities": [],
                "suggestions": ["Please try rephrasing your query"],
                "error": str(e)
            }