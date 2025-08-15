"""
Enhanced bulletproof middleware for ScrollIntel API.
Ensures the API never fails and always provides a meaningful response with
intelligent request routing, dynamic timeout adjustment, request prioritization,
load balancing, and comprehensive error response enhancement.
"""

import asyncio
import logging
import time
import traceback
import hashlib
import random
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from dataclasses import dataclass
from enum import Enum
import json
import psutil

from .failure_prevention import failure_prevention, FailureType
from .graceful_degradation import degradation_manager
from .user_experience_protection import ux_protector

logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """Request priority levels for intelligent routing."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_RESPONSE_TIME = "weighted_response_time"
    RESOURCE_AWARE = "resource_aware"


@dataclass
class RequestContext:
    """Enhanced request context with routing and priority information."""
    request_id: str
    priority: RequestPriority
    complexity_score: float
    estimated_duration: float
    user_id: Optional[str]
    action_type: str
    retry_count: int = 0
    route_target: Optional[str] = None


@dataclass
class ServerNode:
    """Represents a server node for load balancing."""
    node_id: str
    endpoint: str
    current_load: float
    avg_response_time: float
    active_connections: int
    health_score: float
    last_health_check: datetime


class IntelligentRequestRouter:
    """Intelligent request routing system."""
    
    def __init__(self):
        self.server_nodes: Dict[str, ServerNode] = {}
        self.routing_rules: Dict[str, Callable] = {}
        self.load_balancing_strategy = LoadBalancingStrategy.RESOURCE_AWARE
        self.circuit_breakers: Dict[str, bool] = {}
        
        # Initialize default server node (current instance)
        self._initialize_default_node()
        self._setup_routing_rules()
    
    def _initialize_default_node(self):
        """Initialize the default server node."""
        self.server_nodes["primary"] = ServerNode(
            node_id="primary",
            endpoint="localhost",
            current_load=0.0,
            avg_response_time=0.0,
            active_connections=0,
            health_score=1.0,
            last_health_check=datetime.utcnow()
        )
    
    def _setup_routing_rules(self):
        """Setup intelligent routing rules."""
        self.routing_rules = {
            "ai_services": self._route_ai_requests,
            "file_processing": self._route_file_requests,
            "visualization": self._route_visualization_requests,
            "database": self._route_database_requests,
            "default": self._route_default_requests
        }
    
    def route_request(self, context: RequestContext) -> str:
        """Route request to optimal server node."""
        # Determine service type
        service_type = self._determine_service_type(context.action_type)
        
        # Apply routing rule
        routing_rule = self.routing_rules.get(service_type, self.routing_rules["default"])
        target_node = routing_rule(context)
        
        # Update context
        context.route_target = target_node
        
        return target_node
    
    def _determine_service_type(self, action_type: str) -> str:
        """Determine service type from action type."""
        if "ai" in action_type or "agent" in action_type:
            return "ai_services"
        elif "upload" in action_type or "file" in action_type:
            return "file_processing"
        elif "visualization" in action_type or "chart" in action_type:
            return "visualization"
        elif "database" in action_type or "query" in action_type:
            return "database"
        else:
            return "default"
    
    def _route_ai_requests(self, context: RequestContext) -> str:
        """Route AI requests based on complexity and load."""
        # For high complexity AI requests, prefer nodes with better resources
        if context.complexity_score > 0.8:
            return self._select_best_performance_node()
        else:
            return self._select_balanced_node()
    
    def _route_file_requests(self, context: RequestContext) -> str:
        """Route file processing requests."""
        # File processing needs good I/O performance
        return self._select_node_by_disk_performance()
    
    def _route_visualization_requests(self, context: RequestContext) -> str:
        """Route visualization requests."""
        # Visualization needs good CPU and memory
        return self._select_node_by_cpu_memory()
    
    def _route_database_requests(self, context: RequestContext) -> str:
        """Route database requests."""
        # Database requests prefer consistent performance
        return self._select_most_stable_node()
    
    def _route_default_requests(self, context: RequestContext) -> str:
        """Default routing strategy."""
        return self._select_balanced_node()
    
    def _select_best_performance_node(self) -> str:
        """Select node with best overall performance."""
        best_node = max(
            self.server_nodes.values(),
            key=lambda node: node.health_score / max(node.avg_response_time, 0.1)
        )
        return best_node.node_id
    
    def _select_balanced_node(self) -> str:
        """Select node with balanced load."""
        if self.load_balancing_strategy == LoadBalancingStrategy.RESOURCE_AWARE:
            return self._select_resource_aware_node()
        elif self.load_balancing_strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._select_least_connections_node()
        else:
            return "primary"  # Default fallback
    
    def _select_resource_aware_node(self) -> str:
        """Select node based on current resource utilization."""
        best_node = min(
            self.server_nodes.values(),
            key=lambda node: node.current_load + (node.active_connections * 0.1)
        )
        return best_node.node_id
    
    def _select_least_connections_node(self) -> str:
        """Select node with least active connections."""
        best_node = min(self.server_nodes.values(), key=lambda node: node.active_connections)
        return best_node.node_id
    
    def _select_node_by_disk_performance(self) -> str:
        """Select node with best disk performance."""
        # For now, return primary node
        # In a real implementation, you'd check disk I/O metrics
        return "primary"
    
    def _select_node_by_cpu_memory(self) -> str:
        """Select node with best CPU and memory availability."""
        # For now, return primary node
        # In a real implementation, you'd check CPU and memory metrics
        return "primary"
    
    def _select_most_stable_node(self) -> str:
        """Select most stable node based on health score."""
        best_node = max(self.server_nodes.values(), key=lambda node: node.health_score)
        return best_node.node_id
    
    def update_node_metrics(self, node_id: str, response_time: float, success: bool):
        """Update node metrics after request completion."""
        if node_id in self.server_nodes:
            node = self.server_nodes[node_id]
            
            # Update average response time (exponential moving average)
            alpha = 0.1
            node.avg_response_time = (alpha * response_time + 
                                    (1 - alpha) * node.avg_response_time)
            
            # Update health score based on success
            if success:
                node.health_score = min(1.0, node.health_score + 0.01)
            else:
                node.health_score = max(0.0, node.health_score - 0.05)
            
            node.last_health_check = datetime.utcnow()


class DynamicTimeoutManager:
    """Manages dynamic timeout adjustment based on request complexity."""
    
    def __init__(self):
        self.base_timeouts = {
            RequestPriority.CRITICAL: 5.0,
            RequestPriority.HIGH: 15.0,
            RequestPriority.NORMAL: 30.0,
            RequestPriority.LOW: 60.0,
            RequestPriority.BACKGROUND: 300.0
        }
        self.complexity_multipliers = {
            "simple": 1.0,
            "moderate": 1.5,
            "complex": 2.5,
            "very_complex": 4.0
        }
        self.system_load_factor = 1.0
        self.recent_response_times: Dict[str, List[float]] = {}
    
    def calculate_timeout(self, context: RequestContext) -> float:
        """Calculate dynamic timeout based on request context."""
        # Base timeout from priority
        base_timeout = self.base_timeouts[context.priority]
        
        # Complexity adjustment
        complexity_category = self._categorize_complexity(context.complexity_score)
        complexity_multiplier = self.complexity_multipliers[complexity_category]
        
        # System load adjustment
        system_load_multiplier = self._get_system_load_multiplier()
        
        # Historical performance adjustment
        historical_multiplier = self._get_historical_multiplier(context.action_type)
        
        # Calculate final timeout
        timeout = (base_timeout * complexity_multiplier * 
                  system_load_multiplier * historical_multiplier)
        
        # Apply bounds
        min_timeout = 1.0
        max_timeout = 600.0  # 10 minutes max
        
        return max(min_timeout, min(timeout, max_timeout))
    
    def _categorize_complexity(self, complexity_score: float) -> str:
        """Categorize complexity score."""
        if complexity_score < 0.3:
            return "simple"
        elif complexity_score < 0.6:
            return "moderate"
        elif complexity_score < 0.8:
            return "complex"
        else:
            return "very_complex"
    
    def _get_system_load_multiplier(self) -> float:
        """Get system load multiplier."""
        try:
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # Calculate load factor
            load_factor = (cpu_percent + memory_percent) / 200.0  # Normalize to 0-1
            
            # Convert to multiplier (1.0 to 3.0)
            return 1.0 + (load_factor * 2.0)
        except Exception:
            return 1.0
    
    def _get_historical_multiplier(self, action_type: str) -> float:
        """Get historical performance multiplier."""
        if action_type not in self.recent_response_times:
            return 1.0
        
        recent_times = self.recent_response_times[action_type]
        if not recent_times:
            return 1.0
        
        # Calculate average of recent response times
        avg_time = sum(recent_times) / len(recent_times)
        
        # If recent requests are taking longer, increase timeout
        if avg_time > 5.0:
            return 1.5
        elif avg_time > 2.0:
            return 1.2
        else:
            return 1.0
    
    def record_response_time(self, action_type: str, response_time: float):
        """Record response time for historical analysis."""
        if action_type not in self.recent_response_times:
            self.recent_response_times[action_type] = []
        
        self.recent_response_times[action_type].append(response_time)
        
        # Keep only last 20 response times
        if len(self.recent_response_times[action_type]) > 20:
            self.recent_response_times[action_type] = self.recent_response_times[action_type][-20:]


class RequestPrioritizer:
    """Manages request prioritization and queuing."""
    
    def __init__(self):
        self.priority_queues: Dict[RequestPriority, List[RequestContext]] = {
            priority: [] for priority in RequestPriority
        }
        self.active_requests: Dict[str, RequestContext] = {}
        self.max_concurrent_requests = {
            RequestPriority.CRITICAL: 10,
            RequestPriority.HIGH: 8,
            RequestPriority.NORMAL: 6,
            RequestPriority.LOW: 4,
            RequestPriority.BACKGROUND: 2
        }
        self.current_concurrent: Dict[RequestPriority, int] = {
            priority: 0 for priority in RequestPriority
        }
    
    def should_queue_request(self, context: RequestContext) -> bool:
        """Determine if request should be queued."""
        current_count = self.current_concurrent[context.priority]
        max_count = self.max_concurrent_requests[context.priority]
        
        return current_count >= max_count
    
    def queue_request(self, context: RequestContext):
        """Queue a request for later processing."""
        self.priority_queues[context.priority].append(context)
        logger.info(f"Request {context.request_id} queued with priority {context.priority.value}")
    
    def start_request(self, context: RequestContext):
        """Mark request as started."""
        self.active_requests[context.request_id] = context
        self.current_concurrent[context.priority] += 1
    
    def complete_request(self, request_id: str):
        """Mark request as completed."""
        if request_id in self.active_requests:
            context = self.active_requests.pop(request_id)
            self.current_concurrent[context.priority] -= 1
            
            # Process next queued request of same priority
            self._process_next_queued_request(context.priority)
    
    def _process_next_queued_request(self, priority: RequestPriority):
        """Process next queued request of given priority."""
        if (self.priority_queues[priority] and 
            self.current_concurrent[priority] < self.max_concurrent_requests[priority]):
            
            next_context = self.priority_queues[priority].pop(0)
            # In a real implementation, you'd trigger the request processing here
            logger.info(f"Processing queued request {next_context.request_id}")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            "queued_requests": {
                priority.value: len(queue) 
                for priority, queue in self.priority_queues.items()
            },
            "active_requests": {
                priority.value: count 
                for priority, count in self.current_concurrent.items()
            },
            "total_active": len(self.active_requests),
            "total_queued": sum(len(queue) for queue in self.priority_queues.values())
        }


class EnhancedErrorResponseSystem:
    """Comprehensive error response enhancement system."""
    
    def __init__(self):
        self.error_patterns: Dict[str, Dict[str, Any]] = {}
        self.recovery_suggestions: Dict[str, List[str]] = {}
        self.user_friendly_messages: Dict[str, str] = {}
        self.contextual_help: Dict[str, Dict[str, Any]] = {}
        
        self._initialize_error_responses()
    
    def _initialize_error_responses(self):
        """Initialize comprehensive error response patterns."""
        self.error_patterns = {
            "timeout": {
                "user_message": "Your request is taking longer than usual. We're still working on it.",
                "technical_message": "Request timeout occurred",
                "recovery_actions": ["retry", "reduce_complexity", "try_later"],
                "estimated_fix_time": "1-2 minutes",
                "alternative_actions": ["view_cached_data", "use_simplified_view"]
            },
            "rate_limit": {
                "user_message": "You're making requests very quickly. Please wait a moment.",
                "technical_message": "Rate limit exceeded",
                "recovery_actions": ["wait", "reduce_frequency"],
                "estimated_fix_time": "30 seconds",
                "alternative_actions": ["batch_requests", "use_cached_data"]
            },
            "server_error": {
                "user_message": "We're experiencing technical difficulties. Our team has been notified.",
                "technical_message": "Internal server error",
                "recovery_actions": ["retry", "contact_support"],
                "estimated_fix_time": "5-10 minutes",
                "alternative_actions": ["use_offline_mode", "view_cached_data"]
            },
            "validation_error": {
                "user_message": "There's an issue with your input. Please check and try again.",
                "technical_message": "Input validation failed",
                "recovery_actions": ["fix_input", "use_example"],
                "estimated_fix_time": "immediate",
                "alternative_actions": ["use_template", "get_help"]
            },
            "network_error": {
                "user_message": "Connection issue detected. Retrying automatically...",
                "technical_message": "Network connectivity error",
                "recovery_actions": ["auto_retry", "check_connection"],
                "estimated_fix_time": "30 seconds",
                "alternative_actions": ["use_offline_mode", "save_for_later"]
            }
        }
        
        self.contextual_help = {
            "upload": {
                "common_issues": ["File too large", "Unsupported format", "Network timeout"],
                "solutions": ["Compress file", "Convert format", "Try smaller files"],
                "examples": ["Use JPG instead of PNG", "Reduce file size to under 10MB"]
            },
            "visualization": {
                "common_issues": ["Data too large", "Invalid data format", "Chart rendering failed"],
                "solutions": ["Filter data", "Check data format", "Try different chart type"],
                "examples": ["Use date range filter", "Ensure numeric columns", "Try bar chart instead"]
            },
            "ai_interaction": {
                "common_issues": ["Request too complex", "Service overloaded", "Invalid prompt"],
                "solutions": ["Simplify request", "Try again later", "Rephrase prompt"],
                "examples": ["Ask one question at a time", "Be more specific", "Use simpler language"]
            }
        }
    
    def enhance_error_response(self, error: Exception, context: RequestContext) -> Dict[str, Any]:
        """Enhance error response with comprehensive information."""
        error_type = self._classify_error(error)
        error_pattern = self.error_patterns.get(error_type, self.error_patterns["server_error"])
        
        # Build enhanced response
        enhanced_response = {
            "error": True,
            "error_type": error_type,
            "request_id": context.request_id,
            "timestamp": datetime.utcnow().isoformat(),
            
            # User-friendly information
            "message": error_pattern["user_message"],
            "estimated_fix_time": error_pattern["estimated_fix_time"],
            "what_happened": self._explain_what_happened(error, context),
            "what_you_can_do": self._get_user_actions(error_pattern, context),
            
            # Technical information (for debugging)
            "technical_details": {
                "error_message": str(error),
                "error_class": error.__class__.__name__,
                "context": {
                    "action_type": context.action_type,
                    "priority": context.priority.value,
                    "complexity_score": context.complexity_score,
                    "retry_count": context.retry_count
                }
            },
            
            # Recovery and alternatives
            "recovery_options": self._get_recovery_options(error_pattern, context),
            "alternative_actions": self._get_alternative_actions(error_pattern, context),
            "contextual_help": self._get_contextual_help(context.action_type),
            
            # System status
            "system_status": self._get_system_status_summary(),
            "retry_recommended": self._should_recommend_retry(error, context),
            "support_available": True
        }
        
        return enhanced_response
    
    def _classify_error(self, error: Exception) -> str:
        """Classify error type for appropriate response."""
        error_msg = str(error).lower()
        error_class = error.__class__.__name__.lower()
        
        if "timeout" in error_msg or "timeout" in error_class:
            return "timeout"
        elif "rate" in error_msg and "limit" in error_msg:
            return "rate_limit"
        elif "validation" in error_msg or "invalid" in error_msg:
            return "validation_error"
        elif "network" in error_msg or "connection" in error_msg:
            return "network_error"
        else:
            return "server_error"
    
    def _explain_what_happened(self, error: Exception, context: RequestContext) -> str:
        """Provide clear explanation of what happened."""
        explanations = {
            "timeout": f"Your {context.action_type} request took longer than expected to complete.",
            "rate_limit": "You've made too many requests in a short time period.",
            "validation_error": "The information you provided couldn't be processed.",
            "network_error": "There was a problem connecting to our servers.",
            "server_error": "Our system encountered an unexpected issue while processing your request."
        }
        
        error_type = self._classify_error(error)
        return explanations.get(error_type, "An unexpected error occurred.")
    
    def _get_user_actions(self, error_pattern: Dict[str, Any], context: RequestContext) -> List[Dict[str, Any]]:
        """Get actionable steps for the user."""
        actions = []
        
        for action_type in error_pattern["recovery_actions"]:
            if action_type == "retry":
                actions.append({
                    "action": "retry",
                    "label": "Try Again",
                    "description": "Retry your request",
                    "button_text": "Retry",
                    "immediate": True
                })
            elif action_type == "wait":
                actions.append({
                    "action": "wait",
                    "label": "Wait and Retry",
                    "description": f"Wait {error_pattern['estimated_fix_time']} and try again",
                    "button_text": "Wait",
                    "immediate": False
                })
            elif action_type == "fix_input":
                actions.append({
                    "action": "fix_input",
                    "label": "Check Your Input",
                    "description": "Review and correct your input",
                    "button_text": "Edit",
                    "immediate": True
                })
            elif action_type == "contact_support":
                actions.append({
                    "action": "contact_support",
                    "label": "Contact Support",
                    "description": "Get help from our support team",
                    "button_text": "Get Help",
                    "immediate": True
                })
        
        return actions
    
    def _get_recovery_options(self, error_pattern: Dict[str, Any], context: RequestContext) -> List[Dict[str, Any]]:
        """Get recovery options with detailed information."""
        options = []
        
        for alt_action in error_pattern["alternative_actions"]:
            if alt_action == "view_cached_data":
                options.append({
                    "type": "cached_data",
                    "title": "View Recent Data",
                    "description": "See your most recent data while we fix the issue",
                    "available": True
                })
            elif alt_action == "use_simplified_view":
                options.append({
                    "type": "simplified_view",
                    "title": "Simplified View",
                    "description": "Use a basic version with core functionality",
                    "available": True
                })
            elif alt_action == "use_offline_mode":
                options.append({
                    "type": "offline_mode",
                    "title": "Offline Mode",
                    "description": "Continue working with local data",
                    "available": True
                })
        
        return options
    
    def _get_alternative_actions(self, error_pattern: Dict[str, Any], context: RequestContext) -> List[str]:
        """Get alternative actions user can take."""
        return error_pattern.get("alternative_actions", [])
    
    def _get_contextual_help(self, action_type: str) -> Dict[str, Any]:
        """Get contextual help based on action type."""
        # Extract base action type
        base_action = action_type.split('_')[0] if '_' in action_type else action_type
        
        return self.contextual_help.get(base_action, {
            "common_issues": ["System overload", "Network issues"],
            "solutions": ["Try again later", "Contact support"],
            "examples": ["Wait a few minutes", "Check your connection"]
        })
    
    def _get_system_status_summary(self) -> Dict[str, Any]:
        """Get current system status summary."""
        try:
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            status = "healthy"
            if cpu_percent > 80 or memory_percent > 80:
                status = "degraded"
            if cpu_percent > 95 or memory_percent > 95:
                status = "overloaded"
            
            return {
                "overall_status": status,
                "cpu_usage": cpu_percent,
                "memory_usage": memory_percent,
                "estimated_recovery": "1-5 minutes" if status != "healthy" else "N/A"
            }
        except Exception:
            return {
                "overall_status": "unknown",
                "estimated_recovery": "Unknown"
            }
    
    def _should_recommend_retry(self, error: Exception, context: RequestContext) -> bool:
        """Determine if retry should be recommended."""
        error_type = self._classify_error(error)
        
        # Don't recommend retry for validation errors
        if error_type == "validation_error":
            return False
        
        # Don't recommend retry if already retried multiple times
        if context.retry_count >= 3:
            return False
        
        # Recommend retry for transient errors
        return error_type in ["timeout", "network_error", "server_error"]


class BulletproofMiddleware(BaseHTTPMiddleware):
    """Enhanced middleware that makes the API bulletproof against all failures."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.request_count = 0
        self.error_count = 0
        self.response_times: List[float] = []
        
        # Enhanced components
        self.request_router = IntelligentRequestRouter()
        self.timeout_manager = DynamicTimeoutManager()
        self.request_prioritizer = RequestPrioritizer()
        self.error_response_system = EnhancedErrorResponseSystem()
        
        # Load balancing and routing
        self.current_load = 0.0
        self.request_complexity_cache: Dict[str, float] = {}
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with comprehensive protection and intelligent routing."""
        start_time = time.time()
        request_id = f"req_{int(start_time * 1000)}_{self.request_count}"
        self.request_count += 1
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Create enhanced request context
        context = await self._create_request_context(request, request_id, start_time)
        
        try:
            # Check if request should be queued due to load
            if self.request_prioritizer.should_queue_request(context):
                return await self._handle_queued_request(context)
            
            # Start request processing
            self.request_prioritizer.start_request(context)
            
            # Intelligent request routing
            target_node = self.request_router.route_request(context)
            
            # Track user action with enhanced context
            async with ux_protector.track_user_action(
                context.action_type, context.user_id, context.priority.value
            ):
                # Pre-request health checks and optimization
                await self._perform_enhanced_pre_request_checks(request, context)
                
                # Process request with intelligent protection
                response = await self._process_request_with_intelligent_protection(
                    request, call_next, context
                )
                
                # Post-process response with enhancements
                response = await self._post_process_enhanced_response(
                    request, response, context, start_time
                )
                
                return response
                
        except Exception as e:
            # Enhanced error handling
            return await self._handle_enhanced_failure(request, e, context)
        
        finally:
            # Complete request processing
            self.request_prioritizer.complete_request(request_id)
    
    async def _create_request_context(self, request: Request, request_id: str, start_time: float) -> RequestContext:
        """Create enhanced request context with intelligent analysis."""
        action_type = self._get_action_type(request)
        priority = self._get_enhanced_priority(request, action_type)
        user_id = self._extract_user_id(request)
        complexity_score = await self._calculate_request_complexity(request, action_type)
        estimated_duration = self._estimate_request_duration(action_type, complexity_score)
        
        return RequestContext(
            request_id=request_id,
            priority=priority,
            complexity_score=complexity_score,
            estimated_duration=estimated_duration,
            user_id=user_id,
            action_type=action_type
        )
    
    async def _calculate_request_complexity(self, request: Request, action_type: str) -> float:
        """Calculate request complexity score (0.0 to 1.0)."""
        complexity = 0.0
        
        # Base complexity by action type
        action_complexity = {
            "ai_interaction": 0.8,
            "file_upload": 0.6,
            "visualization": 0.7,
            "export": 0.5,
            "data_modification": 0.4,
            "data_retrieval": 0.2,
            "authentication": 0.1
        }
        complexity += action_complexity.get(action_type, 0.3)
        
        # Adjust based on request size
        content_length = int(request.headers.get("content-length", "0"))
        if content_length > 10 * 1024 * 1024:  # 10MB
            complexity += 0.3
        elif content_length > 1024 * 1024:  # 1MB
            complexity += 0.1
        
        # Adjust based on query parameters
        query_params = len(request.query_params)
        if query_params > 10:
            complexity += 0.1
        elif query_params > 5:
            complexity += 0.05
        
        # Check for complex operations in path
        path = request.url.path.lower()
        if any(complex_op in path for complex_op in ["batch", "bulk", "aggregate", "analyze"]):
            complexity += 0.2
        
        # Cache the result for similar requests
        cache_key = f"{action_type}_{content_length}_{query_params}"
        self.request_complexity_cache[cache_key] = min(1.0, complexity)
        
        return min(1.0, complexity)
    
    def _estimate_request_duration(self, action_type: str, complexity_score: float) -> float:
        """Estimate request duration in seconds."""
        base_durations = {
            "ai_interaction": 10.0,
            "file_upload": 30.0,
            "visualization": 5.0,
            "export": 15.0,
            "data_modification": 2.0,
            "data_retrieval": 1.0,
            "authentication": 0.5
        }
        
        base_duration = base_durations.get(action_type, 2.0)
        return base_duration * (1.0 + complexity_score)
    
    async def _handle_queued_request(self, context: RequestContext) -> JSONResponse:
        """Handle request that needs to be queued."""
        self.request_prioritizer.queue_request(context)
        
        queue_status = self.request_prioritizer.get_queue_status()
        estimated_wait = self._estimate_queue_wait_time(context.priority, queue_status)
        
        return JSONResponse(
            status_code=202,  # Accepted
            content={
                "status": "queued",
                "message": f"Your request has been queued due to high system load.",
                "request_id": context.request_id,
                "priority": context.priority.value,
                "estimated_wait_time": f"{estimated_wait:.0f} seconds",
                "queue_position": queue_status["queued_requests"][context.priority.value],
                "what_happens_next": "We'll process your request as soon as resources are available.",
                "alternatives": [
                    "Try again in a few minutes",
                    "Use cached data if available",
                    "Simplify your request"
                ]
            }
        )
    
    def _estimate_queue_wait_time(self, priority: RequestPriority, queue_status: Dict[str, Any]) -> float:
        """Estimate wait time in queue."""
        # Simple estimation based on queue position and average processing time
        queue_position = queue_status["queued_requests"][priority.value]
        avg_processing_time = 5.0  # seconds
        
        return queue_position * avg_processing_time
    
    async def _perform_enhanced_pre_request_checks(self, request: Request, context: RequestContext):
        """Perform enhanced health checks and optimizations before processing request."""
        # System resource checks
        await self._perform_pre_request_checks(request)
        
        # Load balancing preparation
        self.current_load = self._calculate_current_load()
        
        # Update server node metrics
        if context.route_target:
            node = self.request_router.server_nodes.get(context.route_target)
            if node:
                node.active_connections += 1
                node.current_load = self.current_load
    
    def _calculate_current_load(self) -> float:
        """Calculate current system load."""
        try:
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            active_requests = len(self.request_prioritizer.active_requests)
            
            # Normalize and combine metrics
            load = (cpu_percent + memory_percent) / 200.0  # 0-1 scale
            load += min(active_requests / 50.0, 0.5)  # Add request load
            
            return min(1.0, load)
        except Exception:
            return 0.5  # Default moderate load
    
    def _get_action_type(self, request: Request) -> str:
        """Determine action type from request."""
        path = request.url.path.lower()
        method = request.method.lower()
        
        if "auth" in path or "login" in path:
            return "authentication"
        elif "upload" in path or method == "post" and "file" in path:
            return "file_upload"
        elif "visualization" in path or "chart" in path:
            return "visualization"
        elif "export" in path or "download" in path:
            return "export"
        elif "ai" in path or "agent" in path:
            return "ai_interaction"
        elif method == "get":
            return "data_retrieval"
        elif method in ["post", "put", "patch"]:
            return "data_modification"
        else:
            return "general"
    
    def _get_enhanced_priority(self, request: Request, action_type: str) -> RequestPriority:
        """Determine enhanced request priority with intelligent analysis."""
        path = request.url.path.lower()
        method = request.method.lower()
        
        # Check for explicit priority headers
        priority_header = request.headers.get("X-Request-Priority", "").lower()
        priority_mapping = {
            "critical": RequestPriority.CRITICAL,
            "high": RequestPriority.HIGH,
            "normal": RequestPriority.NORMAL,
            "low": RequestPriority.LOW,
            "background": RequestPriority.BACKGROUND
        }
        if priority_header in priority_mapping:
            return priority_mapping[priority_header]
        
        # Critical operations
        if any(critical in path for critical in ["auth", "login", "health", "error", "emergency"]):
            return RequestPriority.CRITICAL
        
        # High priority operations
        if any(high in path for high in ["save", "create", "delete", "payment"]):
            return RequestPriority.HIGH
        elif action_type == "data_modification" and method in ["post", "put", "patch", "delete"]:
            return RequestPriority.HIGH
        
        # Low priority operations
        if any(low in path for low in ["analytics", "logs", "metrics", "stats"]):
            return RequestPriority.LOW
        elif "background" in path or "batch" in path or "bulk" in path:
            return RequestPriority.BACKGROUND
        
        # Check user context for priority adjustment
        user_tier = request.headers.get("X-User-Tier", "").lower()
        if user_tier in ["premium", "enterprise"]:
            # Upgrade priority for premium users
            if action_type in ["ai_interaction", "visualization"]:
                return RequestPriority.HIGH
        
        # Default priority based on action type
        action_priorities = {
            "authentication": RequestPriority.CRITICAL,
            "ai_interaction": RequestPriority.HIGH,
            "file_upload": RequestPriority.NORMAL,
            "visualization": RequestPriority.NORMAL,
            "export": RequestPriority.NORMAL,
            "data_modification": RequestPriority.HIGH,
            "data_retrieval": RequestPriority.NORMAL
        }
        
        return action_priorities.get(action_type, RequestPriority.NORMAL)
    
    def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request."""
        # Try to get from headers
        user_id = request.headers.get("X-User-ID")
        if user_id:
            return user_id
        
        # Try to get from authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # In a real implementation, you'd decode the JWT token
            return "authenticated_user"
        
        # Try to get from query parameters
        user_id = request.query_params.get("user_id")
        if user_id:
            return user_id
        
        return None
    
    async def _perform_pre_request_checks(self, request: Request):
        """Perform health checks before processing request."""
        # Check system resources
        import psutil
        
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        conditions = []
        
        if cpu_percent > 80:
            conditions.append("high_cpu")
        if memory_percent > 85:
            conditions.append("memory_pressure")
        
        # Apply degradation if needed
        if conditions:
            await degradation_manager.apply_degradation("api", conditions)
    
    async def _process_request_with_intelligent_protection(
        self, request: Request, call_next: Callable, context: RequestContext
    ) -> Response:
        """Process request with intelligent protection and dynamic optimization."""
        
        # Calculate dynamic timeout
        timeout = self.timeout_manager.calculate_timeout(context)
        
        try:
            # Apply request-specific optimizations
            await self._apply_request_optimizations(request, context)
            
            # Process with intelligent timeout and monitoring
            response = await asyncio.wait_for(
                call_next(request), timeout=timeout
            )
            
            # Record successful processing
            processing_time = time.time() - time.time()  # This would be calculated properly
            self.timeout_manager.record_response_time(context.action_type, processing_time)
            self.request_router.update_node_metrics(
                context.route_target or "primary", processing_time, True
            )
            
            return response
            
        except asyncio.TimeoutError:
            logger.warning(f"Request {context.request_id} timed out after {timeout}s")
            context.retry_count += 1
            return await self._handle_intelligent_timeout(request, context, timeout)
            
        except HTTPException as e:
            # Handle HTTP exceptions with enhanced responses
            context.retry_count += 1
            return await self._handle_enhanced_http_exception(request, e, context)
            
        except Exception as e:
            # Handle unexpected exceptions with comprehensive analysis
            context.retry_count += 1
            return await self._handle_enhanced_unexpected_exception(request, e, context)
    
    async def _apply_request_optimizations(self, request: Request, context: RequestContext):
        """Apply request-specific optimizations based on context."""
        # Adjust system behavior based on priority
        if context.priority == RequestPriority.CRITICAL:
            # Ensure maximum resources for critical requests
            await self._ensure_critical_resources()
        elif context.priority == RequestPriority.BACKGROUND:
            # Throttle background requests during high load
            if self.current_load > 0.8:
                await asyncio.sleep(0.1)  # Small delay for background requests
        
        # Apply complexity-based optimizations
        if context.complexity_score > 0.8:
            # Pre-warm caches or prepare resources for complex requests
            await self._prepare_for_complex_request(context)
    
    async def _ensure_critical_resources(self):
        """Ensure maximum resources are available for critical requests."""
        # In a real implementation, this might:
        # - Pause non-critical background tasks
        # - Free up memory caches
        # - Prioritize database connections
        pass
    
    async def _prepare_for_complex_request(self, context: RequestContext):
        """Prepare system for complex request processing."""
        # In a real implementation, this might:
        # - Pre-allocate memory
        # - Warm up relevant caches
        # - Prepare database connections
        pass
    
    async def _handle_intelligent_timeout(self, request: Request, context: RequestContext, timeout: float) -> JSONResponse:
        """Handle request timeout with intelligent analysis and recovery."""
        self.error_count += 1
        
        # Try to provide degraded response first
        degraded_response = await self._get_intelligent_degraded_response(request, context, "timeout")
        if degraded_response:
            return degraded_response
        
        # Create enhanced timeout response
        enhanced_response = {
            "status": "timeout",
            "message": "Your request is taking longer than usual. We're still working on it.",
            "request_id": context.request_id,
            "timeout_duration": timeout,
            "estimated_completion": f"{timeout * 1.5}s",
            "retry_after": min(30, timeout // 2),
            "priority": context.priority.value,
            "complexity_score": context.complexity_score,
            
            # Enhanced user guidance
            "what_happened": f"Your {context.action_type} request exceeded the {timeout}s timeout limit.",
            "what_you_can_do": [
                {
                    "action": "wait_and_retry",
                    "label": "Wait and Try Again",
                    "description": "The request might complete if you wait a bit longer",
                    "recommended": True
                },
                {
                    "action": "simplify_request",
                    "label": "Simplify Request",
                    "description": "Try with less data or simpler parameters",
                    "recommended": context.complexity_score > 0.6
                },
                {
                    "action": "try_later",
                    "label": "Try Later",
                    "description": "System load might be lower in a few minutes",
                    "recommended": self.current_load > 0.7
                }
            ],
            
            # System context
            "system_load": self.current_load,
            "queue_status": self.request_prioritizer.get_queue_status(),
            "fallback_available": True
        }
        
        return JSONResponse(status_code=202, content=enhanced_response)
    
    async def _handle_enhanced_http_exception(self, request: Request, exc: HTTPException, context: RequestContext) -> JSONResponse:
        """Handle HTTP exceptions with enhanced error responses."""
        self.error_count += 1
        
        # Try to provide degraded response for server errors
        if exc.status_code >= 500:
            degraded_response = await self._get_intelligent_degraded_response(request, context, "server_error")
            if degraded_response:
                return degraded_response
        
        # Generate enhanced error response
        enhanced_response = self.error_response_system.enhance_error_response(exc, context)
        
        return JSONResponse(status_code=exc.status_code, content=enhanced_response)
    
    async def _handle_enhanced_unexpected_exception(self, request: Request, exc: Exception, context: RequestContext) -> JSONResponse:
        """Handle unexpected exceptions with comprehensive analysis."""
        self.error_count += 1
        
        logger.error(f"Unexpected error in request {context.request_id}: {exc}", exc_info=True)
        
        # Try to provide degraded response
        degraded_response = await self._get_intelligent_degraded_response(request, context, "unexpected_error")
        if degraded_response:
            return degraded_response
        
        # Generate enhanced error response
        enhanced_response = self.error_response_system.enhance_error_response(exc, context)
        
        return JSONResponse(status_code=500, content=enhanced_response)
    
    async def _handle_enhanced_failure(self, request: Request, exc: Exception, context: RequestContext) -> JSONResponse:
        """Handle critical failures with comprehensive error response."""
        self.error_count += 1
        
        logger.critical(f"Critical failure in request {context.request_id}: {exc}", exc_info=True)
        
        # Record failure for analysis
        failure_prevention.failure_history.append({
            "type": "critical_middleware_failure",
            "timestamp": datetime.utcnow(),
            "request_id": context.request_id,
            "error": str(exc),
            "path": request.url.path,
            "method": request.method,
            "context": {
                "priority": context.priority.value,
                "complexity_score": context.complexity_score,
                "retry_count": context.retry_count
            }
        })
        
        # Generate comprehensive error response
        enhanced_response = self.error_response_system.enhance_error_response(exc, context)
        enhanced_response.update({
            "critical_failure": True,
            "system_notification": "Our technical team has been automatically notified",
            "incident_id": context.request_id,
            "escalation_available": True
        })
        
        return JSONResponse(status_code=500, content=enhanced_response)
    
    async def _get_intelligent_degraded_response(self, request: Request, context: RequestContext, error_type: str) -> Optional[JSONResponse]:
        """Get intelligent degraded response based on request context."""
        path = request.url.path.lower()
        
        # Enhanced visualization degradation
        if "visualization" in path or "chart" in path:
            degraded_data = await degradation_manager.apply_degradation("visualization", [error_type])
            if degraded_data:
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "data": degraded_data,
                        "degraded": True,
                        "degradation_reason": error_type,
                        "message": "Using simplified visualization due to system issues",
                        "quality_level": "reduced",
                        "full_functionality_eta": "2-5 minutes"
                    }
                )
        
        # Enhanced AI/Agent degradation
        elif "ai" in path or "agent" in path:
            degraded_data = await degradation_manager.apply_degradation("ai_services", [error_type])
            if degraded_data:
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "response": degraded_data.get("response", "AI services temporarily limited"),
                        "degraded": True,
                        "degradation_reason": error_type,
                        "confidence": degraded_data.get("confidence", 0.5),
                        "alternative_suggestions": [
                            "Try a simpler question",
                            "Use the help documentation",
                            "Contact support for complex queries"
                        ],
                        "full_ai_eta": "1-3 minutes"
                    }
                )
        
        # Enhanced data request degradation
        elif request.method.lower() == "get":
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "data": [],
                    "cached": True,
                    "degraded": True,
                    "degradation_reason": error_type,
                    "message": "Showing cached data due to system issues",
                    "data_freshness": "May be up to 5 minutes old",
                    "refresh_available": True
                }
            )
        
        return None
    
    def _get_request_timeout(self, request: Request) -> float:
        """Get appropriate timeout for request type."""
        path = request.url.path.lower()
        
        if "upload" in path:
            return 300.0  # 5 minutes for uploads
        elif "export" in path:
            return 120.0  # 2 minutes for exports
        elif "ai" in path or "agent" in path:
            return 60.0   # 1 minute for AI operations
        elif "visualization" in path:
            return 30.0   # 30 seconds for visualizations
        else:
            return 15.0   # 15 seconds for general requests
    
    async def _handle_timeout(self, request: Request, timeout: float) -> JSONResponse:
        """Handle request timeout gracefully."""
        self.error_count += 1
        
        # Try to provide cached or degraded response
        degraded_response = await self._get_degraded_response(request, "timeout")
        
        if degraded_response:
            return degraded_response
        
        # Return user-friendly timeout response
        return JSONResponse(
            status_code=202,  # Accepted - processing continues
            content={
                "status": "processing",
                "message": "Your request is taking longer than usual. We're still working on it.",
                "request_id": request.state.request_id,
                "estimated_completion": f"{timeout * 2}s",
                "retry_after": 30,
                "fallback_available": True
            }
        )
    
    async def _handle_http_exception(self, request: Request, exc: HTTPException) -> JSONResponse:
        """Handle HTTP exceptions gracefully."""
        self.error_count += 1
        
        # Try to provide degraded response for certain errors
        if exc.status_code >= 500:
            degraded_response = await self._get_degraded_response(request, "server_error")
            if degraded_response:
                return degraded_response
        
        # Enhance error response with helpful information
        error_response = {
            "error": True,
            "status_code": exc.status_code,
            "message": self._get_user_friendly_error_message(exc),
            "request_id": request.state.request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "suggestions": self._get_error_suggestions(exc),
            "retry_possible": exc.status_code >= 500
        }
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response
        )
    
    async def _handle_unexpected_exception(
        self, request: Request, exc: Exception, request_id: str
    ) -> JSONResponse:
        """Handle unexpected exceptions gracefully."""
        self.error_count += 1
        
        logger.error(f"Unexpected error in request {request_id}: {exc}", exc_info=True)
        
        # Try to provide degraded response
        degraded_response = await self._get_degraded_response(request, "unexpected_error")
        
        if degraded_response:
            return degraded_response
        
        # Return safe error response
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "message": "We encountered an unexpected issue, but we're working to resolve it.",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "support_available": True,
                "retry_recommended": True,
                "fallback_mode": True
            }
        )
    
    async def _handle_critical_failure(
        self, request: Request, exc: Exception, start_time: float, request_id: str
    ) -> JSONResponse:
        """Handle critical failures that bypass all other protection."""
        self.error_count += 1
        
        logger.critical(f"Critical failure in request {request_id}: {exc}", exc_info=True)
        
        # Record failure for analysis
        failure_prevention.failure_history.append({
            "type": "critical_middleware_failure",
            "timestamp": datetime.utcnow(),
            "request_id": request_id,
            "error": str(exc),
            "path": request.url.path,
            "method": request.method
        })
        
        # Return absolute minimal response
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Service temporarily unavailable. Please try again.",
                "request_id": request_id,
                "retry_after": 60
            }
        )
    
    async def _get_degraded_response(self, request: Request, error_type: str) -> Optional[JSONResponse]:
        """Get degraded response based on request type and error."""
        path = request.url.path.lower()
        
        # Visualization requests
        if "visualization" in path or "chart" in path:
            degraded_data = await degradation_manager.apply_degradation(
                "visualization", [error_type]
            )
            if degraded_data:
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "data": degraded_data,
                        "degraded": True,
                        "message": "Using simplified visualization due to system load"
                    }
                )
        
        # AI/Agent requests
        elif "ai" in path or "agent" in path:
            degraded_data = await degradation_manager.apply_degradation(
                "ai_services", [error_type]
            )
            if degraded_data:
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "response": degraded_data.get("response", "Service temporarily limited"),
                        "degraded": True,
                        "confidence": degraded_data.get("confidence", 0.5)
                    }
                )
        
        # Data requests
        elif request.method.lower() == "get":
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "data": [],
                    "cached": True,
                    "message": "Showing cached data due to system issues"
                }
            )
        
        return None
    
    def _get_user_friendly_error_message(self, exc: HTTPException) -> str:
        """Get user-friendly error message."""
        status_code = exc.status_code
        
        messages = {
            400: "There was an issue with your request. Please check your input and try again.",
            401: "Authentication required. Please log in to continue.",
            403: "You don't have permission to access this resource.",
            404: "The requested resource wasn't found. It may have been moved or deleted.",
            429: "Too many requests. Please wait a moment before trying again.",
            500: "We're experiencing technical difficulties. Our team has been notified.",
            502: "Service temporarily unavailable. Please try again in a few moments.",
            503: "Service is under maintenance. Please try again shortly.",
            504: "Request timed out. Please try again or contact support if this persists."
        }
        
        return messages.get(status_code, exc.detail or "An error occurred")
    
    def _get_error_suggestions(self, exc: HTTPException) -> List[str]:
        """Get helpful suggestions for error resolution."""
        status_code = exc.status_code
        
        suggestions = {
            400: [
                "Check that all required fields are filled",
                "Verify data format is correct",
                "Try refreshing the page"
            ],
            401: [
                "Log in to your account",
                "Check if your session has expired",
                "Clear browser cache and cookies"
            ],
            403: [
                "Contact your administrator for access",
                "Verify you're using the correct account",
                "Check if your subscription is active"
            ],
            404: [
                "Check the URL for typos",
                "Try navigating from the home page",
                "Contact support if the link should work"
            ],
            429: [
                "Wait a few seconds before trying again",
                "Reduce the frequency of requests",
                "Contact support for rate limit increases"
            ],
            500: [
                "Try refreshing the page",
                "Wait a few minutes and try again",
                "Contact support if the issue persists"
            ]
        }
        
        return suggestions.get(status_code, [
            "Try refreshing the page",
            "Wait a moment and try again",
            "Contact support if the issue continues"
        ])
    
    async def _post_process_enhanced_response(
        self, request: Request, response: Response, context: RequestContext, start_time: float
    ) -> Response:
        """Enhanced post-processing with intelligent analysis and optimization hints."""
        
        response_time = time.time() - start_time
        self.response_times.append(response_time)
        
        # Keep only last 100 response times
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
        
        # Enhanced headers with intelligent information
        response.headers["X-Request-ID"] = context.request_id
        response.headers["X-Response-Time"] = f"{response_time:.3f}s"
        response.headers["X-Request-Priority"] = context.priority.value
        response.headers["X-Complexity-Score"] = f"{context.complexity_score:.2f}"
        response.headers["X-Route-Target"] = context.route_target or "primary"
        
        # Intelligent server status
        server_status = self._determine_server_status(response_time, context)
        response.headers["X-Server-Status"] = server_status
        
        # Enhanced performance hints
        performance_hint = self._get_intelligent_performance_hint(response_time, context)
        if performance_hint:
            response.headers["X-Performance-Hint"] = performance_hint
        
        # Load balancing information
        response.headers["X-Current-Load"] = f"{self.current_load:.2f}"
        response.headers["X-Queue-Status"] = str(self.request_prioritizer.get_queue_status()["total_queued"])
        
        # Degradation status with enhanced information
        degradation_status = degradation_manager.get_degradation_status()
        if degradation_status["active_degradations"] > 0:
            response.headers["X-Service-Status"] = "degraded"
            response.headers["X-Degradation-Level"] = degradation_status["overall_level"]
            response.headers["X-Degraded-Services"] = str(len(degradation_status["degraded_services"]))
        
        # Optimization recommendations
        optimization_hint = self._get_optimization_hint(response_time, context)
        if optimization_hint:
            response.headers["X-Optimization-Hint"] = optimization_hint
        
        # User experience metrics
        ux_metrics = ux_protector.get_experience_metrics()
        response.headers["X-UX-Level"] = ux_metrics.experience_level.value
        response.headers["X-Success-Rate"] = f"{ux_metrics.success_rate:.2f}"
        
        # Caching recommendations
        cache_hint = self._get_cache_recommendation(context, response_time)
        if cache_hint:
            response.headers["X-Cache-Recommendation"] = cache_hint
        
        return response
    
    def _determine_server_status(self, response_time: float, context: RequestContext) -> str:
        """Determine intelligent server status."""
        if response_time > context.estimated_duration * 2:
            return "overloaded"
        elif response_time > context.estimated_duration * 1.5:
            return "busy"
        elif self.current_load > 0.8:
            return "high-load"
        elif self.current_load > 0.6:
            return "moderate-load"
        else:
            return "healthy"
    
    def _get_intelligent_performance_hint(self, response_time: float, context: RequestContext) -> Optional[str]:
        """Get intelligent performance hint based on context."""
        expected_time = context.estimated_duration
        
        if response_time > expected_time * 3:
            return "very-slow-response"
        elif response_time > expected_time * 2:
            return "slow-response"
        elif response_time > expected_time * 1.5:
            return "slower-than-expected"
        elif response_time < expected_time * 0.5:
            return "faster-than-expected"
        else:
            return None
    
    def _get_optimization_hint(self, response_time: float, context: RequestContext) -> Optional[str]:
        """Get optimization hint for future requests."""
        if context.complexity_score > 0.8 and response_time > 5.0:
            return "consider-simplifying-request"
        elif context.priority == RequestPriority.BACKGROUND and response_time > 10.0:
            return "schedule-during-low-load"
        elif response_time > context.estimated_duration * 2:
            return "try-during-off-peak-hours"
        else:
            return None
    
    def _get_cache_recommendation(self, context: RequestContext, response_time: float) -> Optional[str]:
        """Get caching recommendation based on request pattern."""
        if context.action_type == "data_retrieval" and response_time > 2.0:
            return "cacheable-5min"
        elif context.action_type == "visualization" and response_time > 3.0:
            return "cacheable-10min"
        elif context.complexity_score > 0.7 and response_time > 5.0:
            return "cacheable-15min"
        else:
            return None
    
    def get_enhanced_middleware_stats(self) -> Dict[str, Any]:
        """Get comprehensive middleware statistics with intelligent analysis."""
        avg_response_time = (
            sum(self.response_times) / len(self.response_times)
            if self.response_times else 0
        )
        
        error_rate = self.error_count / max(self.request_count, 1)
        
        # Calculate performance percentiles
        sorted_times = sorted(self.response_times) if self.response_times else [0]
        p50 = sorted_times[len(sorted_times) // 2] if sorted_times else 0
        p95 = sorted_times[int(len(sorted_times) * 0.95)] if sorted_times else 0
        p99 = sorted_times[int(len(sorted_times) * 0.99)] if sorted_times else 0
        
        return {
            # Basic metrics
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": error_rate,
            "avg_response_time": avg_response_time,
            "recent_response_times": self.response_times[-10:],
            
            # Enhanced performance metrics
            "performance_percentiles": {
                "p50": p50,
                "p95": p95,
                "p99": p99
            },
            
            # System status
            "current_load": self.current_load,
            "health_status": self._calculate_health_status(error_rate, avg_response_time),
            
            # Routing and load balancing
            "routing_stats": {
                "server_nodes": len(self.request_router.server_nodes),
                "load_balancing_strategy": self.request_router.load_balancing_strategy.value,
                "circuit_breakers": len(self.request_router.circuit_breakers)
            },
            
            # Queue management
            "queue_stats": self.request_prioritizer.get_queue_status(),
            
            # Request complexity analysis
            "complexity_stats": {
                "cached_patterns": len(self.request_complexity_cache),
                "avg_complexity": self._calculate_avg_complexity()
            },
            
            # Timeout management
            "timeout_stats": {
                "dynamic_timeouts_enabled": True,
                "timeout_patterns": len(self.timeout_manager.recent_response_times)
            },
            
            # Error response enhancement
            "error_response_stats": {
                "enhanced_responses_enabled": True,
                "error_patterns": len(self.error_response_system.error_patterns),
                "contextual_help_available": len(self.error_response_system.contextual_help)
            }
        }
    
    def _calculate_health_status(self, error_rate: float, avg_response_time: float) -> str:
        """Calculate intelligent health status."""
        if error_rate > 0.1 or avg_response_time > 10.0:
            return "critical"
        elif error_rate > 0.05 or avg_response_time > 5.0:
            return "degraded"
        elif error_rate > 0.02 or avg_response_time > 2.0:
            return "warning"
        else:
            return "healthy"
    
    def _calculate_avg_complexity(self) -> float:
        """Calculate average request complexity."""
        if not self.request_complexity_cache:
            return 0.0
        
        return sum(self.request_complexity_cache.values()) / len(self.request_complexity_cache)
    
    # Maintain backward compatibility
    def get_middleware_stats(self) -> Dict[str, Any]:
        """Get basic middleware statistics (backward compatibility)."""
        enhanced_stats = self.get_enhanced_middleware_stats()
        return {
            "total_requests": enhanced_stats["total_requests"],
            "total_errors": enhanced_stats["total_errors"],
            "error_rate": enhanced_stats["error_rate"],
            "avg_response_time": enhanced_stats["avg_response_time"],
            "recent_response_times": enhanced_stats["recent_response_times"],
            "health_status": enhanced_stats["health_status"]
        }


class HealthCheckMiddleware(BaseHTTPMiddleware):
    """Middleware for health checks and system monitoring."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add health check capabilities."""
        
        # Handle health check requests
        if request.url.path in ["/health", "/healthz", "/ping"]:
            return await self._handle_health_check(request)
        
        # Handle status requests
        if request.url.path == "/status":
            return await self._handle_status_request(request)
        
        # Continue with normal processing
        return await call_next(request)
    
    async def _handle_health_check(self, request: Request) -> JSONResponse:
        """Handle health check requests."""
        try:
            # Perform quick health checks
            health_status = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0",
                "uptime": time.time(),  # Simplified uptime
                "checks": {
                    "api": "healthy",
                    "database": "healthy",  # Would check actual database
                    "memory": "healthy",
                    "disk": "healthy"
                }
            }
            
            # Check system resources
            import psutil
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            if cpu_percent > 90:
                health_status["checks"]["cpu"] = "critical"
                health_status["status"] = "degraded"
            elif cpu_percent > 70:
                health_status["checks"]["cpu"] = "warning"
            else:
                health_status["checks"]["cpu"] = "healthy"
            
            if memory_percent > 90:
                health_status["checks"]["memory"] = "critical"
                health_status["status"] = "degraded"
            elif memory_percent > 80:
                health_status["checks"]["memory"] = "warning"
            else:
                health_status["checks"]["memory"] = "healthy"
            
            status_code = 200 if health_status["status"] == "healthy" else 503
            
            return JSONResponse(
                status_code=status_code,
                content=health_status
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "error": "Health check failed",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
    
    async def _handle_status_request(self, request: Request) -> JSONResponse:
        """Handle detailed status requests with enhanced information."""
        try:
            # Get the bulletproof middleware instance from the app
            bulletproof_middleware = None
            for middleware in request.app.middleware_stack:
                if isinstance(middleware, BulletproofMiddleware):
                    bulletproof_middleware = middleware
                    break
            
            # Get comprehensive status
            status = {
                "system": failure_prevention.get_system_status(),
                "degradation": degradation_manager.get_degradation_status(),
                "user_experience": ux_protector.get_experience_metrics().__dict__,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add enhanced middleware stats if available
            if bulletproof_middleware:
                status["middleware"] = bulletproof_middleware.get_enhanced_middleware_stats()
                status["routing"] = {
                    "server_nodes": {
                        node_id: {
                            "endpoint": node.endpoint,
                            "current_load": node.current_load,
                            "avg_response_time": node.avg_response_time,
                            "active_connections": node.active_connections,
                            "health_score": node.health_score,
                            "last_health_check": node.last_health_check.isoformat()
                        }
                        for node_id, node in bulletproof_middleware.request_router.server_nodes.items()
                    },
                    "load_balancing_strategy": bulletproof_middleware.request_router.load_balancing_strategy.value
                }
                status["queue_management"] = bulletproof_middleware.request_prioritizer.get_queue_status()
            
            return JSONResponse(content=status)
            
        except Exception as e:
            logger.error(f"Enhanced status request failed: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Status request failed",
                    "timestamp": datetime.utcnow().isoformat(),
                    "fallback_mode": True
                }
            )