"""
Enhanced Specialized Agent Framework for ScrollIntel

This module provides the base framework for creating specialized agents
with standardized interfaces, health monitoring, and performance tracking.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Type, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta
import uuid
from abc import ABC, abstractmethod
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Agent status enumeration"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    IDLE = "idle"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    RECOVERING = "recovering"

class AgentCapability(Enum):
    """Agent capability types"""
    ANALYSIS = "analysis"
    GENERATION = "generation"
    OPTIMIZATION = "optimization"
    PREDICTION = "prediction"
    DECISION_MAKING = "decision_making"
    LEARNING = "learning"
    COMMUNICATION = "communication"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    MONITORING = "monitoring"
    ORCHESTRATION = "orchestration"

class RequestPriority(Enum):
    """Request priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

@dataclass
class AgentRequest:
    """Standardized agent request format"""
    id: str
    type: str
    payload: Dict[str, Any]
    capabilities_required: List[str] = field(default_factory=list)
    priority: RequestPriority = RequestPriority.NORMAL
    timeout: int = 300
    retry_count: int = 0
    max_retries: int = 3
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None

@dataclass
class AgentResponse:
    """Standardized agent response format"""
    request_id: str
    agent_id: str
    status: str  # success, error, partial, timeout
    data: Any
    confidence: float
    processing_time: float
    error: Optional[str] = None
    error_code: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    trace_id: Optional[str] = None

@dataclass
class HealthCheck:
    """Individual health check result"""
    name: str
    status: str  # pass, warn, fail
    details: str
    timestamp: datetime = field(default_factory=datetime.now)
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HealthStatus:
    """Agent health status"""
    status: str  # healthy, degraded, unhealthy
    checks: List[HealthCheck] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    uptime: timedelta = field(default_factory=lambda: timedelta(0))
    version: str = "1.0.0"
    build_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    average_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    current_load: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0  # requests per second
    last_activity: datetime = field(default_factory=datetime.now)
    uptime: timedelta = field(default_factory=lambda: timedelta(0))
    custom_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class AgentConfiguration:
    """Agent configuration settings"""
    max_concurrent_requests: int = 10
    request_timeout: int = 300
    health_check_interval: int = 30
    metrics_collection_interval: int = 60
    retry_attempts: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    enable_tracing: bool = True
    enable_metrics: bool = True
    log_level: str = "INFO"
    custom_settings: Dict[str, Any] = field(default_factory=dict)

class CircuitBreaker:
    """Circuit breaker for agent fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == "open":
                if self._should_attempt_reset():
                    self.state = "half-open"
                else:
                    raise Exception("Circuit breaker is open")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self.last_failure_time is None:
            return True
        return (datetime.now() - self.last_failure_time).total_seconds() > self.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution"""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

class EnhancedSpecializedAgent(ABC):
    """
    Enhanced base class for all specialized agents in ScrollIntel.
    
    Provides standardized interfaces, health monitoring, metrics collection,
    lifecycle management, and fault tolerance for all agent implementations.
    """
    
    def __init__(self, agent_id: str, name: str, capabilities: List[AgentCapability], 
                 config: Optional[AgentConfiguration] = None):
        self.agent_id = agent_id
        self.name = name
        self.capabilities = capabilities
        self.config = config or AgentConfiguration()
        self.status = AgentStatus.INITIALIZING
        self.metrics = AgentMetrics()
        self.start_time = datetime.now()
        self.current_requests = 0
        self.version = "1.0.0"
        
        # Request tracking
        self._request_history = []
        self._active_requests: Dict[str, AgentRequest] = {}
        
        # Circuit breaker for fault tolerance
        self._circuit_breaker = CircuitBreaker(
            self.config.circuit_breaker_threshold,
            self.config.circuit_breaker_timeout
        )
        
        # Threading and async support
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_requests)
        self._shutdown_event = asyncio.Event()
        self._health_check_task = None
        self._metrics_task = None
        
        # Schema cache
        self._schema_cache = {}
        
        # Lifecycle hooks
        self._startup_hooks: List[Callable] = []
        self._shutdown_hooks: List[Callable] = []
    
    @abstractmethod
    async def process(self, request: AgentRequest) -> AgentResponse:
        """
        Process an agent request and return a response.
        
        This method must be implemented by all specialized agents.
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Return the JSON schema for request/response validation.
        
        This method must be implemented by all specialized agents.
        """
        pass
    
    async def initialize(self):
        """Initialize the agent"""
        try:
            await self._startup()
            self.status = AgentStatus.IDLE
            
            # Start background tasks
            if self.config.enable_metrics:
                self._health_check_task = asyncio.create_task(self._health_check_loop())
                self._metrics_task = asyncio.create_task(self._metrics_collection_loop())
            
            logger.info(f"Agent {self.agent_id} initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent {self.agent_id}: {e}")
            self.status = AgentStatus.ERROR
            raise
    
    async def handle_request(self, request: AgentRequest) -> AgentResponse:
        """
        Handle an incoming request with validation, processing, and metrics.
        """
        start_time = time.time()
        trace_id = str(uuid.uuid4())
        
        try:
            # Pre-request validation
            validation_result = await self._pre_request_validation(request)
            if not validation_result["valid"]:
                return self._create_error_response(
                    request, "validation_error", 
                    validation_result["error"], 
                    time.time() - start_time,
                    trace_id
                )
            
            # Check capacity and circuit breaker
            if not await self._check_capacity():
                return self._create_error_response(
                    request, "capacity_exceeded", 
                    "Agent at maximum capacity", 
                    time.time() - start_time,
                    trace_id
                )
            
            # Track active request
            self._active_requests[request.id] = request
            self.current_requests += 1
            self.status = AgentStatus.BUSY
            self.metrics.total_requests += 1
            
            # Process with circuit breaker protection
            try:
                response = await self._process_with_timeout(request)
                response.trace_id = trace_id
                
                # Update success metrics
                processing_time = time.time() - start_time
                response.processing_time = processing_time
                self._update_metrics(processing_time, True)
                self.metrics.successful_requests += 1
                
                return response
                
            except asyncio.TimeoutError:
                self.metrics.timeout_requests += 1
                return self._create_error_response(
                    request, "timeout", 
                    f"Request timed out after {request.timeout}s", 
                    time.time() - start_time,
                    trace_id
                )
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} error processing request {request.id}: {e}")
            self.metrics.failed_requests += 1
            self._update_metrics(time.time() - start_time, False)
            
            return self._create_error_response(
                request, "processing_error", str(e), 
                time.time() - start_time, trace_id,
                {"traceback": traceback.format_exc()}
            )
        
        finally:
            # Cleanup
            self._active_requests.pop(request.id, None)
            self.current_requests -= 1
            if self.current_requests == 0:
                self.status = AgentStatus.IDLE
    
    async def _process_with_timeout(self, request: AgentRequest) -> AgentResponse:
        """Process request with timeout protection"""
        try:
            return await asyncio.wait_for(
                self.process(request), 
                timeout=request.timeout
            )
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(f"Request {request.id} timed out")
    
    async def health_check(self) -> HealthStatus:
        """
        Perform comprehensive health check and return status.
        """
        checks = []
        overall_status = "healthy"
        
        # Basic health checks
        checks.append(HealthCheck(
            name="agent_status",
            status="pass" if self.status not in [AgentStatus.ERROR, AgentStatus.OFFLINE] else "fail",
            details=f"Agent status: {self.status.value}"
        ))
        
        checks.append(HealthCheck(
            name="request_capacity",
            status="pass" if self.current_requests < self.config.max_concurrent_requests else "warn",
            details=f"Current requests: {self.current_requests}/{self.config.max_concurrent_requests}"
        ))
        
        # Performance checks
        if self.metrics.average_response_time > 30.0:
            checks.append(HealthCheck(
                name="response_time",
                status="warn",
                details=f"Average response time: {self.metrics.average_response_time:.2f}s"
            ))
            overall_status = "degraded"
        
        # Error rate check
        if self.metrics.total_requests > 0:
            error_rate = self.metrics.failed_requests / self.metrics.total_requests
            if error_rate > 0.1:
                checks.append(HealthCheck(
                    name="error_rate",
                    status="fail",
                    details=f"Error rate: {error_rate:.2%}"
                ))
                overall_status = "unhealthy"
        
        # Circuit breaker check
        if self._circuit_breaker.state == "open":
            checks.append(HealthCheck(
                name="circuit_breaker",
                status="fail",
                details="Circuit breaker is open"
            ))
            overall_status = "unhealthy"
        
        # Custom health checks
        custom_checks = await self._custom_health_checks()
        checks.extend(custom_checks)
        
        # Determine overall status from all checks
        if any(check.status == "fail" for check in checks):
            overall_status = "unhealthy"
        elif any(check.status == "warn" for check in checks) and overall_status == "healthy":
            overall_status = "degraded"
        
        return HealthStatus(
            status=overall_status,
            checks=checks,
            last_updated=datetime.now(),
            uptime=datetime.now() - self.start_time,
            version=self.version,
            build_info={
                "agent_id": self.agent_id,
                "capabilities": [cap.value for cap in self.capabilities],
                "config": self.config.__dict__
            }
        )
    
    def get_metrics(self) -> AgentMetrics:
        """
        Get current agent metrics with calculated values.
        """
        self.metrics.uptime = datetime.now() - self.start_time
        self.metrics.current_load = self.current_requests / self.config.max_concurrent_requests
        
        # Calculate error rate
        if self.metrics.total_requests > 0:
            self.metrics.error_rate = self.metrics.failed_requests / self.metrics.total_requests
        
        # Calculate throughput (requests per second over last minute)
        recent_requests = [
            req for req in self._request_history 
            if (datetime.now() - req["timestamp"]).total_seconds() <= 60
        ]
        self.metrics.throughput = len(recent_requests) / 60.0
        
        return self.metrics
    
    async def shutdown(self):
        """
        Gracefully shutdown the agent.
        """
        logger.info(f"Shutting down agent {self.agent_id}")
        self.status = AgentStatus.OFFLINE
        self._shutdown_event.set()
        
        # Cancel background tasks
        if self._health_check_task:
            self._health_check_task.cancel()
        if self._metrics_task:
            self._metrics_task.cancel()
        
        # Wait for current requests to complete (with timeout)
        timeout = 30  # 30 seconds
        start_time = time.time()
        
        while self.current_requests > 0 and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.1)
        
        if self.current_requests > 0:
            logger.warning(f"Agent {self.agent_id} shutdown with {self.current_requests} active requests")
        
        # Run shutdown hooks
        for hook in self._shutdown_hooks:
            try:
                await hook()
            except Exception as e:
                logger.error(f"Shutdown hook failed: {e}")
        
        # Perform cleanup
        await self._cleanup()
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
    
    def add_startup_hook(self, hook: Callable):
        """Add a startup hook"""
        self._startup_hooks.append(hook)
    
    def add_shutdown_hook(self, hook: Callable):
        """Add a shutdown hook"""
        self._shutdown_hooks.append(hook)
    
    async def _pre_request_validation(self, request: AgentRequest) -> Dict[str, Any]:
        """Validate incoming request"""
        try:
            # Basic validation
            if not request.id or not request.type or request.payload is None:
                return {"valid": False, "error": "Missing required fields"}
            
            # Check required capabilities
            for capability in request.capabilities_required:
                if capability not in [cap.value for cap in self.capabilities]:
                    return {
                        "valid": False, 
                        "error": f"Agent missing capability: {capability}"
                    }
            
            # Check deadline
            if request.deadline and datetime.now() > request.deadline:
                return {"valid": False, "error": "Request deadline exceeded"}
            
            return {"valid": True}
            
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}
    
    async def _check_capacity(self) -> bool:
        """Check if agent has capacity for new requests"""
        return self.current_requests < self.config.max_concurrent_requests
    
    def _create_error_response(self, request: AgentRequest, error_code: str, 
                             error_message: str, processing_time: float, 
                             trace_id: str, metadata: Optional[Dict] = None) -> AgentResponse:
        """Create standardized error response"""
        return AgentResponse(
            request_id=request.id,
            agent_id=self.agent_id,
            status="error",
            data=None,
            confidence=0.0,
            processing_time=processing_time,
            error=error_message,
            error_code=error_code,
            trace_id=trace_id,
            metadata=metadata or {}
        )
    
    def _update_metrics(self, processing_time: float, success: bool):
        """Update agent performance metrics"""
        # Update response time statistics
        if success:
            self.metrics.min_response_time = min(self.metrics.min_response_time, processing_time)
            self.metrics.max_response_time = max(self.metrics.max_response_time, processing_time)
        
        # Update average response time
        total_requests = self.metrics.total_requests
        if total_requests > 0:
            self.metrics.average_response_time = (
                (self.metrics.average_response_time * (total_requests - 1) + processing_time) / total_requests
            )
        else:
            self.metrics.average_response_time = processing_time
        
        self.metrics.last_activity = datetime.now()
        
        # Store request history for analysis
        self._request_history.append({
            "timestamp": datetime.now(),
            "processing_time": processing_time,
            "success": success
        })
        
        # Keep only last 1000 requests
        if len(self._request_history) > 1000:
            self._request_history = self._request_history[-1000:]
    
    async def _health_check_loop(self):
        """Background health check loop"""
        while not self._shutdown_event.is_set():
            try:
                await self.health_check()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(self.config.health_check_interval)
    
    async def _metrics_collection_loop(self):
        """Background metrics collection loop"""
        while not self._shutdown_event.is_set():
            try:
                self.get_metrics()
                await asyncio.sleep(self.config.metrics_collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection loop error: {e}")
                await asyncio.sleep(self.config.metrics_collection_interval)
    
    async def _startup(self):
        """Agent startup initialization"""
        # Run startup hooks
        for hook in self._startup_hooks:
            await hook()
        
        # Custom startup logic
        await self._custom_startup()
    
    async def _custom_startup(self):
        """Override for custom startup logic"""
        pass
    
    async def _custom_health_checks(self) -> List[HealthCheck]:
        """Override for custom health checks"""
        return []
    
    async def _cleanup(self):
        """Override for custom cleanup"""
        pass

# Enhanced Agent Registry
class EnhancedAgentRegistry:
    """
    Enhanced registry for managing specialized agents with advanced features.
    """
    
    def __init__(self):
        self.agents: Dict[str, EnhancedSpecializedAgent] = {}
        self.capabilities_map: Dict[str, List[str]] = {}
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.health_statuses: Dict[str, HealthStatus] = {}
        self._lock = threading.Lock()
    
    async def register_agent(self, agent: EnhancedSpecializedAgent):
        """
        Register a specialized agent.
        """
        with self._lock:
            self.agents[agent.agent_id] = agent
            
            # Update capabilities map
            for capability in agent.capabilities:
                if capability.value not in self.capabilities_map:
                    self.capabilities_map[capability.value] = []
                self.capabilities_map[capability.value].append(agent.agent_id)
        
        # Initialize the agent
        await agent.initialize()
        
        logger.info(f"Registered agent {agent.agent_id} with capabilities: {[c.value for c in agent.capabilities]}")
    
    async def unregister_agent(self, agent_id: str):
        """
        Unregister a specialized agent.
        """
        with self._lock:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                
                # Shutdown agent
                await agent.shutdown()
                
                # Remove from capabilities map
                for capability in agent.capabilities:
                    if capability.value in self.capabilities_map:
                        self.capabilities_map[capability.value].remove(agent_id)
                        if not self.capabilities_map[capability.value]:
                            del self.capabilities_map[capability.value]
                
                del self.agents[agent_id]
                self.agent_metrics.pop(agent_id, None)
                self.health_statuses.pop(agent_id, None)
                
                logger.info(f"Unregistered agent {agent_id}")
    
    def get_agents_by_capability(self, capability: str) -> List[EnhancedSpecializedAgent]:
        """
        Get agents that have a specific capability.
        """
        with self._lock:
            agent_ids = self.capabilities_map.get(capability, [])
            return [self.agents[agent_id] for agent_id in agent_ids if agent_id in self.agents]
    
    def get_agent(self, agent_id: str) -> Optional[EnhancedSpecializedAgent]:
        """
        Get agent by ID.
        """
        with self._lock:
            return self.agents.get(agent_id)
    
    def get_all_agents(self) -> List[EnhancedSpecializedAgent]:
        """
        Get all registered agents.
        """
        with self._lock:
            return list(self.agents.values())
    
    async def health_check_all(self) -> Dict[str, HealthStatus]:
        """
        Perform health check on all agents.
        """
        health_statuses = {}
        
        agents = self.get_all_agents()
        for agent in agents:
            try:
                health_status = await agent.health_check()
                health_statuses[agent.agent_id] = health_status
                self.health_statuses[agent.agent_id] = health_status
            except Exception as e:
                logger.error(f"Health check failed for agent {agent.agent_id}: {e}")
                health_status = HealthStatus(
                    status="unhealthy",
                    checks=[HealthCheck(name="health_check", status="fail", details=str(e))]
                )
                health_statuses[agent.agent_id] = health_status
                self.health_statuses[agent.agent_id] = health_status
        
        return health_statuses
    
    def get_metrics_all(self) -> Dict[str, AgentMetrics]:
        """
        Get metrics for all agents.
        """
        metrics = {}
        agents = self.get_all_agents()
        
        for agent in agents:
            try:
                agent_metrics = agent.get_metrics()
                metrics[agent.agent_id] = agent_metrics
                self.agent_metrics[agent.agent_id] = agent_metrics
            except Exception as e:
                logger.error(f"Failed to get metrics for agent {agent.agent_id}: {e}")
        
        return metrics
    
    async def route_request(self, request: AgentRequest) -> AgentResponse:
        """
        Route a request to the best available agent based on capabilities and performance.
        """
        # Find agents with required capabilities
        suitable_agents = []
        
        for capability in request.capabilities_required:
            agents = self.get_agents_by_capability(capability)
            if not suitable_agents:
                suitable_agents = agents
            else:
                # Intersection of agents that have all required capabilities
                suitable_agents = [agent for agent in suitable_agents if agent in agents]
        
        if not suitable_agents:
            return AgentResponse(
                request_id=request.id,
                agent_id="router",
                status="error",
                data=None,
                confidence=0.0,
                processing_time=0.0,
                error=f"No agents available with capabilities: {request.capabilities_required}",
                error_code="no_suitable_agents"
            )
        
        # Select best agent based on load and performance
        selected_agent = self._select_best_agent(suitable_agents, request)
        
        return await selected_agent.handle_request(request)
    
    def _select_best_agent(self, agents: List[EnhancedSpecializedAgent], 
                          request: AgentRequest) -> EnhancedSpecializedAgent:
        """
        Select the best agent based on current load and performance metrics.
        """
        # For now, simple load-based selection
        # In a full implementation, this would consider multiple factors
        best_agent = None
        lowest_load = float('inf')
        
        for agent in agents:
            if agent.status in [AgentStatus.IDLE, AgentStatus.ACTIVE]:
                current_load = agent.current_requests / agent.config.max_concurrent_requests
                if current_load < lowest_load:
                    lowest_load = current_load
                    best_agent = agent
        
        # Fallback to first available agent
        return best_agent or agents[0]

# Global enhanced agent registry instance
enhanced_agent_registry = EnhancedAgentRegistry()

# Example enhanced specialized agent implementations
class EnhancedAnalysisAgent(EnhancedSpecializedAgent):
    """
    Enhanced specialized agent for data analysis tasks.
    """
    
    def __init__(self):
        super().__init__(
            agent_id=f"analysis-{uuid.uuid4().hex[:8]}",
            name="Enhanced Analysis Agent",
            capabilities=[AgentCapability.ANALYSIS, AgentCapability.PREDICTION]
        )
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        """
        Process analysis request with enhanced capabilities.
        """
        # Simulate analysis processing
        await asyncio.sleep(0.1)
        
        analysis_result = {
            "analysis_type": request.payload.get("type", "general"),
            "data_points": len(request.payload.get("data", [])),
            "insights": ["Enhanced insight 1", "Enhanced insight 2"],
            "confidence_score": 0.85,
            "processing_metadata": {
                "algorithm_used": "enhanced_analysis_v2",
                "processing_time": 0.1,
                "data_quality_score": 0.92
            }
        }
        
        return AgentResponse(
            request_id=request.id,
            agent_id=self.agent_id,
            status="success",
            data=analysis_result,
            confidence=0.85,
            processing_time=0.0  # Will be set by handle_request
        )
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Return enhanced schema for analysis requests.
        """
        return {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["statistical", "predictive", "descriptive", "advanced"]},
                "data": {"type": "array"},
                "parameters": {"type": "object"},
                "quality_threshold": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": ["type", "data"]
        }

class EnhancedGenerationAgent(EnhancedSpecializedAgent):
    """
    Enhanced specialized agent for content generation tasks.
    """
    
    def __init__(self):
        super().__init__(
            agent_id=f"generation-{uuid.uuid4().hex[:8]}",
            name="Enhanced Generation Agent",
            capabilities=[AgentCapability.GENERATION, AgentCapability.COMMUNICATION]
        )
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        """
        Process generation request with enhanced capabilities.
        """
        # Simulate generation processing
        await asyncio.sleep(0.2)
        
        generation_result = {
            "content_type": request.payload.get("type", "text"),
            "generated_content": f"Enhanced generated content for: {request.payload.get('prompt', 'default')}",
            "word_count": 150,
            "quality_score": 0.92,
            "generation_metadata": {
                "model_version": "enhanced_gen_v2",
                "creativity_score": 0.88,
                "coherence_score": 0.94,
                "relevance_score": 0.91
            }
        }
        
        return AgentResponse(
            request_id=request.id,
            agent_id=self.agent_id,
            status="success",
            data=generation_result,
            confidence=0.92,
            processing_time=0.0  # Will be set by handle_request
        )
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Return enhanced schema for generation requests.
        """
        return {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["text", "code", "summary", "creative"]},
                "prompt": {"type": "string"},
                "parameters": {"type": "object"},
                "style": {"type": "string", "enum": ["formal", "casual", "technical", "creative"]},
                "max_length": {"type": "integer", "minimum": 1}
            },
            "required": ["type", "prompt"]
        }

# Utility functions
async def create_enhanced_analysis_agent() -> EnhancedAnalysisAgent:
    """Create and register an enhanced analysis agent."""
    agent = EnhancedAnalysisAgent()
    await enhanced_agent_registry.register_agent(agent)
    return agent

async def create_enhanced_generation_agent() -> EnhancedGenerationAgent:
    """Create and register an enhanced generation agent."""
    agent = EnhancedGenerationAgent()
    await enhanced_agent_registry.register_agent(agent)
    return agent

async def route_enhanced_request(request: AgentRequest) -> AgentResponse:
    """
    Route a request using the enhanced agent registry.
    """
    return await enhanced_agent_registry.route_request(request)