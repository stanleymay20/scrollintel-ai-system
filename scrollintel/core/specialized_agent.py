"""
SpecializedAgent Base Class for ScrollIntel Architecture Improvements
Implements standardized agent interface with schema validation and lifecycle management
Based on requirements 1.1 and 2.1 from scrollintel-architecture-improvements spec
"""

import asyncio
import logging
import json
import uuid
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import jsonschema
from jsonschema import validate, ValidationError

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Agent operational status for lifecycle management"""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    BUSY = "busy"
    IDLE = "idle"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    OFFLINE = "offline"

class AgentCapability(Enum):
    """Standard agent capabilities for precise scoping"""
    ANALYSIS = "analysis"
    GENERATION = "generation"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    OPTIMIZATION = "optimization"
    PREDICTION = "prediction"
    DECISION_MAKING = "decision_making"
    LEARNING = "learning"
    COMMUNICATION = "communication"
    ORCHESTRATION = "orchestration"

@dataclass
class AgentMetrics:
    """Comprehensive agent performance and health metrics"""
    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    
    # Load and capacity metrics
    current_load: float = 0.0
    peak_load: float = 0.0
    concurrent_requests: int = 0
    max_concurrent_requests: int = 0
    
    # Resource metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    
    # Temporal metrics
    last_activity: datetime = field(default_factory=datetime.now)
    uptime: timedelta = field(default_factory=lambda: timedelta(0))
    
    # Quality metrics
    error_rate: float = 0.0
    success_rate: float = 1.0
    throughput: float = 0.0
    performance_score: float = 1.0
    
    # Schema validation metrics
    schema_validation_failures: int = 0
    schema_validation_success_rate: float = 1.0

@dataclass
class AgentConfiguration:
    """Agent configuration settings for lifecycle and performance management"""
    # Capacity settings
    max_concurrent_requests: int = 10
    timeout_seconds: int = 300
    retry_attempts: int = 3
    
    # Health and monitoring
    health_check_interval: int = 30
    metrics_collection_interval: int = 10
    
    # Lifecycle management
    startup_timeout: int = 60
    shutdown_timeout: int = 30
    graceful_degradation_enabled: bool = True
    
    # Schema validation
    strict_schema_validation: bool = True
    schema_version: str = "1.0"
    
    # Performance optimization
    auto_scaling_enabled: bool = True
    priority_level: int = 1
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    
    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentRequest:
    """Standardized agent request format with schema validation support"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    request_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Request metadata
    priority: int = 1
    timeout: int = 300
    retry_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    # User context
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Schema validation
    schema_version: str = "1.0"
    capabilities_required: List[str] = field(default_factory=list)
    
    # Tracing and debugging
    trace_id: Optional[str] = None
    parent_request_id: Optional[str] = None

@dataclass
class AgentResponse:
    """
    Standardized agent response format matching design interface
    
    Design Interface:
    - id: string
    - status: 'success' | 'error' | 'partial'
    - data: any
    - confidence: number
    - processingTime: number
    - metadata: Record<string, any>
    """
    # Core interface properties (matching design)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "success"  # success, error, partial
    data: Any = None
    confidence: float = 1.0
    processingTime: float = 0.0  # Design interface uses camelCase
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Additional properties for enhanced functionality
    request_id: str = ""
    agent_id: str = ""
    
    # Performance metrics
    processing_time: float = 0.0  # Keep snake_case for backward compatibility
    queue_time: float = 0.0
    
    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Schema validation results
    schema_validation_passed: bool = True
    schema_validation_errors: List[str] = field(default_factory=list)
    
    # Tracing
    trace_id: Optional[str] = None
    processing_steps: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Ensure processingTime and processing_time are synchronized"""
        if self.processingTime == 0.0 and self.processing_time > 0.0:
            self.processingTime = self.processing_time
        elif self.processing_time == 0.0 and self.processingTime > 0.0:
            self.processing_time = self.processingTime

@dataclass
class HealthCheck:
    """Individual health check result"""
    name: str
    status: str  # healthy, degraded, unhealthy
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class HealthStatus:
    """Comprehensive agent health status information"""
    status: AgentStatus
    checks: List[HealthCheck] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Temporal information
    uptime: timedelta = field(default_factory=lambda: timedelta(0))
    last_restart: Optional[datetime] = None
    
    # Error tracking
    error_count: int = 0
    warning_count: int = 0
    critical_issues: List[str] = field(default_factory=list)
    
    # Performance indicators
    performance_score: float = 1.0
    capacity_utilization: float = 0.0
    response_time_p95: float = 0.0
    
    # Schema validation health
    schema_validation_health: bool = True

class SpecializedAgent(ABC):
    """
    Base class for all ScrollIntel agents implementing architecture improvements
    
    Provides:
    - Standardized process(), healthCheck(), and getMetrics() methods
    - JSON schema validation for requests and responses
    - Agent lifecycle management (startup, shutdown, graceful degradation)
    - Performance monitoring and health checking
    
    Requirements: 1.1, 2.1 from scrollintel-architecture-improvements
    
    Interface matches design specification:
    - id: string
    - capabilities: string[]
    - schema: JSONSchema
    - process(request: AgentRequest): Promise<AgentResponse>
    - healthCheck(): Promise<HealthStatus>
    - getMetrics(): Promise<AgentMetrics>
    - shutdown(): Promise<void>
    """
    
    def __init__(self, agent_id: str, name: str, capabilities: Set[AgentCapability], 
                 configuration: Optional[AgentConfiguration] = None):
        # Core agent properties (matching design interface)
        self.id = agent_id  # Design interface uses 'id' not 'agent_id'
        self.agent_id = agent_id  # Keep for backward compatibility
        self.name = name
        self.capabilities = capabilities
        self.configuration = configuration or AgentConfiguration()
        
        # Schema property (matching design interface)
        self.schema = self._define_request_schema()
        
        # Agent state
        self.status = AgentStatus.INITIALIZING
        self.metrics = AgentMetrics()
        self.start_time = datetime.now()
        self.last_health_check = datetime.now()
        
        # Schema validation
        self.request_schema = self._define_request_schema()
        self.response_schema = self._define_response_schema()
        self._schema_validator = self._create_schema_validator()
        
        # Request management
        self._active_requests: Dict[str, AgentRequest] = {}
        self._request_history: List[AgentRequest] = []
        self._request_queue: List[AgentRequest] = []
        
        # Lifecycle management
        self._shutdown_event = asyncio.Event()
        self._health_check_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        
        # Initialize agent asynchronously
        self._initialization_task = asyncio.create_task(self._initialize())
    
    # Required abstract methods for all specialized agents
    
    @abstractmethod
    async def process(self, request: AgentRequest) -> AgentResponse:
        """
        Core processing method - must be implemented by all agents
        
        This method should:
        1. Validate the request matches agent's scope of responsibility
        2. Process the request according to agent's single responsibility
        3. Return structured response with proper schema validation
        
        Args:
            request: Validated AgentRequest with required capabilities
            
        Returns:
            AgentResponse: Structured response with validation results
            
        Raises:
            ValueError: If request doesn't match agent's capabilities
            TimeoutError: If processing exceeds timeout
        """
        pass
    
    @abstractmethod
    def _define_request_schema(self) -> Dict[str, Any]:
        """
        Define JSON schema for request validation
        
        Must return a valid JSON Schema that defines:
        - Required fields in request payload
        - Data types and constraints
        - Validation rules specific to this agent's responsibility
        
        Returns:
            Dict: JSON Schema for request validation
        """
        pass
    
    @abstractmethod
    def _define_response_schema(self) -> Dict[str, Any]:
        """
        Define JSON schema for response validation
        
        Must return a valid JSON Schema that defines:
        - Structure of response data
        - Required fields and data types
        - Validation rules for response format
        
        Returns:
            Dict: JSON Schema for response validation
        """
        pass
    
    def get_capabilities(self) -> Set[AgentCapability]:
        """
        Return agent capabilities for precise scoping
        
        Returns:
            Set[AgentCapability]: Agent's specific capabilities
        """
        return self.capabilities
    
    @property
    def capabilities_list(self) -> List[str]:
        """
        Return agent capabilities as string list (matching design interface)
        
        Returns:
            List[str]: Agent's capabilities as strings
        """
        return [cap.value for cap in self.capabilities]
    
    async def _initialize(self):
        """
        Initialize the agent with proper lifecycle management
        Implements startup phase of agent lifecycle
        """
        try:
            logger.info(f"Initializing agent {self.agent_id}")
            
            # Validate configuration
            self._validate_configuration()
            
            # Setup agent-specific initialization
            await asyncio.wait_for(
                self._setup_agent(), 
                timeout=self.configuration.startup_timeout
            )
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Validate schemas
            self._validate_schemas()
            
            # Mark as healthy
            self.status = AgentStatus.HEALTHY
            logger.info(f"Agent {self.agent_id} initialized successfully")
            
        except asyncio.TimeoutError:
            self.status = AgentStatus.ERROR
            logger.error(f"Agent {self.agent_id} initialization timed out")
        except Exception as e:
            self.status = AgentStatus.ERROR
            logger.error(f"Agent {self.agent_id} initialization failed: {e}")
    
    async def _setup_agent(self):
        """
        Setup agent-specific initialization - override in subclasses
        This method should initialize any agent-specific resources,
        connections, or state required for operation.
        """
        pass
    
    def _validate_configuration(self):
        """Validate agent configuration"""
        if not self.agent_id:
            raise ValueError("Agent ID is required")
        if not self.name:
            raise ValueError("Agent name is required")
        if not self.capabilities:
            raise ValueError("Agent must have at least one capability")
    
    def _validate_schemas(self):
        """Validate that request and response schemas are valid JSON schemas"""
        try:
            jsonschema.Draft7Validator.check_schema(self.request_schema)
            jsonschema.Draft7Validator.check_schema(self.response_schema)
        except jsonschema.SchemaError as e:
            raise ValueError(f"Invalid schema definition: {e}")
    
    async def _start_background_tasks(self):
        """Start background monitoring tasks"""
        if self.configuration.health_check_interval > 0:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        if self.configuration.metrics_collection_interval > 0:
            self._metrics_task = asyncio.create_task(self._metrics_collection_loop())
    
    def _create_schema_validator(self):
        """Create schema validator instance"""
        return {
            'request': jsonschema.Draft7Validator({}),  # Will be updated when schema is defined
            'response': jsonschema.Draft7Validator({})  # Will be updated when schema is defined
        }
    
    def _validate_request_schema(self, request: AgentRequest) -> List[str]:
        """
        Validate request against defined schema
        
        Args:
            request: AgentRequest to validate
            
        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        errors = []
        try:
            # Update validator with current schema
            validator = jsonschema.Draft7Validator(self.request_schema)
            
            # Validate payload against schema
            validation_errors = list(validator.iter_errors(request.payload))
            for error in validation_errors:
                errors.append(f"Schema validation error at {'.'.join(str(p) for p in error.path)}: {error.message}")
                
        except Exception as e:
            errors.append(f"Schema validation failed: {str(e)}")
            
        return errors
    
    def _validate_response_schema(self, response: AgentResponse) -> List[str]:
        """
        Validate response against defined schema
        
        Args:
            response: AgentResponse to validate
            
        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        errors = []
        try:
            # Update validator with current schema
            validator = jsonschema.Draft7Validator(self.response_schema)
            
            # Validate data against schema
            if response.data is not None:
                validation_errors = list(validator.iter_errors(response.data))
                for error in validation_errors:
                    errors.append(f"Response schema validation error at {'.'.join(str(p) for p in error.path)}: {error.message}")
                    
        except Exception as e:
            errors.append(f"Response schema validation failed: {str(e)}")
            
        return errors
    
    def _validate_request_capabilities(self, request: AgentRequest) -> bool:
        """
        Validate that request matches agent's scope of responsibility
        
        Args:
            request: AgentRequest to validate
            
        Returns:
            bool: True if request matches agent capabilities
        """
        if not request.capabilities_required:
            return True
            
        required_capabilities = set(request.capabilities_required)
        agent_capabilities = {cap.value for cap in self.capabilities}
        
        return required_capabilities.issubset(agent_capabilities)
    
    async def execute_request(self, request: AgentRequest) -> AgentResponse:
        """
        Execute a request with comprehensive validation, metrics, and error handling
        
        Implements:
        - Schema validation for requests and responses
        - Capability validation to ensure request matches agent scope
        - Performance monitoring and metrics collection
        - Graceful error handling and degradation
        
        Args:
            request: AgentRequest to process
            
        Returns:
            AgentResponse: Processed response with validation results
        """
        start_time = datetime.now()
        queue_start_time = start_time
        
        try:
            # Validate agent is operational
            if self.status in [AgentStatus.OFFLINE, AgentStatus.SHUTTING_DOWN, AgentStatus.ERROR]:
                raise Exception(f"Agent is not operational (status: {self.status.value})")
            
            # Validate request capabilities match agent scope
            if not self._validate_request_capabilities(request):
                raise ValueError(f"Request capabilities {request.capabilities_required} don't match agent capabilities {[c.value for c in self.capabilities]}")
            
            # Validate request schema
            schema_errors = self._validate_request_schema(request)
            if schema_errors and self.configuration.strict_schema_validation:
                raise ValueError(f"Request schema validation failed: {'; '.join(schema_errors)}")
            
            # Check agent capacity
            if len(self._active_requests) >= self.configuration.max_concurrent_requests:
                if self.configuration.graceful_degradation_enabled:
                    # Queue the request
                    self._request_queue.append(request)
                    raise Exception("Agent at maximum capacity - request queued")
                else:
                    raise Exception("Agent at maximum capacity")
            
            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.concurrent_requests = len(self._active_requests) + 1
            self.metrics.max_concurrent_requests = max(
                self.metrics.max_concurrent_requests, 
                self.metrics.concurrent_requests
            )
            
            # Track active request
            self._active_requests[request.request_id] = request
            self.status = AgentStatus.BUSY
            
            # Calculate queue time
            processing_start_time = datetime.now()
            queue_time = (processing_start_time - queue_start_time).total_seconds()
            
            # Process request with timeout
            response = await self._process_with_timeout(request)
            
            # Validate response schema
            response_schema_errors = self._validate_response_schema(response)
            if response_schema_errors:
                response.schema_validation_passed = False
                response.schema_validation_errors = response_schema_errors
                self.metrics.schema_validation_failures += 1
                
                if self.configuration.strict_schema_validation:
                    raise ValueError(f"Response schema validation failed: {'; '.join(response_schema_errors)}")
            
            # Update success metrics
            self.metrics.successful_requests += 1
            processing_time = (datetime.now() - processing_start_time).total_seconds()
            total_time = (datetime.now() - start_time).total_seconds()
            
            self._update_response_time(processing_time)
            
            # Populate response metadata
            response.processing_time = processing_time
            response.queue_time = queue_time
            response.timestamp = datetime.now()
            response.trace_id = request.trace_id
            
            return response
            
        except Exception as e:
            # Handle errors with detailed tracking
            self.metrics.failed_requests += 1
            if self.metrics.total_requests > 0:
                self.metrics.error_rate = self.metrics.failed_requests / self.metrics.total_requests
                self.metrics.success_rate = self.metrics.successful_requests / self.metrics.total_requests
            
            error_response = AgentResponse(
                request_id=request.request_id,
                agent_id=self.agent_id,
                status="error",
                errors=[str(e)],
                processing_time=(datetime.now() - start_time).total_seconds(),
                queue_time=(datetime.now() - queue_start_time).total_seconds(),
                trace_id=request.trace_id
            )
            
            logger.error(f"Agent {self.agent_id} request {request.request_id} failed: {e}")
            
            # Update agent status based on error rate
            if self.metrics.error_rate > 0.5:  # More than 50% error rate
                self.status = AgentStatus.DEGRADED
            elif self.metrics.error_rate > 0.8:  # More than 80% error rate
                self.status = AgentStatus.UNHEALTHY
            
            return error_response
            
        finally:
            # Cleanup and metrics update
            if request.request_id in self._active_requests:
                del self._active_requests[request.request_id]
            
            self.metrics.concurrent_requests = len(self._active_requests)
            
            # Update request history
            self._request_history.append(request)
            if len(self._request_history) > 1000:  # Keep last 1000 requests
                self._request_history = self._request_history[-1000:]
            
            # Update agent status
            if len(self._active_requests) == 0:
                if self.metrics.error_rate < 0.1:
                    self.status = AgentStatus.HEALTHY
                elif self.metrics.error_rate < 0.5:
                    self.status = AgentStatus.DEGRADED
                else:
                    self.status = AgentStatus.UNHEALTHY
            else:
                self.status = AgentStatus.BUSY
                
            self.metrics.last_activity = datetime.now()
            
            # Process queued requests if capacity available
            await self._process_queued_requests()
    
    async def _process_with_timeout(self, request: AgentRequest) -> AgentResponse:
        """Process request with timeout handling and tracing"""
        timeout = request.timeout or self.configuration.timeout_seconds
        
        try:
            response = await asyncio.wait_for(
                self.process(request), 
                timeout=timeout
            )
            
            # Add processing steps for tracing
            if not response.processing_steps:
                response.processing_steps = [f"Processed by {self.agent_id}"]
                
            return response
            
        except asyncio.TimeoutError:
            raise Exception(f"Request timed out after {timeout} seconds")
    
    def _update_response_time(self, processing_time: float):
        """Update response time metrics with min/max tracking"""
        # Update min/max response times
        self.metrics.min_response_time = min(self.metrics.min_response_time, processing_time)
        self.metrics.max_response_time = max(self.metrics.max_response_time, processing_time)
        
        # Update average response time (running average)
        if self.metrics.successful_requests == 1:
            self.metrics.average_response_time = processing_time
        else:
            n = self.metrics.successful_requests
            self.metrics.average_response_time = (
                (self.metrics.average_response_time * (n - 1) + processing_time) / n
            )
    
    async def _process_queued_requests(self):
        """Process queued requests if capacity is available"""
        while (self._request_queue and 
               len(self._active_requests) < self.configuration.max_concurrent_requests):
            queued_request = self._request_queue.pop(0)
            # Process queued request asynchronously
            asyncio.create_task(self.execute_request(queued_request))
    
    async def health_check(self) -> HealthStatus:
        """
        Perform comprehensive health check
        
        Implements standardized health checking with:
        - Agent status validation
        - Performance metrics evaluation
        - Capacity and resource monitoring
        - Schema validation health
        - Custom agent-specific checks
        
        Returns:
            HealthStatus: Comprehensive health status with detailed checks
        """
        checks = []
        critical_issues = []
        
        # Basic agent status check
        agent_status_healthy = self.status in [AgentStatus.HEALTHY, AgentStatus.BUSY, AgentStatus.IDLE]
        checks.append(HealthCheck(
            name="agent_status",
            status="healthy" if agent_status_healthy else "unhealthy",
            message=f"Agent status: {self.status.value}",
            details={"current_status": self.status.value, "uptime": str(datetime.now() - self.start_time)}
        ))
        
        if not agent_status_healthy:
            critical_issues.append(f"Agent status is {self.status.value}")
        
        # Performance health check
        performance_healthy = (
            self.metrics.error_rate < 0.1 and
            self.metrics.average_response_time < 30.0 and
            self.metrics.success_rate > 0.9
        )
        
        performance_status = "healthy"
        if self.metrics.error_rate > 0.5 or self.metrics.average_response_time > 60.0:
            performance_status = "unhealthy"
        elif self.metrics.error_rate > 0.1 or self.metrics.average_response_time > 30.0:
            performance_status = "degraded"
            
        checks.append(HealthCheck(
            name="performance",
            status=performance_status,
            message=f"Error rate: {self.metrics.error_rate:.2%}, Avg response: {self.metrics.average_response_time:.2f}s",
            details={
                "error_rate": self.metrics.error_rate,
                "success_rate": self.metrics.success_rate,
                "average_response_time": self.metrics.average_response_time,
                "min_response_time": self.metrics.min_response_time,
                "max_response_time": self.metrics.max_response_time,
                "total_requests": self.metrics.total_requests
            }
        ))
        
        # Capacity health check
        current_load = len(self._active_requests) / self.configuration.max_concurrent_requests if self.configuration.max_concurrent_requests > 0 else 0
        capacity_status = "healthy"
        if current_load > 0.9:
            capacity_status = "unhealthy"
        elif current_load > 0.8:
            capacity_status = "degraded"
            
        checks.append(HealthCheck(
            name="capacity",
            status=capacity_status,
            message=f"Current load: {current_load:.1%}",
            details={
                "current_load": current_load,
                "active_requests": len(self._active_requests),
                "max_concurrent_requests": self.configuration.max_concurrent_requests,
                "queued_requests": len(self._request_queue),
                "peak_load": self.metrics.peak_load
            }
        ))
        
        # Schema validation health check
        schema_validation_rate = 1.0
        if self.metrics.total_requests > 0:
            schema_validation_rate = 1.0 - (self.metrics.schema_validation_failures / self.metrics.total_requests)
            
        schema_healthy = schema_validation_rate > 0.95
        checks.append(HealthCheck(
            name="schema_validation",
            status="healthy" if schema_healthy else "degraded",
            message=f"Schema validation success rate: {schema_validation_rate:.2%}",
            details={
                "validation_success_rate": schema_validation_rate,
                "validation_failures": self.metrics.schema_validation_failures,
                "strict_validation_enabled": self.configuration.strict_schema_validation
            }
        ))
        
        # Resource health check (if available)
        try:
            import psutil
            process = psutil.Process()
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            resource_healthy = cpu_percent < 80 and memory_mb < 1000  # 1GB limit
            checks.append(HealthCheck(
                name="resources",
                status="healthy" if resource_healthy else "degraded",
                message=f"CPU: {cpu_percent:.1f}%, Memory: {memory_mb:.1f}MB",
                details={
                    "cpu_percent": cpu_percent,
                    "memory_mb": memory_mb,
                    "memory_bytes": memory_info.rss
                }
            ))
        except ImportError:
            # psutil not available, skip resource check
            pass
        
        # Custom health checks from subclasses
        try:
            custom_checks = await self._custom_health_checks()
            checks.extend(custom_checks)
        except Exception as e:
            checks.append(HealthCheck(
                name="custom_checks",
                status="unhealthy",
                message=f"Custom health check failed: {str(e)}",
                details={"error": str(e)}
            ))
        
        # Determine overall status
        overall_status = AgentStatus.HEALTHY
        
        unhealthy_checks = [c for c in checks if c.status == "unhealthy"]
        degraded_checks = [c for c in checks if c.status == "degraded"]
        
        if unhealthy_checks:
            overall_status = AgentStatus.UNHEALTHY
            critical_issues.extend([f"{c.name}: {c.message}" for c in unhealthy_checks])
        elif degraded_checks:
            overall_status = AgentStatus.DEGRADED
        
        # Calculate performance score
        performance_score = self._calculate_performance_score()
        
        # Calculate capacity utilization
        capacity_utilization = current_load
        
        # Calculate 95th percentile response time (simplified)
        response_time_p95 = self.metrics.max_response_time * 0.95  # Simplified calculation
        
        self.last_health_check = datetime.now()
        
        return HealthStatus(
            status=overall_status,
            checks=checks,
            last_updated=self.last_health_check,
            uptime=datetime.now() - self.start_time,
            error_count=self.metrics.failed_requests,
            warning_count=len(degraded_checks),
            critical_issues=critical_issues,
            performance_score=performance_score,
            capacity_utilization=capacity_utilization,
            response_time_p95=response_time_p95,
            schema_validation_health=schema_healthy
        )
    
    async def _custom_health_checks(self) -> List[Dict[str, Any]]:
        """Override in subclasses for custom health checks"""
        return []
    
    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score (0.0 to 1.0)"""
        if self.metrics.total_requests == 0:
            return 1.0
        
        success_rate = self.metrics.successful_requests / self.metrics.total_requests
        response_time_score = max(0, 1 - (self.metrics.average_response_time / 60.0))  # Normalize to 60s
        load_score = 1 - (len(self._active_requests) / self.configuration.max_concurrent_requests)
        
        return (success_rate * 0.5 + response_time_score * 0.3 + load_score * 0.2)
    
    def get_metrics(self) -> AgentMetrics:
        """Get current agent metrics"""
        self.metrics.uptime = datetime.now() - self.start_time
        self.metrics.current_load = len(self._active_requests) / self.configuration.max_concurrent_requests
        return self.metrics
    
    def get_status(self) -> AgentStatus:
        """Get current agent status"""
        return self.status
    
    async def shutdown(self):
        """Gracefully shutdown the agent"""
        self.status = AgentStatus.OFFLINE
        
        # Wait for active requests to complete (with timeout)
        timeout = 30  # 30 seconds
        start_time = datetime.now()
        
        while self._active_requests and (datetime.now() - start_time).total_seconds() < timeout:
            await asyncio.sleep(0.1)
        
        # Force cleanup remaining requests
        self._active_requests.clear()
        
        logger.info(f"Agent {self.agent_id} shutdown completed")
    
    def __str__(self) -> str:
        return f"SpecializedAgent(id={self.agent_id}, name={self.name}, status={self.status.value})"
    
    def __repr__(self) -> str:
        return self.__str__()


class ScrollAlignedAgent(SpecializedAgent):
    """
    Enhanced agent with scroll alignment and spiritual validation
    """
    
    def __init__(self, agent_id: str, name: str, capabilities: Set[AgentCapability], 
                 configuration: Optional[AgentConfiguration] = None):
        super().__init__(agent_id, name, capabilities, configuration)
        self.scroll_alignment_threshold = 0.8
        self.spiritual_validation_enabled = True
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process with scroll alignment validation"""
        # Validate scroll alignment
        if self.configuration.scroll_alignment_required:
            alignment_score = await self._validate_scroll_alignment(request)
            if alignment_score < self.scroll_alignment_threshold:
                return AgentResponse(
                    request_id=request.request_id,
                    agent_id=self.agent_id,
                    status="error",
                    errors=[f"Request failed scroll alignment validation (score: {alignment_score})"],
                    scroll_alignment_score=alignment_score,
                    spiritual_validation=False
                )
        
        # Process the request
        response = await self._process_aligned_request(request)
        
        # Validate response alignment
        if self.configuration.scroll_alignment_required:
            response.scroll_alignment_score = await self._validate_response_alignment(response)
            response.spiritual_validation = response.scroll_alignment_score >= self.scroll_alignment_threshold
        
        return response
    
    @abstractmethod
    async def _process_aligned_request(self, request: AgentRequest) -> AgentResponse:
        """Process request with scroll alignment - implement in subclasses"""
        pass
    
    async def _validate_scroll_alignment(self, request: AgentRequest) -> float:
        """Validate request against scroll principles"""
        # Basic scroll alignment validation
        # This would be enhanced with actual scroll principle checking
        return 1.0  # Default to aligned
    
    async def _validate_response_alignment(self, response: AgentResponse) -> float:
        """Validate response against scroll principles"""
        # Basic response alignment validation
        # This would be enhanced with actual scroll principle checking
        return 1.0  # Default to aligned