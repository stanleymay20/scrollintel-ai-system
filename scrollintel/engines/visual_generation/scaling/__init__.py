"""
Horizontal scaling module for visual generation system.
Provides load balancing, distributed caching, session management, and database optimization.
"""

from .load_balancer import (
    VisualGenerationLoadBalancer,
    WorkerNode,
    WorkerStatus,
    LoadBalancingStrategy,
    load_balancer
)

from .distributed_cache import (
    DistributedVisualCache,
    CacheEntry,
    CacheStrategy,
    SemanticSimilarityEngine,
    distributed_cache
)

from .session_manager import (
    DistributedSessionManager,
    UserSession,
    GenerationRequest,
    SessionStatus,
    RequestStatus,
    session_manager
)

from .database_optimizer import (
    DatabaseOptimizer,
    ConnectionPoolConfig,
    QueryType,
    QueryMetrics,
    database_optimizer
)

from .horizontal_scaler import (
    HorizontalScalingCoordinator,
    ScalingMetrics,
    horizontal_scaler
)

__all__ = [
    # Load Balancer
    'VisualGenerationLoadBalancer',
    'WorkerNode',
    'WorkerStatus',
    'LoadBalancingStrategy',
    'load_balancer',
    
    # Distributed Cache
    'DistributedVisualCache',
    'CacheEntry',
    'CacheStrategy',
    'SemanticSimilarityEngine',
    'distributed_cache',
    
    # Session Manager
    'DistributedSessionManager',
    'UserSession',
    'GenerationRequest',
    'SessionStatus',
    'RequestStatus',
    'session_manager',
    
    # Database Optimizer
    'DatabaseOptimizer',
    'ConnectionPoolConfig',
    'QueryType',
    'QueryMetrics',
    'database_optimizer',
    
    # Horizontal Scaler
    'HorizontalScalingCoordinator',
    'ScalingMetrics',
    'horizontal_scaler'
]