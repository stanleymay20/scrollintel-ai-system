"""
ScrollIntel Models Module
Contains data models, database schemas, and validation classes.
"""

# Database models
from .database import (
    Base,
    User,
    Agent,
    Dataset,
    MLModel,
    Dashboard,
    AgentRequest as DBAgentRequest,
    AgentResponse as DBAgentResponse,
    AuditLog,
)

# Pydantic schemas
from .schemas import (
    UserCreate,
    UserUpdate,
    UserResponse,
    UserLogin,
    AgentCreate,
    AgentUpdate,
    AgentResponse as AgentResponseSchema,
    DatasetCreate,
    DatasetUpdate,
    DatasetResponse,
    MLModelCreate,
    MLModelUpdate,
    MLModelResponse,
    DashboardCreate,
    DashboardUpdate,
    DashboardResponse,
    AgentRequestCreate,
    AgentRequestResponse,
    AgentResponseCreate,
    AgentResponseResponse,
    AuditLogCreate,
    AuditLogResponse,
    Token,
    TokenData,
    FileUploadResponse,
    HealthCheck,
    ErrorResponse,
    PaginationParams,
    PaginatedResponse,
)

# Database utilities
from .database_utils import (
    DatabaseManager,
    db_manager,
    get_db,
    get_redis,
    init_database,
    cleanup_database,
    check_database_health,
    run_migrations,
    TestDatabaseManager,
    get_test_db_manager,
)

# Initialization and seeding
from .init_db import (
    initialize_database,
    reset_database,
    clear_all_data,
    check_database_status,
    migrate_database,
)

from .seed_data import (
    seed_database,
    clear_seed_data,
    create_default_users,
    create_default_agents,
    create_sample_datasets,
    create_sample_dashboards,
)

# Insight models
from .insight_models import (
    Pattern,
    Insight,
    ActionRecommendation,
    BusinessContext,
    Anomaly,
    InsightType,
    PatternType,
    SignificanceLevel,
    ActionPriority,
)

# Dashboard models
from .dashboard_models import (
    AnalyticsDashboard,
    Widget,
    DashboardPermission,
    DashboardTemplate,
    BusinessMetric,
    DashboardType,
    ExecutiveRole,
    WidgetType,
)

# ROI models
from .roi_models import (
    ROIAnalysis,
    CostTracking,
    BenefitTracking,
    ROIReport,
    CloudCostCollection,
    ProductivityMetric,
    EfficiencyGainMetric,
)

# Advanced Prompt models
from .prompt_models import (
    AdvancedPromptTemplate,
    AdvancedPromptVersion,
    AdvancedPromptCategory,
    AdvancedPromptTag,
    PromptVariable,
)

# Consciousness models
from .consciousness_models import (
    ConsciousnessState,
    AwarenessState,
    ConsciousnessLevel,
    AwarenessType,
    Thought,
    MetaCognitiveInsight,
    Goal,
    IntentionalState,
    Experience,
    SelfReflection,
    CognitiveContext,
)

# Intuitive reasoning models
from .intuitive_models import (
    IntuitiveInsight,
    PatternSynthesis,
    Pattern as IntuitivePattern,
    DataPoint,
    Problem,
    CreativeSolution,
    Challenge,
    HolisticInsight,
    Context as IntuitiveContext,
    IntuitiveLeap,
    NeuralArchitecture,
    ValidationResult,
    ConfidenceMetrics,
    InsightType as IntuitiveInsightType,
    PatternComplexity,
    CreativityLevel,
)

# Core interfaces (for compatibility)
from ..core.interfaces import (
    AgentRequest,
    AgentResponse,
    AgentCapability,
    SecurityContext,
    AuditEvent,
)

__all__ = [
    # Database models
    "Base",
    "User",
    "Agent",
    "Dataset",
    "MLModel",
    "Dashboard",
    "DBAgentRequest",
    "DBAgentResponse",
    "AuditLog",
    
    # Pydantic schemas
    "UserCreate",
    "UserUpdate",
    "UserResponse",
    "UserLogin",
    "AgentCreate",
    "AgentUpdate",
    "AgentResponseSchema",
    "DatasetCreate",
    "DatasetUpdate",
    "DatasetResponse",
    "MLModelCreate",
    "MLModelUpdate",
    "MLModelResponse",
    "DashboardCreate",
    "DashboardUpdate",
    "DashboardResponse",
    "AgentRequestCreate",
    "AgentRequestResponse",
    "AgentResponseCreate",
    "AgentResponseResponse",
    "AuditLogCreate",
    "AuditLogResponse",
    "Token",
    "TokenData",
    "FileUploadResponse",
    "HealthCheck",
    "ErrorResponse",
    "PaginationParams",
    "PaginatedResponse",
    
    # Database utilities
    "DatabaseManager",
    "db_manager",
    "get_db",
    "get_redis",
    "init_database",
    "cleanup_database",
    "check_database_health",
    "run_migrations",
    "TestDatabaseManager",
    "get_test_db_manager",
    
    # Initialization and seeding
    "initialize_database",
    "reset_database",
    "clear_all_data",
    "check_database_status",
    "migrate_database",
    "seed_database",
    "clear_seed_data",
    "create_default_users",
    "create_default_agents",
    "create_sample_datasets",
    "create_sample_dashboards",
    
    # Insight models
    "Pattern",
    "Insight",
    "ActionRecommendation",
    "BusinessContext",
    "Anomaly",
    "InsightType",
    "PatternType",
    "SignificanceLevel",
    "ActionPriority",
    
    # Dashboard models
    "AnalyticsDashboard",
    "Widget",
    "DashboardPermission",
    "DashboardTemplate",
    "BusinessMetric",
    "DashboardType",
    "ExecutiveRole",
    "WidgetType",
    
    # ROI models
    "ROIAnalysis",
    "CostTracking",
    "BenefitTracking",
    "ROIReport",
    "CloudCostCollection",
    "ProductivityMetric",
    "EfficiencyGainMetric",
    
    # Advanced Prompt models
    "AdvancedPromptTemplate",
    "AdvancedPromptVersion",
    "AdvancedPromptCategory",
    "AdvancedPromptTag",
    "PromptVariable",
    
    # Consciousness models
    "ConsciousnessState",
    "AwarenessState",
    "ConsciousnessLevel",
    "AwarenessType",
    "Thought",
    "MetaCognitiveInsight",
    "Goal",
    "IntentionalState",
    "Experience",
    "SelfReflection",
    "CognitiveContext",
    
    # Intuitive reasoning models
    "IntuitiveInsight",
    "PatternSynthesis",
    "IntuitivePattern",
    "DataPoint",
    "Problem",
    "CreativeSolution",
    "Challenge",
    "HolisticInsight",
    "IntuitiveContext",
    "IntuitiveLeap",
    "NeuralArchitecture",
    "ValidationResult",
    "ConfidenceMetrics",
    "IntuitiveInsightType",
    "PatternComplexity",
    "CreativityLevel",
    
    # Core interfaces (for compatibility)
    "AgentRequest",
    "AgentResponse",
    "AgentCapability",
    "SecurityContext",
    "AuditEvent",
]