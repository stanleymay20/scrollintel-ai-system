"""
Configuration management system for ScrollIntel.
Handles API keys, database connections, and system settings.
"""

import os
from typing import Optional, Dict, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from functools import lru_cache

from .interfaces import ConfigurationError


class DatabaseConfig(BaseSettings):
    """Database configuration settings."""
    
    # PostgreSQL settings
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_db: str = Field(default="scrollintel", env="POSTGRES_DB")
    postgres_user: str = Field(default="postgres", env="POSTGRES_USER")
    postgres_password: str = Field(default="", env="POSTGRES_PASSWORD")
    
    # Redis settings
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # Vector database settings
    pinecone_api_key: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    pinecone_environment: Optional[str] = Field(default=None, env="PINECONE_ENVIRONMENT")
    supabase_url: Optional[str] = Field(default=None, env="SUPABASE_URL")
    supabase_key: Optional[str] = Field(default=None, env="SUPABASE_KEY")
    
    @property
    def postgres_url(self) -> str:
        """Get PostgreSQL connection URL."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def redis_url(self) -> str:
        """Get Redis connection URL."""
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"


class AIServiceConfig(BaseSettings):
    """AI service configuration settings."""
    
    # OpenAI settings
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    openai_max_tokens: int = Field(default=4000, env="OPENAI_MAX_TOKENS")
    
    # Anthropic Claude settings
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-sonnet-20240229", env="ANTHROPIC_MODEL")
    anthropic_max_tokens: int = Field(default=4000, env="ANTHROPIC_MAX_TOKENS")
    
    # Whisper settings
    whisper_model: str = Field(default="whisper-1", env="WHISPER_MODEL")
    
    @field_validator("openai_api_key", "anthropic_api_key")
    @classmethod
    def validate_api_keys(cls, v):
        """Validate that required API keys are provided."""
        # Allow None for migration purposes, but warn
        if v is None or (isinstance(v, str) and v.strip() == ""):
            import warnings
            warnings.warn("API key not provided - some features may not work")
            return None
        return v


class SecurityConfig(BaseSettings):
    """Security configuration settings for EXOUSIA system."""
    
    # JWT settings
    jwt_secret_key: str = Field(default="", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration_hours: int = Field(default=24, env="JWT_EXPIRATION_HOURS")
    
    # Session settings
    session_timeout_minutes: int = Field(default=60, env="SESSION_TIMEOUT_MINUTES")
    max_sessions_per_user: int = Field(default=5, env="MAX_SESSIONS_PER_USER")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window_minutes: int = Field(default=15, env="RATE_LIMIT_WINDOW_MINUTES")
    
    @field_validator("jwt_secret_key")
    @classmethod
    def validate_jwt_secret(cls, v):
        """Validate JWT secret key is provided."""
        if not v or len(v) < 32:
            raise ValueError("JWT secret key must be at least 32 characters long")
        return v


class SystemConfig(BaseSettings):
    """System-wide configuration settings."""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # API settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    
    # File storage
    upload_max_size_mb: int = Field(default=100, env="UPLOAD_MAX_SIZE_MB")
    storage_path: str = Field(default="./storage", env="STORAGE_PATH")
    
    # Agent settings
    max_concurrent_agents: int = Field(default=10, env="MAX_CONCURRENT_AGENTS")
    agent_timeout_seconds: int = Field(default=300, env="AGENT_TIMEOUT_SECONDS")
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        """Validate environment is one of allowed values."""
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        return v


class ScrollIntelConfig:
    """Main configuration class that combines all configuration sections."""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.ai_services = AIServiceConfig()
        self.security = SecurityConfig()
        self.system = SystemConfig()
    
    @property
    def database_url(self) -> str:
        """Get database URL for SQLAlchemy."""
        return self.database.postgres_url
    
    @property
    def environment(self) -> str:
        """Get current environment."""
        return self.system.environment
    
    @property
    def debug(self) -> bool:
        """Get debug mode status."""
        return self.system.debug
    
    @property
    def db_pool_size(self) -> int:
        """Get database pool size."""
        return 5
    
    @property
    def db_max_overflow(self) -> int:
        """Get database max overflow."""
        return 10
    
    @property
    def redis_host(self) -> str:
        """Get Redis host."""
        return self.database.redis_host
    
    @property
    def redis_port(self) -> int:
        """Get Redis port."""
        return self.database.redis_port
    
    @property
    def redis_password(self) -> Optional[str]:
        """Get Redis password."""
        return self.database.redis_password
    
    @property
    def redis_db(self) -> int:
        """Get Redis database number."""
        return self.database.redis_db
    
    def validate(self) -> None:
        """Validate all configuration sections."""
        try:
            # Validate database config
            if not self.database.postgres_password:
                raise ConfigurationError("PostgreSQL password is required")
            
            # Validate AI service config (warn if missing but don't fail)
            if not self.ai_services.openai_api_key:
                import warnings
                warnings.warn("OpenAI API key not configured - AI features will not work")
            
            if not self.ai_services.anthropic_api_key:
                import warnings
                warnings.warn("Anthropic API key not configured - AI features will not work")
            
            # Validate security config
            if not self.security.jwt_secret_key:
                raise ConfigurationError("JWT secret key is required")
            
            # Validate vector database (at least one must be configured)
            has_pinecone = bool(self.database.pinecone_api_key and self.database.pinecone_environment)
            has_supabase = bool(self.database.supabase_url and self.database.supabase_key)
            
            if not (has_pinecone or has_supabase):
                raise ConfigurationError("At least one vector database (Pinecone or Supabase) must be configured")
            
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {str(e)}")
    
    def get_database_connections(self) -> Dict[str, str]:
        """Get database connection strings."""
        return {
            "postgres": self.database.postgres_url,
            "redis": self.database.redis_url,
        }
    
    def get_ai_service_config(self) -> Dict[str, Any]:
        """Get AI service configuration."""
        return {
            "openai": {
                "api_key": self.ai_services.openai_api_key,
                "model": self.ai_services.openai_model,
                "max_tokens": self.ai_services.openai_max_tokens,
            },
            "anthropic": {
                "api_key": self.ai_services.anthropic_api_key,
                "model": self.ai_services.anthropic_model,
                "max_tokens": self.ai_services.anthropic_max_tokens,
            },
            "whisper": {
                "model": self.ai_services.whisper_model,
            }
        }
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.system.environment == "production"
    
    def is_debug(self) -> bool:
        """Check if debug mode is enabled."""
        return self.system.debug


@lru_cache()
def get_config() -> ScrollIntelConfig:
    """Get cached configuration instance."""
    config = ScrollIntelConfig()
    config.validate()
    return config


@lru_cache()
def get_settings() -> ScrollIntelConfig:
    """Alias for get_config() for compatibility."""
    return get_config()


def load_config_from_file(file_path: str) -> ScrollIntelConfig:
    """Load configuration from a file."""
    if not os.path.exists(file_path):
        raise ConfigurationError(f"Configuration file not found: {file_path}")
    
    # Load environment variables from file
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
    
    return get_config()