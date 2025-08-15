"""
ScrollIntel Configuration Management
Handles environment-specific configuration with validation
"""

import os
from typing import Optional, List
from pydantic import validator
from pydantic_settings import BaseSettings
from enum import Enum


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """
    Application settings with environment-specific configuration
    """
    
    # === Environment Settings ===
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    log_level: LogLevel = LogLevel.INFO
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # === Database Configuration ===
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "scrollintel"
    postgres_user: str = "postgres"
    postgres_password: str
    
    # === Redis Configuration ===
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0
    
    # === JWT & Security Configuration ===
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    session_timeout_minutes: int = 60
    max_sessions_per_user: int = 5
    rate_limit_requests: int = 100
    
    # === AI Services ===
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-sonnet-20240229"
    
    # === Vector Database ===
    pinecone_api_key: Optional[str] = None
    pinecone_environment: str = "us-east-1"
    
    # === Supabase ===
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None
    supabase_service_role: Optional[str] = None
    supabase_jwt_secret: Optional[str] = None
    
    # === File Storage ===
    upload_dir: str = "./uploads"
    max_file_size: str = "100MB"
    
    # === Database Connection Pool Settings ===
    db_pool_size: int = 20
    db_max_overflow: int = 30
    db_pool_timeout: int = 30
    db_pool_recycle: int = 3600
    db_pool_pre_ping: bool = True
    
    # === File Processing Settings ===
    upload_max_size_mb: int = 100
    file_processing_timeout: int = 300  # 5 minutes
    file_processing_chunk_size: int = 8192  # 8KB chunks
    max_concurrent_uploads: int = 10
    background_job_timeout: int = 1800  # 30 minutes
    
    # === Performance Settings ===
    enable_query_cache: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes
    enable_compression: bool = True
    max_memory_usage_mb: int = 1024  # 1GB
    
    # === Monitoring & Analytics ===
    sentry_dsn: Optional[str] = None
    posthog_api_key: Optional[str] = None
    
    # === External Services ===
    stripe_api_key: Optional[str] = None
    stripe_webhook_secret: Optional[str] = None
    
    # === Backup System ===
    backup_directory: str = "./backups"
    backup_retention_days: int = 30
    backup_s3_bucket: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    
    # === Production Settings ===
    workers: int = 1
    worker_class: str = "uvicorn.workers.UvicornWorker"
    worker_connections: int = 1000
    max_requests: int = 1000
    max_requests_jitter: int = 100
    preload_app: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
    @validator("environment", pre=True)
    def validate_environment(cls, v):
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    @validator("log_level", pre=True)
    def validate_log_level(cls, v):
        if isinstance(v, str):
            return LogLevel(v.upper())
        return v
    
    @validator("max_file_size")
    def validate_max_file_size(cls, v):
        """Convert file size string to bytes"""
        if isinstance(v, str):
            v = v.upper()
            if v.endswith("MB"):
                return int(v[:-2]) * 1024 * 1024
            elif v.endswith("GB"):
                return int(v[:-2]) * 1024 * 1024 * 1024
            elif v.endswith("KB"):
                return int(v[:-2]) * 1024
        return int(v)
    
    @property
    def database_url(self) -> str:
        """Get PostgreSQL database URL"""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def async_database_url(self) -> str:
        """Get async PostgreSQL database URL"""
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def redis_url(self) -> str:
        """Get Redis URL"""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_test(self) -> bool:
        return self.environment == Environment.TEST


class DeploymentConfig:
    """
    Deployment-specific configuration management
    """
    
    @staticmethod
    def get_environment_file(env: str) -> str:
        """Get the appropriate environment file"""
        env_files = {
            "development": ".env",
            "staging": ".env.staging",
            "production": ".env.production",
            "test": ".env.test"
        }
        return env_files.get(env, ".env")
    
    @staticmethod
    def load_settings(env: Optional[str] = None) -> Settings:
        """Load settings for specific environment"""
        if env:
            env_file = DeploymentConfig.get_environment_file(env)
            if os.path.exists(env_file):
                return Settings(_env_file=env_file)
        return Settings()
    
    @staticmethod
    def get_docker_compose_file(env: str) -> str:
        """Get appropriate docker-compose file"""
        compose_files = {
            "development": "docker-compose.yml",
            "staging": "docker-compose.staging.yml",
            "production": "docker-compose.prod.yml",
            "test": "docker-compose.test.yml"
        }
        return compose_files.get(env, "docker-compose.yml")


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings


def reload_settings(env: Optional[str] = None) -> Settings:
    """Reload settings for specific environment"""
    global settings
    settings = DeploymentConfig.load_settings(env)
    return settings


# Legacy config classes for backward compatibility
class ScrollIntelConfig:
    """Legacy config class - use Settings instead"""
    def __init__(self):
        self.settings = get_settings()
    
    def __getattr__(self, name):
        return getattr(self.settings, name)

class DatabaseConfig:
    """Legacy database config class"""
    def __init__(self):
        self.settings = get_settings()
    
    @property
    def url(self):
        return self.settings.database_url
    
    @property
    def async_url(self):
        return self.settings.async_database_url

class AIServiceConfig:
    """Legacy AI service config class"""
    def __init__(self):
        self.settings = get_settings()
    
    @property
    def openai_api_key(self):
        return self.settings.openai_api_key
    
    @property
    def anthropic_api_key(self):
        return self.settings.anthropic_api_key

class SecurityConfig:
    """Legacy security config class"""
    def __init__(self):
        self.settings = get_settings()
    
    @property
    def jwt_secret_key(self):
        return self.settings.jwt_secret_key

class SystemConfig:
    """Legacy system config class"""
    def __init__(self):
        self.settings = get_settings()

def get_config():
    """Get legacy config object"""
    return ScrollIntelConfig()

def load_config_from_file(file_path: str):
    """Load config from file - legacy function"""
    return ScrollIntelConfig()