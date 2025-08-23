"""
Configuration Management for ScrollIntel
Handles loading and validation of configuration settings
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Import the new configuration manager
from .configuration_manager import get_config as get_new_config, ScrollIntelConfig

logger = logging.getLogger(__name__)


def get_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file or environment (legacy compatibility)"""
    try:
        # Use the new configuration manager
        new_config = get_new_config()
        
        # Convert to legacy format for backward compatibility
        return _convert_to_legacy_format(new_config)
        
    except Exception as e:
        logger.warning(f"New configuration manager failed, falling back to legacy: {e}")
        return _get_legacy_config(config_file)


def _convert_to_legacy_format(config: ScrollIntelConfig) -> Dict[str, Any]:
    """Convert new configuration format to legacy format."""
    return {
        "environment": config.environment,
        "debug": config.debug,
        "is_production": config.environment == "production",
        "database_url": config.database.primary_url,
        "db_pool_size": config.database.pool_size,
        "db_max_overflow": config.database.max_overflow,
        "redis_host": "localhost",  # Default for legacy compatibility
        "redis_port": 6379,
        "redis_password": "",
        "redis_db": 0,
        "skip_redis": True,  # Skip Redis for now to avoid connection issues
        "jwt_secret": config.session.secret_key,
        "session_timeout_minutes": config.session.timeout_minutes,
        
        "infrastructure": {
            "redis_host": "localhost",
            "redis_port": 6379,
            "database_url": config.database.primary_url,
            "scaling": {
                "min_instances": int(os.getenv("MIN_INSTANCES", "2")),
                "max_instances": int(os.getenv("MAX_INSTANCES", "10")),
                "target_cpu": float(os.getenv("TARGET_CPU", "70.0")),
                "target_memory": float(os.getenv("TARGET_MEMORY", "80.0"))
            }
        },
        
        "onboarding": {
            "jwt_secret": config.session.secret_key,
            "jwt_algorithm": config.session.algorithm,
            "email": {
                "smtp_server": os.getenv("SMTP_SERVER", "localhost"),
                "smtp_port": int(os.getenv("SMTP_PORT", "587")),
                "username": os.getenv("EMAIL_USERNAME", ""),
                "password": os.getenv("EMAIL_PASSWORD", ""),
                "from_email": os.getenv("FROM_EMAIL", "noreply@scrollintel.com"),
                "base_url": os.getenv("BASE_URL", os.getenv("API_URL", "http://localhost:3000"))
            }
        },
        
        "api_stability": {
            "redis_host": "localhost",
            "redis_port": 6379,
            "rate_limiting": {
                "default_limits": {
                    "requests_per_second": int(os.getenv("RATE_LIMIT_PER_SECOND", "10")),
                    "requests_per_minute": int(os.getenv("RATE_LIMIT_PER_MINUTE", "100")),
                    "requests_per_hour": int(os.getenv("RATE_LIMIT_PER_HOUR", "1000")),
                    "requests_per_day": int(os.getenv("RATE_LIMIT_PER_DAY", "10000"))
                },
                "endpoint_limits": {}
            }
        }
    }


def _get_legacy_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Legacy configuration loading (fallback)"""
    if config_file is None:
        # Try different config file locations
        config_paths = [
            "config/production.json",
            "/opt/scrollintel/config/production.json",
            "config/development.json"
        ]
        
        for path in config_paths:
            if os.path.exists(path):
                config_file = path
                break
    
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # Override with environment variables
    config = override_with_env(config)
    
    return config


def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return {
        "environment": os.getenv("ENVIRONMENT", "development"),
        "debug": os.getenv("DEBUG", "false").lower() == "true",
        "database_url": os.getenv("DATABASE_URL", os.getenv("DATABASE_URL", os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/scrollintel"))),
        "db_pool_size": int(os.getenv("DB_POOL_SIZE", "5")),
        "db_max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "10")),
        "redis_host": os.getenv("REDIS_HOST", "localhost"),
        "redis_port": int(os.getenv("REDIS_PORT", "6379")),
        "redis_password": os.getenv("REDIS_PASSWORD", ""),
        "redis_db": int(os.getenv("REDIS_DB", "0")),
        "skip_redis": os.getenv("SKIP_REDIS", "true").lower() == "true",
        "jwt_secret": os.getenv("JWT_SECRET", os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")),
        "session_timeout_minutes": int(os.getenv("SESSION_TIMEOUT_MINUTES", "60")),
        
        "infrastructure": {
            "redis_host": os.getenv("REDIS_HOST", "localhost"),
            "redis_port": int(os.getenv("REDIS_PORT", "6379")),
            "database_url": os.getenv("DATABASE_URL", os.getenv("DATABASE_URL", os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/scrollintel"))),
            "scaling": {
                "min_instances": int(os.getenv("MIN_INSTANCES", "2")),
                "max_instances": int(os.getenv("MAX_INSTANCES", "10")),
                "target_cpu": float(os.getenv("TARGET_CPU", "70.0")),
                "target_memory": float(os.getenv("TARGET_MEMORY", "80.0"))
            }
        },
        
        "onboarding": {
            "jwt_secret": os.getenv("JWT_SECRET", os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")),
            "jwt_algorithm": "HS256",
            "email": {
                "smtp_server": os.getenv("SMTP_SERVER", "localhost"),
                "smtp_port": int(os.getenv("SMTP_PORT", "587")),
                "username": os.getenv("EMAIL_USERNAME", ""),
                "password": os.getenv("EMAIL_PASSWORD", ""),
                "from_email": os.getenv("FROM_EMAIL", "noreply@scrollintel.com"),
                "base_url": os.getenv("BASE_URL", os.getenv("API_URL", "http://localhost:3000"))
            }
        },
        
        "api_stability": {
            "redis_host": os.getenv("REDIS_HOST", "localhost"),
            "redis_port": int(os.getenv("REDIS_PORT", "6379")),
            "rate_limiting": {
                "default_limits": {
                    "requests_per_second": int(os.getenv("RATE_LIMIT_PER_SECOND", "10")),
                    "requests_per_minute": int(os.getenv("RATE_LIMIT_PER_MINUTE", "100")),
                    "requests_per_hour": int(os.getenv("RATE_LIMIT_PER_HOUR", "1000")),
                    "requests_per_day": int(os.getenv("RATE_LIMIT_PER_DAY", "10000"))
                },
                "endpoint_limits": {}
            }
        }
    }


def override_with_env(config: Dict[str, Any]) -> Dict[str, Any]:
    """Override configuration with environment variables"""
    # This function would recursively override config values
    # with environment variables following a naming convention
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration"""
    required_keys = [
        "infrastructure",
        "onboarding", 
        "api_stability"
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    return True


# Compatibility functions for legacy code
def get_settings(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Get settings (alias for get_config for backward compatibility)."""
    return get_config(config_file)


def reload_settings(environment: Optional[str] = None) -> Dict[str, Any]:
    """Reload settings (for backward compatibility)."""
    if environment:
        os.environ["ENVIRONMENT"] = environment
    
    # Clear any cached configuration
    from .configuration_manager import reload_configuration
    try:
        new_config = reload_configuration()
        return _convert_to_legacy_format(new_config)
    except Exception as e:
        logger.warning(f"New configuration manager failed during reload, falling back to legacy: {e}")
        return _get_legacy_config()

# Fallback configuration for testing
class FallbackSettings:
    """Fallback settings when main config fails"""
    
    def __init__(self):
        import os
        self.database_url = os.getenv('DATABASE_URL', 'sqlite:///./scrollintel.db')
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.jwt_secret_key = os.getenv('JWT_SECRET_KEY', 'fallback-secret-key')
        self.debug = os.getenv('DEBUG', 'false').lower() == 'true'
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self.db_pool_size = int(os.getenv('DB_POOL_SIZE', '5'))
        self.db_max_overflow = int(os.getenv('DB_MAX_OVERFLOW', '10'))
        self.redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = int(os.getenv('REDIS_PORT', '6379'))
        self.redis_password = os.getenv('REDIS_PASSWORD', '')
        self.redis_db = int(os.getenv('REDIS_DB', '0'))
        self.skip_redis = os.getenv('SKIP_REDIS', 'false').lower() == 'true'
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return getattr(self, key, default)

def get_fallback_settings():
    """Get fallback settings for testing"""
    return FallbackSettings()
