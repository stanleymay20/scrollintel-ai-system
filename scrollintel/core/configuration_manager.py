"""
Configuration Management System for ScrollIntel
Handles loading, validation, and fallback mechanisms for configuration settings
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

# Load .env file at module import
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, continue without it

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


class DatabaseType(Enum):
    """Supported database types."""
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_keys: List[str] = field(default_factory=list)


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    primary_url: str
    fallback_url: str
    pool_size: int = 5
    max_overflow: int = 10
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class SessionConfig:
    """Session configuration settings."""
    timeout_minutes: int = 60
    secret_key: str = ""
    algorithm: str = "HS256"
    secure_cookies: bool = True


@dataclass
class ServiceConfig:
    """Service configuration settings."""
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    frontend_port: int = 3000
    startup_timeout: int = 60
    health_check_interval: int = 30


@dataclass
class ScrollIntelConfig:
    """Main ScrollIntel configuration."""
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    database: DatabaseConfig = None
    session: SessionConfig = None
    services: ServiceConfig = None
    
    def __post_init__(self):
        """Initialize nested configurations if not provided."""
        if self.database is None:
            self.database = DatabaseConfig(
                primary_url="postgresql://postgres:password@localhost:5432/scrollintel",
                fallback_url="sqlite:///./data/scrollintel.db"
            )
        if self.session is None:
            self.session = SessionConfig()
        if self.services is None:
            self.services = ServiceConfig()


class ConfigurationManager:
    """Manages configuration loading, validation, and fallback mechanisms."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_file = config_file
        self.config: Optional[ScrollIntelConfig] = None
        self._validation_result: Optional[ValidationResult] = None
    
    def load_config(self) -> ScrollIntelConfig:
        """Load configuration from file and environment variables."""
        try:
            # Start with default configuration
            config_dict = self._get_default_config()
            
            # Load from file if specified
            if self.config_file:
                file_config = self._load_from_file(self.config_file)
                config_dict.update(file_config)
            else:
                # Try to find config file automatically
                file_config = self._auto_load_config_file()
                if file_config:
                    config_dict.update(file_config)
            
            # Override with environment variables
            config_dict = self._override_with_env(config_dict)
            
            # Create configuration object
            self.config = self._create_config_object(config_dict)
            
            # Validate configuration
            self._validation_result = self.validate_config(self.config)
            
            # Apply fallbacks for invalid/missing values
            if not self._validation_result.is_valid:
                self.config = self._apply_fallbacks(self.config, self._validation_result)
                # Re-validate after applying fallbacks
                self._validation_result = self.validate_config(self.config)
            
            # Log warnings
            for warning in self._validation_result.warnings:
                logger.warning(f"Configuration warning: {warning}")
            
            # Raise error if still invalid after fallbacks
            if not self._validation_result.is_valid:
                error_msg = "Configuration validation failed:\n" + "\n".join(self._validation_result.errors)
                raise ConfigurationError(error_msg)
            
            logger.info("Configuration loaded successfully")
            return self.config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ConfigurationError(f"Configuration loading failed: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            "environment": "development",
            "debug": False,
            "log_level": "INFO",
            "database": {
                "primary_url": "postgresql://postgres:password@localhost:5432/scrollintel",
                "fallback_url": "sqlite:///./data/scrollintel.db",
                "pool_size": 5,
                "max_overflow": 10,
                "timeout": 30,
                "retry_attempts": 3,
                "retry_delay": 1.0
            },
            "session": {
                "timeout_minutes": 60,
                "secret_key": "",
                "algorithm": "HS256",
                "secure_cookies": True
            },
            "services": {
                "api_host": "0.0.0.0",
                "api_port": 8000,
                "frontend_port": 3000,
                "startup_timeout": 60,
                "health_check_interval": 30
            }
        }
    
    def _load_from_file(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from a JSON file."""
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                logger.warning(f"Configuration file not found: {config_file}")
                return {}
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            logger.info(f"Configuration loaded from file: {config_file}")
            return config
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file {config_file}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load configuration file {config_file}: {e}")
            return {}
    
    def _auto_load_config_file(self) -> Dict[str, Any]:
        """Automatically find and load configuration file."""
        config_paths = [
            "config/production.json",
            "config/staging.json", 
            "config/development.json",
            "/opt/scrollintel/config/production.json",
            "./scrollintel.config.json",
            "./config.json"
        ]
        
        for config_path in config_paths:
            if os.path.exists(config_path):
                logger.info(f"Auto-detected configuration file: {config_path}")
                return self._load_from_file(config_path)
        
        logger.info("No configuration file found, using defaults and environment variables")
        return {}
    
    def _override_with_env(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Override configuration with environment variables."""
        # Environment variable mappings
        env_mappings = {
            "ENVIRONMENT": "environment",
            "DEBUG": ("debug", bool),
            "LOG_LEVEL": "log_level",
            
            # Database settings
            "DATABASE_URL": "database.primary_url",
            "POSTGRES_URL": "database.primary_url", 
            "SQLITE_URL": "database.fallback_url",
            "DB_POOL_SIZE": ("database.pool_size", int),
            "DB_MAX_OVERFLOW": ("database.max_overflow", int),
            "DB_TIMEOUT": ("database.timeout", int),
            
            # Session settings
            "SESSION_TIMEOUT_MINUTES": ("session.timeout_minutes", int),
            "SESSION_SECRET_KEY": "session.secret_key",
            "JWT_SECRET_KEY": "session.secret_key",  # Alternative name
            "SESSION_ALGORITHM": "session.algorithm",
            "SECURE_COOKIES": ("session.secure_cookies", bool),
            
            # Service settings
            "API_HOST": "services.api_host",
            "API_PORT": ("services.api_port", int),
            "FRONTEND_PORT": ("services.frontend_port", int),
            "STARTUP_TIMEOUT": ("services.startup_timeout", int),
            "HEALTH_CHECK_INTERVAL": ("services.health_check_interval", int),
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Handle type conversion
                if isinstance(config_path, tuple):
                    path, value_type = config_path
                    try:
                        if value_type == bool:
                            env_value = env_value.lower() in ('true', '1', 'yes', 'on')
                        elif value_type == int:
                            env_value = int(env_value)
                        elif value_type == float:
                            env_value = float(env_value)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid value for {env_var}: {env_value}, error: {e}")
                        continue
                    config_path = path
                
                # Set nested configuration value
                self._set_nested_value(config, config_path, env_value)
        
        return config
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any) -> None:
        """Set a nested configuration value using dot notation."""
        keys = path.split('.')
        current = config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value
    
    def _create_config_object(self, config_dict: Dict[str, Any]) -> ScrollIntelConfig:
        """Create configuration object from dictionary."""
        try:
            # Create database config
            db_config = DatabaseConfig(
                primary_url=config_dict["database"]["primary_url"],
                fallback_url=config_dict["database"]["fallback_url"],
                pool_size=config_dict["database"]["pool_size"],
                max_overflow=config_dict["database"]["max_overflow"],
                timeout=config_dict["database"]["timeout"],
                retry_attempts=config_dict["database"]["retry_attempts"],
                retry_delay=config_dict["database"]["retry_delay"]
            )
            
            # Create session config
            session_config = SessionConfig(
                timeout_minutes=config_dict["session"]["timeout_minutes"],
                secret_key=config_dict["session"]["secret_key"],
                algorithm=config_dict["session"]["algorithm"],
                secure_cookies=config_dict["session"]["secure_cookies"]
            )
            
            # Create service config
            service_config = ServiceConfig(
                api_host=config_dict["services"]["api_host"],
                api_port=config_dict["services"]["api_port"],
                frontend_port=config_dict["services"]["frontend_port"],
                startup_timeout=config_dict["services"]["startup_timeout"],
                health_check_interval=config_dict["services"]["health_check_interval"]
            )
            
            # Create main config
            return ScrollIntelConfig(
                environment=config_dict["environment"],
                debug=config_dict["debug"],
                log_level=config_dict["log_level"],
                database=db_config,
                session=session_config,
                services=service_config
            )
            
        except KeyError as e:
            raise ConfigurationError(f"Missing required configuration key: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to create configuration object: {e}")
    
    def validate_config(self, config: ScrollIntelConfig) -> ValidationResult:
        """Validate configuration and return validation result."""
        result = ValidationResult(is_valid=True)
        
        # Validate environment
        valid_environments = ["development", "staging", "production", "test"]
        if config.environment not in valid_environments:
            result.errors.append(f"Invalid environment '{config.environment}'. Must be one of: {valid_environments}")
            result.is_valid = False
        
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if config.log_level not in valid_log_levels:
            result.errors.append(f"Invalid log level '{config.log_level}'. Must be one of: {valid_log_levels}")
            result.is_valid = False
        
        # Validate database configuration
        if not config.database.primary_url:
            result.errors.append("Primary database URL is required")
            result.is_valid = False
        
        if not config.database.fallback_url:
            result.errors.append("Fallback database URL is required")
            result.is_valid = False
        
        # Validate database URLs format
        if config.database.primary_url and not self._is_valid_database_url(config.database.primary_url):
            result.errors.append(f"Invalid primary database URL format: {config.database.primary_url}")
            result.is_valid = False
        
        if config.database.fallback_url and not self._is_valid_database_url(config.database.fallback_url):
            result.errors.append(f"Invalid fallback database URL format: {config.database.fallback_url}")
            result.is_valid = False
        
        # Validate session configuration
        if not config.session.secret_key:
            if config.environment == "production":
                result.errors.append("Session secret key is required in production")
                result.is_valid = False
            else:
                result.warnings.append("Session secret key not set, using default (not secure)")
                result.missing_keys.append("session.secret_key")
        
        if config.session.timeout_minutes <= 0:
            result.errors.append("Session timeout must be greater than 0")
            result.is_valid = False
        
        # Validate service configuration
        if config.services.api_port <= 0 or config.services.api_port > 65535:
            result.errors.append(f"Invalid API port: {config.services.api_port}")
            result.is_valid = False
        
        if config.services.frontend_port <= 0 or config.services.frontend_port > 65535:
            result.errors.append(f"Invalid frontend port: {config.services.frontend_port}")
            result.is_valid = False
        
        if config.services.api_port == config.services.frontend_port:
            result.errors.append("API and frontend ports cannot be the same")
            result.is_valid = False
        
        return result
    
    def _is_valid_database_url(self, url: str) -> bool:
        """Check if database URL format is valid."""
        valid_prefixes = [
            "postgresql://", "postgresql+asyncpg://", "postgresql+psycopg2://",
            "sqlite://", "sqlite+aiosqlite://", "sqlite:///"
        ]
        return any(url.startswith(prefix) for prefix in valid_prefixes)
    
    def _apply_fallbacks(self, config: ScrollIntelConfig, validation_result: ValidationResult) -> ScrollIntelConfig:
        """Apply fallback values for invalid or missing configuration."""
        logger.warning("Applying configuration fallbacks...")
        
        # Apply session secret key fallback
        if "session.secret_key" in validation_result.missing_keys:
            import secrets
            config.session.secret_key = secrets.token_urlsafe(32)
            logger.warning("Generated temporary session secret key (not suitable for production)")
        
        # Apply other fallbacks as needed
        # This can be extended based on specific requirements
        
        return config
    
    def get_missing_keys(self) -> List[str]:
        """Get list of missing configuration keys."""
        if self._validation_result:
            return self._validation_result.missing_keys
        return []
    
    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors."""
        if self._validation_result:
            return self._validation_result.errors
        return []
    
    def get_validation_warnings(self) -> List[str]:
        """Get list of validation warnings."""
        if self._validation_result:
            return self._validation_result.warnings
        return []
    
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        if self._validation_result:
            return self._validation_result.is_valid
        return False
    
    def get_database_type(self, url: str) -> DatabaseType:
        """Determine database type from URL."""
        if url.startswith(("postgresql://", "postgresql+asyncpg://", "postgresql+psycopg2://")):
            return DatabaseType.POSTGRESQL
        elif url.startswith(("sqlite://", "sqlite+aiosqlite://", "sqlite:///")):
            return DatabaseType.SQLITE
        else:
            raise ConfigurationError(f"Unsupported database type in URL: {url}")
    
    def export_config(self, file_path: str) -> None:
        """Export current configuration to a file."""
        if not self.config:
            raise ConfigurationError("No configuration loaded")
        
        try:
            config_dict = {
                "environment": self.config.environment,
                "debug": self.config.debug,
                "log_level": self.config.log_level,
                "database": {
                    "primary_url": self.config.database.primary_url,
                    "fallback_url": self.config.database.fallback_url,
                    "pool_size": self.config.database.pool_size,
                    "max_overflow": self.config.database.max_overflow,
                    "timeout": self.config.database.timeout,
                    "retry_attempts": self.config.database.retry_attempts,
                    "retry_delay": self.config.database.retry_delay
                },
                "session": {
                    "timeout_minutes": self.config.session.timeout_minutes,
                    "secret_key": "***REDACTED***",  # Don't export secret
                    "algorithm": self.config.session.algorithm,
                    "secure_cookies": self.config.session.secure_cookies
                },
                "services": {
                    "api_host": self.config.services.api_host,
                    "api_port": self.config.services.api_port,
                    "frontend_port": self.config.services.frontend_port,
                    "startup_timeout": self.config.services.startup_timeout,
                    "health_check_interval": self.config.services.health_check_interval
                }
            }
            
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration exported to: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            raise ConfigurationError(f"Configuration export failed: {e}")


# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None
_config: Optional[ScrollIntelConfig] = None


def get_configuration_manager(config_file: Optional[str] = None) -> ConfigurationManager:
    """Get the global configuration manager."""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigurationManager(config_file)
    
    return _config_manager


def load_configuration(config_file: Optional[str] = None) -> ScrollIntelConfig:
    """Load and return the global configuration."""
    global _config
    
    if _config is None:
        manager = get_configuration_manager(config_file)
        _config = manager.load_config()
    
    return _config


def get_config() -> ScrollIntelConfig:
    """Get the current configuration (load if not already loaded)."""
    global _config
    
    if _config is None:
        _config = load_configuration()
    
    return _config


def reload_configuration(config_file: Optional[str] = None) -> ScrollIntelConfig:
    """Reload configuration from file."""
    global _config_manager, _config
    
    _config_manager = None
    _config = None
    
    return load_configuration(config_file)