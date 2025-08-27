"""
Production Configuration for Visual Generation System
Manages environment variables, API keys, and production settings
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class ModelConfig:
    enabled: bool
    api_key: Optional[str]
    rate_limit_per_minute: int
    timeout_seconds: int
    retry_attempts: int
    warm_up_enabled: bool

@dataclass
class StorageConfig:
    provider: str
    bucket_name: str
    region: str
    access_key: Optional[str]
    secret_key: Optional[str]
    cdn_domain: Optional[str]
    encryption_enabled: bool

@dataclass
class MonitoringConfig:
    enabled: bool
    log_level: str
    metrics_endpoint: Optional[str]
    alert_webhook: Optional[str]
    performance_tracking: bool

class ProductionConfig:
    """Manages production configuration for visual generation system"""
    
    def __init__(self):
        self.environment = Environment(os.getenv("ENVIRONMENT", "development"))
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # Load all configurations
        self.model_configs = self._load_model_configs()
        self.storage_config = self._load_storage_config()
        self.monitoring_config = self._load_monitoring_config()
        self.security_config = self._load_security_config()
        
        # Validate critical configurations
        self._validate_production_config()
    
    def _load_model_configs(self) -> Dict[str, ModelConfig]:
        """Load model-specific configurations"""
        return {
            "stable_diffusion_xl": ModelConfig(
                enabled=os.getenv("STABLE_DIFFUSION_ENABLED", "true").lower() == "true",
                api_key=None,  # Local model, no API key needed
                rate_limit_per_minute=int(os.getenv("STABLE_DIFFUSION_RATE_LIMIT", "60")),
                timeout_seconds=int(os.getenv("STABLE_DIFFUSION_TIMEOUT", "120")),
                retry_attempts=int(os.getenv("STABLE_DIFFUSION_RETRIES", "3")),
                warm_up_enabled=os.getenv("STABLE_DIFFUSION_WARMUP", "true").lower() == "true"
            ),
            "dalle3": ModelConfig(
                enabled=os.getenv("DALLE3_ENABLED", "true").lower() == "true",
                api_key=os.getenv("OPENAI_API_KEY"),
                rate_limit_per_minute=int(os.getenv("DALLE3_RATE_LIMIT", "50")),
                timeout_seconds=int(os.getenv("DALLE3_TIMEOUT", "60")),
                retry_attempts=int(os.getenv("DALLE3_RETRIES", "3")),
                warm_up_enabled=False  # API-based, no warm-up needed
            ),
            "midjourney": ModelConfig(
                enabled=os.getenv("MIDJOURNEY_ENABLED", "true").lower() == "true",
                api_key=os.getenv("MIDJOURNEY_BOT_TOKEN"),
                rate_limit_per_minute=int(os.getenv("MIDJOURNEY_RATE_LIMIT", "20")),
                timeout_seconds=int(os.getenv("MIDJOURNEY_TIMEOUT", "300")),
                retry_attempts=int(os.getenv("MIDJOURNEY_RETRIES", "2")),
                warm_up_enabled=False  # Discord bot, no warm-up needed
            )
        }
    
    def _load_storage_config(self) -> StorageConfig:
        """Load storage configuration"""
        return StorageConfig(
            provider=os.getenv("STORAGE_PROVIDER", "aws_s3"),
            bucket_name=os.getenv("STORAGE_BUCKET", "scrollintel-visual-generation"),
            region=os.getenv("STORAGE_REGION", "us-east-1"),
            access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            cdn_domain=os.getenv("CDN_DOMAIN"),
            encryption_enabled=os.getenv("STORAGE_ENCRYPTION", "true").lower() == "true"
        )
    
    def _load_monitoring_config(self) -> MonitoringConfig:
        """Load monitoring configuration"""
        return MonitoringConfig(
            enabled=os.getenv("MONITORING_ENABLED", "true").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            metrics_endpoint=os.getenv("METRICS_ENDPOINT"),
            alert_webhook=os.getenv("ALERT_WEBHOOK"),
            performance_tracking=os.getenv("PERFORMANCE_TRACKING", "true").lower() == "true"
        )
    
    def _load_security_config(self) -> Dict[str, Any]:
        """Load security configuration"""
        return {
            "content_safety_enabled": os.getenv("CONTENT_SAFETY_ENABLED", "true").lower() == "true",
            "rate_limiting_enabled": os.getenv("RATE_LIMITING_ENABLED", "true").lower() == "true",
            "api_key_required": os.getenv("API_KEY_REQUIRED", "true").lower() == "true",
            "audit_logging_enabled": os.getenv("AUDIT_LOGGING_ENABLED", "true").lower() == "true",
            "encryption_key": os.getenv("ENCRYPTION_KEY"),
            "jwt_secret": os.getenv("JWT_SECRET"),
            "cors_origins": os.getenv("CORS_ORIGINS", "*").split(","),
            "max_file_size_mb": int(os.getenv("MAX_FILE_SIZE_MB", "100")),
            "allowed_file_types": os.getenv("ALLOWED_FILE_TYPES", "jpg,jpeg,png,gif,mp4,mov").split(",")
        }
    
    def _validate_production_config(self):
        """Validate critical production configurations"""
        if self.environment == Environment.PRODUCTION:
            # Validate required API keys
            if self.model_configs["dalle3"].enabled and not self.model_configs["dalle3"].api_key:
                logger.error("DALL-E 3 enabled but OpenAI API key not configured")
                raise ValueError("Missing OpenAI API key for production")
            
            if self.model_configs["midjourney"].enabled and not self.model_configs["midjourney"].api_key:
                logger.error("Midjourney enabled but bot token not configured")
                raise ValueError("Missing Midjourney bot token for production")
            
            # Validate storage configuration
            if not self.storage_config.access_key or not self.storage_config.secret_key:
                logger.error("Storage credentials not configured for production")
                raise ValueError("Missing storage credentials for production")
            
            # Validate security configuration
            if not self.security_config["encryption_key"]:
                logger.error("Encryption key not configured for production")
                raise ValueError("Missing encryption key for production")
            
            if not self.security_config["jwt_secret"]:
                logger.error("JWT secret not configured for production")
                raise ValueError("Missing JWT secret for production")
            
            logger.info("Production configuration validation passed")
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for specific model"""
        return self.model_configs.get(model_name)
    
    def is_model_enabled(self, model_name: str) -> bool:
        """Check if model is enabled"""
        config = self.get_model_config(model_name)
        return config.enabled if config else False
    
    def get_storage_config(self) -> StorageConfig:
        """Get storage configuration"""
        return self.storage_config
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration"""
        return self.monitoring_config
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return self.security_config
    
    def get_gpu_config(self) -> Dict[str, Any]:
        """Get GPU configuration"""
        return {
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", "0"),
            "gpu_memory_fraction": float(os.getenv("GPU_MEMORY_FRACTION", "0.9")),
            "mixed_precision": os.getenv("MIXED_PRECISION", "true").lower() == "true",
            "model_parallel": os.getenv("MODEL_PARALLEL", "false").lower() == "true",
            "batch_size": int(os.getenv("BATCH_SIZE", "1")),
            "max_concurrent_requests": int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return {
            "cache_enabled": os.getenv("CACHE_ENABLED", "true").lower() == "true",
            "cache_ttl_seconds": int(os.getenv("CACHE_TTL_SECONDS", "3600")),
            "preload_models": os.getenv("PRELOAD_MODELS", "true").lower() == "true",
            "async_processing": os.getenv("ASYNC_PROCESSING", "true").lower() == "true",
            "queue_max_size": int(os.getenv("QUEUE_MAX_SIZE", "100")),
            "worker_threads": int(os.getenv("WORKER_THREADS", "4")),
            "auto_scaling_enabled": os.getenv("AUTO_SCALING_ENABLED", "true").lower() == "true",
            "scale_up_threshold": float(os.getenv("SCALE_UP_THRESHOLD", "80.0")),
            "scale_down_threshold": float(os.getenv("SCALE_DOWN_THRESHOLD", "20.0"))
        }
    
    def get_quality_config(self) -> Dict[str, Any]:
        """Get quality control configuration"""
        return {
            "quality_checks_enabled": os.getenv("QUALITY_CHECKS_ENABLED", "true").lower() == "true",
            "min_quality_score": float(os.getenv("MIN_QUALITY_SCORE", "0.7")),
            "safety_checks_enabled": os.getenv("SAFETY_CHECKS_ENABLED", "true").lower() == "true",
            "nsfw_detection_enabled": os.getenv("NSFW_DETECTION_ENABLED", "true").lower() == "true",
            "copyright_check_enabled": os.getenv("COPYRIGHT_CHECK_ENABLED", "true").lower() == "true",
            "content_moderation_strict": os.getenv("CONTENT_MODERATION_STRICT", "true").lower() == "true"
        }
    
    def get_billing_config(self) -> Dict[str, Any]:
        """Get billing configuration"""
        return {
            "billing_enabled": os.getenv("BILLING_ENABLED", "true").lower() == "true",
            "cost_tracking_enabled": os.getenv("COST_TRACKING_ENABLED", "true").lower() == "true",
            "usage_limits_enabled": os.getenv("USAGE_LIMITS_ENABLED", "true").lower() == "true",
            "default_monthly_limit": float(os.getenv("DEFAULT_MONTHLY_LIMIT", "100.0")),
            "cost_per_image": float(os.getenv("COST_PER_IMAGE", "0.02")),
            "cost_per_video_second": float(os.getenv("COST_PER_VIDEO_SECOND", "0.10")),
            "premium_tier_multiplier": float(os.getenv("PREMIUM_TIER_MULTIPLIER", "2.0"))
        }
    
    def export_config(self) -> Dict[str, Any]:
        """Export complete configuration (excluding sensitive data)"""
        return {
            "environment": self.environment.value,
            "debug": self.debug,
            "models": {
                name: {
                    "enabled": config.enabled,
                    "rate_limit_per_minute": config.rate_limit_per_minute,
                    "timeout_seconds": config.timeout_seconds,
                    "retry_attempts": config.retry_attempts,
                    "warm_up_enabled": config.warm_up_enabled
                }
                for name, config in self.model_configs.items()
            },
            "storage": {
                "provider": self.storage_config.provider,
                "bucket_name": self.storage_config.bucket_name,
                "region": self.storage_config.region,
                "encryption_enabled": self.storage_config.encryption_enabled
            },
            "monitoring": {
                "enabled": self.monitoring_config.enabled,
                "log_level": self.monitoring_config.log_level,
                "performance_tracking": self.monitoring_config.performance_tracking
            },
            "gpu": self.get_gpu_config(),
            "performance": self.get_performance_config(),
            "quality": self.get_quality_config(),
            "billing": self.get_billing_config()
        }

# Global configuration instance
production_config = ProductionConfig()

def get_config() -> ProductionConfig:
    """Get production configuration instance"""
    return production_config

def validate_environment():
    """Validate environment configuration"""
    try:
        production_config._validate_production_config()
        return True
    except Exception as e:
        logger.error(f"Environment validation failed: {str(e)}")
        return False