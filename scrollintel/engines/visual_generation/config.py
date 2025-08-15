"""
Configuration management for visual generation engines.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import yaml
from .exceptions import ConfigurationError


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    type: str  # 'image', 'video', 'enhancement'
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    model_path: Optional[str] = None
    max_resolution: tuple = (1024, 1024)
    max_duration: float = 30.0
    batch_size: int = 1
    timeout: float = 300.0
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InfrastructureConfig:
    """Configuration for infrastructure and resources."""
    gpu_enabled: bool = True
    max_concurrent_requests: int = 10
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds
    storage_path: str = "./generated_content"
    temp_path: str = "./temp"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    cleanup_interval: int = 3600  # seconds
    # Redis configuration
    redis_url: Optional[str] = "redis://localhost:6379/0"
    redis_enabled: bool = True
    # Semantic similarity configuration
    semantic_similarity_enabled: bool = True
    similarity_threshold: float = 0.85
    semantic_model_name: str = "all-MiniLM-L6-v2"


@dataclass
class SafetyConfig:
    """Configuration for content safety and filtering."""
    enabled: bool = True
    nsfw_detection: bool = True
    violence_detection: bool = True
    copyright_check: bool = True
    prompt_filtering: bool = True
    confidence_threshold: float = 0.8
    blocked_keywords: list = field(default_factory=list)


@dataclass
class QualityConfig:
    """Configuration for quality assessment."""
    enabled: bool = True
    min_quality_score: float = 0.7
    technical_quality_weight: float = 0.3
    aesthetic_weight: float = 0.3
    prompt_adherence_weight: float = 0.4
    auto_enhance: bool = False
    quality_metrics: list = field(default_factory=lambda: [
        "sharpness", "color_balance", "composition"
    ])


@dataclass
class CostConfig:
    """Configuration for cost management."""
    enabled: bool = True
    cost_per_image: float = 0.01
    cost_per_video_second: float = 0.1
    cost_per_enhancement: float = 0.005
    budget_alerts: bool = True
    daily_budget_limit: float = 100.0
    user_budget_limits: Dict[str, float] = field(default_factory=dict)


class VisualGenerationConfig:
    """Main configuration class for visual generation system."""
    
    def __init__(self, config_path_or_dict: Optional[str | Dict[str, Any]] = None):
        if isinstance(config_path_or_dict, dict):
            # Direct configuration dictionary provided
            self.config_path = None
            self.models: Dict[str, ModelConfig] = {}
            self.infrastructure = InfrastructureConfig()
            self.safety = SafetyConfig()
            self.quality = QualityConfig()
            self.cost = CostConfig()
            self._parse_config(config_path_or_dict)
        else:
            # File path provided or use default
            self.config_path = config_path_or_dict or self._get_default_config_path()
            self.models: Dict[str, ModelConfig] = {}
            self.infrastructure = InfrastructureConfig()
            self.safety = SafetyConfig()
            self.quality = QualityConfig()
            self.cost = CostConfig()
            self._load_config()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key (for backward compatibility)."""
        # Handle direct key access for simple configurations
        if hasattr(self, key):
            return getattr(self, key)
        
        # Check in model parameters
        for model_config in self.models.values():
            if key in model_config.parameters:
                return model_config.parameters[key]
        
        return default
    
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        return os.path.join(os.path.dirname(__file__), "config.yaml")
    
    def _load_config(self):
        """Load configuration from file."""
        if not os.path.exists(self.config_path):
            self._create_default_config()
            return
        
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            self._parse_config(config_data)
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load config from {self.config_path}: {str(e)}")
    
    def _parse_config(self, config_data: Dict[str, Any]):
        """Parse configuration data into structured objects."""
        # Load model configurations
        if 'models' in config_data:
            for model_name, model_data in config_data['models'].items():
                self.models[model_name] = ModelConfig(
                    name=model_name,
                    **model_data
                )
        
        # Load infrastructure config
        if 'infrastructure' in config_data:
            infra_data = config_data['infrastructure']
            self.infrastructure = InfrastructureConfig(**infra_data)
        
        # Load safety config
        if 'safety' in config_data:
            safety_data = config_data['safety']
            self.safety = SafetyConfig(**safety_data)
        
        # Load quality config
        if 'quality' in config_data:
            quality_data = config_data['quality']
            self.quality = QualityConfig(**quality_data)
        
        # Load cost config
        if 'cost' in config_data:
            cost_data = config_data['cost']
            self.cost = CostConfig(**cost_data)
    
    def _create_default_config(self):
        """Create a default configuration file."""
        default_config = {
            'models': {
                'stable_diffusion_xl': {
                    'type': 'image',
                    'api_key': '${STABILITY_API_KEY}',
                    'api_url': 'https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image',
                    'max_resolution': [1024, 1024],
                    'batch_size': 1,
                    'timeout': 300.0,
                    'enabled': True,
                    'parameters': {
                        'cfg_scale': 7.5,
                        'steps': 50,
                        'sampler': 'K_DPM_2_ANCESTRAL'
                    }
                },
                'dalle3': {
                    'type': 'image',
                    'api_key': '${OPENAI_API_KEY}',
                    'api_url': 'https://api.openai.com/v1/images/generations',
                    'max_resolution': [1024, 1024],
                    'batch_size': 1,
                    'timeout': 300.0,
                    'enabled': True,
                    'parameters': {
                        'quality': 'hd',
                        'style': 'vivid'
                    }
                },
                'midjourney': {
                    'type': 'image',
                    'api_key': '${MIDJOURNEY_API_KEY}',
                    'api_url': 'https://api.midjourney.com/v1/imagine',
                    'max_resolution': [1024, 1024],
                    'batch_size': 4,
                    'timeout': 600.0,
                    'enabled': False,
                    'parameters': {
                        'version': '6',
                        'quality': '1'
                    }
                },
                'proprietary_neural_renderer': {
                    'type': 'video',
                    'model_path': './models/proprietary_neural_renderer',
                    'max_resolution': [3840, 2160],  # 4K
                    'max_duration': 600.0,  # 10 minutes
                    'batch_size': 1,
                    'timeout': 1800.0,  # 30 minutes
                    'enabled': True,
                    'parameters': {
                        'fps': 60,
                        'quality': 'photorealistic_plus',
                        'temporal_consistency': 'ultra_high',
                        'neural_rendering': True,
                        'physics_simulation': True
                    }
                }
            },
            'infrastructure': {
                'gpu_enabled': True,
                'max_concurrent_requests': 10,
                'cache_enabled': True,
                'cache_ttl': 3600,
                'storage_path': './generated_content',
                'temp_path': './temp',
                'max_file_size': 104857600,  # 100MB
                'cleanup_interval': 3600,
                'redis_url': '${REDIS_URL:redis://localhost:6379/0}',
                'redis_enabled': True,
                'semantic_similarity_enabled': True,
                'similarity_threshold': 0.85,
                'semantic_model_name': 'all-MiniLM-L6-v2'
            },
            'safety': {
                'enabled': True,
                'nsfw_detection': True,
                'violence_detection': True,
                'copyright_check': True,
                'prompt_filtering': True,
                'confidence_threshold': 0.8,
                'blocked_keywords': [
                    'explicit', 'nsfw', 'violence', 'harmful'
                ]
            },
            'quality': {
                'enabled': True,
                'min_quality_score': 0.7,
                'technical_quality_weight': 0.3,
                'aesthetic_weight': 0.3,
                'prompt_adherence_weight': 0.4,
                'auto_enhance': False,
                'quality_metrics': [
                    'sharpness', 'color_balance', 'composition',
                    'realism_score', 'temporal_consistency'
                ]
            },
            'cost': {
                'enabled': True,
                'cost_per_image': 0.01,
                'cost_per_video_second': 0.1,
                'cost_per_enhancement': 0.005,
                'budget_alerts': True,
                'daily_budget_limit': 100.0,
                'user_budget_limits': {}
            }
        }
        
        # Create config directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Write default config
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        return self.models.get(model_name)
    
    def get_enabled_models(self, model_type: Optional[str] = None) -> Dict[str, ModelConfig]:
        """Get all enabled models, optionally filtered by type."""
        enabled_models = {
            name: config for name, config in self.models.items() 
            if config.enabled
        }
        
        if model_type:
            enabled_models = {
                name: config for name, config in enabled_models.items()
                if config.type == model_type
            }
        
        return enabled_models
    
    def update_model_config(self, model_name: str, updates: Dict[str, Any]):
        """Update configuration for a specific model."""
        if model_name not in self.models:
            raise ConfigurationError(f"Model {model_name} not found in configuration")
        
        model_config = self.models[model_name]
        for key, value in updates.items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
            else:
                model_config.parameters[key] = value
    
    def save_config(self):
        """Save current configuration to file."""
        config_data = {
            'models': {
                name: {
                    'type': config.type,
                    'api_key': config.api_key,
                    'api_url': config.api_url,
                    'model_path': config.model_path,
                    'max_resolution': list(config.max_resolution),
                    'max_duration': config.max_duration,
                    'batch_size': config.batch_size,
                    'timeout': config.timeout,
                    'enabled': config.enabled,
                    'parameters': config.parameters
                }
                for name, config in self.models.items()
            },
            'infrastructure': {
                'gpu_enabled': self.infrastructure.gpu_enabled,
                'max_concurrent_requests': self.infrastructure.max_concurrent_requests,
                'cache_enabled': self.infrastructure.cache_enabled,
                'cache_ttl': self.infrastructure.cache_ttl,
                'storage_path': self.infrastructure.storage_path,
                'temp_path': self.infrastructure.temp_path,
                'max_file_size': self.infrastructure.max_file_size,
                'cleanup_interval': self.infrastructure.cleanup_interval,
                'redis_url': self.infrastructure.redis_url,
                'redis_enabled': self.infrastructure.redis_enabled,
                'semantic_similarity_enabled': self.infrastructure.semantic_similarity_enabled,
                'similarity_threshold': self.infrastructure.similarity_threshold,
                'semantic_model_name': self.infrastructure.semantic_model_name
            },
            'safety': {
                'enabled': self.safety.enabled,
                'nsfw_detection': self.safety.nsfw_detection,
                'violence_detection': self.safety.violence_detection,
                'copyright_check': self.safety.copyright_check,
                'prompt_filtering': self.safety.prompt_filtering,
                'confidence_threshold': self.safety.confidence_threshold,
                'blocked_keywords': self.safety.blocked_keywords
            },
            'quality': {
                'enabled': self.quality.enabled,
                'min_quality_score': self.quality.min_quality_score,
                'technical_quality_weight': self.quality.technical_quality_weight,
                'aesthetic_weight': self.quality.aesthetic_weight,
                'prompt_adherence_weight': self.quality.prompt_adherence_weight,
                'auto_enhance': self.quality.auto_enhance,
                'quality_metrics': self.quality.quality_metrics
            },
            'cost': {
                'enabled': self.cost.enabled,
                'cost_per_image': self.cost.cost_per_image,
                'cost_per_video_second': self.cost.cost_per_video_second,
                'cost_per_enhancement': self.cost.cost_per_enhancement,
                'budget_alerts': self.cost.budget_alerts,
                'daily_budget_limit': self.cost.daily_budget_limit,
                'user_budget_limits': self.cost.user_budget_limits
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    def resolve_environment_variables(self, value: str) -> str:
        """Resolve environment variables in configuration values."""
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]
            resolved_value = os.getenv(env_var)
            if resolved_value is None:
                raise ConfigurationError(f"Environment variable {env_var} not found")
            return resolved_value
        return value
    
    def validate_config(self) -> bool:
        """Validate the current configuration."""
        errors = []
        
        # Validate models
        for model_name, model_config in self.models.items():
            if model_config.enabled:
                if model_config.api_key and model_config.api_key.startswith('${'):
                    try:
                        self.resolve_environment_variables(model_config.api_key)
                    except ConfigurationError as e:
                        errors.append(f"Model {model_name}: {str(e)}")
                
                if model_config.model_path and not os.path.exists(model_config.model_path):
                    errors.append(f"Model {model_name}: Model path does not exist: {model_config.model_path}")
        
        # Validate infrastructure
        if not os.path.exists(self.infrastructure.storage_path):
            try:
                os.makedirs(self.infrastructure.storage_path, exist_ok=True)
            except Exception as e:
                errors.append(f"Infrastructure: Cannot create storage path: {str(e)}")
        
        if not os.path.exists(self.infrastructure.temp_path):
            try:
                os.makedirs(self.infrastructure.temp_path, exist_ok=True)
            except Exception as e:
                errors.append(f"Infrastructure: Cannot create temp path: {str(e)}")
        
        if errors:
            raise ConfigurationError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return True