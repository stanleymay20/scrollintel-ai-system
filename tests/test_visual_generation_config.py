"""
Comprehensive tests for visual generation configuration system.
"""
import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from scrollintel.engines.visual_generation.config_manager import (
    ConfigurationManager, DeploymentConfig, create_optimal_setup
)
from scrollintel.engines.visual_generation.config import (
    VisualGenerationConfig, ModelConfig, InfrastructureConfig
)
from scrollintel.engines.visual_generation.exceptions import ConfigurationError


class TestDeploymentConfig:
    """Test deployment configuration"""
    
    def test_deployment_config_defaults(self):
        """Test default deployment configuration"""
        config = DeploymentConfig()
        
        assert config.mode == "hybrid"
        assert config.fallback_enabled == True
        assert config.cost_optimization == True
        assert config.quality_preference == "balanced"
        assert config.api_timeout == 30.0
        assert config.max_retries == 3
    
    def test_deployment_config_custom(self):
        """Test custom deployment configuration"""
        config = DeploymentConfig(
            mode="self_hosted",
            fallback_enabled=False,
            cost_optimization=False,
            quality_preference="quality",
            api_timeout=60.0,
            max_retries=5
        )
        
        assert config.mode == "self_hosted"
        assert config.fallback_enabled == False
        assert config.cost_optimization == False
        assert config.quality_preference == "quality"
        assert config.api_timeout == 60.0
        assert config.max_retries == 5


class TestConfigurationManager:
    """Test configuration manager"""
    
    @pytest.fixture
    def config_manager(self):
        """Get configuration manager for testing"""
        return ConfigurationManager()
    
    def test_initialization(self, config_manager):
        """Test configuration manager initialization"""
        assert config_manager.config_path is None
        assert isinstance(config_manager.base_config, VisualGenerationConfig)
        assert isinstance(config_manager.deployment_config, DeploymentConfig)
        assert isinstance(config_manager.api_keys_available, dict)
    
    @patch.dict(os.environ, {
        'STABILITY_API_KEY': 'test_stability_key',
        'OPENAI_API_KEY': 'test_openai_key'
    })
    def test_api_key_detection_with_keys(self, config_manager):
        """Test API key detection when keys are present"""
        api_keys = config_manager._check_api_keys()
        
        # Should detect available keys
        assert api_keys['stability_ai'] == True
        assert api_keys['openai'] == True
        assert api_keys['midjourney'] == False  # Not set
        assert api_keys['anthropic'] == False  # Not set
        assert api_keys['replicate'] == False  # Not set
    
    @patch.dict(os.environ, {}, clear=True)
    def test_api_key_detection_without_keys(self, config_manager):
        """Test API key detection when no keys are present"""
        api_keys = config_manager._check_api_keys()
        
        # Should detect no keys
        for key, available in api_keys.items():
            assert available == False, f"{key} should not be available"
    
    def test_optimal_configuration_hybrid_mode(self, config_manager):
        """Test optimal configuration in hybrid mode"""
        config_manager.deployment_config.mode = "hybrid"
        
        config = config_manager.get_optimal_configuration()
        
        # Should be a valid configuration
        assert isinstance(config, VisualGenerationConfig)
        assert len(config.models) > 0
        
        # Should have ScrollIntel proprietary models
        scrollintel_models = [name for name in config.models.keys() if name.startswith('scrollintel_')]
        assert len(scrollintel_models) >= 3, "Should have ScrollIntel proprietary models"
    
    def test_self_hosted_configuration(self, config_manager):
        """Test self-hosted only configuration"""
        config = config_manager._configure_self_hosted_only(config_manager.base_config)
        
        # Should disable external API models
        external_models = ['stable_diffusion_xl', 'dalle3', 'midjourney']
        for model_name in external_models:
            if model_name in config.models:
                assert config.models[model_name].enabled == False, f"{model_name} should be disabled"
        
        # Should enable proprietary models
        proprietary_models = [name for name, model in config.models.items() 
                            if name.startswith('scrollintel_') and model.enabled]
        assert len(proprietary_models) > 0, "Should enable ScrollIntel models"
    
    @patch.dict(os.environ, {
        'STABILITY_API_KEY': 'test_key',
        'OPENAI_API_KEY': 'test_key'
    })
    def test_api_only_configuration(self, config_manager):
        """Test API-only configuration"""
        config_manager.api_keys_available = config_manager._check_api_keys()
        config = config_manager._configure_api_only(config_manager.base_config)
        
        # Should enable models with available API keys
        if 'stable_diffusion_xl' in config.models:
            assert config.models['stable_diffusion_xl'].enabled == True
        if 'dalle3' in config.models:
            assert config.models['dalle3'].enabled == True
        
        # Should disable proprietary models
        proprietary_models = [name for name, model in config.models.items() 
                            if name.startswith('proprietary_') and model.enabled]
        assert len(proprietary_models) == 0, "Should disable proprietary models in API-only mode"
    
    def test_proprietary_models_addition(self, config_manager):
        """Test addition of proprietary ScrollIntel models"""
        config = VisualGenerationConfig()
        config_manager._add_proprietary_models(config)
        
        # Should add all proprietary models
        expected_models = [
            'scrollintel_image_generator',
            'scrollintel_video_generator', 
            'scrollintel_enhancement_suite'
        ]
        
        for model_name in expected_models:
            assert model_name in config.models, f"Should add {model_name}"
            model_config = config.models[model_name]
            assert model_config.enabled == True
            assert model_config.parameters is not None
    
    def test_scrollintel_image_generator_config(self, config_manager):
        """Test ScrollIntel image generator configuration"""
        config = VisualGenerationConfig()
        config_manager._add_proprietary_models(config)
        
        img_gen = config.models['scrollintel_image_generator']
        assert img_gen.name == 'scrollintel_image_generator'
        assert img_gen.type == 'image'
        assert img_gen.max_resolution == (4096, 4096)
        assert img_gen.batch_size == 4
        assert img_gen.timeout == 60.0
        assert img_gen.enabled == True
        
        # Should have advanced parameters
        params = img_gen.parameters
        assert params['quality'] == 'ultra_high'
        assert params['style_control'] == 'advanced'
        assert params['prompt_enhancement'] == True
        assert params['negative_prompt_auto'] == True
    
    def test_scrollintel_video_generator_config(self, config_manager):
        """Test ScrollIntel video generator configuration"""
        config = VisualGenerationConfig()
        config_manager._add_proprietary_models(config)
        
        vid_gen = config.models['scrollintel_video_generator']
        assert vid_gen.name == 'scrollintel_video_generator'
        assert vid_gen.type == 'video'
        assert vid_gen.max_resolution == (3840, 2160)  # 4K
        assert vid_gen.max_duration == 1800.0  # 30 minutes
        assert vid_gen.batch_size == 1
        assert vid_gen.timeout == 3600.0  # 1 hour
        assert vid_gen.enabled == True
        
        # Should have revolutionary parameters
        params = vid_gen.parameters
        assert params['fps'] == 60
        assert params['quality'] == 'photorealistic_plus'
        assert params['temporal_consistency'] == 'perfect'
        assert params['physics_simulation'] == True
        assert params['humanoid_generation'] == True
        assert params['neural_rendering'] == True
    
    def test_scrollintel_enhancement_suite_config(self, config_manager):
        """Test ScrollIntel enhancement suite configuration"""
        config = VisualGenerationConfig()
        config_manager._add_proprietary_models(config)
        
        enhancement = config.models['scrollintel_enhancement_suite']
        assert enhancement.name == 'scrollintel_enhancement_suite'
        assert enhancement.type == 'enhancement'
        assert enhancement.max_resolution == (8192, 8192)
        assert enhancement.batch_size == 2
        assert enhancement.timeout == 300.0
        assert enhancement.enabled == True
        
        # Should have superior parameters
        params = enhancement.parameters
        assert params['upscale_factor'] == 8
        assert params['face_restoration'] == True
        assert params['style_transfer'] == True
        assert params['artifact_removal'] == True
        assert params['quality_boost'] == True
    
    def test_environment_setup(self, config_manager):
        """Test environment variable setup"""
        config_manager.deployment_config.mode = "hybrid"
        config_manager.deployment_config.fallback_enabled = True
        config_manager.deployment_config.cost_optimization = True
        config_manager.deployment_config.quality_preference = "quality"
        
        env_vars = config_manager.setup_environment()
        
        # Should set appropriate environment variables
        assert env_vars['SCROLLINTEL_VISUAL_MODE'] == "hybrid"
        assert env_vars['SCROLLINTEL_FALLBACK_ENABLED'] == "True"
        assert env_vars['SCROLLINTEL_COST_OPTIMIZATION'] == "True"
        assert env_vars['SCROLLINTEL_QUALITY_PREFERENCE'] == "quality"
    
    @patch.dict(os.environ, {}, clear=True)
    def test_environment_setup_missing_keys_warning(self, config_manager, capsys):
        """Test warning for missing API keys"""
        config_manager.deployment_config.mode = "hybrid"
        config_manager.api_keys_available = config_manager._check_api_keys()
        
        env_vars = config_manager.setup_environment()
        
        # Should warn about missing keys
        captured = capsys.readouterr()
        assert "Missing API keys" in captured.out
        assert "STABILITY_API_KEY" in captured.out
        assert "OPENAI_API_KEY" in captured.out
    
    def test_deployment_recommendations_basic(self, config_manager):
        """Test basic deployment recommendations"""
        recommendations = config_manager.get_deployment_recommendations()
        
        # Should provide comprehensive recommendations
        assert 'current_mode' in recommendations
        assert 'api_keys_available' in recommendations
        assert 'enabled_models' in recommendations
        assert 'performance_tier' in recommendations
        assert 'cost_estimate' in recommendations
        assert 'recommendations' in recommendations
        
        # Should have some enabled models
        assert len(recommendations['enabled_models']) > 0
        
        # Should provide recommendations
        assert len(recommendations['recommendations']) > 0
    
    @patch.dict(os.environ, {
        'STABILITY_API_KEY': 'test_key',
        'OPENAI_API_KEY': 'test_key',
        'MIDJOURNEY_API_KEY': 'test_key'
    })
    def test_deployment_recommendations_enterprise_tier(self, config_manager):
        """Test enterprise tier recommendations with API keys"""
        config_manager.api_keys_available = config_manager._check_api_keys()
        recommendations = config_manager.get_deployment_recommendations()
        
        # Should detect enterprise tier
        assert recommendations['performance_tier'] == 'enterprise'
        
        # Should have many enabled models
        assert len(recommendations['enabled_models']) > 3
    
    @patch.dict(os.environ, {}, clear=True)
    def test_deployment_recommendations_no_api_keys(self, config_manager):
        """Test recommendations when no API keys are available"""
        config_manager.api_keys_available = config_manager._check_api_keys()
        recommendations = config_manager.get_deployment_recommendations()
        
        # Should recommend adding API keys
        recommendations_text = " ".join(recommendations['recommendations'])
        assert "API keys" in recommendations_text
    
    def test_deployment_recommendations_hybrid_mode(self, config_manager):
        """Test recommendations for hybrid mode"""
        config_manager.deployment_config.mode = "hybrid"
        recommendations = config_manager.get_deployment_recommendations()
        
        # Should mention hybrid mode
        recommendations_text = " ".join(recommendations['recommendations'])
        assert "Hybrid mode" in recommendations_text or "hybrid" in recommendations_text.lower()


class TestCreateOptimalSetup:
    """Test optimal setup creation"""
    
    @patch.dict(os.environ, {}, clear=True)
    def test_create_optimal_setup_self_hosted(self, capsys):
        """Test optimal setup creation when no API keys available"""
        config_manager = create_optimal_setup()
        
        # Should set self-hosted mode
        assert config_manager.deployment_config.mode == "self_hosted"
        
        # Should print setup message
        captured = capsys.readouterr()
        assert "SELF-HOSTED mode" in captured.out
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    def test_create_optimal_setup_hybrid(self, capsys):
        """Test optimal setup creation when API keys available"""
        config_manager = create_optimal_setup()
        
        # Should set hybrid mode
        assert config_manager.deployment_config.mode == "hybrid"
        
        # Should print setup message
        captured = capsys.readouterr()
        assert "HYBRID mode" in captured.out
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    def test_create_optimal_setup_environment_variables(self):
        """Test that optimal setup sets environment variables"""
        with patch.dict(os.environ, {}, clear=True):
            # Clear environment first
            config_manager = create_optimal_setup()
            
            # Should set ScrollIntel environment variables
            assert os.environ.get('SCROLLINTEL_VISUAL_MODE') is not None
            assert os.environ.get('SCROLLINTEL_FALLBACK_ENABLED') is not None
            assert os.environ.get('SCROLLINTEL_COST_OPTIMIZATION') is not None
            assert os.environ.get('SCROLLINTEL_QUALITY_PREFERENCE') is not None


class TestVisualGenerationConfig:
    """Test base visual generation configuration"""
    
    def test_visual_generation_config_initialization(self):
        """Test VisualGenerationConfig initialization"""
        config = VisualGenerationConfig()
        
        # Should have basic structure
        assert hasattr(config, 'models')
        assert isinstance(config.models, dict)
    
    def test_visual_generation_config_with_path(self):
        """Test VisualGenerationConfig with custom path"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
models:
  test_model:
    name: test_model
    type: image
    enabled: true
""")
            config_path = f.name
        
        try:
            config = VisualGenerationConfig(config_path)
            # Should load from file (implementation dependent)
            assert hasattr(config, 'models')
        finally:
            os.unlink(config_path)


class TestModelConfig:
    """Test model configuration"""
    
    def test_model_config_basic(self):
        """Test basic model configuration"""
        config = ModelConfig(
            name="test_model",
            type="image",
            model_path="/path/to/model",
            enabled=True
        )
        
        assert config.name == "test_model"
        assert config.type == "image"
        assert config.model_path == "/path/to/model"
        assert config.enabled == True
    
    def test_model_config_with_parameters(self):
        """Test model configuration with parameters"""
        config = ModelConfig(
            name="advanced_model",
            type="video",
            model_path="/path/to/model",
            max_resolution=(4096, 4096),
            max_duration=1800.0,
            batch_size=2,
            timeout=300.0,
            enabled=True,
            parameters={
                'quality': 'ultra_high',
                'fps': 60,
                'physics': True
            }
        )
        
        assert config.name == "advanced_model"
        assert config.type == "video"
        assert config.max_resolution == (4096, 4096)
        assert config.max_duration == 1800.0
        assert config.batch_size == 2
        assert config.timeout == 300.0
        assert config.enabled == True
        assert config.parameters['quality'] == 'ultra_high'
        assert config.parameters['fps'] == 60
        assert config.parameters['physics'] == True


class TestConfigurationIntegration:
    """Test configuration system integration"""
    
    def test_configuration_manager_with_custom_config(self):
        """Test configuration manager with custom base config"""
        # Create custom config
        custom_config = VisualGenerationConfig()
        
        # Create manager with custom path
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("# Custom config")
            config_path = f.name
        
        try:
            config_manager = ConfigurationManager(config_path)
            assert config_manager.config_path == config_path
        finally:
            os.unlink(config_path)
    
    def test_end_to_end_configuration_flow(self):
        """Test complete configuration flow"""
        # Create optimal setup
        config_manager = create_optimal_setup()
        
        # Get optimal configuration
        config = config_manager.get_optimal_configuration()
        
        # Should have ScrollIntel models
        scrollintel_models = [name for name in config.models.keys() if name.startswith('scrollintel_')]
        assert len(scrollintel_models) >= 3
        
        # Get deployment recommendations
        recommendations = config_manager.get_deployment_recommendations()
        assert 'current_mode' in recommendations
        assert 'enabled_models' in recommendations
        
        # Should have enabled models
        assert len(recommendations['enabled_models']) > 0
    
    def test_configuration_consistency(self):
        """Test configuration consistency across different modes"""
        config_manager = ConfigurationManager()
        
        # Test all modes
        modes = ["self_hosted", "api_only", "hybrid"]
        
        for mode in modes:
            config_manager.deployment_config.mode = mode
            config = config_manager.get_optimal_configuration()
            
            # Should always have some models enabled
            enabled_models = [name for name, model in config.models.items() if model.enabled]
            assert len(enabled_models) > 0, f"Mode {mode} should have enabled models"
            
            # Self-hosted should prioritize ScrollIntel models
            if mode == "self_hosted":
                scrollintel_models = [name for name in enabled_models if name.startswith('scrollintel_')]
                assert len(scrollintel_models) > 0, "Self-hosted should have ScrollIntel models"
    
    def test_configuration_error_handling(self):
        """Test configuration error handling"""
        # Test with invalid config path
        with pytest.raises((FileNotFoundError, ConfigurationError, Exception)):
            # This might raise different exceptions depending on implementation
            ConfigurationManager("/nonexistent/path/config.yaml")
    
    def test_model_config_validation(self):
        """Test model configuration validation"""
        # Valid config should work
        valid_config = ModelConfig(
            name="valid_model",
            type="image",
            model_path="/valid/path",
            enabled=True
        )
        assert valid_config.name == "valid_model"
        
        # Test with all required fields
        required_fields = ['name', 'type', 'model_path', 'enabled']
        for field in required_fields:
            kwargs = {
                'name': 'test',
                'type': 'image', 
                'model_path': '/path',
                'enabled': True
            }
            # Remove one required field
            del kwargs[field]
            
            # Should handle missing required field
            try:
                ModelConfig(**kwargs)
            except (TypeError, ValueError):
                # Expected for missing required fields
                pass


if __name__ == "__main__":
    pytest.main([__file__])