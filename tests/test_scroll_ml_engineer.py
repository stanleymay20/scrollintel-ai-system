"""
Unit tests for ScrollMLEngineer agent.
Tests ML engineering workflows and pipeline management capabilities.
"""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from scrollintel.agents.scroll_ml_engineer import (
    ScrollMLEngineer, MLFramework, PipelineStage, DeploymentTarget,
    MLPipeline, ModelDeployment, ModelMonitoring
)
from scrollintel.core.interfaces import AgentRequest, AgentType, ResponseStatus


class TestScrollMLEngineer:
    """Test cases for ScrollMLEngineer agent."""
    
    @pytest.fixture
    def agent(self):
        """Create ScrollMLEngineer agent instance."""
        return ScrollMLEngineer()
    
    @pytest.fixture
    def sample_request(self):
        """Create sample agent request."""
        return AgentRequest(
            id="test-request-1",
            user_id="test-user",
            agent_id="scroll-ml-engineer",
            prompt="Set up ML pipeline for customer churn prediction",
            context={
                "action": "setup_pipeline",
                "dataset_path": "data/customers.csv",
                "target_column": "churn",
                "framework": MLFramework.SCIKIT_LEARN
            },
            priority=1,
            created_at=datetime.now()
        )
    
    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.agent_id == "scroll-ml-engineer"
        assert agent.name == "ScrollMLEngineer Agent"
        assert agent.agent_type == AgentType.ML_ENGINEER
        assert len(agent.capabilities) == 5
        assert agent.active_pipelines == []
        assert agent.active_deployments == []
        assert agent.monitoring_jobs == []
    
    def test_capabilities(self, agent):
        """Test agent capabilities."""
        capabilities = agent.get_capabilities()
        capability_names = [cap.name for cap in capabilities]
        
        expected_capabilities = [
            "ml_pipeline_setup",
            "model_deployment", 
            "model_monitoring",
            "framework_integration",
            "mlops_automation"
        ]
        
        for expected in expected_capabilities:
            assert expected in capability_names
    
    @pytest.mark.asyncio
    async def test_health_check(self, agent):
        """Test agent health check."""
        is_healthy = await agent.health_check()
        assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_setup_ml_pipeline(self, agent, sample_request):
        """Test ML pipeline setup."""
        with patch.object(agent, '_get_gpt4_pipeline_advice', return_value="AI recommendations"):
            response = await agent.process_request(sample_request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "ML Pipeline Setup Report" in response.content
            assert "Pipeline Configuration" in response.content
            assert len(agent.active_pipelines) == 1
            
            pipeline = agent.active_pipelines[0]
            assert pipeline.dataset_path == "data/customers.csv"
            assert pipeline.target_column == "churn"
            assert pipeline.framework == MLFramework.SCIKIT_LEARN
    
    @pytest.mark.asyncio
    async def test_deploy_model(self, agent):
        """Test model deployment."""
        request = AgentRequest(
            id="test-deploy-1",
            user_id="test-user",
            agent_id="scroll-ml-engineer",
            prompt="Deploy my trained model",
            context={
                "action": "deploy_model",
                "model_name": "churn_model",
                "model_path": "models/churn_model.pkl",
                "deployment_target": DeploymentTarget.FASTAPI
            },
            priority=1,
            created_at=datetime.now()
        )
        
        with patch.object(agent, '_get_gpt4_deployment_advice', return_value="Deployment recommendations"):
            response = await agent.process_request(request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "Model Deployment Report" in response.content
            assert "Deployment Configuration" in response.content
            assert len(agent.active_deployments) == 1
            
            deployment = agent.active_deployments[0]
            assert deployment.model_name == "churn_model"
            assert deployment.target == DeploymentTarget.FASTAPI
    
    @pytest.mark.asyncio
    async def test_monitor_model(self, agent):
        """Test model monitoring."""
        request = AgentRequest(
            id="test-monitor-1",
            user_id="test-user",
            agent_id="scroll-ml-engineer",
            prompt="Monitor my deployed model",
            context={
                "action": "monitor_model",
                "model_name": "churn_model",
                "monitoring_config": {
                    "accuracy": 0.92,
                    "drift_detected": False,
                    "performance_degradation": False
                }
            },
            priority=1,
            created_at=datetime.now()
        )
        
        with patch.object(agent, '_get_gpt4_monitoring_advice', return_value="Monitoring recommendations"):
            response = await agent.process_request(request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "Model Monitoring Report" in response.content
            assert "Performance Analysis" in response.content
            assert len(agent.monitoring_jobs) == 1
            
            monitoring = agent.monitoring_jobs[0]
            assert monitoring.model_name == "churn_model"
            assert not monitoring.drift_detected
    
    @pytest.mark.asyncio
    async def test_framework_integration(self, agent):
        """Test framework integration."""
        request = AgentRequest(
            id="test-framework-1",
            user_id="test-user",
            agent_id="scroll-ml-engineer",
            prompt="Help me integrate TensorFlow",
            context={
                "action": "framework_integration",
                "framework": MLFramework.TENSORFLOW,
                "integration_type": "deep_learning",
                "requirements": ["neural_networks", "gpu_support"]
            },
            priority=1,
            created_at=datetime.now()
        )
        
        with patch.object(agent, '_get_gpt4_framework_advice', return_value="TensorFlow integration advice"):
            response = await agent.process_request(request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "Framework Integration Report" in response.content
            assert "tensorflow" in response.content.lower()
            assert "Integration Code" in response.content
    
    @pytest.mark.asyncio
    async def test_mlops_automation(self, agent):
        """Test MLOps automation."""
        request = AgentRequest(
            id="test-mlops-1",
            user_id="test-user",
            agent_id="scroll-ml-engineer",
            prompt="Set up MLOps automation",
            context={
                "action": "mlops_automation",
                "project_name": "ml_project",
                "mlops_dir": Path("test_mlops"),
                "config": {"ci_cd": True, "monitoring": True}
            },
            priority=1,
            created_at=datetime.now()
        )
        
        with patch.object(agent, '_get_gpt4_mlops_advice', return_value="MLOps automation advice"):
            response = await agent.process_request(request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "MLOps Automation Report" in response.content
            assert "Generated Artifacts" in response.content
    
    @pytest.mark.asyncio
    async def test_general_advice(self, agent):
        """Test general ML engineering advice."""
        request = AgentRequest(
            id="test-general-1",
            user_id="test-user",
            agent_id="scroll-ml-engineer",
            prompt="What's the best approach for model versioning?",
            context={},
            priority=1,
            created_at=datetime.now()
        )
        
        with patch.object(agent, '_get_gpt4_general_advice', return_value="Model versioning best practices"):
            response = await agent.process_request(request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "ML Engineering Consultation" in response.content
            assert "System Status" in response.content
    
    def test_generate_sklearn_pipeline(self, agent):
        """Test scikit-learn pipeline code generation."""
        code = agent._generate_sklearn_pipeline("data/test.csv", "target")
        
        assert "import pandas as pd" in code
        assert "import numpy as np" in code
        assert "from sklearn" in code
        assert "RandomForestClassifier" in code
        assert "train_test_split" in code
        assert "data/test.csv" in code
        assert "target" in code
    
    def test_generate_tensorflow_pipeline(self, agent):
        """Test TensorFlow pipeline code generation."""
        code = agent._generate_tensorflow_pipeline("data/test.csv", "target")
        
        assert "import tensorflow as tf" in code
        assert "keras" in code
        assert "Sequential" in code
        assert "Dense" in code
    
    def test_generate_pytorch_pipeline(self, agent):
        """Test PyTorch pipeline code generation."""
        code = agent._generate_pytorch_pipeline("data/test.csv", "target")
        
        assert "import torch" in code
        assert "torch.nn" in code
        assert "nn.Module" in code
        assert "SimpleNet" in code
    
    def test_generate_deployment_artifacts_fastapi(self, agent):
        """Test FastAPI deployment artifacts generation."""
        artifacts = agent._generate_deployment_artifacts(
            "test_model", "models/test.pkl", DeploymentTarget.FASTAPI, {}
        )
        
        assert "FastAPI App" in artifacts
        assert "Dockerfile" in artifacts
        assert "Requirements" in artifacts
        
        app_code = artifacts["FastAPI App"]
        assert "from fastapi import FastAPI" in app_code
        assert "test_model" in app_code
        assert "/predict" in app_code
    
    def test_analyze_retraining_needs(self, agent):
        """Test retraining needs analysis."""
        config = {
            "accuracy": 0.85,
            "drift_detected": True,
            "new_data_available": True
        }
        
        analysis = agent._analyze_retraining_needs("test_model", config)
        
        assert "drift_analysis" in analysis
        assert "performance_metrics" in analysis
        assert "data_quality" in analysis
        assert analysis["drift_detected"] is True
    
    def test_framework_configs(self, agent):
        """Test framework configurations."""
        configs = agent.framework_configs
        
        assert MLFramework.SCIKIT_LEARN in configs
        assert MLFramework.TENSORFLOW in configs
        assert MLFramework.PYTORCH in configs
        assert MLFramework.XGBOOST in configs
        
        sklearn_config = configs[MLFramework.SCIKIT_LEARN]
        assert "requirements" in sklearn_config
        assert "features" in sklearn_config
        assert "pandas" in str(sklearn_config["requirements"])
    
    def test_format_pipeline_stages(self, agent):
        """Test pipeline stages formatting."""
        stages = [PipelineStage.DATA_INGESTION, PipelineStage.MODEL_TRAINING]
        formatted = agent._format_pipeline_stages(stages)
        
        assert "Data Ingestion" in formatted
        assert "Model Training" in formatted
    
    def test_format_monitoring_analysis(self, agent):
        """Test monitoring analysis formatting."""
        analysis = {
            "performance_metrics": {
                "accuracy": 0.95,
                "latency_ms": 45
            }
        }
        
        formatted = agent._format_monitoring_analysis(analysis)
        assert "Accuracy" in formatted
        assert "0.95" in formatted
        assert "Latency Ms" in formatted
    
    @pytest.mark.asyncio
    async def test_error_handling(self, agent):
        """Test error handling in request processing."""
        request = AgentRequest(
            id="test-error-1",
            user_id="test-user",
            agent_id="scroll-ml-engineer",
            prompt="Invalid request",
            context={"action": "invalid_action"},
            priority=1,
            created_at=datetime.now()
        )
        
        # Mock GPT-4 to raise an exception
        with patch.object(agent, '_get_gpt4_general_advice', side_effect=Exception("API Error")):
            response = await agent.process_request(request)
            
            assert response.status == ResponseStatus.ERROR
            assert "Error processing request" in response.content
            assert response.error_message is not None
    
    @pytest.mark.asyncio
    async def test_gpt4_integration_without_api_key(self, agent):
        """Test GPT-4 integration when API key is not available."""
        # Set openai_client to None to simulate missing API key
        agent.openai_client = None
        
        advice = await agent._get_gpt4_general_advice("test prompt", {})
        assert "Enhanced AI analysis unavailable" in advice
    
    def test_mlpipeline_dataclass(self):
        """Test MLPipeline dataclass."""
        pipeline = MLPipeline(
            name="test_pipeline",
            dataset_path="data/test.csv",
            target_column="target",
            framework=MLFramework.SCIKIT_LEARN,
            stages=[PipelineStage.DATA_INGESTION],
            config={"test": True},
            created_at=datetime.now()
        )
        
        assert pipeline.name == "test_pipeline"
        assert pipeline.framework == MLFramework.SCIKIT_LEARN
        assert pipeline.status == "created"
    
    def test_model_deployment_dataclass(self):
        """Test ModelDeployment dataclass."""
        deployment = ModelDeployment(
            model_name="test_model",
            model_path="models/test.pkl",
            target=DeploymentTarget.FASTAPI,
            created_at=datetime.now()
        )
        
        assert deployment.model_name == "test_model"
        assert deployment.target == DeploymentTarget.FASTAPI
        assert deployment.status == "pending"
    
    def test_model_monitoring_dataclass(self):
        """Test ModelMonitoring dataclass."""
        monitoring = ModelMonitoring(
            model_name="test_model",
            metrics={"accuracy": 0.95},
            drift_detected=False,
            performance_degradation=False,
            alerts=[],
            last_check=datetime.now()
        )
        
        assert monitoring.model_name == "test_model"
        assert monitoring.metrics["accuracy"] == 0.95
        assert not monitoring.drift_detected


class TestMLFrameworkIntegration:
    """Test ML framework integration capabilities."""
    
    @pytest.fixture
    def agent(self):
        """Create ScrollMLEngineer agent instance."""
        return ScrollMLEngineer()
    
    def test_sklearn_integration(self, agent):
        """Test scikit-learn integration."""
        code = agent._generate_pipeline_code("data.csv", "target", MLFramework.SCIKIT_LEARN)
        
        assert "sklearn" in code
        assert "RandomForestClassifier" in code
        assert "train_test_split" in code
    
    def test_tensorflow_integration(self, agent):
        """Test TensorFlow integration."""
        code = agent._generate_pipeline_code("data.csv", "target", MLFramework.TENSORFLOW)
        
        assert "tensorflow" in code
        assert "keras" in code
    
    def test_pytorch_integration(self, agent):
        """Test PyTorch integration."""
        code = agent._generate_pipeline_code("data.csv", "target", MLFramework.PYTORCH)
        
        assert "torch" in code
        assert "nn.Module" in code
    
    def test_framework_requirements(self, agent):
        """Test framework requirements generation."""
        requirements = agent._generate_framework_requirements(MLFramework.SCIKIT_LEARN)
        
        assert "pandas" in requirements
        assert "scikit-learn" in requirements
        assert "numpy" in requirements


class TestDeploymentArtifacts:
    """Test deployment artifacts generation."""
    
    @pytest.fixture
    def agent(self):
        """Create ScrollMLEngineer agent instance."""
        return ScrollMLEngineer()
    
    def test_fastapi_artifacts(self, agent):
        """Test FastAPI deployment artifacts."""
        artifacts = agent._generate_deployment_artifacts(
            "test_model", "models/test.pkl", DeploymentTarget.FASTAPI, {}
        )
        
        assert "FastAPI App" in artifacts
        assert "Dockerfile" in artifacts
        assert "Requirements" in artifacts
        
        # Check FastAPI app structure
        app_code = artifacts["FastAPI App"]
        assert "from fastapi import FastAPI" in app_code
        assert "@app.post(\"/predict\")" in app_code
        assert "PredictionRequest" in app_code
        assert "PredictionResponse" in app_code
        
        # Check Dockerfile
        dockerfile = artifacts["Dockerfile"]
        assert "FROM python:3.9-slim" in dockerfile
        assert "uvicorn" in dockerfile
        
        # Check requirements
        requirements = artifacts["Requirements"]
        assert "fastapi" in requirements
        assert "uvicorn" in requirements


class TestMLOpsAutomation:
    """Test MLOps automation capabilities."""
    
    @pytest.fixture
    def agent(self):
        """Create ScrollMLEngineer agent instance."""
        return ScrollMLEngineer()
    
    def test_mlops_artifacts_generation(self, agent, tmp_path):
        """Test MLOps artifacts generation."""
        artifacts = agent._generate_mlops_artifacts_files("test_project", tmp_path, {})
        
        assert "GitHub Actions Workflow" in artifacts
        assert "Requirements" in artifacts
        assert "Dockerfile" in artifacts
        
        # Check if files were created
        assert (tmp_path / ".github" / "workflows" / "mlops.yml").exists()
        assert (tmp_path / "requirements.txt").exists()
        assert (tmp_path / "deployment" / "Dockerfile").exists()


if __name__ == "__main__":
    pytest.main([__file__])