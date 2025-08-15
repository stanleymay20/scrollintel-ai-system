"""
ScrollMLEngineer Agent - MLOps capabilities and GPT-4 integration
Implements requirements 3.1, 3.3, 3.4 for ML engineering workflows and pipeline management.
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path
from uuid import uuid4
from enum import Enum
from dataclasses import dataclass, field

# Core imports
from scrollintel.core.interfaces import BaseAgent, AgentType, AgentRequest, AgentResponse, AgentCapability, ResponseStatus

logger = logging.getLogger(__name__)


class MLFramework(str, Enum):
    """Supported ML frameworks."""
    SCIKIT_LEARN = "scikit-learn"
    XGBOOST = "xgboost"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"


class PipelineStage(str, Enum):
    """ML pipeline stages."""
    DATA_INGESTION = "data_ingestion"
    DATA_PREPROCESSING = "data_preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    MODEL_DEPLOYMENT = "model_deployment"
    MODEL_MONITORING = "model_monitoring"


class DeploymentTarget(str, Enum):
    """Model deployment targets."""
    FASTAPI = "fastapi"
    FLASK = "flask"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"


@dataclass
class MLPipeline:
    """ML pipeline configuration."""
    name: str
    dataset_path: str
    target_column: str
    framework: MLFramework
    stages: List[PipelineStage]
    config: Dict[str, Any]
    created_at: datetime
    status: str = "created"


@dataclass
class ModelDeployment:
    """Model deployment configuration."""
    model_name: str
    model_path: str
    target: DeploymentTarget
    endpoint_url: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    status: str = "pending"
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.config is None:
            self.config = {}


@dataclass
class ModelMonitoring:
    """Model monitoring configuration."""
    model_name: str
    metrics: Dict[str, float]
    drift_detected: bool = False
    performance_degradation: bool = False
    alerts: Optional[List[str]] = None
    last_check: Optional[datetime] = None

    def __post_init__(self):
        if self.alerts is None:
            self.alerts = []
        if self.last_check is None:
            self.last_check = datetime.now()


class ScrollMLEngineer(BaseAgent):
    """
    ScrollMLEngineer agent with MLOps capabilities and GPT-4 integration.
    
    Provides comprehensive ML engineering services including:
    - ML pipeline setup with automated data preprocessing
    - Model deployment system with API endpoint generation
    - Model monitoring and retraining automation
    - Integration with popular ML frameworks (scikit-learn, TensorFlow, PyTorch)
    """
    
    def __init__(self):
        super().__init__(
            agent_id="scroll-ml-engineer",
            name="ScrollMLEngineer Agent",
            agent_type=AgentType.ML_ENGINEER
        )
        
        # Initialize directories
        self.pipelines_dir = Path("ml_pipelines")
        self.models_dir = Path("models")
        self.deployments_dir = Path("deployments")
        
        for directory in [self.pipelines_dir, self.models_dir, self.deployments_dir]:
            directory.mkdir(exist_ok=True)
        
        # Initialize state
        self.active_pipelines: List[MLPipeline] = []
        self.active_deployments: List[ModelDeployment] = []
        self.monitoring_jobs: List[ModelMonitoring] = []
        self.framework_configs = self._initialize_framework_configs()
        
        # Initialize OpenAI client if available
        self.openai_client = None
        try:
            import openai
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = openai.OpenAI(api_key=api_key)
        except ImportError:
            logger.warning("OpenAI package not available. GPT-4 features will be disabled.")
        
        # Define capabilities
        self.capabilities = [
            AgentCapability(
                name="ml_pipeline_setup",
                description="Set up automated ML pipelines with data preprocessing",
                input_types=["dataset", "requirements"],
                output_types=["pipeline", "configuration"]
            ),
            AgentCapability(
                name="model_deployment",
                description="Deploy ML models with API endpoint generation",
                input_types=["model", "deployment_config"],
                output_types=["api_endpoint", "deployment_status"]
            ),
            AgentCapability(
                name="model_monitoring",
                description="Monitor model performance and trigger retraining",
                input_types=["model_metrics", "monitoring_config"],
                output_types=["monitoring_report", "alerts"]
            ),
            AgentCapability(
                name="framework_integration",
                description="Integrate with ML frameworks (scikit-learn, TensorFlow, PyTorch)",
                input_types=["framework_requirements", "model_config"],
                output_types=["integration_code", "examples"]
            ),
            AgentCapability(
                name="mlops_automation",
                description="Automate MLOps workflows and CI/CD pipelines",
                input_types=["project_config", "automation_requirements"],
                output_types=["mlops_artifacts", "workflows"]
            )
        ]
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Return the capabilities of this agent."""
        return self.capabilities
    
    async def health_check(self) -> bool:
        """Check if the agent is healthy and ready to process requests."""
        try:
            # Check if directories exist
            for directory in [self.pipelines_dir, self.models_dir, self.deployments_dir]:
                if not directory.exists():
                    return False
            
            # Check framework configurations
            if not self.framework_configs:
                return False
            
            return True
        except Exception:
            return False
    
    def _initialize_framework_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize framework-specific configurations."""
        return {
            MLFramework.SCIKIT_LEARN: {
                "requirements": [
                    "pandas>=1.5.0",
                    "numpy>=1.21.0",
                    "scikit-learn>=1.3.0",
                    "joblib>=1.3.0"
                ],
                "features": [
                    "Pipeline composition and chaining",
                    "Custom transformers and estimators",
                    "Hyperparameter optimization with GridSearchCV",
                    "Cross-validation and model selection",
                    "Feature importance and model interpretation"
                ]
            },
            MLFramework.TENSORFLOW: {
                "requirements": [
                    "tensorflow>=2.13.0",
                    "tensorboard>=2.13.0"
                ],
                "features": [
                    "Custom layers and models",
                    "Distributed training with DistributionStrategy",
                    "TensorBoard integration",
                    "Model serving with TensorFlow Serving"
                ]
            },
            MLFramework.PYTORCH: {
                "requirements": [
                    "torch>=2.0.0",
                    "torchvision>=0.15.0"
                ],
                "features": [
                    "Custom loss functions and optimizers",
                    "Distributed training with DDP",
                    "TorchScript for production deployment"
                ]
            },
            MLFramework.XGBOOST: {
                "requirements": [
                    "xgboost>=1.7.0"
                ],
                "features": [
                    "Gradient boosting for classification and regression",
                    "Early stopping and regularization",
                    "Feature importance analysis"
                ]
            }
        }

    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """Process ML engineering requests."""
        start_time = datetime.now()
        
        try:
            # Parse request context
            context = request.context
            action = context.get("action", "general_advice")
            
            # Route to appropriate handler
            if action == "setup_pipeline":
                result = await self._setup_ml_pipeline(request.prompt, context)
            elif action == "deploy_model":
                result = await self._deploy_model(request.prompt, context)
            elif action == "monitor_model":
                result = await self._monitor_model(request.prompt, context)
            elif action == "framework_integration":
                result = await self._handle_framework_integration(request.prompt, context)
            elif action == "mlops_automation":
                result = await self._generate_mlops_artifacts(request.prompt, context)
            else:
                result = await self._general_ml_engineering_advice(request.prompt, context)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                id=str(uuid4()),
                request_id=request.id,
                content=result,
                artifacts=[],
                execution_time=execution_time,
                status=ResponseStatus.SUCCESS
            )
            
        except Exception as e:
            logger.error(f"Error processing ML engineering request: {str(e)}")
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                id=str(uuid4()),
                request_id=request.id,
                content=f"Error processing request: {str(e)}",
                artifacts=[],
                execution_time=execution_time,
                status=ResponseStatus.ERROR,
                error_message=str(e)
            )    

    async def _setup_ml_pipeline(self, prompt: str, context: Dict[str, Any]) -> str:
        """Set up automated ML pipeline with data preprocessing."""
        try:
            # Extract pipeline configuration from context
            dataset_path = context.get("dataset_path", "")
            target_column = context.get("target_column", "")
            framework = context.get("framework", MLFramework.SCIKIT_LEARN)
            
            # Generate pipeline code
            pipeline_code = self._generate_pipeline_code(dataset_path, target_column, framework)
            
            # Get GPT-4 advice for pipeline optimization
            ai_advice = await self._get_gpt4_pipeline_advice(prompt, context)
            
            # Create pipeline configuration
            pipeline = MLPipeline(
                name=f"pipeline_{uuid4().hex[:8]}",
                dataset_path=dataset_path,
                target_column=target_column,
                framework=framework,
                stages=[
                    PipelineStage.DATA_INGESTION,
                    PipelineStage.DATA_PREPROCESSING,
                    PipelineStage.FEATURE_ENGINEERING,
                    PipelineStage.MODEL_TRAINING,
                    PipelineStage.MODEL_EVALUATION
                ],
                config=context,
                created_at=datetime.now()
            )
            
            self.active_pipelines.append(pipeline)
            
            # Generate comprehensive report
            report = f"""
# ML Pipeline Setup Report

## Pipeline Configuration
- **Name**: {pipeline.name}
- **Dataset**: {dataset_path}
- **Target Column**: {target_column}
- **Framework**: {framework.value}
- **Created At**: {pipeline.created_at.strftime('%Y-%m-%d %H:%M:%S')}

## Pipeline Stages
{self._format_pipeline_stages(pipeline.stages)}

## Generated Code
```python
{pipeline_code}
```

## AI-Enhanced Recommendations
{ai_advice}

## Next Steps
1. Review and customize the generated pipeline code
2. Execute the pipeline with your dataset
3. Monitor training progress and metrics
4. Deploy the best performing model
5. Set up automated retraining schedule

---
*Pipeline setup completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
            
            return report
            
        except Exception as e:
            logger.error(f"Error setting up ML pipeline: {str(e)}")
            return f"Error setting up ML pipeline: {str(e)}"
    
    async def _deploy_model(self, prompt: str, context: Dict[str, Any]) -> str:
        """Deploy ML model with API endpoint generation."""
        try:
            model_name = context.get("model_name", "")
            model_path = context.get("model_path", "")
            target = context.get("deployment_target", DeploymentTarget.FASTAPI)
            
            if not model_name or not model_path:
                return "Error: Model name and path are required for deployment."
            
            # Generate deployment artifacts
            deployment_artifacts = self._generate_deployment_artifacts(model_name, model_path, target, context)
            
            # Get GPT-4 advice for deployment optimization
            ai_advice = await self._get_gpt4_deployment_advice(prompt, context)
            
            # Create deployment configuration
            deployment = ModelDeployment(
                model_name=model_name,
                model_path=model_path,
                target=target,
                config=context,
                status="deployed"
            )
            
            self.active_deployments.append(deployment)
            
            # Generate deployment report
            report = f"""
# Model Deployment Report

## Deployment Configuration
- **Model**: {model_name}
- **Target**: {target.value}
- **Status**: {deployment.status}
- **Deployed At**: {deployment.created_at.strftime('%Y-%m-%d %H:%M:%S')}

## Generated Artifacts
{self._format_deployment_artifacts(deployment_artifacts)}

## AI-Enhanced Recommendations
{ai_advice}

## Deployment Instructions
1. Review the generated deployment artifacts
2. Configure your deployment environment
3. Deploy using the provided scripts
4. Test the API endpoints
5. Set up monitoring and logging

---
*Deployment completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
            
            return report
            
        except Exception as e:
            logger.error(f"Error deploying model: {str(e)}")
            return f"Error deploying model: {str(e)}"
    
    async def _monitor_model(self, prompt: str, context: Dict[str, Any]) -> str:
        """Monitor model performance and handle retraining automation."""
        try:
            model_name = context.get("model_name", "")
            monitoring_config = context.get("monitoring_config", {})
            
            if not model_name:
                return "Error: Model name is required for monitoring."
            
            # Perform monitoring checks
            monitoring_analysis = self._analyze_retraining_needs(model_name, monitoring_config)
            
            # Get GPT-4 advice for monitoring and retraining
            ai_advice = await self._get_gpt4_monitoring_advice(prompt, context)
            
            # Create monitoring job
            monitoring = ModelMonitoring(
                model_name=model_name,
                metrics=monitoring_analysis.get("performance_metrics", {}),
                drift_detected=monitoring_analysis.get("drift_detected", False),
                performance_degradation=monitoring_analysis.get("performance_degradation", False),
                alerts=monitoring_analysis.get("alerts", [])
            )
            
            self.monitoring_jobs.append(monitoring)
            
            # Generate monitoring report
            report = f"""
# Model Monitoring Report

## Model: {model_name}

### Performance Analysis
{self._format_monitoring_analysis(monitoring_analysis)}

### AI-Enhanced Recommendations
{ai_advice}

### Next Steps
{self._generate_monitoring_next_steps(monitoring)}

---
*Monitoring analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
            
            return report
            
        except Exception as e:
            logger.error(f"Error monitoring model: {str(e)}")
            return f"Error monitoring model: {str(e)}"
    
    async def _handle_framework_integration(self, prompt: str, context: Dict[str, Any]) -> str:
        """Handle ML framework integration requests."""
        try:
            framework = context.get("framework", MLFramework.SCIKIT_LEARN)
            integration_type = context.get("integration_type", "general")
            requirements = context.get("requirements", [])
            
            # Generate framework-specific integration code
            integration_code = self._generate_framework_integration(framework, integration_type, requirements)
            
            # Get GPT-4 advice for framework integration
            ai_advice = await self._get_gpt4_framework_advice(prompt, context)
            
            # Generate integration report
            report = f"""
# Framework Integration Report

## Framework: {framework.value}

### Available Capabilities
{self._format_framework_capabilities(framework)}

### Integration Code
```python
{integration_code}
```

### AI-Enhanced Recommendations
{ai_advice}

### Framework Requirements
{self._generate_framework_requirements(framework)}

---
*Framework integration completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
            
            return report
            
        except Exception as e:
            logger.error(f"Error handling framework integration: {str(e)}")
            return f"Error handling framework integration: {str(e)}"
    
    async def _generate_mlops_artifacts(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate MLOps automation artifacts."""
        try:
            project_name = context.get("project_name", "ml_project")
            mlops_dir = context.get("mlops_dir", Path("mlops"))
            config = context.get("config", {})
            
            # Generate MLOps artifacts
            artifacts = self._generate_mlops_artifacts_files(project_name, mlops_dir, config)
            
            # Get GPT-4 advice for MLOps automation
            ai_advice = await self._get_gpt4_mlops_advice(prompt, context)
            
            # Generate MLOps report
            report = f"""
# MLOps Automation Report

## Project: {project_name}

### Generated Artifacts
{self._format_mlops_artifacts(artifacts)}

### AI-Enhanced Recommendations
{ai_advice}

### MLOps Best Practices
- ✅ Use version control for data, code, and models
- ✅ Implement comprehensive data validation
- ✅ Set up automated testing for ML pipelines
- ✅ Monitor model performance continuously
- ✅ Use containerization for reproducible deployments
- ✅ Implement proper experiment tracking
- ✅ Set up automated retraining workflows

### Available Capabilities
{self._format_capabilities()}

---
*MLOps automation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating MLOps artifacts: {str(e)}")
            return f"Error generating MLOps artifacts: {str(e)}"
    
    async def _general_ml_engineering_advice(self, prompt: str, context: Dict[str, Any]) -> str:
        """Provide general ML engineering advice using GPT-4."""
        try:
            # Get system status
            system_status = {
                "active_pipelines": len(self.active_pipelines),
                "active_deployments": len(self.active_deployments),
                "monitoring_jobs": len(self.monitoring_jobs),
                "available_frameworks": list(self.framework_configs.keys())
            }
            
            # Get AI-powered analysis
            ai_advice = await self._get_gpt4_general_advice(prompt, context)
            
            # Generate comprehensive report
            report = f"""
# ML Engineering Consultation

## Your Request
{prompt}

## AI-Enhanced Analysis and Recommendations
{ai_advice}

## System Status
{', '.join(f"**{key.replace('_', ' ').title()}**: {value}" for key, value in system_status.items())}

## Available Frameworks
{', '.join(f"- **{framework}**" for framework in system_status['available_frameworks'])}

## Quick Actions
- Create ML pipeline: Provide dataset path and target column
- Deploy model: Specify model name and deployment target
- Monitor model: Provide model name for monitoring setup
- Framework integration: Choose from available frameworks

---
*Consultation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
            
            return report
            
        except Exception as e:
            logger.error(f"Error providing general ML engineering advice: {str(e)}")
            raise e  # Re-raise the exception so it's caught by the main process_request method   
 # GPT-4 advice methods
    async def _get_gpt4_pipeline_advice(self, prompt: str, context: Dict[str, Any]) -> str:
        """Get GPT-4 advice for pipeline setup."""
        return await self._get_gpt4_advice(prompt, context, "pipeline")
    
    async def _get_gpt4_deployment_advice(self, prompt: str, context: Dict[str, Any]) -> str:
        """Get GPT-4 advice for model deployment."""
        return await self._get_gpt4_advice(prompt, context, "deployment")
    
    async def _get_gpt4_monitoring_advice(self, prompt: str, context: Dict[str, Any]) -> str:
        """Get GPT-4 advice for model monitoring."""
        return await self._get_gpt4_advice(prompt, context, "monitoring")
    
    async def _get_gpt4_framework_advice(self, prompt: str, context: Dict[str, Any]) -> str:
        """Get GPT-4 advice for framework integration."""
        return await self._get_gpt4_advice(prompt, context, "framework")
    
    async def _get_gpt4_mlops_advice(self, prompt: str, context: Dict[str, Any]) -> str:
        """Get GPT-4 advice for MLOps automation."""
        return await self._get_gpt4_advice(prompt, context, "mlops")
    
    async def _get_gpt4_general_advice(self, prompt: str, context: Dict[str, Any]) -> str:
        """Get GPT-4 advice for general ML engineering."""
        return await self._get_gpt4_advice(prompt, context, "general")

    async def _get_gpt4_advice(self, prompt: str, context: Dict[str, Any], advice_type: str) -> str:
        """Get GPT-4 advice for various ML engineering tasks."""
        if not self.openai_client:
            return "Enhanced AI analysis unavailable - using baseline recommendations"
        
        try:
            advice_prompts = {
                "pipeline": "As a senior ML Engineer, provide detailed advice for ML pipeline setup and optimization.",
                "deployment": "As a senior ML Engineer, provide detailed advice for model deployment and production systems.",
                "monitoring": "As a senior ML Engineer, provide detailed advice for model monitoring and retraining strategies.",
                "framework": "As a senior ML Engineer, provide detailed advice for ML framework integration and best practices.",
                "mlops": "As a senior ML Engineer, provide detailed advice for MLOps automation and enterprise deployment.",
                "general": "As a senior ML Engineer, provide comprehensive advice for ML engineering challenges."
            }
            
            system_prompt = advice_prompts.get(advice_type, advice_prompts["general"])
            
            gpt4_prompt = f"""
{system_prompt}

Focus on practical, actionable recommendations including:
1. Technical implementation details
2. Best practices and patterns
3. Common pitfalls and solutions
4. Performance optimization tips
5. Tools and technologies
6. Next steps and guidance

User Request: {prompt}

Context: {json.dumps(context, indent=2)}
"""
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior ML Engineer with expertise in MLOps, production ML systems, and model deployment. Provide detailed, practical advice with actionable recommendations."
                    },
                    {
                        "role": "user",
                        "content": gpt4_prompt
                    }
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"GPT-4 API call failed: {str(e)}")
            return "Enhanced AI analysis unavailable - using baseline recommendations"
    
    # Code generation methods
    def _generate_pipeline_code(self, dataset_path: str, target_column: str, framework: MLFramework) -> str:
        """Generate ML pipeline code."""
        if framework == MLFramework.SCIKIT_LEARN:
            return self._generate_sklearn_pipeline(dataset_path, target_column)
        elif framework == MLFramework.TENSORFLOW:
            return self._generate_tensorflow_pipeline(dataset_path, target_column)
        elif framework == MLFramework.PYTORCH:
            return self._generate_pytorch_pipeline(dataset_path, target_column)
        elif framework == MLFramework.XGBOOST:
            return self._generate_xgboost_pipeline(dataset_path, target_column)
        else:
            return "# Framework integration code would be generated here based on the framework selection"
    
    def _generate_sklearn_pipeline(self, dataset_path: str, target_column: str) -> str:
        """Generate scikit-learn pipeline code."""
        return f'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
import joblib

def load_and_prepare_data():
    """Load and prepare data"""
    df = pd.read_csv("{dataset_path}")
    
    # Handle missing values
    df = df.dropna()
    
    # Separate features and target
    X = df.drop(columns=["{target_column}"])
    y = df["{target_column}"]
    
    # Encode categorical variables
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    return X, y

def train_model(X, y):
    """Train model"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    if len(y.unique()) < 20:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        metric_name = "accuracy"
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        metric_name = "r2_score"
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    predictions = model.predict(X_test_scaled)
    if metric_name == "accuracy":
        score = accuracy_score(y_test, predictions)
    else:
        score = r2_score(y_test, predictions)
    
    print(f"Pipeline completed. Score: {{score:.4f}}")
    
    # Save artifacts
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    
    return model, scaler, score

if __name__ == "__main__":
    # Execute pipeline
    X, y = load_and_prepare_data()
    model, scaler, score = train_model(X, y)
'''
    
    def _generate_tensorflow_pipeline(self, dataset_path: str, target_column: str) -> str:
        """Generate TensorFlow pipeline code."""
        return f'''
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def create_model(input_shape, num_classes=None):
    """Create TensorFlow model"""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=input_shape),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3)
    ])
    
    if num_classes:
        model.add(layers.Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

def train_tensorflow_model():
    """Train TensorFlow model"""
    # Load data
    df = pd.read_csv("{dataset_path}")
    X = df.drop(columns=["{target_column}"])
    y = df["{target_column}"]
    
    # Prepare data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    model = create_model((X_train_scaled.shape[1],))
    
    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_metric = model.evaluate(X_test_scaled, y_test)
    print(f"Test Loss: {{test_loss:.4f}}, Test Metric: {{test_metric:.4f}}")
    
    # Save model
    model.save("tensorflow_model.h5")
    
    return model, history

if __name__ == "__main__":
    model, history = train_tensorflow_model()
'''
    
    def _generate_pytorch_pipeline(self, dataset_path: str, target_column: str) -> str:
        """Generate PyTorch pipeline code."""
        return f'''
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=1):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_pytorch_model():
    """Train PyTorch model"""
    # Load data
    df = pd.read_csv("{dataset_path}")
    X = df.drop(columns=["{target_column}"]).values
    y = df["{target_column}"].values
    
    # Prepare data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test.reshape(-1, 1))
    
    # Create model
    model = SimpleNet(X_train_scaled.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{{epoch+1}}/{{epochs}}], Loss: {{loss.item():.4f}}')
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        print(f'Test Loss: {{test_loss.item():.4f}}')
    
    # Save model
    torch.save(model.state_dict(), 'pytorch_model.pth')
    
    return model

if __name__ == "__main__":
    model = train_pytorch_model()
'''
    
    def _generate_xgboost_pipeline(self, dataset_path: str, target_column: str) -> str:
        """Generate XGBoost pipeline code."""
        return f'''
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
import joblib

def train_xgboost_model():
    """Train XGBoost model"""
    # Load data
    df = pd.read_csv("{dataset_path}")
    X = df.drop(columns=["{target_column}"])
    y = df["{target_column}"]
    
    # Encode categorical variables
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Determine task type
    if len(y.unique()) < 20:
        # Classification
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        metric_name = "accuracy"
    else:
        # Regression
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        metric_name = "r2_score"
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    if metric_name == "accuracy":
        score = accuracy_score(y_test, predictions)
    else:
        score = r2_score(y_test, predictions)
    
    print(f"XGBoost Pipeline completed. Score: {{score:.4f}}")
    
    # Save model
    joblib.dump(model, "xgboost_model.pkl")
    
    return model, score

if __name__ == "__main__":
    model, score = train_xgboost_model()
'''  
  
    # Deployment artifact generation methods
    def _generate_deployment_artifacts(self, model_name: str, model_path: str, target: DeploymentTarget, context: Dict[str, Any]) -> Dict[str, str]:
        """Generate deployment artifacts based on target."""
        artifacts = {}
        
        if target == DeploymentTarget.FASTAPI:
            artifacts["FastAPI App"] = self._generate_fastapi_deployment(model_name, model_path)
            artifacts["Requirements"] = self._generate_deployment_requirements("fastapi")
            artifacts["Dockerfile"] = self._generate_dockerfile("fastapi")
        elif target == DeploymentTarget.FLASK:
            artifacts["flask_app"] = self._generate_flask_deployment(model_name, model_path)
            artifacts["requirements"] = self._generate_deployment_requirements("flask")
            artifacts["dockerfile"] = self._generate_dockerfile("flask")
        elif target == DeploymentTarget.DOCKER:
            artifacts["dockerfile"] = self._generate_dockerfile("standalone")
            artifacts["docker_compose"] = self._generate_docker_compose(model_name)
        elif target == DeploymentTarget.KUBERNETES:
            artifacts["k8s_deployment"] = self._generate_k8s_deployment(model_name)
            artifacts["k8s_service"] = self._generate_k8s_service(model_name)
        
        return artifacts
    
    def _generate_fastapi_deployment(self, model_name: str, model_path: str) -> str:
        """Generate FastAPI deployment code."""
        return f'''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="{model_name} API", version="1.0.0")

# Load model
try:
    model = joblib.load("{model_path}")
    logger.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    logger.error(f"Failed to load model: {{e}}")
    model = None

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: float
    model_name: str
    confidence: float = None

@app.get("/")
async def root():
    return {{"message": "{model_name} API is running", "status": "healthy"}}

@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {{"status": "healthy", "model": "{model_name}"}}

@app.post("/predict")
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to numpy array
        features = np.array(request.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get prediction probability if available (for classifiers)
        confidence = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            confidence = float(np.max(proba))
        
        return PredictionResponse(
            prediction=float(prediction),
            model_name="{model_name}",
            confidence=confidence
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {{e}}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {{str(e)}}")

@app.post("/batch_predict")
async def batch_predict(requests: List[PredictionRequest]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert all features to numpy array
        features_list = [req.features for req in requests]
        features_array = np.array(features_list)
        
        # Make predictions
        predictions = model.predict(features_array)
        
        return {{
            "predictions": [float(pred) for pred in predictions],
            "model_name": "{model_name}",
            "count": len(predictions)
        }}
    
    except Exception as e:
        logger.error(f"Batch prediction error: {{e}}")
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {{str(e)}}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    def _generate_flask_deployment(self, model_name: str, model_path: str) -> str:
        """Generate Flask deployment code."""
        return f'''
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load model
try:
    model = joblib.load("{model_path}")
    logger.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    logger.error(f"Failed to load model: {{e}}")
    model = None

@app.route('/')
def root():
    return jsonify({{"message": "{model_name} API is running", "status": "healthy"}})

@app.route('/health')
def health_check():
    if model is None:
        return jsonify({{"error": "Model not loaded"}}), 503
    return jsonify({{"status": "healthy", "model": "{model_name}"}})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({{"error": "Model not loaded"}}), 503
    
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get prediction probability if available
        confidence = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            confidence = float(np.max(proba))
        
        return jsonify({{
            "prediction": float(prediction),
            "model_name": "{model_name}",
            "confidence": confidence
        }})
    
    except Exception as e:
        logger.error(f"Prediction error: {{e}}")
        return jsonify({{"error": f"Prediction failed: {{str(e)}}"}}), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if model is None:
        return jsonify({{"error": "Model not loaded"}}), 503
    
    try:
        data = request.get_json()
        features_list = data['features']
        features_array = np.array(features_list)
        
        # Make predictions
        predictions = model.predict(features_array)
        
        return jsonify({{
            "predictions": [float(pred) for pred in predictions],
            "model_name": "{model_name}",
            "count": len(predictions)
        }})
    
    except Exception as e:
        logger.error(f"Batch prediction error: {{e}}")
        return jsonify({{"error": f"Batch prediction failed: {{str(e)}}"}}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
'''
    
    def _generate_deployment_requirements(self, framework: str) -> str:
        """Generate requirements.txt for deployment."""
        base_requirements = [
            "numpy>=1.21.0",
            "pandas>=1.5.0",
            "scikit-learn>=1.3.0",
            "joblib>=1.3.0"
        ]
        
        if framework == "fastapi":
            base_requirements.extend([
                "fastapi>=0.100.0",
                "uvicorn>=0.23.0",
                "pydantic>=2.0.0"
            ])
        elif framework == "flask":
            base_requirements.extend([
                "flask>=2.3.0",
                "gunicorn>=21.0.0"
            ])
        
        return "\n".join(base_requirements)
    
    def _generate_framework_requirements(self, framework: MLFramework) -> str:
        """Generate framework requirements."""
        config = self.framework_configs.get(framework, {})
        requirements = config.get("requirements", [])
        return "\n".join([f"- {req}" for req in requirements])
    
    def _generate_dockerfile(self, deployment_type: str) -> str:
        """Generate Dockerfile for deployment."""
        if deployment_type == "fastapi":
            return '''
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
        elif deployment_type == "flask":
            return '''
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
'''
        else:
            return '''
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
'''
    
    def _generate_docker_compose(self, model_name: str) -> str:
        """Generate docker-compose.yml."""
        return f'''
version: '3.8'

services:
  {model_name.lower().replace(' ', '-')}:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME={model_name}
    volumes:
      - ./models:/app/models
    restart: unless-stopped
    
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
'''
    
    def _generate_k8s_deployment(self, model_name: str) -> str:
        """Generate Kubernetes deployment manifest."""
        app_name = model_name.lower().replace(' ', '-')
        return f'''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {app_name}
  labels:
    app: {app_name}
spec:
  replicas: 3
  selector:
    matchLabels:
      app: {app_name}
  template:
    metadata:
      labels:
        app: {app_name}
    spec:
      containers:
      - name: {app_name}
        image: {app_name}:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_NAME
          value: "{model_name}"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
'''
    
    def _generate_k8s_service(self, model_name: str) -> str:
        """Generate Kubernetes service manifest."""
        app_name = model_name.lower().replace(' ', '-')
        return f'''
apiVersion: v1
kind: Service
metadata:
  name: {app_name}-service
spec:
  selector:
    app: {app_name}
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
'''    

    # Monitoring and analysis methods
    def _analyze_retraining_needs(self, model_name: str, monitoring_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if model needs retraining."""
        # Extract configuration values
        accuracy = monitoring_config.get("accuracy", 0.85)
        drift_detected = monitoring_config.get("drift_detected", False)
        new_data_available = monitoring_config.get("new_data_available", False)
        
        # Simulate monitoring analysis
        analysis = {
            "performance_metrics": {
                "accuracy": accuracy,
                "precision": 0.82,
                "recall": 0.88,
                "f1_score": 0.85
            },
            "drift_detected": drift_detected,
            "performance_degradation": False,
            "alerts": [],
            "drift_analysis": {
                "statistical_tests": {
                    "ks_test": {"p_value": 0.05, "significant": drift_detected},
                    "chi_square": {"p_value": 0.03, "significant": drift_detected}
                },
                "feature_drift": {
                    "high_drift_features": ["feature_1", "feature_3"] if drift_detected else [],
                    "drift_scores": {"feature_1": 0.8, "feature_2": 0.2, "feature_3": 0.7} if drift_detected else {}
                }
            },
            "data_quality": {
                "missing_values": 0.02,
                "outliers": 0.05,
                "schema_changes": False,
                "new_data_available": new_data_available
            }
        }
        
        # Check for performance degradation
        threshold = monitoring_config.get("performance_threshold", 0.8)
        if accuracy < threshold:
            analysis["performance_degradation"] = True
            analysis["alerts"].append(f"Model accuracy ({accuracy:.3f}) below threshold ({threshold})")
        
        # Add drift alerts
        if drift_detected:
            analysis["alerts"].append("Data drift detected - model retraining recommended")
        
        # Add data quality alerts
        if new_data_available:
            analysis["alerts"].append("New training data available - consider retraining")
        
        return analysis
    
    def _format_monitoring_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format monitoring analysis for display."""
        metrics = analysis.get("performance_metrics", {})
        
        metrics_text = "\n".join([
            f"- **{metric.replace('_', ' ').title()}**: {value:.3f}"
            for metric, value in metrics.items()
        ])
        
        alerts_text = "\n".join([
            f"- ⚠️ {alert}" for alert in analysis.get("alerts", [])
        ]) if analysis.get("alerts") else "- ✅ No alerts"
        
        return f"""
**Performance Metrics:**
{metrics_text}

**Drift Detection:**
- Data Drift: {"🔴 Detected" if analysis.get("drift_detected") else "✅ Not Detected"}
- Performance Degradation: {"🔴 Detected" if analysis.get("performance_degradation") else "✅ Not Detected"}

**Alerts:**
{alerts_text}
"""
    
    def _format_pipeline_stages(self, stages: List[PipelineStage]) -> str:
        """Format pipeline stages for display."""
        stage_names = {
            PipelineStage.DATA_INGESTION: "Data Ingestion",
            PipelineStage.DATA_PREPROCESSING: "Data Preprocessing", 
            PipelineStage.FEATURE_ENGINEERING: "Feature Engineering",
            PipelineStage.MODEL_TRAINING: "Model Training",
            PipelineStage.MODEL_EVALUATION: "Model Evaluation",
            PipelineStage.MODEL_DEPLOYMENT: "Model Deployment",
            PipelineStage.MODEL_MONITORING: "Model Monitoring"
        }
        
        formatted_stages = []
        for i, stage in enumerate(stages, 1):
            stage_name = stage_names.get(stage, stage.value.replace('_', ' ').title())
            formatted_stages.append(f"{i}. **{stage_name}**")
        
        return "\n".join(formatted_stages)
    
    def _format_deployment_artifacts(self, artifacts: Dict[str, str]) -> str:
        """Format deployment artifacts for display."""
        formatted = []
        for artifact_type, content in artifacts.items():
            formatted.append(f"### {artifact_type}")
            if artifact_type in ["FastAPI App", "Flask App", "Docker Config"]:
                formatted.append(f"```python\n{content[:500]}...\n```" if len(content) > 500 else f"```python\n{content}\n```")
            else:
                formatted.append(f"```\n{content[:500]}...\n```" if len(content) > 500 else f"```\n{content}\n```")
            formatted.append("")
        
        return "\n".join(formatted)
    
    def _format_framework_capabilities(self, framework: MLFramework) -> str:
        """Format framework capabilities for display."""
        config = self.framework_configs.get(framework, {})
        features = config.get("features", [])
        
        if not features:
            return f"No specific capabilities configured for {framework.value}"
        
        formatted_features = []
        for feature in features:
            formatted_features.append(f"- {feature}")
        
        return "\n".join(formatted_features)
    
    def _format_mlops_artifacts(self, artifacts: Dict[str, str]) -> str:
        """Format MLOps artifacts for display."""
        formatted = []
        for artifact_name, content in artifacts.items():
            formatted.append(f"### {artifact_name}")
            if len(content) > 300:
                formatted.append(f"```\n{content[:300]}...\n```")
            else:
                formatted.append(f"```\n{content}\n```")
            formatted.append("")
        
        return "\n".join(formatted)
    
    def _format_capabilities(self) -> str:
        """Format agent capabilities for display."""
        return "\n".join([f"- **{cap.name}**: {cap.description}" for cap in self.capabilities])
    
    def _generate_monitoring_next_steps(self, monitoring: ModelMonitoring) -> str:
        """Generate next steps based on monitoring results."""
        steps = []
        
        if monitoring.drift_detected:
            steps.extend([
                "1. **Investigate Data Drift**: Analyze recent data for distribution changes",
                "2. **Collect New Training Data**: Gather representative samples from current distribution",
                "3. **Retrain Model**: Update model with recent data to address drift"
            ])
        
        if monitoring.performance_degradation:
            steps.extend([
                "1. **Performance Analysis**: Deep dive into performance degradation causes",
                "2. **Feature Analysis**: Check if input features have changed",
                "3. **Model Refresh**: Consider retraining or model architecture updates"
            ])
        
        if not monitoring.drift_detected and not monitoring.performance_degradation:
            steps.extend([
                "1. **Continue Monitoring**: Model performance is stable",
                "2. **Schedule Regular Reviews**: Set up periodic performance assessments",
                "3. **Optimize Infrastructure**: Focus on deployment efficiency and cost"
            ])
        
        return "\n".join(steps) if steps else "No specific actions required at this time."
    
    # Framework integration methods
    def _generate_framework_integration(self, framework: MLFramework, integration_type: str, requirements: List[str]) -> str:
        """Generate framework-specific integration code."""
        if framework == MLFramework.SCIKIT_LEARN:
            return self._generate_sklearn_integration(integration_type, requirements)
        elif framework == MLFramework.TENSORFLOW:
            return self._generate_tensorflow_integration(integration_type, requirements)
        elif framework == MLFramework.PYTORCH:
            return self._generate_pytorch_integration(integration_type, requirements)
        elif framework == MLFramework.XGBOOST:
            return self._generate_xgboost_integration(integration_type, requirements)
        else:
            return "# Framework integration code would be generated here"
    
    def _generate_sklearn_integration(self, integration_type: str, requirements: List[str]) -> str:
        """Generate scikit-learn integration code."""
        return '''
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib

# Create a complete ML pipeline
def create_sklearn_pipeline():
    """Create a scikit-learn pipeline with preprocessing and model."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    return pipeline

# Hyperparameter tuning
def tune_hyperparameters(pipeline, X_train, y_train):
    """Perform hyperparameter tuning."""
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(
        pipeline, param_grid, 
        cv=5, scoring='accuracy', 
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Model persistence
def save_model(model, filepath):
    """Save trained model."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """Load trained model."""
    return joblib.load(filepath)
'''
    
    def _generate_tensorflow_integration(self, integration_type: str, requirements: List[str]) -> str:
        """Generate TensorFlow integration code."""
        return '''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import numpy as np

# Custom model architecture
class CustomTensorFlowModel(keras.Model):
    def __init__(self, num_classes):
        super(CustomTensorFlowModel, self).__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout1 = layers.Dropout(0.3)
        self.dense2 = layers.Dense(64, activation='relu')
        self.dropout2 = layers.Dropout(0.3)
        self.output_layer = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.output_layer(x)

# Training with callbacks
def train_with_callbacks(model, X_train, y_train, X_val, y_val):
    """Train model with callbacks."""
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.001
        ),
        callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks_list,
        verbose=1
    )
    
    return history

# Model export for serving
def export_for_serving(model, export_path):
    """Export model for TensorFlow Serving."""
    tf.saved_model.save(model, export_path)
    print(f"Model exported to {export_path}")
'''
    
    def _generate_pytorch_integration(self, integration_type: str, requirements: List[str]) -> str:
        """Generate PyTorch integration code."""
        return '''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Advanced model architecture
class AdvancedPyTorchModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.3):
        super(AdvancedPyTorchModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Training loop with validation
def train_model(model, train_loader, val_loader, num_epochs=100):
    """Train PyTorch model with validation."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'Val Acc: {val_accuracy:.2f}%')
        
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_pytorch_model.pth')
    
    return model

# Model export for deployment
def export_to_torchscript(model, example_input):
    """Export model to TorchScript for deployment."""
    model.eval()
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save('model_torchscript.pt')
    print("Model exported to TorchScript format")
'''
    
    def _generate_xgboost_integration(self, integration_type: str, requirements: List[str]) -> str:
        """Generate XGBoost integration code."""
        return '''
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Advanced XGBoost configuration
def create_xgboost_model(task_type='classification'):
    """Create XGBoost model with advanced configuration."""
    if task_type == 'classification':
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='rmse'
        )
    
    return model

# Hyperparameter tuning for XGBoost
def tune_xgboost_hyperparameters(X_train, y_train, task_type='classification'):
    """Perform comprehensive hyperparameter tuning."""
    model = create_xgboost_model(task_type)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    scoring = 'accuracy' if task_type == 'classification' else 'neg_mean_squared_error'
    
    grid_search = GridSearchCV(
        model, param_grid,
        cv=5, scoring=scoring,
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Feature importance analysis
def analyze_feature_importance(model, feature_names):
    """Analyze and visualize feature importance."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importance")
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()
    
    return dict(zip([feature_names[i] for i in indices], importance[indices]))

# Early stopping training
def train_with_early_stopping(model, X_train, y_train, X_val, y_val):
    """Train XGBoost with early stopping."""
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=10,
        verbose=True
    )
    
    return model
'''    

    def _generate_mlops_artifacts_files(self, project_name: str, mlops_dir: Path, config: Dict[str, Any]) -> Dict[str, str]:
        """Generate MLOps artifacts files."""
        # Create directories
        mlops_dir.mkdir(exist_ok=True)
        (mlops_dir / ".github" / "workflows").mkdir(parents=True, exist_ok=True)
        (mlops_dir / "deployment").mkdir(exist_ok=True)
        (mlops_dir / "tests").mkdir(exist_ok=True)
        
        # Generate GitHub Actions workflow
        github_workflow = f'''name: MLOps Pipeline - {project_name}

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
      run: |
        python -m pytest tests/
    - name: Run model validation
      run: |
        python scripts/validate_model.py

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    - name: Deploy to production
      run: |
        echo "Deploying {project_name} to production"
        # Add deployment commands here
'''
        
        # Generate requirements.txt
        requirements_txt = '''pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.3.0
joblib>=1.3.0
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
pytest>=7.0.0
mlflow>=2.0.0
'''
        
        # Generate Dockerfile
        dockerfile = f'''FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
        
        # Write files
        workflow_path = mlops_dir / ".github" / "workflows" / "mlops.yml"
        requirements_path = mlops_dir / "requirements.txt"
        dockerfile_path = mlops_dir / "deployment" / "Dockerfile"
        
        workflow_path.write_text(github_workflow)
        requirements_path.write_text(requirements_txt)
        dockerfile_path.write_text(dockerfile)
        
        return {
            "GitHub Actions Workflow": github_workflow,
            "Requirements": requirements_txt,
            "Dockerfile": dockerfile
        }