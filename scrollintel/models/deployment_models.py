"""
Deployment automation models for the automated code generation system.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class CloudProvider(str, Enum):
    """Supported cloud providers."""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"


class DeploymentEnvironment(str, Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class ContainerConfig(BaseModel):
    """Docker container configuration."""
    base_image: str = Field(..., description="Base Docker image")
    dockerfile_content: str = Field(..., description="Generated Dockerfile content")
    build_args: Dict[str, str] = Field(default_factory=dict, description="Docker build arguments")
    environment_vars: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    exposed_ports: List[int] = Field(default_factory=list, description="Exposed container ports")
    volumes: List[str] = Field(default_factory=list, description="Volume mounts")
    health_check: Optional[Dict[str, Any]] = Field(None, description="Health check configuration")
    resource_limits: Dict[str, str] = Field(default_factory=dict, description="Resource limits")


class InfrastructureCode(BaseModel):
    """Infrastructure as Code configuration."""
    provider: CloudProvider = Field(..., description="Cloud provider")
    template_type: str = Field(..., description="Template type (terraform, cloudformation, etc.)")
    template_content: str = Field(..., description="Generated IaC template content")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Template variables")
    outputs: Dict[str, str] = Field(default_factory=dict, description="Template outputs")
    dependencies: List[str] = Field(default_factory=list, description="Resource dependencies")


class CICDPipeline(BaseModel):
    """CI/CD pipeline configuration."""
    platform: str = Field(..., description="CI/CD platform (github, gitlab, jenkins)")
    pipeline_content: str = Field(..., description="Generated pipeline configuration")
    stages: List[str] = Field(default_factory=list, description="Pipeline stages")
    triggers: List[str] = Field(default_factory=list, description="Pipeline triggers")
    environment_configs: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Environment-specific configurations"
    )
    secrets: List[str] = Field(default_factory=list, description="Required secrets")


class DeploymentConfig(BaseModel):
    """Complete deployment configuration."""
    id: str = Field(..., description="Unique deployment configuration ID")
    name: str = Field(..., description="Deployment configuration name")
    application_id: str = Field(..., description="Associated application ID")
    environment: DeploymentEnvironment = Field(..., description="Target environment")
    cloud_provider: CloudProvider = Field(..., description="Target cloud provider")
    
    # Container configuration
    container_config: ContainerConfig = Field(..., description="Container configuration")
    
    # Infrastructure configuration
    infrastructure_code: InfrastructureCode = Field(..., description="Infrastructure as code")
    
    # CI/CD configuration
    cicd_pipeline: CICDPipeline = Field(..., description="CI/CD pipeline configuration")
    
    # Deployment settings
    auto_scaling: Dict[str, Any] = Field(default_factory=dict, description="Auto-scaling configuration")
    load_balancing: Dict[str, Any] = Field(default_factory=dict, description="Load balancing configuration")
    monitoring: Dict[str, Any] = Field(default_factory=dict, description="Monitoring configuration")
    security: Dict[str, Any] = Field(default_factory=dict, description="Security configuration")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = Field(..., description="User who created the configuration")


class DeploymentResult(BaseModel):
    """Deployment execution result."""
    deployment_id: str = Field(..., description="Deployment ID")
    status: str = Field(..., description="Deployment status")
    environment: DeploymentEnvironment = Field(..., description="Target environment")
    cloud_provider: CloudProvider = Field(..., description="Cloud provider")
    
    # Deployment details
    container_image: Optional[str] = Field(None, description="Built container image")
    infrastructure_resources: List[Dict[str, Any]] = Field(
        default_factory=list, description="Created infrastructure resources"
    )
    endpoints: List[str] = Field(default_factory=list, description="Application endpoints")
    
    # Execution metadata
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None)
    logs: List[str] = Field(default_factory=list, description="Deployment logs")
    errors: List[str] = Field(default_factory=list, description="Deployment errors")


class DeploymentTemplate(BaseModel):
    """Reusable deployment template."""
    id: str = Field(..., description="Template ID")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    cloud_provider: CloudProvider = Field(..., description="Target cloud provider")
    
    # Template content
    dockerfile_template: str = Field(..., description="Dockerfile template")
    infrastructure_template: str = Field(..., description="Infrastructure template")
    cicd_template: str = Field(..., description="CI/CD pipeline template")
    
    # Template variables
    variables: Dict[str, Any] = Field(default_factory=dict, description="Template variables")
    
    # Metadata
    version: str = Field(default="1.0.0", description="Template version")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DeploymentValidation(BaseModel):
    """Deployment validation result."""
    is_valid: bool = Field(..., description="Whether deployment configuration is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    recommendations: List[str] = Field(default_factory=list, description="Optimization recommendations")
    estimated_cost: Optional[float] = Field(None, description="Estimated monthly cost")
    security_score: Optional[int] = Field(None, description="Security score (0-100)")