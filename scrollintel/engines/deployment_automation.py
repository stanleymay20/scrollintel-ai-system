"""
Deployment automation engine for generating and managing application deployments.
"""

import json
import yaml
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..models.deployment_models import (
    DeploymentConfig, ContainerConfig, InfrastructureCode, CICDPipeline,
    DeploymentResult, DeploymentTemplate, DeploymentValidation,
    CloudProvider, DeploymentEnvironment
)
from ..models.code_generation_models import GeneratedApplication


class ContainerBuilder:
    """Builds Docker containers with optimization."""
    
    def __init__(self):
        self.base_images = {
            "python": "python:3.11-slim",
            "node": "node:18-alpine",
            "java": "openjdk:17-jre-slim",
            "go": "golang:1.21-alpine",
            "rust": "rust:1.75-slim"
        }
    
    def generate_dockerfile(self, application: GeneratedApplication) -> str:
        """Generate optimized Dockerfile for the application."""
        language = self._detect_language(application)
        base_image = self.base_images.get(language, "ubuntu:22.04")
        
        dockerfile_content = f"""# Generated Dockerfile for {application.name}
FROM {base_image}

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

"""
        
        # Add language-specific setup
        if language == "python":
            dockerfile_content += self._generate_python_dockerfile_content(application)
        elif language == "node":
            dockerfile_content += self._generate_node_dockerfile_content(application)
        elif language == "java":
            dockerfile_content += self._generate_java_dockerfile_content(application)
        
        # Add language-specific health check and CMD
        if language == "node":
            port = 3000
            cmd = '["node", "app.js"]'
        elif language == "java":
            port = 8080
            cmd = '["java", "-jar", "app.jar"]'
        else:  # Python and others
            port = 8000
            cmd = '["python", "main.py"]'
        
        dockerfile_content += f"""
# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{port}/health || exit 1

# Run the application
CMD {cmd}
"""
        
        return dockerfile_content
    
    def _detect_language(self, application: GeneratedApplication) -> str:
        """Detect the primary programming language of the application."""
        # Simple detection based on file extensions and component names
        for component in application.code_components:
            if component.language:
                lang = component.language.lower()
                if lang in ["javascript", "js", "node"]:
                    return "node"
                elif lang in ["python", "py"]:
                    return "python"
                elif lang in ["java"]:
                    return "java"
                elif lang in ["go", "golang"]:
                    return "go"
                elif lang in ["rust"]:
                    return "rust"
                else:
                    return lang
            
            # Check file names for language hints
            if component.name.endswith('.js') or component.name == 'package.json':
                return "node"
            elif component.name.endswith('.py') or component.name == 'requirements.txt':
                return "python"
            elif component.name.endswith('.java'):
                return "java"
            elif component.name.endswith('.go'):
                return "go"
            elif component.name.endswith('.rs'):
                return "rust"
        
        return "python"  # Default
    
    def _generate_python_dockerfile_content(self, application: GeneratedApplication) -> str:
        """Generate Python-specific Dockerfile content."""
        return """# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

"""
    
    def _generate_node_dockerfile_content(self, application: GeneratedApplication) -> str:
        """Generate Node.js-specific Dockerfile content."""
        return """# Copy package files and install dependencies
COPY package*.json ./
RUN npm ci --only=production

# Copy application code
COPY . .

# Create non-root user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001
USER nextjs

# Expose port
EXPOSE 3000

"""
    
    def _generate_java_dockerfile_content(self, application: GeneratedApplication) -> str:
        """Generate Java-specific Dockerfile content."""
        return """# Copy JAR file
COPY target/*.jar app.jar

# Create non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Expose port
EXPOSE 8080

"""
    
    def build_container_config(self, application: GeneratedApplication) -> ContainerConfig:
        """Build complete container configuration."""
        language = self._detect_language(application)
        
        # Set language-specific defaults
        if language == "node":
            port = 3000
            health_check_port = 3000
            cmd = ["node", "app.js"]
        elif language == "java":
            port = 8080
            health_check_port = 8080
            cmd = ["java", "-jar", "app.jar"]
        else:  # Python and others
            port = 8000
            health_check_port = 8000
            cmd = ["python", "main.py"]
        
        return ContainerConfig(
            base_image=self.base_images.get(language, "ubuntu:22.04"),
            dockerfile_content=self.generate_dockerfile(application),
            build_args={
                "APP_NAME": application.name,
                "BUILD_DATE": datetime.utcnow().isoformat()
            },
            environment_vars={
                "APP_ENV": "production",
                "PORT": str(port)
            },
            exposed_ports=[port],
            volumes=["/app/data"],
            health_check={
                "test": ["CMD", "curl", "-f", f"http://localhost:{health_check_port}/health"],
                "interval": "30s",
                "timeout": "3s",
                "retries": 3
            },
            resource_limits={
                "memory": "512m",
                "cpu": "0.5"
            }
        )


class InfrastructureGenerator:
    """Generates infrastructure-as-code templates."""
    
    def generate_terraform_aws(self, application: GeneratedApplication, config: Dict[str, Any]) -> str:
        """Generate Terraform configuration for AWS."""
        return f"""# Generated Terraform configuration for {application.name}
terraform {{
  required_version = ">= 1.0"
  required_providers {{
    aws = {{
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }}
  }}
}}

provider "aws" {{
  region = var.aws_region
}}

# Variables
variable "aws_region" {{
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}}

variable "environment" {{
  description = "Environment name"
  type        = string
  default     = "production"
}}

# VPC
resource "aws_vpc" "main" {{
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {{
    Name        = "${{var.environment}}-{application.name}-vpc"
    Environment = var.environment
  }}
}}

# Internet Gateway
resource "aws_internet_gateway" "main" {{
  vpc_id = aws_vpc.main.id

  tags = {{
    Name        = "${{var.environment}}-{application.name}-igw"
    Environment = var.environment
  }}
}}

# Subnets
resource "aws_subnet" "public" {{
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${{count.index + 1}}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  map_public_ip_on_launch = true

  tags = {{
    Name        = "${{var.environment}}-{application.name}-public-${{count.index + 1}}"
    Environment = var.environment
  }}
}}

# Route Table
resource "aws_route_table" "public" {{
  vpc_id = aws_vpc.main.id

  route {{
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }}

  tags = {{
    Name        = "${{var.environment}}-{application.name}-public-rt"
    Environment = var.environment
  }}
}}

resource "aws_route_table_association" "public" {{
  count          = length(aws_subnet.public)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}}

# Security Group
resource "aws_security_group" "app" {{
  name_prefix = "${{var.environment}}-{application.name}-"
  vpc_id      = aws_vpc.main.id

  ingress {{
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }}

  ingress {{
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }}

  egress {{
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }}

  tags = {{
    Name        = "${{var.environment}}-{application.name}-sg"
    Environment = var.environment
  }}
}}

# ECS Cluster
resource "aws_ecs_cluster" "main" {{
  name = "${{var.environment}}-{application.name}"

  setting {{
    name  = "containerInsights"
    value = "enabled"
  }}

  tags = {{
    Name        = "${{var.environment}}-{application.name}-cluster"
    Environment = var.environment
  }}
}}

# Application Load Balancer
resource "aws_lb" "main" {{
  name               = "${{var.environment}}-{application.name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.app.id]
  subnets            = aws_subnet.public[*].id

  enable_deletion_protection = false

  tags = {{
    Name        = "${{var.environment}}-{application.name}-alb"
    Environment = var.environment
  }}
}}

# Outputs
output "vpc_id" {{
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}}

output "load_balancer_dns" {{
  description = "DNS name of the load balancer"
  value       = aws_lb.main.dns_name
}}

output "ecs_cluster_name" {{
  description = "Name of the ECS cluster"
  value       = aws_ecs_cluster.main.name
}}

# Data sources
data "aws_availability_zones" "available" {{
  state = "available"
}}
"""
    
    def generate_cloudformation_aws(self, application: GeneratedApplication, config: Dict[str, Any]) -> str:
        """Generate CloudFormation template for AWS."""
        template = {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": f"CloudFormation template for {application.name}",
            "Parameters": {
                "Environment": {
                    "Type": "String",
                    "Default": "production",
                    "Description": "Environment name"
                },
                "InstanceType": {
                    "Type": "String",
                    "Default": "t3.micro",
                    "Description": "EC2 instance type"
                }
            },
            "Resources": {
                "VPC": {
                    "Type": "AWS::EC2::VPC",
                    "Properties": {
                        "CidrBlock": "10.0.0.0/16",
                        "EnableDnsHostnames": True,
                        "EnableDnsSupport": True,
                        "Tags": [
                            {
                                "Key": "Name",
                                "Value": {"Fn::Sub": "${Environment}-" + application.name + "-vpc"}
                            }
                        ]
                    }
                },
                "InternetGateway": {
                    "Type": "AWS::EC2::InternetGateway",
                    "Properties": {
                        "Tags": [
                            {
                                "Key": "Name",
                                "Value": {"Fn::Sub": "${Environment}-" + application.name + "-igw"}
                            }
                        ]
                    }
                },
                "AttachGateway": {
                    "Type": "AWS::EC2::VPCGatewayAttachment",
                    "Properties": {
                        "VpcId": {"Ref": "VPC"},
                        "InternetGatewayId": {"Ref": "InternetGateway"}
                    }
                }
            },
            "Outputs": {
                "VPCId": {
                    "Description": "VPC ID",
                    "Value": {"Ref": "VPC"},
                    "Export": {
                        "Name": {"Fn::Sub": "${AWS::StackName}-VPC-ID"}
                    }
                }
            }
        }
        
        return json.dumps(template, indent=2)
    
    def generate_azure_arm(self, application: GeneratedApplication, config: Dict[str, Any]) -> str:
        """Generate Azure ARM template."""
        template = {
            "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
            "contentVersion": "1.0.0.0",
            "parameters": {
                "appName": {
                    "type": "string",
                    "defaultValue": application.name,
                    "metadata": {
                        "description": "Name of the application"
                    }
                },
                "environment": {
                    "type": "string",
                    "defaultValue": "production",
                    "metadata": {
                        "description": "Environment name"
                    }
                }
            },
            "variables": {
                "resourceGroupName": "[concat(parameters('environment'), '-', parameters('appName'), '-rg')]",
                "appServicePlanName": "[concat(parameters('environment'), '-', parameters('appName'), '-plan')]",
                "webAppName": "[concat(parameters('environment'), '-', parameters('appName'), '-app')]"
            },
            "resources": [
                {
                    "type": "Microsoft.Web/serverfarms",
                    "apiVersion": "2021-02-01",
                    "name": "[variables('appServicePlanName')]",
                    "location": "[resourceGroup().location]",
                    "sku": {
                        "name": "B1",
                        "tier": "Basic"
                    },
                    "properties": {
                        "reserved": True
                    }
                },
                {
                    "type": "Microsoft.Web/sites",
                    "apiVersion": "2021-02-01",
                    "name": "[variables('webAppName')]",
                    "location": "[resourceGroup().location]",
                    "dependsOn": [
                        "[resourceId('Microsoft.Web/serverfarms', variables('appServicePlanName'))]"
                    ],
                    "properties": {
                        "serverFarmId": "[resourceId('Microsoft.Web/serverfarms', variables('appServicePlanName'))]",
                        "siteConfig": {
                            "linuxFxVersion": "DOCKER|nginx:latest"
                        }
                    }
                }
            ],
            "outputs": {
                "webAppUrl": {
                    "type": "string",
                    "value": "[concat('https://', variables('webAppName'), '.azurewebsites.net')]"
                }
            }
        }
        
        return json.dumps(template, indent=2)
    
    def generate_gcp_deployment_manager(self, application: GeneratedApplication, config: Dict[str, Any]) -> str:
        """Generate GCP Deployment Manager template."""
        return f"""# Generated GCP Deployment Manager template for {application.name}
resources:
- name: {application.name}-network
  type: compute.v1.network
  properties:
    autoCreateSubnetworks: false

- name: {application.name}-subnet
  type: compute.v1.subnetwork
  properties:
    network: $(ref.{application.name}-network.selfLink)
    ipCidrRange: 10.0.0.0/24
    region: us-central1

- name: {application.name}-firewall
  type: compute.v1.firewall
  properties:
    network: $(ref.{application.name}-network.selfLink)
    allowed:
    - IPProtocol: TCP
      ports: ["80", "443", "8080"]
    sourceRanges: ["0.0.0.0/0"]

- name: {application.name}-instance-template
  type: compute.v1.instanceTemplate
  properties:
    properties:
      machineType: e2-micro
      disks:
      - deviceName: boot
        type: PERSISTENT
        boot: true
        autoDelete: true
        initializeParams:
          sourceImage: projects/cos-cloud/global/images/family/cos-stable
      networkInterfaces:
      - network: $(ref.{application.name}-network.selfLink)
        subnetwork: $(ref.{application.name}-subnet.selfLink)
        accessConfigs:
        - name: External NAT
          type: ONE_TO_ONE_NAT

- name: {application.name}-managed-instance-group
  type: compute.v1.instanceGroupManager
  properties:
    zone: us-central1-a
    targetSize: 2
    instanceTemplate: $(ref.{application.name}-instance-template.selfLink)
    baseInstanceName: {application.name}-instance

outputs:
- name: network-name
  value: $(ref.{application.name}-network.name)
- name: subnet-name
  value: $(ref.{application.name}-subnet.name)
"""


class CICDGenerator:
    """Generates CI/CD pipeline configurations."""
    
    def generate_github_actions(self, application: GeneratedApplication, config: Dict[str, Any]) -> str:
        """Generate GitHub Actions workflow."""
        return f"""# Generated GitHub Actions workflow for {application.name}
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{{{ github.repository }}}}

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest --cov=./ --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  security-scan:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Run security scan
      uses: securecodewarrior/github-action-add-sarif@v1
      with:
        sarif-file: 'security-scan-results.sarif'

  build-and-push:
    needs: [test, security-scan]
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{{{ env.REGISTRY }}}}
        username: ${{{{ github.actor }}}}
        password: ${{{{ secrets.GITHUB_TOKEN }}}}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{{{ env.REGISTRY }}}}/${{{{ env.IMAGE_NAME }}}}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{{{ steps.meta.outputs.tags }}}}
        labels: ${{{{ steps.meta.outputs.labels }}}}

  deploy-staging:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment: staging
    
    steps:
    - name: Deploy to staging
      run: |
        echo "Deploying to staging environment"
        # Add deployment commands here

  deploy-production:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - name: Deploy to production
      run: |
        echo "Deploying to production environment"
        # Add deployment commands here
"""
    
    def generate_gitlab_ci(self, application: GeneratedApplication, config: Dict[str, Any]) -> str:
        """Generate GitLab CI configuration."""
        return f"""# Generated GitLab CI configuration for {application.name}
stages:
  - test
  - security
  - build
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"

before_script:
  - python -m pip install --upgrade pip

test:
  stage: test
  image: python:3.11
  script:
    - pip install -r requirements.txt
    - pip install pytest pytest-cov
    - pytest --cov=./ --cov-report=xml --cov-report=term
  coverage: '/TOTAL.+ ([0-9]{{1,3}}%)/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

security-scan:
  stage: security
  image: securecodewarrior/docker-action
  script:
    - echo "Running security scan"
    # Add security scanning commands
  allow_failure: true

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only:
    - main
    - develop

deploy-staging:
  stage: deploy
  image: alpine:latest
  script:
    - echo "Deploying to staging"
    # Add staging deployment commands
  environment:
    name: staging
    url: https://staging.{application.name}.com
  only:
    - develop

deploy-production:
  stage: deploy
  image: alpine:latest
  script:
    - echo "Deploying to production"
    # Add production deployment commands
  environment:
    name: production
    url: https://{application.name}.com
  only:
    - main
  when: manual
"""
    
    def generate_jenkins_pipeline(self, application: GeneratedApplication, config: Dict[str, Any]) -> str:
        """Generate Jenkins pipeline configuration."""
        return f"""// Generated Jenkins pipeline for {application.name}
pipeline {{
    agent any
    
    environment {{
        DOCKER_REGISTRY = 'your-registry.com'
        IMAGE_NAME = '{application.name}'
        KUBECONFIG = credentials('kubeconfig')
    }}
    
    stages {{
        stage('Checkout') {{
            steps {{
                checkout scm
            }}
        }}
        
        stage('Test') {{
            steps {{
                script {{
                    sh '''
                        python -m pip install --upgrade pip
                        pip install -r requirements.txt
                        pip install pytest pytest-cov
                        pytest --cov=./ --cov-report=xml
                    '''
                }}
            }}
            post {{
                always {{
                    publishCoverage adapters: [
                        coberturaAdapter('coverage.xml')
                    ], sourceFileResolver: sourceFiles('STORE_LAST_BUILD')
                }}
            }}
        }}
        
        stage('Security Scan') {{
            steps {{
                script {{
                    sh 'echo "Running security scan"'
                    // Add security scanning commands
                }}
            }}
        }}
        
        stage('Build') {{
            steps {{
                script {{
                    def image = docker.build("${{DOCKER_REGISTRY}}/${{IMAGE_NAME}}:${{BUILD_NUMBER}}")
                    docker.withRegistry("https://${{DOCKER_REGISTRY}}", 'docker-registry-credentials') {{
                        image.push()
                        image.push('latest')
                    }}
                }}
            }}
        }}
        
        stage('Deploy to Staging') {{
            when {{
                branch 'develop'
            }}
            steps {{
                script {{
                    sh '''
                        kubectl set image deployment/{application.name} \\
                            {application.name}=${{DOCKER_REGISTRY}}/${{IMAGE_NAME}}:${{BUILD_NUMBER}} \\
                            --namespace=staging
                        kubectl rollout status deployment/{application.name} --namespace=staging
                    '''
                }}
            }}
        }}
        
        stage('Deploy to Production') {{
            when {{
                branch 'main'
            }}
            steps {{
                input message: 'Deploy to production?', ok: 'Deploy'
                script {{
                    sh '''
                        kubectl set image deployment/{application.name} \\
                            {application.name}=${{DOCKER_REGISTRY}}/${{IMAGE_NAME}}:${{BUILD_NUMBER}} \\
                            --namespace=production
                        kubectl rollout status deployment/{application.name} --namespace=production
                    '''
                }}
            }}
        }}
    }}
    
    post {{
        always {{
            cleanWs()
        }}
        failure {{
            emailext (
                subject: "Pipeline Failed: ${{env.JOB_NAME}} - ${{env.BUILD_NUMBER}}",
                body: "The pipeline has failed. Please check the logs.",
                to: "team@company.com"
            )
        }}
    }}
}}
"""


class DeploymentAutomation:
    """Main deployment automation engine."""
    
    def __init__(self):
        self.container_builder = ContainerBuilder()
        self.infrastructure_generator = InfrastructureGenerator()
        self.cicd_generator = CICDGenerator()
    
    def generate_deployment_config(
        self,
        application: GeneratedApplication,
        environment: DeploymentEnvironment,
        cloud_provider: CloudProvider,
        config: Dict[str, Any]
    ) -> DeploymentConfig:
        """Generate complete deployment configuration."""
        
        # Build container configuration
        container_config = self.container_builder.build_container_config(application)
        
        # Generate infrastructure code
        infrastructure_code = self._generate_infrastructure_code(
            application, cloud_provider, config
        )
        
        # Generate CI/CD pipeline
        cicd_pipeline = self._generate_cicd_pipeline(application, config)
        
        # Merge default configurations with user-provided config
        default_auto_scaling = {
            "min_instances": 1,
            "max_instances": 10,
            "target_cpu_utilization": 70
        }
        auto_scaling = {**default_auto_scaling, **config.get("auto_scaling", {})}
        
        default_load_balancing = {
            "type": "application",
            "health_check_path": "/health"
        }
        load_balancing = {**default_load_balancing, **config.get("load_balancing", {})}
        
        default_monitoring = {
            "metrics_enabled": True,
            "logging_enabled": True,
            "alerting_enabled": True
        }
        monitoring = {**default_monitoring, **config.get("monitoring", {})}
        
        default_security = {
            "https_only": True,
            "security_headers": True,
            "waf_enabled": True
        }
        security = {**default_security, **config.get("security", {})}

        return DeploymentConfig(
            id=f"deploy-{application.id}-{environment.value}",
            name=f"{application.name}-{environment.value}",
            application_id=application.id,
            environment=environment,
            cloud_provider=cloud_provider,
            container_config=container_config,
            infrastructure_code=infrastructure_code,
            cicd_pipeline=cicd_pipeline,
            auto_scaling=auto_scaling,
            load_balancing=load_balancing,
            monitoring=monitoring,
            security=security,
            created_by="system"
        )
    
    def _generate_infrastructure_code(
        self,
        application: GeneratedApplication,
        cloud_provider: CloudProvider,
        config: Dict[str, Any]
    ) -> InfrastructureCode:
        """Generate infrastructure as code based on cloud provider."""
        
        if cloud_provider == CloudProvider.AWS:
            template_content = self.infrastructure_generator.generate_terraform_aws(application, config)
            template_type = "terraform"
        elif cloud_provider == CloudProvider.AZURE:
            template_content = self.infrastructure_generator.generate_azure_arm(application, config)
            template_type = "arm"
        elif cloud_provider == CloudProvider.GCP:
            template_content = self.infrastructure_generator.generate_gcp_deployment_manager(application, config)
            template_type = "deployment_manager"
        else:
            template_content = self.infrastructure_generator.generate_terraform_aws(application, config)
            template_type = "terraform"
        
        return InfrastructureCode(
            provider=cloud_provider,
            template_type=template_type,
            template_content=template_content,
            variables={
                "app_name": application.name,
                "environment": config.get("environment", "production"),
                "region": config.get("region", "us-west-2")
            },
            outputs={
                "app_url": "Application URL",
                "database_endpoint": "Database endpoint"
            },
            dependencies=["vpc", "security_groups", "load_balancer"]
        )
    
    def _generate_cicd_pipeline(
        self,
        application: GeneratedApplication,
        config: Dict[str, Any]
    ) -> CICDPipeline:
        """Generate CI/CD pipeline configuration."""
        
        platform = config.get("cicd_platform", "github")
        
        if platform == "github":
            pipeline_content = self.cicd_generator.generate_github_actions(application, config)
        elif platform == "gitlab":
            pipeline_content = self.cicd_generator.generate_gitlab_ci(application, config)
        elif platform == "jenkins":
            pipeline_content = self.cicd_generator.generate_jenkins_pipeline(application, config)
        else:
            pipeline_content = self.cicd_generator.generate_github_actions(application, config)
        
        return CICDPipeline(
            platform=platform,
            pipeline_content=pipeline_content,
            stages=["test", "security", "build", "deploy"],
            triggers=["push", "pull_request"],
            environment_configs={
                "staging": {"auto_deploy": True},
                "production": {"manual_approval": True}
            },
            secrets=["DOCKER_REGISTRY_TOKEN", "CLOUD_CREDENTIALS", "DATABASE_URL"]
        )
    
    def validate_deployment_config(self, config: DeploymentConfig) -> DeploymentValidation:
        """Validate deployment configuration."""
        errors = []
        warnings = []
        recommendations = []
        
        # Validate container configuration
        if not config.container_config.dockerfile_content:
            errors.append("Dockerfile content is required")
        
        if not config.container_config.exposed_ports:
            warnings.append("No exposed ports defined")
        
        # Validate infrastructure configuration
        if not config.infrastructure_code.template_content:
            errors.append("Infrastructure template content is required")
        
        # Validate CI/CD configuration
        if not config.cicd_pipeline.pipeline_content:
            errors.append("CI/CD pipeline content is required")
        
        # Security recommendations
        if not config.security.get("https_only"):
            recommendations.append("Enable HTTPS-only for better security")
        
        if not config.security.get("waf_enabled"):
            recommendations.append("Enable Web Application Firewall for enhanced security")
        
        # Performance recommendations
        if config.auto_scaling.get("max_instances", 0) < 3:
            recommendations.append("Consider increasing max instances for better availability")
        
        return DeploymentValidation(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations,
            estimated_cost=self._estimate_cost(config),
            security_score=self._calculate_security_score(config)
        )
    
    def _estimate_cost(self, config: DeploymentConfig) -> float:
        """Estimate monthly deployment cost."""
        # Simple cost estimation based on configuration
        base_cost = 50.0  # Base infrastructure cost
        
        # Add cost based on auto-scaling
        max_instances = config.auto_scaling.get("max_instances", 1)
        instance_cost = max_instances * 30.0  # $30 per instance per month
        
        # Add cost based on cloud provider
        if config.cloud_provider == CloudProvider.AWS:
            multiplier = 1.0
        elif config.cloud_provider == CloudProvider.AZURE:
            multiplier = 0.95
        elif config.cloud_provider == CloudProvider.GCP:
            multiplier = 0.90
        else:
            multiplier = 1.0
        
        return (base_cost + instance_cost) * multiplier
    
    def _calculate_security_score(self, config: DeploymentConfig) -> int:
        """Calculate security score (0-100)."""
        score = 0
        
        # HTTPS enabled
        if config.security.get("https_only"):
            score += 20
        
        # WAF enabled
        if config.security.get("waf_enabled"):
            score += 20
        
        # Security headers
        if config.security.get("security_headers"):
            score += 15
        
        # Health checks
        if config.container_config.health_check:
            score += 10
        
        # Resource limits
        if config.container_config.resource_limits:
            score += 10
        
        # Non-root user (check Dockerfile)
        if "USER" in config.container_config.dockerfile_content:
            score += 15
        
        # Monitoring enabled
        if config.monitoring.get("metrics_enabled"):
            score += 10
        
        return min(score, 100)
    
    def deploy_application(
        self,
        config: DeploymentConfig,
        dry_run: bool = False
    ) -> DeploymentResult:
        """Deploy application using the provided configuration."""
        
        deployment_result = DeploymentResult(
            deployment_id=f"deploy-{config.id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            status="in_progress",
            environment=config.environment,
            cloud_provider=config.cloud_provider,
            started_at=datetime.utcnow()
        )
        
        try:
            if dry_run:
                deployment_result.status = "dry_run_success"
                deployment_result.logs.append("Dry run completed successfully")
            else:
                # In a real implementation, this would execute the actual deployment
                deployment_result.status = "success"
                deployment_result.container_image = f"{config.name}:latest"
                deployment_result.endpoints = [f"https://{config.name}.example.com"]
                deployment_result.logs.append("Deployment completed successfully")
            
            deployment_result.completed_at = datetime.utcnow()
            
        except Exception as e:
            deployment_result.status = "failed"
            deployment_result.errors.append(str(e))
            deployment_result.completed_at = datetime.utcnow()
        
        return deployment_result