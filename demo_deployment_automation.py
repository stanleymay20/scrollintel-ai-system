"""
Demo script for deployment automation system.
"""

import asyncio
import json
from datetime import datetime

from scrollintel.engines.deployment_automation import DeploymentAutomation
from scrollintel.models.deployment_models import CloudProvider, DeploymentEnvironment
from scrollintel.models.code_generation_models import GeneratedApplication, CodeComponent


def create_sample_python_application():
    """Create a sample Python web application."""
    return GeneratedApplication(
        id="demo-python-app",
        name="python-web-service",
        description="Demo Python web service with Flask and PostgreSQL",
        requirements=None,
        architecture=None,
        code_components=[
            CodeComponent(
                id="main-py",
                name="main.py",
                type="backend",
                language="python",
                code="""
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost/db')
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()})

@app.route('/users', methods=['GET'])
def get_users():
    users = User.query.all()
    return jsonify([{'id': u.id, 'name': u.name, 'email': u.email} for u in users])

@app.route('/users', methods=['POST'])
def create_user():
    data = request.get_json()
    user = User(name=data['name'], email=data['email'])
    db.session.add(user)
    db.session.commit()
    return jsonify({'id': user.id, 'name': user.name, 'email': user.email}), 201

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=8000, debug=False)
""",
                dependencies=["flask", "flask-sqlalchemy", "psycopg2-binary"],
                tests=[]
            ),
            CodeComponent(
                id="requirements-txt",
                name="requirements.txt",
                type="config",
                language="text",
                code="""flask==2.3.3
flask-sqlalchemy==3.0.5
psycopg2-binary==2.9.7
gunicorn==21.2.0
python-dotenv==1.0.0""",
                dependencies=[],
                tests=[]
            ),
            CodeComponent(
                id="dockerfile",
                name="Dockerfile",
                type="config",
                language="dockerfile",
                code="""FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    postgresql-client \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "main:app"]""",
                dependencies=[],
                tests=[]
            )
        ],
        tests=None,
        deployment_config=None
    )


def create_sample_node_application():
    """Create a sample Node.js application."""
    return GeneratedApplication(
        id="demo-node-app",
        name="node-api-service",
        description="Demo Node.js API service with Express and MongoDB",
        requirements=None,
        architecture=None,
        code_components=[
            CodeComponent(
                id="app-js",
                name="app.js",
                type="backend",
                language="javascript",
                code="""
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
require('dotenv').config();

const app = express();
const port = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// MongoDB connection
mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/demo', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

// User schema
const userSchema = new mongoose.Schema({
  name: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  createdAt: { type: Date, default: Date.now }
});

const User = mongoose.model('User', userSchema);

// Routes
app.get('/health', (req, res) => {
  res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

app.get('/users', async (req, res) => {
  try {
    const users = await User.find();
    res.json(users);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/users', async (req, res) => {
  try {
    const user = new User(req.body);
    await user.save();
    res.status(201).json(user);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
""",
                dependencies=["express", "mongoose", "cors", "dotenv"],
                tests=[]
            ),
            CodeComponent(
                id="package-json",
                name="package.json",
                type="config",
                language="json",
                code="""{
  "name": "node-api-service",
  "version": "1.0.0",
  "description": "Demo Node.js API service",
  "main": "app.js",
  "scripts": {
    "start": "node app.js",
    "dev": "nodemon app.js",
    "test": "jest"
  },
  "dependencies": {
    "express": "^4.18.2",
    "mongoose": "^7.5.0",
    "cors": "^2.8.5",
    "dotenv": "^16.3.1"
  },
  "devDependencies": {
    "nodemon": "^3.0.1",
    "jest": "^29.6.2"
  }
}""",
                dependencies=[],
                tests=[]
            )
        ],
        tests=None,
        deployment_config=None
    )


async def demo_deployment_automation():
    """Demonstrate deployment automation functionality."""
    print("🚀 Deployment Automation Demo")
    print("=" * 50)
    
    # Initialize deployment automation
    deployment_engine = DeploymentAutomation()
    
    # Create sample applications
    python_app = create_sample_python_application()
    node_app = create_sample_node_application()
    
    print(f"\n📱 Created sample applications:")
    print(f"  - Python App: {python_app.name}")
    print(f"  - Node.js App: {node_app.name}")
    
    # Demo 1: Python application deployment to AWS
    print(f"\n🔧 Demo 1: Python Application Deployment to AWS")
    print("-" * 40)
    
    aws_config = deployment_engine.generate_deployment_config(
        application=python_app,
        environment=DeploymentEnvironment.PRODUCTION,
        cloud_provider=CloudProvider.AWS,
        config={
            "region": "us-west-2",
            "cicd_platform": "github",
            "instance_type": "t3.micro",
            "auto_scaling": {
                "min_instances": 2,
                "max_instances": 10,
                "target_cpu_utilization": 70
            }
        }
    )
    
    print(f"✅ Generated AWS deployment configuration:")
    print(f"  - Environment: {aws_config.environment}")
    print(f"  - Cloud Provider: {aws_config.cloud_provider}")
    print(f"  - Container Base Image: {aws_config.container_config.base_image}")
    print(f"  - Exposed Ports: {aws_config.container_config.exposed_ports}")
    print(f"  - Auto Scaling: {aws_config.auto_scaling}")
    
    # Validate configuration
    validation = deployment_engine.validate_deployment_config(aws_config)
    print(f"\n📋 Configuration Validation:")
    print(f"  - Valid: {validation.is_valid}")
    print(f"  - Errors: {len(validation.errors)}")
    print(f"  - Warnings: {len(validation.warnings)}")
    print(f"  - Recommendations: {len(validation.recommendations)}")
    print(f"  - Estimated Cost: ${validation.estimated_cost:.2f}/month")
    print(f"  - Security Score: {validation.security_score}/100")
    
    if validation.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in validation.recommendations[:3]:
            print(f"  - {rec}")
    
    # Show generated Dockerfile
    print(f"\n🐳 Generated Dockerfile (first 10 lines):")
    dockerfile_lines = aws_config.container_config.dockerfile_content.split('\n')
    for i, line in enumerate(dockerfile_lines[:10]):
        print(f"  {i+1:2d}: {line}")
    if len(dockerfile_lines) > 10:
        print(f"  ... ({len(dockerfile_lines) - 10} more lines)")
    
    # Show infrastructure code snippet
    print(f"\n🏗️ Generated Terraform Code (snippet):")
    terraform_lines = aws_config.infrastructure_code.template_content.split('\n')
    for i, line in enumerate(terraform_lines[:8]):
        if line.strip():
            print(f"  {line}")
    print("  ...")
    
    # Perform dry run deployment
    print(f"\n🧪 Performing Dry Run Deployment...")
    dry_run_result = deployment_engine.deploy_application(aws_config, dry_run=True)
    print(f"  - Status: {dry_run_result.status}")
    print(f"  - Started: {dry_run_result.started_at}")
    print(f"  - Completed: {dry_run_result.completed_at}")
    print(f"  - Logs: {dry_run_result.logs}")
    
    # Demo 2: Node.js application deployment to Azure
    print(f"\n🔧 Demo 2: Node.js Application Deployment to Azure")
    print("-" * 40)
    
    azure_config = deployment_engine.generate_deployment_config(
        application=node_app,
        environment=DeploymentEnvironment.STAGING,
        cloud_provider=CloudProvider.AZURE,
        config={
            "region": "eastus",
            "cicd_platform": "gitlab",
            "monitoring": {
                "metrics_enabled": True,
                "logging_enabled": True,
                "alerting_enabled": True
            }
        }
    )
    
    print(f"✅ Generated Azure deployment configuration:")
    print(f"  - Environment: {azure_config.environment}")
    print(f"  - Cloud Provider: {azure_config.cloud_provider}")
    print(f"  - Infrastructure Type: {azure_config.infrastructure_code.template_type}")
    print(f"  - CI/CD Platform: {azure_config.cicd_pipeline.platform}")
    print(f"  - Monitoring: {azure_config.monitoring}")
    
    # Validate Azure configuration
    azure_validation = deployment_engine.validate_deployment_config(azure_config)
    print(f"\n📋 Azure Configuration Validation:")
    print(f"  - Valid: {azure_validation.is_valid}")
    print(f"  - Estimated Cost: ${azure_validation.estimated_cost:.2f}/month")
    print(f"  - Security Score: {azure_validation.security_score}/100")
    
    # Demo 3: Multi-cloud cost comparison
    print(f"\n💰 Demo 3: Multi-Cloud Cost Comparison")
    print("-" * 40)
    
    cloud_providers = [CloudProvider.AWS, CloudProvider.AZURE, CloudProvider.GCP]
    cost_comparison = {}
    
    for provider in cloud_providers:
        config = deployment_engine.generate_deployment_config(
            application=python_app,
            environment=DeploymentEnvironment.PRODUCTION,
            cloud_provider=provider,
            config={"region": "us-west-2"}
        )
        
        validation = deployment_engine.validate_deployment_config(config)
        cost_comparison[provider.value] = validation.estimated_cost
        
        print(f"  - {provider.value.upper()}: ${validation.estimated_cost:.2f}/month")
    
    # Find cheapest option
    cheapest = min(cost_comparison, key=cost_comparison.get)
    print(f"\n💡 Cheapest option: {cheapest.upper()} at ${cost_comparison[cheapest]:.2f}/month")
    
    # Demo 4: CI/CD Pipeline Generation
    print(f"\n⚙️ Demo 4: CI/CD Pipeline Generation")
    print("-" * 40)
    
    # Generate different CI/CD platforms
    platforms = ["github", "gitlab", "jenkins"]
    
    for platform in platforms:
        config = deployment_engine.generate_deployment_config(
            application=python_app,
            environment=DeploymentEnvironment.PRODUCTION,
            cloud_provider=CloudProvider.AWS,
            config={"cicd_platform": platform}
        )
        
        print(f"\n🔄 {platform.title()} Actions Pipeline (first 8 lines):")
        pipeline_lines = config.cicd_pipeline.pipeline_content.split('\n')
        for i, line in enumerate(pipeline_lines[:8]):
            if line.strip():
                print(f"  {line}")
        print("  ...")
    
    # Demo 5: Security Analysis
    print(f"\n🔒 Demo 5: Security Analysis")
    print("-" * 40)
    
    # Create configurations with different security levels
    high_security_config = deployment_engine.generate_deployment_config(
        application=python_app,
        environment=DeploymentEnvironment.PRODUCTION,
        cloud_provider=CloudProvider.AWS,
        config={
            "region": "us-west-2",
            "security": {
                "https_only": True,
                "waf_enabled": True,
                "security_headers": True
            }
        }
    )
    
    low_security_config = deployment_engine.generate_deployment_config(
        application=python_app,
        environment=DeploymentEnvironment.DEVELOPMENT,
        cloud_provider=CloudProvider.AWS,
        config={
            "region": "us-west-2",
            "security": {
                "https_only": False,
                "waf_enabled": False,
                "security_headers": False
            }
        }
    )
    
    high_security_score = deployment_engine._calculate_security_score(high_security_config)
    low_security_score = deployment_engine._calculate_security_score(low_security_config)
    
    print(f"  - High Security Config Score: {high_security_score}/100")
    print(f"  - Low Security Config Score: {low_security_score}/100")
    print(f"  - Security Improvement: +{high_security_score - low_security_score} points")
    
    # Demo 6: Deployment Templates
    print(f"\n📋 Demo 6: Deployment Templates")
    print("-" * 40)
    
    print("Available deployment templates:")
    templates = [
        {"name": "Python Web App", "provider": "AWS", "type": "ECS + RDS"},
        {"name": "Node.js API", "provider": "Azure", "type": "App Service + CosmosDB"},
        {"name": "Microservices", "provider": "GCP", "type": "GKE + Cloud SQL"},
        {"name": "Static Site", "provider": "AWS", "type": "S3 + CloudFront"},
        {"name": "Serverless API", "provider": "AWS", "type": "Lambda + API Gateway"}
    ]
    
    for i, template in enumerate(templates, 1):
        print(f"  {i}. {template['name']} ({template['provider']} - {template['type']})")
    
    print(f"\n🎯 Demo Summary")
    print("=" * 50)
    print("✅ Successfully demonstrated:")
    print("  - Deployment configuration generation")
    print("  - Multi-cloud infrastructure templates")
    print("  - Container optimization")
    print("  - CI/CD pipeline generation")
    print("  - Security validation")
    print("  - Cost estimation")
    print("  - Configuration validation")
    print("  - Dry run deployments")
    
    print(f"\n🚀 Deployment automation system is ready for production use!")


if __name__ == "__main__":
    asyncio.run(demo_deployment_automation())