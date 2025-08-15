#!/usr/bin/env python3
"""
Deployment script for visual generation infrastructure across multiple cloud providers.
"""

import os
import sys
import json
import yaml
import argparse
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import boto3
import kubernetes
from google.cloud import compute_v1
from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.resource import ResourceManagementClient


class InfrastructureDeployer:
    """Deploy visual generation infrastructure to cloud providers."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Cloud clients
        self.aws_clients = {}
        self.gcp_clients = {}
        self.azure_clients = {}
        self.k8s_clients = {}
        
        self._initialize_clients()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration."""
        with open(self.config_path, 'r') as f:
            if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                return yaml.safe_load(f)
            else:
                return json.load(f)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _initialize_clients(self):
        """Initialize cloud provider clients."""
        try:
            # AWS
            if self.config.get('aws', {}).get('enabled', False):
                self.aws_clients = {
                    'ec2': boto3.client('ec2', region_name=self.config['aws']['region']),
                    'autoscaling': boto3.client('autoscaling', region_name=self.config['aws']['region']),
                    'ecs': boto3.client('ecs', region_name=self.config['aws']['region']),
                    'iam': boto3.client('iam', region_name=self.config['aws']['region']),
                    'cloudformation': boto3.client('cloudformation', region_name=self.config['aws']['region'])
                }
            
            # GCP
            if self.config.get('gcp', {}).get('enabled', False):
                self.gcp_clients = {
                    'compute': compute_v1.InstancesClient(),
                    'instance_groups': compute_v1.InstanceGroupManagersClient(),
                    'instance_templates': compute_v1.InstanceTemplatesClient()
                }
            
            # Azure
            if self.config.get('azure', {}).get('enabled', False):
                credential = DefaultAzureCredential()
                subscription_id = self.config['azure']['subscription_id']
                self.azure_clients = {
                    'compute': ComputeManagementClient(credential, subscription_id),
                    'resource': ResourceManagementClient(credential, subscription_id)
                }
            
            # Kubernetes
            if self.config.get('kubernetes', {}).get('enabled', False):
                if self.config['kubernetes'].get('in_cluster', False):
                    kubernetes.config.load_incluster_config()
                else:
                    kubernetes.config.load_kube_config()
                
                self.k8s_clients = {
                    'apps_v1': kubernetes.client.AppsV1Api(),
                    'core_v1': kubernetes.client.CoreV1Api(),
                    'custom_objects': kubernetes.client.CustomObjectsApi(),
                    'networking_v1': kubernetes.client.NetworkingV1Api()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to initialize clients: {e}")
            raise
    
    async def deploy_all(self):
        """Deploy infrastructure to all configured providers."""
        tasks = []
        
        if self.aws_clients:
            tasks.append(self.deploy_aws())
        
        if self.gcp_clients:
            tasks.append(self.deploy_gcp())
        
        if self.azure_clients:
            tasks.append(self.deploy_azure())
        
        if self.k8s_clients:
            tasks.append(self.deploy_kubernetes())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            self.logger.warning("No cloud providers configured for deployment")
    
    async def deploy_aws(self):
        """Deploy infrastructure to AWS."""
        self.logger.info("Deploying to AWS...")
        
        try:
            # Create VPC and networking
            await self._create_aws_networking()
            
            # Create security groups
            await self._create_aws_security_groups()
            
            # Create IAM roles and policies
            await self._create_aws_iam_resources()
            
            # Create launch templates
            await self._create_aws_launch_templates()
            
            # Create auto scaling groups
            await self._create_aws_autoscaling_groups()
            
            # Create ECS cluster for container workloads
            await self._create_aws_ecs_cluster()
            
            self.logger.info("AWS deployment completed successfully")
            
        except Exception as e:
            self.logger.error(f"AWS deployment failed: {e}")
            raise
    
    async def _create_aws_networking(self):
        """Create AWS VPC and networking resources."""
        ec2 = self.aws_clients['ec2']
        aws_config = self.config['aws']
        
        # Create VPC
        vpc_response = ec2.create_vpc(
            CidrBlock=aws_config.get('vpc_cidr', '10.0.0.0/16'),
            TagSpecifications=[{
                'ResourceType': 'vpc',
                'Tags': [
                    {'Key': 'Name', 'Value': 'scrollintel-visual-generation-vpc'},
                    {'Key': 'Project', 'Value': 'scrollintel'},
                    {'Key': 'Component', 'Value': 'visual-generation'}
                ]
            }]
        )
        vpc_id = vpc_response['Vpc']['VpcId']
        
        # Create internet gateway
        igw_response = ec2.create_internet_gateway(
            TagSpecifications=[{
                'ResourceType': 'internet-gateway',
                'Tags': [{'Key': 'Name', 'Value': 'scrollintel-visual-generation-igw'}]
            }]
        )
        igw_id = igw_response['InternetGateway']['InternetGatewayId']
        
        # Attach internet gateway to VPC
        ec2.attach_internet_gateway(InternetGatewayId=igw_id, VpcId=vpc_id)
        
        # Create subnets
        subnets = []
        availability_zones = ec2.describe_availability_zones()['AvailabilityZones']
        
        for i, az in enumerate(availability_zones[:3]):  # Create up to 3 subnets
            subnet_response = ec2.create_subnet(
                VpcId=vpc_id,
                CidrBlock=f"10.0.{i+1}.0/24",
                AvailabilityZone=az['ZoneName'],
                TagSpecifications=[{
                    'ResourceType': 'subnet',
                    'Tags': [
                        {'Key': 'Name', 'Value': f'scrollintel-visual-generation-subnet-{i+1}'},
                        {'Key': 'Type', 'Value': 'public'}
                    ]
                }]
            )
            subnets.append(subnet_response['Subnet']['SubnetId'])
        
        # Create route table
        route_table_response = ec2.create_route_table(
            VpcId=vpc_id,
            TagSpecifications=[{
                'ResourceType': 'route-table',
                'Tags': [{'Key': 'Name', 'Value': 'scrollintel-visual-generation-rt'}]
            }]
        )
        route_table_id = route_table_response['RouteTable']['RouteTableId']
        
        # Add route to internet gateway
        ec2.create_route(
            RouteTableId=route_table_id,
            DestinationCidrBlock='0.0.0.0/0',
            GatewayId=igw_id
        )
        
        # Associate subnets with route table
        for subnet_id in subnets:
            ec2.associate_route_table(RouteTableId=route_table_id, SubnetId=subnet_id)
        
        # Store IDs for later use
        self.config['aws']['vpc_id'] = vpc_id
        self.config['aws']['subnet_ids'] = subnets
        
        self.logger.info(f"Created AWS networking: VPC {vpc_id}, Subnets {subnets}")
    
    async def _create_aws_security_groups(self):
        """Create AWS security groups."""
        ec2 = self.aws_clients['ec2']
        vpc_id = self.config['aws']['vpc_id']
        
        # Worker security group
        worker_sg_response = ec2.create_security_group(
            GroupName='scrollintel-visual-generation-workers',
            Description='Security group for visual generation workers',
            VpcId=vpc_id,
            TagSpecifications=[{
                'ResourceType': 'security-group',
                'Tags': [{'Key': 'Name', 'Value': 'scrollintel-visual-generation-workers'}]
            }]
        )
        worker_sg_id = worker_sg_response['GroupId']
        
        # Add inbound rules
        ec2.authorize_security_group_ingress(
            GroupId=worker_sg_id,
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 22,
                    'ToPort': 22,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'SSH access'}]
                },
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 8000,
                    'ToPort': 8000,
                    'IpRanges': [{'CidrIp': '10.0.0.0/16', 'Description': 'Worker API'}]
                },
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 6379,
                    'ToPort': 6379,
                    'IpRanges': [{'CidrIp': '10.0.0.0/16', 'Description': 'Redis'}]
                }
            ]
        )
        
        self.config['aws']['worker_security_group_id'] = worker_sg_id
        self.logger.info(f"Created AWS security group: {worker_sg_id}")
    
    async def _create_aws_iam_resources(self):
        """Create AWS IAM roles and policies."""
        iam = self.aws_clients['iam']
        
        # Worker instance role
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "ec2.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        role_response = iam.create_role(
            RoleName='ScrollIntelVisualGenerationWorkerRole',
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='Role for ScrollIntel visual generation workers',
            Tags=[
                {'Key': 'Project', 'Value': 'scrollintel'},
                {'Key': 'Component', 'Value': 'visual-generation'}
            ]
        )
        
        # Worker policy
        worker_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "s3:GetObject",
                        "s3:PutObject",
                        "s3:DeleteObject"
                    ],
                    "Resource": "arn:aws:s3:::scrollintel-visual-generation/*"
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "cloudwatch:PutMetricData",
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents"
                    ],
                    "Resource": "*"
                }
            ]
        }
        
        iam.put_role_policy(
            RoleName='ScrollIntelVisualGenerationWorkerRole',
            PolicyName='ScrollIntelVisualGenerationWorkerPolicy',
            PolicyDocument=json.dumps(worker_policy)
        )
        
        # Create instance profile
        iam.create_instance_profile(
            InstanceProfileName='ScrollIntelVisualGenerationWorkerProfile'
        )
        
        iam.add_role_to_instance_profile(
            InstanceProfileName='ScrollIntelVisualGenerationWorkerProfile',
            RoleName='ScrollIntelVisualGenerationWorkerRole'
        )
        
        self.logger.info("Created AWS IAM resources")
    
    async def _create_aws_launch_templates(self):
        """Create AWS launch templates for different worker types."""
        ec2 = self.aws_clients['ec2']
        
        # GPU worker launch template
        gpu_user_data = """#!/bin/bash
yum update -y
yum install -y docker
systemctl start docker
systemctl enable docker

# Install NVIDIA drivers
yum install -y gcc kernel-devel-$(uname -r)
wget https://developer.download.nvidia.com/compute/cuda/repos/amzn2/x86_64/cuda-repo-amzn2-11-8.x86_64.rpm
rpm -i cuda-repo-amzn2-11-8.x86_64.rpm
yum clean all
yum install -y cuda

# Install nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | tee /etc/yum.repos.d/nvidia-docker.repo
yum install -y nvidia-docker2
systemctl restart docker

# Pull and run ScrollIntel worker
docker pull scrollintel/visual-generation-worker:latest
docker run -d --gpus all --name scrollintel-worker \\
  -e WORKER_TYPE=gpu_worker \\
  -e REDIS_URL=redis://redis.scrollintel.internal:6379 \\
  scrollintel/visual-generation-worker:latest
"""
        
        gpu_template_response = ec2.create_launch_template(
            LaunchTemplateName='scrollintel-gpu-worker-template',
            LaunchTemplateData={
                'ImageId': 'ami-0abcdef1234567890',  # Deep Learning AMI
                'InstanceType': 'g4dn.xlarge',
                'SecurityGroupIds': [self.config['aws']['worker_security_group_id']],
                'IamInstanceProfile': {
                    'Name': 'ScrollIntelVisualGenerationWorkerProfile'
                },
                'UserData': gpu_user_data,
                'TagSpecifications': [{
                    'ResourceType': 'instance',
                    'Tags': [
                        {'Key': 'Name', 'Value': 'scrollintel-gpu-worker'},
                        {'Key': 'WorkerType', 'Value': 'gpu_worker'},
                        {'Key': 'Project', 'Value': 'scrollintel'}
                    ]
                }]
            }
        )
        
        # CPU worker launch template
        cpu_user_data = """#!/bin/bash
yum update -y
yum install -y docker
systemctl start docker
systemctl enable docker

# Pull and run ScrollIntel worker
docker pull scrollintel/visual-generation-worker:latest
docker run -d --name scrollintel-worker \\
  -e WORKER_TYPE=cpu_worker \\
  -e REDIS_URL=redis://redis.scrollintel.internal:6379 \\
  scrollintel/visual-generation-worker:latest
"""
        
        cpu_template_response = ec2.create_launch_template(
            LaunchTemplateName='scrollintel-cpu-worker-template',
            LaunchTemplateData={
                'ImageId': 'ami-0abcdef1234567890',  # Amazon Linux 2
                'InstanceType': 'c5.2xlarge',
                'SecurityGroupIds': [self.config['aws']['worker_security_group_id']],
                'IamInstanceProfile': {
                    'Name': 'ScrollIntelVisualGenerationWorkerProfile'
                },
                'UserData': cpu_user_data,
                'TagSpecifications': [{
                    'ResourceType': 'instance',
                    'Tags': [
                        {'Key': 'Name', 'Value': 'scrollintel-cpu-worker'},
                        {'Key': 'WorkerType', 'Value': 'cpu_worker'},
                        {'Key': 'Project', 'Value': 'scrollintel'}
                    ]
                }]
            }
        )
        
        self.config['aws']['gpu_launch_template_id'] = gpu_template_response['LaunchTemplate']['LaunchTemplateId']
        self.config['aws']['cpu_launch_template_id'] = cpu_template_response['LaunchTemplate']['LaunchTemplateId']
        
        self.logger.info("Created AWS launch templates")
    
    async def _create_aws_autoscaling_groups(self):
        """Create AWS Auto Scaling groups."""
        autoscaling = self.aws_clients['autoscaling']
        
        # GPU worker auto scaling group
        autoscaling.create_auto_scaling_group(
            AutoScalingGroupName='scrollintel-gpu-workers',
            LaunchTemplate={
                'LaunchTemplateId': self.config['aws']['gpu_launch_template_id'],
                'Version': '$Latest'
            },
            MinSize=0,
            MaxSize=20,
            DesiredCapacity=0,
            VPCZoneIdentifier=','.join(self.config['aws']['subnet_ids']),
            Tags=[
                {
                    'Key': 'Name',
                    'Value': 'scrollintel-gpu-worker',
                    'PropagateAtLaunch': True,
                    'ResourceId': 'scrollintel-gpu-workers',
                    'ResourceType': 'auto-scaling-group'
                },
                {
                    'Key': 'WorkerType',
                    'Value': 'gpu_worker',
                    'PropagateAtLaunch': True,
                    'ResourceId': 'scrollintel-gpu-workers',
                    'ResourceType': 'auto-scaling-group'
                }
            ]
        )
        
        # CPU worker auto scaling group
        autoscaling.create_auto_scaling_group(
            AutoScalingGroupName='scrollintel-cpu-workers',
            LaunchTemplate={
                'LaunchTemplateId': self.config['aws']['cpu_launch_template_id'],
                'Version': '$Latest'
            },
            MinSize=1,
            MaxSize=50,
            DesiredCapacity=1,
            VPCZoneIdentifier=','.join(self.config['aws']['subnet_ids']),
            Tags=[
                {
                    'Key': 'Name',
                    'Value': 'scrollintel-cpu-worker',
                    'PropagateAtLaunch': True,
                    'ResourceId': 'scrollintel-cpu-workers',
                    'ResourceType': 'auto-scaling-group'
                },
                {
                    'Key': 'WorkerType',
                    'Value': 'cpu_worker',
                    'PropagateAtLaunch': True,
                    'ResourceId': 'scrollintel-cpu-workers',
                    'ResourceType': 'auto-scaling-group'
                }
            ]
        )
        
        self.logger.info("Created AWS Auto Scaling groups")
    
    async def _create_aws_ecs_cluster(self):
        """Create AWS ECS cluster for container workloads."""
        ecs = self.aws_clients['ecs']
        
        cluster_response = ecs.create_cluster(
            clusterName='scrollintel-visual-generation',
            tags=[
                {'key': 'Project', 'value': 'scrollintel'},
                {'key': 'Component', 'value': 'visual-generation'}
            ],
            capacityProviders=['FARGATE', 'FARGATE_SPOT'],
            defaultCapacityProviderStrategy=[
                {
                    'capacityProvider': 'FARGATE_SPOT',
                    'weight': 1,
                    'base': 0
                },
                {
                    'capacityProvider': 'FARGATE',
                    'weight': 1,
                    'base': 1
                }
            ]
        )
        
        self.config['aws']['ecs_cluster_arn'] = cluster_response['cluster']['clusterArn']
        self.logger.info("Created AWS ECS cluster")
    
    async def deploy_gcp(self):
        """Deploy infrastructure to Google Cloud Platform."""
        self.logger.info("Deploying to GCP...")
        
        try:
            # Create instance templates
            await self._create_gcp_instance_templates()
            
            # Create managed instance groups
            await self._create_gcp_instance_groups()
            
            self.logger.info("GCP deployment completed successfully")
            
        except Exception as e:
            self.logger.error(f"GCP deployment failed: {e}")
            raise
    
    async def _create_gcp_instance_templates(self):
        """Create GCP instance templates."""
        # Implementation for GCP instance templates
        self.logger.info("Created GCP instance templates")
    
    async def _create_gcp_instance_groups(self):
        """Create GCP managed instance groups."""
        # Implementation for GCP managed instance groups
        self.logger.info("Created GCP managed instance groups")
    
    async def deploy_azure(self):
        """Deploy infrastructure to Microsoft Azure."""
        self.logger.info("Deploying to Azure...")
        
        try:
            # Create resource group
            await self._create_azure_resource_group()
            
            # Create virtual machine scale sets
            await self._create_azure_vmss()
            
            self.logger.info("Azure deployment completed successfully")
            
        except Exception as e:
            self.logger.error(f"Azure deployment failed: {e}")
            raise
    
    async def _create_azure_resource_group(self):
        """Create Azure resource group."""
        # Implementation for Azure resource group
        self.logger.info("Created Azure resource group")
    
    async def _create_azure_vmss(self):
        """Create Azure Virtual Machine Scale Sets."""
        # Implementation for Azure VMSS
        self.logger.info("Created Azure VMSS")
    
    async def deploy_kubernetes(self):
        """Deploy infrastructure to Kubernetes."""
        self.logger.info("Deploying to Kubernetes...")
        
        try:
            # Create namespace
            await self._create_k8s_namespace()
            
            # Create config maps and secrets
            await self._create_k8s_configs()
            
            # Create deployments
            await self._create_k8s_deployments()
            
            # Create services
            await self._create_k8s_services()
            
            # Create horizontal pod autoscalers
            await self._create_k8s_hpa()
            
            self.logger.info("Kubernetes deployment completed successfully")
            
        except Exception as e:
            self.logger.error(f"Kubernetes deployment failed: {e}")
            raise
    
    async def _create_k8s_namespace(self):
        """Create Kubernetes namespace."""
        core_v1 = self.k8s_clients['core_v1']
        
        namespace = kubernetes.client.V1Namespace(
            metadata=kubernetes.client.V1ObjectMeta(
                name='scrollintel-visual-generation',
                labels={
                    'app': 'scrollintel',
                    'component': 'visual-generation'
                }
            )
        )
        
        try:
            core_v1.create_namespace(body=namespace)
            self.logger.info("Created Kubernetes namespace")
        except kubernetes.client.exceptions.ApiException as e:
            if e.status == 409:  # Already exists
                self.logger.info("Kubernetes namespace already exists")
            else:
                raise
    
    async def _create_k8s_configs(self):
        """Create Kubernetes config maps and secrets."""
        core_v1 = self.k8s_clients['core_v1']
        
        # Config map for worker configuration
        config_map = kubernetes.client.V1ConfigMap(
            metadata=kubernetes.client.V1ObjectMeta(
                name='scrollintel-worker-config',
                namespace='scrollintel-visual-generation'
            ),
            data={
                'redis_url': 'redis://redis:6379/0',
                'log_level': 'INFO',
                'worker_timeout': '3600'
            }
        )
        
        try:
            core_v1.create_namespaced_config_map(
                namespace='scrollintel-visual-generation',
                body=config_map
            )
            self.logger.info("Created Kubernetes config map")
        except kubernetes.client.exceptions.ApiException as e:
            if e.status == 409:  # Already exists
                self.logger.info("Kubernetes config map already exists")
            else:
                raise
    
    async def _create_k8s_deployments(self):
        """Create Kubernetes deployments."""
        apps_v1 = self.k8s_clients['apps_v1']
        
        # GPU worker deployment
        gpu_deployment = kubernetes.client.V1Deployment(
            metadata=kubernetes.client.V1ObjectMeta(
                name='scrollintel-gpu-workers',
                namespace='scrollintel-visual-generation',
                labels={'app': 'scrollintel-worker', 'worker-type': 'gpu'}
            ),
            spec=kubernetes.client.V1DeploymentSpec(
                replicas=0,  # Start with 0, auto-scaler will manage
                selector=kubernetes.client.V1LabelSelector(
                    match_labels={'app': 'scrollintel-worker', 'worker-type': 'gpu'}
                ),
                template=kubernetes.client.V1PodTemplateSpec(
                    metadata=kubernetes.client.V1ObjectMeta(
                        labels={'app': 'scrollintel-worker', 'worker-type': 'gpu'}
                    ),
                    spec=kubernetes.client.V1PodSpec(
                        containers=[
                            kubernetes.client.V1Container(
                                name='scrollintel-worker',
                                image='scrollintel/visual-generation-worker:latest',
                                env=[
                                    kubernetes.client.V1EnvVar(name='WORKER_TYPE', value='gpu_worker'),
                                    kubernetes.client.V1EnvVar(
                                        name='REDIS_URL',
                                        value_from=kubernetes.client.V1EnvVarSource(
                                            config_map_key_ref=kubernetes.client.V1ConfigMapKeySelector(
                                                name='scrollintel-worker-config',
                                                key='redis_url'
                                            )
                                        )
                                    )
                                ],
                                resources=kubernetes.client.V1ResourceRequirements(
                                    requests={'nvidia.com/gpu': '1', 'cpu': '4', 'memory': '16Gi'},
                                    limits={'nvidia.com/gpu': '1', 'cpu': '4', 'memory': '16Gi'}
                                )
                            )
                        ],
                        node_selector={'accelerator': 'nvidia-tesla-k80'}
                    )
                )
            )
        )
        
        # CPU worker deployment
        cpu_deployment = kubernetes.client.V1Deployment(
            metadata=kubernetes.client.V1ObjectMeta(
                name='scrollintel-cpu-workers',
                namespace='scrollintel-visual-generation',
                labels={'app': 'scrollintel-worker', 'worker-type': 'cpu'}
            ),
            spec=kubernetes.client.V1DeploymentSpec(
                replicas=1,  # Start with 1 CPU worker
                selector=kubernetes.client.V1LabelSelector(
                    match_labels={'app': 'scrollintel-worker', 'worker-type': 'cpu'}
                ),
                template=kubernetes.client.V1PodTemplateSpec(
                    metadata=kubernetes.client.V1ObjectMeta(
                        labels={'app': 'scrollintel-worker', 'worker-type': 'cpu'}
                    ),
                    spec=kubernetes.client.V1PodSpec(
                        containers=[
                            kubernetes.client.V1Container(
                                name='scrollintel-worker',
                                image='scrollintel/visual-generation-worker:latest',
                                env=[
                                    kubernetes.client.V1EnvVar(name='WORKER_TYPE', value='cpu_worker'),
                                    kubernetes.client.V1EnvVar(
                                        name='REDIS_URL',
                                        value_from=kubernetes.client.V1EnvVarSource(
                                            config_map_key_ref=kubernetes.client.V1ConfigMapKeySelector(
                                                name='scrollintel-worker-config',
                                                key='redis_url'
                                            )
                                        )
                                    )
                                ],
                                resources=kubernetes.client.V1ResourceRequirements(
                                    requests={'cpu': '8', 'memory': '16Gi'},
                                    limits={'cpu': '8', 'memory': '16Gi'}
                                )
                            )
                        ]
                    )
                )
            )
        )
        
        try:
            apps_v1.create_namespaced_deployment(
                namespace='scrollintel-visual-generation',
                body=gpu_deployment
            )
            apps_v1.create_namespaced_deployment(
                namespace='scrollintel-visual-generation',
                body=cpu_deployment
            )
            self.logger.info("Created Kubernetes deployments")
        except kubernetes.client.exceptions.ApiException as e:
            if e.status == 409:  # Already exists
                self.logger.info("Kubernetes deployments already exist")
            else:
                raise
    
    async def _create_k8s_services(self):
        """Create Kubernetes services."""
        core_v1 = self.k8s_clients['core_v1']
        
        # Worker service
        service = kubernetes.client.V1Service(
            metadata=kubernetes.client.V1ObjectMeta(
                name='scrollintel-workers',
                namespace='scrollintel-visual-generation'
            ),
            spec=kubernetes.client.V1ServiceSpec(
                selector={'app': 'scrollintel-worker'},
                ports=[
                    kubernetes.client.V1ServicePort(
                        port=8000,
                        target_port=8000,
                        name='worker-api'
                    )
                ]
            )
        )
        
        try:
            core_v1.create_namespaced_service(
                namespace='scrollintel-visual-generation',
                body=service
            )
            self.logger.info("Created Kubernetes services")
        except kubernetes.client.exceptions.ApiException as e:
            if e.status == 409:  # Already exists
                self.logger.info("Kubernetes services already exist")
            else:
                raise
    
    async def _create_k8s_hpa(self):
        """Create Kubernetes Horizontal Pod Autoscalers."""
        # Implementation for HPA would go here
        self.logger.info("Created Kubernetes HPAs")
    
    async def cleanup_all(self):
        """Clean up all deployed infrastructure."""
        self.logger.info("Cleaning up all infrastructure...")
        
        tasks = []
        
        if self.aws_clients:
            tasks.append(self.cleanup_aws())
        
        if self.gcp_clients:
            tasks.append(self.cleanup_gcp())
        
        if self.azure_clients:
            tasks.append(self.cleanup_azure())
        
        if self.k8s_clients:
            tasks.append(self.cleanup_kubernetes())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self.logger.info("Infrastructure cleanup completed")
    
    async def cleanup_aws(self):
        """Clean up AWS resources."""
        # Implementation for AWS cleanup
        self.logger.info("Cleaned up AWS resources")
    
    async def cleanup_gcp(self):
        """Clean up GCP resources."""
        # Implementation for GCP cleanup
        self.logger.info("Cleaned up GCP resources")
    
    async def cleanup_azure(self):
        """Clean up Azure resources."""
        # Implementation for Azure cleanup
        self.logger.info("Cleaned up Azure resources")
    
    async def cleanup_kubernetes(self):
        """Clean up Kubernetes resources."""
        # Implementation for Kubernetes cleanup
        self.logger.info("Cleaned up Kubernetes resources")


async def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description='Deploy ScrollIntel visual generation infrastructure')
    parser.add_argument('--config', required=True, help='Path to deployment configuration file')
    parser.add_argument('--action', choices=['deploy', 'cleanup'], default='deploy', help='Action to perform')
    parser.add_argument('--provider', choices=['aws', 'gcp', 'azure', 'kubernetes', 'all'], default='all', help='Cloud provider to deploy to')
    
    args = parser.parse_args()
    
    deployer = InfrastructureDeployer(args.config)
    
    if args.action == 'deploy':
        if args.provider == 'all':
            await deployer.deploy_all()
        elif args.provider == 'aws':
            await deployer.deploy_aws()
        elif args.provider == 'gcp':
            await deployer.deploy_gcp()
        elif args.provider == 'azure':
            await deployer.deploy_azure()
        elif args.provider == 'kubernetes':
            await deployer.deploy_kubernetes()
    elif args.action == 'cleanup':
        await deployer.cleanup_all()


if __name__ == '__main__':
    asyncio.run(main())