terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
  }

  backend "s3" {
    bucket         = "scrollintel-terraform-state"
    key            = "agent-steering-system/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "agent-steering-system"
      Environment = var.environment
      ManagedBy   = "terraform"
      Owner       = "scrollintel"
    }
  }
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
  }
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    
    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

data "aws_caller_identity" "current" {}

# Local values
locals {
  name   = "agent-steering-${var.environment}"
  region = var.aws_region

  vpc_cidr = "10.0.0.0/16"
  azs      = slice(data.aws_availability_zones.available.names, 0, 3)

  tags = {
    Project     = "agent-steering-system"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# VPC Module
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = local.name
  cidr = local.vpc_cidr

  azs             = local.azs
  private_subnets = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, 4, k)]
  public_subnets  = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, 8, k + 48)]
  intra_subnets   = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, 8, k + 52)]

  enable_nat_gateway = true
  single_nat_gateway = false
  enable_dns_hostnames = true
  enable_dns_support = true

  # VPC Flow Logs
  enable_flow_log                      = true
  create_flow_log_cloudwatch_iam_role  = true
  create_flow_log_cloudwatch_log_group = true

  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1
  }

  tags = local.tags
}

# EKS Module
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  version = "~> 19.15"

  cluster_name    = local.name
  cluster_version = "1.27"

  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true

  # EKS Managed Node Groups
  eks_managed_node_groups = {
    # Core system nodes
    core = {
      name = "core-nodes"
      
      instance_types = ["m6i.2xlarge"]
      
      min_size     = 3
      max_size     = 10
      desired_size = 6

      pre_bootstrap_user_data = <<-EOT
        #!/bin/bash
        /etc/eks/bootstrap.sh ${local.name}
      EOT

      vpc_security_group_ids = [aws_security_group.node_group_one.id]
      
      labels = {
        role = "core"
        workload = "agent-steering"
      }

      taints = {
        dedicated = {
          key    = "core"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }

      tags = {
        ExtraTag = "core-nodes"
      }
    }

    # Intelligence engine nodes (GPU-enabled)
    intelligence = {
      name = "intelligence-nodes"
      
      instance_types = ["g5.2xlarge"]
      
      min_size     = 2
      max_size     = 8
      desired_size = 4

      labels = {
        role = "intelligence"
        workload = "ml-inference"
        "nvidia.com/gpu" = "true"
      }

      taints = {
        dedicated = {
          key    = "intelligence"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }

      tags = {
        ExtraTag = "intelligence-nodes"
      }
    }

    # Monitoring and observability nodes
    monitoring = {
      name = "monitoring-nodes"
      
      instance_types = ["m6i.xlarge"]
      
      min_size     = 2
      max_size     = 6
      desired_size = 3

      labels = {
        role = "monitoring"
        workload = "observability"
      }

      tags = {
        ExtraTag = "monitoring-nodes"
      }
    }
  }

  # Cluster access entry
  access_entries = {
    admin = {
      kubernetes_groups = []
      principal_arn     = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"

      policy_associations = {
        admin = {
          policy_arn = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"
          access_scope = {
            type = "cluster"
          }
        }
      }
    }
  }

  tags = local.tags
}

# Additional security group for node groups
resource "aws_security_group" "node_group_one" {
  name_prefix = "${local.name}-node-group-one"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port = 22
    to_port   = 22
    protocol  = "tcp"

    cidr_blocks = [
      "10.0.0.0/8",
    ]
  }

  tags = local.tags
}

# RDS PostgreSQL for persistent data
module "db" {
  source = "terraform-aws-modules/rds/aws"
  version = "~> 6.0"

  identifier = "${local.name}-postgres"

  engine            = "postgres"
  engine_version    = "15.3"
  instance_class    = "db.r6g.2xlarge"
  allocated_storage = 100
  max_allocated_storage = 1000

  storage_encrypted = true

  db_name  = "agent_steering"
  username = "postgres"
  port     = "5432"

  iam_database_authentication_enabled = true

  vpc_security_group_ids = [aws_security_group.rds.id]

  maintenance_window = "Mon:00:00-Mon:03:00"
  backup_window      = "03:00-06:00"

  # Enhanced Monitoring
  monitoring_interval    = "60"
  monitoring_role_name   = "${local.name}-rds-monitoring-role"
  create_monitoring_role = true

  tags = local.tags

  # DB subnet group
  create_db_subnet_group = true
  subnet_ids             = module.vpc.private_subnets

  # DB parameter group
  family = "postgres15"

  # DB option group
  major_engine_version = "15"

  # Database Deletion Protection
  deletion_protection = var.environment == "production" ? true : false

  parameters = [
    {
      name  = "log_checkpoints"
      value = 1
    },
    {
      name  = "log_connections"
      value = 1
    },
    {
      name  = "log_disconnections"
      value = 1
    },
    {
      name  = "log_lock_waits"
      value = 1
    },
    {
      name  = "log_temp_files"
      value = 0
    },
    {
      name  = "slow_query_log"
      value = 1
    }
  ]
}

# Security group for RDS
resource "aws_security_group" "rds" {
  name_prefix = "${local.name}-rds"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.tags
}

# ElastiCache Redis for caching
resource "aws_elasticache_subnet_group" "redis" {
  name       = "${local.name}-redis"
  subnet_ids = module.vpc.private_subnets

  tags = local.tags
}

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id         = "${local.name}-redis"
  description                  = "Redis cluster for agent steering system"
  
  node_type                    = "cache.r7g.xlarge"
  port                         = 6379
  parameter_group_name         = "default.redis7"
  
  num_cache_clusters           = 3
  automatic_failover_enabled   = true
  multi_az_enabled            = true
  
  subnet_group_name           = aws_elasticache_subnet_group.redis.name
  security_group_ids          = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled  = true
  transit_encryption_enabled  = true
  
  maintenance_window          = "sun:05:00-sun:09:00"
  snapshot_retention_limit    = 5
  snapshot_window             = "09:10-10:10"
  
  tags = local.tags
}

# Security group for Redis
resource "aws_security_group" "redis" {
  name_prefix = "${local.name}-redis"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.tags
}

# S3 bucket for artifacts and backups
resource "aws_s3_bucket" "artifacts" {
  bucket = "${local.name}-artifacts-${random_string.bucket_suffix.result}"

  tags = local.tags
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "orchestration" {
  name              = "/aws/eks/${local.name}/orchestration"
  retention_in_days = 30

  tags = local.tags
}

resource "aws_cloudwatch_log_group" "intelligence" {
  name              = "/aws/eks/${local.name}/intelligence"
  retention_in_days = 30

  tags = local.tags
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "${local.name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = module.vpc.public_subnets

  enable_deletion_protection = var.environment == "production" ? true : false

  tags = local.tags
}

resource "aws_security_group" "alb" {
  name_prefix = "${local.name}-alb"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.tags
}