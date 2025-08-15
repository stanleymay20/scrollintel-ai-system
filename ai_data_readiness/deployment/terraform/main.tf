terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC Configuration
resource "aws_vpc" "ai_data_readiness_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = "ai-data-readiness-vpc"
    Environment = var.environment
  }
}

# Internet Gateway
resource "aws_internet_gateway" "ai_data_readiness_igw" {
  vpc_id = aws_vpc.ai_data_readiness_vpc.id

  tags = {
    Name        = "ai-data-readiness-igw"
    Environment = var.environment
  }
}

# Public Subnets
resource "aws_subnet" "public_subnets" {
  count             = length(var.availability_zones)
  vpc_id            = aws_vpc.ai_data_readiness_vpc.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = var.availability_zones[count.index]

  map_public_ip_on_launch = true

  tags = {
    Name        = "ai-data-readiness-public-${count.index + 1}"
    Environment = var.environment
  }
}

# Private Subnets
resource "aws_subnet" "private_subnets" {
  count             = length(var.availability_zones)
  vpc_id            = aws_vpc.ai_data_readiness_vpc.id
  route_table_id    = aws_route_table.private_route_table.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = var.availability_zones[count.index]

  tags = {
    Name        = "ai-data-readiness-private-${count.index + 1}"
    Environment = var.environment
  }
}

# Route Tables
resource "aws_route_table" "public_route_table" {
  vpc_id = aws_vpc.ai_data_readiness_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.ai_data_readiness_igw.id
  }

  tags = {
    Name        = "ai-data-readiness-public-rt"
    Environment = var.environment
  }
}

resource "aws_route_table" "private_route_table" {
  vpc_id = aws_vpc.ai_data_readiness_vpc.id

  tags = {
    Name        = "ai-data-readiness-private-rt"
    Environment = var.environment
  }
}

# Route Table Associations
resource "aws_route_table_association" "public_associations" {
  count          = length(aws_subnet.public_subnets)
  subnet_id      = aws_subnet.public_subnets[count.index].id
  route_table_id = aws_route_table.public_route_table.id
}

# EKS Cluster
resource "aws_eks_cluster" "ai_data_readiness_cluster" {
  name     = "ai-data-readiness-cluster"
  role_arn = aws_iam_role.eks_cluster_role.arn
  version  = "1.28"

  vpc_config {
    subnet_ids = concat(aws_subnet.public_subnets[*].id, aws_subnet.private_subnets[*].id)
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
    aws_iam_role_policy_attachment.eks_vpc_resource_controller,
  ]

  tags = {
    Name        = "ai-data-readiness-cluster"
    Environment = var.environment
  }
}

# EKS Node Group
resource "aws_eks_node_group" "ai_data_readiness_nodes" {
  cluster_name    = aws_eks_cluster.ai_data_readiness_cluster.name
  node_group_name = "ai-data-readiness-nodes"
  node_role_arn   = aws_iam_role.eks_node_role.arn
  subnet_ids      = aws_subnet.private_subnets[*].id

  scaling_config {
    desired_size = 3
    max_size     = 10
    min_size     = 1
  }

  instance_types = ["t3.medium"]

  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.eks_container_registry_policy,
  ]

  tags = {
    Name        = "ai-data-readiness-nodes"
    Environment = var.environment
  }
}

# RDS Instance
resource "aws_db_instance" "ai_data_readiness_db" {
  identifier = "ai-data-readiness-db"

  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.t3.micro"

  allocated_storage     = 20
  max_allocated_storage = 100
  storage_type          = "gp2"
  storage_encrypted     = true

  db_name  = "ai_data_readiness"
  username = "postgres"
  password = var.db_password

  vpc_security_group_ids = [aws_security_group.rds_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.ai_data_readiness_db_subnet_group.name

  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  skip_final_snapshot = true

  tags = {
    Name        = "ai-data-readiness-db"
    Environment = var.environment
  }
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "ai_data_readiness_cache_subnet" {
  name       = "ai-data-readiness-cache-subnet"
  subnet_ids = aws_subnet.private_subnets[*].id
}

resource "aws_elasticache_cluster" "ai_data_readiness_cache" {
  cluster_id           = "ai-data-readiness-cache"
  engine               = "redis"
  node_type            = "cache.t3.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.ai_data_readiness_cache_subnet.name
  security_group_ids   = [aws_security_group.cache_sg.id]

  tags = {
    Name        = "ai-data-readiness-cache"
    Environment = var.environment
  }
}