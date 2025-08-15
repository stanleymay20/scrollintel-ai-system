# RDS Security Group
resource "aws_security_group" "rds_sg" {
  name        = "ai-data-readiness-rds-sg"
  description = "Security group for RDS database"
  vpc_id      = aws_vpc.ai_data_readiness_vpc.id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.ai_data_readiness_vpc.cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "ai-data-readiness-rds-sg"
    Environment = var.environment
  }
}

# ElastiCache Security Group
resource "aws_security_group" "cache_sg" {
  name        = "ai-data-readiness-cache-sg"
  description = "Security group for ElastiCache"
  vpc_id      = aws_vpc.ai_data_readiness_vpc.id

  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.ai_data_readiness_vpc.cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "ai-data-readiness-cache-sg"
    Environment = var.environment
  }
}

# DB Subnet Group
resource "aws_db_subnet_group" "ai_data_readiness_db_subnet_group" {
  name       = "ai-data-readiness-db-subnet-group"
  subnet_ids = aws_subnet.private_subnets[*].id

  tags = {
    Name        = "ai-data-readiness-db-subnet-group"
    Environment = var.environment
  }
}