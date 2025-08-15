output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = aws_eks_cluster.ai_data_readiness_cluster.endpoint
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = aws_eks_cluster.ai_data_readiness_cluster.name
}

output "cluster_arn" {
  description = "EKS cluster ARN"
  value       = aws_eks_cluster.ai_data_readiness_cluster.arn
}

output "database_endpoint" {
  description = "RDS database endpoint"
  value       = aws_db_instance.ai_data_readiness_db.endpoint
}

output "cache_endpoint" {
  description = "ElastiCache endpoint"
  value       = aws_elasticache_cluster.ai_data_readiness_cache.cache_nodes[0].address
}

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.ai_data_readiness_vpc.id
}