variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  validation {
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "Environment must be one of: development, staging, production."
  }
}

variable "cluster_version" {
  description = "Kubernetes version to use for the EKS cluster"
  type        = string
  default     = "1.27"
}

variable "node_groups" {
  description = "EKS node groups configuration"
  type = map(object({
    instance_types = list(string)
    min_size      = number
    max_size      = number
    desired_size  = number
    labels        = map(string)
    taints = map(object({
      key    = string
      value  = string
      effect = string
    }))
  }))
  default = {
    core = {
      instance_types = ["m6i.2xlarge"]
      min_size      = 3
      max_size      = 10
      desired_size  = 6
      labels = {
        role     = "core"
        workload = "agent-steering"
      }
      taints = {
        dedicated = {
          key    = "core"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }
    }
    intelligence = {
      instance_types = ["g5.2xlarge"]
      min_size      = 2
      max_size      = 8
      desired_size  = 4
      labels = {
        role     = "intelligence"
        workload = "ml-inference"
      }
      taints = {
        dedicated = {
          key    = "intelligence"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }
    }
  }
}

variable "database_config" {
  description = "RDS PostgreSQL configuration"
  type = object({
    instance_class        = string
    allocated_storage     = number
    max_allocated_storage = number
    engine_version        = string
    backup_retention      = number
    multi_az             = bool
  })
  default = {
    instance_class        = "db.r6g.2xlarge"
    allocated_storage     = 100
    max_allocated_storage = 1000
    engine_version        = "15.3"
    backup_retention      = 7
    multi_az             = true
  }
}

variable "redis_config" {
  description = "ElastiCache Redis configuration"
  type = object({
    node_type              = string
    num_cache_clusters     = number
    parameter_group_name   = string
    engine_version         = string
  })
  default = {
    node_type              = "cache.r7g.xlarge"
    num_cache_clusters     = 3
    parameter_group_name   = "default.redis7"
    engine_version         = "7.0"
  }
}

variable "monitoring_config" {
  description = "Monitoring and observability configuration"
  type = object({
    prometheus_retention_days = number
    grafana_admin_password   = string
    log_retention_days       = number
    enable_detailed_monitoring = bool
  })
  default = {
    prometheus_retention_days = 15
    grafana_admin_password   = "change-me-in-production"
    log_retention_days       = 30
    enable_detailed_monitoring = true
  }
  sensitive = true
}

variable "security_config" {
  description = "Security configuration"
  type = object({
    enable_encryption_at_rest = bool
    enable_encryption_in_transit = bool
    enable_network_policies = bool
    allowed_cidr_blocks = list(string)
  })
  default = {
    enable_encryption_at_rest = true
    enable_encryption_in_transit = true
    enable_network_policies = true
    allowed_cidr_blocks = ["0.0.0.0/0"]
  }
}

variable "backup_config" {
  description = "Backup and disaster recovery configuration"
  type = object({
    backup_retention_days = number
    backup_window        = string
    maintenance_window   = string
    enable_point_in_time_recovery = bool
  })
  default = {
    backup_retention_days = 7
    backup_window        = "03:00-06:00"
    maintenance_window   = "Mon:00:00-Mon:03:00"
    enable_point_in_time_recovery = true
  }
}

variable "scaling_config" {
  description = "Auto-scaling configuration"
  type = object({
    enable_cluster_autoscaler = bool
    enable_horizontal_pod_autoscaler = bool
    enable_vertical_pod_autoscaler = bool
    max_nodes_per_group = number
  })
  default = {
    enable_cluster_autoscaler = true
    enable_horizontal_pod_autoscaler = true
    enable_vertical_pod_autoscaler = false
    max_nodes_per_group = 20
  }
}

variable "domain_config" {
  description = "Domain and SSL configuration"
  type = object({
    domain_name = string
    subdomain_api = string
    subdomain_monitoring = string
    enable_ssl = bool
    certificate_arn = string
  })
  default = {
    domain_name = "scrollintel.com"
    subdomain_api = "api.agent-steering"
    subdomain_monitoring = "monitoring.agent-steering"
    enable_ssl = true
    certificate_arn = ""
  }
}

variable "cost_optimization" {
  description = "Cost optimization settings"
  type = object({
    enable_spot_instances = bool
    spot_instance_percentage = number
    enable_scheduled_scaling = bool
    enable_resource_quotas = bool
  })
  default = {
    enable_spot_instances = false
    spot_instance_percentage = 50
    enable_scheduled_scaling = true
    enable_resource_quotas = true
  }
}

variable "compliance_config" {
  description = "Compliance and governance configuration"
  type = object({
    enable_audit_logging = bool
    enable_policy_enforcement = bool
    enable_vulnerability_scanning = bool
    compliance_framework = string
  })
  default = {
    enable_audit_logging = true
    enable_policy_enforcement = true
    enable_vulnerability_scanning = true
    compliance_framework = "SOC2"
  }
}

variable "tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}