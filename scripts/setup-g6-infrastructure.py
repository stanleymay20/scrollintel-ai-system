#!/usr/bin/env python3
"""
Setup script for ScrollIntel-G6 Core Infrastructure
"""
import subprocess
import time
import requests
import logging
import sys
import os
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InfrastructureSetup:
    """Setup and initialize ScrollIntel-G6 infrastructure"""
    
    def __init__(self):
        self.services_health = {}
    
    def run_command(self, command: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run shell command"""
        logger.info(f"Running: {command}")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if check and result.returncode != 0:
            logger.error(f"Command failed: {command}")
            logger.error(f"Error: {result.stderr}")
            sys.exit(1)
        
        return result
    
    def wait_for_service(self, name: str, url: str, timeout: int = 300) -> bool:
        """Wait for service to be healthy"""
        logger.info(f"Waiting for {name} to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"{name} is ready!")
                    self.services_health[name] = True
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(5)
        
        logger.error(f"{name} failed to start within {timeout} seconds")
        self.services_health[name] = False
        return False
    
    def setup_docker_infrastructure(self):
        """Setup Docker infrastructure"""
        logger.info("Setting up Docker infrastructure...")
        
        # Stop any existing containers
        self.run_command("docker-compose -f docker-compose-g6-infrastructure.yml down", check=False)
        
        # Start infrastructure services
        self.run_command("docker-compose -f docker-compose-g6-infrastructure.yml up -d")
        
        # Wait for services to be ready
        services_to_check = [
            ("PostgreSQL", "http://localhost:5432"),  # Will fail but that's expected
            ("Elasticsearch", "http://localhost:9200"),
            ("Neo4j", "http://localhost:7474"),
            ("Prometheus", "http://localhost:9090"),
            ("Grafana", "http://localhost:3000"),
        ]
        
        # Check Elasticsearch
        self.wait_for_service("Elasticsearch", "http://localhost:9200")
        
        # Check Neo4j
        self.wait_for_service("Neo4j", "http://localhost:7474")
        
        # Check Prometheus
        self.wait_for_service("Prometheus", "http://localhost:9090")
        
        # Check Grafana
        self.wait_for_service("Grafana", "http://localhost:3000")
    
    def setup_redis_cluster(self):
        """Setup Redis cluster"""
        logger.info("Setting up Redis cluster...")
        
        # Wait for Redis nodes to start
        time.sleep(10)
        
        # Create Redis cluster
        cluster_command = """
        docker exec scrollintel-redis-1 redis-cli --cluster create \
        172.20.0.2:6379 172.20.0.3:6379 172.20.0.4:6379 \
        --cluster-replicas 0 --cluster-yes
        """
        
        result = self.run_command(cluster_command, check=False)
        if result.returncode == 0:
            logger.info("Redis cluster created successfully")
            self.services_health["Redis Cluster"] = True
        else:
            logger.warning("Redis cluster creation failed, continuing with single nodes")
            self.services_health["Redis Cluster"] = False
    
    def setup_databases(self):
        """Setup databases and initial schemas"""
        logger.info("Setting up databases...")
        
        # Wait for PostgreSQL to be ready
        time.sleep(15)
        
        # Create database schemas
        sql_commands = [
            "CREATE EXTENSION IF NOT EXISTS 'uuid-ossp';",
            """
            CREATE TABLE IF NOT EXISTS data_products (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                name VARCHAR(255) NOT NULL,
                version VARCHAR(50) NOT NULL,
                schema_definition JSONB,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                username VARCHAR(100) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                roles JSONB,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS audit_logs (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                action VARCHAR(100) NOT NULL,
                resource VARCHAR(255) NOT NULL,
                user_id UUID,
                details JSONB,
                ip_address INET
            );
            """
        ]
        
        for sql in sql_commands:
            cmd = f'docker exec scrollintel-postgres psql -U scrollintel -d scrollintel_g6 -c "{sql}"'
            result = self.run_command(cmd, check=False)
            if result.returncode != 0:
                logger.warning(f"SQL command failed: {sql}")
    
    def setup_elasticsearch_indices(self):
        """Setup Elasticsearch indices"""
        logger.info("Setting up Elasticsearch indices...")
        
        indices = [
            {
                "name": "data-products",
                "mapping": {
                    "mappings": {
                        "properties": {
                            "name": {"type": "text", "analyzer": "standard"},
                            "description": {"type": "text", "analyzer": "standard"},
                            "tags": {"type": "keyword"},
                            "schema": {"type": "object"},
                            "created_at": {"type": "date"}
                        }
                    }
                }
            },
            {
                "name": "audit-logs",
                "mapping": {
                    "mappings": {
                        "properties": {
                            "timestamp": {"type": "date"},
                            "action": {"type": "keyword"},
                            "resource": {"type": "keyword"},
                            "user_id": {"type": "keyword"},
                            "details": {"type": "object"}
                        }
                    }
                }
            }
        ]
        
        for index in indices:
            try:
                response = requests.put(
                    f"http://localhost:9200/{index['name']}",
                    json=index["mapping"],
                    timeout=10
                )
                if response.status_code in [200, 201]:
                    logger.info(f"Created Elasticsearch index: {index['name']}")
                else:
                    logger.warning(f"Failed to create index {index['name']}: {response.text}")
            except Exception as e:
                logger.warning(f"Failed to create index {index['name']}: {e}")
    
    def setup_neo4j_constraints(self):
        """Setup Neo4j constraints and indices"""
        logger.info("Setting up Neo4j constraints...")
        
        constraints = [
            "CREATE CONSTRAINT data_product_id IF NOT EXISTS FOR (dp:DataProduct) REQUIRE dp.id IS UNIQUE",
            "CREATE CONSTRAINT transformation_id IF NOT EXISTS FOR (t:Transformation) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT source_id IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE",
        ]
        
        for constraint in constraints:
            cmd = f'docker exec scrollintel-neo4j cypher-shell -u neo4j -p scrollintel_graph_pass "{constraint}"'
            result = self.run_command(cmd, check=False)
            if result.returncode == 0:
                logger.info(f"Created Neo4j constraint: {constraint}")
            else:
                logger.warning(f"Failed to create constraint: {constraint}")
    
    def setup_grafana_datasources(self):
        """Setup Grafana datasources"""
        logger.info("Setting up Grafana datasources...")
        
        # Wait for Grafana to be fully ready
        time.sleep(10)
        
        datasources = [
            {
                "name": "Prometheus",
                "type": "prometheus",
                "url": "http://prometheus:9090",
                "access": "proxy",
                "isDefault": True
            }
        ]
        
        for datasource in datasources:
            try:
                response = requests.post(
                    "http://admin:scrollintel_admin@localhost:3000/api/datasources",
                    json=datasource,
                    timeout=10
                )
                if response.status_code in [200, 201]:
                    logger.info(f"Created Grafana datasource: {datasource['name']}")
                else:
                    logger.warning(f"Failed to create datasource: {response.text}")
            except Exception as e:
                logger.warning(f"Failed to create datasource: {e}")
    
    def verify_infrastructure(self) -> Dict[str, Any]:
        """Verify infrastructure health"""
        logger.info("Verifying infrastructure health...")
        
        health_status = {
            "overall_status": "healthy",
            "services": self.services_health,
            "issues": []
        }
        
        # Check for any failed services
        failed_services = [name for name, status in self.services_health.items() if not status]
        
        if failed_services:
            health_status["overall_status"] = "degraded"
            health_status["issues"] = failed_services
        
        return health_status
    
    def run_setup(self):
        """Run complete infrastructure setup"""
        logger.info("Starting ScrollIntel-G6 infrastructure setup...")
        
        try:
            # Setup Docker infrastructure
            self.setup_docker_infrastructure()
            
            # Setup Redis cluster
            self.setup_redis_cluster()
            
            # Setup databases
            self.setup_databases()
            
            # Setup Elasticsearch
            self.setup_elasticsearch_indices()
            
            # Setup Neo4j
            self.setup_neo4j_constraints()
            
            # Setup Grafana
            self.setup_grafana_datasources()
            
            # Verify everything
            health = self.verify_infrastructure()
            
            logger.info("Infrastructure setup completed!")
            logger.info(f"Health status: {health}")
            
            if health["overall_status"] == "healthy":
                logger.info("✅ All services are healthy and ready!")
                self.print_access_info()
            else:
                logger.warning("⚠️  Some services have issues. Check the logs above.")
                
        except Exception as e:
            logger.error(f"Infrastructure setup failed: {e}")
            sys.exit(1)
    
    def print_access_info(self):
        """Print access information for services"""
        print("\n" + "="*60)
        print("ScrollIntel-G6 Infrastructure Access Information")
        print("="*60)
        print("🔍 Grafana Dashboard: http://localhost:3000")
        print("   Username: admin")
        print("   Password: scrollintel_admin")
        print()
        print("📊 Prometheus: http://localhost:9090")
        print("🔔 AlertManager: http://localhost:9093")
        print("🗄️  Neo4j Browser: http://localhost:7474")
        print("   Username: neo4j")
        print("   Password: scrollintel_graph_pass")
        print()
        print("🔍 Elasticsearch: http://localhost:9200")
        print("📋 Jaeger Tracing: http://localhost:16686")
        print("="*60)

if __name__ == "__main__":
    setup = InfrastructureSetup()
    setup.run_setup()