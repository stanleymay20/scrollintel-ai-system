#!/usr/bin/env python3
"""
Health Check Script for Agent Steering System
Performs comprehensive health checks on all system components
"""

import asyncio
import aiohttp
import argparse
import json
import logging
import sys
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"

@dataclass
class HealthCheckResult:
    service: str
    status: HealthStatus
    response_time: float
    details: Dict
    error: Optional[str] = None

class HealthChecker:
    def __init__(self, environment: str = "production", timeout: int = 30):
        self.environment = environment
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Service endpoints based on environment
        self.endpoints = self._get_endpoints()
    
    def _get_endpoints(self) -> Dict[str, str]:
        """Get service endpoints based on environment"""
        if self.environment == "production":
            return {
                "orchestration": "https://api.agent-steering.scrollintel.com/api/v1/orchestration/health",
                "intelligence": "https://api.agent-steering.scrollintel.com/api/v1/intelligence/health",
                "prometheus": "https://monitoring.agent-steering.scrollintel.com/api/v1/query?query=up",
                "grafana": "https://monitoring.agent-steering.scrollintel.com/api/health"
            }
        elif self.environment == "staging":
            return {
                "orchestration": "https://api.staging.agent-steering.scrollintel.com/api/v1/orchestration/health",
                "intelligence": "https://api.staging.agent-steering.scrollintel.com/api/v1/intelligence/health",
                "prometheus": "https://monitoring.staging.agent-steering.scrollintel.com/api/v1/query?query=up",
                "grafana": "https://monitoring.staging.agent-steering.scrollintel.com/api/health"
            }
        else:  # development or local
            return {
                "orchestration": "http://localhost:8080/health",
                "intelligence": "http://localhost:8081/health",
                "prometheus": "http://localhost:9090/api/v1/query?query=up",
                "grafana": "http://localhost:3000/api/health"
            }
    
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": "AgentSteering-HealthChecker/1.0"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def check_service_health(self, service: str, url: str) -> HealthCheckResult:
        """Check health of a single service"""
        start_time = time.time()
        
        try:
            async with self.session.get(url) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    try:
                        data = await response.json()
                        return HealthCheckResult(
                            service=service,
                            status=HealthStatus.HEALTHY,
                            response_time=response_time,
                            details=data
                        )
                    except json.JSONDecodeError:
                        # Some services return plain text
                        text = await response.text()
                        return HealthCheckResult(
                            service=service,
                            status=HealthStatus.HEALTHY,
                            response_time=response_time,
                            details={"response": text}
                        )
                else:
                    return HealthCheckResult(
                        service=service,
                        status=HealthStatus.UNHEALTHY,
                        response_time=response_time,
                        details={"status_code": response.status},
                        error=f"HTTP {response.status}"
                    )
        
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            return HealthCheckResult(
                service=service,
                status=HealthStatus.UNHEALTHY,
                response_time=response_time,
                details={},
                error="Timeout"
            )
        
        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheckResult(
                service=service,
                status=HealthStatus.UNHEALTHY,
                response_time=response_time,
                details={},
                error=str(e)
            )
    
    async def check_database_connectivity(self) -> HealthCheckResult:
        """Check database connectivity through orchestration service"""
        try:
            url = self.endpoints["orchestration"].replace("/health", "/health/database")
            return await self.check_service_health("database", url)
        except Exception as e:
            return HealthCheckResult(
                service="database",
                status=HealthStatus.UNKNOWN,
                response_time=0.0,
                details={},
                error=str(e)
            )
    
    async def check_cache_connectivity(self) -> HealthCheckResult:
        """Check Redis cache connectivity"""
        try:
            url = self.endpoints["orchestration"].replace("/health", "/health/cache")
            return await self.check_service_health("cache", url)
        except Exception as e:
            return HealthCheckResult(
                service="cache",
                status=HealthStatus.UNKNOWN,
                response_time=0.0,
                details={},
                error=str(e)
            )
    
    async def check_message_queue(self) -> HealthCheckResult:
        """Check Kafka message queue connectivity"""
        try:
            url = self.endpoints["orchestration"].replace("/health", "/health/kafka")
            return await self.check_service_health("kafka", url)
        except Exception as e:
            return HealthCheckResult(
                service="kafka",
                status=HealthStatus.UNKNOWN,
                response_time=0.0,
                details={},
                error=str(e)
            )
    
    async def check_all_services(self) -> List[HealthCheckResult]:
        """Check health of all services"""
        tasks = []
        
        # Core services
        for service, url in self.endpoints.items():
            tasks.append(self.check_service_health(service, url))
        
        # Infrastructure services
        tasks.extend([
            self.check_database_connectivity(),
            self.check_cache_connectivity(),
            self.check_message_queue()
        ])
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        health_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                service_name = list(self.endpoints.keys())[i] if i < len(self.endpoints) else "infrastructure"
                health_results.append(HealthCheckResult(
                    service=service_name,
                    status=HealthStatus.UNHEALTHY,
                    response_time=0.0,
                    details={},
                    error=str(result)
                ))
            else:
                health_results.append(result)
        
        return health_results
    
    def print_results(self, results: List[HealthCheckResult]) -> bool:
        """Print health check results and return overall health status"""
        print(f"\n{'='*60}")
        print(f"Agent Steering System Health Check - {self.environment.upper()}")
        print(f"{'='*60}")
        
        healthy_count = 0
        total_count = len(results)
        
        for result in results:
            status_color = {
                HealthStatus.HEALTHY: "\033[92m",  # Green
                HealthStatus.DEGRADED: "\033[93m",  # Yellow
                HealthStatus.UNHEALTHY: "\033[91m",  # Red
                HealthStatus.UNKNOWN: "\033[94m"  # Blue
            }.get(result.status, "\033[0m")
            
            reset_color = "\033[0m"
            
            print(f"\n{result.service.upper():<15} {status_color}{result.status.value:<10}{reset_color} "
                  f"({result.response_time:.3f}s)")
            
            if result.error:
                print(f"  Error: {result.error}")
            
            if result.details and result.status == HealthStatus.HEALTHY:
                if "version" in result.details:
                    print(f"  Version: {result.details['version']}")
                if "uptime" in result.details:
                    print(f"  Uptime: {result.details['uptime']}")
            
            if result.status == HealthStatus.HEALTHY:
                healthy_count += 1
        
        print(f"\n{'='*60}")
        
        overall_health = healthy_count == total_count
        health_percentage = (healthy_count / total_count) * 100
        
        if overall_health:
            print(f"\033[92m✓ SYSTEM HEALTHY\033[0m - All services operational ({health_percentage:.0f}%)")
        elif health_percentage >= 80:
            print(f"\033[93m⚠ SYSTEM DEGRADED\033[0m - Some services down ({health_percentage:.0f}%)")
        else:
            print(f"\033[91m✗ SYSTEM UNHEALTHY\033[0m - Critical services down ({health_percentage:.0f}%)")
        
        print(f"Services: {healthy_count}/{total_count} healthy")
        print(f"{'='*60}\n")
        
        return overall_health

async def main():
    parser = argparse.ArgumentParser(description="Agent Steering System Health Checker")
    parser.add_argument(
        "--environment", "-e",
        choices=["production", "staging", "development"],
        default="production",
        help="Environment to check"
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=30,
        help="Request timeout in seconds"
    )
    parser.add_argument(
        "--continuous", "-c",
        action="store_true",
        help="Run continuous health checks"
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=60,
        help="Interval between checks in continuous mode (seconds)"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results in JSON format"
    )
    
    args = parser.parse_args()
    
    async with HealthChecker(args.environment, args.timeout) as checker:
        if args.continuous:
            logger.info(f"Starting continuous health checks every {args.interval} seconds...")
            try:
                while True:
                    results = await checker.check_all_services()
                    
                    if args.json:
                        output = {
                            "timestamp": time.time(),
                            "environment": args.environment,
                            "results": [
                                {
                                    "service": r.service,
                                    "status": r.status.value,
                                    "response_time": r.response_time,
                                    "details": r.details,
                                    "error": r.error
                                }
                                for r in results
                            ]
                        }
                        print(json.dumps(output, indent=2))
                    else:
                        overall_health = checker.print_results(results)
                    
                    await asyncio.sleep(args.interval)
            
            except KeyboardInterrupt:
                logger.info("Health check monitoring stopped")
        
        else:
            # Single health check
            results = await checker.check_all_services()
            
            if args.json:
                output = {
                    "timestamp": time.time(),
                    "environment": args.environment,
                    "results": [
                        {
                            "service": r.service,
                            "status": r.status.value,
                            "response_time": r.response_time,
                            "details": r.details,
                            "error": r.error
                        }
                        for r in results
                    ]
                }
                print(json.dumps(output, indent=2))
            else:
                overall_health = checker.print_results(results)
            
            # Exit with appropriate code
            sys.exit(0 if overall_health else 1)

if __name__ == "__main__":
    asyncio.run(main())