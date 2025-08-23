#!/usr/bin/env python3
"""
Smoke Tests for Agent Steering System
Validates basic functionality after deployment
"""

import asyncio
import aiohttp
import argparse
import json
import logging
import sys
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"

@dataclass
class SmokeTestResult:
    test_name: str
    result: TestResult
    duration: float
    details: Dict[str, Any]
    error: Optional[str] = None

class SmokeTestSuite:
    def __init__(self, environment: str = "production", timeout: int = 30):
        self.environment = environment
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.base_urls = self._get_base_urls()
        self.test_results: List[SmokeTestResult] = []
    
    def _get_base_urls(self) -> Dict[str, str]:
        """Get base URLs based on environment"""
        if self.environment == "production":
            return {
                "api": "https://api.agent-steering.scrollintel.com",
                "monitoring": "https://monitoring.agent-steering.scrollintel.com"
            }
        elif self.environment == "staging":
            return {
                "api": "https://api.staging.agent-steering.scrollintel.com",
                "monitoring": "https://monitoring.staging.agent-steering.scrollintel.com"
            }
        else:  # development or local
            return {
                "api": "http://localhost:8080",
                "monitoring": "http://localhost:3000"
            }
    
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": "AgentSteering-SmokeTests/1.0"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def run_test(self, test_name: str, test_func) -> SmokeTestResult:
        """Run a single test and record the result"""
        start_time = time.time()
        
        try:
            logger.info(f"Running test: {test_name}")
            result = await test_func()
            duration = time.time() - start_time
            
            test_result = SmokeTestResult(
                test_name=test_name,
                result=TestResult.PASS,
                duration=duration,
                details=result if isinstance(result, dict) else {"result": result}
            )
            
            logger.info(f"✓ {test_name} - PASSED ({duration:.3f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            test_result = SmokeTestResult(
                test_name=test_name,
                result=TestResult.FAIL,
                duration=duration,
                details={},
                error=str(e)
            )
            
            logger.error(f"✗ {test_name} - FAILED ({duration:.3f}s): {e}")
        
        self.test_results.append(test_result)
        return test_result
    
    async def test_orchestration_health(self) -> Dict[str, Any]:
        """Test orchestration engine health endpoint"""
        url = f"{self.base_urls['api']}/api/v1/orchestration/health"
        
        async with self.session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Health check failed with status {response.status}")
            
            data = await response.json()
            return {
                "status": data.get("status"),
                "version": data.get("version"),
                "uptime": data.get("uptime")
            }
    
    async def test_intelligence_health(self) -> Dict[str, Any]:
        """Test intelligence engine health endpoint"""
        url = f"{self.base_urls['api']}/api/v1/intelligence/health"
        
        async with self.session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Health check failed with status {response.status}")
            
            data = await response.json()
            return {
                "status": data.get("status"),
                "version": data.get("version"),
                "models_loaded": data.get("models_loaded", 0)
            }
    
    async def test_agent_registration(self) -> Dict[str, Any]:
        """Test agent registration functionality"""
        url = f"{self.base_urls['api']}/api/v1/orchestration/agents/register"
        
        test_agent = {
            "name": "smoke-test-agent",
            "type": "test",
            "capabilities": ["test-capability"],
            "version": "1.0.0"
        }
        
        async with self.session.post(url, json=test_agent) as response:
            if response.status not in [200, 201]:
                raise Exception(f"Agent registration failed with status {response.status}")
            
            data = await response.json()
            agent_id = data.get("agent_id")
            
            if not agent_id:
                raise Exception("No agent ID returned from registration")
            
            # Clean up - deregister the test agent
            deregister_url = f"{self.base_urls['api']}/api/v1/orchestration/agents/{agent_id}/deregister"
            async with self.session.delete(deregister_url) as cleanup_response:
                pass  # Ignore cleanup errors
            
            return {
                "agent_id": agent_id,
                "registration_successful": True
            }
    
    async def test_task_submission(self) -> Dict[str, Any]:
        """Test task submission and execution"""
        url = f"{self.base_urls['api']}/api/v1/orchestration/tasks/submit"
        
        test_task = {
            "title": "Smoke Test Task",
            "description": "Test task for smoke testing",
            "priority": "low",
            "requirements": {
                "capabilities": ["test-capability"],
                "timeout": 30
            }
        }
        
        async with self.session.post(url, json=test_task) as response:
            if response.status not in [200, 201, 202]:
                raise Exception(f"Task submission failed with status {response.status}")
            
            data = await response.json()
            task_id = data.get("task_id")
            
            if not task_id:
                raise Exception("No task ID returned from submission")
            
            return {
                "task_id": task_id,
                "status": data.get("status"),
                "submission_successful": True
            }
    
    async def test_intelligence_decision(self) -> Dict[str, Any]:
        """Test intelligence engine decision making"""
        url = f"{self.base_urls['api']}/api/v1/intelligence/decisions/analyze"
        
        test_context = {
            "scenario": "smoke_test",
            "data": {
                "test_parameter": "test_value"
            },
            "options": [
                {"id": "option_1", "name": "Test Option 1"},
                {"id": "option_2", "name": "Test Option 2"}
            ]
        }
        
        async with self.session.post(url, json=test_context) as response:
            if response.status not in [200, 201]:
                raise Exception(f"Decision analysis failed with status {response.status}")
            
            data = await response.json()
            decision_id = data.get("decision_id")
            
            if not decision_id:
                raise Exception("No decision ID returned from analysis")
            
            return {
                "decision_id": decision_id,
                "recommended_option": data.get("recommended_option"),
                "confidence": data.get("confidence"),
                "analysis_successful": True
            }
    
    async def test_metrics_collection(self) -> Dict[str, Any]:
        """Test metrics collection and availability"""
        url = f"{self.base_urls['api']}/metrics"
        
        async with self.session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Metrics endpoint failed with status {response.status}")
            
            metrics_text = await response.text()
            
            # Check for key metrics
            required_metrics = [
                "http_requests_total",
                "http_request_duration_seconds",
                "agent_registry_active_agents_total",
                "task_executions_total"
            ]
            
            missing_metrics = []
            for metric in required_metrics:
                if metric not in metrics_text:
                    missing_metrics.append(metric)
            
            if missing_metrics:
                raise Exception(f"Missing required metrics: {missing_metrics}")
            
            return {
                "metrics_available": True,
                "metrics_count": len(metrics_text.split('\n')),
                "required_metrics_present": True
            }
    
    async def test_database_connectivity(self) -> Dict[str, Any]:
        """Test database connectivity through API"""
        url = f"{self.base_urls['api']}/api/v1/orchestration/health/database"
        
        async with self.session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Database health check failed with status {response.status}")
            
            data = await response.json()
            
            if not data.get("connected"):
                raise Exception("Database not connected")
            
            return {
                "connected": True,
                "connection_pool_size": data.get("pool_size"),
                "active_connections": data.get("active_connections")
            }
    
    async def test_cache_connectivity(self) -> Dict[str, Any]:
        """Test Redis cache connectivity"""
        url = f"{self.base_urls['api']}/api/v1/orchestration/health/cache"
        
        async with self.session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Cache health check failed with status {response.status}")
            
            data = await response.json()
            
            if not data.get("connected"):
                raise Exception("Cache not connected")
            
            return {
                "connected": True,
                "memory_usage": data.get("memory_usage"),
                "keyspace_hits": data.get("keyspace_hits")
            }
    
    async def test_monitoring_dashboard(self) -> Dict[str, Any]:
        """Test monitoring dashboard accessibility"""
        url = f"{self.base_urls['monitoring']}/api/health"
        
        async with self.session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Monitoring dashboard failed with status {response.status}")
            
            data = await response.json()
            
            return {
                "dashboard_accessible": True,
                "version": data.get("version"),
                "database": data.get("database")
            }
    
    async def run_all_tests(self) -> List[SmokeTestResult]:
        """Run all smoke tests"""
        tests = [
            ("Orchestration Health", self.test_orchestration_health),
            ("Intelligence Health", self.test_intelligence_health),
            ("Agent Registration", self.test_agent_registration),
            ("Task Submission", self.test_task_submission),
            ("Intelligence Decision", self.test_intelligence_decision),
            ("Metrics Collection", self.test_metrics_collection),
            ("Database Connectivity", self.test_database_connectivity),
            ("Cache Connectivity", self.test_cache_connectivity),
            ("Monitoring Dashboard", self.test_monitoring_dashboard)
        ]
        
        logger.info(f"Running {len(tests)} smoke tests for {self.environment} environment...")
        
        for test_name, test_func in tests:
            await self.run_test(test_name, test_func)
        
        return self.test_results
    
    def print_summary(self) -> bool:
        """Print test summary and return overall success status"""
        print(f"\n{'='*70}")
        print(f"Agent Steering System Smoke Test Results - {self.environment.upper()}")
        print(f"{'='*70}")
        
        passed = sum(1 for r in self.test_results if r.result == TestResult.PASS)
        failed = sum(1 for r in self.test_results if r.result == TestResult.FAIL)
        skipped = sum(1 for r in self.test_results if r.result == TestResult.SKIP)
        total = len(self.test_results)
        
        for result in self.test_results:
            status_color = {
                TestResult.PASS: "\033[92m",  # Green
                TestResult.FAIL: "\033[91m",  # Red
                TestResult.SKIP: "\033[93m"   # Yellow
            }.get(result.result, "\033[0m")
            
            reset_color = "\033[0m"
            
            print(f"{result.test_name:<30} {status_color}{result.result.value:<6}{reset_color} "
                  f"({result.duration:.3f}s)")
            
            if result.error:
                print(f"  Error: {result.error}")
        
        print(f"\n{'='*70}")
        
        success_rate = (passed / total * 100) if total > 0 else 0
        overall_success = failed == 0
        
        if overall_success:
            print(f"\033[92m✓ ALL TESTS PASSED\033[0m")
        else:
            print(f"\033[91m✗ SOME TESTS FAILED\033[0m")
        
        print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Total Duration: {sum(r.duration for r in self.test_results):.3f}s")
        print(f"{'='*70}\n")
        
        return overall_success

async def main():
    parser = argparse.ArgumentParser(description="Agent Steering System Smoke Tests")
    parser.add_argument(
        "--environment", "-e",
        choices=["production", "staging", "development"],
        default="production",
        help="Environment to test"
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=30,
        help="Request timeout in seconds"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results in JSON format"
    )
    
    args = parser.parse_args()
    
    async with SmokeTestSuite(args.environment, args.timeout) as suite:
        results = await suite.run_all_tests()
        
        if args.json:
            output = {
                "timestamp": time.time(),
                "environment": args.environment,
                "summary": {
                    "total": len(results),
                    "passed": sum(1 for r in results if r.result == TestResult.PASS),
                    "failed": sum(1 for r in results if r.result == TestResult.FAIL),
                    "skipped": sum(1 for r in results if r.result == TestResult.SKIP)
                },
                "results": [
                    {
                        "test_name": r.test_name,
                        "result": r.result.value,
                        "duration": r.duration,
                        "details": r.details,
                        "error": r.error
                    }
                    for r in results
                ]
            }
            print(json.dumps(output, indent=2))
        else:
            overall_success = suite.print_summary()
        
        # Exit with appropriate code
        sys.exit(0 if overall_success else 1)

if __name__ == "__main__":
    asyncio.run(main())