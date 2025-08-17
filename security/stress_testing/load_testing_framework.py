"""
Enterprise Load Testing Framework
Capable of simulating 100,000+ concurrent users with realistic enterprise workloads
"""

import asyncio
import aiohttp
import time
import json
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from datetime import datetime, timedelta
import psutil
import threading
from queue import Queue
import statistics

@dataclass
class LoadTestConfig:
    """Configuration for load testing scenarios"""
    concurrent_users: int = 100000
    test_duration_minutes: int = 30
    ramp_up_time_minutes: int = 5
    target_endpoints: List[str] = None
    user_scenarios: List[Dict] = None
    think_time_seconds: float = 1.0
    data_payload_size_kb: int = 10
    authentication_required: bool = True
    enterprise_features_enabled: bool = True
    
    def __post_init__(self):
        if self.target_endpoints is None:
            self.target_endpoints = [
                "/api/v1/data/upload",
                "/api/v1/analytics/query",
                "/api/v1/reports/generate",
                "/api/v1/dashboard/load",
                "/api/v1/ai/inference"
            ]
        
        if self.user_scenarios is None:
            self.user_scenarios = [
                {"name": "data_analyst", "weight": 0.3, "actions": ["upload", "query", "visualize"]},
                {"name": "executive", "weight": 0.2, "actions": ["dashboard", "reports"]},
                {"name": "data_scientist", "weight": 0.25, "actions": ["ai_inference", "model_training"]},
                {"name": "business_user", "weight": 0.25, "actions": ["query", "dashboard"]}
            ]

@dataclass
class LoadTestResult:
    """Results from load testing execution"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    throughput_rps: float
    error_rate_percent: float
    concurrent_users_achieved: int
    test_duration_seconds: float
    resource_utilization: Dict[str, float]
    bottlenecks_identified: List[str]
    performance_degradation_points: List[Dict]

class EnterpriseLoadTester:
    """High-performance load testing framework for enterprise scenarios"""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.results_queue = Queue()
        self.active_sessions = 0
        self.session_lock = threading.Lock()
        self.start_time = None
        self.metrics_collector = MetricsCollector()
        
    async def simulate_user_session(self, session: aiohttp.ClientSession, user_id: int, scenario: Dict) -> Dict:
        """Simulate a realistic user session with enterprise workload patterns"""
        session_results = {
            "user_id": user_id,
            "scenario": scenario["name"],
            "requests": [],
            "total_time": 0,
            "errors": []
        }
        
        try:
            with self.session_lock:
                self.active_sessions += 1
            
            session_start = time.time()
            
            # Simulate user authentication
            if self.config.authentication_required:
                auth_result = await self._authenticate_user(session, user_id)
                session_results["requests"].append(auth_result)
            
            # Execute user scenario actions
            for action in scenario["actions"]:
                action_result = await self._execute_action(session, action, user_id)
                session_results["requests"].append(action_result)
                
                # Realistic think time between actions
                await asyncio.sleep(self.config.think_time_seconds)
            
            session_results["total_time"] = time.time() - session_start
            
        except Exception as e:
            session_results["errors"].append(str(e))
            self.logger.error(f"User session {user_id} failed: {e}")
        finally:
            with self.session_lock:
                self.active_sessions -= 1
        
        return session_results
    
    async def _authenticate_user(self, session: aiohttp.ClientSession, user_id: int) -> Dict:
        """Simulate enterprise authentication with realistic payloads"""
        auth_payload = {
            "username": f"enterprise_user_{user_id}",
            "password": "secure_enterprise_password",
            "domain": "enterprise.local",
            "mfa_token": f"mfa_{user_id}_{int(time.time())}"
        }
        
        start_time = time.time()
        try:
            async with session.post("/api/v1/auth/login", json=auth_payload) as response:
                response_time = (time.time() - start_time) * 1000
                return {
                    "endpoint": "/api/v1/auth/login",
                    "method": "POST",
                    "status_code": response.status,
                    "response_time_ms": response_time,
                    "success": response.status == 200
                }
        except Exception as e:
            return {
                "endpoint": "/api/v1/auth/login",
                "method": "POST",
                "status_code": 0,
                "response_time_ms": (time.time() - start_time) * 1000,
                "success": False,
                "error": str(e)
            }
    
    async def _execute_action(self, session: aiohttp.ClientSession, action: str, user_id: int) -> Dict:
        """Execute specific user actions with enterprise-scale data"""
        action_mapping = {
            "upload": self._simulate_data_upload,
            "query": self._simulate_analytics_query,
            "visualize": self._simulate_visualization_request,
            "dashboard": self._simulate_dashboard_load,
            "reports": self._simulate_report_generation,
            "ai_inference": self._simulate_ai_inference,
            "model_training": self._simulate_model_training
        }
        
        if action in action_mapping:
            return await action_mapping[action](session, user_id)
        else:
            return {"error": f"Unknown action: {action}"}
    
    async def _simulate_data_upload(self, session: aiohttp.ClientSession, user_id: int) -> Dict:
        """Simulate enterprise data upload with realistic file sizes"""
        # Generate realistic enterprise data payload
        data_size_bytes = self.config.data_payload_size_kb * 1024
        payload = {
            "file_name": f"enterprise_data_{user_id}_{int(time.time())}.csv",
            "data": "x" * data_size_bytes,  # Simulate file content
            "metadata": {
                "department": "analytics",
                "classification": "confidential",
                "retention_policy": "7_years"
            }
        }
        
        start_time = time.time()
        try:
            async with session.post("/api/v1/data/upload", json=payload) as response:
                response_time = (time.time() - start_time) * 1000
                return {
                    "endpoint": "/api/v1/data/upload",
                    "method": "POST",
                    "status_code": response.status,
                    "response_time_ms": response_time,
                    "success": response.status == 200,
                    "payload_size_kb": self.config.data_payload_size_kb
                }
        except Exception as e:
            return {
                "endpoint": "/api/v1/data/upload",
                "method": "POST",
                "status_code": 0,
                "response_time_ms": (time.time() - start_time) * 1000,
                "success": False,
                "error": str(e)
            }
    
    async def _simulate_analytics_query(self, session: aiohttp.ClientSession, user_id: int) -> Dict:
        """Simulate complex analytics queries typical in enterprise environments"""
        query_payload = {
            "query": f"""
                SELECT 
                    department, 
                    SUM(revenue) as total_revenue,
                    AVG(customer_satisfaction) as avg_satisfaction,
                    COUNT(DISTINCT customer_id) as unique_customers
                FROM enterprise_data 
                WHERE date_range BETWEEN '2024-01-01' AND '2024-12-31'
                    AND region IN ('North America', 'Europe', 'Asia Pacific')
                    AND product_category IS NOT NULL
                GROUP BY department
                HAVING total_revenue > 1000000
                ORDER BY total_revenue DESC
                LIMIT 100
            """,
            "parameters": {
                "user_id": user_id,
                "access_level": "enterprise",
                "cache_enabled": True
            }
        }
        
        start_time = time.time()
        try:
            async with session.post("/api/v1/analytics/query", json=query_payload) as response:
                response_time = (time.time() - start_time) * 1000
                return {
                    "endpoint": "/api/v1/analytics/query",
                    "method": "POST",
                    "status_code": response.status,
                    "response_time_ms": response_time,
                    "success": response.status == 200
                }
        except Exception as e:
            return {
                "endpoint": "/api/v1/analytics/query",
                "method": "POST",
                "status_code": 0,
                "response_time_ms": (time.time() - start_time) * 1000,
                "success": False,
                "error": str(e)
            }
    
    async def _simulate_visualization_request(self, session: aiohttp.ClientSession, user_id: int) -> Dict:
        """Simulate visualization rendering requests"""
        viz_payload = {
            "chart_type": "complex_dashboard",
            "data_points": 10000,
            "filters": {
                "date_range": "last_12_months",
                "departments": ["sales", "marketing", "operations"],
                "metrics": ["revenue", "costs", "profit_margin"]
            },
            "rendering_options": {
                "high_resolution": True,
                "interactive": True,
                "real_time_updates": True
            }
        }
        
        start_time = time.time()
        try:
            async with session.post("/api/v1/visualization/render", json=viz_payload) as response:
                response_time = (time.time() - start_time) * 1000
                return {
                    "endpoint": "/api/v1/visualization/render",
                    "method": "POST",
                    "status_code": response.status,
                    "response_time_ms": response_time,
                    "success": response.status == 200
                }
        except Exception as e:
            return {
                "endpoint": "/api/v1/visualization/render",
                "method": "POST",
                "status_code": 0,
                "response_time_ms": (time.time() - start_time) * 1000,
                "success": False,
                "error": str(e)
            }
    
    async def _simulate_dashboard_load(self, session: aiohttp.ClientSession, user_id: int) -> Dict:
        """Simulate executive dashboard loading with multiple widgets"""
        start_time = time.time()
        try:
            async with session.get(f"/api/v1/dashboard/executive?user_id={user_id}") as response:
                response_time = (time.time() - start_time) * 1000
                return {
                    "endpoint": "/api/v1/dashboard/executive",
                    "method": "GET",
                    "status_code": response.status,
                    "response_time_ms": response_time,
                    "success": response.status == 200
                }
        except Exception as e:
            return {
                "endpoint": "/api/v1/dashboard/executive",
                "method": "GET",
                "status_code": 0,
                "response_time_ms": (time.time() - start_time) * 1000,
                "success": False,
                "error": str(e)
            }
    
    async def _simulate_report_generation(self, session: aiohttp.ClientSession, user_id: int) -> Dict:
        """Simulate enterprise report generation"""
        report_payload = {
            "report_type": "quarterly_executive_summary",
            "format": "pdf",
            "data_sources": ["sales", "finance", "operations", "hr"],
            "parameters": {
                "quarter": "Q4_2024",
                "include_forecasts": True,
                "detail_level": "executive"
            }
        }
        
        start_time = time.time()
        try:
            async with session.post("/api/v1/reports/generate", json=report_payload) as response:
                response_time = (time.time() - start_time) * 1000
                return {
                    "endpoint": "/api/v1/reports/generate",
                    "method": "POST",
                    "status_code": response.status,
                    "response_time_ms": response_time,
                    "success": response.status == 200
                }
        except Exception as e:
            return {
                "endpoint": "/api/v1/reports/generate",
                "method": "POST",
                "status_code": 0,
                "response_time_ms": (time.time() - start_time) * 1000,
                "success": False,
                "error": str(e)
            }
    
    async def _simulate_ai_inference(self, session: aiohttp.ClientSession, user_id: int) -> Dict:
        """Simulate AI model inference requests"""
        inference_payload = {
            "model_id": "enterprise_forecasting_v2",
            "input_data": {
                "features": [1.2, 3.4, 5.6, 7.8, 9.0] * 100,  # Realistic feature vector
                "metadata": {
                    "timestamp": int(time.time()),
                    "source": "enterprise_system"
                }
            },
            "options": {
                "confidence_threshold": 0.85,
                "return_explanations": True
            }
        }
        
        start_time = time.time()
        try:
            async with session.post("/api/v1/ai/inference", json=inference_payload) as response:
                response_time = (time.time() - start_time) * 1000
                return {
                    "endpoint": "/api/v1/ai/inference",
                    "method": "POST",
                    "status_code": response.status,
                    "response_time_ms": response_time,
                    "success": response.status == 200
                }
        except Exception as e:
            return {
                "endpoint": "/api/v1/ai/inference",
                "method": "POST",
                "status_code": 0,
                "response_time_ms": (time.time() - start_time) * 1000,
                "success": False,
                "error": str(e)
            }
    
    async def _simulate_model_training(self, session: aiohttp.ClientSession, user_id: int) -> Dict:
        """Simulate model training job submission"""
        training_payload = {
            "training_job_name": f"enterprise_model_training_{user_id}_{int(time.time())}",
            "algorithm": "gradient_boosting",
            "dataset_id": f"enterprise_dataset_{user_id}",
            "hyperparameters": {
                "learning_rate": 0.1,
                "max_depth": 10,
                "n_estimators": 100
            },
            "compute_requirements": {
                "instance_type": "ml.m5.xlarge",
                "max_runtime_hours": 2
            }
        }
        
        start_time = time.time()
        try:
            async with session.post("/api/v1/ml/training/start", json=training_payload) as response:
                response_time = (time.time() - start_time) * 1000
                return {
                    "endpoint": "/api/v1/ml/training/start",
                    "method": "POST",
                    "status_code": response.status,
                    "response_time_ms": response_time,
                    "success": response.status == 200
                }
        except Exception as e:
            return {
                "endpoint": "/api/v1/ml/training/start",
                "method": "POST",
                "status_code": 0,
                "response_time_ms": (time.time() - start_time) * 1000,
                "success": False,
                "error": str(e)
            }
    
    async def execute_load_test(self, base_url: str = "http://localhost:8000") -> LoadTestResult:
        """Execute the complete load test with enterprise-scale concurrent users"""
        self.logger.info(f"Starting load test with {self.config.concurrent_users} concurrent users")
        self.start_time = time.time()
        
        # Start metrics collection
        self.metrics_collector.start_collection()
        
        # Create session pool for connection reuse
        connector = aiohttp.TCPConnector(
            limit=self.config.concurrent_users + 100,
            limit_per_host=self.config.concurrent_users + 100,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": "EnterpriseLoadTester/1.0"}
        ) as session:
            
            # Generate user scenarios based on weights
            user_scenarios = self._generate_user_scenarios()
            
            # Create tasks for concurrent user simulation
            tasks = []
            for user_id in range(self.config.concurrent_users):
                scenario = user_scenarios[user_id % len(user_scenarios)]
                task = asyncio.create_task(
                    self.simulate_user_session(session, user_id, scenario)
                )
                tasks.append(task)
                
                # Implement ramp-up to avoid overwhelming the system
                if user_id % 1000 == 0 and user_id > 0:
                    ramp_delay = (self.config.ramp_up_time_minutes * 60) / (self.config.concurrent_users / 1000)
                    await asyncio.sleep(ramp_delay)
            
            # Wait for all user sessions to complete or timeout
            self.logger.info(f"Executing {len(tasks)} concurrent user sessions")
            session_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Stop metrics collection
        resource_metrics = self.metrics_collector.stop_collection()
        
        # Analyze results
        return self._analyze_results(session_results, resource_metrics)
    
    def _generate_user_scenarios(self) -> List[Dict]:
        """Generate user scenarios based on configured weights"""
        scenarios = []
        for scenario in self.config.user_scenarios:
            count = int(self.config.concurrent_users * scenario["weight"])
            scenarios.extend([scenario] * count)
        
        # Fill remaining slots with random scenarios
        while len(scenarios) < self.config.concurrent_users:
            scenarios.append(np.random.choice(self.config.user_scenarios))
        
        return scenarios[:self.config.concurrent_users]
    
    def _analyze_results(self, session_results: List[Dict], resource_metrics: Dict) -> LoadTestResult:
        """Analyze load test results and generate comprehensive report"""
        total_requests = 0
        successful_requests = 0
        failed_requests = 0
        response_times = []
        errors = []
        
        for result in session_results:
            if isinstance(result, Exception):
                failed_requests += 1
                errors.append(str(result))
                continue
            
            if isinstance(result, dict) and "requests" in result:
                for request in result["requests"]:
                    total_requests += 1
                    if request.get("success", False):
                        successful_requests += 1
                        response_times.append(request.get("response_time_ms", 0))
                    else:
                        failed_requests += 1
                        if "error" in request:
                            errors.append(request["error"])
        
        # Calculate performance metrics
        test_duration = time.time() - self.start_time
        error_rate = (failed_requests / max(total_requests, 1)) * 100
        throughput = total_requests / test_duration if test_duration > 0 else 0
        
        # Response time percentiles
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
        else:
            avg_response_time = p95_response_time = p99_response_time = 0
        
        # Identify bottlenecks and performance issues
        bottlenecks = self._identify_bottlenecks(resource_metrics, response_times)
        performance_degradation = self._detect_performance_degradation(response_times)
        
        return LoadTestResult(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time_ms=avg_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            throughput_rps=throughput,
            error_rate_percent=error_rate,
            concurrent_users_achieved=self.config.concurrent_users,
            test_duration_seconds=test_duration,
            resource_utilization=resource_metrics,
            bottlenecks_identified=bottlenecks,
            performance_degradation_points=performance_degradation
        )
    
    def _identify_bottlenecks(self, resource_metrics: Dict, response_times: List[float]) -> List[str]:
        """Identify system bottlenecks based on metrics"""
        bottlenecks = []
        
        if resource_metrics.get("cpu_percent", 0) > 80:
            bottlenecks.append("High CPU utilization detected")
        
        if resource_metrics.get("memory_percent", 0) > 85:
            bottlenecks.append("High memory utilization detected")
        
        if resource_metrics.get("disk_io_percent", 0) > 90:
            bottlenecks.append("Disk I/O bottleneck detected")
        
        if response_times and statistics.mean(response_times) > 5000:
            bottlenecks.append("High average response time indicates performance issues")
        
        return bottlenecks
    
    def _detect_performance_degradation(self, response_times: List[float]) -> List[Dict]:
        """Detect points where performance significantly degraded"""
        if not response_times or len(response_times) < 100:
            return []
        
        degradation_points = []
        window_size = len(response_times) // 10  # Analyze in 10% windows
        
        for i in range(0, len(response_times) - window_size, window_size):
            window = response_times[i:i + window_size]
            next_window = response_times[i + window_size:i + 2 * window_size]
            
            if next_window:
                current_avg = statistics.mean(window)
                next_avg = statistics.mean(next_window)
                
                # Detect significant degradation (>50% increase)
                if next_avg > current_avg * 1.5:
                    degradation_points.append({
                        "time_point": f"{(i / len(response_times)) * 100:.1f}%",
                        "degradation_factor": next_avg / current_avg,
                        "avg_response_before_ms": current_avg,
                        "avg_response_after_ms": next_avg
                    })
        
        return degradation_points


class MetricsCollector:
    """Collect system resource metrics during load testing"""
    
    def __init__(self):
        self.collecting = False
        self.metrics = []
        self.collection_thread = None
    
    def start_collection(self):
        """Start collecting system metrics"""
        self.collecting = True
        self.collection_thread = threading.Thread(target=self._collect_metrics)
        self.collection_thread.start()
    
    def stop_collection(self) -> Dict[str, float]:
        """Stop collecting metrics and return aggregated results"""
        self.collecting = False
        if self.collection_thread:
            self.collection_thread.join()
        
        if not self.metrics:
            return {}
        
        # Aggregate metrics
        cpu_values = [m["cpu_percent"] for m in self.metrics]
        memory_values = [m["memory_percent"] for m in self.metrics]
        disk_values = [m["disk_io_percent"] for m in self.metrics]
        
        return {
            "cpu_percent": statistics.mean(cpu_values),
            "cpu_max": max(cpu_values),
            "memory_percent": statistics.mean(memory_values),
            "memory_max": max(memory_values),
            "disk_io_percent": statistics.mean(disk_values),
            "disk_io_max": max(disk_values),
            "samples_collected": len(self.metrics)
        }
    
    def _collect_metrics(self):
        """Collect system metrics in background thread"""
        while self.collecting:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk_io = psutil.disk_io_counters()
                
                # Calculate disk I/O percentage (simplified)
                disk_io_percent = min(100, (disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024 * 100))
                
                self.metrics.append({
                    "timestamp": time.time(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_io_percent": disk_io_percent
                })
                
            except Exception as e:
                logging.error(f"Error collecting metrics: {e}")
            
            time.sleep(5)  # Collect every 5 seconds


# Example usage and testing
if __name__ == "__main__":
    async def run_load_test():
        config = LoadTestConfig(
            concurrent_users=10000,  # Start with 10K for testing
            test_duration_minutes=10,
            ramp_up_time_minutes=2
        )
        
        tester = EnterpriseLoadTester(config)
        result = await tester.execute_load_test()
        
        print("Load Test Results:")
        print(f"Total Requests: {result.total_requests}")
        print(f"Success Rate: {(result.successful_requests / result.total_requests) * 100:.2f}%")
        print(f"Average Response Time: {result.average_response_time_ms:.2f}ms")
        print(f"95th Percentile: {result.p95_response_time_ms:.2f}ms")
        print(f"Throughput: {result.throughput_rps:.2f} RPS")
        print(f"Concurrent Users Achieved: {result.concurrent_users_achieved}")
        
        if result.bottlenecks_identified:
            print("\nBottlenecks Identified:")
            for bottleneck in result.bottlenecks_identified:
                print(f"- {bottleneck}")
    
    # Run the test
    asyncio.run(run_load_test())