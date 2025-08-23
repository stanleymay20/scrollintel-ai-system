"""
Security Performance Testing Under Load
Tests security controls performance impact and effectiveness under high load
"""

import asyncio
import logging
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import concurrent.futures
import threading
import psutil
import requests
import json
import random

from .security_test_framework import SecurityTestResult, SecurityTestType, SecuritySeverity

logger = logging.getLogger(__name__)

@dataclass
class LoadProfile:
    name: str
    concurrent_users: int
    requests_per_second: int
    duration: int  # seconds
    ramp_up_time: int  # seconds
    security_features_enabled: bool

@dataclass
class SecurityPerformanceMetrics:
    throughput: float  # requests per second
    response_time_avg: float
    response_time_p95: float
    response_time_p99: float
    error_rate: float
    security_overhead: float  # percentage
    cpu_usage: float
    memory_usage: float
    security_events_processed: int
    false_positive_rate: float

class SecurityPerformanceTester:
    """Security performance testing engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.load_profiles = self._define_load_profiles()
        self.security_scenarios = [
            "baseline_performance",
            "waf_enabled",
            "rate_limiting_active",
            "encryption_overhead",
            "authentication_load",
            "audit_logging_impact",
            "vulnerability_scanning_active",
            "ddos_protection_enabled",
            "full_security_stack"
        ]
    
    async def test_security_performance(self, target_config: Dict[str, Any]) -> List[SecurityTestResult]:
        """Execute comprehensive security performance tests"""
        results = []
        
        logger.info("Starting security performance testing")
        
        for scenario in self.security_scenarios:
            try:
                result = await self._execute_performance_scenario(scenario, target_config)
                results.append(result)
            except Exception as e:
                logger.error(f"Security performance test {scenario} failed: {e}")
                results.append(self._create_error_result(scenario, str(e)))
        
        logger.info(f"Security performance testing completed with {len(results)} scenarios")
        return results
    
    def _define_load_profiles(self) -> List[LoadProfile]:
        """Define load testing profiles"""
        return [
            LoadProfile(
                name="light_load",
                concurrent_users=50,
                requests_per_second=100,
                duration=60,
                ramp_up_time=10,
                security_features_enabled=True
            ),
            LoadProfile(
                name="moderate_load",
                concurrent_users=200,
                requests_per_second=500,
                duration=120,
                ramp_up_time=20,
                security_features_enabled=True
            ),
            LoadProfile(
                name="heavy_load",
                concurrent_users=1000,
                requests_per_second=2000,
                duration=300,
                ramp_up_time=60,
                security_features_enabled=True
            ),
            LoadProfile(
                name="stress_load",
                concurrent_users=5000,
                requests_per_second=10000,
                duration=180,
                ramp_up_time=30,
                security_features_enabled=True
            )
        ]
    
    async def _execute_performance_scenario(self, scenario: str, target_config: Dict[str, Any]) -> SecurityTestResult:
        """Execute specific security performance scenario"""
        start_time = time.time()
        test_id = f"secperf_{scenario}_{int(time.time())}"
        
        logger.info(f"Executing security performance scenario: {scenario}")
        
        # Select appropriate load profile based on scenario
        load_profile = self._select_load_profile(scenario)
        
        # Configure security features for scenario
        security_config = self._configure_security_features(scenario)
        
        # Execute performance test
        baseline_metrics = await self._run_baseline_test(target_config, load_profile)
        security_metrics = await self._run_security_test(target_config, load_profile, security_config)
        
        # Analyze performance impact
        impact_analysis = self._analyze_security_impact(baseline_metrics, security_metrics)
        
        # Generate findings
        findings = self._generate_performance_findings(scenario, impact_analysis, security_metrics)
        
        execution_time = time.time() - start_time
        severity = self._determine_performance_severity(impact_analysis)
        status = "passed" if impact_analysis.get("acceptable_overhead", True) else "failed"
        
        return SecurityTestResult(
            test_id=test_id,
            test_type=SecurityTestType.PERFORMANCE,
            test_name=f"Security Performance: {scenario.replace('_', ' ').title()}",
            status=status,
            severity=severity,
            findings=findings,
            execution_time=execution_time,
            timestamp=datetime.now(),
            metadata={
                "load_profile": load_profile.name,
                "security_overhead": impact_analysis.get("security_overhead", 0),
                "throughput_impact": impact_analysis.get("throughput_impact", 0)
            },
            recommendations=self._generate_performance_recommendations(scenario, impact_analysis)
        )
    
    def _select_load_profile(self, scenario: str) -> LoadProfile:
        """Select appropriate load profile for scenario"""
        if scenario in ["baseline_performance", "waf_enabled"]:
            return self.load_profiles[1]  # moderate_load
        elif scenario in ["full_security_stack", "ddos_protection_enabled"]:
            return self.load_profiles[2]  # heavy_load
        elif scenario == "vulnerability_scanning_active":
            return self.load_profiles[3]  # stress_load
        else:
            return self.load_profiles[0]  # light_load
    
    def _configure_security_features(self, scenario: str) -> Dict[str, Any]:
        """Configure security features for specific scenario"""
        config = {
            "waf_enabled": False,
            "rate_limiting": False,
            "encryption": False,
            "authentication": False,
            "audit_logging": False,
            "vulnerability_scanning": False,
            "ddos_protection": False
        }
        
        if scenario == "baseline_performance":
            # All security features disabled for baseline
            pass
        elif scenario == "waf_enabled":
            config["waf_enabled"] = True
        elif scenario == "rate_limiting_active":
            config["rate_limiting"] = True
        elif scenario == "encryption_overhead":
            config["encryption"] = True
        elif scenario == "authentication_load":
            config["authentication"] = True
        elif scenario == "audit_logging_impact":
            config["audit_logging"] = True
        elif scenario == "vulnerability_scanning_active":
            config["vulnerability_scanning"] = True
        elif scenario == "ddos_protection_enabled":
            config["ddos_protection"] = True
        elif scenario == "full_security_stack":
            # Enable all security features
            for key in config:
                config[key] = True
        
        return config
    
    async def _run_baseline_test(self, target_config: Dict[str, Any], load_profile: LoadProfile) -> SecurityPerformanceMetrics:
        """Run baseline performance test without security features"""
        logger.info("Running baseline performance test")
        
        # Simulate baseline test
        metrics = await self._execute_load_test(
            target_config, 
            load_profile, 
            {"security_enabled": False}
        )
        
        return metrics
    
    async def _run_security_test(self, target_config: Dict[str, Any], 
                                load_profile: LoadProfile, 
                                security_config: Dict[str, Any]) -> SecurityPerformanceMetrics:
        """Run performance test with security features enabled"""
        logger.info("Running security-enabled performance test")
        
        # Simulate security test
        metrics = await self._execute_load_test(
            target_config, 
            load_profile, 
            {**security_config, "security_enabled": True}
        )
        
        return metrics
    
    async def _execute_load_test(self, target_config: Dict[str, Any], 
                                load_profile: LoadProfile, 
                                test_config: Dict[str, Any]) -> SecurityPerformanceMetrics:
        """Execute load test and collect metrics"""
        base_url = target_config.get('base_url', 'http://localhost:8000')
        
        # Initialize metrics collection
        response_times = []
        error_count = 0
        total_requests = 0
        security_events = 0
        false_positives = 0
        
        # System resource monitoring
        cpu_usage_samples = []
        memory_usage_samples = []
        
        # Start system monitoring
        monitoring_task = asyncio.create_task(
            self._monitor_system_resources(cpu_usage_samples, memory_usage_samples, load_profile.duration)
        )
        
        # Execute load test
        start_time = time.time()
        
        # Ramp up phase
        await self._ramp_up_load(base_url, load_profile, test_config, response_times)
        
        # Sustained load phase
        await self._sustained_load(
            base_url, load_profile, test_config, 
            response_times, error_count, total_requests, security_events, false_positives
        )
        
        # Wait for monitoring to complete
        await monitoring_task
        
        # Calculate metrics
        total_time = time.time() - start_time
        
        # Calculate performance metrics
        throughput = total_requests / total_time if total_time > 0 else 0
        response_time_avg = statistics.mean(response_times) if response_times else 0
        response_time_p95 = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else 0
        response_time_p99 = statistics.quantiles(response_times, n=100)[98] if len(response_times) >= 100 else 0
        error_rate = error_count / max(total_requests, 1)
        
        # Calculate security overhead (simulated)
        security_overhead = self._calculate_security_overhead(test_config)
        
        # Calculate resource usage
        avg_cpu = statistics.mean(cpu_usage_samples) if cpu_usage_samples else 0
        avg_memory = statistics.mean(memory_usage_samples) if memory_usage_samples else 0
        
        # Calculate false positive rate
        false_positive_rate = false_positives / max(security_events, 1) if security_events > 0 else 0
        
        return SecurityPerformanceMetrics(
            throughput=throughput,
            response_time_avg=response_time_avg,
            response_time_p95=response_time_p95,
            response_time_p99=response_time_p99,
            error_rate=error_rate,
            security_overhead=security_overhead,
            cpu_usage=avg_cpu,
            memory_usage=avg_memory,
            security_events_processed=security_events,
            false_positive_rate=false_positive_rate
        )
    
    async def _ramp_up_load(self, base_url: str, load_profile: LoadProfile, 
                           test_config: Dict[str, Any], response_times: List[float]):
        """Execute ramp-up phase of load test"""
        ramp_up_time = load_profile.ramp_up_time
        target_rps = load_profile.requests_per_second
        
        # Gradually increase load
        for second in range(ramp_up_time):
            current_rps = int((second + 1) / ramp_up_time * target_rps)
            
            # Send requests for this second
            tasks = []
            for _ in range(current_rps):
                task = asyncio.create_task(self._send_request(base_url, test_config))
                tasks.append(task)
            
            # Wait for requests to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect response times
            for result in results:
                if isinstance(result, float):
                    response_times.append(result)
            
            # Wait for next second
            await asyncio.sleep(1)
    
    async def _sustained_load(self, base_url: str, load_profile: LoadProfile, 
                             test_config: Dict[str, Any], response_times: List[float],
                             error_count: int, total_requests: int, 
                             security_events: int, false_positives: int):
        """Execute sustained load phase"""
        duration = load_profile.duration - load_profile.ramp_up_time
        rps = load_profile.requests_per_second
        
        for second in range(duration):
            # Send requests for this second
            tasks = []
            for _ in range(rps):
                task = asyncio.create_task(self._send_request(base_url, test_config))
                tasks.append(task)
            
            # Wait for requests to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                total_requests += 1
                
                if isinstance(result, Exception):
                    error_count += 1
                elif isinstance(result, float):
                    response_times.append(result)
                    
                    # Simulate security event processing
                    if test_config.get("security_enabled", False):
                        if random.random() < 0.1:  # 10% of requests trigger security events
                            security_events += 1
                            if random.random() < 0.05:  # 5% false positive rate
                                false_positives += 1
            
            # Wait for next second
            await asyncio.sleep(1)
    
    async def _send_request(self, base_url: str, test_config: Dict[str, Any]) -> float:
        """Send individual HTTP request and measure response time"""
        start_time = time.time()
        
        try:
            # Simulate HTTP request with security processing
            base_delay = 0.1  # Base response time
            
            # Add security overhead
            security_delay = 0
            if test_config.get("waf_enabled", False):
                security_delay += 0.02  # WAF processing time
            if test_config.get("rate_limiting", False):
                security_delay += 0.01  # Rate limiting check
            if test_config.get("encryption", False):
                security_delay += 0.05  # Encryption/decryption overhead
            if test_config.get("authentication", False):
                security_delay += 0.03  # Authentication processing
            if test_config.get("audit_logging", False):
                security_delay += 0.01  # Audit logging overhead
            if test_config.get("vulnerability_scanning", False):
                security_delay += 0.1   # Vulnerability scanning overhead
            if test_config.get("ddos_protection", False):
                security_delay += 0.02  # DDoS protection processing
            
            # Simulate request processing
            total_delay = base_delay + security_delay
            await asyncio.sleep(total_delay)
            
            response_time = time.time() - start_time
            return response_time
        
        except Exception as e:
            # Return exception to indicate error
            return e
    
    async def _monitor_system_resources(self, cpu_samples: List[float], 
                                       memory_samples: List[float], duration: int):
        """Monitor system resource usage during load test"""
        monitoring_interval = 1  # second
        
        for _ in range(duration):
            try:
                cpu_usage = psutil.cpu_percent(interval=0.1)
                memory_usage = psutil.virtual_memory().percent
                
                cpu_samples.append(cpu_usage)
                memory_samples.append(memory_usage)
                
                await asyncio.sleep(monitoring_interval)
            
            except Exception as e:
                logger.error(f"Resource monitoring failed: {e}")
    
    def _calculate_security_overhead(self, test_config: Dict[str, Any]) -> float:
        """Calculate security overhead percentage"""
        overhead = 0.0
        
        if test_config.get("waf_enabled", False):
            overhead += 5.0  # 5% overhead for WAF
        if test_config.get("rate_limiting", False):
            overhead += 2.0  # 2% overhead for rate limiting
        if test_config.get("encryption", False):
            overhead += 15.0  # 15% overhead for encryption
        if test_config.get("authentication", False):
            overhead += 8.0   # 8% overhead for authentication
        if test_config.get("audit_logging", False):
            overhead += 3.0   # 3% overhead for audit logging
        if test_config.get("vulnerability_scanning", False):
            overhead += 25.0  # 25% overhead for vulnerability scanning
        if test_config.get("ddos_protection", False):
            overhead += 10.0  # 10% overhead for DDoS protection
        
        return overhead
    
    def _analyze_security_impact(self, baseline_metrics: SecurityPerformanceMetrics, 
                                security_metrics: SecurityPerformanceMetrics) -> Dict[str, Any]:
        """Analyze security impact on performance"""
        analysis = {}
        
        # Throughput impact
        throughput_impact = ((baseline_metrics.throughput - security_metrics.throughput) / 
                           max(baseline_metrics.throughput, 1)) * 100
        analysis["throughput_impact"] = throughput_impact
        
        # Response time impact
        response_time_impact = ((security_metrics.response_time_avg - baseline_metrics.response_time_avg) / 
                              max(baseline_metrics.response_time_avg, 0.001)) * 100
        analysis["response_time_impact"] = response_time_impact
        
        # Error rate comparison
        error_rate_change = security_metrics.error_rate - baseline_metrics.error_rate
        analysis["error_rate_change"] = error_rate_change
        
        # Resource usage impact
        cpu_impact = security_metrics.cpu_usage - baseline_metrics.cpu_usage
        memory_impact = security_metrics.memory_usage - baseline_metrics.memory_usage
        analysis["cpu_impact"] = cpu_impact
        analysis["memory_impact"] = memory_impact
        
        # Security overhead
        analysis["security_overhead"] = security_metrics.security_overhead
        
        # Overall acceptability
        acceptable_overhead = (
            throughput_impact < 30 and  # Less than 30% throughput loss
            response_time_impact < 50 and  # Less than 50% response time increase
            error_rate_change < 0.05 and  # Less than 5% error rate increase
            cpu_impact < 20 and  # Less than 20% CPU increase
            memory_impact < 15  # Less than 15% memory increase
        )
        analysis["acceptable_overhead"] = acceptable_overhead
        
        # Performance grade
        if acceptable_overhead and throughput_impact < 10:
            analysis["performance_grade"] = "A"
        elif acceptable_overhead and throughput_impact < 20:
            analysis["performance_grade"] = "B"
        elif acceptable_overhead:
            analysis["performance_grade"] = "C"
        else:
            analysis["performance_grade"] = "F"
        
        return analysis
    
    def _generate_performance_findings(self, scenario: str, impact_analysis: Dict[str, Any], 
                                     security_metrics: SecurityPerformanceMetrics) -> List[Dict[str, Any]]:
        """Generate performance findings"""
        findings = []
        
        # Main performance finding
        grade = impact_analysis.get("performance_grade", "F")
        throughput_impact = impact_analysis.get("throughput_impact", 0)
        
        findings.append({
            "type": "security_performance_impact",
            "severity": self._get_severity_from_grade(grade).value,
            "title": f"Security Performance Impact: {scenario.replace('_', ' ').title()}",
            "description": f"Performance grade: {grade}, Throughput impact: {throughput_impact:.1f}%",
            "impact": f"Security overhead: {security_metrics.security_overhead:.1f}%",
            "remediation": "Optimize security controls for better performance"
        })
        
        # Specific impact findings
        if impact_analysis.get("throughput_impact", 0) > 30:
            findings.append({
                "type": "high_throughput_impact",
                "severity": "high",
                "title": "High Throughput Impact",
                "description": f"Throughput reduced by {impact_analysis['throughput_impact']:.1f}%",
                "remediation": "Optimize security processing or scale infrastructure"
            })
        
        if impact_analysis.get("response_time_impact", 0) > 50:
            findings.append({
                "type": "high_latency_impact",
                "severity": "medium",
                "title": "High Response Time Impact",
                "description": f"Response time increased by {impact_analysis['response_time_impact']:.1f}%",
                "remediation": "Optimize security control processing time"
            })
        
        if impact_analysis.get("error_rate_change", 0) > 0.05:
            findings.append({
                "type": "increased_error_rate",
                "severity": "high",
                "title": "Increased Error Rate",
                "description": f"Error rate increased by {impact_analysis['error_rate_change']:.2%}",
                "remediation": "Investigate security control configuration issues"
            })
        
        # Resource usage findings
        if impact_analysis.get("cpu_impact", 0) > 20:
            findings.append({
                "type": "high_cpu_impact",
                "severity": "medium",
                "title": "High CPU Usage Impact",
                "description": f"CPU usage increased by {impact_analysis['cpu_impact']:.1f}%",
                "remediation": "Optimize CPU-intensive security operations"
            })
        
        if impact_analysis.get("memory_impact", 0) > 15:
            findings.append({
                "type": "high_memory_impact",
                "severity": "medium",
                "title": "High Memory Usage Impact",
                "description": f"Memory usage increased by {impact_analysis['memory_impact']:.1f}%",
                "remediation": "Optimize memory usage in security controls"
            })
        
        # Security effectiveness findings
        if security_metrics.false_positive_rate > 0.1:
            findings.append({
                "type": "high_false_positive_rate",
                "severity": "medium",
                "title": "High False Positive Rate",
                "description": f"False positive rate: {security_metrics.false_positive_rate:.2%}",
                "remediation": "Tune security controls to reduce false positives"
            })
        
        return findings
    
    def _get_severity_from_grade(self, grade: str) -> SecuritySeverity:
        """Convert performance grade to security severity"""
        grade_mapping = {
            "A": SecuritySeverity.INFO,
            "B": SecuritySeverity.LOW,
            "C": SecuritySeverity.MEDIUM,
            "D": SecuritySeverity.HIGH,
            "F": SecuritySeverity.CRITICAL
        }
        return grade_mapping.get(grade, SecuritySeverity.MEDIUM)
    
    def _determine_performance_severity(self, impact_analysis: Dict[str, Any]) -> SecuritySeverity:
        """Determine overall severity based on performance impact"""
        if not impact_analysis.get("acceptable_overhead", True):
            return SecuritySeverity.HIGH
        
        grade = impact_analysis.get("performance_grade", "F")
        return self._get_severity_from_grade(grade)
    
    def _generate_performance_recommendations(self, scenario: str, impact_analysis: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        grade = impact_analysis.get("performance_grade", "F")
        
        if grade in ["D", "F"]:
            recommendations.append("URGENT: Security performance optimization required")
            recommendations.append("Consider scaling infrastructure or optimizing security controls")
        elif grade == "C":
            recommendations.append("MEDIUM: Security performance needs improvement")
            recommendations.append("Optimize security control configuration")
        else:
            recommendations.append("GOOD: Security performance is acceptable")
            recommendations.append("Monitor performance trends and maintain current configuration")
        
        # Scenario-specific recommendations
        if scenario == "waf_enabled":
            recommendations.extend([
                "Optimize WAF rules for performance",
                "Consider WAF caching strategies",
                "Review WAF rule complexity"
            ])
        elif scenario == "encryption_overhead":
            recommendations.extend([
                "Consider hardware acceleration for encryption",
                "Optimize cipher suite selection",
                "Implement connection pooling"
            ])
        elif scenario == "authentication_load":
            recommendations.extend([
                "Implement authentication caching",
                "Optimize token validation",
                "Consider session management improvements"
            ])
        elif scenario == "vulnerability_scanning_active":
            recommendations.extend([
                "Schedule vulnerability scans during low-traffic periods",
                "Implement scan throttling",
                "Use incremental scanning approaches"
            ])
        elif scenario == "full_security_stack":
            recommendations.extend([
                "Prioritize security controls by risk and performance impact",
                "Implement security control orchestration",
                "Consider security appliance consolidation"
            ])
        
        # General recommendations
        recommendations.extend([
            "Implement performance monitoring for security controls",
            "Regular performance testing with security enabled",
            "Establish performance baselines and SLAs"
        ])
        
        return recommendations
    
    def _create_error_result(self, scenario: str, error: str) -> SecurityTestResult:
        """Create error result for failed performance tests"""
        return SecurityTestResult(
            test_id=f"secperf_{scenario}_error_{int(time.time())}",
            test_type=SecurityTestType.PERFORMANCE,
            test_name=f"Security Performance: {scenario.replace('_', ' ').title()} (Error)",
            status="error",
            severity=SecuritySeverity.INFO,
            findings=[{
                "type": "performance_test_error",
                "severity": "info",
                "description": f"Performance test failed: {error}",
                "remediation": "Fix test configuration and retry"
            }],
            execution_time=0.0,
            timestamp=datetime.now(),
            recommendations=["Fix performance test issues and retry"]
        )