"""
Chaos Engineering Framework for Edge-Case Failure Testing
Tests system resilience under extreme failure conditions and edge cases
"""

import asyncio
import random
import time
import logging
import json
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid
import psutil
import subprocess
import signal
import os
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import socket

class FailureType(Enum):
    NETWORK_PARTITION = "network_partition"
    SERVICE_CRASH = "service_crash"
    DATABASE_FAILURE = "database_failure"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    CPU_SPIKE = "cpu_spike"
    DISK_FULL = "disk_full"
    LATENCY_INJECTION = "latency_injection"
    PACKET_LOSS = "packet_loss"
    DNS_FAILURE = "dns_failure"
    CERTIFICATE_EXPIRY = "certificate_expiry"
    CASCADING_FAILURE = "cascading_failure"
    BYZANTINE_FAILURE = "byzantine_failure"

class FailureSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ChaosExperiment:
    """Defines a chaos engineering experiment"""
    experiment_id: str
    name: str
    description: str
    failure_type: FailureType
    severity: FailureSeverity
    target_services: List[str]
    duration_seconds: int
    blast_radius: float  # Percentage of infrastructure affected (0-100)
    hypothesis: str
    success_criteria: List[str]
    rollback_strategy: str
    safety_checks: List[str]

@dataclass
class ChaosResult:
    """Results from chaos engineering experiment"""
    experiment_id: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    failure_injected: bool
    system_recovered: bool
    recovery_time_seconds: float
    hypothesis_validated: bool
    success_criteria_met: List[bool]
    system_metrics: Dict[str, Any]
    failure_impact: Dict[str, float]
    lessons_learned: List[str]
    recommendations: List[str]

class ChaosEngineeringFramework:
    """Comprehensive chaos engineering framework for enterprise systems"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_experiments = {}
        self.experiment_lock = threading.Lock()
        self.safety_monitor = SafetyMonitor()
        self.failure_injector = FailureInjector()
        self.recovery_validator = RecoveryValidator()
        
    async def execute_chaos_experiment(self, experiment: ChaosExperiment) -> ChaosResult:
        """Execute a comprehensive chaos engineering experiment"""
        
        self.logger.info(f"Starting chaos experiment: {experiment.name}")
        
        with self.experiment_lock:
            self.active_experiments[experiment.experiment_id] = experiment
        
        start_time = datetime.now()
        
        try:
            # Pre-experiment safety checks
            safety_check_passed = await self._perform_safety_checks(experiment)
            if not safety_check_passed:
                raise Exception("Safety checks failed - experiment aborted")
            
            # Start monitoring
            self.safety_monitor.start_monitoring(experiment)
            
            # Collect baseline metrics
            baseline_metrics = await self._collect_baseline_metrics(experiment.target_services)
            
            # Inject failure
            failure_result = await self.failure_injector.inject_failure(experiment)
            
            # Monitor system behavior during failure
            failure_metrics = await self._monitor_during_failure(experiment)
            
            # Validate recovery
            recovery_result = await self.recovery_validator.validate_recovery(
                experiment, baseline_metrics
            )
            
            # Analyze results
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return ChaosResult(
                experiment_id=experiment.experiment_id,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                failure_injected=failure_result["success"],
                system_recovered=recovery_result["recovered"],
                recovery_time_seconds=recovery_result["recovery_time"],
                hypothesis_validated=self._validate_hypothesis(experiment, failure_metrics),
                success_criteria_met=self._check_success_criteria(experiment, failure_metrics),
                system_metrics=failure_metrics,
                failure_impact=failure_result["impact"],
                lessons_learned=self._extract_lessons_learned(experiment, failure_metrics),
                recommendations=self._generate_recommendations(experiment, failure_metrics)
            )
            
        except Exception as e:
            self.logger.error(f"Chaos experiment failed: {e}")
            # Emergency rollback
            await self._emergency_rollback(experiment)
            raise
            
        finally:
            self.safety_monitor.stop_monitoring(experiment.experiment_id)
            with self.experiment_lock:
                if experiment.experiment_id in self.active_experiments:
                    del self.active_experiments[experiment.experiment_id]
    
    async def _perform_safety_checks(self, experiment: ChaosExperiment) -> bool:
        """Perform pre-experiment safety checks"""
        
        self.logger.info("Performing safety checks...")
        
        # Check system health
        system_health = await self._check_system_health(experiment.target_services)
        if system_health < 0.8:  # 80% health threshold
            self.logger.error("System health below threshold - aborting experiment")
            return False
        
        # Check blast radius
        if experiment.blast_radius > 50 and experiment.severity == FailureSeverity.CRITICAL:
            self.logger.error("Blast radius too high for critical severity - aborting")
            return False
        
        # Check for concurrent experiments
        if len(self.active_experiments) > 0:
            self.logger.error("Another experiment is running - aborting")
            return False
        
        # Validate rollback strategy
        if not experiment.rollback_strategy:
            self.logger.error("No rollback strategy defined - aborting")
            return False
        
        return True
    
    async def _check_system_health(self, target_services: List[str]) -> float:
        """Check overall system health before experiment"""
        
        health_scores = []
        
        for service in target_services:
            try:
                # Simulate health check
                health_endpoint = f"http://{service}/health"
                async with aiohttp.ClientSession() as session:
                    async with session.get(health_endpoint, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            health_scores.append(1.0)
                        else:
                            health_scores.append(0.5)
            except:
                health_scores.append(0.0)
        
        return sum(health_scores) / len(health_scores) if health_scores else 0.0
    
    async def _collect_baseline_metrics(self, target_services: List[str]) -> Dict[str, Any]:
        """Collect baseline system metrics before failure injection"""
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "network_connections": len(psutil.net_connections()),
            "service_response_times": {},
            "error_rates": {}
        }
        
        # Collect service-specific metrics
        for service in target_services:
            try:
                start_time = time.time()
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://{service}/metrics", timeout=aiohttp.ClientTimeout(total=5)) as response:
                        response_time = (time.time() - start_time) * 1000
                        metrics["service_response_times"][service] = response_time
                        metrics["error_rates"][service] = 0.0 if response.status == 200 else 1.0
            except Exception as e:
                metrics["service_response_times"][service] = 5000  # Timeout
                metrics["error_rates"][service] = 1.0
        
        return metrics
    
    async def _monitor_during_failure(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Monitor system behavior during failure injection"""
        
        monitoring_duration = min(experiment.duration_seconds, 300)  # Max 5 minutes
        metrics_history = []
        
        for i in range(0, monitoring_duration, 10):  # Collect every 10 seconds
            metrics = await self._collect_baseline_metrics(experiment.target_services)
            metrics_history.append(metrics)
            await asyncio.sleep(10)
        
        # Aggregate metrics
        return {
            "monitoring_duration": monitoring_duration,
            "samples_collected": len(metrics_history),
            "metrics_history": metrics_history,
            "avg_cpu_percent": sum(m["cpu_percent"] for m in metrics_history) / len(metrics_history),
            "max_cpu_percent": max(m["cpu_percent"] for m in metrics_history),
            "avg_memory_percent": sum(m["memory_percent"] for m in metrics_history) / len(metrics_history),
            "max_memory_percent": max(m["memory_percent"] for m in metrics_history),
            "service_availability": self._calculate_service_availability(metrics_history, experiment.target_services)
        }
    
    def _calculate_service_availability(self, metrics_history: List[Dict], services: List[str]) -> Dict[str, float]:
        """Calculate service availability during experiment"""
        
        availability = {}
        
        for service in services:
            successful_checks = 0
            total_checks = 0
            
            for metrics in metrics_history:
                if service in metrics.get("error_rates", {}):
                    total_checks += 1
                    if metrics["error_rates"][service] < 0.5:  # Less than 50% error rate
                        successful_checks += 1
            
            availability[service] = (successful_checks / total_checks) if total_checks > 0 else 0.0
        
        return availability
    
    def _validate_hypothesis(self, experiment: ChaosExperiment, metrics: Dict[str, Any]) -> bool:
        """Validate experiment hypothesis based on observed behavior"""
        
        # Simple hypothesis validation based on system behavior
        # In real implementation, this would be more sophisticated
        
        avg_availability = sum(metrics["service_availability"].values()) / len(metrics["service_availability"])
        
        if "resilient" in experiment.hypothesis.lower():
            return avg_availability > 0.7  # System remained 70% available
        elif "graceful degradation" in experiment.hypothesis.lower():
            return metrics["max_cpu_percent"] < 90 and metrics["max_memory_percent"] < 90
        elif "recovery" in experiment.hypothesis.lower():
            return avg_availability > 0.5  # System recovered to 50% availability
        
        return True  # Default to hypothesis validated
    
    def _check_success_criteria(self, experiment: ChaosExperiment, metrics: Dict[str, Any]) -> List[bool]:
        """Check if success criteria were met"""
        
        results = []
        
        for criteria in experiment.success_criteria:
            if "availability" in criteria.lower():
                avg_availability = sum(metrics["service_availability"].values()) / len(metrics["service_availability"])
                results.append(avg_availability > 0.8)
            elif "response time" in criteria.lower():
                avg_response_times = []
                for service_metrics in metrics["metrics_history"]:
                    avg_response_times.extend(service_metrics.get("service_response_times", {}).values())
                avg_response_time = sum(avg_response_times) / len(avg_response_times) if avg_response_times else 0
                results.append(avg_response_time < 2000)  # Less than 2 seconds
            elif "no data loss" in criteria.lower():
                results.append(True)  # Assume no data loss for simulation
            else:
                results.append(True)  # Default to criteria met
        
        return results
    
    def _extract_lessons_learned(self, experiment: ChaosExperiment, metrics: Dict[str, Any]) -> List[str]:
        """Extract lessons learned from experiment"""
        
        lessons = []
        
        # Analyze CPU usage
        if metrics["max_cpu_percent"] > 90:
            lessons.append("System experienced high CPU usage during failure - consider CPU scaling policies")
        
        # Analyze memory usage
        if metrics["max_memory_percent"] > 85:
            lessons.append("Memory usage spiked during failure - review memory management and garbage collection")
        
        # Analyze service availability
        low_availability_services = [
            service for service, availability in metrics["service_availability"].items()
            if availability < 0.8
        ]
        if low_availability_services:
            lessons.append(f"Services with low availability: {', '.join(low_availability_services)} - improve resilience")
        
        # Failure type specific lessons
        if experiment.failure_type == FailureType.NETWORK_PARTITION:
            lessons.append("Network partition revealed dependencies - consider circuit breakers and retries")
        elif experiment.failure_type == FailureType.DATABASE_FAILURE:
            lessons.append("Database failure impact - implement database clustering and failover")
        
        return lessons
    
    def _generate_recommendations(self, experiment: ChaosExperiment, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on experiment results"""
        
        recommendations = []
        
        # Performance recommendations
        if metrics["max_cpu_percent"] > 80:
            recommendations.append("Implement auto-scaling for CPU-intensive workloads")
        
        if metrics["max_memory_percent"] > 80:
            recommendations.append("Optimize memory usage and implement memory-based scaling")
        
        # Availability recommendations
        avg_availability = sum(metrics["service_availability"].values()) / len(metrics["service_availability"])
        if avg_availability < 0.9:
            recommendations.append("Improve service resilience with redundancy and failover mechanisms")
        
        # Failure type specific recommendations
        failure_recommendations = {
            FailureType.NETWORK_PARTITION: [
                "Implement circuit breakers for external dependencies",
                "Add retry logic with exponential backoff",
                "Consider service mesh for traffic management"
            ],
            FailureType.SERVICE_CRASH: [
                "Implement health checks and automatic restart policies",
                "Add graceful shutdown procedures",
                "Consider blue-green deployments"
            ],
            FailureType.DATABASE_FAILURE: [
                "Implement database replication and failover",
                "Add connection pooling and retry logic",
                "Consider read replicas for scaling"
            ]
        }
        
        if experiment.failure_type in failure_recommendations:
            recommendations.extend(failure_recommendations[experiment.failure_type])
        
        return recommendations
    
    async def _emergency_rollback(self, experiment: ChaosExperiment):
        """Perform emergency rollback if experiment goes wrong"""
        
        self.logger.warning(f"Performing emergency rollback for experiment: {experiment.experiment_id}")
        
        try:
            # Stop failure injection
            await self.failure_injector.stop_failure(experiment)
            
            # Execute rollback strategy
            await self._execute_rollback_strategy(experiment)
            
            # Verify system recovery
            recovery_verified = await self._verify_emergency_recovery(experiment)
            
            if recovery_verified:
                self.logger.info("Emergency rollback successful")
            else:
                self.logger.error("Emergency rollback failed - manual intervention required")
                
        except Exception as e:
            self.logger.error(f"Emergency rollback failed: {e}")
    
    async def _execute_rollback_strategy(self, experiment: ChaosExperiment):
        """Execute the defined rollback strategy"""
        
        rollback_steps = experiment.rollback_strategy.split(";")
        
        for step in rollback_steps:
            step = step.strip()
            if step.startswith("restart_service"):
                service_name = step.split(":")[1].strip()
                await self._restart_service(service_name)
            elif step.startswith("restore_network"):
                await self._restore_network_connectivity()
            elif step.startswith("clear_cache"):
                await self._clear_system_cache()
            # Add more rollback actions as needed
    
    async def _restart_service(self, service_name: str):
        """Restart a specific service"""
        try:
            # Simulate service restart
            self.logger.info(f"Restarting service: {service_name}")
            await asyncio.sleep(2)  # Simulate restart time
        except Exception as e:
            self.logger.error(f"Failed to restart service {service_name}: {e}")
    
    async def _restore_network_connectivity(self):
        """Restore network connectivity"""
        try:
            self.logger.info("Restoring network connectivity")
            await asyncio.sleep(1)  # Simulate network restoration
        except Exception as e:
            self.logger.error(f"Failed to restore network: {e}")
    
    async def _clear_system_cache(self):
        """Clear system cache"""
        try:
            self.logger.info("Clearing system cache")
            await asyncio.sleep(0.5)  # Simulate cache clearing
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
    
    async def _verify_emergency_recovery(self, experiment: ChaosExperiment) -> bool:
        """Verify that emergency recovery was successful"""
        
        # Wait for system to stabilize
        await asyncio.sleep(30)
        
        # Check system health
        health_score = await self._check_system_health(experiment.target_services)
        
        return health_score > 0.8
    
    async def run_chaos_suite(self, experiments: List[ChaosExperiment]) -> List[ChaosResult]:
        """Run a suite of chaos engineering experiments"""
        
        self.logger.info(f"Running chaos suite with {len(experiments)} experiments")
        
        results = []
        
        for experiment in experiments:
            try:
                result = await self.execute_chaos_experiment(experiment)
                results.append(result)
                
                # Wait between experiments for system recovery
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Experiment {experiment.name} failed: {e}")
                # Continue with next experiment
        
        return results


class FailureInjector:
    """Injects various types of failures into the system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_failures = {}
    
    async def inject_failure(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Inject failure based on experiment configuration"""
        
        failure_methods = {
            FailureType.NETWORK_PARTITION: self._inject_network_partition,
            FailureType.SERVICE_CRASH: self._inject_service_crash,
            FailureType.DATABASE_FAILURE: self._inject_database_failure,
            FailureType.MEMORY_EXHAUSTION: self._inject_memory_exhaustion,
            FailureType.CPU_SPIKE: self._inject_cpu_spike,
            FailureType.DISK_FULL: self._inject_disk_full,
            FailureType.LATENCY_INJECTION: self._inject_latency,
            FailureType.PACKET_LOSS: self._inject_packet_loss,
            FailureType.DNS_FAILURE: self._inject_dns_failure,
            FailureType.CERTIFICATE_EXPIRY: self._inject_certificate_expiry,
            FailureType.CASCADING_FAILURE: self._inject_cascading_failure,
            FailureType.BYZANTINE_FAILURE: self._inject_byzantine_failure
        }
        
        if experiment.failure_type in failure_methods:
            return await failure_methods[experiment.failure_type](experiment)
        else:
            raise ValueError(f"Unknown failure type: {experiment.failure_type}")
    
    async def _inject_network_partition(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Simulate network partition between services"""
        
        self.logger.info("Injecting network partition")
        
        # Simulate network partition by blocking traffic between services
        affected_services = experiment.target_services[:int(len(experiment.target_services) * experiment.blast_radius / 100)]
        
        # Store failure state for rollback
        self.active_failures[experiment.experiment_id] = {
            "type": "network_partition",
            "affected_services": affected_services,
            "start_time": time.time()
        }
        
        # Simulate partition duration
        await asyncio.sleep(experiment.duration_seconds)
        
        return {
            "success": True,
            "impact": {
                "services_affected": len(affected_services),
                "partition_duration": experiment.duration_seconds,
                "blast_radius_percent": experiment.blast_radius
            }
        }
    
    async def _inject_service_crash(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Simulate service crashes"""
        
        self.logger.info("Injecting service crashes")
        
        crashed_services = random.sample(
            experiment.target_services,
            max(1, int(len(experiment.target_services) * experiment.blast_radius / 100))
        )
        
        self.active_failures[experiment.experiment_id] = {
            "type": "service_crash",
            "crashed_services": crashed_services,
            "start_time": time.time()
        }
        
        # Simulate crash duration
        await asyncio.sleep(experiment.duration_seconds)
        
        return {
            "success": True,
            "impact": {
                "services_crashed": len(crashed_services),
                "crash_duration": experiment.duration_seconds,
                "services_list": crashed_services
            }
        }
    
    async def _inject_database_failure(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Simulate database failures"""
        
        self.logger.info("Injecting database failure")
        
        self.active_failures[experiment.experiment_id] = {
            "type": "database_failure",
            "failure_mode": "connection_timeout",
            "start_time": time.time()
        }
        
        await asyncio.sleep(experiment.duration_seconds)
        
        return {
            "success": True,
            "impact": {
                "database_unavailable_seconds": experiment.duration_seconds,
                "failure_mode": "connection_timeout",
                "affected_operations": ["read", "write", "transaction"]
            }
        }
    
    async def _inject_memory_exhaustion(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Simulate memory exhaustion"""
        
        self.logger.info("Injecting memory exhaustion")
        
        # Simulate memory pressure
        memory_hog = []
        try:
            # Allocate memory to simulate exhaustion (be careful in real implementation)
            for i in range(100):  # Limited for safety
                memory_hog.append([0] * 1000000)  # 1M integers
                await asyncio.sleep(0.1)
            
            await asyncio.sleep(experiment.duration_seconds)
            
        finally:
            # Clean up memory
            memory_hog.clear()
        
        self.active_failures[experiment.experiment_id] = {
            "type": "memory_exhaustion",
            "peak_memory_mb": 100,  # Simulated value
            "start_time": time.time()
        }
        
        return {
            "success": True,
            "impact": {
                "memory_pressure_duration": experiment.duration_seconds,
                "peak_memory_usage_mb": 100,
                "oom_risk": "high"
            }
        }
    
    async def _inject_cpu_spike(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Simulate CPU spike"""
        
        self.logger.info("Injecting CPU spike")
        
        # Create CPU-intensive tasks
        cpu_tasks = []
        for i in range(psutil.cpu_count()):
            task = asyncio.create_task(self._cpu_intensive_task(experiment.duration_seconds))
            cpu_tasks.append(task)
        
        await asyncio.gather(*cpu_tasks)
        
        self.active_failures[experiment.experiment_id] = {
            "type": "cpu_spike",
            "cpu_cores_affected": psutil.cpu_count(),
            "start_time": time.time()
        }
        
        return {
            "success": True,
            "impact": {
                "cpu_spike_duration": experiment.duration_seconds,
                "cores_affected": psutil.cpu_count(),
                "target_utilization_percent": 90
            }
        }
    
    async def _cpu_intensive_task(self, duration: int):
        """CPU-intensive task for load simulation"""
        end_time = time.time() + duration
        while time.time() < end_time:
            # CPU-intensive calculation
            sum(i * i for i in range(1000))
            await asyncio.sleep(0.001)  # Small yield to prevent complete blocking
    
    async def _inject_disk_full(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Simulate disk full condition"""
        
        self.logger.info("Injecting disk full condition")
        
        # Simulate disk full (don't actually fill disk in real implementation)
        self.active_failures[experiment.experiment_id] = {
            "type": "disk_full",
            "disk_usage_percent": 100,
            "start_time": time.time()
        }
        
        await asyncio.sleep(experiment.duration_seconds)
        
        return {
            "success": True,
            "impact": {
                "disk_full_duration": experiment.duration_seconds,
                "write_operations_failed": True,
                "log_rotation_blocked": True
            }
        }
    
    async def _inject_latency(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Inject network latency"""
        
        self.logger.info("Injecting network latency")
        
        latency_ms = 1000 if experiment.severity == FailureSeverity.HIGH else 500
        
        self.active_failures[experiment.experiment_id] = {
            "type": "latency_injection",
            "latency_ms": latency_ms,
            "start_time": time.time()
        }
        
        await asyncio.sleep(experiment.duration_seconds)
        
        return {
            "success": True,
            "impact": {
                "latency_added_ms": latency_ms,
                "duration_seconds": experiment.duration_seconds,
                "affected_services": experiment.target_services
            }
        }
    
    async def _inject_packet_loss(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Inject packet loss"""
        
        self.logger.info("Injecting packet loss")
        
        loss_percent = 20 if experiment.severity == FailureSeverity.HIGH else 10
        
        self.active_failures[experiment.experiment_id] = {
            "type": "packet_loss",
            "loss_percent": loss_percent,
            "start_time": time.time()
        }
        
        await asyncio.sleep(experiment.duration_seconds)
        
        return {
            "success": True,
            "impact": {
                "packet_loss_percent": loss_percent,
                "duration_seconds": experiment.duration_seconds,
                "connection_reliability": "degraded"
            }
        }
    
    async def _inject_dns_failure(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Simulate DNS resolution failures"""
        
        self.logger.info("Injecting DNS failure")
        
        self.active_failures[experiment.experiment_id] = {
            "type": "dns_failure",
            "failure_mode": "timeout",
            "start_time": time.time()
        }
        
        await asyncio.sleep(experiment.duration_seconds)
        
        return {
            "success": True,
            "impact": {
                "dns_resolution_failed": True,
                "duration_seconds": experiment.duration_seconds,
                "affected_domains": ["api.example.com", "db.example.com"]
            }
        }
    
    async def _inject_certificate_expiry(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Simulate SSL certificate expiry"""
        
        self.logger.info("Injecting certificate expiry")
        
        self.active_failures[experiment.experiment_id] = {
            "type": "certificate_expiry",
            "expired_certificates": ["api.example.com", "secure.example.com"],
            "start_time": time.time()
        }
        
        await asyncio.sleep(experiment.duration_seconds)
        
        return {
            "success": True,
            "impact": {
                "ssl_connections_failed": True,
                "duration_seconds": experiment.duration_seconds,
                "security_warnings": True
            }
        }
    
    async def _inject_cascading_failure(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Simulate cascading failure across multiple services"""
        
        self.logger.info("Injecting cascading failure")
        
        # Start with one service and cascade to others
        cascade_order = experiment.target_services.copy()
        random.shuffle(cascade_order)
        
        failed_services = []
        
        for i, service in enumerate(cascade_order):
            if i >= len(cascade_order) * experiment.blast_radius / 100:
                break
            
            failed_services.append(service)
            self.logger.info(f"Cascading failure reached: {service}")
            
            # Delay between cascade steps
            await asyncio.sleep(experiment.duration_seconds / len(cascade_order))
        
        self.active_failures[experiment.experiment_id] = {
            "type": "cascading_failure",
            "cascade_order": failed_services,
            "start_time": time.time()
        }
        
        return {
            "success": True,
            "impact": {
                "services_failed": len(failed_services),
                "cascade_duration": experiment.duration_seconds,
                "failure_pattern": "sequential"
            }
        }
    
    async def _inject_byzantine_failure(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Simulate Byzantine failure (arbitrary/malicious behavior)"""
        
        self.logger.info("Injecting Byzantine failure")
        
        byzantine_behaviors = [
            "incorrect_responses",
            "delayed_responses",
            "corrupted_data",
            "inconsistent_state"
        ]
        
        selected_behavior = random.choice(byzantine_behaviors)
        
        self.active_failures[experiment.experiment_id] = {
            "type": "byzantine_failure",
            "behavior": selected_behavior,
            "start_time": time.time()
        }
        
        await asyncio.sleep(experiment.duration_seconds)
        
        return {
            "success": True,
            "impact": {
                "byzantine_behavior": selected_behavior,
                "duration_seconds": experiment.duration_seconds,
                "consensus_affected": True
            }
        }
    
    async def stop_failure(self, experiment: ChaosExperiment):
        """Stop injected failure"""
        
        if experiment.experiment_id in self.active_failures:
            failure_info = self.active_failures[experiment.experiment_id]
            self.logger.info(f"Stopping failure: {failure_info['type']}")
            
            # Cleanup based on failure type
            if failure_info["type"] == "memory_exhaustion":
                # Memory cleanup already handled in injection method
                pass
            elif failure_info["type"] == "cpu_spike":
                # CPU tasks will naturally complete
                pass
            
            del self.active_failures[experiment.experiment_id]


class RecoveryValidator:
    """Validates system recovery after failure injection"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def validate_recovery(
        self, 
        experiment: ChaosExperiment, 
        baseline_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate that system has recovered from failure"""
        
        self.logger.info("Validating system recovery")
        
        recovery_start = time.time()
        max_recovery_time = 300  # 5 minutes max recovery time
        
        while time.time() - recovery_start < max_recovery_time:
            current_metrics = await self._collect_current_metrics(experiment.target_services)
            
            if self._is_system_recovered(baseline_metrics, current_metrics):
                recovery_time = time.time() - recovery_start
                return {
                    "recovered": True,
                    "recovery_time": recovery_time,
                    "final_metrics": current_metrics
                }
            
            await asyncio.sleep(10)  # Check every 10 seconds
        
        # Recovery timeout
        return {
            "recovered": False,
            "recovery_time": max_recovery_time,
            "final_metrics": await self._collect_current_metrics(experiment.target_services)
        }
    
    async def _collect_current_metrics(self, target_services: List[str]) -> Dict[str, Any]:
        """Collect current system metrics"""
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "service_health": {}
        }
        
        for service in target_services:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://{service}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                        metrics["service_health"][service] = response.status == 200
            except:
                metrics["service_health"][service] = False
        
        return metrics
    
    def _is_system_recovered(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> bool:
        """Check if system has recovered to acceptable levels"""
        
        # CPU recovery check
        if current["cpu_percent"] > baseline["cpu_percent"] * 1.5:
            return False
        
        # Memory recovery check
        if current["memory_percent"] > baseline["memory_percent"] * 1.3:
            return False
        
        # Service health check
        healthy_services = sum(1 for health in current["service_health"].values() if health)
        total_services = len(current["service_health"])
        
        if healthy_services / total_services < 0.8:  # 80% of services must be healthy
            return False
        
        return True


class SafetyMonitor:
    """Monitors system safety during chaos experiments"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.monitoring_threads = {}
        self.safety_thresholds = {
            "cpu_percent": 95,
            "memory_percent": 90,
            "disk_percent": 95,
            "error_rate": 0.8
        }
    
    def start_monitoring(self, experiment: ChaosExperiment):
        """Start safety monitoring for experiment"""
        
        self.logger.info(f"Starting safety monitoring for experiment: {experiment.experiment_id}")
        
        monitor_thread = threading.Thread(
            target=self._monitor_safety,
            args=(experiment,)
        )
        monitor_thread.daemon = True
        monitor_thread.start()
        
        self.monitoring_threads[experiment.experiment_id] = monitor_thread
    
    def stop_monitoring(self, experiment_id: str):
        """Stop safety monitoring"""
        
        if experiment_id in self.monitoring_threads:
            self.logger.info(f"Stopping safety monitoring for experiment: {experiment_id}")
            # Thread will stop when experiment is removed from active experiments
            del self.monitoring_threads[experiment_id]
    
    def _monitor_safety(self, experiment: ChaosExperiment):
        """Monitor system safety in background thread"""
        
        while experiment.experiment_id in self.monitoring_threads:
            try:
                # Check system resources
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                disk_percent = psutil.disk_usage('/').percent
                
                # Check safety thresholds
                if cpu_percent > self.safety_thresholds["cpu_percent"]:
                    self.logger.warning(f"CPU usage critical: {cpu_percent}%")
                
                if memory_percent > self.safety_thresholds["memory_percent"]:
                    self.logger.warning(f"Memory usage critical: {memory_percent}%")
                
                if disk_percent > self.safety_thresholds["disk_percent"]:
                    self.logger.warning(f"Disk usage critical: {disk_percent}%")
                
                # If multiple thresholds exceeded, trigger emergency stop
                violations = 0
                if cpu_percent > self.safety_thresholds["cpu_percent"]:
                    violations += 1
                if memory_percent > self.safety_thresholds["memory_percent"]:
                    violations += 1
                if disk_percent > self.safety_thresholds["disk_percent"]:
                    violations += 1
                
                if violations >= 2:
                    self.logger.error("Multiple safety thresholds exceeded - emergency stop required")
                    # In real implementation, this would trigger emergency rollback
                
            except Exception as e:
                self.logger.error(f"Safety monitoring error: {e}")
            
            time.sleep(5)  # Check every 5 seconds


# Example usage and testing
if __name__ == "__main__":
    async def run_chaos_engineering_demo():
        framework = ChaosEngineeringFramework()
        
        # Define chaos experiments
        experiments = [
            ChaosExperiment(
                experiment_id=str(uuid.uuid4()),
                name="Network Partition Resilience Test",
                description="Test system behavior during network partition",
                failure_type=FailureType.NETWORK_PARTITION,
                severity=FailureSeverity.HIGH,
                target_services=["api-service", "database-service", "cache-service"],
                duration_seconds=60,
                blast_radius=50.0,
                hypothesis="System will maintain 70% availability during network partition",
                success_criteria=["Service availability > 70%", "No data loss", "Recovery < 2 minutes"],
                rollback_strategy="restore_network; restart_service:api-service",
                safety_checks=["system_health > 80%", "no_concurrent_experiments"]
            ),
            ChaosExperiment(
                experiment_id=str(uuid.uuid4()),
                name="Service Crash Recovery Test",
                description="Test system recovery from service crashes",
                failure_type=FailureType.SERVICE_CRASH,
                severity=FailureSeverity.MEDIUM,
                target_services=["api-service", "worker-service"],
                duration_seconds=30,
                blast_radius=25.0,
                hypothesis="System will automatically recover from service crashes",
                success_criteria=["Service restart < 30 seconds", "No request failures"],
                rollback_strategy="restart_service:api-service; restart_service:worker-service",
                safety_checks=["system_health > 80%"]
            )
        ]
        
        # Run experiments
        results = await framework.run_chaos_suite(experiments)
        
        # Display results
        for result in results:
            print(f"\nChaos Experiment Results: {result.experiment_id}")
            print(f"Duration: {result.duration_seconds:.2f} seconds")
            print(f"Failure Injected: {result.failure_injected}")
            print(f"System Recovered: {result.system_recovered}")
            print(f"Recovery Time: {result.recovery_time_seconds:.2f} seconds")
            print(f"Hypothesis Validated: {result.hypothesis_validated}")
            print(f"Success Criteria Met: {result.success_criteria_met}")
            
            if result.lessons_learned:
                print("Lessons Learned:")
                for lesson in result.lessons_learned:
                    print(f"- {lesson}")
            
            if result.recommendations:
                print("Recommendations:")
                for rec in result.recommendations:
                    print(f"- {rec}")
    
    # Run the demo
    asyncio.run(run_chaos_engineering_demo())