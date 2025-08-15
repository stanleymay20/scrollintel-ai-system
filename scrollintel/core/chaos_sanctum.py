"""
Chaos Sanctum for ScrollIntel-G6.
Implements scheduled chaos tests and auto-learned failover playbooks.
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import json

from .proof_of_workflow import create_workflow_attestation

logger = logging.getLogger(__name__)


class ChaosType(Enum):
    LATENCY_SPIKE = "latency_spike"
    MODEL_OUTAGE = "model_outage"
    BAD_WEIGHTS = "bad_weights"
    CONNECTOR_FAILURE = "connector_failure"
    MEMORY_PRESSURE = "memory_pressure"
    NETWORK_PARTITION = "network_partition"
    DATABASE_SLOWDOWN = "database_slowdown"
    CACHE_INVALIDATION = "cache_invalidation"


class ChaosImpact(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ChaosExperiment:
    """Definition of a chaos experiment."""
    
    id: str
    name: str
    chaos_type: ChaosType
    impact_level: ChaosImpact
    duration_seconds: int
    target_components: List[str]
    parameters: Dict[str, Any]
    success_criteria: Dict[str, Any]
    rollback_strategy: str
    enabled: bool = True


@dataclass
class ChaosResult:
    """Result of a chaos experiment."""
    
    experiment_id: str
    start_time: datetime
    end_time: datetime
    success: bool
    impact_observed: ChaosImpact
    metrics_before: Dict[str, float]
    metrics_during: Dict[str, float]
    metrics_after: Dict[str, float]
    recovery_time_seconds: float
    lessons_learned: List[str]
    playbook_updates: List[str]


class FailoverPlaybook:
    """Automated failover playbook."""
    
    def __init__(self, name: str, triggers: List[str], actions: List[Callable]):
        self.name = name
        self.triggers = triggers
        self.actions = actions
        self.execution_count = 0
        self.success_rate = 0.0
        self.last_updated = datetime.utcnow()
    
    async def execute(self, context: Dict[str, Any]) -> bool:
        """Execute the failover playbook."""
        logger.info(f"Executing failover playbook: {self.name}")
        
        try:
            for action in self.actions:
                await action(context)
            
            self.execution_count += 1
            # Update success rate (simplified)
            self.success_rate = min(1.0, self.success_rate + 0.1)
            
            logger.info(f"Failover playbook {self.name} executed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failover playbook {self.name} failed: {e}")
            self.success_rate = max(0.0, self.success_rate - 0.2)
            return False
    
    def update_from_experiment(self, result: ChaosResult) -> None:
        """Update playbook based on chaos experiment results."""
        if result.success and result.recovery_time_seconds < 60:
            # Experiment showed good recovery, no changes needed
            return
        
        # Learn from failures
        for lesson in result.lessons_learned:
            if "timeout" in lesson.lower():
                # Add timeout handling
                self.triggers.append("timeout_detected")
            elif "retry" in lesson.lower():
                # Add retry logic
                self.triggers.append("retry_exhausted")
        
        self.last_updated = datetime.utcnow()
        logger.info(f"Updated playbook {self.name} based on experiment {result.experiment_id}")


class MetricsCollector:
    """Collects system metrics for chaos experiments."""
    
    def __init__(self):
        self.metrics_history: Dict[str, List[Tuple[datetime, float]]] = {}
    
    async def collect_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        metrics = {}
        
        # Response time metrics
        metrics["avg_response_time"] = await self._measure_response_time()
        
        # Error rate metrics
        metrics["error_rate"] = await self._measure_error_rate()
        
        # Throughput metrics
        metrics["requests_per_second"] = await self._measure_throughput()
        
        # Resource utilization
        metrics["cpu_usage"] = await self._measure_cpu_usage()
        metrics["memory_usage"] = await self._measure_memory_usage()
        
        # Model availability
        metrics["model_availability"] = await self._measure_model_availability()
        
        # Cache hit rate
        metrics["cache_hit_rate"] = await self._measure_cache_hit_rate()
        
        # Store in history
        timestamp = datetime.utcnow()
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
            self.metrics_history[metric_name].append((timestamp, value))
            
            # Keep only last 1000 measurements
            if len(self.metrics_history[metric_name]) > 1000:
                self.metrics_history[metric_name] = self.metrics_history[metric_name][-1000:]
        
        return metrics
    
    async def _measure_response_time(self) -> float:
        """Measure average response time."""
        # Simulate measurement
        return random.uniform(0.1, 2.0)
    
    async def _measure_error_rate(self) -> float:
        """Measure error rate."""
        return random.uniform(0.0, 0.05)
    
    async def _measure_throughput(self) -> float:
        """Measure requests per second."""
        return random.uniform(10.0, 100.0)
    
    async def _measure_cpu_usage(self) -> float:
        """Measure CPU usage percentage."""
        return random.uniform(20.0, 80.0)
    
    async def _measure_memory_usage(self) -> float:
        """Measure memory usage percentage."""
        return random.uniform(30.0, 70.0)
    
    async def _measure_model_availability(self) -> float:
        """Measure model availability percentage."""
        return random.uniform(0.95, 1.0)
    
    async def _measure_cache_hit_rate(self) -> float:
        """Measure cache hit rate."""
        return random.uniform(0.6, 0.9)
    
    def get_baseline_metrics(self, window_minutes: int = 30) -> Dict[str, float]:
        """Get baseline metrics from recent history."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
        baseline = {}
        
        for metric_name, history in self.metrics_history.items():
            recent_values = [value for timestamp, value in history if timestamp >= cutoff_time]
            if recent_values:
                baseline[metric_name] = sum(recent_values) / len(recent_values)
            else:
                baseline[metric_name] = 0.0
        
        return baseline


class ChaosInjector:
    """Injects chaos into the system."""
    
    def __init__(self):
        self.active_experiments: Dict[str, ChaosExperiment] = {}
        self.injection_methods = {
            ChaosType.LATENCY_SPIKE: self._inject_latency_spike,
            ChaosType.MODEL_OUTAGE: self._inject_model_outage,
            ChaosType.BAD_WEIGHTS: self._inject_bad_weights,
            ChaosType.CONNECTOR_FAILURE: self._inject_connector_failure,
            ChaosType.MEMORY_PRESSURE: self._inject_memory_pressure,
            ChaosType.NETWORK_PARTITION: self._inject_network_partition,
            ChaosType.DATABASE_SLOWDOWN: self._inject_database_slowdown,
            ChaosType.CACHE_INVALIDATION: self._inject_cache_invalidation,
        }
    
    async def inject_chaos(self, experiment: ChaosExperiment) -> None:
        """Inject chaos according to experiment specification."""
        logger.warning(f"Injecting chaos: {experiment.name} ({experiment.chaos_type.value})")
        
        self.active_experiments[experiment.id] = experiment
        
        try:
            injection_method = self.injection_methods[experiment.chaos_type]
            await injection_method(experiment)
            
            # Wait for experiment duration
            await asyncio.sleep(experiment.duration_seconds)
            
        finally:
            # Always clean up
            await self._cleanup_experiment(experiment)
            del self.active_experiments[experiment.id]
    
    async def _inject_latency_spike(self, experiment: ChaosExperiment) -> None:
        """Inject artificial latency."""
        delay_ms = experiment.parameters.get("delay_ms", 1000)
        
        # This would integrate with actual request handling to add delays
        logger.info(f"Injecting {delay_ms}ms latency spike")
        
        # Simulate by adding global delay flag
        import os
        os.environ["CHAOS_LATENCY_MS"] = str(delay_ms)
    
    async def _inject_model_outage(self, experiment: ChaosExperiment) -> None:
        """Simulate model outage."""
        model_name = experiment.parameters.get("model_name", "gpt-4")
        
        logger.info(f"Simulating outage for model: {model_name}")
        
        # This would integrate with model routing to block specific models
        import os
        os.environ[f"CHAOS_MODEL_OUTAGE_{model_name.upper()}"] = "true"
    
    async def _inject_bad_weights(self, experiment: ChaosExperiment) -> None:
        """Inject corrupted model weights."""
        corruption_rate = experiment.parameters.get("corruption_rate", 0.1)
        
        logger.info(f"Injecting weight corruption at rate: {corruption_rate}")
        
        # This would integrate with model loading to corrupt weights
        import os
        os.environ["CHAOS_WEIGHT_CORRUPTION"] = str(corruption_rate)
    
    async def _inject_connector_failure(self, experiment: ChaosExperiment) -> None:
        """Simulate connector failure."""
        connector_type = experiment.parameters.get("connector_type", "database")
        
        logger.info(f"Simulating failure for connector: {connector_type}")
        
        import os
        os.environ[f"CHAOS_CONNECTOR_FAILURE_{connector_type.upper()}"] = "true"
    
    async def _inject_memory_pressure(self, experiment: ChaosExperiment) -> None:
        """Simulate memory pressure."""
        pressure_mb = experiment.parameters.get("pressure_mb", 1000)
        
        logger.info(f"Simulating memory pressure: {pressure_mb}MB")
        
        # Allocate memory to simulate pressure
        self._memory_ballast = bytearray(pressure_mb * 1024 * 1024)
    
    async def _inject_network_partition(self, experiment: ChaosExperiment) -> None:
        """Simulate network partition."""
        target_hosts = experiment.parameters.get("target_hosts", ["api.openai.com"])
        
        logger.info(f"Simulating network partition for hosts: {target_hosts}")
        
        # This would integrate with network layer to block specific hosts
        import os
        os.environ["CHAOS_BLOCKED_HOSTS"] = ",".join(target_hosts)
    
    async def _inject_database_slowdown(self, experiment: ChaosExperiment) -> None:
        """Simulate database slowdown."""
        delay_ms = experiment.parameters.get("delay_ms", 500)
        
        logger.info(f"Simulating database slowdown: {delay_ms}ms")
        
        import os
        os.environ["CHAOS_DB_DELAY_MS"] = str(delay_ms)
    
    async def _inject_cache_invalidation(self, experiment: ChaosExperiment) -> None:
        """Simulate cache invalidation."""
        invalidation_rate = experiment.parameters.get("invalidation_rate", 0.5)
        
        logger.info(f"Simulating cache invalidation at rate: {invalidation_rate}")
        
        import os
        os.environ["CHAOS_CACHE_INVALIDATION"] = str(invalidation_rate)
    
    async def _cleanup_experiment(self, experiment: ChaosExperiment) -> None:
        """Clean up after chaos experiment."""
        logger.info(f"Cleaning up chaos experiment: {experiment.name}")
        
        # Remove environment variables
        import os
        chaos_vars = [key for key in os.environ.keys() if key.startswith("CHAOS_")]
        for var in chaos_vars:
            del os.environ[var]
        
        # Clean up memory ballast
        if hasattr(self, '_memory_ballast'):
            del self._memory_ballast


class ChaosSanctum:
    """Main chaos engineering orchestrator."""
    
    def __init__(self):
        self.experiments = self._load_default_experiments()
        self.playbooks = self._load_default_playbooks()
        self.metrics_collector = MetricsCollector()
        self.chaos_injector = ChaosInjector()
        self.experiment_history: List[ChaosResult] = []
        self.is_running = False
    
    def _load_default_experiments(self) -> List[ChaosExperiment]:
        """Load default chaos experiments."""
        return [
            ChaosExperiment(
                id="latency_spike_low",
                name="Low Impact Latency Spike",
                chaos_type=ChaosType.LATENCY_SPIKE,
                impact_level=ChaosImpact.LOW,
                duration_seconds=30,
                target_components=["api_gateway"],
                parameters={"delay_ms": 200},
                success_criteria={"max_error_rate": 0.05, "max_response_time": 3.0},
                rollback_strategy="automatic"
            ),
            ChaosExperiment(
                id="model_outage_medium",
                name="Medium Impact Model Outage",
                chaos_type=ChaosType.MODEL_OUTAGE,
                impact_level=ChaosImpact.MEDIUM,
                duration_seconds=60,
                target_components=["model_router"],
                parameters={"model_name": "gpt-4"},
                success_criteria={"max_error_rate": 0.1, "min_availability": 0.9},
                rollback_strategy="automatic"
            ),
            ChaosExperiment(
                id="connector_failure_high",
                name="High Impact Connector Failure",
                chaos_type=ChaosType.CONNECTOR_FAILURE,
                impact_level=ChaosImpact.HIGH,
                duration_seconds=45,
                target_components=["database_connector"],
                parameters={"connector_type": "database"},
                success_criteria={"max_error_rate": 0.15, "min_availability": 0.8},
                rollback_strategy="manual"
            ),
        ]
    
    def _load_default_playbooks(self) -> List[FailoverPlaybook]:
        """Load default failover playbooks."""
        return [
            FailoverPlaybook(
                name="Model Failover",
                triggers=["model_outage", "model_timeout"],
                actions=[self._switch_to_backup_model, self._notify_operations]
            ),
            FailoverPlaybook(
                name="Database Failover",
                triggers=["database_timeout", "connection_error"],
                actions=[self._switch_to_read_replica, self._enable_cache_fallback]
            ),
            FailoverPlaybook(
                name="High Latency Response",
                triggers=["latency_spike", "timeout_detected"],
                actions=[self._enable_circuit_breaker, self._scale_up_resources]
            ),
        ]
    
    async def start_chaos_testing(self, schedule_interval_hours: int = 24) -> None:
        """Start scheduled chaos testing."""
        logger.info("Starting Chaos Sanctum with scheduled testing")
        self.is_running = True
        
        while self.is_running:
            try:
                # Run a random experiment
                experiment = random.choice([e for e in self.experiments if e.enabled])
                result = await self.run_experiment(experiment)
                
                # Learn from the result
                await self._learn_from_experiment(result)
                
                # Wait for next scheduled run
                await asyncio.sleep(schedule_interval_hours * 3600)
                
            except Exception as e:
                logger.error(f"Error in chaos testing cycle: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retrying
    
    async def run_experiment(self, experiment: ChaosExperiment) -> ChaosResult:
        """Run a single chaos experiment."""
        logger.info(f"Running chaos experiment: {experiment.name}")
        
        start_time = datetime.utcnow()
        
        # Collect baseline metrics
        metrics_before = await self.metrics_collector.collect_metrics()
        
        # Start chaos injection
        chaos_task = asyncio.create_task(self.chaos_injector.inject_chaos(experiment))
        
        # Monitor during experiment
        await asyncio.sleep(5)  # Let chaos settle
        metrics_during = await self.metrics_collector.collect_metrics()
        
        # Wait for experiment to complete
        await chaos_task
        
        # Collect post-experiment metrics
        await asyncio.sleep(10)  # Let system recover
        metrics_after = await self.metrics_collector.collect_metrics()
        
        end_time = datetime.utcnow()
        
        # Analyze results
        success = self._evaluate_experiment_success(experiment, metrics_before, metrics_during, metrics_after)
        impact_observed = self._assess_impact(metrics_before, metrics_during)
        recovery_time = self._calculate_recovery_time(metrics_during, metrics_after)
        lessons_learned = self._extract_lessons(experiment, metrics_before, metrics_during, metrics_after)
        
        result = ChaosResult(
            experiment_id=experiment.id,
            start_time=start_time,
            end_time=end_time,
            success=success,
            impact_observed=impact_observed,
            metrics_before=metrics_before,
            metrics_during=metrics_during,
            metrics_after=metrics_after,
            recovery_time_seconds=recovery_time,
            lessons_learned=lessons_learned,
            playbook_updates=[]
        )
        
        self.experiment_history.append(result)
        
        # Create attestation
        create_workflow_attestation(
            action_type="chaos_experiment",
            agent_id="chaos_sanctum",
            user_id="system",
            prompt=f"Chaos experiment: {experiment.name}",
            tools_used=["chaos_injector", "metrics_collector"],
            datasets_used=[],
            model_version="chaos-v1.0",
            verifier_evidence={
                "experiment_success": success,
                "impact_level": impact_observed.value,
                "recovery_time": recovery_time
            },
            content=result.__dict__
        )
        
        logger.info(f"Chaos experiment completed: {experiment.name} (Success: {success})")
        return result
    
    def _evaluate_experiment_success(
        self,
        experiment: ChaosExperiment,
        before: Dict[str, float],
        during: Dict[str, float],
        after: Dict[str, float]
    ) -> bool:
        """Evaluate if experiment met success criteria."""
        
        criteria = experiment.success_criteria
        
        # Check error rate
        if "max_error_rate" in criteria:
            if during.get("error_rate", 0) > criteria["max_error_rate"]:
                return False
        
        # Check response time
        if "max_response_time" in criteria:
            if during.get("avg_response_time", 0) > criteria["max_response_time"]:
                return False
        
        # Check availability
        if "min_availability" in criteria:
            if during.get("model_availability", 1) < criteria["min_availability"]:
                return False
        
        # Check recovery
        recovery_threshold = 1.2  # 20% above baseline
        for metric, before_value in before.items():
            after_value = after.get(metric, before_value)
            if after_value > before_value * recovery_threshold:
                return False
        
        return True
    
    def _assess_impact(self, before: Dict[str, float], during: Dict[str, float]) -> ChaosImpact:
        """Assess the actual impact of the chaos experiment."""
        
        max_degradation = 0.0
        
        for metric, before_value in before.items():
            during_value = during.get(metric, before_value)
            
            if metric in ["error_rate"]:
                # Higher is worse
                degradation = (during_value - before_value) / max(before_value, 0.01)
            else:
                # Lower is worse (for response time, availability, etc.)
                degradation = (before_value - during_value) / max(before_value, 0.01)
            
            max_degradation = max(max_degradation, degradation)
        
        if max_degradation < 0.1:
            return ChaosImpact.LOW
        elif max_degradation < 0.3:
            return ChaosImpact.MEDIUM
        elif max_degradation < 0.6:
            return ChaosImpact.HIGH
        else:
            return ChaosImpact.CRITICAL
    
    def _calculate_recovery_time(self, during: Dict[str, float], after: Dict[str, float]) -> float:
        """Calculate recovery time in seconds."""
        # Simplified calculation - in practice this would be more sophisticated
        return 30.0  # Assume 30 seconds recovery time
    
    def _extract_lessons(
        self,
        experiment: ChaosExperiment,
        before: Dict[str, float],
        during: Dict[str, float],
        after: Dict[str, float]
    ) -> List[str]:
        """Extract lessons learned from the experiment."""
        
        lessons = []
        
        # Check for slow recovery
        if after.get("avg_response_time", 0) > before.get("avg_response_time", 0) * 1.5:
            lessons.append("System recovery is slower than expected - consider timeout adjustments")
        
        # Check for error spikes
        if during.get("error_rate", 0) > before.get("error_rate", 0) * 3:
            lessons.append("Error rate spikes significantly - improve retry logic")
        
        # Check for availability issues
        if during.get("model_availability", 1) < 0.9:
            lessons.append("Model availability drops below threshold - implement better failover")
        
        return lessons
    
    async def _learn_from_experiment(self, result: ChaosResult) -> None:
        """Learn from experiment results and update playbooks."""
        
        # Update relevant playbooks
        for playbook in self.playbooks:
            if any(trigger in result.lessons_learned[0].lower() if result.lessons_learned else False 
                   for trigger in playbook.triggers):
                playbook.update_from_experiment(result)
        
        # Create new playbooks if needed
        if not result.success and result.impact_observed in [ChaosImpact.HIGH, ChaosImpact.CRITICAL]:
            await self._create_emergency_playbook(result)
    
    async def _create_emergency_playbook(self, result: ChaosResult) -> None:
        """Create emergency playbook for critical failures."""
        
        playbook_name = f"Emergency Response - {result.experiment_id}"
        triggers = [f"critical_failure_{result.experiment_id}"]
        actions = [self._emergency_shutdown, self._notify_on_call]
        
        emergency_playbook = FailoverPlaybook(playbook_name, triggers, actions)
        self.playbooks.append(emergency_playbook)
        
        logger.warning(f"Created emergency playbook: {playbook_name}")
    
    # Playbook action methods
    async def _switch_to_backup_model(self, context: Dict[str, Any]) -> None:
        """Switch to backup model."""
        logger.info("Switching to backup model")
        # Implementation would update model routing
    
    async def _notify_operations(self, context: Dict[str, Any]) -> None:
        """Notify operations team."""
        logger.info("Notifying operations team")
        # Implementation would send alerts
    
    async def _switch_to_read_replica(self, context: Dict[str, Any]) -> None:
        """Switch to database read replica."""
        logger.info("Switching to database read replica")
        # Implementation would update database routing
    
    async def _enable_cache_fallback(self, context: Dict[str, Any]) -> None:
        """Enable cache fallback mode."""
        logger.info("Enabling cache fallback mode")
        # Implementation would enable cache-only responses
    
    async def _enable_circuit_breaker(self, context: Dict[str, Any]) -> None:
        """Enable circuit breaker."""
        logger.info("Enabling circuit breaker")
        # Implementation would enable circuit breaker pattern
    
    async def _scale_up_resources(self, context: Dict[str, Any]) -> None:
        """Scale up resources."""
        logger.info("Scaling up resources")
        # Implementation would trigger auto-scaling
    
    async def _emergency_shutdown(self, context: Dict[str, Any]) -> None:
        """Emergency shutdown procedure."""
        logger.critical("Executing emergency shutdown")
        # Implementation would safely shut down critical components
    
    async def _notify_on_call(self, context: Dict[str, Any]) -> None:
        """Notify on-call engineer."""
        logger.critical("Notifying on-call engineer")
        # Implementation would send urgent alerts
    
    def stop_chaos_testing(self) -> None:
        """Stop scheduled chaos testing."""
        logger.info("Stopping Chaos Sanctum")
        self.is_running = False
    
    def get_experiment_history(self) -> List[ChaosResult]:
        """Get history of chaos experiments."""
        return self.experiment_history.copy()
    
    def get_playbook_status(self) -> List[Dict[str, Any]]:
        """Get status of all failover playbooks."""
        return [
            {
                "name": playbook.name,
                "triggers": playbook.triggers,
                "execution_count": playbook.execution_count,
                "success_rate": playbook.success_rate,
                "last_updated": playbook.last_updated.isoformat()
            }
            for playbook in self.playbooks
        ]


# Global chaos sanctum instance
chaos_sanctum = ChaosSanctum()


async def start_chaos_testing(schedule_interval_hours: int = 24) -> None:
    """Start chaos testing (convenience function)."""
    await chaos_sanctum.start_chaos_testing(schedule_interval_hours)


async def run_chaos_experiment(experiment_id: str) -> ChaosResult:
    """Run a specific chaos experiment."""
    experiment = next((e for e in chaos_sanctum.experiments if e.id == experiment_id), None)
    if not experiment:
        raise ValueError(f"Experiment not found: {experiment_id}")
    
    return await chaos_sanctum.run_experiment(experiment)


def get_chaos_status() -> Dict[str, Any]:
    """Get current chaos testing status."""
    return {
        "is_running": chaos_sanctum.is_running,
        "experiments_available": len(chaos_sanctum.experiments),
        "playbooks_available": len(chaos_sanctum.playbooks),
        "experiments_completed": len(chaos_sanctum.experiment_history),
        "last_experiment": chaos_sanctum.experiment_history[-1].experiment_id if chaos_sanctum.experiment_history else None
    }