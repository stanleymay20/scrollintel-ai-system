"""
Enterprise Stress Testing Orchestrator
Coordinates and executes comprehensive stress testing scenarios
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid

from .load_testing_framework import EnterpriseLoadTester, LoadTestConfig, LoadTestResult
from .compliance_audit_simulator import ComplianceAuditSimulator, ComplianceFramework, AuditScenario, AuditResult
from .chaos_engineering_framework import ChaosEngineeringFramework, ChaosExperiment, FailureType, FailureSeverity, ChaosResult
from .petabyte_data_processor import PetabyteDataProcessor, ProcessingWorkload, ProcessingType, DataType, ProcessingResult
from .multi_tenant_isolation_tester import MultiTenantIsolationTester, TenantConfig, TenantTier, IsolationTest, AttackType, IsolationType, IsolationResult

class TestPhase(Enum):
    LOAD_TESTING = "load_testing"
    COMPLIANCE_AUDIT = "compliance_audit"
    CHAOS_ENGINEERING = "chaos_engineering"
    PETABYTE_PROCESSING = "petabyte_processing"
    MULTI_TENANT_ISOLATION = "multi_tenant_isolation"

@dataclass
class StressTestSuite:
    """Defines a comprehensive stress test suite"""
    suite_id: str
    name: str
    description: str
    phases: List[TestPhase]
    target_system: str
    duration_hours: float
    concurrent_users: int
    data_volume_tb: float
    compliance_frameworks: List[ComplianceFramework]
    chaos_scenarios: List[str]
    tenant_count: int
    performance_targets: Dict[str, float]

@dataclass
class StressTestResults:
    """Comprehensive results from stress testing suite"""
    suite_id: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    phases_executed: List[TestPhase]
    load_test_results: Optional[LoadTestResult]
    compliance_results: List[AuditResult]
    chaos_results: List[ChaosResult]
    processing_results: List[ProcessingResult]
    isolation_results: List[IsolationResult]
    overall_success_rate: float
    performance_score: float
    resilience_score: float
    compliance_score: float
    isolation_score: float
    recommendations: List[str]
    critical_issues: List[str]

class EnterpriseStressTestOrchestrator:
    """Orchestrates comprehensive enterprise stress testing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.load_tester = EnterpriseLoadTester(LoadTestConfig())
        self.compliance_simulator = ComplianceAuditSimulator()
        self.chaos_framework = ChaosEngineeringFramework()
        self.data_processor = PetabyteDataProcessor()
        self.isolation_tester = MultiTenantIsolationTester()
        
    async def execute_stress_test_suite(self, suite: StressTestSuite) -> StressTestResults:
        """Execute a comprehensive stress test suite"""
        
        self.logger.info(f"Starting stress test suite: {suite.name}")
        start_time = datetime.now()
        
        results = StressTestResults(
            suite_id=suite.suite_id,
            start_time=start_time,
            end_time=start_time,  # Will be updated
            duration_seconds=0,
            phases_executed=[],
            load_test_results=None,
            compliance_results=[],
            chaos_results=[],
            processing_results=[],
            isolation_results=[],
            overall_success_rate=0.0,
            performance_score=0.0,
            resilience_score=0.0,
            compliance_score=0.0,
            isolation_score=0.0,
            recommendations=[],
            critical_issues=[]
        )
        
        try:
            # Execute test phases in sequence
            for phase in suite.phases:
                self.logger.info(f"Executing phase: {phase.value}")
                
                if phase == TestPhase.LOAD_TESTING:
                    results.load_test_results = await self._execute_load_testing(suite)
                    results.phases_executed.append(phase)
                    
                elif phase == TestPhase.COMPLIANCE_AUDIT:
                    results.compliance_results = await self._execute_compliance_testing(suite)
                    results.phases_executed.append(phase)
                    
                elif phase == TestPhase.CHAOS_ENGINEERING:
                    results.chaos_results = await self._execute_chaos_testing(suite)
                    results.phases_executed.append(phase)
                    
                elif phase == TestPhase.PETABYTE_PROCESSING:
                    results.processing_results = await self._execute_petabyte_testing(suite)
                    results.phases_executed.append(phase)
                    
                elif phase == TestPhase.MULTI_TENANT_ISOLATION:
                    results.isolation_results = await self._execute_isolation_testing(suite)
                    results.phases_executed.append(phase)
                
                # Brief pause between phases
                await asyncio.sleep(30)
            
            # Calculate final scores and metrics
            end_time = datetime.now()
            results.end_time = end_time
            results.duration_seconds = (end_time - start_time).total_seconds()
            
            results = self._calculate_final_scores(results)
            results = self._generate_recommendations(results, suite)
            
            self.logger.info(f"Stress test suite completed in {results.duration_seconds:.2f} seconds")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Stress test suite failed: {e}")
            results.critical_issues.append(f"Suite execution failed: {str(e)}")
            return results
    
    async def _execute_load_testing(self, suite: StressTestSuite) -> LoadTestResult:
        """Execute load testing phase"""
        
        self.logger.info("Executing load testing phase")
        
        config = LoadTestConfig(
            concurrent_users=suite.concurrent_users,
            test_duration_minutes=int(suite.duration_hours * 60 / len(suite.phases)),
            ramp_up_time_minutes=5,
            enterprise_features_enabled=True
        )
        
        tester = EnterpriseLoadTester(config)
        result = await tester.execute_load_test(f"http://{suite.target_system}")
        
        return result
    
    async def _execute_compliance_testing(self, suite: StressTestSuite) -> List[AuditResult]:
        """Execute compliance audit testing phase"""
        
        self.logger.info("Executing compliance testing phase")
        
        results = []
        
        for framework in suite.compliance_frameworks:
            # Test different audit scenarios
            scenarios = [
                AuditScenario.SURPRISE_INSPECTION,
                AuditScenario.SCHEDULED_AUDIT,
                AuditScenario.COMPLIANCE_VALIDATION
            ]
            
            for scenario in scenarios:
                try:
                    result = await self.compliance_simulator.simulate_regulatory_inspection(
                        framework=framework,
                        scenario=scenario,
                        concurrent_inspectors=3,
                        system_load_percent=75
                    )
                    results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Compliance test failed for {framework.value}: {e}")
        
        return results
    
    async def _execute_chaos_testing(self, suite: StressTestSuite) -> List[ChaosResult]:
        """Execute chaos engineering testing phase"""
        
        self.logger.info("Executing chaos engineering phase")
        
        # Define chaos experiments based on suite configuration
        experiments = [
            ChaosExperiment(
                experiment_id=str(uuid.uuid4()),
                name="Network Partition Test",
                description="Test system resilience during network partitions",
                failure_type=FailureType.NETWORK_PARTITION,
                severity=FailureSeverity.HIGH,
                target_services=["api-service", "database-service", "cache-service"],
                duration_seconds=120,
                blast_radius=30.0,
                hypothesis="System maintains 80% availability during network partition",
                success_criteria=["Availability > 80%", "Recovery < 3 minutes"],
                rollback_strategy="restore_network; restart_service:api-service",
                safety_checks=["system_health > 70%"]
            ),
            ChaosExperiment(
                experiment_id=str(uuid.uuid4()),
                name="Service Crash Recovery",
                description="Test automatic recovery from service crashes",
                failure_type=FailureType.SERVICE_CRASH,
                severity=FailureSeverity.MEDIUM,
                target_services=["api-service"],
                duration_seconds=60,
                blast_radius=20.0,
                hypothesis="System automatically recovers from service crashes",
                success_criteria=["Recovery < 60 seconds", "No data loss"],
                rollback_strategy="restart_service:api-service",
                safety_checks=["system_health > 70%"]
            ),
            ChaosExperiment(
                experiment_id=str(uuid.uuid4()),
                name="Resource Exhaustion Test",
                description="Test system behavior under resource exhaustion",
                failure_type=FailureType.MEMORY_EXHAUSTION,
                severity=FailureSeverity.HIGH,
                target_services=["worker-service"],
                duration_seconds=90,
                blast_radius=25.0,
                hypothesis="System gracefully degrades under memory pressure",
                success_criteria=["No system crash", "Graceful degradation"],
                rollback_strategy="restart_service:worker-service; clear_cache",
                safety_checks=["system_health > 60%"]
            )
        ]
        
        results = await self.chaos_framework.run_chaos_suite(experiments)
        return results
    
    async def _execute_petabyte_testing(self, suite: StressTestSuite) -> List[ProcessingResult]:
        """Execute petabyte-scale data processing testing phase"""
        
        self.logger.info("Executing petabyte processing phase")
        
        # Define processing workloads
        workloads = [
            ProcessingWorkload(
                workload_id=str(uuid.uuid4()),
                name="Batch Analytics Processing",
                processing_type=ProcessingType.BATCH,
                datasets=["enterprise_dataset_1", "enterprise_dataset_2"],
                operations=["aggregate", "transform", "join"],
                parallelism_level=16,
                memory_requirement_gb=64,
                cpu_cores_required=16,
                expected_duration_hours=1.0,
                performance_targets={"min_throughput_gbps": 8.0, "min_success_rate": 0.95}
            ),
            ProcessingWorkload(
                workload_id=str(uuid.uuid4()),
                name="Real-time Stream Processing",
                processing_type=ProcessingType.STREAMING,
                datasets=["stream_dataset_1"],
                operations=["filter", "enrich", "aggregate"],
                parallelism_level=8,
                memory_requirement_gb=32,
                cpu_cores_required=8,
                expected_duration_hours=0.5,
                performance_targets={"max_latency_ms": 500.0, "min_success_rate": 0.99}
            ),
            ProcessingWorkload(
                workload_id=str(uuid.uuid4()),
                name="Hybrid Processing Pipeline",
                processing_type=ProcessingType.HYBRID,
                datasets=["hybrid_dataset_1", "hybrid_dataset_2"],
                operations=["validate", "transform", "store"],
                parallelism_level=12,
                memory_requirement_gb=48,
                cpu_cores_required=12,
                expected_duration_hours=0.75,
                performance_targets={"min_throughput_gbps": 6.0, "min_success_rate": 0.97}
            )
        ]
        
        # Set target performance based on suite requirements
        target_performance = {
            "min_throughput_gbps": suite.performance_targets.get("min_throughput_gbps", 5.0),
            "max_latency_ms": suite.performance_targets.get("max_latency_ms", 1000.0),
            "min_success_rate": suite.performance_targets.get("min_success_rate", 0.95)
        }
        
        results = await self.data_processor.validate_petabyte_processing(workloads, target_performance)
        return results
    
    async def _execute_isolation_testing(self, suite: StressTestSuite) -> List[IsolationResult]:
        """Execute multi-tenant isolation testing phase"""
        
        self.logger.info("Executing multi-tenant isolation phase")
        
        # Generate test tenants
        tenants = self._generate_test_tenants(suite.tenant_count)
        
        # Define isolation tests
        tests = [
            IsolationTest(
                test_id=str(uuid.uuid4()),
                name="Resource Exhaustion Isolation",
                description="Test resource isolation during exhaustion attacks",
                attack_type=AttackType.RESOURCE_EXHAUSTION,
                isolation_types=[IsolationType.COMPUTE, IsolationType.MEMORY],
                attacker_tenant=tenants[0].tenant_id,
                target_tenants=[t.tenant_id for t in tenants[1:3]],
                duration_seconds=90,
                intensity_level=8,
                expected_isolation=True,
                success_criteria=["No resource leakage", "SLA maintained"]
            ),
            IsolationTest(
                test_id=str(uuid.uuid4()),
                name="Data Access Isolation",
                description="Test data isolation between tenants",
                attack_type=AttackType.DATA_ACCESS_ATTEMPT,
                isolation_types=[IsolationType.DATABASE, IsolationType.STORAGE],
                attacker_tenant=tenants[1].tenant_id,
                target_tenants=[tenants[0].tenant_id],
                duration_seconds=60,
                intensity_level=6,
                expected_isolation=True,
                success_criteria=["No unauthorized access", "Encryption intact"]
            ),
            IsolationTest(
                test_id=str(uuid.uuid4()),
                name="Noisy Neighbor Impact",
                description="Test performance isolation during noisy neighbor",
                attack_type=AttackType.NOISY_NEIGHBOR,
                isolation_types=[IsolationType.COMPUTE, IsolationType.NETWORK],
                attacker_tenant=tenants[2].tenant_id if len(tenants) > 2 else tenants[0].tenant_id,
                target_tenants=[t.tenant_id for t in tenants[:2]],
                duration_seconds=120,
                intensity_level=9,
                expected_isolation=True,
                success_criteria=["Performance maintained", "Resource quotas enforced"]
            )
        ]
        
        results = await self.isolation_tester.run_isolation_test_suite(tests, tenants)
        return results
    
    def _generate_test_tenants(self, tenant_count: int) -> List[TenantConfig]:
        """Generate test tenant configurations"""
        
        tenants = []
        tiers = [TenantTier.FREE, TenantTier.BASIC, TenantTier.PREMIUM, TenantTier.ENTERPRISE]
        
        for i in range(tenant_count):
            tier = tiers[i % len(tiers)]
            
            # Configure resources based on tier
            if tier == TenantTier.ENTERPRISE:
                cpu_cores, memory_gb, storage_gb = 16, 64, 1000
                api_calls, connections = 10000, 1000
            elif tier == TenantTier.PREMIUM:
                cpu_cores, memory_gb, storage_gb = 8, 32, 500
                api_calls, connections = 5000, 500
            elif tier == TenantTier.BASIC:
                cpu_cores, memory_gb, storage_gb = 4, 16, 100
                api_calls, connections = 1000, 100
            else:  # FREE
                cpu_cores, memory_gb, storage_gb = 2, 8, 50
                api_calls, connections = 500, 50
            
            tenant = TenantConfig(
                tenant_id=f"test_tenant_{i:03d}",
                name=f"Test Tenant {i+1}",
                tier=tier,
                max_cpu_cores=cpu_cores,
                max_memory_gb=memory_gb,
                max_storage_gb=storage_gb,
                max_api_calls_per_minute=api_calls,
                max_concurrent_connections=connections,
                allowed_regions=["us-east-1", "us-west-2"],
                encryption_key=f"test_key_{i:03d}",
                isolation_requirements=[IsolationType.COMPUTE, IsolationType.STORAGE, IsolationType.DATABASE]
            )
            
            tenants.append(tenant)
        
        return tenants
    
    def _calculate_final_scores(self, results: StressTestResults) -> StressTestResults:
        """Calculate final performance and resilience scores"""
        
        scores = []
        
        # Load testing score
        if results.load_test_results:
            load_score = min(100, (results.load_test_results.successful_requests / 
                                 max(results.load_test_results.total_requests, 1)) * 100)
            results.performance_score = load_score
            scores.append(load_score)
        
        # Compliance score
        if results.compliance_results:
            compliance_scores = [r.compliance_score for r in results.compliance_results]
            results.compliance_score = sum(compliance_scores) / len(compliance_scores)
            scores.append(results.compliance_score)
        
        # Chaos engineering score (resilience)
        if results.chaos_results:
            chaos_scores = []
            for result in results.chaos_results:
                # Score based on hypothesis validation and recovery
                score = 0
                if result.hypothesis_validated:
                    score += 50
                if result.system_recovered:
                    score += 30
                if result.recovery_time_seconds < 180:  # Less than 3 minutes
                    score += 20
                chaos_scores.append(score)
            
            results.resilience_score = sum(chaos_scores) / len(chaos_scores)
            scores.append(results.resilience_score)
        
        # Processing performance score
        if results.processing_results:
            processing_scores = [r.success_rate * 100 for r in results.processing_results]
            processing_score = sum(processing_scores) / len(processing_scores)
            scores.append(processing_score)
        
        # Isolation score
        if results.isolation_results:
            isolation_scores = [r.isolation_effectiveness for r in results.isolation_results]
            results.isolation_score = sum(isolation_scores) / len(isolation_scores)
            scores.append(results.isolation_score)
        
        # Overall success rate
        results.overall_success_rate = sum(scores) / len(scores) if scores else 0
        
        return results
    
    def _generate_recommendations(self, results: StressTestResults, suite: StressTestSuite) -> StressTestResults:
        """Generate recommendations based on test results"""
        
        recommendations = []
        critical_issues = []
        
        # Load testing recommendations
        if results.load_test_results:
            if results.load_test_results.error_rate_percent > 5:
                critical_issues.append(f"High error rate: {results.load_test_results.error_rate_percent:.2f}%")
                recommendations.append("Investigate and fix high error rate issues")
            
            if results.load_test_results.p95_response_time_ms > 2000:
                recommendations.append("Optimize response times - 95th percentile exceeds 2 seconds")
            
            if results.load_test_results.bottlenecks_identified:
                recommendations.extend([f"Address bottleneck: {b}" for b in results.load_test_results.bottlenecks_identified])
        
        # Compliance recommendations
        if results.compliance_results:
            avg_compliance = sum(r.compliance_score for r in results.compliance_results) / len(results.compliance_results)
            if avg_compliance < 90:
                critical_issues.append(f"Low compliance score: {avg_compliance:.1f}%")
                recommendations.append("Address compliance gaps to achieve >90% compliance score")
        
        # Chaos engineering recommendations
        if results.chaos_results:
            failed_experiments = [r for r in results.chaos_results if not r.system_recovered]
            if failed_experiments:
                critical_issues.append(f"{len(failed_experiments)} chaos experiments failed recovery")
                recommendations.append("Improve system resilience and recovery mechanisms")
            
            slow_recovery = [r for r in results.chaos_results if r.recovery_time_seconds > 300]
            if slow_recovery:
                recommendations.append("Optimize recovery time - some experiments took >5 minutes to recover")
        
        # Processing recommendations
        if results.processing_results:
            low_performance = [r for r in results.processing_results if r.success_rate < 0.95]
            if low_performance:
                recommendations.append("Improve data processing reliability - some workloads below 95% success rate")
            
            bottlenecked_workloads = [r for r in results.processing_results if r.bottlenecks_identified]
            if bottlenecked_workloads:
                recommendations.append("Address data processing bottlenecks for better performance")
        
        # Isolation recommendations
        if results.isolation_results:
            isolation_failures = [r for r in results.isolation_results if not r.isolation_maintained]
            if isolation_failures:
                critical_issues.append(f"{len(isolation_failures)} isolation tests failed")
                recommendations.append("Strengthen tenant isolation mechanisms immediately")
            
            violations = sum(len(r.violations_detected) for r in results.isolation_results)
            if violations > 0:
                critical_issues.append(f"{violations} security violations detected")
                recommendations.append("Address all security violations before production deployment")
        
        # Overall recommendations
        if results.overall_success_rate < 80:
            critical_issues.append(f"Overall success rate too low: {results.overall_success_rate:.1f}%")
            recommendations.append("System not ready for enterprise deployment - address critical issues")
        elif results.overall_success_rate < 90:
            recommendations.append("Good progress but improvements needed before enterprise deployment")
        else:
            recommendations.append("System demonstrates enterprise readiness")
        
        results.recommendations = recommendations
        results.critical_issues = critical_issues
        
        return results
    
    def generate_test_report(self, results: StressTestResults) -> str:
        """Generate comprehensive test report"""
        
        report = f"""
# Enterprise Stress Test Report

## Test Suite: {results.suite_id}
- **Duration**: {results.duration_seconds:.2f} seconds ({results.duration_seconds/3600:.2f} hours)
- **Phases Executed**: {len(results.phases_executed)}
- **Overall Success Rate**: {results.overall_success_rate:.1f}%

## Performance Metrics
- **Performance Score**: {results.performance_score:.1f}%
- **Resilience Score**: {results.resilience_score:.1f}%
- **Compliance Score**: {results.compliance_score:.1f}%
- **Isolation Score**: {results.isolation_score:.1f}%

## Load Testing Results
"""
        
        if results.load_test_results:
            report += f"""
- **Total Requests**: {results.load_test_results.total_requests:,}
- **Success Rate**: {(results.load_test_results.successful_requests/results.load_test_results.total_requests)*100:.2f}%
- **Average Response Time**: {results.load_test_results.average_response_time_ms:.2f}ms
- **95th Percentile**: {results.load_test_results.p95_response_time_ms:.2f}ms
- **Throughput**: {results.load_test_results.throughput_rps:.2f} RPS
- **Concurrent Users Achieved**: {results.load_test_results.concurrent_users_achieved:,}
"""
        
        report += f"""
## Compliance Audit Results
- **Audits Conducted**: {len(results.compliance_results)}
"""
        
        for audit in results.compliance_results:
            report += f"""
- **{audit.framework.value}**: {audit.compliance_score:.1f}% compliance
  - Violations: {len(audit.violations_found)}
  - Inspector Satisfaction: {audit.inspector_satisfaction:.1f}%
"""
        
        report += f"""
## Chaos Engineering Results
- **Experiments Conducted**: {len(results.chaos_results)}
"""
        
        for chaos in results.chaos_results:
            report += f"""
- **{chaos.experiment_id}**: 
  - System Recovered: {chaos.system_recovered}
  - Recovery Time: {chaos.recovery_time_seconds:.2f}s
  - Hypothesis Validated: {chaos.hypothesis_validated}
"""
        
        report += f"""
## Data Processing Results
- **Workloads Executed**: {len(results.processing_results)}
"""
        
        for processing in results.processing_results:
            report += f"""
- **{processing.workload_id}**:
  - Data Processed: {processing.data_processed_tb:.3f} TB
  - Throughput: {processing.throughput_gbps:.2f} GB/s
  - Success Rate: {processing.success_rate:.2%}
  - Records Processed: {processing.records_processed:,}
"""
        
        report += f"""
## Multi-Tenant Isolation Results
- **Isolation Tests**: {len(results.isolation_results)}
"""
        
        for isolation in results.isolation_results:
            report += f"""
- **{isolation.test_id}**:
  - Isolation Maintained: {isolation.isolation_maintained}
  - Security Boundaries Intact: {isolation.security_boundaries_intact}
  - Violations Detected: {len(isolation.violations_detected)}
  - Effectiveness: {isolation.isolation_effectiveness:.1f}%
"""
        
        if results.critical_issues:
            report += f"""
## Critical Issues
"""
            for issue in results.critical_issues:
                report += f"- {issue}\n"
        
        if results.recommendations:
            report += f"""
## Recommendations
"""
            for rec in results.recommendations:
                report += f"- {rec}\n"
        
        report += f"""
## Conclusion
The enterprise stress testing suite has been completed with an overall success rate of {results.overall_success_rate:.1f}%.
"""
        
        if results.overall_success_rate >= 90:
            report += "The system demonstrates excellent enterprise readiness."
        elif results.overall_success_rate >= 80:
            report += "The system shows good enterprise potential with some improvements needed."
        else:
            report += "The system requires significant improvements before enterprise deployment."
        
        return report


# Example usage and testing
if __name__ == "__main__":
    async def run_enterprise_stress_test_demo():
        orchestrator = EnterpriseStressTestOrchestrator()
        
        # Define comprehensive stress test suite
        test_suite = StressTestSuite(
            suite_id=str(uuid.uuid4()),
            name="Enterprise Readiness Validation",
            description="Comprehensive stress testing for enterprise deployment readiness",
            phases=[
                TestPhase.LOAD_TESTING,
                TestPhase.COMPLIANCE_AUDIT,
                TestPhase.CHAOS_ENGINEERING,
                TestPhase.PETABYTE_PROCESSING,
                TestPhase.MULTI_TENANT_ISOLATION
            ],
            target_system="localhost:8000",
            duration_hours=2.0,
            concurrent_users=50000,  # Reduced for demo
            data_volume_tb=1.0,      # Reduced for demo
            compliance_frameworks=[
                ComplianceFramework.SOC2_TYPE_II,
                ComplianceFramework.GDPR,
                ComplianceFramework.ISO_27001
            ],
            chaos_scenarios=["network_partition", "service_crash", "resource_exhaustion"],
            tenant_count=5,
            performance_targets={
                "min_throughput_gbps": 5.0,
                "max_latency_ms": 1000.0,
                "min_success_rate": 0.95,
                "min_compliance_score": 90.0
            }
        )
        
        # Execute stress test suite
        results = await orchestrator.execute_stress_test_suite(test_suite)
        
        # Generate and display report
        report = orchestrator.generate_test_report(results)
        print(report)
        
        # Save results to file
        with open(f"stress_test_results_{results.suite_id}.json", "w") as f:
            # Convert results to dict for JSON serialization
            results_dict = asdict(results)
            # Handle datetime serialization
            results_dict["start_time"] = results.start_time.isoformat()
            results_dict["end_time"] = results.end_time.isoformat()
            json.dump(results_dict, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: stress_test_results_{results.suite_id}.json")
    
    # Run the demo
    asyncio.run(run_enterprise_stress_test_demo())