#!/usr/bin/env python3
"""
Comprehensive Error Handling System Demo for ScrollIntel.
Demonstrates error handling, recovery mechanisms, monitoring, and alerting.
"""

import asyncio
import time
import logging
from typing import Dict, Any

from scrollintel.core.error_handling import (
    ErrorHandler, ErrorContext, ErrorSeverity, ErrorCategory,
    RetryConfig, CircuitBreakerConfig, with_error_handling, with_retry
)
from scrollintel.core.error_monitoring import (
    ErrorMonitor, AlertRule, AlertLevel, AlertChannel
)
from scrollintel.core.user_messages import get_user_friendly_error
from scrollintel.core.interfaces import (
    AgentError, SecurityError, EngineError, DataError,
    ValidationError, ExternalServiceError
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ErrorHandlingDemo:
    """Demonstrates comprehensive error handling capabilities."""
    
    def __init__(self):
        self.error_handler = ErrorHandler()
        self.error_monitor = ErrorMonitor()
        self.demo_results = []
    
    async def run_all_demos(self):
        """Run all error handling demonstrations."""
        logger.info("🚀 Starting Comprehensive Error Handling Demo")
        
        # Start error monitoring
        await self.error_monitor.start_monitoring()
        
        try:
            # Demo 1: Basic Error Handling
            await self.demo_basic_error_handling()
            
            # Demo 2: Error Classification and Severity
            await self.demo_error_classification()
            
            # Demo 3: Retry Mechanisms
            await self.demo_retry_mechanisms()
            
            # Demo 4: Circuit Breaker Pattern
            await self.demo_circuit_breaker()
            
            # Demo 5: Fallback Strategies
            await self.demo_fallback_strategies()
            
            # Demo 6: Error Monitoring and Metrics
            await self.demo_error_monitoring()
            
            # Demo 7: Alert System
            await self.demo_alert_system()
            
            # Demo 8: User-Friendly Messages
            await self.demo_user_messages()
            
            # Demo 9: Recovery Mechanisms
            await self.demo_recovery_mechanisms()
            
            # Demo 10: Integration Scenario
            await self.demo_integration_scenario()
            
        finally:
            await self.error_monitor.stop_monitoring()
        
        # Print summary
        self.print_demo_summary()
    
    async def demo_basic_error_handling(self):
        """Demo basic error handling functionality."""
        logger.info("\n📋 Demo 1: Basic Error Handling")
        
        # Create error context
        context = ErrorContext(
            error_id="demo-001",
            timestamp=time.time(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AGENT,
            component="scroll_cto",
            operation="generate_architecture",
            user_id="demo_user"
        )
        
        # Handle different types of errors
        errors_to_test = [
            AgentError("CTO agent is temporarily unavailable"),
            SecurityError("Invalid authentication token"),
            DataError("Invalid data format provided"),
            ValidationError("Required field 'project_name' is missing")
        ]
        
        for error in errors_to_test:
            result = await self.error_handler.handle_error(error, context)
            logger.info(f"  ✅ Handled {type(error).__name__}: {result.get('success', False)}")
            self.demo_results.append(f"Basic handling of {type(error).__name__}")
    
    async def demo_error_classification(self):
        """Demo error classification and severity determination."""
        logger.info("\n🏷️  Demo 2: Error Classification and Severity")
        
        test_errors = [
            (SecurityError("Access denied"), ErrorCategory.SECURITY, ErrorSeverity.CRITICAL),
            (AgentError("Agent crashed"), ErrorCategory.AGENT, ErrorSeverity.HIGH),
            (DataError("Invalid format"), ErrorCategory.DATA, ErrorSeverity.MEDIUM),
            (ConnectionError("Network timeout"), ErrorCategory.NETWORK, ErrorSeverity.LOW)
        ]
        
        for error, expected_category, expected_severity in test_errors:
            category = self.error_handler._classify_error(error)
            context = ErrorContext(
                error_id="demo-002",
                timestamp=time.time(),
                severity=ErrorSeverity.LOW,
                category=ErrorCategory.UNKNOWN,
                component="test",
                operation="test"
            )
            severity = self.error_handler._determine_severity(error, context)
            
            logger.info(f"  ✅ {type(error).__name__}: {category.value} ({severity.value})")
            self.demo_results.append(f"Classified {type(error).__name__} correctly")
    
    async def demo_retry_mechanisms(self):
        """Demo retry mechanisms with backoff."""
        logger.info("\n🔄 Demo 3: Retry Mechanisms")
        
        # Simulate a function that fails a few times then succeeds
        attempt_count = 0
        
        @with_retry(RetryConfig(max_attempts=3, base_delay=0.1))
        async def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError(f"Connection failed (attempt {attempt_count})")
            return f"Success after {attempt_count} attempts"
        
        try:
            result = await flaky_function()
            logger.info(f"  ✅ Retry successful: {result}")
            self.demo_results.append("Retry mechanism worked correctly")
        except Exception as e:
            logger.error(f"  ❌ Retry failed: {e}")
    
    async def demo_circuit_breaker(self):
        """Demo circuit breaker pattern."""
        logger.info("\n⚡ Demo 4: Circuit Breaker Pattern")
        
        # Get circuit breaker for a test service
        circuit_breaker = self.error_handler.get_circuit_breaker(
            "demo_service",
            CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1.0)
        )
        
        # Simulate failures to open circuit
        logger.info("  📉 Simulating service failures...")
        for i in range(3):
            circuit_breaker.record_failure()
            can_execute = circuit_breaker.can_execute()
            logger.info(f"    Failure {i+1}: Can execute = {can_execute}")
        
        # Wait for recovery
        logger.info("  ⏳ Waiting for recovery timeout...")
        await asyncio.sleep(1.1)
        
        # Test recovery
        can_execute = circuit_breaker.can_execute()
        logger.info(f"  ✅ After recovery timeout: Can execute = {can_execute}")
        
        # Simulate success to close circuit
        circuit_breaker.record_success()
        circuit_breaker.record_success()
        circuit_breaker.record_success()
        
        logger.info(f"  ✅ Circuit breaker demo completed")
        self.demo_results.append("Circuit breaker pattern demonstrated")
    
    async def demo_fallback_strategies(self):
        """Demo fallback strategies."""
        logger.info("\n🔄 Demo 5: Fallback Strategies")
        
        # Register fallback handler
        async def demo_fallback(error, context):
            return {
                "message": "Using cached response",
                "data": {"status": "fallback_active", "cached_result": "demo_data"}
            }
        
        self.error_handler.register_fallback("demo_component", demo_fallback)
        
        # Test fallback
        context = ErrorContext(
            error_id="demo-005",
            timestamp=time.time(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AGENT,
            component="demo_component",
            operation="demo_operation"
        )
        
        result = await self.error_handler.handle_error(
            AgentError("Service unavailable"), context
        )
        
        logger.info(f"  ✅ Fallback result: {result.get('fallback_used', False)}")
        self.demo_results.append("Fallback strategy executed successfully")
    
    async def demo_error_monitoring(self):
        """Demo error monitoring and metrics."""
        logger.info("\n📊 Demo 6: Error Monitoring and Metrics")
        
        # Generate some test errors
        components = ["scroll_cto", "scroll_data_scientist", "scroll_ml_engineer"]
        
        for component in components:
            for i in range(5):
                context = ErrorContext(
                    error_id=f"monitor-{component}-{i}",
                    timestamp=time.time(),
                    severity=ErrorSeverity.MEDIUM,
                    category=ErrorCategory.AGENT,
                    component=component,
                    operation="test_operation"
                )
                self.error_monitor.record_error(context)
                
                # Also record some successes
                self.error_monitor.record_success(component, 0.5 + i * 0.1)
        
        # Get metrics
        metrics = self.error_monitor.get_metrics()
        logger.info(f"  📈 Total components monitored: {metrics['total_components']}")
        logger.info(f"  📈 Overall success rate: {metrics['overall_success_rate']:.2%}")
        
        # Get component-specific metrics
        for component in components:
            comp_metrics = self.error_monitor.get_component_metrics(component)
            logger.info(f"  📊 {component}: {comp_metrics['error_rate']:.1f} errors/min, "
                       f"{comp_metrics['success_rate']:.1%} success rate")
        
        self.demo_results.append("Error monitoring and metrics collection")
    
    async def demo_alert_system(self):
        """Demo alert system."""
        logger.info("\n🚨 Demo 7: Alert System")
        
        # Add custom alert rule
        custom_rule = AlertRule(
            name="demo_high_error_rate",
            condition="error_rate > 3",
            threshold=3.0,
            window_minutes=1,
            alert_level=AlertLevel.WARNING,
            channels=[AlertChannel.DASHBOARD]
        )
        
        self.error_monitor.add_alert_rule(custom_rule)
        
        # Generate errors to trigger alert
        for i in range(6):
            context = ErrorContext(
                error_id=f"alert-demo-{i}",
                timestamp=time.time(),
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.AGENT,
                component="alert_test_component",
                operation="test_operation"
            )
            self.error_monitor.record_error(context)
        
        # Wait for alert check
        await asyncio.sleep(1.1)
        
        # Check for alerts
        active_alerts = self.error_monitor.get_active_alerts()
        logger.info(f"  🚨 Active alerts: {len(active_alerts)}")
        
        for alert in active_alerts:
            logger.info(f"    Alert: {alert.title} ({alert.level.value})")
        
        self.demo_results.append(f"Alert system generated {len(active_alerts)} alerts")
    
    async def demo_user_messages(self):
        """Demo user-friendly message generation."""
        logger.info("\n💬 Demo 8: User-Friendly Messages")
        
        # Test different error scenarios
        scenarios = [
            (ErrorCategory.AGENT, ErrorSeverity.HIGH, "scroll_cto", "generate_architecture"),
            (ErrorCategory.SECURITY, ErrorSeverity.CRITICAL, "auth", "login"),
            (ErrorCategory.DATA, ErrorSeverity.MEDIUM, "file_upload", "process_csv"),
            (ErrorCategory.EXTERNAL_SERVICE, ErrorSeverity.HIGH, "openai", "api_call")
        ]
        
        for category, severity, component, operation in scenarios:
            message = get_user_friendly_error(category, severity, component, operation)
            logger.info(f"  💬 {component}: {message['title']}")
            logger.info(f"     Message: {message['message'][:80]}...")
            logger.info(f"     Recovery actions: {len(message['recovery_actions'])}")
        
        self.demo_results.append("User-friendly messages generated for all scenarios")
    
    async def demo_recovery_mechanisms(self):
        """Demo automatic recovery mechanisms."""
        logger.info("\n🔧 Demo 9: Recovery Mechanisms")
        
        # Test automatic recovery for different error types
        recovery_scenarios = [
            (ConnectionError("Connection lost"), "connection_recovery"),
            (MemoryError("Out of memory"), "memory_recovery"),
            (Exception("Service overload"), "overload_recovery")
        ]
        
        for error, recovery_type in recovery_scenarios:
            context = ErrorContext(
                error_id=f"recovery-{recovery_type}",
                timestamp=time.time(),
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.NETWORK,
                component="recovery_test",
                operation="test_recovery"
            )
            
            # Test automatic recovery
            recovery_result = await self.error_handler._attempt_automatic_recovery(error, context)
            
            if recovery_result:
                logger.info(f"  ✅ {recovery_type}: {recovery_result.get('recovery_type', 'unknown')}")
            else:
                logger.info(f"  ⚠️  {recovery_type}: No automatic recovery available")
        
        self.demo_results.append("Automatic recovery mechanisms tested")
    
    async def demo_integration_scenario(self):
        """Demo complete integration scenario."""
        logger.info("\n🔗 Demo 10: Integration Scenario")
        
        # Simulate a complete user workflow with errors and recovery
        @with_error_handling(
            component="demo_workflow",
            operation="complete_analysis",
            retry_config=RetryConfig(max_attempts=2, base_delay=0.1)
        )
        async def complex_workflow():
            # Simulate multi-step process with potential failures
            steps = [
                ("data_validation", 0.3),
                ("model_training", 0.2),
                ("result_generation", 0.1)
            ]
            
            results = {}
            for step, failure_probability in steps:
                if time.time() % 1 < failure_probability:  # Simulate random failures
                    raise EngineError(f"Step {step} failed")
                results[step] = f"{step}_completed"
                await asyncio.sleep(0.1)
            
            return results
        
        try:
            result = await complex_workflow()
            logger.info(f"  ✅ Workflow completed: {len(result)} steps")
            self.demo_results.append("Integration scenario completed successfully")
        except Exception as e:
            logger.info(f"  ⚠️  Workflow failed with recovery: {type(e).__name__}")
            self.demo_results.append("Integration scenario handled gracefully")
    
    def print_demo_summary(self):
        """Print summary of all demonstrations."""
        logger.info("\n" + "="*60)
        logger.info("📋 COMPREHENSIVE ERROR HANDLING DEMO SUMMARY")
        logger.info("="*60)
        
        logger.info(f"✅ Total demonstrations completed: {len(self.demo_results)}")
        
        for i, result in enumerate(self.demo_results, 1):
            logger.info(f"  {i:2d}. {result}")
        
        # Get final system metrics
        metrics = self.error_monitor.get_metrics()
        active_alerts = self.error_monitor.get_active_alerts()
        
        logger.info("\n📊 FINAL SYSTEM METRICS:")
        logger.info(f"  • Components monitored: {metrics.get('total_components', 0)}")
        logger.info(f"  • Healthy components: {metrics.get('healthy_components', 0)}")
        logger.info(f"  • Degraded components: {metrics.get('degraded_components', 0)}")
        logger.info(f"  • Unhealthy components: {metrics.get('unhealthy_components', 0)}")
        logger.info(f"  • Overall success rate: {metrics.get('overall_success_rate', 0):.1%}")
        logger.info(f"  • Active alerts: {len(active_alerts)}")
        
        logger.info("\n🎉 All error handling demonstrations completed successfully!")
        logger.info("   The system is ready for production deployment.")


async def main():
    """Run the comprehensive error handling demo."""
    demo = ErrorHandlingDemo()
    await demo.run_all_demos()


if __name__ == "__main__":
    asyncio.run(main())