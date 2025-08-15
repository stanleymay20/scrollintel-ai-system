"""
Upgrade Planner

Creates systematic upgrade plans for components based on assessment results.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from pathlib import Path

from .production_upgrade_framework import (
    ComponentAssessment, UpgradePlan, UpgradeStep, ValidationCriterion,
    ProductionStandards, UpgradeCategory, Priority, RiskLevel, Severity
)

logger = logging.getLogger(__name__)


class UpgradePlanner:
    """Creates systematic upgrade plans for components"""
    
    def __init__(self):
        self.step_templates = self._initialize_step_templates()
        self.dependency_graph = self._initialize_dependency_graph()
    
    async def create_plan(self, assessment: ComponentAssessment, 
                         standards: ProductionStandards) -> UpgradePlan:
        """
        Create a comprehensive upgrade plan based on assessment results
        
        Args:
            assessment: Component assessment results
            standards: Target production standards
            
        Returns:
            UpgradePlan with ordered steps and dependencies
        """
        try:
            logger.info(f"Creating upgrade plan for {assessment.component_path}")
            
            # Analyze gaps between current state and standards
            gaps = await self._analyze_gaps(assessment, standards)
            
            # Generate upgrade steps based on gaps
            steps = await self._generate_upgrade_steps(gaps, assessment)
            
            # Order steps by dependencies and priority
            ordered_steps = await self._order_steps(steps)
            
            # Calculate effort and risk
            total_effort = sum((step.estimated_duration for step in ordered_steps), timedelta())
            risk_level = await self._assess_risk_level(assessment, ordered_steps)
            
            # Identify dependencies
            dependencies = await self._identify_dependencies(assessment.component_path)
            
            plan = UpgradePlan(
                component_path=assessment.component_path,
                current_assessment=assessment,
                target_state=standards,
                upgrade_steps=ordered_steps,
                estimated_effort=total_effort,
                dependencies=dependencies,
                risk_level=risk_level
            )
            
            logger.info(f"Upgrade plan created with {len(ordered_steps)} steps, "
                       f"estimated effort: {total_effort}, risk level: {risk_level.value}")
            
            return plan
            
        except Exception as e:
            logger.error(f"Failed to create upgrade plan for {assessment.component_path}: {str(e)}")
            raise
    
    async def _analyze_gaps(self, assessment: ComponentAssessment, 
                           standards: ProductionStandards) -> Dict[str, float]:
        """Analyze gaps between current state and production standards"""
        gaps = {}
        
        # Code quality gap
        if assessment.code_quality_score < standards.min_code_quality_score:
            gaps['code_quality'] = standards.min_code_quality_score - assessment.code_quality_score
        
        # Security gap
        if assessment.security_score < standards.min_security_score:
            gaps['security'] = standards.min_security_score - assessment.security_score
        
        # Performance gap
        if assessment.performance_score < standards.min_performance_score:
            gaps['performance'] = standards.min_performance_score - assessment.performance_score
        
        # Test coverage gap
        if assessment.test_coverage < standards.min_test_coverage:
            gaps['test_coverage'] = standards.min_test_coverage - assessment.test_coverage
        
        # Documentation gap
        if assessment.documentation_score < standards.min_documentation_score:
            gaps['documentation'] = standards.min_documentation_score - assessment.documentation_score
        
        # Check for missing required features
        if standards.required_error_handling:
            gaps['error_handling'] = await self._check_error_handling_gap(assessment.component_path)
        
        if standards.required_logging:
            gaps['logging'] = await self._check_logging_gap(assessment.component_path)
        
        if standards.required_monitoring:
            gaps['monitoring'] = await self._check_monitoring_gap(assessment.component_path)
        
        if standards.required_security_controls:
            gaps['security_controls'] = await self._check_security_controls_gap(assessment.component_path)
        
        if standards.required_input_validation:
            gaps['input_validation'] = await self._check_input_validation_gap(assessment.component_path)
        
        return gaps
    
    async def _check_error_handling_gap(self, component_path: str) -> float:
        """Check if component has proper error handling"""
        try:
            with open(component_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for try-except blocks
            try_count = content.count('try:')
            except_count = content.count('except')
            
            # Look for function definitions
            func_count = content.count('def ') + content.count('async def ')
            
            if func_count == 0:
                return 0.0
            
            # Calculate error handling coverage
            coverage = min(try_count / func_count, 1.0)
            return max(0.0, 1.0 - coverage)
            
        except Exception as e:
            logger.warning(f"Error handling gap analysis failed for {component_path}: {str(e)}")
            return 1.0
    
    async def _check_logging_gap(self, component_path: str) -> float:
        """Check if component has proper logging"""
        try:
            with open(component_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for logging imports and usage
            has_logging_import = 'import logging' in content or 'from logging' in content
            has_logger = 'logger' in content or 'log.' in content
            
            if has_logging_import and has_logger:
                return 0.0
            elif has_logging_import or has_logger:
                return 0.5
            else:
                return 1.0
                
        except Exception as e:
            logger.warning(f"Logging gap analysis failed for {component_path}: {str(e)}")
            return 1.0
    
    async def _check_monitoring_gap(self, component_path: str) -> float:
        """Check if component has monitoring capabilities"""
        try:
            with open(component_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for monitoring-related patterns
            monitoring_patterns = [
                'metrics', 'prometheus', 'health_check', 'status',
                'monitor', 'telemetry', 'observability'
            ]
            
            found_patterns = sum(1 for pattern in monitoring_patterns if pattern in content.lower())
            
            if found_patterns >= 2:
                return 0.0
            elif found_patterns == 1:
                return 0.5
            else:
                return 1.0
                
        except Exception as e:
            logger.warning(f"Monitoring gap analysis failed for {component_path}: {str(e)}")
            return 1.0
    
    async def _check_security_controls_gap(self, component_path: str) -> float:
        """Check if component has security controls"""
        try:
            with open(component_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for security-related patterns
            security_patterns = [
                'authenticate', 'authorize', 'validate', 'sanitize',
                'encrypt', 'hash', 'token', 'permission'
            ]
            
            found_patterns = sum(1 for pattern in security_patterns if pattern in content.lower())
            
            if found_patterns >= 3:
                return 0.0
            elif found_patterns >= 1:
                return 0.5
            else:
                return 1.0
                
        except Exception as e:
            logger.warning(f"Security controls gap analysis failed for {component_path}: {str(e)}")
            return 1.0
    
    async def _check_input_validation_gap(self, component_path: str) -> float:
        """Check if component has input validation"""
        try:
            with open(component_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for validation patterns
            validation_patterns = [
                'validate', 'check', 'verify', 'isinstance',
                'assert', 'raise ValueError', 'raise TypeError'
            ]
            
            found_patterns = sum(1 for pattern in validation_patterns if pattern in content.lower())
            
            if found_patterns >= 2:
                return 0.0
            elif found_patterns == 1:
                return 0.5
            else:
                return 1.0
                
        except Exception as e:
            logger.warning(f"Input validation gap analysis failed for {component_path}: {str(e)}")
            return 1.0
    
    async def _generate_upgrade_steps(self, gaps: Dict[str, float], 
                                    assessment: ComponentAssessment) -> List[UpgradeStep]:
        """Generate specific upgrade steps based on identified gaps"""
        steps = []
        step_counter = 0
        
        # Critical security issues first
        critical_security_issues = [
            issue for issue in assessment.identified_issues 
            if issue.severity == Severity.CRITICAL and issue.category == UpgradeCategory.SECURITY
        ]
        
        if critical_security_issues:
            step_counter += 1
            steps.append(UpgradeStep(
                step_id=f"step_{step_counter:03d}",
                description="Fix critical security vulnerabilities",
                category=UpgradeCategory.SECURITY,
                priority=Priority.CRITICAL,
                estimated_duration=timedelta(hours=4),
                prerequisites=[],
                validation_criteria=[
                    ValidationCriterion(
                        name="security_scan_clean",
                        description="Security scan shows no critical vulnerabilities",
                        validation_method="security_scan",
                        expected_result=0
                    )
                ]
            ))
        
        # Error handling implementation
        if gaps.get('error_handling', 0) > 0.5:
            step_counter += 1
            steps.append(UpgradeStep(
                step_id=f"step_{step_counter:03d}",
                description="Implement comprehensive error handling",
                category=UpgradeCategory.ERROR_HANDLING,
                priority=Priority.HIGH,
                estimated_duration=timedelta(hours=6),
                prerequisites=[],
                validation_criteria=[
                    ValidationCriterion(
                        name="error_handling_coverage",
                        description="All functions have proper error handling",
                        validation_method="code_analysis",
                        expected_result=True
                    )
                ]
            ))
        
        # Logging implementation
        if gaps.get('logging', 0) > 0.5:
            step_counter += 1
            steps.append(UpgradeStep(
                step_id=f"step_{step_counter:03d}",
                description="Add structured logging",
                category=UpgradeCategory.MONITORING,
                priority=Priority.HIGH,
                estimated_duration=timedelta(hours=3),
                prerequisites=[],
                validation_criteria=[
                    ValidationCriterion(
                        name="logging_implemented",
                        description="Component uses structured logging",
                        validation_method="code_analysis",
                        expected_result=True
                    )
                ]
            ))
        
        # Input validation
        if gaps.get('input_validation', 0) > 0.5:
            step_counter += 1
            steps.append(UpgradeStep(
                step_id=f"step_{step_counter:03d}",
                description="Implement input validation",
                category=UpgradeCategory.SECURITY,
                priority=Priority.HIGH,
                estimated_duration=timedelta(hours=4),
                prerequisites=[],
                validation_criteria=[
                    ValidationCriterion(
                        name="input_validation_coverage",
                        description="All inputs are properly validated",
                        validation_method="code_analysis",
                        expected_result=True
                    )
                ]
            ))
        
        # Code quality improvements
        if gaps.get('code_quality', 0) > 1.0:
            step_counter += 1
            steps.append(UpgradeStep(
                step_id=f"step_{step_counter:03d}",
                description="Refactor code for quality improvements",
                category=UpgradeCategory.CODE_QUALITY,
                priority=Priority.MEDIUM,
                estimated_duration=timedelta(hours=8),
                prerequisites=[],
                validation_criteria=[
                    ValidationCriterion(
                        name="code_quality_score",
                        description="Code quality score meets minimum threshold",
                        validation_method="quality_analysis",
                        expected_result=8.5
                    )
                ]
            ))
        
        # Performance optimization
        if gaps.get('performance', 0) > 1.0:
            step_counter += 1
            steps.append(UpgradeStep(
                step_id=f"step_{step_counter:03d}",
                description="Optimize performance bottlenecks",
                category=UpgradeCategory.PERFORMANCE,
                priority=Priority.MEDIUM,
                estimated_duration=timedelta(hours=6),
                prerequisites=[],
                validation_criteria=[
                    ValidationCriterion(
                        name="performance_score",
                        description="Performance score meets minimum threshold",
                        validation_method="performance_test",
                        expected_result=8.0
                    )
                ]
            ))
        
        # Test coverage improvement
        if gaps.get('test_coverage', 0) > 10.0:
            step_counter += 1
            steps.append(UpgradeStep(
                step_id=f"step_{step_counter:03d}",
                description="Increase test coverage",
                category=UpgradeCategory.TESTING,
                priority=Priority.MEDIUM,
                estimated_duration=timedelta(hours=10),
                prerequisites=[],
                validation_criteria=[
                    ValidationCriterion(
                        name="test_coverage",
                        description="Test coverage meets minimum threshold",
                        validation_method="coverage_test",
                        expected_result=90.0
                    )
                ]
            ))
        
        # Documentation improvement
        if gaps.get('documentation', 0) > 1.0:
            step_counter += 1
            steps.append(UpgradeStep(
                step_id=f"step_{step_counter:03d}",
                description="Improve documentation coverage",
                category=UpgradeCategory.DOCUMENTATION,
                priority=Priority.LOW,
                estimated_duration=timedelta(hours=4),
                prerequisites=[],
                validation_criteria=[
                    ValidationCriterion(
                        name="documentation_score",
                        description="Documentation score meets minimum threshold",
                        validation_method="documentation_analysis",
                        expected_result=8.0
                    )
                ]
            ))
        
        # Monitoring implementation
        if gaps.get('monitoring', 0) > 0.5:
            step_counter += 1
            steps.append(UpgradeStep(
                step_id=f"step_{step_counter:03d}",
                description="Add monitoring and health checks",
                category=UpgradeCategory.MONITORING,
                priority=Priority.MEDIUM,
                estimated_duration=timedelta(hours=5),
                prerequisites=["step_002", "step_003"],  # After error handling and logging
                validation_criteria=[
                    ValidationCriterion(
                        name="monitoring_implemented",
                        description="Component has monitoring capabilities",
                        validation_method="monitoring_check",
                        expected_result=True
                    )
                ]
            ))
        
        return steps
    
    async def _order_steps(self, steps: List[UpgradeStep]) -> List[UpgradeStep]:
        """Order steps by dependencies and priority"""
        # Create dependency graph
        step_map = {step.step_id: step for step in steps}
        
        # Topological sort with priority consideration
        ordered_steps = []
        remaining_steps = steps.copy()
        
        while remaining_steps:
            # Find steps with no unmet prerequisites
            ready_steps = []
            for step in remaining_steps:
                if all(prereq in [s.step_id for s in ordered_steps] for prereq in step.prerequisites):
                    ready_steps.append(step)
            
            if not ready_steps:
                # If no steps are ready, there might be a circular dependency
                # Add the highest priority step
                ready_steps = [min(remaining_steps, key=lambda s: s.priority.value)]
            
            # Sort ready steps by priority
            priority_order = {Priority.CRITICAL: 0, Priority.HIGH: 1, Priority.MEDIUM: 2, Priority.LOW: 3}
            ready_steps.sort(key=lambda s: priority_order[s.priority])
            
            # Add the highest priority ready step
            next_step = ready_steps[0]
            ordered_steps.append(next_step)
            remaining_steps.remove(next_step)
        
        return ordered_steps
    
    async def _assess_risk_level(self, assessment: ComponentAssessment, 
                               steps: List[UpgradeStep]) -> RiskLevel:
        """Assess the risk level of the upgrade plan"""
        risk_factors = 0
        
        # Critical security issues increase risk
        critical_issues = [i for i in assessment.identified_issues if i.severity == Severity.CRITICAL]
        risk_factors += len(critical_issues) * 2
        
        # Low production readiness score increases risk
        if assessment.production_readiness_score < 5.0:
            risk_factors += 3
        elif assessment.production_readiness_score < 7.0:
            risk_factors += 1
        
        # Large number of steps increases risk
        if len(steps) > 8:
            risk_factors += 2
        elif len(steps) > 5:
            risk_factors += 1
        
        # High effort increases risk
        total_effort = sum((step.estimated_duration for step in steps), timedelta())
        if total_effort > timedelta(hours=40):
            risk_factors += 2
        elif total_effort > timedelta(hours=20):
            risk_factors += 1
        
        # Determine risk level
        if risk_factors >= 6:
            return RiskLevel.CRITICAL
        elif risk_factors >= 4:
            return RiskLevel.HIGH
        elif risk_factors >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    async def _identify_dependencies(self, component_path: str) -> List[str]:
        """Identify external dependencies that might be affected"""
        dependencies = []
        
        try:
            with open(component_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for import statements
            import_lines = [line.strip() for line in content.split('\n') if line.strip().startswith(('import ', 'from '))]
            
            # Extract local dependencies (scrollintel modules)
            for line in import_lines:
                if 'scrollintel' in line:
                    # Extract module name
                    if line.startswith('from scrollintel'):
                        parts = line.split()
                        if len(parts) >= 2:
                            module = parts[1].split('.')[0]
                            if module not in dependencies:
                                dependencies.append(f"scrollintel.{module}")
                    elif line.startswith('import scrollintel'):
                        parts = line.split()
                        if len(parts) >= 2:
                            module = parts[1].split('.')[1] if '.' in parts[1] else parts[1]
                            if module not in dependencies:
                                dependencies.append(f"scrollintel.{module}")
            
            return dependencies
            
        except Exception as e:
            logger.warning(f"Dependency analysis failed for {component_path}: {str(e)}")
            return []
    
    def _initialize_step_templates(self) -> Dict[str, Dict]:
        """Initialize templates for common upgrade steps"""
        return {
            'error_handling': {
                'description': 'Implement comprehensive error handling',
                'category': UpgradeCategory.ERROR_HANDLING,
                'priority': Priority.HIGH,
                'estimated_duration': timedelta(hours=6)
            },
            'logging': {
                'description': 'Add structured logging',
                'category': UpgradeCategory.MONITORING,
                'priority': Priority.HIGH,
                'estimated_duration': timedelta(hours=3)
            },
            'input_validation': {
                'description': 'Implement input validation',
                'category': UpgradeCategory.SECURITY,
                'priority': Priority.HIGH,
                'estimated_duration': timedelta(hours=4)
            },
            'monitoring': {
                'description': 'Add monitoring and health checks',
                'category': UpgradeCategory.MONITORING,
                'priority': Priority.MEDIUM,
                'estimated_duration': timedelta(hours=5)
            }
        }
    
    def _initialize_dependency_graph(self) -> Dict[str, List[str]]:
        """Initialize common dependency relationships"""
        return {
            'monitoring': ['error_handling', 'logging'],
            'performance_optimization': ['code_quality'],
            'security_hardening': ['input_validation'],
            'testing': ['error_handling', 'code_quality']
        }