"""
Validation Engine

Validates upgrade results to ensure components meet production standards.
"""

import asyncio
import ast
import logging
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import re
import os

from .production_upgrade_framework import (
    UpgradeResult, ValidationCriterion, ComponentAssessment, 
    ProductionStandards, UpgradeCategory, Severity
)
from .component_assessment_engine import ComponentAssessmentEngine

logger = logging.getLogger(__name__)


class ValidationEngine:
    """Engine for validating upgrade results against production standards"""
    
    def __init__(self):
        self.assessment_engine = ComponentAssessmentEngine()
        self.validators = {
            'security_scan': self._validate_security_scan,
            'code_analysis': self._validate_code_analysis,
            'quality_analysis': self._validate_quality_analysis,
            'performance_test': self._validate_performance_test,
            'coverage_test': self._validate_coverage_test,
            'documentation_analysis': self._validate_documentation_analysis,
            'monitoring_check': self._validate_monitoring_check,
            'error_handling_check': self._validate_error_handling_check,
            'input_validation_check': self._validate_input_validation_check,
            'integration_test': self._validate_integration_test
        }
    
    async def validate(self, result: UpgradeResult) -> List[ValidationCriterion]:
        """
        Validate upgrade results against all criteria
        
        Args:
            result: The upgrade result to validate
            
        Returns:
            List of validation criteria with results
        """
        try:
            logger.info(f"Starting validation for {result.component_path}")
            
            # Collect all validation criteria from completed steps
            all_criteria = []
            for step in result.completed_steps:
                all_criteria.extend(step.validation_criteria)
            
            # Add standard production validation criteria
            standard_criteria = await self._get_standard_validation_criteria(result.component_path)
            all_criteria.extend(standard_criteria)
            
            # Execute validations in parallel
            validation_tasks = []
            for criterion in all_criteria:
                task = self._execute_validation(result.component_path, criterion)
                validation_tasks.append(task)
            
            validated_criteria = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            # Process results
            final_criteria = []
            for i, criterion_result in enumerate(validated_criteria):
                if isinstance(criterion_result, Exception):
                    logger.error(f"Validation failed for {all_criteria[i].name}: {str(criterion_result)}")
                    all_criteria[i].passed = False
                    all_criteria[i].actual_result = f"Error: {str(criterion_result)}"
                else:
                    all_criteria[i] = criterion_result
                
                final_criteria.append(all_criteria[i])
            
            # Log validation summary
            passed_count = sum(1 for c in final_criteria if c.passed)
            total_count = len(final_criteria)
            
            logger.info(f"Validation completed: {passed_count}/{total_count} criteria passed")
            
            return final_criteria
            
        except Exception as e:
            logger.error(f"Validation failed for {result.component_path}: {str(e)}")
            raise
    
    async def _get_standard_validation_criteria(self, component_path: str) -> List[ValidationCriterion]:
        """Get standard validation criteria for production readiness"""
        return [
            ValidationCriterion(
                name="syntax_check",
                description="Python syntax is valid",
                validation_method="code_analysis",
                expected_result=True
            ),
            ValidationCriterion(
                name="import_check",
                description="All imports are valid",
                validation_method="code_analysis",
                expected_result=True
            ),
            ValidationCriterion(
                name="error_handling_present",
                description="Error handling is implemented",
                validation_method="error_handling_check",
                expected_result=True
            ),
            ValidationCriterion(
                name="logging_present",
                description="Logging is implemented",
                validation_method="code_analysis",
                expected_result=True
            ),
            ValidationCriterion(
                name="basic_security_check",
                description="No obvious security vulnerabilities",
                validation_method="security_scan",
                expected_result=0
            ),
            ValidationCriterion(
                name="documentation_present",
                description="Basic documentation is present",
                validation_method="documentation_analysis",
                expected_result=True
            )
        ]
    
    async def _execute_validation(self, component_path: str, 
                                criterion: ValidationCriterion) -> ValidationCriterion:
        """Execute a single validation criterion"""
        try:
            validator = self.validators.get(criterion.validation_method)
            if not validator:
                logger.warning(f"No validator found for method: {criterion.validation_method}")
                criterion.passed = False
                criterion.actual_result = f"No validator for {criterion.validation_method}"
                return criterion
            
            actual_result = await validator(component_path, criterion)
            criterion.actual_result = actual_result
            
            # Compare actual result with expected result
            criterion.passed = await self._compare_results(
                actual_result, criterion.expected_result, criterion.validation_method
            )
            
            return criterion
            
        except Exception as e:
            logger.error(f"Validation execution failed for {criterion.name}: {str(e)}")
            criterion.passed = False
            criterion.actual_result = f"Error: {str(e)}"
            return criterion
    
    async def _compare_results(self, actual: Any, expected: Any, method: str) -> bool:
        """Compare actual and expected results"""
        try:
            if method in ['security_scan']:
                # For security scans, actual should be <= expected (fewer vulnerabilities is better)
                return actual <= expected
            elif method in ['coverage_test', 'quality_analysis', 'performance_test', 'documentation_analysis']:
                # For scores, actual should be >= expected
                return actual >= expected
            else:
                # For boolean checks, direct comparison
                return actual == expected
                
        except Exception as e:
            logger.error(f"Result comparison failed: {str(e)}")
            return False
    
    async def _validate_security_scan(self, component_path: str, 
                                    criterion: ValidationCriterion) -> int:
        """Validate security scan results"""
        try:
            # Use the security scanner from assessment engine
            security_results = await self.assessment_engine.security_scanner.scan(component_path)
            
            # Count critical and high severity vulnerabilities
            critical_count = security_results.get('critical_count', 0)
            high_count = security_results.get('high_count', 0)
            
            # Return total count of serious vulnerabilities
            return critical_count + high_count
            
        except Exception as e:
            logger.error(f"Security scan validation failed: {str(e)}")
            return 999  # High number to indicate failure
    
    async def _validate_code_analysis(self, component_path: str, 
                                    criterion: ValidationCriterion) -> bool:
        """Validate code analysis results"""
        try:
            if not os.path.exists(component_path):
                return False
            
            # Check syntax
            if criterion.name == "syntax_check":
                return await self._check_syntax(component_path)
            
            # Check imports
            elif criterion.name == "import_check":
                return await self._check_imports(component_path)
            
            # Check logging
            elif criterion.name == "logging_present":
                return await self._check_logging_present(component_path)
            
            # Generic code analysis
            else:
                code_quality = await self.assessment_engine.code_analyzer.analyze(component_path)
                return code_quality.get('overall_score', 0) >= 7.0
                
        except Exception as e:
            logger.error(f"Code analysis validation failed: {str(e)}")
            return False
    
    async def _check_syntax(self, component_path: str) -> bool:
        """Check if Python syntax is valid"""
        try:
            with open(component_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to parse the AST
            ast.parse(content)
            return True
            
        except SyntaxError as e:
            logger.error(f"Syntax error in {component_path}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Syntax check failed: {str(e)}")
            return False
    
    async def _check_imports(self, component_path: str) -> bool:
        """Check if all imports are valid"""
        try:
            # Run a basic import check using py_compile
            result = subprocess.run(
                ['python', '-m', 'py_compile', component_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            logger.error("Import check timed out")
            return False
        except Exception as e:
            logger.error(f"Import check failed: {str(e)}")
            return False
    
    async def _check_logging_present(self, component_path: str) -> bool:
        """Check if logging is implemented"""
        try:
            with open(component_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for logging import
            has_logging_import = ('import logging' in content or 
                                'from logging' in content)
            
            # Check for logger usage
            has_logger_usage = ('logger.' in content or 
                              'logging.' in content)
            
            return has_logging_import and has_logger_usage
            
        except Exception as e:
            logger.error(f"Logging check failed: {str(e)}")
            return False
    
    async def _validate_quality_analysis(self, component_path: str, 
                                       criterion: ValidationCriterion) -> float:
        """Validate code quality analysis"""
        try:
            code_quality = await self.assessment_engine.code_analyzer.analyze(component_path)
            return code_quality.get('overall_score', 0.0)
            
        except Exception as e:
            logger.error(f"Quality analysis validation failed: {str(e)}")
            return 0.0
    
    async def _validate_performance_test(self, component_path: str, 
                                       criterion: ValidationCriterion) -> float:
        """Validate performance test results"""
        try:
            performance_results = await self.assessment_engine.performance_profiler.profile(component_path)
            return performance_results.get('overall_score', 0.0)
            
        except Exception as e:
            logger.error(f"Performance test validation failed: {str(e)}")
            return 0.0
    
    async def _validate_coverage_test(self, component_path: str, 
                                    criterion: ValidationCriterion) -> float:
        """Validate test coverage"""
        try:
            # This is a simplified implementation
            # In a real system, this would run coverage tools
            
            component_name = Path(component_path).stem
            test_file = Path(component_path).parent.parent / 'tests' / f'test_{component_name}.py'
            
            if test_file.exists():
                # Check if test file has meaningful content
                with open(test_file, 'r', encoding='utf-8') as f:
                    test_content = f.read()
                
                # Count test functions
                test_count = test_content.count('def test_')
                
                if test_count >= 3:
                    return 85.0  # Assume good coverage if multiple tests exist
                elif test_count >= 1:
                    return 60.0  # Partial coverage
                else:
                    return 20.0  # Minimal coverage
            else:
                return 0.0  # No tests
                
        except Exception as e:
            logger.error(f"Coverage test validation failed: {str(e)}")
            return 0.0
    
    async def _validate_documentation_analysis(self, component_path: str, 
                                             criterion: ValidationCriterion) -> float:
        """Validate documentation analysis"""
        try:
            doc_score = await self.assessment_engine._calculate_documentation_score(component_path)
            return doc_score
            
        except Exception as e:
            logger.error(f"Documentation analysis validation failed: {str(e)}")
            return 0.0
    
    async def _validate_monitoring_check(self, component_path: str, 
                                       criterion: ValidationCriterion) -> bool:
        """Validate monitoring implementation"""
        try:
            with open(component_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for health check function
            has_health_check = 'def health_check' in content or 'async def health_check' in content
            
            # Check for monitoring-related imports or usage
            monitoring_indicators = [
                'time.time()', 'timestamp', 'status', 'health',
                'metrics', 'monitor'
            ]
            
            has_monitoring = any(indicator in content for indicator in monitoring_indicators)
            
            return has_health_check and has_monitoring
            
        except Exception as e:
            logger.error(f"Monitoring check validation failed: {str(e)}")
            return False
    
    async def _validate_error_handling_check(self, component_path: str, 
                                           criterion: ValidationCriterion) -> bool:
        """Validate error handling implementation"""
        try:
            with open(component_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for try-except blocks
            has_try_except = 'try:' in content and 'except' in content
            
            # Check for proper exception handling
            has_exception_handling = 'except Exception' in content or 'except ' in content
            
            # Check for logging in exception handlers
            has_error_logging = ('logger.error' in content or 
                               'logging.error' in content)
            
            return has_try_except and has_exception_handling and has_error_logging
            
        except Exception as e:
            logger.error(f"Error handling check validation failed: {str(e)}")
            return False
    
    async def _validate_input_validation_check(self, component_path: str, 
                                             criterion: ValidationCriterion) -> bool:
        """Validate input validation implementation"""
        try:
            with open(component_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for validation function
            has_validation_function = 'def validate_input' in content
            
            # Check for validation patterns
            validation_patterns = [
                'isinstance(', 'type(', 'ValueError', 'TypeError',
                'validate', 'check', 'verify'
            ]
            
            has_validation_patterns = sum(1 for pattern in validation_patterns 
                                        if pattern in content) >= 2
            
            return has_validation_function or has_validation_patterns
            
        except Exception as e:
            logger.error(f"Input validation check failed: {str(e)}")
            return False
    
    async def _validate_integration_test(self, component_path: str, 
                                       criterion: ValidationCriterion) -> bool:
        """Validate integration test results"""
        try:
            # This would run actual integration tests
            # For now, just check if the component can be imported
            
            component_name = Path(component_path).stem
            parent_module = Path(component_path).parent.name
            
            # Try to import the module
            import_cmd = f"from scrollintel.{parent_module} import {component_name}"
            
            result = subprocess.run(
                ['python', '-c', import_cmd],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Integration test validation failed: {str(e)}")
            return False
    
    async def generate_validation_report(self, criteria: List[ValidationCriterion], 
                                       component_path: str) -> Dict[str, Any]:
        """Generate a comprehensive validation report"""
        try:
            passed_criteria = [c for c in criteria if c.passed]
            failed_criteria = [c for c in criteria if not c.passed]
            
            report = {
                "component_path": component_path,
                "validation_timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_criteria": len(criteria),
                    "passed": len(passed_criteria),
                    "failed": len(failed_criteria),
                    "success_rate": len(passed_criteria) / len(criteria) if criteria else 0.0
                },
                "passed_criteria": [
                    {
                        "name": c.name,
                        "description": c.description,
                        "expected": c.expected_result,
                        "actual": c.actual_result
                    }
                    for c in passed_criteria
                ],
                "failed_criteria": [
                    {
                        "name": c.name,
                        "description": c.description,
                        "expected": c.expected_result,
                        "actual": c.actual_result
                    }
                    for c in failed_criteria
                ],
                "recommendations": await self._generate_validation_recommendations(failed_criteria)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Validation report generation failed: {str(e)}")
            return {"error": str(e)}
    
    async def _generate_validation_recommendations(self, 
                                                 failed_criteria: List[ValidationCriterion]) -> List[str]:
        """Generate recommendations based on failed validation criteria"""
        recommendations = []
        
        for criterion in failed_criteria:
            if criterion.name == "syntax_check":
                recommendations.append("Fix Python syntax errors in the code")
            elif criterion.name == "import_check":
                recommendations.append("Resolve import errors and missing dependencies")
            elif criterion.name == "error_handling_present":
                recommendations.append("Add comprehensive error handling with try-except blocks")
            elif criterion.name == "logging_present":
                recommendations.append("Implement structured logging throughout the component")
            elif criterion.name == "basic_security_check":
                recommendations.append("Address security vulnerabilities identified in the scan")
            elif criterion.name == "documentation_present":
                recommendations.append("Add docstrings and documentation to functions and classes")
            elif criterion.name == "test_coverage":
                recommendations.append("Increase test coverage to meet minimum requirements")
            elif criterion.name == "performance_score":
                recommendations.append("Optimize performance bottlenecks and inefficient algorithms")
            elif criterion.name == "code_quality_score":
                recommendations.append("Refactor code to improve readability and maintainability")
            else:
                recommendations.append(f"Address issues with {criterion.name}")
        
        return recommendations
    
    async def validate_production_readiness(self, component_path: str, 
                                          standards: ProductionStandards) -> Tuple[bool, Dict[str, Any]]:
        """
        Comprehensive production readiness validation
        
        Args:
            component_path: Path to component to validate
            standards: Production standards to validate against
            
        Returns:
            Tuple of (is_ready, detailed_report)
        """
        try:
            logger.info(f"Validating production readiness for {component_path}")
            
            # Perform fresh assessment
            assessment = await self.assessment_engine.assess(component_path)
            
            # Check against standards
            readiness_checks = {
                "code_quality": assessment.code_quality_score >= standards.min_code_quality_score,
                "security": assessment.security_score >= standards.min_security_score,
                "performance": assessment.performance_score >= standards.min_performance_score,
                "test_coverage": assessment.test_coverage >= standards.min_test_coverage,
                "documentation": assessment.documentation_score >= standards.min_documentation_score,
                "error_handling": standards.required_error_handling and await self._check_error_handling_present(component_path),
                "logging": standards.required_logging and await self._check_logging_present(component_path),
                "monitoring": standards.required_monitoring and await self._validate_monitoring_check(component_path, None),
                "security_controls": standards.required_security_controls and await self._validate_input_validation_check(component_path, None),
                "input_validation": standards.required_input_validation and await self._validate_input_validation_check(component_path, None)
            }
            
            # Calculate overall readiness
            passed_checks = sum(1 for check in readiness_checks.values() if check)
            total_checks = len(readiness_checks)
            readiness_score = passed_checks / total_checks
            
            is_ready = readiness_score >= 0.9  # 90% of checks must pass
            
            report = {
                "component_path": component_path,
                "is_production_ready": is_ready,
                "readiness_score": readiness_score,
                "assessment_scores": {
                    "code_quality": assessment.code_quality_score,
                    "security": assessment.security_score,
                    "performance": assessment.performance_score,
                    "test_coverage": assessment.test_coverage,
                    "documentation": assessment.documentation_score,
                    "overall": assessment.production_readiness_score
                },
                "standards_compliance": readiness_checks,
                "failed_checks": [check for check, passed in readiness_checks.items() if not passed],
                "critical_issues": len([i for i in assessment.identified_issues if i.severity == Severity.CRITICAL]),
                "recommendations": [r.title for r in assessment.upgrade_recommendations[:5]]  # Top 5 recommendations
            }
            
            logger.info(f"Production readiness validation completed. Ready: {is_ready}, Score: {readiness_score:.2f}")
            
            return is_ready, report
            
        except Exception as e:
            logger.error(f"Production readiness validation failed: {str(e)}")
            return False, {"error": str(e)}
    
    async def _check_error_handling_present(self, component_path: str) -> bool:
        """Check if error handling is present"""
        return await self._validate_error_handling_check(component_path, None)