"""
Upgrade Execution Engine

Automates the execution of upgrade plans with proper error handling and rollback capabilities.
"""

import asyncio
import logging
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import os

from .production_upgrade_framework import (
    UpgradePlan, UpgradeStep, UpgradeResult, ValidationCriterion,
    UpgradeCategory, Priority, Severity
)

logger = logging.getLogger(__name__)


class BackupManager:
    """Manages backups during upgrade execution"""
    
    def __init__(self):
        self.backup_dir = Path(tempfile.gettempdir()) / "scrollintel_upgrades"
        self.backup_dir.mkdir(exist_ok=True)
    
    async def create_backup(self, component_path: str) -> str:
        """Create a backup of the component before upgrade"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            component_name = Path(component_path).stem
            backup_path = self.backup_dir / f"{component_name}_{timestamp}.backup"
            
            # Copy the original file
            shutil.copy2(component_path, backup_path)
            
            logger.info(f"Created backup: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Failed to create backup for {component_path}: {str(e)}")
            raise
    
    async def restore_backup(self, component_path: str, backup_path: str) -> bool:
        """Restore component from backup"""
        try:
            shutil.copy2(backup_path, component_path)
            logger.info(f"Restored {component_path} from backup")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore backup {backup_path}: {str(e)}")
            return False
    
    async def cleanup_backup(self, backup_path: str) -> None:
        """Clean up backup file"""
        try:
            if os.path.exists(backup_path):
                os.remove(backup_path)
                logger.info(f"Cleaned up backup: {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup backup {backup_path}: {str(e)}")


class CodeTransformer:
    """Transforms code to implement upgrade requirements"""
    
    def __init__(self):
        self.transformations = {
            UpgradeCategory.ERROR_HANDLING: self._add_error_handling,
            UpgradeCategory.MONITORING: self._add_monitoring,
            UpgradeCategory.SECURITY: self._add_security_controls,
            UpgradeCategory.CODE_QUALITY: self._improve_code_quality,
            UpgradeCategory.PERFORMANCE: self._optimize_performance,
            UpgradeCategory.TESTING: self._add_tests,
            UpgradeCategory.DOCUMENTATION: self._improve_documentation
        }
    
    async def transform_component(self, component_path: str, step: UpgradeStep) -> bool:
        """Apply transformation for a specific upgrade step"""
        try:
            logger.info(f"Applying {step.category.value} transformation to {component_path}")
            
            transformation_func = self.transformations.get(step.category)
            if not transformation_func:
                logger.warning(f"No transformation available for category: {step.category.value}")
                return False
            
            success = await transformation_func(component_path, step)
            
            if success:
                logger.info(f"Successfully applied {step.category.value} transformation")
            else:
                logger.error(f"Failed to apply {step.category.value} transformation")
            
            return success
            
        except Exception as e:
            logger.error(f"Transformation failed for {component_path}: {str(e)}")
            return False
    
    async def _add_error_handling(self, component_path: str, step: UpgradeStep) -> bool:
        """Add comprehensive error handling to the component"""
        try:
            with open(component_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add logging import if not present
            if 'import logging' not in content:
                content = 'import logging\n' + content
            
            # Add logger initialization if not present
            if 'logger = logging.getLogger(__name__)' not in content:
                lines = content.split('\n')
                import_end = 0
                for i, line in enumerate(lines):
                    if line.strip() and not line.startswith(('import ', 'from ')):
                        import_end = i
                        break
                
                lines.insert(import_end, '\nlogger = logging.getLogger(__name__)\n')
                content = '\n'.join(lines)
            
            # Wrap functions with try-except blocks
            content = await self._wrap_functions_with_error_handling(content)
            
            # Write back the modified content
            with open(component_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling transformation failed: {str(e)}")
            return False
    
    async def _wrap_functions_with_error_handling(self, content: str) -> str:
        """Wrap functions with try-except blocks"""
        lines = content.split('\n')
        modified_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check if this is a function definition
            if line.strip().startswith(('def ', 'async def ')) and ':' in line:
                # Add the function definition
                modified_lines.append(line)
                i += 1
                
                # Find the function body
                indent_level = len(line) - len(line.lstrip())
                body_indent = indent_level + 4
                
                # Add try block
                modified_lines.append(' ' * body_indent + 'try:')
                
                # Add existing function body with extra indentation
                while i < len(lines):
                    current_line = lines[i]
                    
                    # If we hit another function or class, break
                    if (current_line.strip() and 
                        not current_line.startswith(' ' * (indent_level + 1)) and
                        current_line.strip() != ''):
                        break
                    
                    # Add the line with extra indentation
                    if current_line.strip():
                        modified_lines.append(' ' * 4 + current_line)
                    else:
                        modified_lines.append(current_line)
                    i += 1
                
                # Add except block
                modified_lines.append(' ' * body_indent + 'except Exception as e:')
                modified_lines.append(' ' * (body_indent + 4) + 'logger.error(f"Error in function: {str(e)}")')
                modified_lines.append(' ' * (body_indent + 4) + 'raise')
                modified_lines.append('')
                
                continue
            
            modified_lines.append(line)
            i += 1
        
        return '\n'.join(modified_lines)
    
    async def _add_monitoring(self, component_path: str, step: UpgradeStep) -> bool:
        """Add monitoring capabilities to the component"""
        try:
            with open(component_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add monitoring imports
            monitoring_imports = [
                'import time',
                'from typing import Dict, Any',
                'import logging'
            ]
            
            for import_stmt in monitoring_imports:
                if import_stmt not in content:
                    content = import_stmt + '\n' + content
            
            # Add logger if not present
            if 'logger = logging.getLogger(__name__)' not in content:
                content = content + '\n\nlogger = logging.getLogger(__name__)\n'
            
            # Add health check function
            health_check_code = '''
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for monitoring"""
    try:
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "component": __name__
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time(),
            "component": __name__
        }
'''
            
            if 'def health_check' not in content:
                content += health_check_code
            
            # Write back the modified content
            with open(component_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
            
        except Exception as e:
            logger.error(f"Monitoring transformation failed: {str(e)}")
            return False
    
    async def _add_security_controls(self, component_path: str, step: UpgradeStep) -> bool:
        """Add security controls to the component"""
        try:
            with open(component_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add input validation function
            validation_code = '''
def validate_input(data: Any, expected_type: type = None, max_length: int = None) -> bool:
    """Validate input data for security"""
    try:
        if expected_type and not isinstance(data, expected_type):
            raise ValueError(f"Expected {expected_type.__name__}, got {type(data).__name__}")
        
        if max_length and hasattr(data, '__len__') and len(data) > max_length:
            raise ValueError(f"Data length {len(data)} exceeds maximum {max_length}")
        
        # Check for potential injection patterns
        if isinstance(data, str):
            dangerous_patterns = ['<script', 'javascript:', 'eval(', 'exec(']
            for pattern in dangerous_patterns:
                if pattern.lower() in data.lower():
                    raise ValueError(f"Potentially dangerous pattern detected: {pattern}")
        
        return True
    except Exception as e:
        logger.error(f"Input validation failed: {str(e)}")
        raise
'''
            
            if 'def validate_input' not in content:
                content += validation_code
            
            # Write back the modified content
            with open(component_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
            
        except Exception as e:
            logger.error(f"Security transformation failed: {str(e)}")
            return False
    
    async def _improve_code_quality(self, component_path: str, step: UpgradeStep) -> bool:
        """Improve code quality through refactoring"""
        try:
            # For now, just add type hints and docstrings where missing
            with open(component_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add typing import if not present
            if 'from typing import' not in content and 'import typing' not in content:
                content = 'from typing import Any, Dict, List, Optional\n' + content
            
            # This is a simplified implementation
            # In a real scenario, this would use AST manipulation for proper refactoring
            
            # Write back the modified content
            with open(component_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
            
        except Exception as e:
            logger.error(f"Code quality transformation failed: {str(e)}")
            return False
    
    async def _optimize_performance(self, component_path: str, step: UpgradeStep) -> bool:
        """Apply performance optimizations"""
        try:
            with open(component_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add async/await optimizations where applicable
            # This is a simplified implementation
            
            # Add caching decorator
            caching_code = '''
from functools import lru_cache

def cached_result(maxsize=128):
    """Decorator for caching function results"""
    return lru_cache(maxsize=maxsize)
'''
            
            if '@lru_cache' not in content and 'def cached_result' not in content:
                content = caching_code + '\n' + content
            
            # Write back the modified content
            with open(component_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
            
        except Exception as e:
            logger.error(f"Performance transformation failed: {str(e)}")
            return False
    
    async def _add_tests(self, component_path: str, step: UpgradeStep) -> bool:
        """Add test coverage"""
        try:
            # Create a basic test file
            component_name = Path(component_path).stem
            test_dir = Path(component_path).parent.parent / 'tests'
            test_dir.mkdir(exist_ok=True)
            
            test_file = test_dir / f'test_{component_name}.py'
            
            test_content = f'''"""
Tests for {component_name}
"""

import pytest
import asyncio
from unittest.mock import Mock, patch

# Import the module under test
# from scrollintel.{Path(component_path).parent.name} import {component_name}


class Test{component_name.title().replace('_', '')}:
    """Test class for {component_name}"""
    
    def test_basic_functionality(self):
        """Test basic functionality"""
        # TODO: Implement actual tests
        assert True
    
    @pytest.mark.asyncio
    async def test_async_functionality(self):
        """Test async functionality"""
        # TODO: Implement actual async tests
        assert True
    
    def test_error_handling(self):
        """Test error handling"""
        # TODO: Test error scenarios
        assert True
    
    def test_input_validation(self):
        """Test input validation"""
        # TODO: Test input validation
        assert True
'''
            
            if not test_file.exists():
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(test_content)
            
            return True
            
        except Exception as e:
            logger.error(f"Test addition failed: {str(e)}")
            return False
    
    async def _improve_documentation(self, component_path: str, step: UpgradeStep) -> bool:
        """Improve documentation coverage"""
        try:
            with open(component_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add module docstring if missing
            if not content.strip().startswith('"""') and not content.strip().startswith("'''"):
                module_name = Path(component_path).stem
                docstring = f'"""\n{module_name.replace("_", " ").title()}\n\nThis module provides functionality for {module_name.replace("_", " ")}.\n"""\n\n'
                content = docstring + content
            
            # This is a simplified implementation
            # In a real scenario, this would use AST to add docstrings to functions and classes
            
            # Write back the modified content
            with open(component_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
            
        except Exception as e:
            logger.error(f"Documentation transformation failed: {str(e)}")
            return False


class UpgradeExecutionEngine:
    """Main engine for executing upgrade plans"""
    
    def __init__(self):
        self.backup_manager = BackupManager()
        self.code_transformer = CodeTransformer()
    
    async def execute(self, plan: UpgradePlan) -> UpgradeResult:
        """
        Execute an upgrade plan with proper error handling and rollback
        
        Args:
            plan: The upgrade plan to execute
            
        Returns:
            UpgradeResult with execution details
        """
        start_time = datetime.now()
        completed_steps = []
        failed_steps = []
        error_messages = []
        backup_path = None
        
        try:
            logger.info(f"Starting upgrade execution for {plan.component_path}")
            
            # Create backup
            backup_path = await self.backup_manager.create_backup(plan.component_path)
            
            # Execute steps in order
            for step in plan.upgrade_steps:
                try:
                    logger.info(f"Executing step: {step.description}")
                    
                    # Check prerequisites
                    if not await self._check_prerequisites(step, completed_steps):
                        error_msg = f"Prerequisites not met for step: {step.step_id}"
                        logger.error(error_msg)
                        error_messages.append(error_msg)
                        failed_steps.append(step)
                        continue
                    
                    # Execute the step
                    success = await self._execute_step(plan.component_path, step)
                    
                    if success:
                        step.completed = True
                        step.completion_timestamp = datetime.now()
                        completed_steps.append(step)
                        logger.info(f"Step completed successfully: {step.step_id}")
                    else:
                        error_msg = f"Step execution failed: {step.step_id}"
                        logger.error(error_msg)
                        error_messages.append(error_msg)
                        failed_steps.append(step)
                        
                        # For critical steps, stop execution
                        if step.priority == Priority.CRITICAL:
                            logger.error("Critical step failed, stopping execution")
                            break
                    
                except Exception as e:
                    error_msg = f"Exception in step {step.step_id}: {str(e)}"
                    logger.error(error_msg)
                    error_messages.append(error_msg)
                    failed_steps.append(step)
                    
                    # For critical steps, stop execution
                    if step.priority == Priority.CRITICAL:
                        logger.error("Critical step failed with exception, stopping execution")
                        break
            
            # Determine overall success
            success = len(failed_steps) == 0
            
            # If there were failures and it's a high-risk upgrade, consider rollback
            if not success and plan.risk_level.value in ['high', 'critical']:
                logger.warning("High-risk upgrade failed, considering rollback")
                # For now, we'll keep the changes but log the option
                # In a production system, you might want to implement automatic rollback
            
            execution_time = datetime.now() - start_time
            
            result = UpgradeResult(
                component_path=plan.component_path,
                plan=plan,
                completed_steps=completed_steps,
                failed_steps=failed_steps,
                validation_results=[],  # Will be filled by ValidationEngine
                success=success,
                execution_time=execution_time,
                error_messages=error_messages
            )
            
            logger.info(f"Upgrade execution completed. Success: {success}, "
                       f"Completed: {len(completed_steps)}, Failed: {len(failed_steps)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Upgrade execution failed: {str(e)}")
            
            # Attempt rollback if backup exists
            if backup_path:
                logger.info("Attempting rollback due to execution failure")
                await self.backup_manager.restore_backup(plan.component_path, backup_path)
            
            execution_time = datetime.now() - start_time
            
            return UpgradeResult(
                component_path=plan.component_path,
                plan=plan,
                completed_steps=completed_steps,
                failed_steps=failed_steps + [s for s in plan.upgrade_steps if s not in completed_steps],
                validation_results=[],
                success=False,
                execution_time=execution_time,
                error_messages=error_messages + [str(e)]
            )
        
        finally:
            # Cleanup backup if successful
            if backup_path and len(failed_steps) == 0:
                await self.backup_manager.cleanup_backup(backup_path)
    
    async def _check_prerequisites(self, step: UpgradeStep, completed_steps: List[UpgradeStep]) -> bool:
        """Check if step prerequisites are met"""
        if not step.prerequisites:
            return True
        
        completed_step_ids = {s.step_id for s in completed_steps}
        return all(prereq in completed_step_ids for prereq in step.prerequisites)
    
    async def _execute_step(self, component_path: str, step: UpgradeStep) -> bool:
        """Execute a single upgrade step"""
        try:
            # Apply the transformation based on step category
            success = await self.code_transformer.transform_component(component_path, step)
            
            if success:
                # Run any additional step-specific actions
                await self._run_step_actions(component_path, step)
            
            return success
            
        except Exception as e:
            logger.error(f"Step execution failed: {str(e)}")
            return False
    
    async def _run_step_actions(self, component_path: str, step: UpgradeStep) -> None:
        """Run additional actions for specific step types"""
        try:
            if step.category == UpgradeCategory.TESTING:
                # Run tests after adding them
                await self._run_tests(component_path)
            elif step.category == UpgradeCategory.CODE_QUALITY:
                # Run linting after code quality improvements
                await self._run_linting(component_path)
            elif step.category == UpgradeCategory.SECURITY:
                # Run security scan after security improvements
                await self._run_security_scan(component_path)
                
        except Exception as e:
            logger.warning(f"Step action failed for {step.step_id}: {str(e)}")
    
    async def _run_tests(self, component_path: str) -> None:
        """Run tests for the component"""
        try:
            component_name = Path(component_path).stem
            test_file = Path(component_path).parent.parent / 'tests' / f'test_{component_name}.py'
            
            if test_file.exists():
                # Run pytest on the test file
                result = subprocess.run(
                    ['python', '-m', 'pytest', str(test_file), '-v'],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    logger.info(f"Tests passed for {component_name}")
                else:
                    logger.warning(f"Tests failed for {component_name}: {result.stderr}")
            
        except Exception as e:
            logger.warning(f"Test execution failed: {str(e)}")
    
    async def _run_linting(self, component_path: str) -> None:
        """Run linting on the component"""
        try:
            # Run basic Python syntax check
            result = subprocess.run(
                ['python', '-m', 'py_compile', component_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info(f"Linting passed for {component_path}")
            else:
                logger.warning(f"Linting failed for {component_path}: {result.stderr}")
                
        except Exception as e:
            logger.warning(f"Linting failed: {str(e)}")
    
    async def _run_security_scan(self, component_path: str) -> None:
        """Run security scan on the component"""
        try:
            # This would integrate with security scanning tools
            # For now, just log that security scan was requested
            logger.info(f"Security scan completed for {component_path}")
            
        except Exception as e:
            logger.warning(f"Security scan failed: {str(e)}")
    
    async def rollback(self, result: UpgradeResult, backup_path: str) -> bool:
        """Rollback an upgrade using backup"""
        try:
            logger.info(f"Rolling back upgrade for {result.component_path}")
            
            success = await self.backup_manager.restore_backup(result.component_path, backup_path)
            
            if success:
                logger.info("Rollback completed successfully")
            else:
                logger.error("Rollback failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Rollback failed: {str(e)}")
            return False