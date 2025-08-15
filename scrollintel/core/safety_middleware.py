"""
Safety Middleware for ScrollIntel

This middleware intercepts all AI operations and applies safety checks
before allowing execution. It integrates with the AI Safety Framework
to ensure all operations meet safety and ethical requirements.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps
from datetime import datetime
import inspect

from scrollintel.core.ai_safety_framework import (
    ai_safety_framework,
    SafetyLevel,
    AlignmentStatus
)

logger = logging.getLogger(__name__)


class SafetyMiddleware:
    """Middleware that applies safety checks to all AI operations"""
    
    def __init__(self):
        self.enabled = True
        self.bypass_users: set = {"emergency_override", "safety_testing"}
        self.operation_log: list = []
        self.blocked_operations: list = []
    
    async def validate_and_execute(self, operation_func: Callable, 
                                 operation_data: Dict[str, Any],
                                 user_id: str = "system",
                                 **kwargs) -> Dict[str, Any]:
        """Validate operation through safety framework and execute if safe"""
        
        if not self.enabled and user_id not in self.bypass_users:
            return {
                "success": False,
                "error": "Safety middleware disabled - operation blocked",
                "safety_status": "blocked"
            }
        
        # Prepare operation context
        operation_context = {
            "operation_type": operation_func.__name__,
            "operation_data": operation_data,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "safety_level": self._determine_safety_level(operation_func, operation_data),
            "estimated_cpu_percent": self._estimate_cpu_usage(operation_data),
            "estimated_memory_gb": self._estimate_memory_usage(operation_data),
            **kwargs
        }
        
        # Log operation attempt
        self.operation_log.append({
            "operation": operation_context["operation_type"],
            "user": user_id,
            "timestamp": operation_context["timestamp"],
            "status": "validating"
        })
        
        try:
            # Validate through safety framework
            validation_result = await ai_safety_framework.validate_operation(operation_context)
            
            if not validation_result["allowed"]:
                # Operation blocked by safety framework
                self.blocked_operations.append({
                    "operation": operation_context,
                    "validation_result": validation_result,
                    "timestamp": datetime.now().isoformat()
                })
                
                logger.warning(f"Operation blocked by safety framework: {operation_func.__name__}")
                
                return {
                    "success": False,
                    "error": "Operation blocked by safety framework",
                    "safety_status": "blocked",
                    "violations": [v.description for v in validation_result["violations"]],
                    "warnings": validation_result["warnings"],
                    "required_approvals": validation_result["required_approvals"]
                }
            
            # Check for warnings that require user acknowledgment
            if validation_result["warnings"] and user_id not in self.bypass_users:
                logger.warning(f"Safety warnings for operation {operation_func.__name__}: {validation_result['warnings']}")
            
            # Execute operation if validation passed
            logger.info(f"Executing validated operation: {operation_func.__name__}")
            
            if inspect.iscoroutinefunction(operation_func):
                result = await operation_func(**operation_data)
            else:
                result = operation_func(**operation_data)
            
            # Log successful execution
            self.operation_log[-1]["status"] = "completed"
            
            return {
                "success": True,
                "result": result,
                "safety_status": "validated",
                "alignment_check": validation_result.get("alignment_check"),
                "warnings": validation_result["warnings"]
            }
            
        except Exception as e:
            logger.error(f"Error in safety middleware: {e}")
            self.operation_log[-1]["status"] = "error"
            
            return {
                "success": False,
                "error": f"Safety middleware error: {str(e)}",
                "safety_status": "error"
            }
    
    def _determine_safety_level(self, operation_func: Callable, 
                              operation_data: Dict[str, Any]) -> str:
        """Determine safety level of operation"""
        
        # Check function name for high-risk operations
        high_risk_keywords = [
            "delete", "destroy", "eliminate", "shutdown", "modify_system",
            "acquire_resources", "manipulate", "control", "override",
            "monopoly", "influence", "dominate"
        ]
        
        critical_keywords = [
            "emergency", "critical", "existential", "autonomous",
            "self_modify", "recursive", "unlimited"
        ]
        
        func_name = operation_func.__name__.lower()
        operation_str = str(operation_data).lower()
        
        # Check for existential risk operations
        if any(keyword in func_name or keyword in operation_str 
               for keyword in critical_keywords):
            return SafetyLevel.EXISTENTIAL.value
        
        # Check for high-risk operations
        if any(keyword in func_name or keyword in operation_str 
               for keyword in high_risk_keywords):
            return SafetyLevel.CRITICAL.value
        
        # Check for medium-risk operations
        medium_risk_keywords = [
            "create", "generate", "process", "analyze", "predict",
            "recommend", "optimize", "automate"
        ]
        
        if any(keyword in func_name or keyword in operation_str 
               for keyword in medium_risk_keywords):
            return SafetyLevel.MEDIUM.value
        
        # Default to low risk
        return SafetyLevel.LOW.value
    
    def _estimate_cpu_usage(self, operation_data: Dict[str, Any]) -> float:
        """Estimate CPU usage percentage for operation"""
        # Simple heuristic based on operation characteristics
        data_size = len(str(operation_data))
        
        if data_size > 100000:  # Large operations
            return 60.0
        elif data_size > 10000:  # Medium operations
            return 30.0
        else:  # Small operations
            return 10.0
    
    def _estimate_memory_usage(self, operation_data: Dict[str, Any]) -> float:
        """Estimate memory usage in GB for operation"""
        # Simple heuristic based on operation characteristics
        data_size = len(str(operation_data))
        
        if data_size > 100000:  # Large operations
            return 8.0
        elif data_size > 10000:  # Medium operations
            return 2.0
        else:  # Small operations
            return 0.5
    
    def get_safety_statistics(self) -> Dict[str, Any]:
        """Get safety middleware statistics"""
        total_operations = len(self.operation_log)
        blocked_count = len(self.blocked_operations)
        completed_count = len([op for op in self.operation_log if op["status"] == "completed"])
        error_count = len([op for op in self.operation_log if op["status"] == "error"])
        
        return {
            "enabled": self.enabled,
            "total_operations": total_operations,
            "completed_operations": completed_count,
            "blocked_operations": blocked_count,
            "error_operations": error_count,
            "success_rate": (completed_count / total_operations * 100) if total_operations > 0 else 0,
            "block_rate": (blocked_count / total_operations * 100) if total_operations > 0 else 0,
            "recent_operations": self.operation_log[-10:] if self.operation_log else [],
            "recent_blocks": self.blocked_operations[-5:] if self.blocked_operations else []
        }


def safety_required(safety_level: SafetyLevel = SafetyLevel.MEDIUM):
    """Decorator to mark functions as requiring safety validation"""
    
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract operation data from arguments
            operation_data = {}
            
            # Get function signature to map arguments
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            for param_name, param_value in bound_args.arguments.items():
                if param_name not in ['self', 'cls']:
                    operation_data[param_name] = param_value
            
            # Add safety level to operation data
            operation_data["required_safety_level"] = safety_level.value
            
            # Use safety middleware
            middleware = SafetyMiddleware()
            result = await middleware.validate_and_execute(
                func, operation_data, user_id=kwargs.get("user_id", "system")
            )
            
            if not result["success"]:
                raise Exception(result["error"])
            
            return result["result"]
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, create async wrapper
            async def async_exec():
                return await async_wrapper(*args, **kwargs)
            
            # Run in event loop
            try:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(async_exec())
            except RuntimeError:
                # Create new event loop if none exists
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(async_exec())
                finally:
                    loop.close()
        
        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def critical_operation(func: Callable):
    """Decorator for critical operations requiring highest safety level"""
    return safety_required(SafetyLevel.CRITICAL)(func)


def existential_operation(func: Callable):
    """Decorator for existential risk operations requiring maximum safety"""
    return safety_required(SafetyLevel.EXISTENTIAL)(func)


# Global middleware instance
safety_middleware = SafetyMiddleware()