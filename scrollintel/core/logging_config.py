"""
ScrollIntel Centralized Logging System
Structured logging with JSON format and multiple handlers
"""

import logging
import logging.config
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import traceback
from pythonjsonlogger import jsonlogger

from ..core.config import get_settings

settings = get_settings()

class ScrollIntelFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for ScrollIntel logs"""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]):
        super().add_fields(log_record, record, message_dict)
        
        # Add standard fields
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno
        
        # Add process info
        log_record['process_id'] = os.getpid()
        log_record['thread_id'] = record.thread
        
        # Add exception info if present
        if record.exc_info:
            log_record['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
            
        # Add custom fields from extra
        if hasattr(record, 'user_id'):
            log_record['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_record['request_id'] = record.request_id
        if hasattr(record, 'agent_type'):
            log_record['agent_type'] = record.agent_type
        if hasattr(record, 'operation'):
            log_record['operation'] = record.operation
        if hasattr(record, 'duration'):
            log_record['duration'] = record.duration
        if hasattr(record, 'status_code'):
            log_record['status_code'] = record.status_code

class ContextFilter(logging.Filter):
    """Filter to add context information to log records"""
    
    def __init__(self):
        super().__init__()
        self.context = {}
        
    def set_context(self, **kwargs):
        """Set context information"""
        self.context.update(kwargs)
        
    def clear_context(self):
        """Clear context information"""
        self.context.clear()
        
    def filter(self, record):
        # Add context to record
        for key, value in self.context.items():
            setattr(record, key, value)
        return True

# Global context filter
context_filter = ContextFilter()

def setup_logging():
    """Setup centralized logging configuration"""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Logging configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'json': {
                '()': ScrollIntelFormatter,
                'format': '%(timestamp)s %(level)s %(logger)s %(message)s'
            },
            'console': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'filters': {
            'context': {
                '()': ContextFilter
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'console',
                'stream': sys.stdout,
                'filters': ['context']
            },
            'file_json': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'json',
                'filename': 'logs/scrollintel.json',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'filters': ['context']
            },
            'error_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'json',
                'filename': 'logs/errors.json',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 10,
                'filters': ['context']
            },
            'audit_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'json',
                'filename': 'logs/audit.json',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 20,
                'filters': ['context']
            },
            'performance_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'json',
                'filename': 'logs/performance.json',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 10,
                'filters': ['context']
            }
        },
        'loggers': {
            'scrollintel': {
                'level': 'DEBUG',
                'handlers': ['console', 'file_json', 'error_file'],
                'propagate': False
            },
            'scrollintel.audit': {
                'level': 'INFO',
                'handlers': ['audit_file'],
                'propagate': False
            },
            'scrollintel.performance': {
                'level': 'INFO',
                'handlers': ['performance_file'],
                'propagate': False
            },
            'scrollintel.agents': {
                'level': 'DEBUG',
                'handlers': ['console', 'file_json'],
                'propagate': False
            },
            'scrollintel.api': {
                'level': 'INFO',
                'handlers': ['console', 'file_json'],
                'propagate': False
            },
            'scrollintel.security': {
                'level': 'INFO',
                'handlers': ['console', 'file_json', 'audit_file'],
                'propagate': False
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['console']
        }
    }
    
    logging.config.dictConfig(config)

class StructuredLogger:
    """Structured logger with context management"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.context = {}
        
    def set_context(self, **kwargs):
        """Set logging context"""
        self.context.update(kwargs)
        context_filter.set_context(**kwargs)
        
    def clear_context(self):
        """Clear logging context"""
        self.context.clear()
        context_filter.clear_context()
        
    def debug(self, message: str, **kwargs):
        """Log debug message with context"""
        self.logger.debug(message, extra={**self.context, **kwargs})
        
    def info(self, message: str, **kwargs):
        """Log info message with context"""
        self.logger.info(message, extra={**self.context, **kwargs})
        
    def warning(self, message: str, **kwargs):
        """Log warning message with context"""
        self.logger.warning(message, extra={**self.context, **kwargs})
        
    def error(self, message: str, **kwargs):
        """Log error message with context"""
        self.logger.error(message, extra={**self.context, **kwargs})
        
    def critical(self, message: str, **kwargs):
        """Log critical message with context"""
        self.logger.critical(message, extra={**self.context, **kwargs})
        
    def log_request(self, method: str, path: str, status_code: int, duration: float, user_id: Optional[str] = None):
        """Log HTTP request"""
        self.info(
            f"{method} {path} - {status_code}",
            operation="http_request",
            method=method,
            path=path,
            status_code=status_code,
            duration=duration,
            user_id=user_id
        )
        
    def log_agent_request(self, agent_type: str, operation: str, duration: float, status: str, user_id: Optional[str] = None):
        """Log agent request"""
        self.info(
            f"Agent {agent_type} - {operation} - {status}",
            operation="agent_request",
            agent_type=agent_type,
            agent_operation=operation,
            duration=duration,
            status=status,
            user_id=user_id
        )
        
    def log_error(self, error: Exception, operation: str, **kwargs):
        """Log error with full context"""
        extra_data = {
            **self.context,
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            **kwargs
        }
        self.logger.error(f"Error in {operation}: {str(error)}", extra=extra_data, exc_info=True)
        
    def log_security_event(self, event_type: str, user_id: Optional[str], details: Dict[str, Any]):
        """Log security event"""
        security_logger = logging.getLogger('scrollintel.security')
        extra_data = {
            "event_type": event_type,
            "user_id": user_id,
            **details
        }
        security_logger.info(f"Security event: {event_type}", extra=extra_data)
        
    def log_performance_metric(self, metric_name: str, value: float, unit: str, **kwargs):
        """Log performance metric"""
        perf_logger = logging.getLogger('scrollintel.performance')
        extra_data = {
            "metric_name": metric_name,
            "metric_value": value,
            "metric_unit": unit,
            **kwargs
        }
        perf_logger.info(f"Performance metric: {metric_name} = {value} {unit}", extra=extra_data)

def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance"""
    return StructuredLogger(name)

# Audit logger for security events
audit_logger = logging.getLogger('scrollintel.audit')

def log_audit_event(event_type: str, user_id: Optional[str], resource: str, action: str, details: Dict[str, Any] = None):
    """Log audit event for security compliance"""
    extra_data = {
        "event_type": event_type,
        "user_id": user_id,
        "resource": resource,
        "action": action,
        "details": details or {},
        "timestamp": datetime.utcnow().isoformat()
    }
    audit_logger.info(f"Audit: {event_type} - {action} on {resource}", extra=extra_data)

# Performance logger
performance_logger = logging.getLogger('scrollintel.performance')

def log_performance_event(operation: str, duration: float, success: bool, **kwargs):
    """Log performance event"""
    extra_data = {
        "operation": operation,
        "duration": duration,
        "success": success,
        **kwargs
    }
    performance_logger.info(f"Performance: {operation} - {duration:.3f}s - {'SUCCESS' if success else 'FAILED'}", extra=extra_data)

# Initialize logging on import
setup_logging()