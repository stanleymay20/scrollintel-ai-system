"""
Pipeline Monitoring and Observability System
Provides structured logging, metrics collection, and health checks.
"""
import logging
import time
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from functools import wraps
import psutil
import threading
from collections import defaultdict, deque

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(extra)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/pipeline.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class PipelineMetric:
    """Pipeline performance metric"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str]
    unit: str = ""

@dataclass
class PipelineEvent:
    """Pipeline event for audit trail"""
    event_type: str
    pipeline_id: str
    user_id: str
    timestamp: datetime
    details: Dict[str, Any]
    success: bool = True

class MetricsCollector:
    """Collects and stores pipeline metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.events: deque = deque(maxlen=5000)
        self.lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None, unit: str = ""):
        """Record a metric"""
        metric = PipelineMetric(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            labels=labels or {},
            unit=unit
        )
        
        with self.lock:
            self.metrics[name].append(metric)
    
    def record_event(self, event_type: str, pipeline_id: str, user_id: str, 
                    details: Dict[str, Any], success: bool = True):
        """Record a pipeline event"""
        event = PipelineEvent(
            event_type=event_type,
            pipeline_id=pipeline_id,
            user_id=user_id,
            timestamp=datetime.utcnow(),
            details=details,
            success=success
        )
        
        with self.lock:
            self.events.append(event)
    
    def get_metrics(self, name: str, since: Optional[datetime] = None) -> List[PipelineMetric]:
        """Get metrics by name"""
        with self.lock:
            metrics = list(self.metrics[name])
            if since:
                metrics = [m for m in metrics if m.timestamp >= since]
            return metrics
    
    def get_events(self, pipeline_id: Optional[str] = None, 
                  since: Optional[datetime] = None) -> List[PipelineEvent]:
        """Get events with optional filtering"""
        with self.lock:
            events = list(self.events)
            if pipeline_id:
                events = [e for e in events if e.pipeline_id == pipeline_id]
            if since:
                events = [e for e in events if e.timestamp >= since]
            return events
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics"""
        with self.lock:
            return {
                'total_metrics': sum(len(metrics) for metrics in self.metrics.values()),
                'total_events': len(self.events),
                'metric_types': list(self.metrics.keys()),
                'recent_events': len([e for e in self.events 
                                    if e.timestamp >= datetime.utcnow() - timedelta(hours=1)]),
                'error_events': len([e for e in self.events if not e.success])
            }

class StructuredLogger:
    """Structured logging for pipeline operations"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.metrics_collector = metrics_collector
    
    def log_pipeline_operation(self, operation: str, pipeline_id: str, user_id: str, 
                             details: Dict[str, Any], success: bool = True, 
                             duration: Optional[float] = None):
        """Log a pipeline operation with structured data"""
        log_data = {
            'operation': operation,
            'pipeline_id': pipeline_id,
            'user_id': user_id,
            'success': success,
            'timestamp': datetime.utcnow().isoformat(),
            **details
        }
        
        if duration is not None:
            log_data['duration_ms'] = duration * 1000
            self.metrics_collector.record_metric(
                f"pipeline_operation_duration",
                duration,
                labels={'operation': operation, 'pipeline_id': pipeline_id},
                unit="seconds"
            )
        
        level = logging.INFO if success else logging.ERROR
        self.logger.log(level, f"Pipeline {operation}", extra=log_data)
        
        # Record event
        self.metrics_collector.record_event(
            event_type=operation,
            pipeline_id=pipeline_id,
            user_id=user_id,
            details=details,
            success=success
        )
    
    def log_validation_result(self, pipeline_id: str, is_valid: bool, 
                            errors: List[str], warnings: List[str]):
        """Log pipeline validation results"""
        self.log_pipeline_operation(
            operation="validation",
            pipeline_id=pipeline_id,
            user_id="system",
            details={
                'is_valid': is_valid,
                'error_count': len(errors),
                'warning_count': len(warnings),
                'errors': errors[:5],  # Limit to first 5 errors
                'warnings': warnings[:5]  # Limit to first 5 warnings
            },
            success=is_valid
        )
    
    def log_performance_metric(self, metric_name: str, value: float, 
                             pipeline_id: str, labels: Dict[str, str] = None):
        """Log a performance metric"""
        self.metrics_collector.record_metric(
            name=metric_name,
            value=value,
            labels={**(labels or {}), 'pipeline_id': pipeline_id}
        )

class HealthChecker:
    """System health monitoring"""
    
    def __init__(self):
        self.checks = {}
        self.last_check = None
        self.check_interval = 60  # seconds
    
    def register_check(self, name: str, check_func):
        """Register a health check function"""
        self.checks[name] = check_func
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        for name, check_func in self.checks.items():
            try:
                start_time = time.time()
                result = check_func()
                duration = time.time() - start_time
                
                results['checks'][name] = {
                    'status': 'healthy' if result else 'unhealthy',
                    'duration_ms': duration * 1000,
                    'details': result if isinstance(result, dict) else {}
                }
                
                if not result:
                    results['overall_status'] = 'unhealthy'
                    
            except Exception as e:
                results['checks'][name] = {
                    'status': 'error',
                    'error': str(e)
                }
                results['overall_status'] = 'unhealthy'
        
        # Add system metrics
        results['system'] = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent
        }
        
        self.last_check = results
        return results
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        if (not self.last_check or 
            datetime.fromisoformat(self.last_check['timestamp']) < 
            datetime.utcnow() - timedelta(seconds=self.check_interval)):
            return self.run_health_checks()
        return self.last_check

def monitor_performance(operation_name: str):
    """Decorator to monitor function performance"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error = None
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                duration = time.time() - start_time
                
                # Extract pipeline_id and user_id from kwargs if available
                pipeline_id = kwargs.get('pipeline_id', 'unknown')
                user_id = kwargs.get('current_user', {}).get('user_id', 'unknown')
                
                structured_logger.log_pipeline_operation(
                    operation=operation_name,
                    pipeline_id=pipeline_id,
                    user_id=user_id,
                    details={'error': error} if error else {},
                    success=success,
                    duration=duration
                )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                duration = time.time() - start_time
                
                pipeline_id = kwargs.get('pipeline_id', 'unknown')
                user_id = kwargs.get('current_user', {}).get('user_id', 'unknown')
                
                structured_logger.log_pipeline_operation(
                    operation=operation_name,
                    pipeline_id=pipeline_id,
                    user_id=user_id,
                    details={'error': error} if error else {},
                    success=success,
                    duration=duration
                )
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator

# Global instances
metrics_collector = MetricsCollector()
structured_logger = StructuredLogger('pipeline_system')
health_checker = HealthChecker()

# Register default health checks
def database_health_check():
    """Check database connectivity"""
    try:
        from scrollintel.core.database_optimization import db_optimizer
        return db_optimizer.health_check()
    except Exception:
        return False

def system_resources_check():
    """Check system resource usage"""
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    disk_percent = psutil.disk_usage('/').percent
    
    return {
        'cpu_ok': cpu_percent < 80,
        'memory_ok': memory_percent < 80,
        'disk_ok': disk_percent < 80,
        'cpu_percent': cpu_percent,
        'memory_percent': memory_percent,
        'disk_percent': disk_percent
    }

health_checker.register_check('database', database_health_check)
health_checker.register_check('system_resources', system_resources_check)