"""
Performance monitoring engine for data pipeline automation.
Provides real-time monitoring, SLA tracking, and automated optimization.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import psutil
import time
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from ..models.performance_models import (
    PerformanceMetrics, ResourceUsage, SLAViolation, PerformanceAlert,
    OptimizationRecommendation, PerformanceTuningConfig
)
from ..core.database import SessionLocal

logger = logging.getLogger(__name__)

class PerformanceMonitoringEngine:
    """Engine for monitoring pipeline performance and generating optimization recommendations."""
    
    def __init__(self):
        self.monitoring_active = False
        self.monitoring_interval = 30  # seconds
        self.alert_thresholds = {
            'cpu_usage': 85.0,
            'memory_usage': 90.0,
            'error_rate': 0.05,
            'latency_ms': 10000
        }
        
    async def start_monitoring(self, pipeline_id: str, execution_id: str) -> Dict[str, Any]:
        """Start performance monitoring for a pipeline execution."""
        try:
            self.monitoring_active = True
            
            # Create initial performance metrics record
            with SessionLocal() as db:
                metrics = PerformanceMetrics(
                    pipeline_id=pipeline_id,
                    execution_id=execution_id,
                    start_time=datetime.utcnow()
                )
                db.add(metrics)
                db.commit()
                db.refresh(metrics)
                
                logger.info(f"Started monitoring for pipeline {pipeline_id}, execution {execution_id}")
                
                return {
                    "status": "monitoring_started",
                    "metrics_id": metrics.id,
                    "pipeline_id": pipeline_id,
                    "execution_id": execution_id,
                    "start_time": metrics.start_time
                }
                
        except Exception as e:
            logger.error(f"Error starting monitoring: {str(e)}")
            raise
    
    async def collect_metrics(self, metrics_id: int) -> Dict[str, Any]:
        """Collect current system metrics and update performance record."""
        try:
            # Collect system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            network_io = psutil.net_io_counters()
            
            current_time = datetime.utcnow()
            
            with SessionLocal() as db:
                # Update performance metrics
                metrics = db.query(PerformanceMetrics).filter(
                    PerformanceMetrics.id == metrics_id
                ).first()
                
                if metrics:
                    metrics.cpu_usage_percent = cpu_percent
                    metrics.memory_usage_mb = memory.used / (1024 * 1024)
                    
                    if disk_io:
                        metrics.disk_io_mb = (disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024)
                    
                    if network_io:
                        metrics.network_io_mb = (network_io.bytes_sent + network_io.bytes_recv) / (1024 * 1024)
                    
                    # Create detailed resource usage record
                    resource_records = [
                        ResourceUsage(
                            performance_metrics_id=metrics_id,
                            resource_type="cpu",
                            timestamp=current_time,
                            usage_value=cpu_percent,
                            usage_unit="percent",
                            allocated_limit=100.0,
                            warning_threshold=self.alert_thresholds['cpu_usage'],
                            critical_threshold=95.0
                        ),
                        ResourceUsage(
                            performance_metrics_id=metrics_id,
                            resource_type="memory",
                            timestamp=current_time,
                            usage_value=memory.percent,
                            usage_unit="percent",
                            allocated_limit=100.0,
                            warning_threshold=self.alert_thresholds['memory_usage'],
                            critical_threshold=95.0
                        )
                    ]
                    
                    for record in resource_records:
                        db.add(record)
                    
                    db.commit()
                    
                    # Check for SLA violations
                    await self._check_sla_violations(db, metrics)
                    
                    return {
                        "timestamp": current_time,
                        "cpu_usage": cpu_percent,
                        "memory_usage": memory.percent,
                        "disk_io_mb": metrics.disk_io_mb,
                        "network_io_mb": metrics.network_io_mb
                    }
                    
        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
            raise
    
    async def _check_sla_violations(self, db: Session, metrics: PerformanceMetrics):
        """Check for SLA violations and create alerts if necessary."""
        violations = []
        
        # Check CPU usage violation
        if metrics.cpu_usage_percent and metrics.cpu_usage_percent > self.alert_thresholds['cpu_usage']:
            violations.append({
                'sla_type': 'cpu_usage',
                'threshold': self.alert_thresholds['cpu_usage'],
                'actual': metrics.cpu_usage_percent,
                'severity': 'critical' if metrics.cpu_usage_percent > 95 else 'warning'
            })
        
        # Check memory usage violation
        if metrics.memory_usage_mb:
            memory_percent = (metrics.memory_usage_mb / (psutil.virtual_memory().total / (1024 * 1024))) * 100
            if memory_percent > self.alert_thresholds['memory_usage']:
                violations.append({
                    'sla_type': 'memory_usage',
                    'threshold': self.alert_thresholds['memory_usage'],
                    'actual': memory_percent,
                    'severity': 'critical' if memory_percent > 95 else 'warning'
                })
        
        # Check error rate violation
        if metrics.error_rate and metrics.error_rate > self.alert_thresholds['error_rate']:
            violations.append({
                'sla_type': 'error_rate',
                'threshold': self.alert_thresholds['error_rate'],
                'actual': metrics.error_rate,
                'severity': 'critical' if metrics.error_rate > 0.1 else 'warning'
            })
        
        # Create SLA violation records and alerts
        for violation in violations:
            sla_violation = SLAViolation(
                performance_metrics_id=metrics.id,
                sla_type=violation['sla_type'],
                sla_threshold=violation['threshold'],
                actual_value=violation['actual'],
                violation_severity=violation['severity'],
                violation_start=datetime.utcnow(),
                affected_pipelines=[metrics.pipeline_id]
            )
            db.add(sla_violation)
            db.flush()
            
            # Create alert
            alert = PerformanceAlert(
                sla_violation_id=sla_violation.id,
                alert_type=violation['sla_type'],
                alert_level=violation['severity'],
                alert_message=f"{violation['sla_type']} exceeded threshold: {violation['actual']:.2f} > {violation['threshold']:.2f}",
                notification_channels=['email', 'dashboard']
            )
            db.add(alert)
    
    async def stop_monitoring(self, metrics_id: int) -> Dict[str, Any]:
        """Stop monitoring and finalize performance metrics."""
        try:
            self.monitoring_active = False
            
            with SessionLocal() as db:
                metrics = db.query(PerformanceMetrics).filter(
                    PerformanceMetrics.id == metrics_id
                ).first()
                
                if metrics:
                    end_time = datetime.utcnow()
                    metrics.end_time = end_time
                    
                    if metrics.start_time:
                        duration = (end_time - metrics.start_time).total_seconds()
                        metrics.duration_seconds = duration
                        
                        # Calculate records per second if available
                        if metrics.records_processed and duration > 0:
                            metrics.records_per_second = metrics.records_processed / duration
                    
                    db.commit()
                    
                    # Generate optimization recommendations
                    recommendations = await self._generate_optimization_recommendations(db, metrics)
                    
                    logger.info(f"Stopped monitoring for metrics ID {metrics_id}")
                    
                    return {
                        "status": "monitoring_stopped",
                        "metrics_id": metrics_id,
                        "end_time": end_time,
                        "duration_seconds": metrics.duration_seconds,
                        "recommendations_generated": len(recommendations)
                    }
                    
        except Exception as e:
            logger.error(f"Error stopping monitoring: {str(e)}")
            raise
    
    async def _generate_optimization_recommendations(self, db: Session, metrics: PerformanceMetrics) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on performance metrics."""
        recommendations = []
        
        try:
            # CPU optimization recommendations
            if metrics.cpu_usage_percent and metrics.cpu_usage_percent > 80:
                recommendations.append({
                    'type': 'cpu_optimization',
                    'priority': 'high' if metrics.cpu_usage_percent > 90 else 'medium',
                    'title': 'High CPU Usage Detected',
                    'description': f'CPU usage is {metrics.cpu_usage_percent:.1f}%. Consider scaling up or optimizing processing logic.',
                    'expected_improvement': 25.0,
                    'implementation_steps': [
                        'Analyze CPU-intensive operations',
                        'Consider horizontal scaling',
                        'Optimize data processing algorithms',
                        'Implement caching for repeated operations'
                    ]
                })
            
            # Memory optimization recommendations
            if metrics.memory_usage_mb:
                memory_gb = metrics.memory_usage_mb / 1024
                if memory_gb > 8:  # Assuming high memory usage threshold
                    recommendations.append({
                        'type': 'memory_optimization',
                        'priority': 'medium',
                        'title': 'High Memory Usage Detected',
                        'description': f'Memory usage is {memory_gb:.1f}GB. Consider memory optimization strategies.',
                        'expected_improvement': 30.0,
                        'implementation_steps': [
                            'Implement data streaming instead of batch loading',
                            'Add memory-efficient data structures',
                            'Implement garbage collection optimization',
                            'Consider data partitioning'
                        ]
                    })
            
            # Performance optimization based on duration
            if metrics.duration_seconds and metrics.duration_seconds > 3600:  # More than 1 hour
                recommendations.append({
                    'type': 'performance_optimization',
                    'priority': 'high',
                    'title': 'Long Execution Time Detected',
                    'description': f'Pipeline took {metrics.duration_seconds/60:.1f} minutes to complete.',
                    'expected_improvement': 40.0,
                    'implementation_steps': [
                        'Implement parallel processing',
                        'Add data indexing for faster queries',
                        'Optimize transformation logic',
                        'Consider pipeline segmentation'
                    ]
                })
            
            # Error rate optimization
            if metrics.error_rate and metrics.error_rate > 0.01:
                recommendations.append({
                    'type': 'reliability_optimization',
                    'priority': 'critical',
                    'title': 'High Error Rate Detected',
                    'description': f'Error rate is {metrics.error_rate*100:.2f}%. Reliability improvements needed.',
                    'expected_improvement': 50.0,
                    'implementation_steps': [
                        'Implement robust error handling',
                        'Add data validation checks',
                        'Implement retry mechanisms',
                        'Add comprehensive logging'
                    ]
                })
            
            # Save recommendations to database
            for rec in recommendations:
                db_rec = OptimizationRecommendation(
                    pipeline_id=metrics.pipeline_id,
                    recommendation_type=rec['type'],
                    priority=rec['priority'],
                    title=rec['title'],
                    description=rec['description'],
                    expected_improvement=rec['expected_improvement'],
                    implementation_steps=rec['implementation_steps']
                )
                db.add(db_rec)
            
            db.commit()
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return []
    
    async def get_performance_dashboard_data(self, pipeline_id: Optional[str] = None, 
                                           time_range_hours: int = 24) -> Dict[str, Any]:
        """Get performance dashboard data for monitoring interface."""
        try:
            with SessionLocal() as db:
                # Calculate time range
                start_time = datetime.utcnow() - timedelta(hours=time_range_hours)
                
                # Base query
                query = db.query(PerformanceMetrics).filter(
                    PerformanceMetrics.start_time >= start_time
                )
                
                if pipeline_id:
                    query = query.filter(PerformanceMetrics.pipeline_id == pipeline_id)
                
                metrics = query.all()
                
                # Calculate aggregated statistics
                total_executions = len(metrics)
                avg_duration = sum(m.duration_seconds or 0 for m in metrics) / max(total_executions, 1)
                avg_cpu = sum(m.cpu_usage_percent or 0 for m in metrics) / max(total_executions, 1)
                avg_memory = sum(m.memory_usage_mb or 0 for m in metrics) / max(total_executions, 1)
                total_records = sum(m.records_processed or 0 for m in metrics)
                total_errors = sum(m.error_count or 0 for m in metrics)
                
                # Get recent SLA violations
                violations = db.query(SLAViolation).join(PerformanceMetrics).filter(
                    PerformanceMetrics.start_time >= start_time
                ).all()
                
                # Get active alerts
                active_alerts = db.query(PerformanceAlert).join(SLAViolation).join(PerformanceMetrics).filter(
                    and_(
                        PerformanceMetrics.start_time >= start_time,
                        PerformanceAlert.acknowledged == False
                    )
                ).all()
                
                return {
                    "summary": {
                        "total_executions": total_executions,
                        "avg_duration_seconds": round(avg_duration, 2),
                        "avg_cpu_usage": round(avg_cpu, 2),
                        "avg_memory_usage_mb": round(avg_memory, 2),
                        "total_records_processed": total_records,
                        "total_errors": total_errors,
                        "error_rate": round(total_errors / max(total_records, 1), 4)
                    },
                    "violations": [
                        {
                            "id": v.id,
                            "sla_type": v.sla_type,
                            "severity": v.violation_severity,
                            "threshold": v.sla_threshold,
                            "actual_value": v.actual_value,
                            "start_time": v.violation_start,
                            "is_resolved": v.is_resolved
                        } for v in violations
                    ],
                    "active_alerts": [
                        {
                            "id": a.id,
                            "type": a.alert_type,
                            "level": a.alert_level,
                            "message": a.alert_message,
                            "created_at": a.created_at
                        } for a in active_alerts
                    ],
                    "time_range_hours": time_range_hours,
                    "generated_at": datetime.utcnow()
                }
                
        except Exception as e:
            logger.error(f"Error getting dashboard data: {str(e)}")
            raise
    
    async def apply_auto_tuning(self, pipeline_id: str) -> Dict[str, Any]:
        """Apply automated performance tuning based on historical data."""
        try:
            with SessionLocal() as db:
                # Get tuning configuration
                config = db.query(PerformanceTuningConfig).filter(
                    PerformanceTuningConfig.pipeline_id == pipeline_id,
                    PerformanceTuningConfig.is_active == True
                ).first()
                
                if not config:
                    return {"status": "no_config", "message": "No active tuning configuration found"}
                
                # Get recent performance data
                recent_metrics = db.query(PerformanceMetrics).filter(
                    PerformanceMetrics.pipeline_id == pipeline_id,
                    PerformanceMetrics.start_time >= datetime.utcnow() - timedelta(hours=24)
                ).all()
                
                if not recent_metrics:
                    return {"status": "no_data", "message": "No recent performance data available"}
                
                # Calculate performance trends
                avg_cpu = sum(m.cpu_usage_percent or 0 for m in recent_metrics) / len(recent_metrics)
                avg_memory = sum(m.memory_usage_mb or 0 for m in recent_metrics) / len(recent_metrics)
                avg_duration = sum(m.duration_seconds or 0 for m in recent_metrics) / len(recent_metrics)
                
                tuning_actions = []
                
                # Auto-scaling decisions
                if config.auto_scaling_enabled:
                    if avg_cpu > config.target_cpu_utilization:
                        # Scale up recommendation
                        tuning_actions.append({
                            "action": "scale_up",
                            "reason": f"CPU usage ({avg_cpu:.1f}%) exceeds target ({config.target_cpu_utilization}%)",
                            "recommendation": "Increase instance count or upgrade instance type"
                        })
                    elif avg_cpu < config.target_cpu_utilization * 0.5:
                        # Scale down recommendation
                        tuning_actions.append({
                            "action": "scale_down",
                            "reason": f"CPU usage ({avg_cpu:.1f}%) is well below target",
                            "recommendation": "Consider reducing instance count to optimize costs"
                        })
                
                # Performance optimization
                if avg_duration > config.latency_threshold_ms / 1000:
                    tuning_actions.append({
                        "action": "optimize_performance",
                        "reason": f"Average duration ({avg_duration:.1f}s) exceeds threshold",
                        "recommendation": "Implement parallel processing or optimize algorithms"
                    })
                
                # Update last tuned timestamp
                config.last_tuned = datetime.utcnow()
                db.commit()
                
                return {
                    "status": "tuning_applied",
                    "pipeline_id": pipeline_id,
                    "actions": tuning_actions,
                    "performance_summary": {
                        "avg_cpu": avg_cpu,
                        "avg_memory_mb": avg_memory,
                        "avg_duration_seconds": avg_duration
                    },
                    "tuned_at": config.last_tuned
                }
                
        except Exception as e:
            logger.error(f"Error applying auto-tuning: {str(e)}")
            raise