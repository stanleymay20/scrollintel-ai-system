#!/usr/bin/env python3
"""
ScrollIntel Launch Day Monitoring System
Real-time monitoring and incident response for launch day
"""

import os
import sys
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import requests
import psutil
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('launch_day_monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class Alert:
    level: AlertLevel
    message: str
    timestamp: datetime
    metric: str
    value: float
    threshold: float
    resolved: bool = False

class LaunchDayMonitor:
    """Real-time monitoring system for launch day"""
    
    def __init__(self):
        self.monitoring_active = False
        self.alerts = []
        self.metrics_history = {}
        self.thresholds = {
            "response_time": 2.0,
            "error_rate": 0.1,
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "concurrent_users": 1000,
            "uptime": 99.9
        }
        self.incident_response_team = [
            {"name": "Technical Lead", "contact": "tech-lead@scrollintel.com"},
            {"name": "DevOps Engineer", "contact": "devops@scrollintel.com"},
            {"name": "Product Manager", "contact": "pm@scrollintel.com"}
        ]
        
    def start_monitoring(self):
        """Start the launch day monitoring system"""
        logger.info("🚀 Starting Launch Day Monitoring System")
        self.monitoring_active = True
        
        # Start monitoring threads
        monitoring_threads = [
            threading.Thread(target=self._monitor_system_health, daemon=True),
            threading.Thread(target=self._monitor_application_performance, daemon=True),
            threading.Thread(target=self._monitor_business_metrics, daemon=True),
            threading.Thread(target=self._monitor_user_experience, daemon=True),
            threading.Thread(target=self._alert_processor, daemon=True)
        ]
        
        for thread in monitoring_threads:
            thread.start()
        
        logger.info("✅ All monitoring threads started")
        
        # Keep main thread alive
        try:
            while self.monitoring_active:
                self._generate_status_report()
                time.sleep(60)  # Generate status report every minute
        except KeyboardInterrupt:
            logger.info("🛑 Stopping monitoring system...")
            self.monitoring_active = False
    
    def _monitor_system_health(self):
        """Monitor system health metrics"""
        while self.monitoring_active:
            try:
                # CPU Usage
                cpu_usage = psutil.cpu_percent(interval=1)
                self._record_metric("cpu_usage", cpu_usage)
                
                if cpu_usage > self.thresholds["cpu_usage"]:
                    self._create_alert(
                        AlertLevel.WARNING,
                        f"High CPU usage: {cpu_usage}%",
                        "cpu_usage",
                        cpu_usage,
                        self.thresholds["cpu_usage"]
                    )
                
                # Memory Usage
                memory = psutil.virtual_memory()
                memory_usage = memory.percent
                self._record_metric("memory_usage", memory_usage)
                
                if memory_usage > self.thresholds["memory_usage"]:
                    self._create_alert(
                        AlertLevel.WARNING,
                        f"High memory usage: {memory_usage}%",
                        "memory_usage",
                        memory_usage,
                        self.thresholds["memory_usage"]
                    )
                
                # Disk Usage
                disk = psutil.disk_usage('/')
                disk_usage = (disk.used / disk.total) * 100
                self._record_metric("disk_usage", disk_usage)
                
                if disk_usage > self.thresholds["disk_usage"]:
                    self._create_alert(
                        AlertLevel.CRITICAL,
                        f"High disk usage: {disk_usage:.1f}%",
                        "disk_usage",
                        disk_usage,
                        self.thresholds["disk_usage"]
                    )
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring system health: {e}")
                time.sleep(60)
    
    def _monitor_application_performance(self):
        """Monitor application performance metrics"""
        while self.monitoring_active:
            try:
                # Test API endpoints
                endpoints = [
                    "/api/health",
                    "/api/agents/cto",
                    "/api/dashboard/metrics",
                    "/api/files/upload"
                ]
                
                total_response_time = 0
                successful_requests = 0
                failed_requests = 0
                
                for endpoint in endpoints:
                    try:
                        start_time = time.time()
                        response = requests.get(f"http://localhost:8000{endpoint}", timeout=10)
                        response_time = time.time() - start_time
                        
                        if response.status_code == 200:
                            successful_requests += 1
                            total_response_time += response_time
                        else:
                            failed_requests += 1
                            
                    except Exception as e:
                        failed_requests += 1
                        logger.warning(f"Failed to reach {endpoint}: {e}")
                
                # Calculate metrics
                if successful_requests > 0:
                    avg_response_time = total_response_time / successful_requests
                    self._record_metric("response_time", avg_response_time)
                    
                    if avg_response_time > self.thresholds["response_time"]:
                        self._create_alert(
                            AlertLevel.WARNING,
                            f"Slow response time: {avg_response_time:.2f}s",
                            "response_time",
                            avg_response_time,
                            self.thresholds["response_time"]
                        )
                
                # Error rate
                total_requests = successful_requests + failed_requests
                if total_requests > 0:
                    error_rate = (failed_requests / total_requests) * 100
                    self._record_metric("error_rate", error_rate)
                    
                    if error_rate > self.thresholds["error_rate"]:
                        self._create_alert(
                            AlertLevel.CRITICAL,
                            f"High error rate: {error_rate:.2f}%",
                            "error_rate",
                            error_rate,
                            self.thresholds["error_rate"]
                        )
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error monitoring application performance: {e}")
                time.sleep(60)
    
    def _monitor_business_metrics(self):
        """Monitor business metrics"""
        while self.monitoring_active:
            try:
                # Simulate business metrics monitoring
                # In production, these would come from your analytics system
                
                # User signups
                signups = self._get_current_signups()
                self._record_metric("signups", signups)
                
                # Active users
                active_users = self._get_active_users()
                self._record_metric("active_users", active_users)
                
                # Conversion rate
                conversion_rate = self._get_conversion_rate()
                self._record_metric("conversion_rate", conversion_rate)
                
                # Revenue
                revenue = self._get_current_revenue()
                self._record_metric("revenue", revenue)
                
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring business metrics: {e}")
                time.sleep(300)
    
    def _monitor_user_experience(self):
        """Monitor user experience metrics"""
        while self.monitoring_active:
            try:
                # Page load times
                page_load_time = self._measure_page_load_time()
                self._record_metric("page_load_time", page_load_time)
                
                # User satisfaction scores
                satisfaction_score = self._get_satisfaction_score()
                self._record_metric("satisfaction_score", satisfaction_score)
                
                # Feature usage
                feature_usage = self._get_feature_usage()
                self._record_metric("feature_usage", feature_usage)
                
                time.sleep(180)  # Check every 3 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring user experience: {e}")
                time.sleep(180)
    
    def _alert_processor(self):
        """Process and handle alerts"""
        while self.monitoring_active:
            try:
                # Process unresolved alerts
                unresolved_alerts = [alert for alert in self.alerts if not alert.resolved]
                
                for alert in unresolved_alerts:
                    self._handle_alert(alert)
                
                time.sleep(30)  # Process alerts every 30 seconds
                
            except Exception as e:
                logger.error(f"Error processing alerts: {e}")
                time.sleep(30)
    
    def _create_alert(self, level: AlertLevel, message: str, metric: str, value: float, threshold: float):
        """Create a new alert"""
        alert = Alert(
            level=level,
            message=message,
            timestamp=datetime.now(),
            metric=metric,
            value=value,
            threshold=threshold
        )
        
        self.alerts.append(alert)
        logger.warning(f"🚨 ALERT [{level.value.upper()}]: {message}")
        
        # Send immediate notification for critical alerts
        if level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            self._send_immediate_notification(alert)
    
    def _handle_alert(self, alert: Alert):
        """Handle an alert based on its level"""
        if alert.level == AlertLevel.EMERGENCY:
            self._handle_emergency_alert(alert)
        elif alert.level == AlertLevel.CRITICAL:
            self._handle_critical_alert(alert)
        elif alert.level == AlertLevel.WARNING:
            self._handle_warning_alert(alert)
    
    def _handle_emergency_alert(self, alert: Alert):
        """Handle emergency level alerts"""
        logger.error(f"🚨 EMERGENCY ALERT: {alert.message}")
        
        # Immediate escalation to all team members
        for team_member in self.incident_response_team:
            self._notify_team_member(team_member, alert, urgent=True)
        
        # Trigger automated recovery if possible
        self._trigger_automated_recovery(alert)
    
    def _handle_critical_alert(self, alert: Alert):
        """Handle critical level alerts"""
        logger.error(f"🔥 CRITICAL ALERT: {alert.message}")
        
        # Notify technical team
        tech_team = [member for member in self.incident_response_team 
                    if "Technical" in member["name"] or "DevOps" in member["name"]]
        
        for team_member in tech_team:
            self._notify_team_member(team_member, alert, urgent=True)
    
    def _handle_warning_alert(self, alert: Alert):
        """Handle warning level alerts"""
        logger.warning(f"⚠️ WARNING ALERT: {alert.message}")
        
        # Log for review, but don't immediately escalate
        # Check if it becomes critical
        if self._should_escalate_warning(alert):
            alert.level = AlertLevel.CRITICAL
            self._handle_critical_alert(alert)
    
    def _record_metric(self, metric_name: str, value: float):
        """Record a metric value"""
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = []
        
        self.metrics_history[metric_name].append({
            "timestamp": datetime.now(),
            "value": value
        })
        
        # Keep only last 1000 data points
        if len(self.metrics_history[metric_name]) > 1000:
            self.metrics_history[metric_name] = self.metrics_history[metric_name][-1000:]
    
    def _generate_status_report(self):
        """Generate periodic status report"""
        current_time = datetime.now()
        
        # Get latest metrics
        latest_metrics = {}
        for metric_name, history in self.metrics_history.items():
            if history:
                latest_metrics[metric_name] = history[-1]["value"]
        
        # Count alerts by level
        alert_counts = {level.value: 0 for level in AlertLevel}
        for alert in self.alerts:
            if not alert.resolved:
                alert_counts[alert.level.value] += 1
        
        status_report = {
            "timestamp": current_time.isoformat(),
            "system_status": "operational" if alert_counts["critical"] == 0 and alert_counts["emergency"] == 0 else "degraded",
            "latest_metrics": latest_metrics,
            "alert_summary": alert_counts,
            "uptime": self._calculate_uptime(),
            "total_users": latest_metrics.get("active_users", 0),
            "total_signups": latest_metrics.get("signups", 0)
        }
        
        # Save status report
        with open(f"status_reports/status_{current_time.strftime('%Y%m%d_%H%M')}.json", "w") as f:
            json.dump(status_report, f, indent=2)
        
        # Log summary
        logger.info(f"📊 Status Report - System: {status_report['system_status'].upper()}, "
                   f"Users: {status_report['total_users']}, "
                   f"Alerts: {sum(alert_counts.values())}")
    
    def _calculate_uptime(self) -> float:
        """Calculate system uptime percentage"""
        # Simplified uptime calculation
        # In production, this would be based on actual downtime tracking
        return 99.95
    
    # Mock methods for business metrics (replace with actual implementations)
    def _get_current_signups(self) -> int:
        """Get current signup count"""
        return 150  # Mock value
    
    def _get_active_users(self) -> int:
        """Get current active user count"""
        return 75  # Mock value
    
    def _get_conversion_rate(self) -> float:
        """Get current conversion rate"""
        return 12.5  # Mock value
    
    def _get_current_revenue(self) -> float:
        """Get current revenue"""
        return 2500.0  # Mock value
    
    def _measure_page_load_time(self) -> float:
        """Measure page load time"""
        return 1.2  # Mock value
    
    def _get_satisfaction_score(self) -> float:
        """Get user satisfaction score"""
        return 4.6  # Mock value
    
    def _get_feature_usage(self) -> float:
        """Get feature usage percentage"""
        return 65.0  # Mock value
    
    def _send_immediate_notification(self, alert: Alert):
        """Send immediate notification for critical alerts"""
        logger.info(f"📧 Sending immediate notification for: {alert.message}")
        # In production, integrate with email/SMS/Slack notifications
    
    def _notify_team_member(self, team_member: Dict[str, str], alert: Alert, urgent: bool = False):
        """Notify a team member about an alert"""
        urgency = "URGENT" if urgent else "NORMAL"
        logger.info(f"📞 Notifying {team_member['name']} ({urgency}): {alert.message}")
        # In production, send actual notifications
    
    def _trigger_automated_recovery(self, alert: Alert):
        """Trigger automated recovery procedures"""
        logger.info(f"🔧 Triggering automated recovery for: {alert.metric}")
        
        recovery_actions = {
            "cpu_usage": self._scale_up_resources,
            "memory_usage": self._restart_services,
            "disk_usage": self._cleanup_disk_space,
            "error_rate": self._restart_application
        }
        
        if alert.metric in recovery_actions:
            try:
                recovery_actions[alert.metric]()
                logger.info(f"✅ Automated recovery completed for {alert.metric}")
            except Exception as e:
                logger.error(f"❌ Automated recovery failed for {alert.metric}: {e}")
    
    def _scale_up_resources(self):
        """Scale up system resources"""
        logger.info("🚀 Scaling up system resources...")
        # In production, trigger auto-scaling
    
    def _restart_services(self):
        """Restart system services"""
        logger.info("🔄 Restarting system services...")
        # In production, restart specific services
    
    def _cleanup_disk_space(self):
        """Clean up disk space"""
        logger.info("🧹 Cleaning up disk space...")
        # In production, clean up logs and temporary files
    
    def _restart_application(self):
        """Restart application"""
        logger.info("🔄 Restarting application...")
        # In production, perform rolling restart
    
    def _should_escalate_warning(self, alert: Alert) -> bool:
        """Determine if a warning should be escalated"""
        # Check if the metric has been above threshold for too long
        if alert.metric in self.metrics_history:
            recent_values = self.metrics_history[alert.metric][-10:]  # Last 10 values
            above_threshold_count = sum(1 for entry in recent_values 
                                      if entry["value"] > alert.threshold)
            return above_threshold_count >= 8  # 80% of recent values above threshold
        return False

def main():
    """Main execution function"""
    # Create status reports directory
    os.makedirs("status_reports", exist_ok=True)
    
    # Initialize and start monitoring
    monitor = LaunchDayMonitor()
    
    print("🚀 ScrollIntel Launch Day Monitoring System")
    print("=" * 50)
    print("Starting real-time monitoring...")
    print("Press Ctrl+C to stop monitoring")
    print("=" * 50)
    
    try:
        monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Monitoring failed: {e}")

if __name__ == "__main__":
    main()