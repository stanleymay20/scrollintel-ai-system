#!/usr/bin/env python3
"""
ScrollIntel Launch Metrics Dashboard
Real-time monitoring and tracking of launch success metrics
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import threading
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricStatus(Enum):
    BELOW_TARGET = "below_target"
    ON_TARGET = "on_target"
    ABOVE_TARGET = "above_target"
    CRITICAL = "critical"

@dataclass
class Metric:
    name: str
    current_value: float
    target_value: float
    unit: str
    status: MetricStatus
    trend: str
    last_updated: datetime

class LaunchMetricsDashboard:
    """Real-time dashboard for monitoring launch success metrics"""
    
    def __init__(self):
        self.launch_date = datetime(2025, 8, 22)
        self.monitoring_active = False
        self.metrics = {}
        self.success_criteria = self._define_success_criteria()
        self.alerts = []
        
    def _define_success_criteria(self) -> Dict[str, Dict[str, Any]]:
        """Define success criteria for launch metrics"""
        return {
            "technical_metrics": {
                "system_uptime": {"target": 99.9, "unit": "%", "critical_threshold": 99.0},
                "response_time": {"target": 2.0, "unit": "seconds", "critical_threshold": 5.0},
                "file_processing_time": {"target": 30.0, "unit": "seconds", "critical_threshold": 60.0},
                "concurrent_users": {"target": 100, "unit": "users", "critical_threshold": 50},
                "error_rate": {"target": 0.1, "unit": "%", "critical_threshold": 1.0}
            },
            "user_experience_metrics": {
                "onboarding_completion_rate": {"target": 80.0, "unit": "%", "critical_threshold": 50.0},
                "user_activation_rate": {"target": 60.0, "unit": "%", "critical_threshold": 30.0},
                "user_satisfaction_score": {"target": 4.5, "unit": "/5", "critical_threshold": 3.0},
                "support_ticket_resolution_time": {"target": 24.0, "unit": "hours", "critical_threshold": 48.0},
                "feature_adoption_rate": {"target": 50.0, "unit": "%", "critical_threshold": 25.0}
            },
            "business_metrics": {
                "launch_day_signups": {"target": 100, "unit": "signups", "critical_threshold": 25},
                "week_1_paying_customers": {"target": 10, "unit": "customers", "critical_threshold": 3},
                "month_1_revenue": {"target": 1000.0, "unit": "USD", "critical_threshold": 250.0},
                "customer_acquisition_cost": {"target": 100.0, "unit": "USD", "critical_threshold": 200.0},
                "monthly_recurring_revenue_growth": {"target": 20.0, "unit": "%", "critical_threshold": 5.0}
            }
        }
    
    def start_monitoring(self):
        """Start real-time metrics monitoring"""
        logger.info("📊 Starting Launch Metrics Dashboard")
        self.monitoring_active = True
        
        # Initialize metrics
        self._initialize_metrics()
        
        # Start monitoring threads
        monitoring_threads = [
            threading.Thread(target=self._monitor_technical_metrics, daemon=True),
            threading.Thread(target=self._monitor_user_experience_metrics, daemon=True),
            threading.Thread(target=self._monitor_business_metrics, daemon=True),
            threading.Thread(target=self._generate_periodic_reports, daemon=True),
            threading.Thread(target=self._check_success_criteria, daemon=True)
        ]
        
        for thread in monitoring_threads:
            thread.start()
        
        logger.info("✅ Launch metrics monitoring started")
        
        # Keep main thread alive and display dashboard
        try:
            while self.monitoring_active:
                self._display_dashboard()
                time.sleep(30)  # Update dashboard every 30 seconds
        except KeyboardInterrupt:
            logger.info("🛑 Stopping metrics monitoring...")
            self.monitoring_active = False
    
    def _initialize_metrics(self):
        """Initialize all metrics with default values"""
        for category, metrics in self.success_criteria.items():
            for metric_name, config in metrics.items():
                self.metrics[metric_name] = Metric(
                    name=metric_name,
                    current_value=0.0,
                    target_value=config["target"],
                    unit=config["unit"],
                    status=MetricStatus.BELOW_TARGET,
                    trend="stable",
                    last_updated=datetime.now()
                )
    
    def _monitor_technical_metrics(self):
        """Monitor technical performance metrics"""
        while self.monitoring_active:
            try:
                # System Uptime
                uptime = self._calculate_system_uptime()
                self._update_metric("system_uptime", uptime)
                
                # Response Time
                response_time = self._measure_average_response_time()
                self._update_metric("response_time", response_time)
                
                # File Processing Time
                file_processing_time = self._measure_file_processing_time()
                self._update_metric("file_processing_time", file_processing_time)
                
                # Concurrent Users
                concurrent_users = self._count_concurrent_users()
                self._update_metric("concurrent_users", concurrent_users)
                
                # Error Rate
                error_rate = self._calculate_error_rate()
                self._update_metric("error_rate", error_rate)
                
                time.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error monitoring technical metrics: {e}")
                time.sleep(60)
    
    def _monitor_user_experience_metrics(self):
        """Monitor user experience metrics"""
        while self.monitoring_active:
            try:
                # Onboarding Completion Rate
                onboarding_rate = self._calculate_onboarding_completion_rate()
                self._update_metric("onboarding_completion_rate", onboarding_rate)
                
                # User Activation Rate
                activation_rate = self._calculate_user_activation_rate()
                self._update_metric("user_activation_rate", activation_rate)
                
                # User Satisfaction Score
                satisfaction_score = self._get_user_satisfaction_score()
                self._update_metric("user_satisfaction_score", satisfaction_score)
                
                # Support Ticket Resolution Time
                resolution_time = self._calculate_support_resolution_time()
                self._update_metric("support_ticket_resolution_time", resolution_time)
                
                # Feature Adoption Rate
                adoption_rate = self._calculate_feature_adoption_rate()
                self._update_metric("feature_adoption_rate", adoption_rate)
                
                time.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring user experience metrics: {e}")
                time.sleep(300)
    
    def _monitor_business_metrics(self):
        """Monitor business performance metrics"""
        while self.monitoring_active:
            try:
                # Launch Day Signups
                if self._is_launch_day():
                    signups = self._count_launch_day_signups()
                    self._update_metric("launch_day_signups", signups)
                
                # Week 1 Paying Customers
                if self._is_within_week_1():
                    paying_customers = self._count_paying_customers()
                    self._update_metric("week_1_paying_customers", paying_customers)
                
                # Month 1 Revenue
                if self._is_within_month_1():
                    revenue = self._calculate_month_1_revenue()
                    self._update_metric("month_1_revenue", revenue)
                
                # Customer Acquisition Cost
                cac = self._calculate_customer_acquisition_cost()
                self._update_metric("customer_acquisition_cost", cac)
                
                # MRR Growth
                mrr_growth = self._calculate_mrr_growth()
                self._update_metric("monthly_recurring_revenue_growth", mrr_growth)
                
                time.sleep(600)  # Update every 10 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring business metrics: {e}")
                time.sleep(600)
    
    def _update_metric(self, metric_name: str, value: float):
        """Update a metric with new value and status"""
        if metric_name not in self.metrics:
            return
        
        metric = self.metrics[metric_name]
        old_value = metric.current_value
        metric.current_value = value
        metric.last_updated = datetime.now()
        
        # Calculate trend
        if value > old_value:
            metric.trend = "increasing"
        elif value < old_value:
            metric.trend = "decreasing"
        else:
            metric.trend = "stable"
        
        # Update status
        metric.status = self._calculate_metric_status(metric_name, value)
        
        # Check for alerts
        self._check_metric_alerts(metric)
    
    def _calculate_metric_status(self, metric_name: str, value: float) -> MetricStatus:
        """Calculate status based on metric value and targets"""
        # Find the metric configuration
        config = None
        for category, metrics in self.success_criteria.items():
            if metric_name in metrics:
                config = metrics[metric_name]
                break
        
        if not config:
            return MetricStatus.BELOW_TARGET
        
        target = config["target"]
        critical_threshold = config["critical_threshold"]
        
        # For metrics where lower is better (like error_rate, response_time)
        if metric_name in ["error_rate", "response_time", "file_processing_time", 
                          "support_ticket_resolution_time", "customer_acquisition_cost"]:
            if value >= critical_threshold:
                return MetricStatus.CRITICAL
            elif value <= target:
                return MetricStatus.ABOVE_TARGET
            else:
                return MetricStatus.ON_TARGET
        
        # For metrics where higher is better
        else:
            if value <= critical_threshold:
                return MetricStatus.CRITICAL
            elif value >= target:
                return MetricStatus.ABOVE_TARGET
            else:
                return MetricStatus.ON_TARGET
    
    def _check_metric_alerts(self, metric: Metric):
        """Check if metric requires alerting"""
        if metric.status == MetricStatus.CRITICAL:
            alert = {
                "timestamp": datetime.now().isoformat(),
                "metric": metric.name,
                "value": metric.current_value,
                "target": metric.target_value,
                "status": "CRITICAL",
                "message": f"{metric.name} is critically below target: {metric.current_value} {metric.unit}"
            }
            self.alerts.append(alert)
            logger.error(f"🚨 CRITICAL ALERT: {alert['message']}")
    
    def _display_dashboard(self):
        """Display real-time metrics dashboard"""
        os.system('clear' if os.name == 'posix' else 'cls')  # Clear screen
        
        print("🚀 SCROLLINTEL LAUNCH METRICS DASHBOARD")
        print("=" * 80)
        print(f"Launch Date: {self.launch_date.strftime('%Y-%m-%d')}")
        print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Days Since Launch: {(datetime.now() - self.launch_date).days}")
        print("=" * 80)
        
        # Technical Metrics
        print("\n📊 TECHNICAL METRICS")
        print("-" * 40)
        self._display_metric_category("technical_metrics")
        
        # User Experience Metrics
        print("\n👥 USER EXPERIENCE METRICS")
        print("-" * 40)
        self._display_metric_category("user_experience_metrics")
        
        # Business Metrics
        print("\n💰 BUSINESS METRICS")
        print("-" * 40)
        self._display_metric_category("business_metrics")
        
        # Recent Alerts
        if self.alerts:
            print("\n🚨 RECENT ALERTS")
            print("-" * 40)
            recent_alerts = sorted(self.alerts, key=lambda x: x["timestamp"], reverse=True)[:5]
            for alert in recent_alerts:
                print(f"⚠️  {alert['message']}")
        
        # Overall Status
        print("\n🎯 OVERALL LAUNCH STATUS")
        print("-" * 40)
        overall_status = self._calculate_overall_status()
        status_emoji = "✅" if overall_status == "SUCCESS" else "⚠️" if overall_status == "WARNING" else "❌"
        print(f"{status_emoji} Status: {overall_status}")
        
        success_rate = self._calculate_success_rate()
        print(f"📈 Success Rate: {success_rate:.1f}%")
        
        print("=" * 80)
    
    def _display_metric_category(self, category: str):
        """Display metrics for a specific category"""
        category_metrics = self.success_criteria[category]
        
        for metric_name in category_metrics.keys():
            if metric_name in self.metrics:
                metric = self.metrics[metric_name]
                status_emoji = self._get_status_emoji(metric.status)
                trend_emoji = self._get_trend_emoji(metric.trend)
                
                print(f"{status_emoji} {metric.name.replace('_', ' ').title()}: "
                      f"{metric.current_value} {metric.unit} "
                      f"(Target: {metric.target_value} {metric.unit}) {trend_emoji}")
    
    def _get_status_emoji(self, status: MetricStatus) -> str:
        """Get emoji for metric status"""
        emoji_map = {
            MetricStatus.ABOVE_TARGET: "✅",
            MetricStatus.ON_TARGET: "🟡",
            MetricStatus.BELOW_TARGET: "🟠",
            MetricStatus.CRITICAL: "❌"
        }
        return emoji_map.get(status, "❓")
    
    def _get_trend_emoji(self, trend: str) -> str:
        """Get emoji for metric trend"""
        trend_map = {
            "increasing": "📈",
            "decreasing": "📉",
            "stable": "➡️"
        }
        return trend_map.get(trend, "")
    
    def _calculate_overall_status(self) -> str:
        """Calculate overall launch status"""
        critical_count = sum(1 for metric in self.metrics.values() 
                           if metric.status == MetricStatus.CRITICAL)
        
        if critical_count > 0:
            return "CRITICAL"
        
        success_rate = self._calculate_success_rate()
        
        if success_rate >= 80:
            return "SUCCESS"
        elif success_rate >= 60:
            return "WARNING"
        else:
            return "NEEDS_ATTENTION"
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate"""
        if not self.metrics:
            return 0.0
        
        successful_metrics = sum(1 for metric in self.metrics.values() 
                               if metric.status in [MetricStatus.ABOVE_TARGET, MetricStatus.ON_TARGET])
        
        return (successful_metrics / len(self.metrics)) * 100
    
    def _generate_periodic_reports(self):
        """Generate periodic reports"""
        while self.monitoring_active:
            try:
                # Generate hourly report
                report = self._generate_metrics_report()
                
                # Save report
                timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                with open(f"launch_metrics/reports/metrics_report_{timestamp}.json", "w") as f:
                    json.dump(report, f, indent=2)
                
                time.sleep(3600)  # Generate report every hour
                
            except Exception as e:
                logger.error(f"Error generating periodic reports: {e}")
                time.sleep(3600)
    
    def _generate_metrics_report(self) -> Dict[str, Any]:
        """Generate comprehensive metrics report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "launch_date": self.launch_date.isoformat(),
            "days_since_launch": (datetime.now() - self.launch_date).days,
            "overall_status": self._calculate_overall_status(),
            "success_rate": self._calculate_success_rate(),
            "metrics": {},
            "alerts": self.alerts[-10:],  # Last 10 alerts
            "summary": {
                "total_metrics": len(self.metrics),
                "successful_metrics": sum(1 for m in self.metrics.values() 
                                        if m.status in [MetricStatus.ABOVE_TARGET, MetricStatus.ON_TARGET]),
                "critical_metrics": sum(1 for m in self.metrics.values() 
                                      if m.status == MetricStatus.CRITICAL)
            }
        }
        
        # Add detailed metrics
        for metric_name, metric in self.metrics.items():
            report["metrics"][metric_name] = {
                "current_value": metric.current_value,
                "target_value": metric.target_value,
                "unit": metric.unit,
                "status": metric.status.value,
                "trend": metric.trend,
                "last_updated": metric.last_updated.isoformat()
            }
        
        return report
    
    def _check_success_criteria(self):
        """Continuously check if success criteria are being met"""
        while self.monitoring_active:
            try:
                success_rate = self._calculate_success_rate()
                
                if success_rate >= 90:
                    logger.info(f"🎉 Excellent performance! Success rate: {success_rate:.1f}%")
                elif success_rate >= 70:
                    logger.info(f"✅ Good performance! Success rate: {success_rate:.1f}%")
                elif success_rate >= 50:
                    logger.warning(f"⚠️ Moderate performance. Success rate: {success_rate:.1f}%")
                else:
                    logger.error(f"❌ Poor performance! Success rate: {success_rate:.1f}%")
                
                time.sleep(1800)  # Check every 30 minutes
                
            except Exception as e:
                logger.error(f"Error checking success criteria: {e}")
                time.sleep(1800)
    
    # Mock data methods (replace with actual data collection in production)
    def _calculate_system_uptime(self) -> float:
        return 99.95  # Mock uptime
    
    def _measure_average_response_time(self) -> float:
        return 1.2  # Mock response time
    
    def _measure_file_processing_time(self) -> float:
        return 25.0  # Mock processing time
    
    def _count_concurrent_users(self) -> int:
        return 150  # Mock concurrent users
    
    def _calculate_error_rate(self) -> float:
        return 0.05  # Mock error rate
    
    def _calculate_onboarding_completion_rate(self) -> float:
        return 85.0  # Mock completion rate
    
    def _calculate_user_activation_rate(self) -> float:
        return 65.0  # Mock activation rate
    
    def _get_user_satisfaction_score(self) -> float:
        return 4.6  # Mock satisfaction score
    
    def _calculate_support_resolution_time(self) -> float:
        return 18.0  # Mock resolution time in hours
    
    def _calculate_feature_adoption_rate(self) -> float:
        return 55.0  # Mock adoption rate
    
    def _count_launch_day_signups(self) -> int:
        return 125  # Mock signups
    
    def _count_paying_customers(self) -> int:
        return 15  # Mock paying customers
    
    def _calculate_month_1_revenue(self) -> float:
        return 1250.0  # Mock revenue
    
    def _calculate_customer_acquisition_cost(self) -> float:
        return 85.0  # Mock CAC
    
    def _calculate_mrr_growth(self) -> float:
        return 25.0  # Mock MRR growth
    
    def _is_launch_day(self) -> bool:
        return datetime.now().date() == self.launch_date.date()
    
    def _is_within_week_1(self) -> bool:
        return (datetime.now() - self.launch_date).days <= 7
    
    def _is_within_month_1(self) -> bool:
        return (datetime.now() - self.launch_date).days <= 30

def main():
    """Main execution function"""
    # Create reports directory
    os.makedirs("launch_metrics/reports", exist_ok=True)
    
    # Initialize and start dashboard
    dashboard = LaunchMetricsDashboard()
    
    print("📊 ScrollIntel Launch Metrics Dashboard")
    print("=" * 50)
    print("Starting real-time metrics monitoring...")
    print("Press Ctrl+C to stop monitoring")
    print("=" * 50)
    
    try:
        dashboard.start_monitoring()
    except KeyboardInterrupt:
        print("\n🛑 Metrics monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Metrics monitoring failed: {e}")

if __name__ == "__main__":
    main()