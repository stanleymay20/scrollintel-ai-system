"""
Comprehensive Reporting System for Production Monitoring

This module provides comprehensive reporting capabilities for continuous improvement,
including automated report generation, trend analysis, and actionable insights.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
from collections import defaultdict, Counter
import uuid
import base64
from io import BytesIO

logger = logging.getLogger(__name__)

class ReportType(Enum):
    SYSTEM_HEALTH = "system_health"
    USER_EXPERIENCE = "user_experience"
    FAILURE_ANALYSIS = "failure_analysis"
    PERFORMANCE_TRENDS = "performance_trends"
    OPTIMIZATION_IMPACT = "optimization_impact"
    EXECUTIVE_SUMMARY = "executive_summary"

class ReportFrequency(Enum):
    REAL_TIME = "real_time"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

class InsightType(Enum):
    TREND = "trend"
    ANOMALY = "anomaly"
    RECOMMENDATION = "recommendation"
    ALERT = "alert"
    SUCCESS = "success"

@dataclass
class ReportInsight:
    """Individual insight within a report"""
    id: str
    insight_type: InsightType
    title: str
    description: str
    impact_level: str  # high, medium, low
    confidence: float
    data_points: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime

@dataclass
class Report:
    """Comprehensive report structure"""
    id: str
    report_type: ReportType
    title: str
    summary: str
    time_period: Tuple[datetime, datetime]
    generated_at: datetime
    insights: List[ReportInsight]
    metrics: Dict[str, Any]
    charts: Dict[str, Any]
    recommendations: List[str]
    next_actions: List[str]
    metadata: Dict[str, Any]

@dataclass
class ReportSchedule:
    """Report scheduling configuration"""
    schedule_id: str
    report_type: ReportType
    frequency: ReportFrequency
    recipients: List[str]
    filters: Dict[str, Any]
    enabled: bool
    last_generated: Optional[datetime]
    next_generation: datetime

class ComprehensiveReporter:
    """
    Comprehensive reporting system for production monitoring and continuous improvement
    """
    
    def __init__(self, production_monitor=None, ux_monitor=None, pattern_detector=None):
        self.production_monitor = production_monitor
        self.ux_monitor = ux_monitor
        self.pattern_detector = pattern_detector
        
        self.generated_reports = {}
        self.report_schedules = {}
        self.insight_templates = {}
        self.reporting_active = False
        
        # Initialize insight templates
        self._setup_insight_templates()
        
        # Initialize default report schedules
        self._setup_default_schedules()
    
    def _setup_insight_templates(self):
        """Setup templates for generating insights"""
        self.insight_templates = {
            "performance_degradation": {
                "type": InsightType.TREND,
                "title_template": "Performance degradation detected in {component}",
                "description_template": "Response times have increased by {percentage}% over the last {period}",
                "recommendations": [
                    "Investigate resource utilization",
                    "Check for memory leaks",
                    "Review recent deployments",
                    "Consider scaling resources"
                ]
            },
            "user_satisfaction_decline": {
                "type": InsightType.ALERT,
                "title_template": "User satisfaction declining",
                "description_template": "User satisfaction has dropped to {score} from {previous_score}",
                "recommendations": [
                    "Review user feedback",
                    "Analyze user journey issues",
                    "Check for recent UI changes",
                    "Implement user experience improvements"
                ]
            },
            "failure_pattern_detected": {
                "type": InsightType.ANOMALY,
                "title_template": "Recurring failure pattern identified",
                "description_template": "Pattern '{pattern_type}' detected with {frequency} occurrences",
                "recommendations": [
                    "Implement preventive measures",
                    "Review root cause analysis",
                    "Update monitoring rules",
                    "Consider architectural changes"
                ]
            },
            "optimization_success": {
                "type": InsightType.SUCCESS,
                "title_template": "Optimization showing positive results",
                "description_template": "Optimization '{optimization}' improved {metric} by {improvement}%",
                "recommendations": [
                    "Continue monitoring results",
                    "Consider expanding optimization",
                    "Document successful approach",
                    "Share learnings with team"
                ]
            }
        }
    
    def _setup_default_schedules(self):
        """Setup default report schedules"""
        current_time = datetime.now()
        
        self.report_schedules = {
            "daily_health": ReportSchedule(
                schedule_id="daily_health",
                report_type=ReportType.SYSTEM_HEALTH,
                frequency=ReportFrequency.DAILY,
                recipients=["ops-team@company.com"],
                filters={},
                enabled=True,
                last_generated=None,
                next_generation=current_time.replace(hour=8, minute=0, second=0) + timedelta(days=1)
            ),
            "weekly_ux": ReportSchedule(
                schedule_id="weekly_ux",
                report_type=ReportType.USER_EXPERIENCE,
                frequency=ReportFrequency.WEEKLY,
                recipients=["product-team@company.com"],
                filters={},
                enabled=True,
                last_generated=None,
                next_generation=current_time.replace(hour=9, minute=0, second=0) + timedelta(days=7)
            ),
            "monthly_executive": ReportSchedule(
                schedule_id="monthly_executive",
                report_type=ReportType.EXECUTIVE_SUMMARY,
                frequency=ReportFrequency.MONTHLY,
                recipients=["executives@company.com"],
                filters={},
                enabled=True,
                last_generated=None,
                next_generation=current_time.replace(day=1, hour=10, minute=0, second=0) + timedelta(days=32)
            )
        }
    
    async def start_reporting(self):
        """Start the comprehensive reporting system"""
        self.reporting_active = True
        logger.info("Comprehensive Reporter started")
        
        tasks = [
            asyncio.create_task(self._process_scheduled_reports()),
            asyncio.create_task(self._generate_real_time_insights()),
            asyncio.create_task(self._cleanup_old_reports())
        ]
        
        await asyncio.gather(*tasks)
    
    async def stop_reporting(self):
        """Stop the comprehensive reporting system"""
        self.reporting_active = False
        logger.info("Comprehensive Reporter stopped")
    
    async def generate_report(self, report_type: ReportType, time_period: Tuple[datetime, datetime] = None,
                            filters: Dict[str, Any] = None) -> Report:
        """Generate a comprehensive report"""
        if not time_period:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
            time_period = (start_time, end_time)
        
        start_time, end_time = time_period
        
        # Generate report based on type
        if report_type == ReportType.SYSTEM_HEALTH:
            return await self._generate_system_health_report(start_time, end_time, filters)
        elif report_type == ReportType.USER_EXPERIENCE:
            return await self._generate_ux_report(start_time, end_time, filters)
        elif report_type == ReportType.FAILURE_ANALYSIS:
            return await self._generate_failure_analysis_report(start_time, end_time, filters)
        elif report_type == ReportType.PERFORMANCE_TRENDS:
            return await self._generate_performance_trends_report(start_time, end_time, filters)
        elif report_type == ReportType.OPTIMIZATION_IMPACT:
            return await self._generate_optimization_impact_report(start_time, end_time, filters)
        elif report_type == ReportType.EXECUTIVE_SUMMARY:
            return await self._generate_executive_summary_report(start_time, end_time, filters)
        else:
            raise ValueError(f"Unknown report type: {report_type}")
    
    async def _generate_system_health_report(self, start_time: datetime, end_time: datetime,
                                           filters: Dict[str, Any] = None) -> Report:
        """Generate system health report"""
        report_id = str(uuid.uuid4())
        insights = []
        metrics = {}
        recommendations = []
        
        # Get system health data
        if self.production_monitor:
            health_data = self.production_monitor.get_system_health()
            metrics.update(health_data)
            
            # Generate insights based on health data
            if health_data.get("health_score", 100) < 80:
                insight = ReportInsight(
                    id=str(uuid.uuid4()),
                    insight_type=InsightType.ALERT,
                    title="System Health Below Optimal",
                    description=f"System health score is {health_data.get('health_score', 0):.1f}/100",
                    impact_level="high",
                    confidence=0.9,
                    data_points={"health_score": health_data.get("health_score", 0)},
                    recommendations=[
                        "Investigate resource usage",
                        "Check for active alerts",
                        "Review recent changes"
                    ],
                    timestamp=datetime.now()
                )
                insights.append(insight)
            
            # Check for resource usage trends
            if "metrics" in health_data:
                for metric_name, metric_data in health_data["metrics"].items():
                    if isinstance(metric_data, dict) and "current" in metric_data:
                        current_value = metric_data["current"]
                        
                        if metric_name == "cpu_usage" and current_value > 80:
                            insight = ReportInsight(
                                id=str(uuid.uuid4()),
                                insight_type=InsightType.TREND,
                                title="High CPU Usage Detected",
                                description=f"CPU usage is at {current_value:.1f}%",
                                impact_level="medium",
                                confidence=0.8,
                                data_points={"cpu_usage": current_value},
                                recommendations=[
                                    "Consider scaling resources",
                                    "Optimize CPU-intensive processes",
                                    "Review application performance"
                                ],
                                timestamp=datetime.now()
                            )
                            insights.append(insight)
                        
                        elif metric_name == "memory_usage" and current_value > 85:
                            insight = ReportInsight(
                                id=str(uuid.uuid4()),
                                insight_type=InsightType.TREND,
                                title="High Memory Usage Detected",
                                description=f"Memory usage is at {current_value:.1f}%",
                                impact_level="medium",
                                confidence=0.8,
                                data_points={"memory_usage": current_value},
                                recommendations=[
                                    "Check for memory leaks",
                                    "Optimize memory usage",
                                    "Consider increasing memory allocation"
                                ],
                                timestamp=datetime.now()
                            )
                            insights.append(insight)
        
        # Generate overall recommendations
        if insights:
            recommendations.extend([
                "Monitor system metrics closely",
                "Implement proactive scaling",
                "Review alerting thresholds"
            ])
        else:
            recommendations.extend([
                "Continue current monitoring practices",
                "Consider optimizing for better performance",
                "Plan for future capacity needs"
            ])
        
        # Create summary
        summary = f"System health report for {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}. "
        if insights:
            summary += f"Identified {len(insights)} areas for attention. "
        summary += f"Overall health score: {metrics.get('health_score', 'N/A')}/100."
        
        report = Report(
            id=report_id,
            report_type=ReportType.SYSTEM_HEALTH,
            title="System Health Report",
            summary=summary,
            time_period=(start_time, end_time),
            generated_at=datetime.now(),
            insights=insights,
            metrics=metrics,
            charts={},  # Would contain chart data in real implementation
            recommendations=recommendations,
            next_actions=[
                "Review and address high-priority insights",
                "Update monitoring thresholds if needed",
                "Plan capacity adjustments"
            ],
            metadata={"filters": filters or {}}
        )
        
        self.generated_reports[report_id] = report
        return report
    
    async def _generate_ux_report(self, start_time: datetime, end_time: datetime,
                                filters: Dict[str, Any] = None) -> Report:
        """Generate user experience report"""
        report_id = str(uuid.uuid4())
        insights = []
        metrics = {}
        recommendations = []
        
        # Get UX data
        if self.ux_monitor:
            ux_data = self.ux_monitor.get_ux_dashboard()
            metrics.update(ux_data)
            
            # Generate UX insights
            satisfaction = ux_data.get("average_satisfaction", 0.8)
            if satisfaction < 0.8:
                insight = ReportInsight(
                    id=str(uuid.uuid4()),
                    insight_type=InsightType.ALERT,
                    title="User Satisfaction Below Target",
                    description=f"Average user satisfaction is {satisfaction:.2f}, below target of 0.8",
                    impact_level="high",
                    confidence=0.9,
                    data_points={"satisfaction": satisfaction},
                    recommendations=[
                        "Analyze user feedback",
                        "Review recent UI changes",
                        "Implement user experience improvements"
                    ],
                    timestamp=datetime.now()
                )
                insights.append(insight)
            
            # Check load times
            load_time = ux_data.get("average_load_time", 2000)
            if load_time > 3000:
                insight = ReportInsight(
                    id=str(uuid.uuid4()),
                    insight_type=InsightType.TREND,
                    title="Slow Page Load Times",
                    description=f"Average load time is {load_time:.0f}ms, above target of 3000ms",
                    impact_level="medium",
                    confidence=0.8,
                    data_points={"load_time": load_time},
                    recommendations=[
                        "Optimize page loading",
                        "Implement caching strategies",
                        "Compress assets"
                    ],
                    timestamp=datetime.now()
                )
                insights.append(insight)
            
            # Check for optimization opportunities
            optimizations = self.ux_monitor.get_optimization_recommendations()
            if optimizations:
                insight = ReportInsight(
                    id=str(uuid.uuid4()),
                    insight_type=InsightType.RECOMMENDATION,
                    title="UX Optimization Opportunities",
                    description=f"Found {len(optimizations)} optimization opportunities",
                    impact_level="medium",
                    confidence=0.7,
                    data_points={"optimization_count": len(optimizations)},
                    recommendations=[opt.get("description", "") for opt in optimizations[:3]],
                    timestamp=datetime.now()
                )
                insights.append(insight)
        
        # Generate recommendations
        if insights:
            recommendations.extend([
                "Prioritize user experience improvements",
                "Implement A/B testing for changes",
                "Monitor user feedback closely"
            ])
        else:
            recommendations.extend([
                "Continue monitoring user satisfaction",
                "Look for proactive improvement opportunities",
                "Maintain current UX standards"
            ])
        
        summary = f"User experience report for {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}. "
        summary += f"Average satisfaction: {metrics.get('average_satisfaction', 'N/A'):.2f}. "
        if insights:
            summary += f"Identified {len(insights)} improvement opportunities."
        
        report = Report(
            id=report_id,
            report_type=ReportType.USER_EXPERIENCE,
            title="User Experience Report",
            summary=summary,
            time_period=(start_time, end_time),
            generated_at=datetime.now(),
            insights=insights,
            metrics=metrics,
            charts={},
            recommendations=recommendations,
            next_actions=[
                "Implement high-priority UX improvements",
                "Set up A/B tests for proposed changes",
                "Review user feedback trends"
            ],
            metadata={"filters": filters or {}}
        )
        
        self.generated_reports[report_id] = report
        return report
    
    async def _generate_failure_analysis_report(self, start_time: datetime, end_time: datetime,
                                              filters: Dict[str, Any] = None) -> Report:
        """Generate failure analysis report"""
        report_id = str(uuid.uuid4())
        insights = []
        metrics = {}
        recommendations = []
        
        # Get failure pattern data
        if self.pattern_detector:
            patterns = self.pattern_detector.get_detected_patterns()
            component_health = self.pattern_detector.get_component_health()
            prevention_status = self.pattern_detector.get_prevention_status()
            
            metrics.update({
                "total_patterns": len(patterns),
                "component_health": component_health,
                "prevention_status": prevention_status
            })
            
            # Analyze critical patterns
            critical_patterns = [p for p in patterns if p.get("severity") == "critical"]
            if critical_patterns:
                insight = ReportInsight(
                    id=str(uuid.uuid4()),
                    insight_type=InsightType.ALERT,
                    title="Critical Failure Patterns Detected",
                    description=f"Found {len(critical_patterns)} critical failure patterns",
                    impact_level="high",
                    confidence=0.9,
                    data_points={"critical_patterns": len(critical_patterns)},
                    recommendations=[
                        "Address critical patterns immediately",
                        "Implement preventive measures",
                        "Review system architecture"
                    ],
                    timestamp=datetime.now()
                )
                insights.append(insight)
            
            # Analyze component health
            unhealthy_components = [
                comp for comp, health in component_health.items()
                if health.get("health_status") == "degraded"
            ]
            
            if unhealthy_components:
                insight = ReportInsight(
                    id=str(uuid.uuid4()),
                    insight_type=InsightType.TREND,
                    title="Component Health Degradation",
                    description=f"Components showing degraded health: {', '.join(unhealthy_components)}",
                    impact_level="medium",
                    confidence=0.8,
                    data_points={"unhealthy_components": unhealthy_components},
                    recommendations=[
                        "Investigate degraded components",
                        "Review component dependencies",
                        "Consider component restart or scaling"
                    ],
                    timestamp=datetime.now()
                )
                insights.append(insight)
        
        # Generate recommendations
        recommendations.extend([
            "Implement automated failure recovery",
            "Enhance monitoring coverage",
            "Review and update prevention rules"
        ])
        
        summary = f"Failure analysis report for {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}. "
        summary += f"Analyzed {metrics.get('total_patterns', 0)} failure patterns. "
        if insights:
            summary += f"Identified {len(insights)} areas requiring attention."
        
        report = Report(
            id=report_id,
            report_type=ReportType.FAILURE_ANALYSIS,
            title="Failure Analysis Report",
            summary=summary,
            time_period=(start_time, end_time),
            generated_at=datetime.now(),
            insights=insights,
            metrics=metrics,
            charts={},
            recommendations=recommendations,
            next_actions=[
                "Address critical failure patterns",
                "Update prevention rules",
                "Improve monitoring coverage"
            ],
            metadata={"filters": filters or {}}
        )
        
        self.generated_reports[report_id] = report
        return report
    
    async def _generate_executive_summary_report(self, start_time: datetime, end_time: datetime,
                                               filters: Dict[str, Any] = None) -> Report:
        """Generate executive summary report"""
        report_id = str(uuid.uuid4())
        insights = []
        metrics = {}
        recommendations = []
        
        # Collect high-level metrics from all systems
        if self.production_monitor:
            health_data = self.production_monitor.get_system_health()
            metrics["system_health"] = health_data.get("health_score", 0)
            metrics["active_alerts"] = health_data.get("active_alerts", 0)
        
        if self.ux_monitor:
            ux_data = self.ux_monitor.get_ux_dashboard()
            metrics["user_satisfaction"] = ux_data.get("average_satisfaction", 0)
            metrics["active_sessions"] = ux_data.get("active_sessions", 0)
        
        if self.pattern_detector:
            patterns = self.pattern_detector.get_detected_patterns()
            metrics["failure_patterns"] = len(patterns)
            metrics["critical_patterns"] = len([p for p in patterns if p.get("severity") == "critical"])
        
        # Generate executive insights
        overall_health = metrics.get("system_health", 0)
        if overall_health < 80:
            insight = ReportInsight(
                id=str(uuid.uuid4()),
                insight_type=InsightType.ALERT,
                title="System Health Requires Attention",
                description=f"Overall system health is {overall_health:.1f}%, below optimal threshold",
                impact_level="high",
                confidence=0.9,
                data_points={"health_score": overall_health},
                recommendations=[
                    "Immediate investigation required",
                    "Allocate resources for system optimization",
                    "Review operational procedures"
                ],
                timestamp=datetime.now()
            )
            insights.append(insight)
        
        user_satisfaction = metrics.get("user_satisfaction", 0)
        if user_satisfaction < 0.8:
            insight = ReportInsight(
                id=str(uuid.uuid4()),
                insight_type=InsightType.TREND,
                title="User Satisfaction Below Target",
                description=f"User satisfaction is {user_satisfaction:.2f}, impacting customer experience",
                impact_level="high",
                confidence=0.8,
                data_points={"satisfaction": user_satisfaction},
                recommendations=[
                    "Invest in user experience improvements",
                    "Conduct user research",
                    "Prioritize customer-facing issues"
                ],
                timestamp=datetime.now()
            )
            insights.append(insight)
        
        critical_patterns = metrics.get("critical_patterns", 0)
        if critical_patterns > 0:
            insight = ReportInsight(
                id=str(uuid.uuid4()),
                insight_type=InsightType.ALERT,
                title="Critical Issues Detected",
                description=f"{critical_patterns} critical failure patterns require immediate attention",
                impact_level="high",
                confidence=0.9,
                data_points={"critical_patterns": critical_patterns},
                recommendations=[
                    "Immediate technical review required",
                    "Allocate engineering resources",
                    "Consider emergency response procedures"
                ],
                timestamp=datetime.now()
            )
            insights.append(insight)
        
        # Generate executive recommendations
        if insights:
            recommendations.extend([
                "Prioritize system stability investments",
                "Increase monitoring and alerting coverage",
                "Review incident response procedures"
            ])
        else:
            recommendations.extend([
                "Continue current operational excellence",
                "Look for optimization opportunities",
                "Plan for future growth and scaling"
            ])
        
        # Create executive summary
        summary = f"Executive summary for {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}. "
        summary += f"System health: {overall_health:.1f}%, User satisfaction: {user_satisfaction:.2f}. "
        if insights:
            summary += f"{len(insights)} critical areas identified for executive attention."
        else:
            summary += "Systems operating within acceptable parameters."
        
        report = Report(
            id=report_id,
            report_type=ReportType.EXECUTIVE_SUMMARY,
            title="Executive Summary Report",
            summary=summary,
            time_period=(start_time, end_time),
            generated_at=datetime.now(),
            insights=insights,
            metrics=metrics,
            charts={},
            recommendations=recommendations,
            next_actions=[
                "Review and approve recommended actions",
                "Allocate resources for critical issues",
                "Schedule follow-up review"
            ],
            metadata={"filters": filters or {}}
        )
        
        self.generated_reports[report_id] = report
        return report
    
    async def _process_scheduled_reports(self):
        """Process scheduled report generation"""
        while self.reporting_active:
            try:
                current_time = datetime.now()
                
                for schedule in self.report_schedules.values():
                    if schedule.enabled and current_time >= schedule.next_generation:
                        # Generate scheduled report
                        logger.info(f"Generating scheduled report: {schedule.schedule_id}")
                        
                        # Calculate time period based on frequency
                        if schedule.frequency == ReportFrequency.DAILY:
                            start_time = current_time - timedelta(days=1)
                        elif schedule.frequency == ReportFrequency.WEEKLY:
                            start_time = current_time - timedelta(weeks=1)
                        elif schedule.frequency == ReportFrequency.MONTHLY:
                            start_time = current_time - timedelta(days=30)
                        else:
                            start_time = current_time - timedelta(hours=1)
                        
                        report = await self.generate_report(
                            schedule.report_type,
                            (start_time, current_time),
                            schedule.filters
                        )
                        
                        # Send report to recipients
                        await self._send_report(report, schedule.recipients)
                        
                        # Update schedule
                        schedule.last_generated = current_time
                        if schedule.frequency == ReportFrequency.DAILY:
                            schedule.next_generation = current_time + timedelta(days=1)
                        elif schedule.frequency == ReportFrequency.WEEKLY:
                            schedule.next_generation = current_time + timedelta(weeks=1)
                        elif schedule.frequency == ReportFrequency.MONTHLY:
                            schedule.next_generation = current_time + timedelta(days=30)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error processing scheduled reports: {e}")
                await asyncio.sleep(600)
    
    async def _generate_real_time_insights(self):
        """Generate real-time insights"""
        while self.reporting_active:
            try:
                # This would generate real-time insights based on current system state
                # For now, just log that we're monitoring
                logger.debug("Monitoring for real-time insights...")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error generating real-time insights: {e}")
                await asyncio.sleep(300)
    
    async def _cleanup_old_reports(self):
        """Clean up old reports"""
        while self.reporting_active:
            try:
                current_time = datetime.now()
                cleanup_threshold = current_time - timedelta(days=90)  # Keep reports for 90 days
                
                old_reports = [
                    report_id for report_id, report in self.generated_reports.items()
                    if report.generated_at < cleanup_threshold
                ]
                
                for report_id in old_reports:
                    del self.generated_reports[report_id]
                
                if old_reports:
                    logger.info(f"Cleaned up {len(old_reports)} old reports")
                
                await asyncio.sleep(86400)  # Clean up daily
                
            except Exception as e:
                logger.error(f"Error cleaning up old reports: {e}")
                await asyncio.sleep(86400)
    
    async def _send_report(self, report: Report, recipients: List[str]):
        """Send report to recipients"""
        # In a real implementation, this would send emails, Slack messages, etc.
        logger.info(f"Sending report {report.id} to {len(recipients)} recipients")
        
        report_summary = {
            "report_id": report.id,
            "title": report.title,
            "summary": report.summary,
            "insights_count": len(report.insights),
            "recommendations_count": len(report.recommendations)
        }
        
        logger.info(f"Report summary: {json.dumps(report_summary, indent=2)}")
    
    # Public API methods
    
    def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific report"""
        report = self.generated_reports.get(report_id)
        return asdict(report) if report else None
    
    def list_reports(self, report_type: ReportType = None, limit: int = 50) -> List[Dict[str, Any]]:
        """List generated reports"""
        reports = list(self.generated_reports.values())
        
        if report_type:
            reports = [r for r in reports if r.report_type == report_type]
        
        # Sort by generation time, most recent first
        reports.sort(key=lambda x: x.generated_at, reverse=True)
        
        return [asdict(report) for report in reports[:limit]]
    
    def get_report_schedules(self) -> List[Dict[str, Any]]:
        """Get report schedules"""
        return [asdict(schedule) for schedule in self.report_schedules.values()]
    
    def update_report_schedule(self, schedule_id: str, updates: Dict[str, Any]) -> bool:
        """Update a report schedule"""
        if schedule_id not in self.report_schedules:
            return False
        
        schedule = self.report_schedules[schedule_id]
        
        for key, value in updates.items():
            if hasattr(schedule, key):
                setattr(schedule, key, value)
        
        return True
    
    def get_insights_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get summary of insights from recent reports"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        recent_reports = [
            report for report in self.generated_reports.values()
            if report.generated_at >= cutoff_time
        ]
        
        all_insights = []
        for report in recent_reports:
            all_insights.extend(report.insights)
        
        # Categorize insights
        insight_counts = Counter(insight.insight_type.value for insight in all_insights)
        impact_counts = Counter(insight.impact_level for insight in all_insights)
        
        return {
            "total_insights": len(all_insights),
            "reports_analyzed": len(recent_reports),
            "insight_types": dict(insight_counts),
            "impact_levels": dict(impact_counts),
            "average_confidence": statistics.mean([i.confidence for i in all_insights]) if all_insights else 0,
            "top_recommendations": Counter([
                rec for insight in all_insights for rec in insight.recommendations
            ]).most_common(5)
        }

# Global comprehensive reporter instance
comprehensive_reporter = ComprehensiveReporter()