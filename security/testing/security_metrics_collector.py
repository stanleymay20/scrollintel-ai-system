"""
Security Metrics Collection and Reporting with Trend Analysis
"""

import asyncio
import logging
import time
import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import os

from .security_test_framework import SecurityTestResult, SecurityTestType, SecuritySeverity

logger = logging.getLogger(__name__)

class MetricType(Enum):
    SECURITY_POSTURE = "security_posture"
    VULNERABILITY_TRENDS = "vulnerability_trends"
    THREAT_DETECTION = "threat_detection"
    COMPLIANCE_STATUS = "compliance_status"
    INCIDENT_METRICS = "incident_metrics"
    PERFORMANCE_IMPACT = "performance_impact"

@dataclass
class SecurityMetric:
    metric_id: str
    metric_type: MetricType
    name: str
    value: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

@dataclass
class TrendAnalysis:
    metric_name: str
    trend_direction: str  # "improving", "declining", "stable"
    trend_strength: float  # 0.0 to 1.0
    change_percentage: float
    time_period: str
    confidence_level: float
    recommendations: List[str]

class SecurityMetricsCollector:
    """Security metrics collection and analysis engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config.get('metrics_db_path', 'security_metrics.db')
        self.metrics_history: List[SecurityMetric] = []
        
        # Initialize database
        self._init_database()
        
        # Load historical metrics
        self._load_historical_metrics()
    
    def _init_database(self):
        """Initialize metrics database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS security_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        metric_id TEXT NOT NULL,
                        metric_type TEXT NOT NULL,
                        name TEXT NOT NULL,
                        value REAL NOT NULL,
                        unit TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        metadata TEXT,
                        tags TEXT
                    )
                ''')
                
                # Create trends table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trend_analysis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        metric_name TEXT NOT NULL,
                        trend_direction TEXT NOT NULL,
                        trend_strength REAL NOT NULL,
                        change_percentage REAL NOT NULL,
                        time_period TEXT NOT NULL,
                        confidence_level REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        recommendations TEXT
                    )
                ''')
                
                # Create indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_metric_type ON security_metrics(metric_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON security_metrics(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_metric_name ON security_metrics(name)')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to initialize metrics database: {e}")
    
    def _load_historical_metrics(self):
        """Load historical metrics from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Load recent metrics (last 30 days)
                thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
                
                cursor.execute('''
                    SELECT metric_id, metric_type, name, value, unit, timestamp, metadata, tags
                    FROM security_metrics
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                ''', (thirty_days_ago,))
                
                for row in cursor.fetchall():
                    metric_id, metric_type, name, value, unit, timestamp, metadata_json, tags_json = row
                    
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    tags = json.loads(tags_json) if tags_json else []
                    
                    metric = SecurityMetric(
                        metric_id=metric_id,
                        metric_type=MetricType(metric_type),
                        name=name,
                        value=value,
                        unit=unit,
                        timestamp=datetime.fromisoformat(timestamp),
                        metadata=metadata,
                        tags=tags
                    )
                    
                    self.metrics_history.append(metric)
                
                logger.info(f"Loaded {len(self.metrics_history)} historical metrics")
                
        except Exception as e:
            logger.error(f"Failed to load historical metrics: {e}")
    
    async def collect_security_metrics(self, test_results: List[SecurityTestResult], 
                                     target_config: Dict[str, Any]) -> Dict[str, Any]:
        """Collect comprehensive security metrics"""
        logger.info("Collecting security metrics")
        
        current_time = datetime.now()
        collected_metrics = []
        
        # Collect basic security posture metrics
        posture_metrics = self._collect_security_posture_metrics(test_results, current_time)
        collected_metrics.extend(posture_metrics)
        
        # Collect vulnerability trend metrics
        vulnerability_metrics = self._collect_vulnerability_metrics(test_results, current_time)
        collected_metrics.extend(vulnerability_metrics)
        
        # Collect threat detection metrics
        threat_metrics = self._collect_threat_detection_metrics(test_results, current_time)
        collected_metrics.extend(threat_metrics)
        
        # Collect compliance metrics
        compliance_metrics = self._collect_compliance_metrics(test_results, current_time)
        collected_metrics.extend(compliance_metrics)
        
        # Collect incident metrics
        incident_metrics = self._collect_incident_metrics(test_results, current_time)
        collected_metrics.extend(incident_metrics)
        
        # Collect performance impact metrics
        performance_metrics = self._collect_performance_metrics(test_results, current_time)
        collected_metrics.extend(performance_metrics)
        
        # Store metrics
        self._store_metrics(collected_metrics)
        
        # Add to history
        self.metrics_history.extend(collected_metrics)
        
        # Perform trend analysis
        trend_analysis = await self._perform_trend_analysis()
        
        # Generate metrics report
        report = self._generate_metrics_report(collected_metrics, trend_analysis)
        
        logger.info(f"Collected {len(collected_metrics)} security metrics")
        return report
    
    def _collect_security_posture_metrics(self, test_results: List[SecurityTestResult], 
                                        timestamp: datetime) -> List[SecurityMetric]:
        """Collect security posture metrics"""
        metrics = []
        
        # Overall security score
        total_tests = len(test_results)
        passed_tests = len([r for r in test_results if r.status == "passed"])
        security_score = (passed_tests / max(total_tests, 1)) * 100
        
        metrics.append(SecurityMetric(
            metric_id=f"security_score_{int(timestamp.timestamp())}",
            metric_type=MetricType.SECURITY_POSTURE,
            name="Overall Security Score",
            value=security_score,
            unit="percentage",
            timestamp=timestamp,
            metadata={"total_tests": total_tests, "passed_tests": passed_tests},
            tags=["security_posture", "overall"]
        ))
        
        # Security test coverage
        test_types = set(r.test_type for r in test_results)
        coverage_score = len(test_types) / len(SecurityTestType) * 100
        
        metrics.append(SecurityMetric(
            metric_id=f"test_coverage_{int(timestamp.timestamp())}",
            metric_type=MetricType.SECURITY_POSTURE,
            name="Security Test Coverage",
            value=coverage_score,
            unit="percentage",
            timestamp=timestamp,
            metadata={"covered_types": list(test_types)},
            tags=["security_posture", "coverage"]
        ))
        
        # Critical findings count
        critical_findings = sum(len([f for f in r.findings if f.get('severity') == 'critical']) 
                              for r in test_results)
        
        metrics.append(SecurityMetric(
            metric_id=f"critical_findings_{int(timestamp.timestamp())}",
            metric_type=MetricType.SECURITY_POSTURE,
            name="Critical Security Findings",
            value=critical_findings,
            unit="count",
            timestamp=timestamp,
            metadata={"severity": "critical"},
            tags=["security_posture", "critical"]
        ))
        
        # High findings count
        high_findings = sum(len([f for f in r.findings if f.get('severity') == 'high']) 
                           for r in test_results)
        
        metrics.append(SecurityMetric(
            metric_id=f"high_findings_{int(timestamp.timestamp())}",
            metric_type=MetricType.SECURITY_POSTURE,
            name="High Security Findings",
            value=high_findings,
            unit="count",
            timestamp=timestamp,
            metadata={"severity": "high"},
            tags=["security_posture", "high"]
        ))
        
        return metrics
    
    def _collect_vulnerability_metrics(self, test_results: List[SecurityTestResult], 
                                     timestamp: datetime) -> List[SecurityMetric]:
        """Collect vulnerability trend metrics"""
        metrics = []
        
        # Vulnerability scan results
        vuln_results = [r for r in test_results if r.test_type == SecurityTestType.VULNERABILITY]
        
        if vuln_results:
            # Total vulnerabilities found
            total_vulns = sum(len(r.findings) for r in vuln_results)
            
            metrics.append(SecurityMetric(
                metric_id=f"total_vulnerabilities_{int(timestamp.timestamp())}",
                metric_type=MetricType.VULNERABILITY_TRENDS,
                name="Total Vulnerabilities",
                value=total_vulns,
                unit="count",
                timestamp=timestamp,
                metadata={"scan_count": len(vuln_results)},
                tags=["vulnerabilities", "total"]
            ))
            
            # Vulnerability density (vulnerabilities per scan)
            vuln_density = total_vulns / len(vuln_results)
            
            metrics.append(SecurityMetric(
                metric_id=f"vulnerability_density_{int(timestamp.timestamp())}",
                metric_type=MetricType.VULNERABILITY_TRENDS,
                name="Vulnerability Density",
                value=vuln_density,
                unit="vulnerabilities_per_scan",
                timestamp=timestamp,
                metadata={"total_vulns": total_vulns, "scan_count": len(vuln_results)},
                tags=["vulnerabilities", "density"]
            ))
            
            # Average CVSS score (simulated)
            avg_cvss = sum(r.metadata.get('cvss_score', 5.0) for r in vuln_results) / len(vuln_results)
            
            metrics.append(SecurityMetric(
                metric_id=f"avg_cvss_score_{int(timestamp.timestamp())}",
                metric_type=MetricType.VULNERABILITY_TRENDS,
                name="Average CVSS Score",
                value=avg_cvss,
                unit="cvss_score",
                timestamp=timestamp,
                metadata={"scan_count": len(vuln_results)},
                tags=["vulnerabilities", "cvss"]
            ))
        
        return metrics
    
    def _collect_threat_detection_metrics(self, test_results: List[SecurityTestResult], 
                                        timestamp: datetime) -> List[SecurityMetric]:
        """Collect threat detection metrics"""
        metrics = []
        
        # Penetration test results
        pentest_results = [r for r in test_results if r.test_type == SecurityTestType.PENETRATION]
        
        if pentest_results:
            # Attack success rate
            successful_attacks = len([r for r in pentest_results if r.status == "failed"])  # Failed test = successful attack
            attack_success_rate = (successful_attacks / len(pentest_results)) * 100
            
            metrics.append(SecurityMetric(
                metric_id=f"attack_success_rate_{int(timestamp.timestamp())}",
                metric_type=MetricType.THREAT_DETECTION,
                name="Attack Success Rate",
                value=attack_success_rate,
                unit="percentage",
                timestamp=timestamp,
                metadata={"successful_attacks": successful_attacks, "total_attacks": len(pentest_results)},
                tags=["threat_detection", "attack_success"]
            ))
            
            # Defense effectiveness
            defense_effectiveness = 100 - attack_success_rate
            
            metrics.append(SecurityMetric(
                metric_id=f"defense_effectiveness_{int(timestamp.timestamp())}",
                metric_type=MetricType.THREAT_DETECTION,
                name="Defense Effectiveness",
                value=defense_effectiveness,
                unit="percentage",
                timestamp=timestamp,
                metadata={"blocked_attacks": len(pentest_results) - successful_attacks},
                tags=["threat_detection", "defense"]
            ))
        
        # Chaos engineering results
        chaos_results = [r for r in test_results if r.test_type == SecurityTestType.CHAOS]
        
        if chaos_results:
            # Average resilience score
            resilience_scores = [r.metadata.get('resilience_score', 0.5) for r in chaos_results]
            avg_resilience = statistics.mean(resilience_scores) * 100
            
            metrics.append(SecurityMetric(
                metric_id=f"avg_resilience_score_{int(timestamp.timestamp())}",
                metric_type=MetricType.THREAT_DETECTION,
                name="Average Resilience Score",
                value=avg_resilience,
                unit="percentage",
                timestamp=timestamp,
                metadata={"chaos_tests": len(chaos_results)},
                tags=["threat_detection", "resilience"]
            ))
            
            # Average recovery time
            recovery_times = [r.metadata.get('recovery_time', 60) for r in chaos_results]
            avg_recovery_time = statistics.mean(recovery_times)
            
            metrics.append(SecurityMetric(
                metric_id=f"avg_recovery_time_{int(timestamp.timestamp())}",
                metric_type=MetricType.THREAT_DETECTION,
                name="Average Recovery Time",
                value=avg_recovery_time,
                unit="seconds",
                timestamp=timestamp,
                metadata={"chaos_tests": len(chaos_results)},
                tags=["threat_detection", "recovery"]
            ))
        
        return metrics
    
    def _collect_compliance_metrics(self, test_results: List[SecurityTestResult], 
                                   timestamp: datetime) -> List[SecurityMetric]:
        """Collect compliance metrics"""
        metrics = []
        
        # Compliance test results
        compliance_results = [r for r in test_results if 'compliance' in r.test_name.lower()]
        
        if compliance_results:
            # Compliance score
            passed_compliance = len([r for r in compliance_results if r.status == "passed"])
            compliance_score = (passed_compliance / len(compliance_results)) * 100
            
            metrics.append(SecurityMetric(
                metric_id=f"compliance_score_{int(timestamp.timestamp())}",
                metric_type=MetricType.COMPLIANCE_STATUS,
                name="Compliance Score",
                value=compliance_score,
                unit="percentage",
                timestamp=timestamp,
                metadata={"passed_tests": passed_compliance, "total_tests": len(compliance_results)},
                tags=["compliance", "score"]
            ))
        
        # Audit readiness (simulated)
        audit_readiness = min(100, security_score * 0.9 + 10)  # Based on overall security score
        
        metrics.append(SecurityMetric(
            metric_id=f"audit_readiness_{int(timestamp.timestamp())}",
            metric_type=MetricType.COMPLIANCE_STATUS,
            name="Audit Readiness",
            value=audit_readiness,
            unit="percentage",
            timestamp=timestamp,
            metadata={"based_on": "security_score"},
            tags=["compliance", "audit"]
        ))
        
        return metrics
    
    def _collect_incident_metrics(self, test_results: List[SecurityTestResult], 
                                timestamp: datetime) -> List[SecurityMetric]:
        """Collect incident metrics"""
        metrics = []
        
        # Security incidents (based on failed tests)
        failed_tests = [r for r in test_results if r.status == "failed"]
        incident_count = len(failed_tests)
        
        metrics.append(SecurityMetric(
            metric_id=f"security_incidents_{int(timestamp.timestamp())}",
            metric_type=MetricType.INCIDENT_METRICS,
            name="Security Incidents",
            value=incident_count,
            unit="count",
            timestamp=timestamp,
            metadata={"failed_tests": incident_count},
            tags=["incidents", "count"]
        ))
        
        # Mean time to detection (simulated)
        mttr = statistics.mean([r.execution_time for r in test_results]) * 60  # Convert to minutes
        
        metrics.append(SecurityMetric(
            metric_id=f"mean_time_to_detection_{int(timestamp.timestamp())}",
            metric_type=MetricType.INCIDENT_METRICS,
            name="Mean Time to Detection",
            value=mttr,
            unit="minutes",
            timestamp=timestamp,
            metadata={"based_on": "test_execution_time"},
            tags=["incidents", "detection_time"]
        ))
        
        return metrics
    
    def _collect_performance_metrics(self, test_results: List[SecurityTestResult], 
                                   timestamp: datetime) -> List[SecurityMetric]:
        """Collect performance impact metrics"""
        metrics = []
        
        # Performance test results
        perf_results = [r for r in test_results if r.test_type == SecurityTestType.PERFORMANCE]
        
        if perf_results:
            # Average security overhead
            overhead_values = [r.metadata.get('security_overhead', 10) for r in perf_results]
            avg_overhead = statistics.mean(overhead_values)
            
            metrics.append(SecurityMetric(
                metric_id=f"avg_security_overhead_{int(timestamp.timestamp())}",
                metric_type=MetricType.PERFORMANCE_IMPACT,
                name="Average Security Overhead",
                value=avg_overhead,
                unit="percentage",
                timestamp=timestamp,
                metadata={"performance_tests": len(perf_results)},
                tags=["performance", "overhead"]
            ))
            
            # Throughput impact
            throughput_impacts = [r.metadata.get('throughput_impact', 5) for r in perf_results]
            avg_throughput_impact = statistics.mean(throughput_impacts)
            
            metrics.append(SecurityMetric(
                metric_id=f"avg_throughput_impact_{int(timestamp.timestamp())}",
                metric_type=MetricType.PERFORMANCE_IMPACT,
                name="Average Throughput Impact",
                value=avg_throughput_impact,
                unit="percentage",
                timestamp=timestamp,
                metadata={"performance_tests": len(perf_results)},
                tags=["performance", "throughput"]
            ))
        
        return metrics
    
    def _store_metrics(self, metrics: List[SecurityMetric]):
        """Store metrics in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for metric in metrics:
                    cursor.execute('''
                        INSERT INTO security_metrics 
                        (metric_id, metric_type, name, value, unit, timestamp, metadata, tags)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        metric.metric_id,
                        metric.metric_type.value,
                        metric.name,
                        metric.value,
                        metric.unit,
                        metric.timestamp.isoformat(),
                        json.dumps(metric.metadata),
                        json.dumps(metric.tags)
                    ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")
    
    async def _perform_trend_analysis(self) -> List[TrendAnalysis]:
        """Perform trend analysis on historical metrics"""
        trend_analyses = []
        
        # Group metrics by name
        metrics_by_name = {}
        for metric in self.metrics_history:
            if metric.name not in metrics_by_name:
                metrics_by_name[metric.name] = []
            metrics_by_name[metric.name].append(metric)
        
        # Analyze trends for each metric
        for metric_name, metric_list in metrics_by_name.items():
            if len(metric_list) < 3:  # Need at least 3 data points
                continue
            
            # Sort by timestamp
            metric_list.sort(key=lambda x: x.timestamp)
            
            # Calculate trend
            trend_analysis = self._calculate_trend(metric_name, metric_list)
            if trend_analysis:
                trend_analyses.append(trend_analysis)
        
        # Store trend analysis
        self._store_trend_analysis(trend_analyses)
        
        return trend_analyses
    
    def _calculate_trend(self, metric_name: str, metrics: List[SecurityMetric]) -> Optional[TrendAnalysis]:
        """Calculate trend for a specific metric"""
        try:
            values = [m.value for m in metrics]
            timestamps = [m.timestamp.timestamp() for m in metrics]
            
            # Simple linear regression for trend
            n = len(values)
            sum_x = sum(timestamps)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(timestamps, values))
            sum_x2 = sum(x * x for x in timestamps)
            
            # Calculate slope
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            # Determine trend direction
            if abs(slope) < 0.001:  # Threshold for stability
                trend_direction = "stable"
                trend_strength = 0.0
            elif slope > 0:
                trend_direction = "improving" if "score" in metric_name.lower() or "effectiveness" in metric_name.lower() else "increasing"
                trend_strength = min(1.0, abs(slope) * 1000)  # Normalize
            else:
                trend_direction = "declining" if "score" in metric_name.lower() or "effectiveness" in metric_name.lower() else "decreasing"
                trend_strength = min(1.0, abs(slope) * 1000)  # Normalize
            
            # Calculate change percentage
            if len(values) >= 2:
                change_percentage = ((values[-1] - values[0]) / max(abs(values[0]), 0.001)) * 100
            else:
                change_percentage = 0.0
            
            # Calculate confidence level (simplified)
            confidence_level = min(1.0, len(values) / 10.0)  # More data points = higher confidence
            
            # Generate recommendations
            recommendations = self._generate_trend_recommendations(metric_name, trend_direction, change_percentage)
            
            return TrendAnalysis(
                metric_name=metric_name,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                change_percentage=change_percentage,
                time_period=f"{len(values)} data points",
                confidence_level=confidence_level,
                recommendations=recommendations
            )
        
        except Exception as e:
            logger.error(f"Failed to calculate trend for {metric_name}: {e}")
            return None
    
    def _generate_trend_recommendations(self, metric_name: str, trend_direction: str, 
                                      change_percentage: float) -> List[str]:
        """Generate recommendations based on trend analysis"""
        recommendations = []
        
        if "security score" in metric_name.lower():
            if trend_direction == "declining":
                recommendations.extend([
                    "URGENT: Security posture is declining",
                    "Review and strengthen security controls",
                    "Increase security testing frequency"
                ])
            elif trend_direction == "improving":
                recommendations.extend([
                    "POSITIVE: Security posture is improving",
                    "Continue current security practices",
                    "Document successful security measures"
                ])
        
        elif "vulnerabilities" in metric_name.lower():
            if trend_direction == "increasing":
                recommendations.extend([
                    "WARNING: Vulnerability count is increasing",
                    "Accelerate vulnerability remediation",
                    "Review security development practices"
                ])
            elif trend_direction == "decreasing":
                recommendations.extend([
                    "GOOD: Vulnerability count is decreasing",
                    "Maintain current remediation pace",
                    "Continue proactive security measures"
                ])
        
        elif "attack success rate" in metric_name.lower():
            if trend_direction == "increasing":
                recommendations.extend([
                    "CRITICAL: Attack success rate is increasing",
                    "Strengthen defensive measures immediately",
                    "Review and update security controls"
                ])
            elif trend_direction == "decreasing":
                recommendations.extend([
                    "EXCELLENT: Attack success rate is decreasing",
                    "Maintain current defensive posture",
                    "Document effective security measures"
                ])
        
        # Add general recommendations based on change magnitude
        if abs(change_percentage) > 50:
            recommendations.append("Significant change detected - investigate root causes")
        elif abs(change_percentage) > 20:
            recommendations.append("Notable change detected - monitor closely")
        
        return recommendations
    
    def _store_trend_analysis(self, trend_analyses: List[TrendAnalysis]):
        """Store trend analysis in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for trend in trend_analyses:
                    cursor.execute('''
                        INSERT INTO trend_analysis 
                        (metric_name, trend_direction, trend_strength, change_percentage, 
                         time_period, confidence_level, timestamp, recommendations)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        trend.metric_name,
                        trend.trend_direction,
                        trend.trend_strength,
                        trend.change_percentage,
                        trend.time_period,
                        trend.confidence_level,
                        datetime.now().isoformat(),
                        json.dumps(trend.recommendations)
                    ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store trend analysis: {e}")
    
    def _generate_metrics_report(self, current_metrics: List[SecurityMetric], 
                               trend_analyses: List[TrendAnalysis]) -> Dict[str, Any]:
        """Generate comprehensive metrics report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_metrics": len(current_metrics),
                "metric_types": len(set(m.metric_type for m in current_metrics)),
                "trends_analyzed": len(trend_analyses)
            },
            "current_metrics": {},
            "trend_analysis": {},
            "recommendations": [],
            "alerts": []
        }
        
        # Group current metrics by type
        for metric in current_metrics:
            metric_type = metric.metric_type.value
            if metric_type not in report["current_metrics"]:
                report["current_metrics"][metric_type] = []
            
            report["current_metrics"][metric_type].append({
                "name": metric.name,
                "value": metric.value,
                "unit": metric.unit,
                "tags": metric.tags,
                "metadata": metric.metadata
            })
        
        # Add trend analysis
        for trend in trend_analyses:
            report["trend_analysis"][trend.metric_name] = {
                "direction": trend.trend_direction,
                "strength": trend.trend_strength,
                "change_percentage": trend.change_percentage,
                "confidence_level": trend.confidence_level,
                "recommendations": trend.recommendations
            }
            
            # Add trend recommendations to overall recommendations
            report["recommendations"].extend(trend.recommendations)
        
        # Generate alerts for critical trends
        critical_trends = [t for t in trend_analyses 
                          if t.trend_direction in ["declining", "increasing"] and 
                          abs(t.change_percentage) > 30]
        
        for trend in critical_trends:
            report["alerts"].append({
                "type": "trend_alert",
                "severity": "high",
                "metric": trend.metric_name,
                "message": f"{trend.metric_name} is {trend.trend_direction} by {trend.change_percentage:.1f}%",
                "recommendations": trend.recommendations[:2]  # Top 2 recommendations
            })
        
        # Add key performance indicators
        report["kpis"] = self._calculate_kpis(current_metrics)
        
        return report
    
    def _calculate_kpis(self, metrics: List[SecurityMetric]) -> Dict[str, Any]:
        """Calculate key performance indicators"""
        kpis = {}
        
        # Security posture KPIs
        security_scores = [m.value for m in metrics if m.name == "Overall Security Score"]
        if security_scores:
            kpis["security_score"] = security_scores[0]
        
        # Vulnerability KPIs
        total_vulns = [m.value for m in metrics if m.name == "Total Vulnerabilities"]
        if total_vulns:
            kpis["total_vulnerabilities"] = total_vulns[0]
        
        # Threat detection KPIs
        defense_effectiveness = [m.value for m in metrics if m.name == "Defense Effectiveness"]
        if defense_effectiveness:
            kpis["defense_effectiveness"] = defense_effectiveness[0]
        
        # Performance KPIs
        security_overhead = [m.value for m in metrics if m.name == "Average Security Overhead"]
        if security_overhead:
            kpis["security_overhead"] = security_overhead[0]
        
        # Compliance KPIs
        compliance_scores = [m.value for m in metrics if m.name == "Compliance Score"]
        if compliance_scores:
            kpis["compliance_score"] = compliance_scores[0]
        
        return kpis
    
    def get_metrics_dashboard_data(self) -> Dict[str, Any]:
        """Get data for metrics dashboard"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get recent metrics (last 7 days)
                seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()
                
                cursor.execute('''
                    SELECT name, value, timestamp, unit
                    FROM security_metrics
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                ''', (seven_days_ago,))
                
                recent_metrics = cursor.fetchall()
                
                # Get latest trend analysis
                cursor.execute('''
                    SELECT metric_name, trend_direction, change_percentage, confidence_level
                    FROM trend_analysis
                    ORDER BY timestamp DESC
                    LIMIT 10
                ''')
                
                recent_trends = cursor.fetchall()
                
                return {
                    "recent_metrics": [
                        {
                            "name": name,
                            "value": value,
                            "timestamp": timestamp,
                            "unit": unit
                        }
                        for name, value, timestamp, unit in recent_metrics
                    ],
                    "recent_trends": [
                        {
                            "metric_name": metric_name,
                            "trend_direction": trend_direction,
                            "change_percentage": change_percentage,
                            "confidence_level": confidence_level
                        }
                        for metric_name, trend_direction, change_percentage, confidence_level in recent_trends
                    ]
                }
        
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return {"recent_metrics": [], "recent_trends": []}