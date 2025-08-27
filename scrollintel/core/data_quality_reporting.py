"""
Data Quality Reporting System for Advanced Analytics Dashboard

Provides comprehensive reporting capabilities for data quality metrics,
trends, and automated report generation with multiple output formats.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
import json
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

from .data_quality_monitor import QualityReport, QualityIssue, QualitySeverity

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"


class ReportFrequency(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass
class ReportTemplate:
    """Defines a report template"""
    id: str
    name: str
    description: str
    sections: List[str]
    format: ReportFormat
    frequency: ReportFrequency
    recipients: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class QualityTrend:
    """Represents quality trends over time"""
    dataset_name: str
    time_period: str
    score_trend: List[Tuple[datetime, float]]
    issue_trend: List[Tuple[datetime, int]]
    dimension_trends: Dict[str, List[Tuple[datetime, float]]]
    trend_direction: str  # improving, declining, stable
    trend_confidence: float


class DataQualityReporting:
    """
    Comprehensive data quality reporting system with automated generation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.report_templates: Dict[str, ReportTemplate] = {}
        self.quality_history: List[QualityReport] = []
        self.generated_reports: List[Dict[str, Any]] = []
        
    def register_report_template(self, template: ReportTemplate) -> bool:
        """Register a report template"""
        try:
            self.report_templates[template.id] = template
            logger.info(f"Registered report template: {template.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register template {template.name}: {str(e)}")
            return False
    
    def add_quality_report(self, report: QualityReport) -> None:
        """Add a quality report to the history"""
        self.quality_history.append(report)
        
        # Keep only recent history (configurable retention)
        retention_days = self.config.get("retention_days", 90)
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        self.quality_history = [
            r for r in self.quality_history 
            if r.assessment_timestamp >= cutoff_date
        ]
    
    def generate_quality_summary_report(self, 
                                      dataset_names: Optional[List[str]] = None,
                                      time_range: Optional[Tuple[datetime, datetime]] = None,
                                      format: ReportFormat = ReportFormat.HTML) -> Dict[str, Any]:
        """Generate a comprehensive quality summary report"""
        try:
            # Filter reports
            filtered_reports = self._filter_reports(dataset_names, time_range)
            
            if not filtered_reports:
                return {"error": "No quality reports found for specified criteria"}
            
            # Generate report content
            report_data = {
                "title": "Data Quality Summary Report",
                "generated_at": datetime.utcnow().isoformat(),
                "time_range": self._format_time_range(time_range),
                "datasets_included": list(set(r.dataset_name for r in filtered_reports)),
                "executive_summary": self._generate_executive_summary(filtered_reports),
                "quality_metrics": self._calculate_aggregate_metrics(filtered_reports),
                "trend_analysis": self._analyze_quality_trends(filtered_reports),
                "issue_analysis": self._analyze_issues(filtered_reports),
                "recommendations": self._generate_recommendations(filtered_reports),
                "detailed_findings": self._generate_detailed_findings(filtered_reports)
            }
            
            # Format report
            if format == ReportFormat.HTML:
                formatted_report = self._format_html_report(report_data)
            elif format == ReportFormat.JSON:
                formatted_report = json.dumps(report_data, indent=2, default=str)
            elif format == ReportFormat.CSV:
                formatted_report = self._format_csv_report(report_data)
            else:
                formatted_report = str(report_data)
            
            # Store generated report
            generated_report = {
                "id": f"report_{datetime.utcnow().timestamp()}",
                "type": "quality_summary",
                "format": format.value,
                "generated_at": datetime.utcnow(),
                "content": formatted_report,
                "metadata": {
                    "datasets_count": len(report_data["datasets_included"]),
                    "reports_analyzed": len(filtered_reports),
                    "time_range": report_data["time_range"]
                }
            }
            
            self.generated_reports.append(generated_report)
            
            return {
                "success": True,
                "report_id": generated_report["id"],
                "content": formatted_report,
                "metadata": generated_report["metadata"]
            }
            
        except Exception as e:
            logger.error(f"Failed to generate quality summary report: {str(e)}")
            return {"error": str(e)}
    
    def generate_trend_analysis_report(self, 
                                     dataset_name: str,
                                     days: int = 30,
                                     format: ReportFormat = ReportFormat.HTML) -> Dict[str, Any]:
        """Generate a detailed trend analysis report for a specific dataset"""
        try:
            # Filter reports for the dataset
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            dataset_reports = [
                r for r in self.quality_history
                if r.dataset_name == dataset_name and r.assessment_timestamp >= cutoff_date
            ]
            
            if not dataset_reports:
                return {"error": f"No quality reports found for dataset {dataset_name}"}
            
            # Sort by timestamp
            dataset_reports.sort(key=lambda x: x.assessment_timestamp)
            
            # Calculate trends
            quality_trend = self._calculate_quality_trend(dataset_reports)
            
            # Generate report content
            report_data = {
                "title": f"Quality Trend Analysis: {dataset_name}",
                "dataset_name": dataset_name,
                "generated_at": datetime.utcnow().isoformat(),
                "analysis_period": f"{days} days",
                "assessments_count": len(dataset_reports),
                "trend_summary": {
                    "overall_direction": quality_trend.trend_direction,
                    "confidence": quality_trend.trend_confidence,
                    "current_score": dataset_reports[-1].overall_score if dataset_reports else 0,
                    "score_change": self._calculate_score_change(dataset_reports),
                    "volatility": self._calculate_score_volatility(dataset_reports)
                },
                "detailed_trends": {
                    "overall_scores": [(r.assessment_timestamp.isoformat(), r.overall_score) 
                                     for r in dataset_reports],
                    "dimension_trends": self._calculate_dimension_trends(dataset_reports),
                    "issue_trends": self._calculate_issue_trends(dataset_reports)
                },
                "statistical_analysis": self._perform_statistical_analysis(dataset_reports),
                "forecasting": self._generate_quality_forecast(dataset_reports),
                "recommendations": self._generate_trend_recommendations(quality_trend, dataset_reports)
            }
            
            # Add visualizations if HTML format
            if format == ReportFormat.HTML:
                report_data["visualizations"] = self._generate_trend_visualizations(dataset_reports)
            
            # Format report
            if format == ReportFormat.HTML:
                formatted_report = self._format_trend_html_report(report_data)
            elif format == ReportFormat.JSON:
                formatted_report = json.dumps(report_data, indent=2, default=str)
            else:
                formatted_report = str(report_data)
            
            return {
                "success": True,
                "content": formatted_report,
                "trend_data": quality_trend
            }
            
        except Exception as e:
            logger.error(f"Failed to generate trend analysis report: {str(e)}")
            return {"error": str(e)}
    
    def generate_automated_reports(self) -> List[Dict[str, Any]]:
        """Generate all scheduled automated reports"""
        generated_reports = []
        
        try:
            current_time = datetime.utcnow()
            
            for template in self.report_templates.values():
                if not template.enabled:
                    continue
                
                # Check if report should be generated based on frequency
                if self._should_generate_report(template, current_time):
                    try:
                        report = self._generate_template_report(template)
                        if report:
                            generated_reports.append(report)
                    except Exception as e:
                        logger.error(f"Failed to generate report from template {template.name}: {str(e)}")
            
            return generated_reports
            
        except Exception as e:
            logger.error(f"Failed to generate automated reports: {str(e)}")
            return []
    
    def _filter_reports(self, dataset_names: Optional[List[str]], 
                       time_range: Optional[Tuple[datetime, datetime]]) -> List[QualityReport]:
        """Filter quality reports based on criteria"""
        filtered_reports = self.quality_history.copy()
        
        if dataset_names:
            filtered_reports = [r for r in filtered_reports if r.dataset_name in dataset_names]
        
        if time_range:
            start_time, end_time = time_range
            filtered_reports = [
                r for r in filtered_reports 
                if start_time <= r.assessment_timestamp <= end_time
            ]
        
        return filtered_reports
    
    def _generate_executive_summary(self, reports: List[QualityReport]) -> Dict[str, Any]:
        """Generate executive summary from quality reports"""
        try:
            if not reports:
                return {}
            
            # Calculate overall statistics
            total_datasets = len(set(r.dataset_name for r in reports))
            avg_score = np.mean([r.overall_score for r in reports])
            total_issues = sum(len(r.issues) for r in reports)
            critical_issues = sum(1 for r in reports for issue in r.issues 
                                if issue.severity == QualitySeverity.CRITICAL)
            
            # Identify best and worst performing datasets
            dataset_scores = defaultdict(list)
            for report in reports:
                dataset_scores[report.dataset_name].append(report.overall_score)
            
            avg_dataset_scores = {
                dataset: np.mean(scores) 
                for dataset, scores in dataset_scores.items()
            }
            
            best_dataset = max(avg_dataset_scores, key=avg_dataset_scores.get) if avg_dataset_scores else None
            worst_dataset = min(avg_dataset_scores, key=avg_dataset_scores.get) if avg_dataset_scores else None
            
            return {
                "total_datasets": total_datasets,
                "total_assessments": len(reports),
                "average_quality_score": round(avg_score, 2),
                "total_issues_found": total_issues,
                "critical_issues": critical_issues,
                "best_performing_dataset": {
                    "name": best_dataset,
                    "score": round(avg_dataset_scores.get(best_dataset, 0), 2)
                } if best_dataset else None,
                "worst_performing_dataset": {
                    "name": worst_dataset,
                    "score": round(avg_dataset_scores.get(worst_dataset, 0), 2)
                } if worst_dataset else None,
                "quality_status": self._determine_overall_status(avg_score, critical_issues)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate executive summary: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_aggregate_metrics(self, reports: List[QualityReport]) -> Dict[str, Any]:
        """Calculate aggregate quality metrics"""
        try:
            if not reports:
                return {}
            
            # Overall score statistics
            scores = [r.overall_score for r in reports]
            
            # Dimension score statistics
            all_dimensions = set()
            for report in reports:
                all_dimensions.update(report.dimension_scores.keys())
            
            dimension_stats = {}
            for dimension in all_dimensions:
                dimension_scores = [
                    r.dimension_scores.get(dimension, 0) 
                    for r in reports 
                    if dimension in r.dimension_scores
                ]
                if dimension_scores:
                    dimension_stats[dimension] = {
                        "average": round(np.mean(dimension_scores), 2),
                        "min": round(min(dimension_scores), 2),
                        "max": round(max(dimension_scores), 2),
                        "std_dev": round(np.std(dimension_scores), 2)
                    }
            
            # Issue statistics
            issue_counts_by_severity = defaultdict(int)
            for report in reports:
                for issue in report.issues:
                    issue_counts_by_severity[issue.severity.value] += 1
            
            return {
                "overall_scores": {
                    "average": round(np.mean(scores), 2),
                    "median": round(np.median(scores), 2),
                    "min": round(min(scores), 2),
                    "max": round(max(scores), 2),
                    "std_dev": round(np.std(scores), 2)
                },
                "dimension_statistics": dimension_stats,
                "issue_statistics": dict(issue_counts_by_severity),
                "data_coverage": {
                    "total_records_assessed": sum(r.total_records for r in reports),
                    "unique_datasets": len(set(r.dataset_name for r in reports)),
                    "assessment_frequency": len(reports) / max(1, len(set(r.dataset_name for r in reports)))
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate aggregate metrics: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_quality_trends(self, reports: List[QualityReport]) -> Dict[str, Any]:
        """Analyze quality trends across all reports"""
        try:
            if len(reports) < 2:
                return {"message": "Insufficient data for trend analysis"}
            
            # Sort reports by timestamp
            sorted_reports = sorted(reports, key=lambda x: x.assessment_timestamp)
            
            # Calculate overall trend
            scores = [r.overall_score for r in sorted_reports]
            timestamps = [r.assessment_timestamp for r in sorted_reports]
            
            # Simple linear trend calculation
            x = np.arange(len(scores))
            slope, intercept = np.polyfit(x, scores, 1)
            
            trend_direction = "improving" if slope > 0.5 else "declining" if slope < -0.5 else "stable"
            
            # Calculate trend by dataset
            dataset_trends = {}
            datasets = set(r.dataset_name for r in sorted_reports)
            
            for dataset in datasets:
                dataset_reports = [r for r in sorted_reports if r.dataset_name == dataset]
                if len(dataset_reports) >= 2:
                    dataset_scores = [r.overall_score for r in dataset_reports]
                    dataset_x = np.arange(len(dataset_scores))
                    dataset_slope, _ = np.polyfit(dataset_x, dataset_scores, 1)
                    
                    dataset_trends[dataset] = {
                        "slope": round(dataset_slope, 3),
                        "direction": "improving" if dataset_slope > 0.5 else "declining" if dataset_slope < -0.5 else "stable",
                        "score_change": round(dataset_scores[-1] - dataset_scores[0], 2),
                        "assessments": len(dataset_reports)
                    }
            
            return {
                "overall_trend": {
                    "direction": trend_direction,
                    "slope": round(slope, 3),
                    "score_change": round(scores[-1] - scores[0], 2),
                    "volatility": round(np.std(scores), 2)
                },
                "dataset_trends": dataset_trends,
                "trend_confidence": min(1.0, len(reports) / 10.0)  # More data = higher confidence
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze quality trends: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_issues(self, reports: List[QualityReport]) -> Dict[str, Any]:
        """Analyze issues across all reports"""
        try:
            all_issues = []
            for report in reports:
                all_issues.extend(report.issues)
            
            if not all_issues:
                return {"message": "No issues found in reports"}
            
            # Issue frequency analysis
            issue_types = defaultdict(int)
            field_issues = defaultdict(int)
            severity_distribution = defaultdict(int)
            
            for issue in all_issues:
                issue_types[issue.rule_name] += 1
                field_issues[issue.field_name] += 1
                severity_distribution[issue.severity.value] += 1
            
            # Most problematic fields
            top_problematic_fields = sorted(
                field_issues.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            # Most common issue types
            top_issue_types = sorted(
                issue_types.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            return {
                "total_issues": len(all_issues),
                "severity_distribution": dict(severity_distribution),
                "most_problematic_fields": top_problematic_fields,
                "most_common_issue_types": top_issue_types,
                "average_issues_per_assessment": round(len(all_issues) / len(reports), 2),
                "critical_issue_rate": severity_distribution.get("critical", 0) / len(all_issues) if all_issues else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze issues: {str(e)}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, reports: List[QualityReport]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        try:
            if not reports:
                return recommendations
            
            # Analyze patterns to generate recommendations
            avg_score = np.mean([r.overall_score for r in reports])
            total_issues = sum(len(r.issues) for r in reports)
            critical_issues = sum(1 for r in reports for issue in r.issues 
                                if issue.severity == QualitySeverity.CRITICAL)
            
            # Score-based recommendations
            if avg_score < 70:
                recommendations.append("Overall data quality is below acceptable threshold. Implement comprehensive data validation at source systems.")
            elif avg_score < 85:
                recommendations.append("Data quality is moderate. Focus on addressing high-severity issues and improving data collection processes.")
            
            # Issue-based recommendations
            if critical_issues > 0:
                recommendations.append(f"Address {critical_issues} critical data quality issues immediately to prevent business impact.")
            
            if total_issues / len(reports) > 5:
                recommendations.append("High issue frequency detected. Consider implementing automated data quality monitoring and real-time validation.")
            
            # Trend-based recommendations
            trend_analysis = self._analyze_quality_trends(reports)
            if trend_analysis.get("overall_trend", {}).get("direction") == "declining":
                recommendations.append("Quality trend is declining. Investigate root causes and implement corrective measures.")
            
            # Dataset-specific recommendations
            dataset_scores = defaultdict(list)
            for report in reports:
                dataset_scores[report.dataset_name].append(report.overall_score)
            
            for dataset, scores in dataset_scores.items():
                avg_dataset_score = np.mean(scores)
                if avg_dataset_score < 60:
                    recommendations.append(f"Dataset '{dataset}' requires immediate attention due to poor quality scores.")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {str(e)}")
            return ["Error generating recommendations. Please review data quality manually."]
    
    def _generate_detailed_findings(self, reports: List[QualityReport]) -> List[Dict[str, Any]]:
        """Generate detailed findings from reports"""
        findings = []
        
        try:
            # Group reports by dataset
            dataset_reports = defaultdict(list)
            for report in reports:
                dataset_reports[report.dataset_name].append(report)
            
            for dataset_name, dataset_reports_list in dataset_reports.items():
                latest_report = max(dataset_reports_list, key=lambda x: x.assessment_timestamp)
                
                finding = {
                    "dataset_name": dataset_name,
                    "latest_assessment": latest_report.assessment_timestamp.isoformat(),
                    "current_score": latest_report.overall_score,
                    "total_records": latest_report.total_records,
                    "issues_count": len(latest_report.issues),
                    "dimension_scores": latest_report.dimension_scores,
                    "critical_issues": [
                        {
                            "field": issue.field_name,
                            "description": issue.issue_description,
                            "affected_records": issue.affected_records
                        }
                        for issue in latest_report.issues
                        if issue.severity == QualitySeverity.CRITICAL
                    ],
                    "assessment_history_count": len(dataset_reports_list)
                }
                
                findings.append(finding)
            
            # Sort by score (worst first)
            findings.sort(key=lambda x: x["current_score"])
            
            return findings
            
        except Exception as e:
            logger.error(f"Failed to generate detailed findings: {str(e)}")
            return []
    
    def _format_html_report(self, report_data: Dict[str, Any]) -> str:
        """Format report as HTML"""
        try:
            html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>{title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
                    .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 3px; }}
                    .critical {{ color: #dc3545; font-weight: bold; }}
                    .good {{ color: #28a745; font-weight: bold; }}
                    .warning {{ color: #ffc107; font-weight: bold; }}
                    table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{title}</h1>
                    <p>Generated: {generated_at}</p>
                    <p>Time Range: {time_range}</p>
                </div>
                
                {executive_summary_html}
                {metrics_html}
                {trends_html}
                {issues_html}
                {recommendations_html}
                {findings_html}
            </body>
            </html>
            """
            
            # Generate HTML sections
            executive_summary_html = self._format_executive_summary_html(report_data.get("executive_summary", {}))
            metrics_html = self._format_metrics_html(report_data.get("quality_metrics", {}))
            trends_html = self._format_trends_html(report_data.get("trend_analysis", {}))
            issues_html = self._format_issues_html(report_data.get("issue_analysis", {}))
            recommendations_html = self._format_recommendations_html(report_data.get("recommendations", []))
            findings_html = self._format_findings_html(report_data.get("detailed_findings", []))
            
            return html_template.format(
                title=report_data.get("title", "Data Quality Report"),
                generated_at=report_data.get("generated_at", ""),
                time_range=report_data.get("time_range", ""),
                executive_summary_html=executive_summary_html,
                metrics_html=metrics_html,
                trends_html=trends_html,
                issues_html=issues_html,
                recommendations_html=recommendations_html,
                findings_html=findings_html
            )
            
        except Exception as e:
            logger.error(f"Failed to format HTML report: {str(e)}")
            return f"<html><body><h1>Report Generation Error</h1><p>{str(e)}</p></body></html>"
    
    def _format_executive_summary_html(self, summary: Dict[str, Any]) -> str:
        """Format executive summary as HTML"""
        if not summary:
            return ""
        
        status_class = "good" if summary.get("quality_status") == "excellent" else "warning" if summary.get("quality_status") == "good" else "critical"
        
        return f"""
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="metric">Total Datasets: <strong>{summary.get('total_datasets', 0)}</strong></div>
            <div class="metric">Average Quality Score: <strong class="{status_class}">{summary.get('average_quality_score', 0)}%</strong></div>
            <div class="metric">Total Issues: <strong>{summary.get('total_issues_found', 0)}</strong></div>
            <div class="metric">Critical Issues: <strong class="critical">{summary.get('critical_issues', 0)}</strong></div>
            <p><strong>Overall Status:</strong> <span class="{status_class}">{summary.get('quality_status', 'Unknown').title()}</span></p>
        </div>
        """
    
    def _format_metrics_html(self, metrics: Dict[str, Any]) -> str:
        """Format metrics as HTML"""
        if not metrics:
            return ""
        
        overall_scores = metrics.get("overall_scores", {})
        
        return f"""
        <div class="section">
            <h2>Quality Metrics</h2>
            <h3>Overall Score Statistics</h3>
            <div class="metric">Average: <strong>{overall_scores.get('average', 0)}%</strong></div>
            <div class="metric">Median: <strong>{overall_scores.get('median', 0)}%</strong></div>
            <div class="metric">Range: <strong>{overall_scores.get('min', 0)}% - {overall_scores.get('max', 0)}%</strong></div>
            <div class="metric">Std Dev: <strong>{overall_scores.get('std_dev', 0)}</strong></div>
        </div>
        """
    
    def _format_trends_html(self, trends: Dict[str, Any]) -> str:
        """Format trends as HTML"""
        if not trends:
            return ""
        
        overall_trend = trends.get("overall_trend", {})
        direction = overall_trend.get("direction", "unknown")
        direction_class = "good" if direction == "improving" else "critical" if direction == "declining" else "warning"
        
        return f"""
        <div class="section">
            <h2>Trend Analysis</h2>
            <p><strong>Overall Trend:</strong> <span class="{direction_class}">{direction.title()}</span></p>
            <p><strong>Score Change:</strong> {overall_trend.get('score_change', 0)}</p>
            <p><strong>Volatility:</strong> {overall_trend.get('volatility', 0)}</p>
        </div>
        """
    
    def _format_issues_html(self, issues: Dict[str, Any]) -> str:
        """Format issues as HTML"""
        if not issues:
            return ""
        
        return f"""
        <div class="section">
            <h2>Issue Analysis</h2>
            <p><strong>Total Issues:</strong> {issues.get('total_issues', 0)}</p>
            <p><strong>Critical Issue Rate:</strong> {issues.get('critical_issue_rate', 0):.1%}</p>
        </div>
        """
    
    def _format_recommendations_html(self, recommendations: List[str]) -> str:
        """Format recommendations as HTML"""
        if not recommendations:
            return ""
        
        recommendations_html = "<ul>"
        for rec in recommendations:
            recommendations_html += f"<li>{rec}</li>"
        recommendations_html += "</ul>"
        
        return f"""
        <div class="section">
            <h2>Recommendations</h2>
            {recommendations_html}
        </div>
        """
    
    def _format_findings_html(self, findings: List[Dict[str, Any]]) -> str:
        """Format detailed findings as HTML"""
        if not findings:
            return ""
        
        findings_html = "<table><tr><th>Dataset</th><th>Score</th><th>Issues</th><th>Records</th></tr>"
        
        for finding in findings[:10]:  # Limit to top 10
            score_class = "good" if finding["current_score"] >= 80 else "warning" if finding["current_score"] >= 60 else "critical"
            findings_html += f"""
            <tr>
                <td>{finding['dataset_name']}</td>
                <td class="{score_class}">{finding['current_score']:.1f}%</td>
                <td>{finding['issues_count']}</td>
                <td>{finding['total_records']:,}</td>
            </tr>
            """
        
        findings_html += "</table>"
        
        return f"""
        <div class="section">
            <h2>Dataset Details</h2>
            {findings_html}
        </div>
        """
    
    def _determine_overall_status(self, avg_score: float, critical_issues: int) -> str:
        """Determine overall quality status"""
        if critical_issues > 0:
            return "critical"
        elif avg_score >= 90:
            return "excellent"
        elif avg_score >= 75:
            return "good"
        elif avg_score >= 60:
            return "fair"
        else:
            return "poor"
    
    def _format_time_range(self, time_range: Optional[Tuple[datetime, datetime]]) -> str:
        """Format time range for display"""
        if not time_range:
            return "All available data"
        
        start_time, end_time = time_range
        return f"{start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}"
    
    def _calculate_quality_trend(self, reports: List[QualityReport]) -> QualityTrend:
        """Calculate quality trend for a dataset"""
        if len(reports) < 2:
            return QualityTrend(
                dataset_name=reports[0].dataset_name if reports else "",
                time_period="insufficient_data",
                score_trend=[],
                issue_trend=[],
                dimension_trends={},
                trend_direction="unknown",
                trend_confidence=0.0
            )
        
        # Calculate score trend
        score_trend = [(r.assessment_timestamp, r.overall_score) for r in reports]
        
        # Calculate issue trend
        issue_trend = [(r.assessment_timestamp, len(r.issues)) for r in reports]
        
        # Calculate dimension trends
        dimension_trends = {}
        all_dimensions = set()
        for report in reports:
            all_dimensions.update(report.dimension_scores.keys())
        
        for dimension in all_dimensions:
            dimension_trend = []
            for report in reports:
                if dimension in report.dimension_scores:
                    dimension_trend.append((report.assessment_timestamp, report.dimension_scores[dimension]))
            dimension_trends[dimension] = dimension_trend
        
        # Determine trend direction
        scores = [r.overall_score for r in reports]
        if len(scores) >= 2:
            slope, _ = np.polyfit(range(len(scores)), scores, 1)
            if slope > 0.5:
                trend_direction = "improving"
            elif slope < -0.5:
                trend_direction = "declining"
            else:
                trend_direction = "stable"
        else:
            trend_direction = "unknown"
        
        # Calculate confidence based on data points and consistency
        confidence = min(1.0, len(reports) / 10.0)
        
        return QualityTrend(
            dataset_name=reports[0].dataset_name,
            time_period=f"{len(reports)} assessments",
            score_trend=score_trend,
            issue_trend=issue_trend,
            dimension_trends=dimension_trends,
            trend_direction=trend_direction,
            trend_confidence=confidence
        )
    
    def _should_generate_report(self, template: ReportTemplate, current_time: datetime) -> bool:
        """Check if a report should be generated based on schedule"""
        # This is a simplified implementation
        # In production, you'd want more sophisticated scheduling logic
        return True
    
    def _generate_template_report(self, template: ReportTemplate) -> Optional[Dict[str, Any]]:
        """Generate a report from a template"""
        try:
            # Apply template filters
            filtered_reports = self.quality_history
            
            if template.filters.get("datasets"):
                filtered_reports = [
                    r for r in filtered_reports 
                    if r.dataset_name in template.filters["datasets"]
                ]
            
            if template.filters.get("days"):
                cutoff_date = datetime.utcnow() - timedelta(days=template.filters["days"])
                filtered_reports = [
                    r for r in filtered_reports 
                    if r.assessment_timestamp >= cutoff_date
                ]
            
            # Generate report based on template format
            if template.format == ReportFormat.HTML:
                return self.generate_quality_summary_report(
                    dataset_names=template.filters.get("datasets"),
                    format=ReportFormat.HTML
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to generate template report: {str(e)}")
            return None