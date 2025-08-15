"""
Interactive Dashboard Generator for AI Data Readiness Platform

This module generates interactive web-based dashboards for data readiness visualization,
real-time monitoring displays, and customizable reporting interfaces.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..engines.ai_readiness_reporting_engine import AIReadinessReport, IndustryStandard
from ..models.base_models import AIReadinessScore, QualityReport, BiasReport

logger = logging.getLogger(__name__)


class DashboardGenerator:
    """
    Interactive dashboard generator for AI data readiness visualization
    """
    
    def __init__(self):
        self.template_dir = Path(__file__).parent / "templates"
        self.static_dir = Path(__file__).parent / "static"
        
    def generate_readiness_dashboard(
        self,
        report: AIReadinessReport,
        include_real_time: bool = True,
        customization_options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate interactive AI readiness dashboard
        
        Args:
            report: AI readiness report data
            include_real_time: Include real-time monitoring components
            customization_options: Dashboard customization options
            
        Returns:
            HTML content for the dashboard
        """
        try:
            logger.info(f"Generating readiness dashboard for dataset {report.dataset_id}")
            
            # Prepare dashboard data
            dashboard_data = self._prepare_dashboard_data(report)
            
            # Generate dashboard HTML
            html_content = self._generate_dashboard_html(
                dashboard_data,
                include_real_time,
                customization_options or {}
            )
            
            logger.info("Readiness dashboard generated successfully")
            return html_content
            
        except Exception as e:
            logger.error(f"Error generating readiness dashboard: {str(e)}")
            raise
    
    def generate_monitoring_dashboard(
        self,
        datasets: List[str],
        refresh_interval: int = 30,
        alert_thresholds: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Generate real-time monitoring dashboard
        
        Args:
            datasets: List of dataset IDs to monitor
            refresh_interval: Dashboard refresh interval in seconds
            alert_thresholds: Custom alert thresholds
            
        Returns:
            HTML content for the monitoring dashboard
        """
        try:
            logger.info(f"Generating monitoring dashboard for {len(datasets)} datasets")
            
            # Prepare monitoring data structure
            monitoring_data = {
                "datasets": datasets,
                "refresh_interval": refresh_interval,
                "alert_thresholds": alert_thresholds or self._get_default_alert_thresholds(),
                "generated_at": datetime.utcnow().isoformat()
            }
            
            # Generate monitoring dashboard HTML
            html_content = self._generate_monitoring_html(monitoring_data)
            
            logger.info("Monitoring dashboard generated successfully")
            return html_content
            
        except Exception as e:
            logger.error(f"Error generating monitoring dashboard: {str(e)}")
            raise
    
    def generate_comparison_dashboard(
        self,
        reports: List[AIReadinessReport],
        comparison_dimensions: List[str] = None
    ) -> str:
        """
        Generate comparison dashboard for multiple reports
        
        Args:
            reports: List of AI readiness reports to compare
            comparison_dimensions: Dimensions to compare
            
        Returns:
            HTML content for the comparison dashboard
        """
        try:
            logger.info(f"Generating comparison dashboard for {len(reports)} reports")
            
            if comparison_dimensions is None:
                comparison_dimensions = [
                    "overall_score", "data_quality_score", "feature_quality_score",
                    "bias_score", "compliance_score"
                ]
            
            # Prepare comparison data
            comparison_data = self._prepare_comparison_data(reports, comparison_dimensions)
            
            # Generate comparison dashboard HTML
            html_content = self._generate_comparison_html(comparison_data)
            
            logger.info("Comparison dashboard generated successfully")
            return html_content
            
        except Exception as e:
            logger.error(f"Error generating comparison dashboard: {str(e)}")
            raise
    
    def _prepare_dashboard_data(self, report: AIReadinessReport) -> Dict[str, Any]:
        """Prepare data for dashboard visualization"""
        return {
            "dataset_id": report.dataset_id,
            "overall_score": report.overall_score,
            "dimension_scores": report.dimension_scores,
            "benchmark_comparison": report.benchmark_comparison,
            "improvement_actions": [
                {
                    "title": action.title,
                    "priority": action.priority,
                    "category": action.category,
                    "description": action.description,
                    "timeline": action.estimated_timeline,
                    "impact": action.expected_impact,
                    "effort": action.estimated_effort
                }
                for action in report.improvement_actions
            ],
            "compliance_status": report.compliance_status,
            "risk_assessment": report.risk_assessment,
            "industry": report.industry_benchmark.industry.value,
            "generated_at": report.generated_at.isoformat()
        }
    
    def _prepare_comparison_data(
        self,
        reports: List[AIReadinessReport],
        dimensions: List[str]
    ) -> Dict[str, Any]:
        """Prepare data for comparison dashboard"""
        comparison_data = {
            "datasets": [],
            "dimensions": dimensions,
            "scores": {dim: [] for dim in dimensions},
            "metadata": []
        }
        
        for report in reports:
            comparison_data["datasets"].append(report.dataset_id)
            comparison_data["metadata"].append({
                "dataset_id": report.dataset_id,
                "industry": report.industry_benchmark.industry.value,
                "generated_at": report.generated_at.isoformat()
            })
            
            # Extract scores for each dimension
            for dim in dimensions:
                if dim == "overall_score":
                    comparison_data["scores"][dim].append(report.overall_score)
                else:
                    comparison_data["scores"][dim].append(
                        report.dimension_scores.get(dim, 0.0)
                    )
        
        return comparison_data
    
    def _get_default_alert_thresholds(self) -> Dict[str, float]:
        """Get default alert thresholds for monitoring"""
        return {
            "overall_score_min": 0.7,
            "data_quality_min": 0.8,
            "bias_score_min": 0.75,
            "compliance_score_min": 0.85,
            "drift_threshold": 0.3
        }
    
    def _generate_dashboard_html(
        self,
        data: Dict[str, Any],
        include_real_time: bool,
        customization: Dict[str, Any]
    ) -> str:
        """Generate main dashboard HTML"""
        
        # Get customization options
        theme = customization.get("theme", "professional")
        color_scheme = customization.get("color_scheme", "blue")
        show_details = customization.get("show_details", True)
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Data Readiness Dashboard - {data['dataset_id']}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/date-fns@2.29.3/index.min.js"></script>
    <style>
        {self._get_dashboard_css(theme, color_scheme)}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <header class="dashboard-header">
            <h1>AI Data Readiness Dashboard</h1>
            <div class="dataset-info">
                <h2>{data['dataset_id']}</h2>
                <span class="timestamp">Generated: {data['generated_at']}</span>
                {'<span class="live-indicator">🟢 Live</span>' if include_real_time else ''}
            </div>
        </header>
        
        <div class="dashboard-grid">
            <!-- Overall Score Card -->
            <div class="card score-card">
                <h3>Overall AI Readiness</h3>
                <div class="score-display">
                    <div class="score-circle" data-score="{data['overall_score']}">
                        <span class="score-value">{data['overall_score']:.1%}</span>
                    </div>
                    <div class="score-label">
                        {self._get_readiness_label(data['overall_score'])}
                    </div>
                </div>
            </div>
            
            <!-- Dimension Scores -->
            <div class="card dimensions-card">
                <h3>Dimension Scores</h3>
                <div class="dimensions-grid">
                    {self._generate_dimension_cards(data['dimension_scores'])}
                </div>
            </div>
            
            <!-- Benchmark Comparison -->
            <div class="card benchmark-card">
                <h3>Industry Benchmark Comparison</h3>
                <div class="benchmark-chart">
                    <canvas id="benchmarkChart"></canvas>
                </div>
                <div class="industry-info">
                    Industry: <strong>{data['industry'].replace('_', ' ').title()}</strong>
                </div>
            </div>
            
            <!-- Improvement Actions -->
            <div class="card actions-card">
                <h3>Priority Improvement Actions</h3>
                <div class="actions-list">
                    {self._generate_action_items(data['improvement_actions'][:5])}
                </div>
            </div>
            
            <!-- Compliance Status -->
            <div class="card compliance-card">
                <h3>Compliance Status</h3>
                <div class="compliance-grid">
                    {self._generate_compliance_status(data['compliance_status'])}
                </div>
            </div>
            
            <!-- Risk Assessment -->
            <div class="card risk-card">
                <h3>Risk Assessment</h3>
                <div class="risk-items">
                    {self._generate_risk_items(data['risk_assessment'])}
                </div>
            </div>
            
            {'<div class="card monitoring-card"><h3>Real-time Monitoring</h3><div id="realTimeChart"><canvas id="monitoringChart"></canvas></div></div>' if include_real_time else ''}
        </div>
    </div>
    
    <script>
        {self._generate_dashboard_js(data, include_real_time)}
    </script>
</body>
</html>
        """
        
        return html_template.strip()
    
    def _generate_monitoring_html(self, data: Dict[str, Any]) -> str:
        """Generate monitoring dashboard HTML"""
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Data Readiness Monitoring Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/date-fns@2.29.3/index.min.js"></script>
    <style>
        {self._get_monitoring_css()}
    </style>
</head>
<body>
    <div class="monitoring-container">
        <header class="monitoring-header">
            <h1>Real-time AI Data Readiness Monitoring</h1>
            <div class="monitoring-controls">
                <span class="refresh-indicator">🔄 Auto-refresh: {data['refresh_interval']}s</span>
                <button id="pauseBtn" class="control-btn">⏸️ Pause</button>
                <button id="refreshBtn" class="control-btn">🔄 Refresh</button>
            </div>
        </header>
        
        <div class="monitoring-grid">
            <!-- System Status -->
            <div class="card status-card">
                <h3>System Status</h3>
                <div class="status-indicators">
                    <div class="status-item">
                        <span class="status-dot green"></span>
                        <span>Data Ingestion: Active</span>
                    </div>
                    <div class="status-item">
                        <span class="status-dot green"></span>
                        <span>Quality Assessment: Running</span>
                    </div>
                    <div class="status-item">
                        <span class="status-dot yellow"></span>
                        <span>Bias Analysis: Warning</span>
                    </div>
                    <div class="status-item">
                        <span class="status-dot green"></span>
                        <span>Compliance Check: Passed</span>
                    </div>
                </div>
            </div>
            
            <!-- Dataset Overview -->
            <div class="card datasets-card">
                <h3>Monitored Datasets ({len(data['datasets'])})</h3>
                <div class="datasets-list">
                    {self._generate_dataset_monitoring_items(data['datasets'])}
                </div>
            </div>
            
            <!-- Real-time Metrics -->
            <div class="card metrics-card">
                <h3>Real-time Metrics</h3>
                <div class="metrics-chart">
                    <canvas id="metricsChart"></canvas>
                </div>
            </div>
            
            <!-- Alerts -->
            <div class="card alerts-card">
                <h3>Active Alerts</h3>
                <div class="alerts-list" id="alertsList">
                    <div class="alert-item warning">
                        <span class="alert-icon">⚠️</span>
                        <div class="alert-content">
                            <strong>Bias Score Below Threshold</strong>
                            <p>Dataset: customer_data_2024 - Score: 0.72</p>
                            <span class="alert-time">2 minutes ago</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Performance Trends -->
            <div class="card trends-card">
                <h3>Performance Trends</h3>
                <div class="trends-chart">
                    <canvas id="trendsChart"></canvas>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        {self._generate_monitoring_js(data)}
    </script>
</body>
</html>
        """
        
        return html_template.strip()
    
    def _generate_comparison_html(self, data: Dict[str, Any]) -> str:
        """Generate comparison dashboard HTML"""
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Data Readiness Comparison Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        {self._get_comparison_css()}
    </style>
</head>
<body>
    <div class="comparison-container">
        <header class="comparison-header">
            <h1>Dataset Comparison Dashboard</h1>
            <div class="comparison-info">
                <span>Comparing {len(data['datasets'])} datasets across {len(data['dimensions'])} dimensions</span>
            </div>
        </header>
        
        <div class="comparison-grid">
            <!-- Comparison Chart -->
            <div class="card chart-card">
                <h3>Score Comparison</h3>
                <div class="chart-container">
                    <canvas id="comparisonChart"></canvas>
                </div>
            </div>
            
            <!-- Dataset Rankings -->
            <div class="card rankings-card">
                <h3>Dataset Rankings</h3>
                <div class="rankings-list">
                    {self._generate_rankings(data)}
                </div>
            </div>
            
            <!-- Dimension Analysis -->
            <div class="card analysis-card">
                <h3>Dimension Analysis</h3>
                <div class="dimension-analysis">
                    {self._generate_dimension_analysis(data)}
                </div>
            </div>
            
            <!-- Dataset Details -->
            <div class="card details-card">
                <h3>Dataset Details</h3>
                <div class="details-table">
                    {self._generate_details_table(data)}
                </div>
            </div>
        </div>
    </div>
    
    <script>
        {self._generate_comparison_js(data)}
    </script>
</body>
</html>
        """
        
        return html_template.strip()
    
    def _get_dashboard_css(self, theme: str, color_scheme: str) -> str:
        """Generate CSS for main dashboard"""
        
        # Color schemes
        colors = {
            "blue": {
                "primary": "#2563eb",
                "secondary": "#3b82f6",
                "accent": "#60a5fa",
                "success": "#10b981",
                "warning": "#f59e0b",
                "danger": "#ef4444"
            },
            "green": {
                "primary": "#059669",
                "secondary": "#10b981",
                "accent": "#34d399",
                "success": "#10b981",
                "warning": "#f59e0b",
                "danger": "#ef4444"
            },
            "purple": {
                "primary": "#7c3aed",
                "secondary": "#8b5cf6",
                "accent": "#a78bfa",
                "success": "#10b981",
                "warning": "#f59e0b",
                "danger": "#ef4444"
            }
        }
        
        color_palette = colors.get(color_scheme, colors["blue"])
        
        return f"""
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f8fafc;
            color: #334155;
            line-height: 1.6;
        }}
        
        .dashboard-container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .dashboard-header {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .dashboard-header h1 {{
            color: {color_palette['primary']};
            font-size: 2.5rem;
            font-weight: 700;
        }}
        
        .dataset-info h2 {{
            color: #1e293b;
            font-size: 1.5rem;
            margin-bottom: 5px;
        }}
        
        .timestamp {{
            color: #64748b;
            font-size: 0.9rem;
        }}
        
        .live-indicator {{
            background: #10b981;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            margin-left: 10px;
        }}
        
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
        }}
        
        .card {{
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid #e2e8f0;
        }}
        
        .card h3 {{
            color: #1e293b;
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 20px;
            border-bottom: 2px solid {color_palette['primary']};
            padding-bottom: 10px;
        }}
        
        .score-card {{
            text-align: center;
            background: linear-gradient(135deg, {color_palette['primary']}, {color_palette['secondary']});
            color: white;
        }}
        
        .score-card h3 {{
            color: white;
            border-bottom-color: rgba(255,255,255,0.3);
        }}
        
        .score-circle {{
            width: 120px;
            height: 120px;
            border-radius: 50%;
            background: rgba(255,255,255,0.2);
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 20px auto;
            border: 4px solid rgba(255,255,255,0.3);
        }}
        
        .score-value {{
            font-size: 2rem;
            font-weight: 700;
        }}
        
        .score-label {{
            font-size: 1.1rem;
            margin-top: 10px;
            opacity: 0.9;
        }}
        
        .dimensions-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        
        .dimension-item {{
            text-align: center;
            padding: 15px;
            background: #f8fafc;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }}
        
        .dimension-score {{
            font-size: 1.5rem;
            font-weight: 600;
            color: {color_palette['primary']};
            margin-bottom: 5px;
        }}
        
        .dimension-label {{
            font-size: 0.9rem;
            color: #64748b;
            text-transform: capitalize;
        }}
        
        .actions-list {{
            space-y: 15px;
        }}
        
        .action-item {{
            padding: 15px;
            border-left: 4px solid {color_palette['primary']};
            background: #f8fafc;
            border-radius: 0 8px 8px 0;
            margin-bottom: 15px;
        }}
        
        .action-item.high {{
            border-left-color: {color_palette['danger']};
        }}
        
        .action-item.medium {{
            border-left-color: {color_palette['warning']};
        }}
        
        .action-item.low {{
            border-left-color: {color_palette['success']};
        }}
        
        .action-title {{
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 5px;
        }}
        
        .action-meta {{
            font-size: 0.9rem;
            color: #64748b;
            display: flex;
            gap: 15px;
        }}
        
        .compliance-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .compliance-item {{
            display: flex;
            align-items: center;
            padding: 10px;
            border-radius: 6px;
            background: #f8fafc;
        }}
        
        .compliance-item.compliant {{
            background: #dcfce7;
            color: #166534;
        }}
        
        .compliance-item.non-compliant {{
            background: #fef2f2;
            color: #991b1b;
        }}
        
        .compliance-icon {{
            margin-right: 10px;
            font-size: 1.2rem;
        }}
        
        .risk-items {{
            space-y: 10px;
        }}
        
        .risk-item {{
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 10px;
        }}
        
        .risk-item.low {{
            background: #dcfce7;
            color: #166534;
        }}
        
        .risk-item.medium {{
            background: #fef3c7;
            color: #92400e;
        }}
        
        .risk-item.high {{
            background: #fef2f2;
            color: #991b1b;
        }}
        
        @media (max-width: 768px) {{
            .dashboard-grid {{
                grid-template-columns: 1fr;
            }}
            
            .dashboard-header {{
                flex-direction: column;
                text-align: center;
                gap: 15px;
            }}
        }}
        """
    
    def _get_monitoring_css(self) -> str:
        """Generate CSS for monitoring dashboard"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #0f172a;
            color: #e2e8f0;
            line-height: 1.6;
        }
        
        .monitoring-container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .monitoring-header {
            background: #1e293b;
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 25px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border: 1px solid #334155;
        }
        
        .monitoring-header h1 {
            color: #60a5fa;
            font-size: 2rem;
            font-weight: 700;
        }
        
        .monitoring-controls {
            display: flex;
            gap: 15px;
            align-items: center;
        }
        
        .refresh-indicator {
            color: #10b981;
            font-size: 0.9rem;
        }
        
        .control-btn {
            background: #3b82f6;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9rem;
        }
        
        .control-btn:hover {
            background: #2563eb;
        }
        
        .monitoring-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 25px;
        }
        
        .card {
            background: #1e293b;
            border-radius: 12px;
            padding: 25px;
            border: 1px solid #334155;
        }
        
        .card h3 {
            color: #f1f5f9;
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 20px;
            border-bottom: 2px solid #3b82f6;
            padding-bottom: 10px;
        }
        
        .status-indicators {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        
        .status-item {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
        
        .status-dot.green {
            background: #10b981;
        }
        
        .status-dot.yellow {
            background: #f59e0b;
        }
        
        .status-dot.red {
            background: #ef4444;
        }
        
        .datasets-list {
            max-height: 300px;
            overflow-y: auto;
        }
        
        .dataset-item {
            padding: 12px;
            background: #0f172a;
            border-radius: 6px;
            margin-bottom: 10px;
            border: 1px solid #334155;
        }
        
        .alerts-list {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .alert-item {
            display: flex;
            gap: 12px;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 12px;
        }
        
        .alert-item.warning {
            background: rgba(245, 158, 11, 0.1);
            border: 1px solid #f59e0b;
        }
        
        .alert-item.error {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid #ef4444;
        }
        
        .alert-icon {
            font-size: 1.2rem;
        }
        
        .alert-content strong {
            color: #f1f5f9;
            display: block;
            margin-bottom: 5px;
        }
        
        .alert-time {
            font-size: 0.8rem;
            color: #94a3b8;
        }
        """
    
    def _get_comparison_css(self) -> str:
        """Generate CSS for comparison dashboard"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f8fafc;
            color: #334155;
            line-height: 1.6;
        }
        
        .comparison-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .comparison-header {
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 25px;
            text-align: center;
        }
        
        .comparison-header h1 {
            color: #1e293b;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 10px;
        }
        
        .comparison-info {
            color: #64748b;
            font-size: 1.1rem;
        }
        
        .comparison-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 25px;
        }
        
        .card {
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid #e2e8f0;
        }
        
        .card h3 {
            color: #1e293b;
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 20px;
            border-bottom: 2px solid #3b82f6;
            padding-bottom: 10px;
        }
        
        .chart-card {
            grid-row: span 2;
        }
        
        .chart-container {
            height: 400px;
        }
        
        .rankings-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .ranking-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px;
            background: #f8fafc;
            border-radius: 6px;
            border-left: 4px solid #3b82f6;
        }
        
        .ranking-item:first-child {
            border-left-color: #10b981;
            background: #dcfce7;
        }
        
        .ranking-score {
            font-weight: 600;
            color: #1e293b;
        }
        
        .details-table {
            overflow-x: auto;
        }
        
        .details-table table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .details-table th,
        .details-table td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .details-table th {
            background: #f8fafc;
            font-weight: 600;
            color: #1e293b;
        }
        """
    
    def _generate_dimension_cards(self, dimension_scores: Dict[str, float]) -> str:
        """Generate HTML for dimension score cards"""
        cards_html = ""
        for dimension, score in dimension_scores.items():
            cards_html += f"""
            <div class="dimension-item">
                <div class="dimension-score">{score:.1%}</div>
                <div class="dimension-label">{dimension.replace('_', ' ')}</div>
            </div>
            """
        return cards_html
    
    def _generate_action_items(self, actions: List[Dict[str, Any]]) -> str:
        """Generate HTML for improvement action items"""
        actions_html = ""
        for action in actions:
            actions_html += f"""
            <div class="action-item {action['priority']}">
                <div class="action-title">{action['title']}</div>
                <div class="action-description">{action['description']}</div>
                <div class="action-meta">
                    <span>Priority: {action['priority'].title()}</span>
                    <span>Timeline: {action['timeline']} days</span>
                    <span>Impact: +{action['impact']:.1%}</span>
                </div>
            </div>
            """
        return actions_html
    
    def _generate_compliance_status(self, compliance_status: Dict[str, bool]) -> str:
        """Generate HTML for compliance status"""
        compliance_html = ""
        for regulation, compliant in compliance_status.items():
            status_class = "compliant" if compliant else "non-compliant"
            icon = "✓" if compliant else "✗"
            compliance_html += f"""
            <div class="compliance-item {status_class}">
                <span class="compliance-icon">{icon}</span>
                <span>{regulation.replace('_', ' ').title()}</span>
            </div>
            """
        return compliance_html
    
    def _generate_risk_items(self, risk_assessment: Dict[str, str]) -> str:
        """Generate HTML for risk assessment items"""
        risk_html = ""
        for risk_type, assessment in risk_assessment.items():
            risk_level = "low" if "Low" in assessment else "medium" if "Medium" in assessment else "high"
            risk_html += f"""
            <div class="risk-item {risk_level}">
                <strong>{risk_type.replace('_', ' ').title()}</strong>
                <p>{assessment}</p>
            </div>
            """
        return risk_html
    
    def _generate_dataset_monitoring_items(self, datasets: List[str]) -> str:
        """Generate HTML for dataset monitoring items"""
        items_html = ""
        for dataset in datasets:
            # Simulate some status data
            status = "healthy"  # This would come from real monitoring data
            score = 0.85  # This would come from real monitoring data
            
            items_html += f"""
            <div class="dataset-item">
                <div class="dataset-name">{dataset}</div>
                <div class="dataset-status">Status: {status}</div>
                <div class="dataset-score">Score: {score:.1%}</div>
            </div>
            """
        return items_html
    
    def _generate_rankings(self, data: Dict[str, Any]) -> str:
        """Generate HTML for dataset rankings"""
        # Calculate overall scores for ranking
        overall_scores = data['scores'].get('overall_score', [])
        datasets = data['datasets']
        
        # Create ranking pairs and sort
        rankings = list(zip(datasets, overall_scores))
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        rankings_html = ""
        for i, (dataset, score) in enumerate(rankings, 1):
            rankings_html += f"""
            <div class="ranking-item">
                <span>#{i} {dataset}</span>
                <span class="ranking-score">{score:.1%}</span>
            </div>
            """
        return rankings_html
    
    def _generate_dimension_analysis(self, data: Dict[str, Any]) -> str:
        """Generate HTML for dimension analysis"""
        analysis_html = ""
        for dimension in data['dimensions']:
            scores = data['scores'][dimension]
            avg_score = sum(scores) / len(scores) if scores else 0
            max_score = max(scores) if scores else 0
            min_score = min(scores) if scores else 0
            
            analysis_html += f"""
            <div class="dimension-stats">
                <h4>{dimension.replace('_', ' ').title()}</h4>
                <div class="stats-grid">
                    <div>Average: {avg_score:.1%}</div>
                    <div>Best: {max_score:.1%}</div>
                    <div>Worst: {min_score:.1%}</div>
                </div>
            </div>
            """
        return analysis_html
    
    def _generate_details_table(self, data: Dict[str, Any]) -> str:
        """Generate HTML for details table"""
        table_html = """
        <table>
            <thead>
                <tr>
                    <th>Dataset</th>
                    <th>Industry</th>
                    <th>Overall Score</th>
                    <th>Generated</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for i, metadata in enumerate(data['metadata']):
            overall_score = data['scores']['overall_score'][i] if i < len(data['scores']['overall_score']) else 0
            table_html += f"""
                <tr>
                    <td>{metadata['dataset_id']}</td>
                    <td>{metadata['industry'].replace('_', ' ').title()}</td>
                    <td>{overall_score:.1%}</td>
                    <td>{metadata['generated_at'][:10]}</td>
                </tr>
            """
        
        table_html += """
            </tbody>
        </table>
        """
        return table_html
    
    def _get_readiness_label(self, score: float) -> str:
        """Get readiness label based on score"""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Good"
        elif score >= 0.7:
            return "Fair"
        else:
            return "Needs Improvement"
    
    def _generate_dashboard_js(self, data: Dict[str, Any], include_real_time: bool) -> str:
        """Generate JavaScript for dashboard functionality"""
        
        js_code = f"""
        // Dashboard data
        const dashboardData = {json.dumps(data)};
        
        // Initialize charts when page loads
        document.addEventListener('DOMContentLoaded', function() {{
            initializeBenchmarkChart();
            {'initializeRealTimeMonitoring();' if include_real_time else ''}
            initializeScoreCircles();
        }});
        
        function initializeBenchmarkChart() {{
            const ctx = document.getElementById('benchmarkChart').getContext('2d');
            const benchmarkData = dashboardData.benchmark_comparison;
            
            new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: Object.keys(benchmarkData).map(key => key.replace('_', ' ').replace('gap', '').trim()),
                    datasets: [{{
                        label: 'Gap from Industry Standard',
                        data: Object.values(benchmarkData),
                        backgroundColor: function(context) {{
                            const value = context.parsed.y;
                            return value >= 0 ? '#10b981' : '#ef4444';
                        }},
                        borderColor: function(context) {{
                            const value = context.parsed.y;
                            return value >= 0 ? '#059669' : '#dc2626';
                        }},
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{
                            display: false
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            grid: {{
                                color: '#e2e8f0'
                            }}
                        }},
                        x: {{
                            grid: {{
                                display: false
                            }}
                        }}
                    }}
                }}
            }});
        }}
        
        function initializeScoreCircles() {{
            const circles = document.querySelectorAll('.score-circle');
            circles.forEach(circle => {{
                const score = parseFloat(circle.dataset.score);
                const circumference = 2 * Math.PI * 50; // radius = 50
                const strokeDasharray = circumference;
                const strokeDashoffset = circumference - (score * circumference);
                
                // Add SVG circle animation (simplified version)
                circle.style.background = `conic-gradient(#10b981 ${{score * 360}}deg, rgba(255,255,255,0.2) 0deg)`;
            }});
        }}
        
        {'function initializeRealTimeMonitoring() { setInterval(updateRealTimeData, 30000); }' if include_real_time else ''}
        
        {'function updateRealTimeData() { console.log("Updating real-time data..."); }' if include_real_time else ''}
        """
        
        return js_code
    
    def _generate_monitoring_js(self, data: Dict[str, Any]) -> str:
        """Generate JavaScript for monitoring dashboard"""
        
        js_code = f"""
        // Monitoring data
        const monitoringData = {json.dumps(data)};
        let isPaused = false;
        let refreshInterval;
        
        // Initialize monitoring when page loads
        document.addEventListener('DOMContentLoaded', function() {{
            initializeMonitoringCharts();
            startAutoRefresh();
            setupControls();
        }});
        
        function initializeMonitoringCharts() {{
            initializeMetricsChart();
            initializeTrendsChart();
        }}
        
        function initializeMetricsChart() {{
            const ctx = document.getElementById('metricsChart').getContext('2d');
            
            new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: generateTimeLabels(),
                    datasets: [{{
                        label: 'Overall Score',
                        data: generateRandomData(20, 0.7, 0.9),
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.4
                    }}, {{
                        label: 'Data Quality',
                        data: generateRandomData(20, 0.75, 0.95),
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.4
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{
                            labels: {{
                                color: '#e2e8f0'
                            }}
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 1,
                            grid: {{
                                color: '#334155'
                            }},
                            ticks: {{
                                color: '#e2e8f0'
                            }}
                        }},
                        x: {{
                            grid: {{
                                color: '#334155'
                            }},
                            ticks: {{
                                color: '#e2e8f0'
                            }}
                        }}
                    }}
                }}
            }});
        }}
        
        function initializeTrendsChart() {{
            const ctx = document.getElementById('trendsChart').getContext('2d');
            
            new Chart(ctx, {{
                type: 'doughnut',
                data: {{
                    labels: ['Excellent', 'Good', 'Fair', 'Poor'],
                    datasets: [{{
                        data: [30, 45, 20, 5],
                        backgroundColor: ['#10b981', '#3b82f6', '#f59e0b', '#ef4444']
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{
                            labels: {{
                                color: '#e2e8f0'
                            }}
                        }}
                    }}
                }}
            }});
        }}
        
        function startAutoRefresh() {{
            refreshInterval = setInterval(() => {{
                if (!isPaused) {{
                    updateMonitoringData();
                }}
            }}, monitoringData.refresh_interval * 1000);
        }}
        
        function setupControls() {{
            document.getElementById('pauseBtn').addEventListener('click', togglePause);
            document.getElementById('refreshBtn').addEventListener('click', forceRefresh);
        }}
        
        function togglePause() {{
            isPaused = !isPaused;
            const btn = document.getElementById('pauseBtn');
            btn.textContent = isPaused ? '▶️ Resume' : '⏸️ Pause';
        }}
        
        function forceRefresh() {{
            updateMonitoringData();
        }}
        
        function updateMonitoringData() {{
            // Simulate data updates
            console.log('Updating monitoring data...');
            
            // Add new alert occasionally
            if (Math.random() < 0.1) {{
                addNewAlert();
            }}
        }}
        
        function addNewAlert() {{
            const alertsList = document.getElementById('alertsList');
            const alertTypes = ['warning', 'error'];
            const alertType = alertTypes[Math.floor(Math.random() * alertTypes.length)];
            
            const alertHtml = `
                <div class="alert-item ${{alertType}}">
                    <span class="alert-icon">${{alertType === 'warning' ? '⚠️' : '🚨'}}</span>
                    <div class="alert-content">
                        <strong>${{alertType === 'warning' ? 'Quality Warning' : 'Critical Error'}}</strong>
                        <p>Dataset: dataset_${{Math.floor(Math.random() * 100)}} - Issue detected</p>
                        <span class="alert-time">Just now</span>
                    </div>
                </div>
            `;
            
            alertsList.insertAdjacentHTML('afterbegin', alertHtml);
            
            // Remove old alerts (keep only 5)
            const alerts = alertsList.children;
            while (alerts.length > 5) {{
                alertsList.removeChild(alerts[alerts.length - 1]);
            }}
        }}
        
        function generateTimeLabels() {{
            const labels = [];
            const now = new Date();
            for (let i = 19; i >= 0; i--) {{
                const time = new Date(now.getTime() - i * 60000); // 1 minute intervals
                labels.push(time.toLocaleTimeString([], {{hour: '2-digit', minute: '2-digit'}}));
            }}
            return labels;
        }}
        
        function generateRandomData(count, min, max) {{
            const data = [];
            for (let i = 0; i < count; i++) {{
                data.push(Math.random() * (max - min) + min);
            }}
            return data;
        }}
        """
        
        return js_code
    
    def _generate_comparison_js(self, data: Dict[str, Any]) -> str:
        """Generate JavaScript for comparison dashboard"""
        
        js_code = f"""
        // Comparison data
        const comparisonData = {json.dumps(data)};
        
        // Initialize comparison charts when page loads
        document.addEventListener('DOMContentLoaded', function() {{
            initializeComparisonChart();
        }});
        
        function initializeComparisonChart() {{
            const ctx = document.getElementById('comparisonChart').getContext('2d');
            
            const datasets = comparisonData.dimensions.map((dimension, index) => ({{
                label: dimension.replace('_', ' ').replace(/\\b\\w/g, l => l.toUpperCase()),
                data: comparisonData.scores[dimension],
                backgroundColor: getColorForIndex(index),
                borderColor: getColorForIndex(index, 0.8),
                borderWidth: 2
            }}));
            
            new Chart(ctx, {{
                type: 'radar',
                data: {{
                    labels: comparisonData.datasets,
                    datasets: datasets
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            position: 'top'
                        }}
                    }},
                    scales: {{
                        r: {{
                            beginAtZero: true,
                            max: 1,
                            grid: {{
                                color: '#e2e8f0'
                            }},
                            pointLabels: {{
                                font: {{
                                    size: 12
                                }}
                            }}
                        }}
                    }}
                }}
            }});
        }}
        
        function getColorForIndex(index, alpha = 0.2) {{
            const colors = [
                `rgba(59, 130, 246, ${{alpha}})`,   // blue
                `rgba(16, 185, 129, ${{alpha}})`,   // green
                `rgba(245, 158, 11, ${{alpha}})`,   // yellow
                `rgba(239, 68, 68, ${{alpha}})`,    // red
                `rgba(139, 92, 246, ${{alpha}})`,   // purple
                `rgba(236, 72, 153, ${{alpha}})`    // pink
            ];
            return colors[index % colors.length];
        }}
        """
        
        return js_code