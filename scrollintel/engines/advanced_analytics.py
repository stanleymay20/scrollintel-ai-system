"""
Advanced Analytics Engine
Integrates all analytics components for comprehensive business intelligence.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncio
import pandas as pd
import numpy as np
from dataclasses import asdict

from .comprehensive_reporting_engine import ComprehensiveReportingEngine, ReportConfig, ReportFormat, ReportType
from .advanced_statistical_analytics import AdvancedStatisticalAnalytics, AnalysisType
from .executive_summary_generator import ExecutiveSummaryGenerator, SummaryType
from ..core.automated_report_scheduler import AutomatedReportScheduler
from ..core.dashboard_manager import DashboardManager
from ..engines.roi_calculator import ROICalculator
from ..engines.predictive_engine import PredictiveEngine
from ..engines.insight_generator import InsightGenerator

logger = logging.getLogger(__name__)

class AdvancedAnalyticsEngine:
    """Main analytics engine integrating all advanced analytics capabilities."""
    
    def __init__(self):
        self.reporting_engine = ComprehensiveReportingEngine()
        self.statistical_engine = AdvancedStatisticalAnalytics()
        self.summary_generator = ExecutiveSummaryGenerator()
        self.report_scheduler = AutomatedReportScheduler(self.reporting_engine)
        self.dashboard_manager = DashboardManager()
        self.roi_calculator = ROICalculator()
        self.predictive_engine = PredictiveEngine()
        self.insight_generator = InsightGenerator()
        
        self.logger = logging.getLogger(__name__)
    
    async def perform_comprehensive_analysis(self, 
                                           data: pd.DataFrame,
                                           analysis_config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analytics analysis."""
        try:
            self.logger.info("Starting comprehensive analytics analysis")
            
            # Statistical analysis
            statistical_results = self.statistical_engine.perform_comprehensive_analysis(
                data=data,
                target_column=analysis_config.get('target_column'),
                analysis_types=analysis_config.get('analysis_types', [
                    AnalysisType.DESCRIPTIVE,
                    AnalysisType.CORRELATION,
                    AnalysisType.PREDICTIVE,
                    AnalysisType.TREND_ANALYSIS,
                    AnalysisType.ANOMALY_DETECTION
                ])
            )
            
            # ROI analysis if financial data available
            roi_results = {}
            if 'financial_data' in analysis_config:
                roi_results = await self._perform_roi_analysis(analysis_config['financial_data'])
            
            # Predictive analysis
            predictive_results = {}
            if analysis_config.get('enable_predictions', True):
                predictive_results = await self._perform_predictive_analysis(data, analysis_config)
            
            # Generate insights
            insights = await self._generate_comprehensive_insights(
                statistical_results, roi_results, predictive_results
            )
            
            # Combine all results
            comprehensive_results = {
                'analysis_id': f"analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                'timestamp': datetime.utcnow().isoformat(),
                'statistical_analysis': statistical_results,
                'roi_analysis': roi_results,
                'predictive_analysis': predictive_results,
                'insights': insights,
                'data_summary': {
                    'rows': len(data),
                    'columns': len(data.columns),
                    'period': analysis_config.get('period', 'current')
                }
            }
            
            self.logger.info("Comprehensive analysis completed successfully")
            return comprehensive_results
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {str(e)}")
            raise
    
    async def generate_executive_report(self,
                                      analysis_results: Dict[str, Any],
                                      report_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive report with summary and recommendations."""
        try:
            self.logger.info("Generating executive report")
            
            # Generate executive summary
            summary = self.summary_generator.generate_executive_summary(
                data=analysis_results,
                summary_type=SummaryType(report_config.get('summary_type', 'executive_overview')),
                context=report_config.get('context')
            )
            
            # Create comprehensive report
            report_sections = await self._create_report_sections(analysis_results, summary)
            
            # Generate report in requested format
            report_format = ReportFormat(report_config.get('format', 'pdf'))
            report_config_obj = ReportConfig(
                title=report_config.get('title', 'Advanced Analytics Report'),
                report_type=ReportType.EXECUTIVE_SUMMARY,
                format=report_format,
                sections=report_sections,
                metadata={
                    'generated_by': 'ScrollIntel Advanced Analytics',
                    'analysis_id': analysis_results.get('analysis_id'),
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
            report_output = self.reporting_engine.generate_report(report_config_obj)
            
            return {
                'report_id': f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                'executive_summary': summary,
                'report_output': report_output,
                'analysis_results': analysis_results
            }
            
        except Exception as e:
            self.logger.error(f"Executive report generation failed: {str(e)}")
            raise
    
    async def setup_automated_reporting(self,
                                      schedule_config: Dict[str, Any]) -> str:
        """Setup automated report generation and distribution."""
        try:
            from ..core.automated_report_scheduler import (
                ScheduledReport, ScheduleConfig, DistributionConfig, ScheduleFrequency
            )
            
            # Create schedule configuration
            schedule = ScheduleConfig(
                frequency=ScheduleFrequency(schedule_config['frequency']),
                start_time=datetime.fromisoformat(schedule_config['start_time']),
                end_time=datetime.fromisoformat(schedule_config['end_time']) if schedule_config.get('end_time') else None
            )
            
            # Create distribution configuration
            distribution = DistributionConfig(
                email_recipients=schedule_config['email_recipients'],
                email_subject_template=schedule_config.get('email_subject', 'Analytics Report - {date}'),
                email_body_template=schedule_config.get('email_body', 'Please find the latest analytics report attached.'),
                storage_path=schedule_config.get('storage_path')
            )
            
            # Create report configuration
            report_config = ReportConfig(
                title=schedule_config.get('report_title', 'Automated Analytics Report'),
                report_type=ReportType.EXECUTIVE_SUMMARY,
                format=ReportFormat(schedule_config.get('format', 'pdf')),
                sections=[],  # Will be populated during generation
                metadata={'automated': True}
            )
            
            # Create scheduled report
            scheduled_report = ScheduledReport(
                id=f"scheduled_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                name=schedule_config['name'],
                description=schedule_config.get('description', ''),
                report_config=report_config,
                schedule_config=schedule,
                distribution_config=distribution,
                data_source_config=schedule_config.get('data_sources', {}),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Schedule the report
            report_id = self.report_scheduler.schedule_report(scheduled_report)
            
            # Start scheduler if not running
            if not self.report_scheduler.running:
                await self.report_scheduler.start_scheduler()
            
            self.logger.info(f"Automated reporting setup completed: {report_id}")
            return report_id
            
        except Exception as e:
            self.logger.error(f"Automated reporting setup failed: {str(e)}")
            raise
    
    async def create_executive_dashboard(self,
                                       dashboard_config: Dict[str, Any]) -> str:
        """Create executive dashboard with real-time analytics."""
        try:
            from ..models.dashboard_models import Dashboard, Widget, DashboardConfig
            
            # Create dashboard widgets
            widgets = []
            
            # KPI widgets
            kpi_widgets = [
                Widget(
                    id="roi_widget",
                    type="metric",
                    title="Total ROI",
                    config={
                        "metric": "total_roi",
                        "format": "percentage",
                        "target": 20,
                        "trend": True
                    },
                    position={"x": 0, "y": 0, "w": 3, "h": 2}
                ),
                Widget(
                    id="efficiency_widget",
                    type="metric",
                    title="Efficiency Gain",
                    config={
                        "metric": "efficiency_gain",
                        "format": "percentage",
                        "trend": True
                    },
                    position={"x": 3, "y": 0, "w": 3, "h": 2}
                ),
                Widget(
                    id="user_growth_widget",
                    type="metric",
                    title="User Growth",
                    config={
                        "metric": "user_growth",
                        "format": "percentage",
                        "trend": True
                    },
                    position={"x": 6, "y": 0, "w": 3, "h": 2}
                )
            ]
            widgets.extend(kpi_widgets)
            
            # Chart widgets
            chart_widgets = [
                Widget(
                    id="trend_chart",
                    type="line_chart",
                    title="Performance Trends",
                    config={
                        "metrics": ["roi", "efficiency", "user_satisfaction"],
                        "time_range": "30d"
                    },
                    position={"x": 0, "y": 2, "w": 6, "h": 4}
                ),
                Widget(
                    id="insights_widget",
                    type="insights",
                    title="Key Insights",
                    config={
                        "max_insights": 5,
                        "priority_filter": ["critical", "high"]
                    },
                    position={"x": 6, "y": 2, "w": 6, "h": 4}
                )
            ]
            widgets.extend(chart_widgets)
            
            # Create dashboard
            dashboard = Dashboard(
                id=f"exec_dashboard_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                name=dashboard_config.get('name', 'Executive Analytics Dashboard'),
                description=dashboard_config.get('description', 'Comprehensive executive analytics dashboard'),
                type="executive",
                owner_id=dashboard_config.get('owner_id', 'system'),
                config=DashboardConfig(
                    refresh_interval=dashboard_config.get('refresh_interval', 300),
                    auto_refresh=True,
                    theme=dashboard_config.get('theme', 'light')
                ),
                widgets=widgets,
                permissions=[],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Save dashboard
            dashboard_id = await self.dashboard_manager.create_dashboard(dashboard)
            
            self.logger.info(f"Executive dashboard created: {dashboard_id}")
            return dashboard_id
            
        except Exception as e:
            self.logger.error(f"Executive dashboard creation failed: {str(e)}")
            raise
    
    async def _perform_roi_analysis(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform ROI analysis on financial data."""
        try:
            # Extract financial metrics
            investments = financial_data.get('investments', [])
            benefits = financial_data.get('benefits', [])
            costs = financial_data.get('costs', [])
            
            # Calculate ROI for each project
            project_rois = []
            for investment in investments:
                project_roi = await self.roi_calculator.calculate_project_roi(
                    project_id=investment['project_id'],
                    investment_amount=investment['amount'],
                    benefits=benefits,
                    costs=costs,
                    time_period=investment.get('time_period', 12)
                )
                project_rois.append(project_roi)
            
            # Calculate overall ROI
            total_investment = sum(inv['amount'] for inv in investments)
            total_benefits = sum(ben['amount'] for ben in benefits)
            total_costs = sum(cost['amount'] for cost in costs)
            
            overall_roi = ((total_benefits - total_costs - total_investment) / total_investment) * 100 if total_investment > 0 else 0
            
            return {
                'overall_roi': overall_roi,
                'total_investment': total_investment,
                'total_benefits': total_benefits,
                'total_costs': total_costs,
                'project_rois': project_rois,
                'payback_period': self._calculate_payback_period(investments, benefits, costs)
            }
            
        except Exception as e:
            self.logger.error(f"ROI analysis failed: {str(e)}")
            return {}
    
    async def _perform_predictive_analysis(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform predictive analysis."""
        try:
            target_column = config.get('target_column')
            if not target_column or target_column not in data.columns:
                return {}
            
            # Prepare data for prediction
            forecast_horizon = config.get('forecast_horizon', 30)
            
            # Generate forecasts
            forecasts = await self.predictive_engine.generate_forecasts(
                data=data,
                target_column=target_column,
                horizon=forecast_horizon,
                confidence_intervals=True
            )
            
            # Risk predictions
            risk_predictions = await self.predictive_engine.predict_risks(
                data=data,
                risk_factors=config.get('risk_factors', [])
            )
            
            return {
                'forecasts': forecasts,
                'risk_predictions': risk_predictions,
                'forecast_horizon': forecast_horizon,
                'model_performance': forecasts.get('model_metrics', {})
            }
            
        except Exception as e:
            self.logger.error(f"Predictive analysis failed: {str(e)}")
            return {}
    
    async def _generate_comprehensive_insights(self,
                                             statistical_results: Dict[str, Any],
                                             roi_results: Dict[str, Any],
                                             predictive_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate comprehensive insights from all analysis results."""
        try:
            all_insights = []
            
            # Statistical insights
            if 'ml_insights' in statistical_results:
                all_insights.extend(statistical_results['ml_insights'])
            
            # ROI insights
            if roi_results:
                roi_insights = await self._generate_roi_insights(roi_results)
                all_insights.extend(roi_insights)
            
            # Predictive insights
            if predictive_results:
                predictive_insights = await self._generate_predictive_insights(predictive_results)
                all_insights.extend(predictive_insights)
            
            # Generate additional insights using insight generator
            enhanced_insights = await self.insight_generator.generate_insights(
                data_sources={
                    'statistical': statistical_results,
                    'roi': roi_results,
                    'predictive': predictive_results
                }
            )
            
            all_insights.extend(enhanced_insights)
            
            # Sort by priority and confidence
            all_insights.sort(key=lambda x: (x.get('priority', 'low'), -x.get('confidence', 0)))
            
            return all_insights[:10]  # Return top 10 insights
            
        except Exception as e:
            self.logger.error(f"Insight generation failed: {str(e)}")
            return []
    
    async def _generate_roi_insights(self, roi_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights from ROI analysis."""
        insights = []
        
        overall_roi = roi_results.get('overall_roi', 0)
        
        if overall_roi > 25:
            insights.append({
                'type': 'roi_performance',
                'title': 'Exceptional ROI Achievement',
                'description': f'Overall ROI of {overall_roi:.1f}% significantly exceeds industry benchmarks',
                'priority': 'high',
                'confidence': 0.9,
                'impact': 'positive',
                'recommendations': ['Continue current investment strategy', 'Consider scaling successful initiatives']
            })
        elif overall_roi < 5:
            insights.append({
                'type': 'roi_performance',
                'title': 'ROI Below Target',
                'description': f'Overall ROI of {overall_roi:.1f}% requires strategic intervention',
                'priority': 'critical',
                'confidence': 0.85,
                'impact': 'negative',
                'recommendations': ['Review underperforming projects', 'Optimize resource allocation']
            })
        
        return insights
    
    async def _generate_predictive_insights(self, predictive_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights from predictive analysis."""
        insights = []
        
        forecasts = predictive_results.get('forecasts', {})
        risk_predictions = predictive_results.get('risk_predictions', {})
        
        # Forecast insights
        if forecasts:
            trend = forecasts.get('trend', 'stable')
            if trend == 'increasing':
                insights.append({
                    'type': 'forecast',
                    'title': 'Positive Growth Forecast',
                    'description': 'Predictive models indicate continued positive growth trajectory',
                    'priority': 'medium',
                    'confidence': forecasts.get('confidence', 0.7),
                    'impact': 'positive',
                    'recommendations': ['Prepare for increased capacity needs', 'Plan resource scaling']
                })
        
        # Risk insights
        if risk_predictions:
            high_risk_factors = [r for r in risk_predictions.get('factors', []) if r.get('risk_level') == 'high']
            if high_risk_factors:
                insights.append({
                    'type': 'risk',
                    'title': 'High Risk Factors Identified',
                    'description': f'Predictive analysis identified {len(high_risk_factors)} high-risk factors',
                    'priority': 'high',
                    'confidence': 0.8,
                    'impact': 'negative',
                    'recommendations': ['Implement risk mitigation strategies', 'Monitor risk factors closely']
                })
        
        return insights
    
    async def _create_report_sections(self, analysis_results: Dict[str, Any], 
                                    summary: Any) -> List[Any]:
        """Create report sections from analysis results."""
        from .comprehensive_reporting_engine import ReportSection
        
        sections = []
        
        # Executive Summary Section
        sections.append(ReportSection(
            title="Executive Summary",
            content=summary.executive_overview,
            data={
                'key_findings': [asdict(f) for f in summary.key_findings],
                'recommendations': [asdict(r) for r in summary.recommendations]
            }
        ))
        
        # Statistical Analysis Section
        if 'statistical_analysis' in analysis_results:
            stat_analysis = analysis_results['statistical_analysis']
            sections.append(ReportSection(
                title="Statistical Analysis",
                content="Comprehensive statistical analysis of the dataset including descriptive statistics, correlations, and trend analysis.",
                data=stat_analysis.get('descriptive', {}),
                tables=[{
                    'title': 'Key Statistics',
                    'data': stat_analysis.get('descriptive', {}).get('numeric_summary', {})
                }]
            ))
        
        # ROI Analysis Section
        if 'roi_analysis' in analysis_results and analysis_results['roi_analysis']:
            roi_analysis = analysis_results['roi_analysis']
            sections.append(ReportSection(
                title="ROI Analysis",
                content=f"Return on Investment analysis shows overall ROI of {roi_analysis.get('overall_roi', 0):.1f}%.",
                data=roi_analysis,
                tables=[{
                    'title': 'ROI Summary',
                    'data': {
                        'Metric': ['Total Investment', 'Total Benefits', 'Overall ROI', 'Payback Period'],
                        'Value': [
                            f"${roi_analysis.get('total_investment', 0):,.0f}",
                            f"${roi_analysis.get('total_benefits', 0):,.0f}",
                            f"{roi_analysis.get('overall_roi', 0):.1f}%",
                            f"{roi_analysis.get('payback_period', 0):.1f} months"
                        ]
                    }
                }]
            ))
        
        # Insights Section
        if 'insights' in analysis_results:
            insights = analysis_results['insights']
            sections.append(ReportSection(
                title="Key Insights",
                content="Machine learning-powered insights and recommendations based on comprehensive data analysis.",
                data={'insights': insights},
                tables=[{
                    'title': 'Priority Insights',
                    'data': {
                        'Insight': [i.get('title', '') for i in insights[:5]],
                        'Priority': [i.get('priority', '') for i in insights[:5]],
                        'Confidence': [f"{i.get('confidence', 0)*100:.0f}%" for i in insights[:5]]
                    }
                }]
            ))
        
        return sections
    
    def _calculate_payback_period(self, investments: List[Dict], benefits: List[Dict], costs: List[Dict]) -> float:
        """Calculate average payback period."""
        try:
            total_investment = sum(inv['amount'] for inv in investments)
            monthly_net_benefit = (sum(ben['amount'] for ben in benefits) - sum(cost['amount'] for cost in costs)) / 12
            
            if monthly_net_benefit > 0:
                return total_investment / monthly_net_benefit
            else:
                return float('inf')
        except:
            return 0.0
    
    async def get_analytics_status(self) -> Dict[str, Any]:
        """Get current status of analytics engine."""
        return {
            'scheduler_running': self.report_scheduler.running,
            'scheduled_reports': len(self.report_scheduler.scheduled_reports),
            'available_formats': self.reporting_engine.get_available_formats(),
            'last_analysis': datetime.utcnow().isoformat(),
            'system_status': 'operational'
        }