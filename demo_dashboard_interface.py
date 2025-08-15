#!/usr/bin/env python3
"""
Demo script for AI Data Readiness Dashboard Interface

This script demonstrates the interactive dashboard functionality including
web-based visualization, real-time monitoring, and customizable reporting interfaces.
"""

import logging
from datetime import datetime
from pathlib import Path

from ai_data_readiness.dashboard.dashboard_generator import DashboardGenerator
from ai_data_readiness.engines.ai_readiness_reporting_engine import (
    AIReadinessReportingEngine,
    IndustryStandard,
    ReportType
)
from ai_data_readiness.models.base_models import (
    AIReadinessScore,
    QualityReport,
    BiasReport,
    QualityIssue,
    QualityDimension,
    DimensionScore,
    ImprovementArea
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create sample data for dashboard demonstration"""
    
    # Create AI readiness score
    ai_readiness_score = AIReadinessScore(
        overall_score=0.78,
        data_quality_score=0.82,
        feature_quality_score=0.74,
        bias_score=0.76,
        compliance_score=0.88,
        scalability_score=0.72,
        dimensions={
            "data_quality": DimensionScore(
                dimension="data_quality",
                score=0.82,
                weight=0.25,
                details={
                    "completeness": 0.88,
                    "accuracy": 0.82,
                    "consistency": 0.78,
                    "validity": 0.80
                }
            ),
            "feature_quality": DimensionScore(
                dimension="feature_quality",
                score=0.74,
                weight=0.20,
                details={
                    "feature_correlation": 0.76,
                    "feature_importance": 0.74,
                    "encoding_quality": 0.72
                }
            )
        },
        improvement_areas=[
            ImprovementArea(
                area="feature_engineering",
                current_score=0.74,
                target_score=0.85,
                priority="high",
                estimated_effort="medium"
            )
        ]
    )
    
    # Create quality report
    quality_report = QualityReport(
        dataset_id="customer_analytics_2024",
        overall_score=0.82,
        completeness_score=0.88,
        accuracy_score=0.82,
        consistency_score=0.78,
        validity_score=0.80,
        uniqueness_score=0.92,
        timeliness_score=0.86,
        issues=[
            QualityIssue(
                dimension=QualityDimension.COMPLETENESS,
                severity="medium",
                description="Missing values in customer_segment column",
                affected_columns=["customer_segment"],
                affected_rows=850,
                recommendation="Implement segment classification algorithm"
            )
        ],
        recommendations=[],
        generated_at=datetime.utcnow()
    )
    
    # Create bias report
    bias_report = BiasReport(
        dataset_id="customer_analytics_2024",
        protected_attributes=["gender", "age_group", "location"],
        bias_metrics={
            "demographic_parity": 0.76,
            "equalized_odds": 0.82,
            "statistical_parity": 0.74
        },
        fairness_violations=[],
        mitigation_strategies=[],
        generated_at=datetime.utcnow()
    )
    
    return ai_readiness_score, quality_report, bias_report


def demo_readiness_dashboard():
    """Demonstrate AI readiness dashboard generation"""
    logger.info("=== AI Readiness Dashboard Demo ===")
    
    # Initialize components
    dashboard_generator = DashboardGenerator()
    reporting_engine = AIReadinessReportingEngine()
    
    # Create sample data
    ai_readiness_score, quality_report, bias_report = create_sample_data()
    
    # Generate comprehensive report
    logger.info("Generating comprehensive report...")
    report = reporting_engine.generate_comprehensive_report(
        dataset_id="customer_analytics_2024",
        ai_readiness_score=ai_readiness_score,
        quality_report=quality_report,
        bias_report=bias_report,
        industry=IndustryStandard.RETAIL,
        report_type=ReportType.DETAILED_TECHNICAL
    )
    
    # Generate readiness dashboard
    logger.info("Generating interactive readiness dashboard...")
    customization_options = {
        "theme": "professional",
        "color_scheme": "blue",
        "show_details": True
    }
    
    dashboard_html = dashboard_generator.generate_readiness_dashboard(
        report=report,
        include_real_time=True,
        customization_options=customization_options
    )
    
    # Save dashboard to file
    dashboard_file = Path("ai_readiness_dashboard.html")
    with open(dashboard_file, 'w', encoding='utf-8') as f:
        f.write(dashboard_html)
    
    logger.info(f"Readiness dashboard saved to: {dashboard_file}")
    logger.info(f"Dashboard size: {len(dashboard_html):,} characters")
    
    return dashboard_file


def demo_monitoring_dashboard():
    """Demonstrate real-time monitoring dashboard"""
    logger.info("\n=== Real-time Monitoring Dashboard Demo ===")
    
    # Initialize dashboard generator
    dashboard_generator = DashboardGenerator()
    
    # Sample datasets to monitor
    datasets = [
        "customer_analytics_2024",
        "sales_transactions_2024",
        "product_catalog_2024",
        "user_behavior_2024"
    ]
    
    # Custom alert thresholds
    alert_thresholds = {
        "overall_score_min": 0.75,
        "data_quality_min": 0.80,
        "bias_score_min": 0.70,
        "compliance_score_min": 0.85,
        "drift_threshold": 0.25
    }
    
    # Generate monitoring dashboard
    logger.info(f"Generating monitoring dashboard for {len(datasets)} datasets...")
    dashboard_html = dashboard_generator.generate_monitoring_dashboard(
        datasets=datasets,
        refresh_interval=30,
        alert_thresholds=alert_thresholds
    )
    
    # Save dashboard to file
    dashboard_file = Path("monitoring_dashboard.html")
    with open(dashboard_file, 'w', encoding='utf-8') as f:
        f.write(dashboard_html)
    
    logger.info(f"Monitoring dashboard saved to: {dashboard_file}")
    logger.info(f"Dashboard features:")
    logger.info(f"  - Real-time monitoring for {len(datasets)} datasets")
    logger.info(f"  - Auto-refresh every 30 seconds")
    logger.info(f"  - Custom alert thresholds configured")
    logger.info(f"  - Interactive controls (pause/resume)")
    
    return dashboard_file


def demo_comparison_dashboard():
    """Demonstrate comparison dashboard for multiple datasets"""
    logger.info("\n=== Dataset Comparison Dashboard Demo ===")
    
    # Initialize components
    dashboard_generator = DashboardGenerator()
    reporting_engine = AIReadinessReportingEngine()
    
    # Create sample reports for multiple datasets
    datasets = [
        "customer_analytics_2024",
        "sales_transactions_2024", 
        "product_catalog_2024"
    ]
    
    reports = []
    for i, dataset_id in enumerate(datasets):
        # Create slightly different scores for each dataset
        base_score = 0.75 + (i * 0.05)
        
        ai_readiness_score = AIReadinessScore(
            overall_score=base_score,
            data_quality_score=base_score + 0.05,
            feature_quality_score=base_score - 0.02,
            bias_score=base_score + 0.03,
            compliance_score=base_score + 0.08,
            scalability_score=base_score - 0.03,
            dimensions={
                "data_quality": DimensionScore(
                    dimension="data_quality",
                    score=base_score + 0.05,
                    weight=0.25,
                    details={}
                )
            },
            improvement_areas=[]
        )
        
        quality_report = QualityReport(
            dataset_id=dataset_id,
            overall_score=base_score + 0.05,
            completeness_score=base_score + 0.08,
            accuracy_score=base_score + 0.02,
            consistency_score=base_score,
            validity_score=base_score + 0.03,
            uniqueness_score=base_score + 0.10,
            timeliness_score=base_score + 0.06,
            issues=[],
            recommendations=[],
            generated_at=datetime.utcnow()
        )
        
        bias_report = BiasReport(
            dataset_id=dataset_id,
            protected_attributes=["gender", "age_group"],
            bias_metrics={"demographic_parity": base_score + 0.03},
            fairness_violations=[],
            mitigation_strategies=[],
            generated_at=datetime.utcnow()
        )
        
        # Generate report
        report = reporting_engine.generate_comprehensive_report(
            dataset_id=dataset_id,
            ai_readiness_score=ai_readiness_score,
            quality_report=quality_report,
            bias_report=bias_report,
            industry=IndustryStandard.RETAIL,
            report_type=ReportType.BENCHMARK_COMPARISON
        )
        reports.append(report)
    
    # Generate comparison dashboard
    logger.info(f"Generating comparison dashboard for {len(reports)} datasets...")
    comparison_dimensions = [
        "overall_score",
        "data_quality_score", 
        "feature_quality_score",
        "bias_score",
        "compliance_score"
    ]
    
    dashboard_html = dashboard_generator.generate_comparison_dashboard(
        reports=reports,
        comparison_dimensions=comparison_dimensions
    )
    
    # Save dashboard to file
    dashboard_file = Path("comparison_dashboard.html")
    with open(dashboard_file, 'w', encoding='utf-8') as f:
        f.write(dashboard_html)
    
    logger.info(f"Comparison dashboard saved to: {dashboard_file}")
    logger.info(f"Dashboard features:")
    logger.info(f"  - Radar chart comparison across {len(comparison_dimensions)} dimensions")
    logger.info(f"  - Dataset rankings and performance analysis")
    logger.info(f"  - Detailed comparison table")
    logger.info(f"  - Interactive visualizations")
    
    return dashboard_file


def demo_custom_themes():
    """Demonstrate different dashboard themes and customizations"""
    logger.info("\n=== Custom Dashboard Themes Demo ===")
    
    # Initialize components
    dashboard_generator = DashboardGenerator()
    reporting_engine = AIReadinessReportingEngine()
    
    # Create sample data
    ai_readiness_score, quality_report, bias_report = create_sample_data()
    
    # Generate report
    report = reporting_engine.generate_comprehensive_report(
        dataset_id="theme_demo_dataset",
        ai_readiness_score=ai_readiness_score,
        quality_report=quality_report,
        bias_report=bias_report,
        industry=IndustryStandard.TECHNOLOGY,
        report_type=ReportType.EXECUTIVE_SUMMARY
    )
    
    # Generate dashboards with different themes
    themes = [
        {"theme": "professional", "color_scheme": "blue"},
        {"theme": "modern", "color_scheme": "green"},
        {"theme": "minimal", "color_scheme": "purple"}
    ]
    
    dashboard_files = []
    for theme_config in themes:
        logger.info(f"Generating dashboard with theme: {theme_config['theme']}, color: {theme_config['color_scheme']}")
        
        dashboard_html = dashboard_generator.generate_readiness_dashboard(
            report=report,
            include_real_time=False,
            customization_options=theme_config
        )
        
        # Save themed dashboard
        filename = f"dashboard_{theme_config['theme']}_{theme_config['color_scheme']}.html"
        dashboard_file = Path(filename)
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        dashboard_files.append(dashboard_file)
        logger.info(f"  - Saved to: {dashboard_file}")
    
    return dashboard_files


def demo_dashboard_features():
    """Demonstrate key dashboard features"""
    logger.info("\n=== Dashboard Features Overview ===")
    
    features = {
        "Interactive Components": [
            "Real-time score visualization with animated circles",
            "Interactive charts using Chart.js",
            "Responsive grid layout",
            "Hover effects and tooltips"
        ],
        "Customization Options": [
            "Multiple themes (professional, modern, minimal)",
            "Color schemes (blue, green, purple, red)",
            "Configurable refresh intervals",
            "Custom alert thresholds"
        ],
        "Visualization Types": [
            "Score circles with percentage displays",
            "Bar charts for benchmark comparisons",
            "Radar charts for multi-dataset comparisons",
            "Real-time line charts for monitoring",
            "Doughnut charts for distribution analysis"
        ],
        "Dashboard Types": [
            "AI Readiness Dashboard - comprehensive overview",
            "Monitoring Dashboard - real-time system monitoring",
            "Comparison Dashboard - multi-dataset analysis",
            "Custom Dashboard - user-defined configurations"
        ],
        "Real-time Features": [
            "Auto-refresh capabilities",
            "Live data updates",
            "Alert notifications",
            "Pause/resume controls",
            "Dynamic chart updates"
        ]
    }
    
    for category, items in features.items():
        logger.info(f"\n{category}:")
        for item in items:
            logger.info(f"  • {item}")
    
    logger.info(f"\nTotal dashboard features demonstrated: {sum(len(items) for items in features.values())}")


def main():
    """Main demo function"""
    logger.info("Starting AI Data Readiness Dashboard Interface Demo")
    logger.info("=" * 65)
    
    try:
        # Demo 1: AI Readiness Dashboard
        readiness_file = demo_readiness_dashboard()
        
        # Demo 2: Real-time Monitoring Dashboard
        monitoring_file = demo_monitoring_dashboard()
        
        # Demo 3: Dataset Comparison Dashboard
        comparison_file = demo_comparison_dashboard()
        
        # Demo 4: Custom Themes
        theme_files = demo_custom_themes()
        
        # Demo 5: Feature Overview
        demo_dashboard_features()
        
        # Summary
        logger.info("\n" + "=" * 65)
        logger.info("AI Data Readiness Dashboard Interface Demo completed successfully!")
        
        all_files = [readiness_file, monitoring_file, comparison_file] + theme_files
        logger.info(f"\nGenerated {len(all_files)} dashboard files:")
        for file in all_files:
            logger.info(f"  - {file}")
        
        logger.info(f"\nDemo Summary:")
        logger.info(f"  - Generated {len(all_files)} interactive HTML dashboards")
        logger.info(f"  - Demonstrated 4 different dashboard types")
        logger.info(f"  - Showcased 3 different themes and color schemes")
        logger.info(f"  - Included real-time monitoring capabilities")
        logger.info(f"  - Featured responsive design and interactive charts")
        
        logger.info(f"\nTo view the dashboards:")
        logger.info(f"  1. Open any .html file in a web browser")
        logger.info(f"  2. Dashboards are fully self-contained with embedded CSS/JS")
        logger.info(f"  3. Real-time features work with simulated data")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()