#!/usr/bin/env python3
"""
Demo script for AI Data Readiness Reporting Engine

This script demonstrates the comprehensive AI readiness reporting functionality
including benchmarking against industry standards and improvement roadmap generation.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

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


def create_sample_ai_readiness_score():
    """Create sample AI readiness score for demonstration"""
    return AIReadinessScore(
        overall_score=0.75,
        data_quality_score=0.80,
        feature_quality_score=0.70,
        bias_score=0.75,
        compliance_score=0.85,
        scalability_score=0.70,
        dimensions={
            "data_quality": DimensionScore(
                dimension="data_quality",
                score=0.80,
                weight=0.25,
                details={
                    "completeness": 0.85,
                    "accuracy": 0.80,
                    "consistency": 0.75,
                    "validity": 0.80
                }
            ),
            "feature_quality": DimensionScore(
                dimension="feature_quality",
                score=0.70,
                weight=0.20,
                details={
                    "feature_correlation": 0.75,
                    "feature_importance": 0.70,
                    "encoding_quality": 0.65
                }
            ),
            "bias_score": DimensionScore(
                dimension="bias_score",
                score=0.75,
                weight=0.20,
                details={
                    "demographic_parity": 0.75,
                    "equalized_odds": 0.80,
                    "statistical_parity": 0.70
                }
            ),
            "compliance_score": DimensionScore(
                dimension="compliance_score",
                score=0.85,
                weight=0.20,
                details={
                    "gdpr_compliance": 0.90,
                    "privacy_protection": 0.85,
                    "data_governance": 0.80
                }
            ),
            "scalability_score": DimensionScore(
                dimension="scalability_score",
                score=0.70,
                weight=0.15,
                details={
                    "processing_efficiency": 0.75,
                    "storage_optimization": 0.70,
                    "compute_scalability": 0.65
                }
            )
        },
        improvement_areas=[
            ImprovementArea(
                area="feature_engineering",
                current_score=0.70,
                target_score=0.85,
                priority="high",
                estimated_effort="medium"
            ),
            ImprovementArea(
                area="bias_mitigation",
                current_score=0.75,
                target_score=0.90,
                priority="high",
                estimated_effort="high"
            )
        ]
    )


def create_sample_quality_report():
    """Create sample quality report for demonstration"""
    return QualityReport(
        dataset_id="customer_transactions_2024",
        overall_score=0.80,
        completeness_score=0.85,
        accuracy_score=0.80,
        consistency_score=0.75,
        validity_score=0.80,
        uniqueness_score=0.90,
        timeliness_score=0.85,
        issues=[
            QualityIssue(
                dimension=QualityDimension.COMPLETENESS,
                severity="medium",
                description="Missing values detected in transaction_amount column",
                affected_columns=["transaction_amount"],
                affected_rows=1250,
                recommendation="Implement data validation rules for transaction_amount"
            ),
            QualityIssue(
                dimension=QualityDimension.CONSISTENCY,
                severity="low",
                description="Date format inconsistencies in transaction_date column",
                affected_columns=["transaction_date"],
                affected_rows=350,
                recommendation="Standardize date formats across all date columns"
            ),
            QualityIssue(
                dimension=QualityDimension.VALIDITY,
                severity="medium",
                description="Invalid customer_id values (negative numbers)",
                affected_columns=["customer_id"],
                affected_rows=75,
                recommendation="Add constraints to prevent invalid customer_id values"
            )
        ],
        recommendations=[
            "Implement data validation rules for transaction_amount",
            "Standardize date formats across all date columns",
            "Add constraints to prevent invalid customer_id values"
        ],
        generated_at=datetime.utcnow()
    )


def create_sample_bias_report():
    """Create sample bias report for demonstration"""
    return BiasReport(
        dataset_id="customer_transactions_2024",
        protected_attributes=["gender", "age_group", "income_bracket"],
        bias_metrics={
            "demographic_parity": 0.75,
            "equalized_odds": 0.80,
            "statistical_parity": 0.70,
            "disparate_impact": 0.78
        },
        fairness_violations=[],
        mitigation_strategies=[
            "Apply demographic parity constraints during model training",
            "Use fairness-aware feature selection techniques",
            "Implement post-processing bias correction methods"
        ],
        generated_at=datetime.utcnow()
    )


def demo_comprehensive_report_generation():
    """Demonstrate comprehensive AI readiness report generation"""
    logger.info("=== AI Readiness Comprehensive Report Generation Demo ===")
    
    # Initialize the reporting engine
    reporting_engine = AIReadinessReportingEngine()
    
    # Create sample data
    ai_readiness_score = create_sample_ai_readiness_score()
    quality_report = create_sample_quality_report()
    bias_report = create_sample_bias_report()
    
    # Generate comprehensive report for financial services industry
    logger.info("Generating comprehensive report for financial services industry...")
    report = reporting_engine.generate_comprehensive_report(
        dataset_id="customer_transactions_2024",
        ai_readiness_score=ai_readiness_score,
        quality_report=quality_report,
        bias_report=bias_report,
        industry=IndustryStandard.FINANCIAL_SERVICES,
        report_type=ReportType.DETAILED_TECHNICAL
    )
    
    # Display key results
    logger.info(f"Report generated successfully!")
    logger.info(f"Dataset ID: {report.dataset_id}")
    logger.info(f"Overall AI Readiness Score: {report.overall_score:.2f}")
    logger.info(f"Industry Benchmark Gap: {report.benchmark_comparison['overall_gap']:+.2f}")
    logger.info(f"Number of Improvement Actions: {len(report.improvement_actions)}")
    logger.info(f"Estimated Improvement Timeline: {report.estimated_improvement_timeline} days")
    
    # Display executive summary
    logger.info("\n=== Executive Summary ===")
    print(report.executive_summary)
    
    # Display improvement actions
    logger.info("\n=== Top 3 Improvement Actions ===")
    for i, action in enumerate(report.improvement_actions[:3], 1):
        logger.info(f"{i}. {action.title} (Priority: {action.priority.upper()})")
        logger.info(f"   Description: {action.description}")
        logger.info(f"   Timeline: {action.estimated_timeline} days")
        logger.info(f"   Expected Impact: +{action.expected_impact:.2f}")
        logger.info("")
    
    # Display compliance status
    logger.info("=== Compliance Status ===")
    for regulation, compliant in report.compliance_status.items():
        status = "✓ COMPLIANT" if compliant else "✗ NON-COMPLIANT"
        logger.info(f"{regulation.upper()}: {status}")
    
    # Display risk assessment
    logger.info("\n=== Risk Assessment ===")
    for risk_type, assessment in report.risk_assessment.items():
        logger.info(f"{risk_type.replace('_', ' ').title()}: {assessment}")
    
    return report


def demo_benchmark_comparison():
    """Demonstrate benchmark comparison across multiple industries"""
    logger.info("\n=== Industry Benchmark Comparison Demo ===")
    
    # Initialize the reporting engine
    reporting_engine = AIReadinessReportingEngine()
    
    # Create sample AI readiness score
    ai_readiness_score = create_sample_ai_readiness_score()
    
    # Compare against multiple industries
    industries = [
        IndustryStandard.FINANCIAL_SERVICES,
        IndustryStandard.HEALTHCARE,
        IndustryStandard.RETAIL,
        IndustryStandard.TECHNOLOGY
    ]
    
    logger.info("Comparing dataset against multiple industry standards...")
    comparisons = reporting_engine.generate_benchmark_comparison_report(
        dataset_id="customer_transactions_2024",
        ai_readiness_score=ai_readiness_score,
        industries=industries
    )
    
    # Display comparison results
    logger.info("\n=== Benchmark Comparison Results ===")
    for industry, comparison in comparisons.items():
        logger.info(f"\n{industry.replace('_', ' ').title()}:")
        logger.info(f"  Overall Gap: {comparison['overall_gap']:+.2f}")
        logger.info(f"  Data Quality Gap: {comparison['data_quality_gap']:+.2f}")
        logger.info(f"  Feature Quality Gap: {comparison['feature_quality_gap']:+.2f}")
        logger.info(f"  Bias Score Gap: {comparison['bias_score_gap']:+.2f}")
        logger.info(f"  Compliance Gap: {comparison['compliance_gap']:+.2f}")
    
    # Find best fit industry
    best_industry = min(comparisons.items(), key=lambda x: abs(x[1]['overall_gap']))
    logger.info(f"\nBest Fit Industry: {best_industry[0].replace('_', ' ').title()}")
    logger.info(f"Overall Gap: {best_industry[1]['overall_gap']:+.2f}")
    
    return comparisons


def demo_report_export():
    """Demonstrate report export functionality"""
    logger.info("\n=== Report Export Demo ===")
    
    # Initialize the reporting engine
    reporting_engine = AIReadinessReportingEngine()
    
    # Create sample data and generate report
    ai_readiness_score = create_sample_ai_readiness_score()
    quality_report = create_sample_quality_report()
    bias_report = create_sample_bias_report()
    
    report = reporting_engine.generate_comprehensive_report(
        dataset_id="customer_transactions_2024",
        ai_readiness_score=ai_readiness_score,
        quality_report=quality_report,
        bias_report=bias_report,
        industry=IndustryStandard.FINANCIAL_SERVICES,
        report_type=ReportType.EXECUTIVE_SUMMARY
    )
    
    # Export to JSON
    logger.info("Exporting report to JSON format...")
    json_export = reporting_engine.export_report_to_json(report)
    
    # Save JSON export
    json_file = Path("sample_ai_readiness_report.json")
    with open(json_file, 'w') as f:
        f.write(json_export)
    logger.info(f"JSON report saved to: {json_file}")
    
    # Export to HTML
    logger.info("Exporting report to HTML format...")
    html_export = reporting_engine.export_report_to_html(report)
    
    # Save HTML export
    html_file = Path("sample_ai_readiness_report.html")
    with open(html_file, 'w') as f:
        f.write(html_export)
    logger.info(f"HTML report saved to: {html_file}")
    
    return json_file, html_file


def demo_industry_benchmarks():
    """Demonstrate industry benchmarks functionality"""
    logger.info("\n=== Industry Benchmarks Demo ===")
    
    # Initialize the reporting engine
    reporting_engine = AIReadinessReportingEngine()
    
    # Display all available industry benchmarks
    logger.info("Available Industry Benchmarks:")
    logger.info("=" * 50)
    
    for industry, benchmark in reporting_engine.industry_benchmarks.items():
        logger.info(f"\n{industry.value.replace('_', ' ').title()}:")
        logger.info(f"  Data Quality Threshold: {benchmark.data_quality_threshold:.2f}")
        logger.info(f"  Feature Quality Threshold: {benchmark.feature_quality_threshold:.2f}")
        logger.info(f"  Bias Score Threshold: {benchmark.bias_score_threshold:.2f}")
        logger.info(f"  Compliance Score Threshold: {benchmark.compliance_score_threshold:.2f}")
        logger.info(f"  Overall Readiness Threshold: {benchmark.overall_readiness_threshold:.2f}")
        logger.info(f"  Typical Improvement Timeline: {benchmark.typical_improvement_timeline} days")


def main():
    """Main demo function"""
    logger.info("Starting AI Data Readiness Reporting Engine Demo")
    logger.info("=" * 60)
    
    try:
        # Demo 1: Comprehensive report generation
        report = demo_comprehensive_report_generation()
        
        # Demo 2: Benchmark comparison
        comparisons = demo_benchmark_comparison()
        
        # Demo 3: Report export
        json_file, html_file = demo_report_export()
        
        # Demo 4: Industry benchmarks
        demo_industry_benchmarks()
        
        logger.info("\n" + "=" * 60)
        logger.info("AI Data Readiness Reporting Engine Demo completed successfully!")
        logger.info(f"Generated files:")
        logger.info(f"  - {json_file}")
        logger.info(f"  - {html_file}")
        
        # Display summary statistics
        logger.info(f"\nDemo Summary:")
        logger.info(f"  - Generated comprehensive report with {len(report.improvement_actions)} improvement actions")
        logger.info(f"  - Compared against {len(comparisons)} industry standards")
        logger.info(f"  - Exported report in 2 formats (JSON, HTML)")
        logger.info(f"  - Overall AI Readiness Score: {report.overall_score:.2f}")
        logger.info(f"  - Estimated improvement timeline: {report.estimated_improvement_timeline} days")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()