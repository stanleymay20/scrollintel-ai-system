"""
Demo script for the AI Data Readiness Quality Reporting and Recommendation System.
Shows how to generate comprehensive quality reports with actionable recommendations.
"""

import json
from datetime import datetime

from ai_data_readiness.engines.recommendation_engine import RecommendationEngine
from ai_data_readiness.engines.quality_report_generator import QualityReportGenerator
from ai_data_readiness.models.base_models import (
    QualityReport, QualityIssue, QualityDimension, AIReadinessScore,
    DimensionScore, ImprovementArea
)


def create_sample_quality_report():
    """Create a sample quality report with various issues."""
    
    # Create sample quality issues
    issues = [
        QualityIssue(
            dimension=QualityDimension.COMPLETENESS,
            severity="high",
            description="Critical customer data missing",
            affected_columns=["customer_id", "email"],
            affected_rows=3500,
            recommendation="Urgent data collection needed",
            column="customer_id",
            category="completeness",
            issue_type="missing_values",
            affected_percentage=0.35
        ),
        QualityIssue(
            dimension=QualityDimension.ACCURACY,
            severity="medium",
            description="Phone number format inconsistencies",
            affected_columns=["phone_number"],
            affected_rows=1200,
            recommendation="Standardize phone format",
            column="phone_number",
            category="accuracy",
            issue_type="format_inconsistency",
            affected_percentage=0.12
        ),
        QualityIssue(
            dimension=QualityDimension.CONSISTENCY,
            severity="high",
            description="Duplicate customer records detected",
            affected_columns=["customer_id", "email"],
            affected_rows=800,
            recommendation="Implement deduplication",
            column="customer_id",
            category="consistency",
            issue_type="duplicate_records",
            affected_percentage=0.08
        ),
        QualityIssue(
            dimension=QualityDimension.VALIDITY,
            severity="medium",
            description="Invalid date formats in transaction data",
            affected_columns=["transaction_date"],
            affected_rows=450,
            recommendation="Fix date parsing",
            column="transaction_date",
            category="validity",
            issue_type="invalid_format",
            affected_percentage=0.045
        )
    ]
    
    # Create quality report
    quality_report = QualityReport(
        dataset_id="customer_transactions_2024",
        overall_score=0.62,
        completeness_score=0.65,
        accuracy_score=0.78,
        consistency_score=0.52,
        validity_score=0.73,
        uniqueness_score=0.92,
        timeliness_score=0.68,
        issues=issues
    )
    
    # Add AI-specific attributes
    quality_report.feature_correlations = [
        ("purchase_amount", "total_spent", 0.98),
        ("age", "years_as_customer", 0.96),
        ("income", "credit_limit", 0.94)
    ]
    
    quality_report.class_distribution = {
        "high_value_customer": 1200,
        "medium_value_customer": 3800,
        "low_value_customer": 5000
    }
    
    quality_report.pii_columns = ["customer_id", "email", "phone_number", "ssn"]
    
    return quality_report


def create_sample_ai_readiness_score():
    """Create a sample AI readiness score."""
    
    improvement_areas = [
        ImprovementArea(
            area="Data Completeness",
            current_score=0.65,
            target_score=0.90,
            priority="High",
            estimated_effort="High"
        ),
        ImprovementArea(
            area="Feature Engineering",
            current_score=0.70,
            target_score=0.85,
            priority="Medium",
            estimated_effort="Medium"
        ),
        ImprovementArea(
            area="Bias Mitigation",
            current_score=0.75,
            target_score=0.90,
            priority="High",
            estimated_effort="Medium"
        )
    ]
    
    dimensions = {
        "data_quality": DimensionScore(dimension="data_quality", score=0.62, weight=0.30),
        "feature_quality": DimensionScore(dimension="feature_quality", score=0.70, weight=0.25),
        "bias_assessment": DimensionScore(dimension="bias_assessment", score=0.75, weight=0.20),
        "compliance": DimensionScore(dimension="compliance", score=0.68, weight=0.15),
        "scalability": DimensionScore(dimension="scalability", score=0.80, weight=0.10)
    }
    
    return AIReadinessScore(
        overall_score=0.69,
        data_quality_score=0.62,
        feature_quality_score=0.70,
        bias_score=0.75,
        compliance_score=0.68,
        scalability_score=0.80,
        dimensions=dimensions,
        improvement_areas=improvement_areas
    )


def demo_recommendation_engine():
    """Demonstrate the recommendation engine capabilities."""
    
    print("=" * 80)
    print("AI DATA READINESS RECOMMENDATION ENGINE DEMO")
    print("=" * 80)
    
    # Create sample data
    quality_report = create_sample_quality_report()
    
    # Initialize recommendation engine
    rec_engine = RecommendationEngine()
    
    print(f"\nDataset: {quality_report.dataset_id}")
    print(f"Overall Quality Score: {quality_report.overall_score:.2f}")
    print(f"Total Issues: {len(quality_report.issues)}")
    
    # Generate recommendations
    print("\nGenerating recommendations...")
    recommendations = rec_engine.generate_recommendations(quality_report)
    
    print(f"\nGenerated {len(recommendations)} recommendations:")
    print("-" * 60)
    
    for i, rec in enumerate(recommendations[:8], 1):  # Show top 8
        print(f"\n{i}. {rec.title}")
        print(f"   Priority: {rec.priority.value}")
        print(f"   Type: {rec.type.value}")
        print(f"   Category: {rec.category}")
        print(f"   Impact: {rec.estimated_impact}")
        print(f"   Effort: {rec.implementation_effort}")
        print(f"   Action Items:")
        for action in rec.action_items[:2]:  # Show first 2 actions
            print(f"     • {action}")
        if len(rec.action_items) > 2:
            print(f"     • ... and {len(rec.action_items) - 2} more actions")
    
    # Generate improvement roadmap
    print("\n" + "=" * 60)
    print("IMPROVEMENT ROADMAP")
    print("=" * 60)
    
    roadmap = rec_engine.generate_improvement_roadmap(recommendations)
    
    print(f"\nImmediate Actions ({len(roadmap['immediate_actions'])} items):")
    for action in roadmap['immediate_actions'][:3]:
        print(f"  • {action.title}")
    
    print(f"\nShort-term Improvements ({len(roadmap['short_term_improvements'])} items):")
    for action in roadmap['short_term_improvements'][:3]:
        print(f"  • {action.title}")
    
    print(f"\nResource Requirements:")
    resources = roadmap['resource_requirements']
    print(f"  • Total Recommendations: {resources['total_recommendations']}")
    print(f"  • Estimated Person-Weeks: {resources['estimated_person_weeks']:.1f}")
    print(f"  • Effort Distribution: {resources['effort_distribution']}")


def demo_quality_report_generator():
    """Demonstrate the quality report generator capabilities."""
    
    print("\n\n" + "=" * 80)
    print("AI DATA READINESS QUALITY REPORT GENERATOR DEMO")
    print("=" * 80)
    
    # Create sample data
    quality_report = create_sample_quality_report()
    ai_readiness_score = create_sample_ai_readiness_score()
    
    # Initialize report generator
    report_generator = QualityReportGenerator()
    
    print(f"\nGenerating comprehensive report for: {quality_report.dataset_id}")
    
    # Generate comprehensive report
    report = report_generator.generate_comprehensive_report(
        quality_report,
        ai_readiness_score,
        include_recommendations=True
    )
    
    print(f"Report generated at: {report['report_metadata']['generated_at']}")
    print(f"Total recommendations: {report['recommendations_count']}")
    print(f"Critical issues: {report['critical_issues_count']}")
    
    # Show executive summary
    print("\n" + "-" * 60)
    print("EXECUTIVE SUMMARY")
    print("-" * 60)
    
    exec_summary = report['sections']['Executive Summary']
    print(f"Overall Assessment: {exec_summary['overall_assessment']}")
    print(f"Data Quality Score: {exec_summary['data_quality_score']}")
    print(f"AI Readiness Score: {exec_summary['ai_readiness_score']}")
    
    print(f"\nKey Findings:")
    for finding in exec_summary['key_findings']:
        print(f"  • {finding}")
    
    print(f"\nCritical Actions Needed:")
    for action in exec_summary['critical_actions_needed']:
        print(f"  • {action}")
    
    # Show quality overview
    print("\n" + "-" * 60)
    print("DATA QUALITY OVERVIEW")
    print("-" * 60)
    
    quality_overview = report['sections']['Data Quality Overview']
    overall_score = quality_overview['overall_score']
    print(f"Overall Score: {overall_score['value']} (Grade: {overall_score['grade']})")
    print(f"Interpretation: {overall_score['interpretation']}")
    
    print(f"\nDimension Scores:")
    for dim_name, dim_data in quality_overview['dimension_scores'].items():
        print(f"  • {dim_name.title()}: {dim_data['score']} (Grade: {dim_data['grade']})")
    
    # Show AI readiness assessment
    print("\n" + "-" * 60)
    print("AI READINESS ASSESSMENT")
    print("-" * 60)
    
    ai_section = report['sections']['AI Readiness Assessment']
    overall_readiness = ai_section['overall_readiness']
    print(f"Overall Readiness: {overall_readiness['score']} (Level: {overall_readiness['readiness_level']})")
    
    print(f"\nImprovement Areas:")
    for area in ai_section['improvement_areas']:
        print(f"  • {area['area']}: {area['current_score']:.2f} → {area['target_score']:.2f} (Priority: {area['priority']})")
    
    # Show top issues
    print("\n" + "-" * 60)
    print("TOP CRITICAL ISSUES")
    print("-" * 60)
    
    issues_analysis = report['sections']['Detailed Issues Analysis']
    critical_issues = issues_analysis['critical_issues'][:5]  # Top 5
    
    for i, issue in enumerate(critical_issues, 1):
        print(f"\n{i}. {issue['description']}")
        print(f"   Column: {issue['column']}")
        print(f"   Severity: {issue['severity']}")
        print(f"   Affected: {issue['affected_percentage']:.1%}")
        print(f"   Impact: {issue['impact_assessment']}")
    
    # Export report
    print("\n" + "-" * 60)
    print("EXPORTING REPORT")
    print("-" * 60)
    
    # Export as JSON
    json_path = report_generator.export_report(report, format_type='json')
    print(f"JSON report exported to: {json_path}")
    
    # Export as HTML
    html_path = report_generator.export_report(report, format_type='html')
    print(f"HTML report exported to: {html_path}")
    
    return report


def demo_integration_workflow():
    """Demonstrate the complete integration workflow."""
    
    print("\n\n" + "=" * 80)
    print("COMPLETE INTEGRATION WORKFLOW DEMO")
    print("=" * 80)
    
    # Create sample data
    quality_report = create_sample_quality_report()
    ai_readiness_score = create_sample_ai_readiness_score()
    
    # Initialize engines
    rec_engine = RecommendationEngine()
    report_generator = QualityReportGenerator()
    
    print(f"\nProcessing dataset: {quality_report.dataset_id}")
    print(f"Initial quality score: {quality_report.overall_score:.2f}")
    print(f"Initial AI readiness: {ai_readiness_score.overall_score:.2f}")
    
    # Step 1: Generate recommendations
    print("\nStep 1: Generating recommendations...")
    recommendations = rec_engine.generate_recommendations(quality_report)
    high_priority_count = len([r for r in recommendations if r.priority.value == "HIGH"])
    print(f"Generated {len(recommendations)} recommendations ({high_priority_count} high priority)")
    
    # Step 2: Create improvement roadmap
    print("\nStep 2: Creating improvement roadmap...")
    roadmap = rec_engine.generate_improvement_roadmap(recommendations)
    print(f"Roadmap created with {len(roadmap['immediate_actions'])} immediate actions")
    print(f"Estimated effort: {roadmap['resource_requirements']['estimated_person_weeks']:.1f} person-weeks")
    
    # Step 3: Generate comprehensive report
    print("\nStep 3: Generating comprehensive report...")
    report = report_generator.generate_comprehensive_report(
        quality_report,
        ai_readiness_score,
        include_recommendations=True
    )
    
    # Step 4: Analyze report insights
    print("\nStep 4: Analyzing report insights...")
    summary_metrics = report['summary_metrics']
    key_indicators = summary_metrics['key_indicators']
    
    print(f"Key Indicators:")
    print(f"  • Ready for AI: {key_indicators['ready_for_ai']}")
    print(f"  • Requires immediate attention: {key_indicators['requires_immediate_attention']}")
    print(f"  • Production ready: {key_indicators['production_ready']}")
    print(f"  • Compliance ready: {key_indicators['compliance_ready']}")
    
    improvement_potential = summary_metrics['improvement_potential']
    print(f"\nImprovement Potential:")
    print(f"  • Quick wins available: {improvement_potential['quick_wins_available']}")
    print(f"  • Major improvements needed: {improvement_potential['major_improvements_needed']}")
    print(f"  • Estimated improvement score: {improvement_potential['estimated_improvement_score']:.2f}")
    
    # Step 5: Export final report
    print("\nStep 5: Exporting final report...")
    json_path = report_generator.export_report(report, format_type='json')
    print(f"Complete report exported to: {json_path}")
    
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)
    
    return report, recommendations, roadmap


if __name__ == "__main__":
    # Run all demos
    demo_recommendation_engine()
    demo_quality_report_generator()
    final_report, final_recommendations, final_roadmap = demo_integration_workflow()
    
    print(f"\nDemo completed successfully!")
    print(f"Generated {len(final_recommendations)} recommendations")
    print(f"Created comprehensive report with {len(final_report['sections'])} sections")
    print(f"Improvement roadmap includes {len(final_roadmap['immediate_actions'])} immediate actions")