"""
Quality report generation engine for AI data readiness platform.
Generates comprehensive, actionable quality reports with insights and recommendations.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json
import logging
from datetime import datetime
from pathlib import Path

from ..models.base_models import QualityReport, QualityIssue, Recommendation, AIReadinessScore
from .recommendation_engine import RecommendationEngine


@dataclass
class ReportSection:
    """Represents a section in the quality report."""
    title: str
    content: Dict[str, Any]
    priority: int = 1
    include_in_summary: bool = True


class QualityReportGenerator:
    """
    Generates comprehensive quality reports with actionable insights
    for AI data readiness assessment.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.recommendation_engine = RecommendationEngine()
    
    def generate_comprehensive_report(self, quality_report: QualityReport, 
                                    ai_readiness_score: AIReadinessScore,
                                    include_recommendations: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive quality report with all sections.
        
        Args:
            quality_report: Basic quality assessment results
            ai_readiness_score: AI readiness scoring results
            include_recommendations: Whether to include recommendations
            
        Returns:
            Complete quality report dictionary
        """
        
        # Generate recommendations if requested
        recommendations = []
        if include_recommendations:
            recommendations = self.recommendation_engine.generate_recommendations(quality_report)
        
        # Build report sections
        report_sections = []
        
        # Executive Summary
        exec_summary = self._generate_executive_summary(quality_report, ai_readiness_score, recommendations)
        report_sections.append(ReportSection("Executive Summary", exec_summary, priority=1))
        
        # Data Quality Overview
        quality_overview = self._generate_quality_overview(quality_report)
        report_sections.append(ReportSection("Data Quality Overview", quality_overview, priority=2))
        
        # AI Readiness Assessment
        ai_assessment = self._generate_ai_readiness_section(ai_readiness_score)
        report_sections.append(ReportSection("AI Readiness Assessment", ai_assessment, priority=3))
        
        # Detailed Issues Analysis
        issues_analysis = self._generate_issues_analysis(quality_report.issues)
        report_sections.append(ReportSection("Detailed Issues Analysis", issues_analysis, priority=4))
        
        # Recommendations and Action Plan
        if recommendations:
            recommendations_section = self._generate_recommendations_section(recommendations)
            report_sections.append(ReportSection("Recommendations & Action Plan", recommendations_section, priority=5))
        
        # Data Profiling Results
        profiling_results = self._generate_profiling_section(quality_report)
        report_sections.append(ReportSection("Data Profiling Results", profiling_results, priority=6))
        
        # Compliance and Governance
        compliance_section = self._generate_compliance_section(quality_report)
        report_sections.append(ReportSection("Compliance & Governance", compliance_section, priority=7))
        
        # Build final report
        comprehensive_report = {
            'report_metadata': {
                'dataset_id': quality_report.dataset_id,
                'generated_at': datetime.now().isoformat(),
                'report_version': '1.0',
                'generator': 'AI Data Readiness Platform'
            },
            'sections': {section.title: section.content for section in report_sections},
            'summary_metrics': self._generate_summary_metrics(quality_report, ai_readiness_score),
            'recommendations_count': len(recommendations),
            'critical_issues_count': len([issue for issue in quality_report.issues if issue.severity == 'high'])
        }
        
        self.logger.info(f"Generated comprehensive report for dataset {quality_report.dataset_id}")
        
        return comprehensive_report
    
    def _generate_executive_summary(self, quality_report: QualityReport, 
                                  ai_readiness_score: AIReadinessScore,
                                  recommendations: List[Recommendation]) -> Dict[str, Any]:
        """Generate executive summary section."""
        
        # Determine overall assessment
        overall_score = quality_report.overall_score
        readiness_score = ai_readiness_score.overall_score
        
        if overall_score >= 0.8 and readiness_score >= 0.8:
            assessment = "Excellent"
            status_color = "green"
        elif overall_score >= 0.6 and readiness_score >= 0.6:
            assessment = "Good"
            status_color = "yellow"
        elif overall_score >= 0.4 and readiness_score >= 0.4:
            assessment = "Fair"
            status_color = "orange"
        else:
            assessment = "Poor"
            status_color = "red"
        
        # Key findings
        key_findings = []
        
        if quality_report.completeness_score < 0.7:
            key_findings.append("Significant data completeness issues detected")
        
        if quality_report.accuracy_score < 0.7:
            key_findings.append("Data accuracy concerns require attention")
        
        if quality_report.consistency_score < 0.7:
            key_findings.append("Data consistency issues may impact model performance")
        
        if readiness_score < 0.6:
            key_findings.append("Dataset requires significant preparation for AI applications")
        
        # Critical actions
        critical_actions = [rec.title for rec in recommendations[:3] 
                          if rec.priority.value == "HIGH"]
        
        return {
            'overall_assessment': assessment,
            'status_color': status_color,
            'data_quality_score': round(overall_score, 2),
            'ai_readiness_score': round(readiness_score, 2),
            'key_findings': key_findings,
            'critical_actions_needed': critical_actions,
            'total_issues': len(quality_report.issues),
            'high_priority_issues': len([i for i in quality_report.issues if i.severity == 'high']),
            'recommendations_provided': len(recommendations),
            'estimated_improvement_time': self._estimate_improvement_timeline(recommendations)
        }
    
    def _generate_quality_overview(self, quality_report: QualityReport) -> Dict[str, Any]:
        """Generate data quality overview section."""
        
        return {
            'overall_score': {
                'value': round(quality_report.overall_score, 3),
                'grade': self._score_to_grade(quality_report.overall_score),
                'interpretation': self._interpret_score(quality_report.overall_score)
            },
            'dimension_scores': {
                'completeness': {
                    'score': round(quality_report.completeness_score, 3),
                    'grade': self._score_to_grade(quality_report.completeness_score),
                    'description': 'Measures the extent to which data is present and not missing'
                },
                'accuracy': {
                    'score': round(quality_report.accuracy_score, 3),
                    'grade': self._score_to_grade(quality_report.accuracy_score),
                    'description': 'Measures correctness and validity of data values'
                },
                'consistency': {
                    'score': round(quality_report.consistency_score, 3),
                    'grade': self._score_to_grade(quality_report.consistency_score),
                    'description': 'Measures uniformity and coherence across the dataset'
                },
                'validity': {
                    'score': round(quality_report.validity_score, 3),
                    'grade': self._score_to_grade(quality_report.validity_score),
                    'description': 'Measures adherence to defined formats and constraints'
                }
            },
            'quality_trends': self._analyze_quality_trends(quality_report),
            'benchmark_comparison': self._generate_benchmark_comparison(quality_report)
        }
    
    def _generate_ai_readiness_section(self, ai_readiness_score: AIReadinessScore) -> Dict[str, Any]:
        """Generate AI readiness assessment section."""
        
        return {
            'overall_readiness': {
                'score': round(ai_readiness_score.overall_score, 3),
                'grade': self._score_to_grade(ai_readiness_score.overall_score),
                'readiness_level': self._determine_readiness_level(ai_readiness_score.overall_score)
            },
            'dimension_analysis': {
                'data_quality': {
                    'score': round(ai_readiness_score.data_quality_score, 3),
                    'impact': 'High - Fundamental requirement for AI success'
                },
                'feature_quality': {
                    'score': round(ai_readiness_score.feature_quality_score, 3),
                    'impact': 'High - Directly affects model performance'
                },
                'bias_assessment': {
                    'score': round(ai_readiness_score.bias_score, 3),
                    'impact': 'Critical - Affects model fairness and compliance'
                },
                'compliance': {
                    'score': round(ai_readiness_score.compliance_score, 3),
                    'impact': 'High - Required for production deployment'
                },
                'scalability': {
                    'score': round(ai_readiness_score.scalability_score, 3),
                    'impact': 'Medium - Important for production systems'
                }
            },
            'improvement_areas': [
                {
                    'area': area.area,
                    'current_score': area.current_score,
                    'target_score': area.target_score,
                    'priority': area.priority,
                    'estimated_effort': area.estimated_effort
                }
                for area in ai_readiness_score.improvement_areas
            ],
            'model_suitability': self._assess_model_suitability(ai_readiness_score)
        }
    
    def _generate_issues_analysis(self, issues: List[QualityIssue]) -> Dict[str, Any]:
        """Generate detailed issues analysis section."""
        
        # Group issues by category and severity
        issues_by_category = {}
        issues_by_severity = {'high': [], 'medium': [], 'low': []}
        
        for issue in issues:
            # Group by category
            if issue.category not in issues_by_category:
                issues_by_category[issue.category] = []
            issues_by_category[issue.category].append(issue)
            
            # Group by severity
            if issue.severity in issues_by_severity:
                issues_by_severity[issue.severity].append(issue)
        
        # Generate category analysis
        category_analysis = {}
        for category, category_issues in issues_by_category.items():
            category_analysis[category] = {
                'total_issues': len(category_issues),
                'severity_breakdown': {
                    'high': len([i for i in category_issues if i.severity == 'high']),
                    'medium': len([i for i in category_issues if i.severity == 'medium']),
                    'low': len([i for i in category_issues if i.severity == 'low'])
                },
                'affected_columns': list(set(i.column for i in category_issues if i.column)),
                'most_critical': [
                    {
                        'column': issue.column,
                        'issue_type': issue.issue_type,
                        'description': issue.description,
                        'affected_percentage': issue.affected_percentage
                    }
                    for issue in sorted(category_issues, key=lambda x: x.affected_percentage, reverse=True)[:3]
                ]
            }
        
        return {
            'total_issues': len(issues),
            'severity_distribution': {
                'high': len(issues_by_severity['high']),
                'medium': len(issues_by_severity['medium']),
                'low': len(issues_by_severity['low'])
            },
            'category_analysis': category_analysis,
            'critical_issues': [
                {
                    'column': issue.column,
                    'category': issue.category,
                    'issue_type': issue.issue_type,
                    'severity': issue.severity,
                    'description': issue.description,
                    'affected_percentage': issue.affected_percentage,
                    'impact_assessment': self._assess_issue_impact(issue)
                }
                for issue in sorted(issues, key=lambda x: x.affected_percentage, reverse=True)[:10]
            ],
            'resolution_complexity': self._assess_resolution_complexity(issues)
        }
    
    def _generate_recommendations_section(self, recommendations: List[Recommendation]) -> Dict[str, Any]:
        """Generate recommendations and action plan section."""
        
        # Generate improvement roadmap
        roadmap = self.recommendation_engine.generate_improvement_roadmap(recommendations)
        
        # Group recommendations by type
        recommendations_by_type = {}
        for rec in recommendations:
            rec_type = rec.type.value
            if rec_type not in recommendations_by_type:
                recommendations_by_type[rec_type] = []
            recommendations_by_type[rec_type].append(rec)
        
        return {
            'improvement_roadmap': roadmap,
            'recommendations_by_priority': {
                'high': [self._format_recommendation(r) for r in recommendations if r.priority.value == "HIGH"],
                'medium': [self._format_recommendation(r) for r in recommendations if r.priority.value == "MEDIUM"],
                'low': [self._format_recommendation(r) for r in recommendations if r.priority.value == "LOW"]
            },
            'recommendations_by_type': {
                rec_type: [self._format_recommendation(r) for r in recs]
                for rec_type, recs in recommendations_by_type.items()
            },
            'implementation_guidance': self._generate_implementation_guidance(recommendations),
            'success_metrics': self._define_success_metrics(recommendations)
        }
    
    def _generate_profiling_section(self, quality_report: QualityReport) -> Dict[str, Any]:
        """Generate data profiling results section."""
        
        return {
            'dataset_overview': {
                'total_records': getattr(quality_report, 'total_records', 'N/A'),
                'total_columns': getattr(quality_report, 'total_columns', 'N/A'),
                'data_size': getattr(quality_report, 'data_size_mb', 'N/A'),
                'last_updated': getattr(quality_report, 'last_updated', 'N/A')
            },
            'column_analysis': getattr(quality_report, 'column_profiles', {}),
            'statistical_summary': getattr(quality_report, 'statistical_summary', {}),
            'data_types_distribution': getattr(quality_report, 'data_types', {}),
            'missing_data_patterns': getattr(quality_report, 'missing_patterns', {}),
            'outlier_analysis': getattr(quality_report, 'outlier_summary', {})
        }
    
    def _generate_compliance_section(self, quality_report: QualityReport) -> Dict[str, Any]:
        """Generate compliance and governance section."""
        
        return {
            'privacy_assessment': {
                'pii_detected': getattr(quality_report, 'pii_columns', []),
                'sensitive_data_risk': self._assess_privacy_risk(quality_report),
                'anonymization_required': len(getattr(quality_report, 'pii_columns', [])) > 0
            },
            'regulatory_compliance': {
                'gdpr_compliance': getattr(quality_report, 'gdpr_compliant', 'Unknown'),
                'ccpa_compliance': getattr(quality_report, 'ccpa_compliant', 'Unknown'),
                'industry_standards': getattr(quality_report, 'industry_compliance', {})
            },
            'governance_metrics': {
                'data_lineage_available': getattr(quality_report, 'has_lineage', False),
                'documentation_completeness': getattr(quality_report, 'documentation_score', 0.0),
                'access_controls': getattr(quality_report, 'access_controlled', False)
            }
        }
    
    def _generate_summary_metrics(self, quality_report: QualityReport, 
                                ai_readiness_score: AIReadinessScore) -> Dict[str, Any]:
        """Generate summary metrics for the report."""
        
        return {
            'overall_health': {
                'data_quality': round(quality_report.overall_score, 2),
                'ai_readiness': round(ai_readiness_score.overall_score, 2),
                'combined_score': round((quality_report.overall_score + ai_readiness_score.overall_score) / 2, 2)
            },
            'key_indicators': {
                'ready_for_ai': ai_readiness_score.overall_score >= 0.7,
                'requires_immediate_attention': quality_report.overall_score < 0.5,
                'production_ready': (quality_report.overall_score >= 0.8 and 
                                   ai_readiness_score.overall_score >= 0.8),
                'compliance_ready': ai_readiness_score.compliance_score >= 0.8
            },
            'improvement_potential': {
                'quick_wins_available': len([i for i in quality_report.issues if i.severity == 'low']) > 0,
                'major_improvements_needed': len([i for i in quality_report.issues if i.severity == 'high']) > 3,
                'estimated_improvement_score': min(1.0, quality_report.overall_score + 0.3)
            }
        }
    
    # Helper methods
    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'
    
    def _interpret_score(self, score: float) -> str:
        """Provide interpretation of score."""
        if score >= 0.9:
            return 'Excellent - Dataset exceeds quality standards'
        elif score >= 0.8:
            return 'Good - Dataset meets most quality requirements'
        elif score >= 0.7:
            return 'Acceptable - Dataset has minor quality issues'
        elif score >= 0.6:
            return 'Fair - Dataset requires some improvements'
        else:
            return 'Poor - Dataset needs significant quality improvements'
    
    def _determine_readiness_level(self, score: float) -> str:
        """Determine AI readiness level."""
        if score >= 0.8:
            return 'Production Ready'
        elif score >= 0.6:
            return 'Development Ready'
        elif score >= 0.4:
            return 'Preparation Required'
        else:
            return 'Significant Work Needed'
    
    def _format_recommendation(self, recommendation: Recommendation) -> Dict[str, Any]:
        """Format recommendation for report output."""
        return {
            'title': recommendation.title,
            'description': recommendation.description,
            'priority': recommendation.priority.value,
            'type': recommendation.type.value,
            'action_items': recommendation.action_items,
            'estimated_impact': recommendation.estimated_impact,
            'implementation_effort': recommendation.implementation_effort,
            'affected_columns': recommendation.affected_columns,
            'category': recommendation.category
        }
    
    def _estimate_improvement_timeline(self, recommendations: List[Recommendation]) -> str:
        """Estimate timeline for implementing recommendations."""
        high_priority_count = len([r for r in recommendations if r.priority.value == "HIGH"])
        
        if high_priority_count > 5:
            return "2-3 months"
        elif high_priority_count > 2:
            return "1-2 months"
        else:
            return "2-4 weeks"
    
    def _analyze_quality_trends(self, quality_report: QualityReport) -> Dict[str, str]:
        """Analyze quality trends (placeholder for historical data)."""
        return {
            'completeness': 'Stable',
            'accuracy': 'Improving',
            'consistency': 'Declining',
            'overall': 'Stable'
        }
    
    def _generate_benchmark_comparison(self, quality_report: QualityReport) -> Dict[str, Any]:
        """Generate benchmark comparison."""
        return {
            'industry_average': 0.72,
            'your_score': quality_report.overall_score,
            'percentile': min(95, max(5, int(quality_report.overall_score * 100))),
            'comparison': 'Above Average' if quality_report.overall_score > 0.72 else 'Below Average'
        }
    
    def _assess_model_suitability(self, ai_readiness_score: AIReadinessScore) -> Dict[str, str]:
        """Assess suitability for different model types."""
        score = ai_readiness_score.overall_score
        
        return {
            'supervised_learning': 'Suitable' if score >= 0.7 else 'Needs Improvement',
            'unsupervised_learning': 'Suitable' if score >= 0.6 else 'Needs Improvement',
            'deep_learning': 'Suitable' if score >= 0.8 else 'Needs Improvement',
            'production_deployment': 'Ready' if score >= 0.85 else 'Not Ready'
        }
    
    def _assess_issue_impact(self, issue: QualityIssue) -> str:
        """Assess the impact of a quality issue."""
        if issue.severity == 'high' and issue.affected_percentage > 0.2:
            return 'Critical - Will significantly impact model performance'
        elif issue.severity == 'high':
            return 'High - May cause model training issues'
        elif issue.severity == 'medium':
            return 'Medium - Could affect model accuracy'
        else:
            return 'Low - Minor impact on model performance'
    
    def _assess_resolution_complexity(self, issues: List[QualityIssue]) -> Dict[str, int]:
        """Assess complexity of resolving issues."""
        complexity_counts = {'simple': 0, 'moderate': 0, 'complex': 0}
        
        for issue in issues:
            if issue.category == 'completeness' and issue.affected_percentage < 0.1:
                complexity_counts['simple'] += 1
            elif issue.category in ['accuracy', 'consistency'] and issue.severity == 'low':
                complexity_counts['simple'] += 1
            elif issue.severity == 'high' or issue.affected_percentage > 0.3:
                complexity_counts['complex'] += 1
            else:
                complexity_counts['moderate'] += 1
        
        return complexity_counts
    
    def _assess_privacy_risk(self, quality_report: QualityReport) -> str:
        """Assess privacy risk level."""
        pii_columns = getattr(quality_report, 'pii_columns', [])
        
        if len(pii_columns) > 5:
            return 'High'
        elif len(pii_columns) > 2:
            return 'Medium'
        elif len(pii_columns) > 0:
            return 'Low'
        else:
            return 'None'
    
    def _generate_implementation_guidance(self, recommendations: List[Recommendation]) -> Dict[str, Any]:
        """Generate implementation guidance."""
        return {
            'getting_started': [
                'Review high-priority recommendations first',
                'Assess resource requirements and timeline',
                'Create implementation plan with milestones',
                'Set up monitoring for progress tracking'
            ],
            'best_practices': [
                'Implement changes incrementally',
                'Test impact of each change',
                'Document all modifications',
                'Maintain data backups during changes'
            ],
            'common_pitfalls': [
                'Trying to fix all issues simultaneously',
                'Not validating changes before deployment',
                'Ignoring downstream system impacts',
                'Insufficient testing of data transformations'
            ]
        }
    
    def _define_success_metrics(self, recommendations: List[Recommendation]) -> Dict[str, Any]:
        """Define success metrics for recommendations."""
        return {
            'quality_improvement_targets': {
                'overall_score_increase': 0.2,
                'completeness_target': 0.9,
                'accuracy_target': 0.85,
                'consistency_target': 0.8
            },
            'implementation_metrics': {
                'recommendations_completed': 0,
                'high_priority_resolved': 0,
                'estimated_completion_date': 'TBD'
            },
            'business_impact_metrics': [
                'Model performance improvement',
                'Data processing efficiency gains',
                'Compliance risk reduction',
                'Time to model deployment'
            ]
        }
    
    def export_report(self, report: Dict[str, Any], format_type: str = 'json', 
                     output_path: Optional[str] = None) -> str:
        """
        Export report to specified format.
        
        Args:
            report: Generated report dictionary
            format_type: Export format ('json', 'html', 'pdf')
            output_path: Optional output file path
            
        Returns:
            Path to exported file
        """
        
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_id = report['report_metadata']['dataset_id']
            output_path = f"quality_report_{dataset_id}_{timestamp}.{format_type}"
        
        if format_type == 'json':
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        elif format_type == 'html':
            html_content = self._generate_html_report(report)
            with open(output_path, 'w') as f:
                f.write(html_content)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        self.logger.info(f"Report exported to {output_path}")
        return output_path
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML version of the report."""
        # This would contain HTML template generation logic
        # For now, return a simple HTML structure
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Data Readiness Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ccc; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AI Data Readiness Report</h1>
                <p>Dataset: {report['report_metadata']['dataset_id']}</p>
                <p>Generated: {report['report_metadata']['generated_at']}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="metric">
                    <strong>Overall Assessment:</strong> 
                    {report['sections']['Executive Summary']['overall_assessment']}
                </div>
                <div class="metric">
                    <strong>Data Quality Score:</strong> 
                    {report['sections']['Executive Summary']['data_quality_score']}
                </div>
                <div class="metric">
                    <strong>AI Readiness Score:</strong> 
                    {report['sections']['Executive Summary']['ai_readiness_score']}
                </div>
            </div>
            
            <!-- Additional sections would be added here -->
            
        </body>
        </html>
        """
        
        return html_template