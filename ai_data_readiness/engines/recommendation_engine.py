"""
Automated remediation recommendation engine for AI data readiness.
Provides actionable insights and recommendations based on quality assessment results.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

from ..models.base_models import QualityReport, QualityIssue, Recommendation, RecommendationPriority, RecommendationType


class RecommendationEngine:
    """
    Automated remediation recommendation engine that analyzes quality issues
    and provides actionable recommendations for improving AI data readiness.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialize_recommendation_rules()
    
    def _initialize_recommendation_rules(self):
        """Initialize recommendation rules and patterns."""
        self.completeness_rules = {
            'high_missing_rate': {
                'threshold': 0.2,
                'recommendations': [
                    'Consider data imputation strategies',
                    'Investigate data collection processes',
                    'Evaluate feature importance for missing columns'
                ]
            },
            'critical_missing_features': {
                'threshold': 0.05,
                'recommendations': [
                    'Prioritize collection of missing critical features',
                    'Consider alternative data sources',
                    'Evaluate model performance impact'
                ]
            }
        }
        
        self.accuracy_rules = {
            'data_type_mismatches': {
                'recommendations': [
                    'Implement data type validation',
                    'Add data conversion pipelines',
                    'Review data ingestion processes'
                ]
            },
            'format_inconsistencies': {
                'recommendations': [
                    'Standardize data formats',
                    'Implement format validation rules',
                    'Create data normalization procedures'
                ]
            }
        }
        
        self.consistency_rules = {
            'duplicate_records': {
                'threshold': 0.01,
                'recommendations': [
                    'Implement deduplication procedures',
                    'Review data collection processes',
                    'Add unique constraint validations'
                ]
            },
            'conflicting_values': {
                'recommendations': [
                    'Establish data governance rules',
                    'Implement conflict resolution procedures',
                    'Add data validation checks'
                ]
            }
        }
        
        self.ai_readiness_rules = {
            'feature_correlation_issues': {
                'threshold': 0.95,
                'recommendations': [
                    'Remove highly correlated features',
                    'Apply dimensionality reduction techniques',
                    'Consider feature engineering approaches'
                ]
            },
            'target_leakage': {
                'recommendations': [
                    'Remove features with target leakage',
                    'Review feature engineering process',
                    'Implement temporal validation'
                ]
            },
            'class_imbalance': {
                'threshold': 0.1,
                'recommendations': [
                    'Apply sampling techniques (SMOTE, undersampling)',
                    'Use class-weighted algorithms',
                    'Consider ensemble methods'
                ]
            }
        }
    
    def generate_recommendations(self, quality_report: QualityReport) -> List[Recommendation]:
        """
        Generate comprehensive recommendations based on quality assessment results.
        
        Args:
            quality_report: Quality assessment results
            
        Returns:
            List of prioritized recommendations
        """
        recommendations = []
        
        # Generate completeness recommendations
        completeness_recs = self._generate_completeness_recommendations(quality_report)
        recommendations.extend(completeness_recs)
        
        # Generate accuracy recommendations
        accuracy_recs = self._generate_accuracy_recommendations(quality_report)
        recommendations.extend(accuracy_recs)
        
        # Generate consistency recommendations
        consistency_recs = self._generate_consistency_recommendations(quality_report)
        recommendations.extend(consistency_recs)
        
        # Generate AI-specific recommendations
        ai_recs = self._generate_ai_readiness_recommendations(quality_report)
        recommendations.extend(ai_recs)
        
        # Prioritize and deduplicate recommendations
        prioritized_recs = self._prioritize_recommendations(recommendations, quality_report)
        
        self.logger.info(f"Generated {len(prioritized_recs)} recommendations for dataset {quality_report.dataset_id}")
        
        return prioritized_recs
    
    def _generate_completeness_recommendations(self, quality_report: QualityReport) -> List[Recommendation]:
        """Generate recommendations for completeness issues."""
        recommendations = []
        
        completeness_issues = [issue for issue in quality_report.issues 
                             if issue.category == 'completeness']
        
        for issue in completeness_issues:
            if issue.severity == 'high' and issue.affected_percentage > 0.2:
                recommendations.append(Recommendation(
                    id=f"completeness_{issue.column}_{datetime.now().timestamp()}",
                    type=RecommendationType.DATA_COLLECTION,
                    priority=RecommendationPriority.HIGH,
                    title=f"Address high missing rate in {issue.column}",
                    description=f"Column {issue.column} has {issue.affected_percentage:.1%} missing values",
                    action_items=[
                        "Investigate data collection process for this column",
                        "Consider imputation strategies (mean, median, mode)",
                        "Evaluate impact on model performance",
                        "Consider removing column if not critical"
                    ],
                    estimated_impact="High - Missing data can significantly impact model performance",
                    implementation_effort="Medium",
                    affected_columns=[issue.column],
                    category="completeness"
                ))
            elif issue.severity == 'medium':
                recommendations.append(Recommendation(
                    id=f"completeness_{issue.column}_{datetime.now().timestamp()}",
                    type=RecommendationType.DATA_PREPROCESSING,
                    priority=RecommendationPriority.MEDIUM,
                    title=f"Improve completeness for {issue.column}",
                    description=f"Column {issue.column} has {issue.affected_percentage:.1%} missing values",
                    action_items=[
                        "Apply appropriate imputation technique",
                        "Document imputation strategy",
                        "Monitor impact on data quality"
                    ],
                    estimated_impact="Medium - Moderate impact on model reliability",
                    implementation_effort="Low",
                    affected_columns=[issue.column],
                    category="completeness"
                ))
        
        return recommendations
    
    def _generate_accuracy_recommendations(self, quality_report: QualityReport) -> List[Recommendation]:
        """Generate recommendations for accuracy issues."""
        recommendations = []
        
        accuracy_issues = [issue for issue in quality_report.issues 
                         if issue.category == 'accuracy']
        
        for issue in accuracy_issues:
            if 'type_mismatch' in issue.issue_type:
                recommendations.append(Recommendation(
                    id=f"accuracy_{issue.column}_{datetime.now().timestamp()}",
                    type=RecommendationType.DATA_VALIDATION,
                    priority=RecommendationPriority.HIGH,
                    title=f"Fix data type issues in {issue.column}",
                    description=f"Data type inconsistencies detected in {issue.column}",
                    action_items=[
                        "Implement data type validation",
                        "Add type conversion procedures",
                        "Review data ingestion pipeline",
                        "Add schema validation"
                    ],
                    estimated_impact="High - Type mismatches can cause model failures",
                    implementation_effort="Medium",
                    affected_columns=[issue.column],
                    category="accuracy"
                ))
            elif 'format_inconsistency' in issue.issue_type:
                recommendations.append(Recommendation(
                    id=f"accuracy_{issue.column}_{datetime.now().timestamp()}",
                    type=RecommendationType.DATA_STANDARDIZATION,
                    priority=RecommendationPriority.MEDIUM,
                    title=f"Standardize format in {issue.column}",
                    description=f"Format inconsistencies detected in {issue.column}",
                    action_items=[
                        "Define standard format rules",
                        "Implement format normalization",
                        "Add format validation checks"
                    ],
                    estimated_impact="Medium - Format issues can affect feature engineering",
                    implementation_effort="Low",
                    affected_columns=[issue.column],
                    category="accuracy"
                ))
        
        return recommendations
    
    def _generate_consistency_recommendations(self, quality_report: QualityReport) -> List[Recommendation]:
        """Generate recommendations for consistency issues."""
        recommendations = []
        
        consistency_issues = [issue for issue in quality_report.issues 
                            if issue.category == 'consistency']
        
        for issue in consistency_issues:
            if 'duplicate' in issue.issue_type:
                recommendations.append(Recommendation(
                    id=f"consistency_{issue.column}_{datetime.now().timestamp()}",
                    type=RecommendationType.DATA_CLEANING,
                    priority=RecommendationPriority.HIGH,
                    title=f"Remove duplicate records",
                    description=f"Duplicate records detected affecting {issue.column}",
                    action_items=[
                        "Implement deduplication logic",
                        "Define record matching criteria",
                        "Review data collection processes",
                        "Add unique constraints where appropriate"
                    ],
                    estimated_impact="High - Duplicates can bias model training",
                    implementation_effort="Medium",
                    affected_columns=[issue.column] if issue.column else [],
                    category="consistency"
                ))
        
        return recommendations
    
    def _generate_ai_readiness_recommendations(self, quality_report: QualityReport) -> List[Recommendation]:
        """Generate AI-specific recommendations."""
        recommendations = []
        
        # Check for feature correlation issues
        if hasattr(quality_report, 'feature_correlations') and quality_report.feature_correlations:
            high_corr_pairs = [(col1, col2, corr) for col1, col2, corr in quality_report.feature_correlations 
                             if abs(corr) > 0.95]
            
            if high_corr_pairs:
                recommendations.append(Recommendation(
                    id=f"ai_readiness_correlation_{datetime.now().timestamp()}",
                    type=RecommendationType.FEATURE_ENGINEERING,
                    priority=RecommendationPriority.HIGH,
                    title="Address highly correlated features",
                    description=f"Found {len(high_corr_pairs)} highly correlated feature pairs",
                    action_items=[
                        "Remove redundant highly correlated features",
                        "Apply dimensionality reduction (PCA, LDA)",
                        "Consider feature selection techniques",
                        "Evaluate feature importance"
                    ],
                    estimated_impact="High - Multicollinearity can affect model performance",
                    implementation_effort="Medium",
                    affected_columns=[col for col1, col2, _ in high_corr_pairs for col in [col1, col2]],
                    category="ai_readiness"
                ))
        
        # Check for class imbalance
        if hasattr(quality_report, 'class_distribution') and quality_report.class_distribution:
            min_class_ratio = min(quality_report.class_distribution.values()) / sum(quality_report.class_distribution.values())
            
            if min_class_ratio < 0.1:
                recommendations.append(Recommendation(
                    id=f"ai_readiness_imbalance_{datetime.now().timestamp()}",
                    type=RecommendationType.DATA_BALANCING,
                    priority=RecommendationPriority.HIGH,
                    title="Address class imbalance",
                    description=f"Severe class imbalance detected (min class: {min_class_ratio:.1%})",
                    action_items=[
                        "Apply SMOTE or other oversampling techniques",
                        "Consider undersampling majority class",
                        "Use class-weighted algorithms",
                        "Evaluate stratified sampling"
                    ],
                    estimated_impact="High - Class imbalance can bias model predictions",
                    implementation_effort="Medium",
                    affected_columns=[],
                    category="ai_readiness"
                ))
        
        return recommendations
    
    def _prioritize_recommendations(self, recommendations: List[Recommendation], 
                                  quality_report: QualityReport) -> List[Recommendation]:
        """Prioritize recommendations based on impact and quality scores."""
        
        # Calculate priority scores
        for rec in recommendations:
            priority_score = self._calculate_priority_score(rec, quality_report)
            rec.priority_score = priority_score
        
        # Sort by priority score (descending) and priority level
        priority_order = {
            RecommendationPriority.HIGH: 3,
            RecommendationPriority.MEDIUM: 2,
            RecommendationPriority.LOW: 1
        }
        
        sorted_recs = sorted(recommendations, 
                           key=lambda x: (priority_order.get(x.priority, 0), 
                                        getattr(x, 'priority_score', 0)), 
                           reverse=True)
        
        # Remove duplicates while preserving order
        seen_titles = set()
        deduplicated_recs = []
        
        for rec in sorted_recs:
            if rec.title not in seen_titles:
                seen_titles.add(rec.title)
                deduplicated_recs.append(rec)
        
        return deduplicated_recs[:20]  # Limit to top 20 recommendations
    
    def _calculate_priority_score(self, recommendation: Recommendation, 
                                quality_report: QualityReport) -> float:
        """Calculate priority score for recommendation ranking."""
        score = 0.0
        
        # Base score from priority level
        priority_scores = {
            RecommendationPriority.HIGH: 10.0,
            RecommendationPriority.MEDIUM: 5.0,
            RecommendationPriority.LOW: 1.0
        }
        score += priority_scores.get(recommendation.priority, 0.0)
        
        # Adjust based on overall quality scores
        if quality_report.overall_score < 0.5:
            score *= 1.5  # Boost priority for low-quality datasets
        
        # Adjust based on affected columns count
        if recommendation.affected_columns:
            score += len(recommendation.affected_columns) * 0.1
        
        # Adjust based on recommendation type
        type_multipliers = {
            RecommendationType.DATA_COLLECTION: 1.2,
            RecommendationType.DATA_VALIDATION: 1.1,
            RecommendationType.DATA_CLEANING: 1.0,
            RecommendationType.FEATURE_ENGINEERING: 0.9,
            RecommendationType.DATA_BALANCING: 1.1
        }
        score *= type_multipliers.get(recommendation.type, 1.0)
        
        return score
    
    def generate_improvement_roadmap(self, recommendations: List[Recommendation]) -> Dict[str, Any]:
        """Generate a structured improvement roadmap from recommendations."""
        
        # Group recommendations by category and priority
        roadmap = {
            'immediate_actions': [],
            'short_term_improvements': [],
            'long_term_enhancements': [],
            'estimated_timeline': {},
            'resource_requirements': {}
        }
        
        high_priority = [r for r in recommendations if r.priority == RecommendationPriority.HIGH]
        medium_priority = [r for r in recommendations if r.priority == RecommendationPriority.MEDIUM]
        low_priority = [r for r in recommendations if r.priority == RecommendationPriority.LOW]
        
        roadmap['immediate_actions'] = high_priority[:5]
        roadmap['short_term_improvements'] = medium_priority[:8]
        roadmap['long_term_enhancements'] = low_priority[:7]
        
        # Estimate timeline
        roadmap['estimated_timeline'] = {
            'immediate_actions': '1-2 weeks',
            'short_term_improvements': '1-2 months',
            'long_term_enhancements': '3-6 months'
        }
        
        # Estimate resource requirements
        effort_counts = {}
        for rec in recommendations:
            effort = rec.implementation_effort
            effort_counts[effort] = effort_counts.get(effort, 0) + 1
        
        roadmap['resource_requirements'] = {
            'total_recommendations': len(recommendations),
            'effort_distribution': effort_counts,
            'estimated_person_weeks': self._estimate_effort(recommendations)
        }
        
        return roadmap
    
    def _estimate_effort(self, recommendations: List[Recommendation]) -> float:
        """Estimate total effort in person-weeks."""
        effort_mapping = {
            'Low': 0.5,
            'Medium': 2.0,
            'High': 5.0
        }
        
        total_weeks = sum(effort_mapping.get(rec.implementation_effort, 1.0) 
                         for rec in recommendations)
        
        return total_weeks