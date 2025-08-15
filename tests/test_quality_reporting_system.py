"""
Tests for quality reporting and recommendation system.
Tests recommendation accuracy, relevance, and report generation quality.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch
import json

from ai_data_readiness.engines.recommendation_engine import RecommendationEngine
from ai_data_readiness.engines.quality_report_generator import QualityReportGenerator
from ai_data_readiness.models.base_models import (
    QualityReport, QualityIssue, QualityDimension, AIReadinessScore,
    DimensionScore, ImprovementArea, Recommendation, RecommendationType,
    RecommendationPriority
)


class TestRecommendationEngine:
    """Test the recommendation engine for accuracy and relevance."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = RecommendationEngine()
        
        # Create sample quality issues
        self.completeness_issue = QualityIssue(
            dimension=QualityDimension.COMPLETENESS,
            severity="high",
            description="High missing rate in critical column",
            affected_columns=["customer_id"],
            affected_rows=2000,
            recommendation="Investigate data collection",
            column="customer_id",
            category="completeness",
            issue_type="missing_values",
            affected_percentage=0.25
        )
        
        self.accuracy_issue = QualityIssue(
            dimension=QualityDimension.ACCURACY,
            severity="medium",
            description="Data type mismatches detected",
            affected_columns=["age"],
            affected_rows=500,
            recommendation="Fix data types",
            column="age",
            category="accuracy",
            issue_type="type_mismatch",
            affected_percentage=0.05
        )
        
        self.consistency_issue = QualityIssue(
            dimension=QualityDimension.CONSISTENCY,
            severity="high",
            description="Duplicate records found",
            affected_columns=["email"],
            affected_rows=1000,
            recommendation="Remove duplicates",
            column="email",
            category="consistency",
            issue_type="duplicate_records",
            affected_percentage=0.10
        )
        
        # Create sample quality report
        self.quality_report = QualityReport(
            dataset_id="test_dataset_001",
            overall_score=0.65,
            completeness_score=0.75,
            accuracy_score=0.80,
            consistency_score=0.60,
            validity_score=0.85,
            uniqueness_score=0.90,
            timeliness_score=0.70,
            issues=[self.completeness_issue, self.accuracy_issue, self.consistency_issue]
        )
        
        # Add AI-specific attributes for testing
        self.quality_report.feature_correlations = [
            ("feature_a", "feature_b", 0.98),
            ("feature_c", "feature_d", 0.96)
        ]
        self.quality_report.class_distribution = {
            "class_0": 9000,
            "class_1": 500,  # Severe imbalance
            "class_2": 500
        }
    
    def test_generate_recommendations_completeness(self):
        """Test that completeness recommendations are accurate and relevant."""
        recommendations = self.engine.generate_recommendations(self.quality_report)
        
        # Find completeness recommendations
        completeness_recs = [r for r in recommendations if r.category == "completeness"]
        
        assert len(completeness_recs) > 0, "Should generate completeness recommendations"
        
        # Check high-priority completeness recommendation
        high_priority_completeness = [r for r in completeness_recs 
                                    if r.priority == RecommendationPriority.HIGH]
        
        assert len(high_priority_completeness) > 0, "Should have high-priority completeness recommendations"
        
        rec = high_priority_completeness[0]
        assert "customer_id" in rec.title, "Should mention the affected column"
        assert rec.type == RecommendationType.DATA_COLLECTION, "Should suggest data collection for high missing rate"
        assert len(rec.action_items) >= 3, "Should provide multiple actionable items"
        assert "imputation" in rec.description.lower() or any("imputation" in item.lower() for item in rec.action_items)
    
    def test_generate_recommendations_accuracy(self):
        """Test that accuracy recommendations are relevant."""
        recommendations = self.engine.generate_recommendations(self.quality_report)
        
        accuracy_recs = [r for r in recommendations if r.category == "accuracy"]
        
        assert len(accuracy_recs) > 0, "Should generate accuracy recommendations"
        
        # Check for type mismatch recommendation
        type_mismatch_recs = [r for r in accuracy_recs if "type" in r.title.lower()]
        
        assert len(type_mismatch_recs) > 0, "Should generate type mismatch recommendations"
        
        rec = type_mismatch_recs[0]
        assert rec.type == RecommendationType.DATA_VALIDATION, "Should suggest validation for type issues"
        assert "age" in rec.affected_columns, "Should identify affected column"
        assert any("validation" in item.lower() for item in rec.action_items)
    
    def test_generate_recommendations_consistency(self):
        """Test that consistency recommendations are appropriate."""
        recommendations = self.engine.generate_recommendations(self.quality_report)
        
        consistency_recs = [r for r in recommendations if r.category == "consistency"]
        
        assert len(consistency_recs) > 0, "Should generate consistency recommendations"
        
        # Check for duplicate recommendation
        duplicate_recs = [r for r in consistency_recs if "duplicate" in r.title.lower()]
        
        assert len(duplicate_recs) > 0, "Should generate duplicate removal recommendations"
        
        rec = duplicate_recs[0]
        assert rec.type == RecommendationType.DATA_CLEANING, "Should suggest cleaning for duplicates"
        assert rec.priority == RecommendationPriority.HIGH, "Duplicates should be high priority"
        assert any("deduplication" in item.lower() for item in rec.action_items)
    
    def test_generate_ai_readiness_recommendations(self):
        """Test AI-specific recommendations for correlation and class imbalance."""
        recommendations = self.engine.generate_recommendations(self.quality_report)
        
        ai_recs = [r for r in recommendations if r.category == "ai_readiness"]
        
        assert len(ai_recs) >= 2, "Should generate AI-specific recommendations"
        
        # Check for correlation recommendation
        correlation_recs = [r for r in ai_recs if "correlated" in r.title.lower()]
        assert len(correlation_recs) > 0, "Should detect high correlation issues"
        
        corr_rec = correlation_recs[0]
        assert corr_rec.type == RecommendationType.FEATURE_ENGINEERING
        assert corr_rec.priority == RecommendationPriority.HIGH
        assert any("pca" in item.lower() or "dimensionality" in item.lower() 
                  for item in corr_rec.action_items)
        
        # Check for class imbalance recommendation
        imbalance_recs = [r for r in ai_recs if "imbalance" in r.title.lower()]
        assert len(imbalance_recs) > 0, "Should detect class imbalance"
        
        imbalance_rec = imbalance_recs[0]
        assert imbalance_rec.type == RecommendationType.DATA_BALANCING
        assert any("smote" in item.lower() or "sampling" in item.lower() 
                  for item in imbalance_rec.action_items)
    
    def test_recommendation_prioritization(self):
        """Test that recommendations are properly prioritized."""
        recommendations = self.engine.generate_recommendations(self.quality_report)
        
        # Check that high-priority recommendations come first
        high_priority_count = len([r for r in recommendations[:5] 
                                 if r.priority == RecommendationPriority.HIGH])
        
        assert high_priority_count >= 3, "Top recommendations should be high priority"
        
        # Check priority scores are assigned
        for rec in recommendations:
            assert hasattr(rec, 'priority_score'), "Should have priority score"
            assert rec.priority_score is not None, "Priority score should be calculated"
    
    def test_recommendation_deduplication(self):
        """Test that duplicate recommendations are removed."""
        # Create quality report with similar issues
        duplicate_issue = QualityIssue(
            dimension=QualityDimension.COMPLETENESS,
            severity="high",
            description="Another missing value issue",
            affected_columns=["customer_id"],
            affected_rows=1500,
            recommendation="Investigate data collection",
            column="customer_id",
            category="completeness",
            issue_type="missing_values",
            affected_percentage=0.20
        )
        
        self.quality_report.issues.append(duplicate_issue)
        
        recommendations = self.engine.generate_recommendations(self.quality_report)
        
        # Check that similar recommendations are deduplicated
        titles = [r.title for r in recommendations]
        unique_titles = set(titles)
        
        assert len(titles) == len(unique_titles), "Should not have duplicate recommendation titles"
    
    def test_improvement_roadmap_generation(self):
        """Test improvement roadmap generation."""
        recommendations = self.engine.generate_recommendations(self.quality_report)
        roadmap = self.engine.generate_improvement_roadmap(recommendations)
        
        # Check roadmap structure
        assert 'immediate_actions' in roadmap
        assert 'short_term_improvements' in roadmap
        assert 'long_term_enhancements' in roadmap
        assert 'estimated_timeline' in roadmap
        assert 'resource_requirements' in roadmap
        
        # Check that immediate actions are high priority
        immediate_actions = roadmap['immediate_actions']
        assert all(action.priority == RecommendationPriority.HIGH 
                  for action in immediate_actions)
        
        # Check resource estimation
        resources = roadmap['resource_requirements']
        assert 'total_recommendations' in resources
        assert 'effort_distribution' in resources
        assert 'estimated_person_weeks' in resources
        assert resources['estimated_person_weeks'] > 0
    
    def test_recommendation_relevance_scoring(self):
        """Test that recommendations are relevant to the specific issues."""
        recommendations = self.engine.generate_recommendations(self.quality_report)
        
        for rec in recommendations:
            # Check that recommendations address specific issues
            if rec.category == "completeness":
                assert any(col in rec.affected_columns or col in rec.title 
                          for col in ["customer_id"])
            elif rec.category == "accuracy":
                assert any(col in rec.affected_columns or col in rec.title 
                          for col in ["age"])
            elif rec.category == "consistency":
                assert "duplicate" in rec.title.lower() or "consistency" in rec.description.lower()
            
            # Check that action items are specific and actionable
            assert len(rec.action_items) >= 2, "Should provide multiple action items"
            assert all(len(item.strip()) > 10 for item in rec.action_items), "Action items should be descriptive"
            
            # Check impact assessment
            assert rec.estimated_impact, "Should have impact assessment"
            assert rec.implementation_effort in ["Low", "Medium", "High"], "Should have effort estimate"


class TestQualityReportGenerator:
    """Test the quality report generator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = QualityReportGenerator()
        
        # Create sample AI readiness score
        self.ai_readiness_score = AIReadinessScore(
            overall_score=0.72,
            data_quality_score=0.75,
            feature_quality_score=0.68,
            bias_score=0.80,
            compliance_score=0.65,
            scalability_score=0.75,
            dimensions={
                "data_quality": DimensionScore(dimension="data_quality", score=0.75, weight=0.3),
                "feature_quality": DimensionScore(dimension="feature_quality", score=0.68, weight=0.25)
            },
            improvement_areas=[
                ImprovementArea(
                    area="Feature Engineering",
                    current_score=0.68,
                    target_score=0.85,
                    priority="High",
                    estimated_effort="Medium"
                )
            ]
        )
        
        # Create sample quality report
        self.quality_report = QualityReport(
            dataset_id="test_dataset_001",
            overall_score=0.75,
            completeness_score=0.80,
            accuracy_score=0.85,
            consistency_score=0.70,
            validity_score=0.75,
            uniqueness_score=0.90,
            timeliness_score=0.65,
            issues=[
                QualityIssue(
                    dimension=QualityDimension.COMPLETENESS,
                    severity="medium",
                    description="Some missing values",
                    affected_columns=["optional_field"],
                    affected_rows=100,
                    recommendation="Consider imputation",
                    category="completeness",
                    issue_type="missing_values",
                    affected_percentage=0.05
                )
            ]
        )
    
    def test_generate_comprehensive_report_structure(self):
        """Test that comprehensive report has correct structure."""
        report = self.generator.generate_comprehensive_report(
            self.quality_report, 
            self.ai_readiness_score
        )
        
        # Check main structure
        assert 'report_metadata' in report
        assert 'sections' in report
        assert 'summary_metrics' in report
        assert 'recommendations_count' in report
        assert 'critical_issues_count' in report
        
        # Check metadata
        metadata = report['report_metadata']
        assert metadata['dataset_id'] == "test_dataset_001"
        assert 'generated_at' in metadata
        assert 'report_version' in metadata
        
        # Check required sections
        sections = report['sections']
        required_sections = [
            "Executive Summary",
            "Data Quality Overview", 
            "AI Readiness Assessment",
            "Detailed Issues Analysis",
            "Recommendations & Action Plan",
            "Data Profiling Results",
            "Compliance & Governance"
        ]
        
        for section in required_sections:
            assert section in sections, f"Missing required section: {section}"
    
    def test_executive_summary_accuracy(self):
        """Test executive summary accuracy and completeness."""
        report = self.generator.generate_comprehensive_report(
            self.quality_report, 
            self.ai_readiness_score
        )
        
        exec_summary = report['sections']['Executive Summary']
        
        # Check key metrics
        assert exec_summary['data_quality_score'] == 0.75
        assert exec_summary['ai_readiness_score'] == 0.72
        assert exec_summary['overall_assessment'] in ["Excellent", "Good", "Fair", "Poor"]
        assert 'status_color' in exec_summary
        
        # Check findings and actions
        assert 'key_findings' in exec_summary
        assert 'critical_actions_needed' in exec_summary
        assert 'total_issues' in exec_summary
        assert 'estimated_improvement_time' in exec_summary
        
        # Verify assessment logic
        combined_score = (0.75 + 0.72) / 2
        if combined_score >= 0.8:
            expected_assessment = "Excellent"
        elif combined_score >= 0.6:
            expected_assessment = "Good"
        else:
            expected_assessment = "Fair"
        
        assert exec_summary['overall_assessment'] == expected_assessment
    
    def test_quality_overview_section(self):
        """Test data quality overview section."""
        report = self.generator.generate_comprehensive_report(
            self.quality_report, 
            self.ai_readiness_score
        )
        
        quality_overview = report['sections']['Data Quality Overview']
        
        # Check overall score details
        assert 'overall_score' in quality_overview
        overall_score = quality_overview['overall_score']
        assert overall_score['value'] == 0.75
        assert overall_score['grade'] in ['A', 'B', 'C', 'D', 'F']
        assert 'interpretation' in overall_score
        
        # Check dimension scores
        assert 'dimension_scores' in quality_overview
        dimensions = quality_overview['dimension_scores']
        
        required_dimensions = ['completeness', 'accuracy', 'consistency', 'validity']
        for dim in required_dimensions:
            assert dim in dimensions
            assert 'score' in dimensions[dim]
            assert 'grade' in dimensions[dim]
            assert 'description' in dimensions[dim]
        
        # Verify score accuracy
        assert dimensions['completeness']['score'] == 0.8
        assert dimensions['accuracy']['score'] == 0.85
    
    def test_ai_readiness_section(self):
        """Test AI readiness assessment section."""
        report = self.generator.generate_comprehensive_report(
            self.quality_report, 
            self.ai_readiness_score
        )
        
        ai_section = report['sections']['AI Readiness Assessment']
        
        # Check overall readiness
        assert 'overall_readiness' in ai_section
        overall_readiness = ai_section['overall_readiness']
        assert overall_readiness['score'] == 0.72
        assert 'readiness_level' in overall_readiness
        
        # Check dimension analysis
        assert 'dimension_analysis' in ai_section
        dimensions = ai_section['dimension_analysis']
        
        required_ai_dimensions = [
            'data_quality', 'feature_quality', 'bias_assessment', 
            'compliance', 'scalability'
        ]
        for dim in required_ai_dimensions:
            assert dim in dimensions
            assert 'score' in dimensions[dim]
            assert 'impact' in dimensions[dim]
        
        # Check improvement areas
        assert 'improvement_areas' in ai_section
        improvement_areas = ai_section['improvement_areas']
        assert len(improvement_areas) > 0
        
        area = improvement_areas[0]
        assert 'area' in area
        assert 'current_score' in area
        assert 'target_score' in area
        assert 'priority' in area
    
    def test_recommendations_section(self):
        """Test recommendations section generation."""
        report = self.generator.generate_comprehensive_report(
            self.quality_report, 
            self.ai_readiness_score
        )
        
        rec_section = report['sections']['Recommendations & Action Plan']
        
        # Check roadmap
        assert 'improvement_roadmap' in rec_section
        roadmap = rec_section['improvement_roadmap']
        assert 'immediate_actions' in roadmap
        assert 'short_term_improvements' in roadmap
        assert 'long_term_enhancements' in roadmap
        
        # Check recommendations by priority
        assert 'recommendations_by_priority' in rec_section
        by_priority = rec_section['recommendations_by_priority']
        assert 'high' in by_priority
        assert 'medium' in by_priority
        assert 'low' in by_priority
        
        # Check implementation guidance
        assert 'implementation_guidance' in rec_section
        guidance = rec_section['implementation_guidance']
        assert 'getting_started' in guidance
        assert 'best_practices' in guidance
        assert 'common_pitfalls' in guidance
    
    def test_report_export_json(self):
        """Test JSON report export functionality."""
        report = self.generator.generate_comprehensive_report(
            self.quality_report, 
            self.ai_readiness_score
        )
        
        # Test JSON export
        with patch('builtins.open', create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            output_path = self.generator.export_report(report, format_type='json')
            
            # Verify file operations
            mock_open.assert_called_once()
            mock_file.write.assert_called()
            
            # Verify output path format
            assert output_path.endswith('.json')
            assert 'test_dataset_001' in output_path
    
    def test_report_export_html(self):
        """Test HTML report export functionality."""
        report = self.generator.generate_comprehensive_report(
            self.quality_report, 
            self.ai_readiness_score
        )
        
        # Test HTML export
        with patch('builtins.open', create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            output_path = self.generator.export_report(report, format_type='html')
            
            # Verify file operations
            mock_open.assert_called_once()
            mock_file.write.assert_called()
            
            # Verify output path format
            assert output_path.endswith('.html')
            assert 'test_dataset_001' in output_path
    
    def test_score_interpretation_accuracy(self):
        """Test score interpretation and grading accuracy."""
        # Test various score ranges
        test_scores = [0.95, 0.85, 0.75, 0.65, 0.45]
        expected_grades = ['A', 'B', 'C', 'D', 'F']
        
        for score, expected_grade in zip(test_scores, expected_grades):
            grade = self.generator._score_to_grade(score)
            assert grade == expected_grade, f"Score {score} should get grade {expected_grade}, got {grade}"
        
        # Test interpretations
        interpretations = [
            self.generator._interpret_score(0.95),
            self.generator._interpret_score(0.85),
            self.generator._interpret_score(0.75),
            self.generator._interpret_score(0.65),
            self.generator._interpret_score(0.45)
        ]
        
        for interpretation in interpretations:
            assert len(interpretation) > 10, "Interpretation should be descriptive"
            assert any(word in interpretation.lower() 
                      for word in ['excellent', 'good', 'acceptable', 'fair', 'poor'])
    
    def test_recommendation_formatting(self):
        """Test recommendation formatting for reports."""
        sample_rec = Recommendation(
            id="test_rec_001",
            type=RecommendationType.DATA_CLEANING,
            priority=RecommendationPriority.HIGH,
            title="Test Recommendation",
            description="Test description",
            action_items=["Action 1", "Action 2"],
            estimated_impact="High impact",
            implementation_effort="Medium",
            affected_columns=["col1", "col2"],
            category="test"
        )
        
        formatted = self.generator._format_recommendation(sample_rec)
        
        # Check all required fields are present
        required_fields = [
            'title', 'description', 'priority', 'type', 'action_items',
            'estimated_impact', 'implementation_effort', 'affected_columns', 'category'
        ]
        
        for field in required_fields:
            assert field in formatted, f"Missing field: {field}"
        
        # Check field values
        assert formatted['title'] == "Test Recommendation"
        assert formatted['priority'] == "HIGH"
        assert formatted['type'] == "data_cleaning"
        assert len(formatted['action_items']) == 2


class TestIntegrationReportingSystem:
    """Integration tests for the complete reporting system."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.recommendation_engine = RecommendationEngine()
        self.report_generator = QualityReportGenerator()
        
        # Create comprehensive test data
        self.quality_issues = [
            QualityIssue(
                dimension=QualityDimension.COMPLETENESS,
                severity="high",
                description="Critical missing data",
                affected_columns=["customer_id"],
                affected_rows=5000,
                recommendation="Urgent data collection needed",
                column="customer_id",
                category="completeness",
                issue_type="missing_values",
                affected_percentage=0.30
            ),
            QualityIssue(
                dimension=QualityDimension.ACCURACY,
                severity="medium",
                description="Format inconsistencies",
                affected_columns=["phone_number"],
                affected_rows=1200,
                recommendation="Standardize phone format",
                column="phone_number",
                category="accuracy",
                issue_type="format_inconsistency",
                affected_percentage=0.08
            )
        ]
        
        self.quality_report = QualityReport(
            dataset_id="integration_test_dataset",
            overall_score=0.58,  # Below threshold to trigger recommendations
            completeness_score=0.70,
            accuracy_score=0.75,
            consistency_score=0.45,
            validity_score=0.60,
            uniqueness_score=0.85,
            timeliness_score=0.55,
            issues=self.quality_issues
        )
        
        # Add AI-specific test data
        self.quality_report.feature_correlations = [("feat1", "feat2", 0.97)]
        self.quality_report.class_distribution = {"class_0": 8500, "class_1": 500}
        
        self.ai_readiness_score = AIReadinessScore(
            overall_score=0.62,
            data_quality_score=0.58,
            feature_quality_score=0.65,
            bias_score=0.70,
            compliance_score=0.55,
            scalability_score=0.68,
            improvement_areas=[
                ImprovementArea(
                    area="Data Quality",
                    current_score=0.58,
                    target_score=0.80,
                    priority="High",
                    estimated_effort="High"
                )
            ]
        )
    
    def test_end_to_end_reporting_workflow(self):
        """Test complete end-to-end reporting workflow."""
        # Generate recommendations
        recommendations = self.recommendation_engine.generate_recommendations(self.quality_report)
        
        assert len(recommendations) > 0, "Should generate recommendations for low-quality data"
        
        # Generate comprehensive report
        report = self.report_generator.generate_comprehensive_report(
            self.quality_report,
            self.ai_readiness_score,
            include_recommendations=True
        )
        
        # Verify report completeness
        assert report['recommendations_count'] == len(recommendations)
        assert report['critical_issues_count'] > 0
        
        # Verify recommendations are included in report
        rec_section = report['sections']['Recommendations & Action Plan']
        assert len(rec_section['recommendations_by_priority']['high']) > 0
        
        # Verify executive summary reflects poor quality
        exec_summary = report['sections']['Executive Summary']
        assert exec_summary['overall_assessment'] in ["Fair", "Poor"]
        assert len(exec_summary['key_findings']) > 0
        assert len(exec_summary['critical_actions_needed']) > 0
    
    def test_recommendation_report_consistency(self):
        """Test consistency between recommendations and report content."""
        recommendations = self.recommendation_engine.generate_recommendations(self.quality_report)
        report = self.report_generator.generate_comprehensive_report(
            self.quality_report,
            self.ai_readiness_score,
            include_recommendations=True
        )
        
        # Check that report recommendations match generated recommendations
        report_recs = []
        for priority in ['high', 'medium', 'low']:
            report_recs.extend(report['sections']['Recommendations & Action Plan']['recommendations_by_priority'][priority])
        
        assert len(report_recs) == len(recommendations), "Report should include all generated recommendations"
        
        # Check that issues mentioned in recommendations are reflected in issues analysis
        issues_analysis = report['sections']['Detailed Issues Analysis']
        critical_issues = issues_analysis['critical_issues']
        
        # Verify that high-severity issues have corresponding recommendations
        high_severity_columns = [issue['column'] for issue in critical_issues if issue['severity'] == 'high']
        high_priority_rec_columns = []
        
        for rec in recommendations:
            if rec.priority == RecommendationPriority.HIGH:
                high_priority_rec_columns.extend(rec.affected_columns)
        
        # Should have overlap between critical issues and high-priority recommendations
        overlap = set(high_severity_columns) & set(high_priority_rec_columns)
        assert len(overlap) > 0, "High-priority recommendations should address critical issues"
    
    def test_report_actionability(self):
        """Test that generated reports provide actionable insights."""
        report = self.report_generator.generate_comprehensive_report(
            self.quality_report,
            self.ai_readiness_score
        )
        
        # Check executive summary actionability
        exec_summary = report['sections']['Executive Summary']
        assert len(exec_summary['critical_actions_needed']) > 0, "Should provide critical actions"
        
        # Check recommendations actionability
        rec_section = report['sections']['Recommendations & Action Plan']
        
        # Verify implementation guidance
        guidance = rec_section['implementation_guidance']
        assert len(guidance['getting_started']) > 0, "Should provide getting started guidance"
        assert len(guidance['best_practices']) > 0, "Should provide best practices"
        
        # Verify roadmap has timeline
        roadmap = rec_section['improvement_roadmap']
        assert 'estimated_timeline' in roadmap, "Should provide timeline estimates"
        assert 'resource_requirements' in roadmap, "Should estimate resource needs"
        
        # Check that all recommendations have action items
        for priority in ['high', 'medium', 'low']:
            for rec in rec_section['recommendations_by_priority'][priority]:
                assert len(rec['action_items']) >= 2, "Each recommendation should have multiple action items"
                assert all(len(item.strip()) > 5 for item in rec['action_items']), "Action items should be descriptive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])