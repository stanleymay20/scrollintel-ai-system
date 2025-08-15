"""
Tests for AI Readiness Reporting Engine

This module contains comprehensive tests for the AI readiness reporting functionality.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from ai_data_readiness.engines.ai_readiness_reporting_engine import (
    AIReadinessReportingEngine,
    IndustryStandard,
    ReportType,
    ImprovementAction,
    AIReadinessReport
)
from ai_data_readiness.models.base_models import (
    AIReadinessScore,
    QualityReport,
    BiasReport,
    QualityIssue,
    DimensionScore
)


class TestAIReadinessReportingEngine:
    """Test cases for AI Readiness Reporting Engine"""
    
    @pytest.fixture
    def reporting_engine(self):
        """Create reporting engine instance for testing"""
        return AIReadinessReportingEngine()
    
    @pytest.fixture
    def sample_ai_readiness_score(self):
        """Create sample AI readiness score for testing"""
        return AIReadinessScore(
            overall_score=0.75,
            data_quality_score=0.80,
            feature_quality_score=0.70,
            bias_score=0.75,
            compliance_score=0.85,
            scalability_score=0.70,
            dimensions={
                "data_quality": DimensionScore(
                    area="data_quality",
                    score=0.80,
                    issues=["Missing values in target column"],
                    recommendations=["Implement data validation"]
                )
            },
            improvement_areas=[]
        )
    
    @pytest.fixture
    def sample_quality_report(self):
        """Create sample quality report for testing"""
        return QualityReport(
            dataset_id="test_dataset",
            overall_score=0.80,
            completeness_score=0.85,
            accuracy_score=0.80,
            consistency_score=0.75,
            validity_score=0.80,
            issues=[
                QualityIssue(
                    issue_type="completeness",
                    severity="medium",
                    description="Missing values detected",
                    affected_columns=["target"],
                    impact_score=0.15
                )
            ],
            recommendations=[],
            generated_at=datetime.utcnow()
        )
    
    @pytest.fixture
    def sample_bias_report(self):
        """Create sample bias report for testing"""
        return BiasReport(
            dataset_id="test_dataset",
            protected_attributes=["gender", "age_group"],
            bias_metrics={"demographic_parity": 0.75},
            fairness_violations=[],
            mitigation_strategies=[],
            generated_at=datetime.utcnow()
        )
    
    def test_engine_initialization(self, reporting_engine):
        """Test that the reporting engine initializes correctly"""
        assert reporting_engine is not None
        assert len(reporting_engine.industry_benchmarks) > 0
        assert IndustryStandard.FINANCIAL_SERVICES in reporting_engine.industry_benchmarks
        assert IndustryStandard.HEALTHCARE in reporting_engine.industry_benchmarks
    
    def test_industry_benchmarks_loaded(self, reporting_engine):
        """Test that industry benchmarks are properly loaded"""
        benchmarks = reporting_engine.industry_benchmarks
        
        # Check that all industry standards have benchmarks
        for industry in IndustryStandard:
            assert industry in benchmarks
            benchmark = benchmarks[industry]
            assert 0.0 <= benchmark.data_quality_threshold <= 1.0
            assert 0.0 <= benchmark.feature_quality_threshold <= 1.0
            assert 0.0 <= benchmark.bias_score_threshold <= 1.0
            assert 0.0 <= benchmark.compliance_score_threshold <= 1.0
            assert 0.0 <= benchmark.overall_readiness_threshold <= 1.0
            assert benchmark.typical_improvement_timeline > 0
    
    def test_generate_comprehensive_report(
        self,
        reporting_engine,
        sample_ai_readiness_score,
        sample_quality_report,
        sample_bias_report
    ):
        """Test comprehensive report generation"""
        report = reporting_engine.generate_comprehensive_report(
            dataset_id="test_dataset",
            ai_readiness_score=sample_ai_readiness_score,
            quality_report=sample_quality_report,
            bias_report=sample_bias_report,
            industry=IndustryStandard.FINANCIAL_SERVICES,
            report_type=ReportType.DETAILED_TECHNICAL
        )
        
        assert isinstance(report, AIReadinessReport)
        assert report.dataset_id == "test_dataset"
        assert report.report_type == ReportType.DETAILED_TECHNICAL
        assert report.overall_score == 0.75
        assert len(report.dimension_scores) == 5
        assert report.industry_benchmark.industry == IndustryStandard.FINANCIAL_SERVICES
        assert len(report.benchmark_comparison) > 0
        assert isinstance(report.improvement_actions, list)
        assert len(report.executive_summary) > 0
        assert isinstance(report.detailed_findings, dict)
        assert isinstance(report.compliance_status, dict)
        assert isinstance(report.risk_assessment, dict)
        assert isinstance(report.recommendations, list)
        assert report.estimated_improvement_timeline >= 0
    
    def test_benchmark_comparison_calculation(
        self,
        reporting_engine,
        sample_ai_readiness_score
    ):
        """Test benchmark comparison calculation"""
        benchmark = reporting_engine.industry_benchmarks[IndustryStandard.FINANCIAL_SERVICES]
        comparison = reporting_engine._calculate_benchmark_comparison(
            sample_ai_readiness_score,
            benchmark
        )
        
        assert "data_quality_gap" in comparison
        assert "feature_quality_gap" in comparison
        assert "bias_score_gap" in comparison
        assert "compliance_gap" in comparison
        assert "overall_gap" in comparison
        
        # Check that gaps are calculated correctly
        expected_data_quality_gap = sample_ai_readiness_score.data_quality_score - benchmark.data_quality_threshold
        assert comparison["data_quality_gap"] == expected_data_quality_gap
    
    def test_improvement_actions_generation(
        self,
        reporting_engine,
        sample_ai_readiness_score,
        sample_quality_report,
        sample_bias_report
    ):
        """Test improvement actions generation"""
        benchmark = reporting_engine.industry_benchmarks[IndustryStandard.FINANCIAL_SERVICES]
        actions = reporting_engine._generate_improvement_actions(
            sample_ai_readiness_score,
            sample_quality_report,
            sample_bias_report,
            benchmark
        )
        
        assert isinstance(actions, list)
        
        # Check that actions are properly structured
        for action in actions:
            assert isinstance(action, ImprovementAction)
            assert action.priority in ["high", "medium", "low"]
            assert len(action.title) > 0
            assert len(action.description) > 0
            assert action.estimated_timeline > 0
            assert 0.0 <= action.expected_impact <= 1.0
            assert isinstance(action.dependencies, list)
            assert isinstance(action.resources_required, list)
        
        # Check that actions are sorted by priority and impact
        if len(actions) > 1:
            priority_order = {"high": 3, "medium": 2, "low": 1}
            for i in range(len(actions) - 1):
                current_priority = priority_order[actions[i].priority]
                next_priority = priority_order[actions[i + 1].priority]
                assert current_priority >= next_priority
    
    def test_executive_summary_generation(
        self,
        reporting_engine,
        sample_ai_readiness_score
    ):
        """Test executive summary generation"""
        benchmark_comparison = {
            "overall_gap": -0.17,
            "data_quality_gap": -0.15,
            "feature_quality_gap": -0.20,
            "bias_score_gap": -0.10,
            "compliance_gap": -0.13
        }
        
        improvement_actions = [
            ImprovementAction(
                priority="high",
                category="data_quality",
                title="Test Action",
                description="Test description",
                estimated_effort="medium",
                estimated_timeline=14,
                expected_impact=0.08,
                dependencies=[],
                resources_required=["data_engineer"]
            )
        ]
        
        summary = reporting_engine._generate_executive_summary(
            sample_ai_readiness_score,
            benchmark_comparison,
            improvement_actions
        )
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "EXECUTIVE SUMMARY" in summary
        assert "Overall Readiness Level" in summary
        assert "Current AI Readiness Score" in summary
        assert "Industry Benchmark Gap" in summary
        assert "Key Findings" in summary
        assert "Improvement Plan" in summary
        assert "Recommendation" in summary
    
    def test_compliance_status_assessment(
        self,
        reporting_engine,
        sample_ai_readiness_score,
        sample_quality_report,
        sample_bias_report
    ):
        """Test compliance status assessment"""
        compliance_status = reporting_engine._assess_compliance_status(
            sample_ai_readiness_score,
            sample_quality_report,
            sample_bias_report
        )
        
        assert isinstance(compliance_status, dict)
        assert "gdpr_compliant" in compliance_status
        assert "ccpa_compliant" in compliance_status
        assert "sox_compliant" in compliance_status
        assert "fair_lending_compliant" in compliance_status
        assert "model_governance_ready" in compliance_status
        
        # Check that all values are boolean
        for key, value in compliance_status.items():
            assert isinstance(value, bool)
    
    def test_risk_assessment_generation(
        self,
        reporting_engine,
        sample_ai_readiness_score,
        sample_quality_report,
        sample_bias_report
    ):
        """Test risk assessment generation"""
        risk_assessment = reporting_engine._generate_risk_assessment(
            sample_ai_readiness_score,
            sample_quality_report,
            sample_bias_report
        )
        
        assert isinstance(risk_assessment, dict)
        assert "data_quality_risk" in risk_assessment
        assert "bias_risk" in risk_assessment
        assert "compliance_risk" in risk_assessment
        
        # Check that all risk levels are valid
        valid_risk_levels = ["Low", "Medium", "High"]
        for key, value in risk_assessment.items():
            assert any(level in value for level in valid_risk_levels)
    
    def test_benchmark_comparison_report(
        self,
        reporting_engine,
        sample_ai_readiness_score
    ):
        """Test benchmark comparison across multiple industries"""
        industries = [IndustryStandard.FINANCIAL_SERVICES, IndustryStandard.HEALTHCARE, IndustryStandard.RETAIL]
        
        comparisons = reporting_engine.generate_benchmark_comparison_report(
            dataset_id="test_dataset",
            ai_readiness_score=sample_ai_readiness_score,
            industries=industries
        )
        
        assert isinstance(comparisons, dict)
        assert len(comparisons) == len(industries)
        
        for industry in industries:
            assert industry.value in comparisons
            comparison = comparisons[industry.value]
            assert "data_quality_gap" in comparison
            assert "feature_quality_gap" in comparison
            assert "bias_score_gap" in comparison
            assert "compliance_gap" in comparison
            assert "overall_gap" in comparison
    
    def test_report_export_json(self, reporting_engine, sample_ai_readiness_score, sample_quality_report):
        """Test JSON export functionality"""
        report = reporting_engine.generate_comprehensive_report(
            dataset_id="test_dataset",
            ai_readiness_score=sample_ai_readiness_score,
            quality_report=sample_quality_report,
            industry=IndustryStandard.GENERAL,
            report_type=ReportType.DETAILED_TECHNICAL
        )
        
        json_export = reporting_engine.export_report_to_json(report)
        
        assert isinstance(json_export, str)
        assert len(json_export) > 0
        
        # Verify it's valid JSON by parsing it
        import json
        parsed = json.loads(json_export)
        assert isinstance(parsed, dict)
        assert "dataset_id" in parsed
        assert "overall_score" in parsed
    
    def test_report_export_html(self, reporting_engine, sample_ai_readiness_score, sample_quality_report):
        """Test HTML export functionality"""
        report = reporting_engine.generate_comprehensive_report(
            dataset_id="test_dataset",
            ai_readiness_score=sample_ai_readiness_score,
            quality_report=sample_quality_report,
            industry=IndustryStandard.GENERAL,
            report_type=ReportType.DETAILED_TECHNICAL
        )
        
        html_export = reporting_engine.export_report_to_html(report)
        
        assert isinstance(html_export, str)
        assert len(html_export) > 0
        assert "<!DOCTYPE html>" in html_export
        assert "<html>" in html_export
        assert "AI Data Readiness Report" in html_export
        assert report.dataset_id in html_export
    
    def test_improvement_timeline_estimation(self, reporting_engine):
        """Test improvement timeline estimation"""
        improvement_actions = [
            ImprovementAction(
                priority="high",
                category="data_quality",
                title="High Priority Action 1",
                description="Test description",
                estimated_effort="medium",
                estimated_timeline=14,
                expected_impact=0.08,
                dependencies=[],
                resources_required=["data_engineer"]
            ),
            ImprovementAction(
                priority="high",
                category="bias_mitigation",
                title="High Priority Action 2",
                description="Test description",
                estimated_effort="high",
                estimated_timeline=21,
                expected_impact=0.12,
                dependencies=[],
                resources_required=["ml_engineer"]
            ),
            ImprovementAction(
                priority="medium",
                category="feature_engineering",
                title="Medium Priority Action",
                description="Test description",
                estimated_effort="medium",
                estimated_timeline=10,
                expected_impact=0.06,
                dependencies=[],
                resources_required=["data_scientist"]
            )
        ]
        
        timeline = reporting_engine._estimate_improvement_timeline(improvement_actions)
        
        assert isinstance(timeline, int)
        assert timeline > 0
        # Should be max of high priority (21) + half of medium priority (5) = 26
        assert timeline == 26
    
    def test_error_handling(self, reporting_engine):
        """Test error handling in report generation"""
        with pytest.raises(Exception):
            # This should raise an exception due to invalid input
            reporting_engine.generate_comprehensive_report(
                dataset_id="",  # Invalid empty dataset ID
                ai_readiness_score=None,  # Invalid None score
                quality_report=None,  # Invalid None report
                industry=IndustryStandard.GENERAL,
                report_type=ReportType.DETAILED_TECHNICAL
            )


if __name__ == "__main__":
    pytest.main([__file__])