"""
Validation tests for quality assessment algorithms using synthetic data with known characteristics.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from ai_data_readiness.engines.quality_assessment_engine import QualityAssessmentEngine
from ai_data_readiness.tests.validation.test_synthetic_data_generation import (
    SyntheticDataGenerator, QualityIssueConfig, DataQualityIssueType
)


class TestQualityAlgorithmValidation:
    """Validate quality assessment algorithms against synthetic data with known characteristics."""
    
    @pytest.fixture
    def quality_engine(self, test_config):
        """Create quality assessment engine."""
        return QualityAssessmentEngine(test_config)
    
    @pytest.fixture
    def data_generator(self):
        """Create synthetic data generator."""
        return SyntheticDataGenerator(seed=42)
    
    def test_completeness_detection_accuracy(self, quality_engine, data_generator):
        """Test accuracy of completeness detection."""
        test_cases = [
            {'missing_rate': 0.0, 'expected_completeness': 1.0},
            {'missing_rate': 0.1, 'expected_completeness': 0.9},
            {'missing_rate': 0.25, 'expected_completeness': 0.75},
            {'missing_rate': 0.5, 'expected_completeness': 0.5}
        ]
        
        for case in test_cases:
            # Generate dataset with known missing rate
            df = data_generator.generate_clean_dataset(1000, 5)
            
            if case['missing_rate'] > 0:
                missing_config = QualityIssueConfig(
                    issue_type=DataQualityIssueType.MISSING_VALUES,
                    severity=case['missing_rate']
                )
                df = data_generator.introduce_missing_values(df, missing_config)
            
            # Mock dataset retrieval for quality engine
            dataset_id = f"completeness_test_{case['missing_rate']}"
            
            with pytest.mock.patch.object(quality_engine, '_get_dataset') as mock_get:
                mock_get.return_value = df
                
                result = quality_engine._assess_completeness(df)
                detected_completeness = result['completeness_score']
                
                # Allow 5% tolerance for randomness
                tolerance = 0.05
                assert abs(detected_completeness - case['expected_completeness']) <= tolerance, \
                    f"Expected completeness {case['expected_completeness']}, got {detected_completeness}"
    
    def test_outlier_detection_sensitivity(self, quality_engine, data_generator):
        """Test sensitivity of outlier detection."""
        base_df = data_generator.generate_clean_dataset(1000, 5)
        
        test_cases = [
            {'outlier_rate': 0.0, 'outlier_strength': 0.0, 'should_detect': False},
            {'outlier_rate': 0.01, 'outlier_strength': 2.0, 'should_detect': False},  # Mild outliers
            {'outlier_rate': 0.02, 'outlier_strength': 3.0, 'should_detect': True},   # Moderate outliers
            {'outlier_rate': 0.05, 'outlier_strength': 4.0, 'should_detect': True},   # Strong outliers
            {'outlier_rate': 0.1, 'outlier_strength': 5.0, 'should_detect': True}     # Very strong outliers
        ]
        
        for case in test_cases:
            df = base_df.copy()
            
            if case['outlier_rate'] > 0:
                outlier_config = QualityIssueConfig(
                    issue_type=DataQualityIssueType.OUTLIERS,
                    severity=case['outlier_rate'],
                    parameters={'strength': case['outlier_strength']}
                )
                df = data_generator.introduce_outliers(df, outlier_config)
            
            # Test outlier detection
            outlier_result = quality_engine._detect_outliers(df)
            
            if case['should_detect']:
                assert outlier_result['outliers_detected'] is True, \
                    f"Failed to detect outliers with rate {case['outlier_rate']} and strength {case['outlier_strength']}"
                assert len(outlier_result['outlier_indices']) > 0
            else:
                # For no outliers or very mild outliers, detection might be False
                if case['outlier_rate'] == 0.0:
                    assert outlier_result['outliers_detected'] is False
    
    def test_duplicate_detection_accuracy(self, quality_engine, data_generator):
        """Test accuracy of duplicate detection."""
        test_cases = [
            {'duplicate_rate': 0.0, 'duplicate_type': 'exact'},
            {'duplicate_rate': 0.05, 'duplicate_type': 'exact'},
            {'duplicate_rate': 0.1, 'duplicate_type': 'exact'},
            {'duplicate_rate': 0.05, 'duplicate_type': 'near'},
            {'duplicate_rate': 0.1, 'duplicate_type': 'near'}
        ]
        
        for case in test_cases:
            df = data_generator.generate_clean_dataset(1000, 5)
            original_length = len(df)
            
            if case['duplicate_rate'] > 0:
                duplicate_config = QualityIssueConfig(
                    issue_type=DataQualityIssueType.DUPLICATES,
                    severity=case['duplicate_rate'],
                    parameters={'type': case['duplicate_type']}
                )
                df = data_generator.introduce_duplicates(df, duplicate_config)
            
            # Test duplicate detection
            duplicate_result = quality_engine._detect_duplicates(df)
            
            if case['duplicate_rate'] > 0:
                assert duplicate_result['duplicates_detected'] is True, \
                    f"Failed to detect duplicates with rate {case['duplicate_rate']}"
                
                detected_duplicates = duplicate_result['duplicate_count']
                expected_duplicates = int(original_length * case['duplicate_rate'])
                
                # Allow some tolerance for near duplicates
                tolerance = expected_duplicates * 0.3 if case['duplicate_type'] == 'near' else expected_duplicates * 0.1
                assert abs(detected_duplicates - expected_duplicates) <= tolerance, \
                    f"Expected ~{expected_duplicates} duplicates, detected {detected_duplicates}"
            else:
                assert duplicate_result['duplicates_detected'] is False
    
    def test_consistency_assessment_accuracy(self, quality_engine, data_generator):
        """Test accuracy of consistency assessment."""
        # Create dataset with categorical columns
        feature_types = {
            'category_1': 'categorical',
            'category_2': 'categorical',
            'numerical_1': 'numerical',
            'numerical_2': 'numerical'
        }
        
        test_cases = [
            {'inconsistency_rate': 0.0, 'expected_consistency': 1.0},
            {'inconsistency_rate': 0.1, 'expected_consistency_range': (0.8, 0.95)},
            {'inconsistency_rate': 0.2, 'expected_consistency_range': (0.7, 0.9)},
            {'inconsistency_rate': 0.3, 'expected_consistency_range': (0.6, 0.85)}
        ]
        
        for case in test_cases:
            df = data_generator.generate_clean_dataset(1000, 4, feature_types)
            
            if case['inconsistency_rate'] > 0:
                inconsistency_config = QualityIssueConfig(
                    issue_type=DataQualityIssueType.INCONSISTENT_FORMATS,
                    severity=case['inconsistency_rate'],
                    affected_columns=['category_1', 'category_2']
                )
                df = data_generator.introduce_inconsistent_formats(df, inconsistency_config)
            
            # Test consistency assessment
            consistency_result = quality_engine._assess_consistency(df)
            detected_consistency = consistency_result['consistency_score']
            
            if 'expected_consistency' in case:
                tolerance = 0.05
                assert abs(detected_consistency - case['expected_consistency']) <= tolerance
            else:
                min_expected, max_expected = case['expected_consistency_range']
                assert min_expected <= detected_consistency <= max_expected, \
                    f"Consistency score {detected_consistency} not in expected range {case['expected_consistency_range']}"
    
    def test_validity_assessment_accuracy(self, quality_engine, data_generator):
        """Test accuracy of validity assessment."""
        test_cases = [
            {'invalid_rate': 0.0, 'expected_validity': 1.0},
            {'invalid_rate': 0.05, 'expected_validity_range': (0.9, 0.98)},
            {'invalid_rate': 0.1, 'expected_validity_range': (0.85, 0.95)},
            {'invalid_rate': 0.2, 'expected_validity_range': (0.75, 0.9)}
        ]
        
        for case in test_cases:
            df = data_generator.generate_clean_dataset(1000, 5)
            
            if case['invalid_rate'] > 0:
                invalid_config = QualityIssueConfig(
                    issue_type=DataQualityIssueType.INVALID_VALUES,
                    severity=case['invalid_rate']
                )
                df = data_generator.introduce_invalid_values(df, invalid_config)
            
            # Define validation rules
            validation_rules = {
                'numerical_0': {'min': -100, 'max': 200},
                'categorical_1': {'allowed_values': ['A', 'B', 'C', 'D', 'E']},
                'boolean_2': {'type': 'boolean'}
            }
            
            # Test validity assessment
            validity_result = quality_engine._assess_validity(df, validation_rules)
            detected_validity = validity_result['validity_score']
            
            if 'expected_validity' in case:
                tolerance = 0.05
                assert abs(detected_validity - case['expected_validity']) <= tolerance
            else:
                min_expected, max_expected = case['expected_validity_range']
                assert min_expected <= detected_validity <= max_expected, \
                    f"Validity score {detected_validity} not in expected range {case['expected_validity_range']}"
    
    def test_overall_quality_score_calculation(self, quality_engine, data_generator):
        """Test overall quality score calculation with known dimension scores."""
        # Test with perfect data
        clean_df = data_generator.generate_clean_dataset(1000, 5)
        
        with pytest.mock.patch.object(quality_engine, '_get_dataset') as mock_get:
            mock_get.return_value = clean_df
            
            quality_report = quality_engine.assess_quality("clean_dataset")
            
            # Clean data should have high quality scores
            assert quality_report['overall_score'] > 0.9
            assert quality_report['completeness_score'] > 0.95
            assert quality_report['accuracy_score'] > 0.9
            assert quality_report['consistency_score'] > 0.9
            assert quality_report['validity_score'] > 0.9
        
        # Test with problematic data
        quality_issues = [
            QualityIssueConfig(
                issue_type=DataQualityIssueType.MISSING_VALUES,
                severity=0.2  # 20% missing
            ),
            QualityIssueConfig(
                issue_type=DataQualityIssueType.OUTLIERS,
                severity=0.1,  # 10% outliers
                parameters={'strength': 4.0}
            ),
            QualityIssueConfig(
                issue_type=DataQualityIssueType.INVALID_VALUES,
                severity=0.05  # 5% invalid
            )
        ]
        
        problematic_df = data_generator.generate_dataset_with_issues(
            n_rows=1000,
            n_features=5,
            quality_issues=quality_issues
        )
        
        with pytest.mock.patch.object(quality_engine, '_get_dataset') as mock_get:
            mock_get.return_value = problematic_df
            
            quality_report = quality_engine.assess_quality("problematic_dataset")
            
            # Problematic data should have lower quality scores
            assert quality_report['overall_score'] < 0.8
            assert quality_report['completeness_score'] < 0.85  # Due to missing values
            
            # Should identify issues
            assert len(quality_report['issues']) > 0
    
    def test_quality_threshold_sensitivity(self, quality_engine, data_generator):
        """Test sensitivity to different quality thresholds."""
        # Create dataset with moderate quality issues
        quality_issues = [
            QualityIssueConfig(
                issue_type=DataQualityIssueType.MISSING_VALUES,
                severity=0.15
            )
        ]
        
        df = data_generator.generate_dataset_with_issues(
            n_rows=1000,
            n_features=5,
            quality_issues=quality_issues
        )
        
        # Test with different thresholds
        threshold_configs = [
            {'completeness_threshold': 0.9, 'should_flag': True},   # Strict
            {'completeness_threshold': 0.8, 'should_flag': False},  # Moderate
            {'completeness_threshold': 0.7, 'should_flag': False}   # Lenient
        ]
        
        for config in threshold_configs:
            # Update engine thresholds
            quality_engine.thresholds['completeness_threshold'] = config['completeness_threshold']
            
            with pytest.mock.patch.object(quality_engine, '_get_dataset') as mock_get:
                mock_get.return_value = df
                
                quality_report = quality_engine.assess_quality("threshold_test")
                
                # Check if completeness issues are flagged based on threshold
                completeness_issues = [
                    issue for issue in quality_report['issues'] 
                    if issue['dimension'] == 'completeness'
                ]
                
                if config['should_flag']:
                    assert len(completeness_issues) > 0, \
                        f"Should flag completeness issues with threshold {config['completeness_threshold']}"
                else:
                    # May or may not flag depending on actual completeness score
                    pass
    
    def test_ai_readiness_score_correlation(self, quality_engine, data_generator):
        """Test correlation between data quality and AI readiness scores."""
        datasets = []
        
        # Create datasets with varying quality levels
        quality_levels = [
            {'name': 'high_quality', 'issues': []},
            {'name': 'medium_quality', 'issues': [
                QualityIssueConfig(DataQualityIssueType.MISSING_VALUES, 0.05),
                QualityIssueConfig(DataQualityIssueType.OUTLIERS, 0.02, parameters={'strength': 3.0})
            ]},
            {'name': 'low_quality', 'issues': [
                QualityIssueConfig(DataQualityIssueType.MISSING_VALUES, 0.2),
                QualityIssueConfig(DataQualityIssueType.OUTLIERS, 0.1, parameters={'strength': 4.0}),
                QualityIssueConfig(DataQualityIssueType.INVALID_VALUES, 0.05),
                QualityIssueConfig(DataQualityIssueType.DUPLICATES, 0.03, parameters={'type': 'exact'})
            ]}
        ]
        
        results = {}
        
        for level in quality_levels:
            if level['issues']:
                df = data_generator.generate_dataset_with_issues(
                    n_rows=1000,
                    n_features=8,
                    quality_issues=level['issues']
                )
            else:
                df = data_generator.generate_clean_dataset(1000, 8)
            
            with pytest.mock.patch.object(quality_engine, '_get_dataset') as mock_get:
                mock_get.return_value = df
                
                quality_report = quality_engine.assess_quality(f"{level['name']}_dataset")
                ai_readiness = quality_engine.calculate_ai_readiness_score(f"{level['name']}_dataset")
                
                results[level['name']] = {
                    'quality_score': quality_report['overall_score'],
                    'ai_readiness_score': ai_readiness['overall_score']
                }
        
        # Verify quality ordering
        assert results['high_quality']['quality_score'] > results['medium_quality']['quality_score']
        assert results['medium_quality']['quality_score'] > results['low_quality']['quality_score']
        
        # Verify AI readiness ordering
        assert results['high_quality']['ai_readiness_score'] > results['medium_quality']['ai_readiness_score']
        assert results['medium_quality']['ai_readiness_score'] > results['low_quality']['ai_readiness_score']
        
        # Verify correlation between quality and AI readiness
        for level_name, scores in results.items():
            # AI readiness should be correlated with quality (within reasonable bounds)
            score_diff = abs(scores['quality_score'] - scores['ai_readiness_score'])
            assert score_diff < 0.3, f"Quality and AI readiness scores too different for {level_name}: {score_diff}"
    
    def test_recommendation_accuracy(self, quality_engine, data_generator):
        """Test accuracy of improvement recommendations."""
        # Create dataset with specific known issues
        quality_issues = [
            QualityIssueConfig(
                issue_type=DataQualityIssueType.MISSING_VALUES,
                severity=0.25,
                affected_columns=['numerical_0', 'categorical_1']
            ),
            QualityIssueConfig(
                issue_type=DataQualityIssueType.OUTLIERS,
                severity=0.1,
                affected_columns=['numerical_0'],
                parameters={'strength': 4.0}
            ),
            QualityIssueConfig(
                issue_type=DataQualityIssueType.INCONSISTENT_FORMATS,
                severity=0.15,
                affected_columns=['categorical_1']
            )
        ]
        
        df = data_generator.generate_dataset_with_issues(
            n_rows=1000,
            n_features=5,
            quality_issues=quality_issues
        )
        
        with pytest.mock.patch.object(quality_engine, '_get_dataset') as mock_get:
            mock_get.return_value = df
            
            quality_report = quality_engine.assess_quality("recommendation_test")
            recommendations = quality_engine.recommend_improvements(quality_report)
            
            # Should recommend addressing missing values
            missing_recs = [
                rec for rec in recommendations['recommendations']
                if 'missing' in rec['description'].lower()
            ]
            assert len(missing_recs) > 0, "Should recommend handling missing values"
            
            # Should recommend addressing outliers
            outlier_recs = [
                rec for rec in recommendations['recommendations']
                if 'outlier' in rec['description'].lower()
            ]
            assert len(outlier_recs) > 0, "Should recommend handling outliers"
            
            # Should recommend addressing format inconsistencies
            format_recs = [
                rec for rec in recommendations['recommendations']
                if 'format' in rec['description'].lower() or 'consistency' in rec['description'].lower()
            ]
            assert len(format_recs) > 0, "Should recommend handling format inconsistencies"
            
            # High-severity issues should have high-priority recommendations
            high_priority_recs = [
                rec for rec in recommendations['recommendations']
                if rec['priority'] == 'high'
            ]
            assert len(high_priority_recs) > 0, "Should have high-priority recommendations for severe issues"
    
    def test_edge_cases_handling(self, quality_engine, data_generator):
        """Test handling of edge cases in quality assessment."""
        edge_cases = [
            {
                'name': 'single_row',
                'df': pd.DataFrame({'col1': [1], 'col2': ['A'], 'target': [0]})
            },
            {
                'name': 'single_column',
                'df': pd.DataFrame({'target': [0, 1, 0, 1, 1]})
            },
            {
                'name': 'all_missing',
                'df': pd.DataFrame({
                    'col1': [None, None, None],
                    'col2': [None, None, None],
                    'target': [0, 1, 0]
                })
            },
            {
                'name': 'all_identical',
                'df': pd.DataFrame({
                    'col1': [1, 1, 1, 1, 1],
                    'col2': ['A', 'A', 'A', 'A', 'A'],
                    'target': [0, 0, 0, 0, 0]
                })
            }
        ]
        
        for case in edge_cases:
            with pytest.mock.patch.object(quality_engine, '_get_dataset') as mock_get:
                mock_get.return_value = case['df']
                
                try:
                    quality_report = quality_engine.assess_quality(f"edge_case_{case['name']}")
                    
                    # Should handle edge cases gracefully
                    assert 'overall_score' in quality_report
                    assert 0 <= quality_report['overall_score'] <= 1
                    
                    # Specific checks for edge cases
                    if case['name'] == 'all_missing':
                        assert quality_report['completeness_score'] == 0.0
                    elif case['name'] == 'single_row':
                        # Should handle single row without errors
                        assert quality_report is not None
                    
                except Exception as e:
                    pytest.fail(f"Quality assessment failed for edge case '{case['name']}': {str(e)}")
    
    def test_performance_with_known_complexity(self, quality_engine, data_generator, performance_timer):
        """Test performance scaling with datasets of known complexity."""
        complexity_levels = [
            {'rows': 1000, 'features': 5, 'max_time': 5.0},
            {'rows': 5000, 'features': 10, 'max_time': 15.0},
            {'rows': 10000, 'features': 15, 'max_time': 30.0}
        ]
        
        for level in complexity_levels:
            df = data_generator.generate_clean_dataset(level['rows'], level['features'])
            
            with pytest.mock.patch.object(quality_engine, '_get_dataset') as mock_get:
                mock_get.return_value = df
                
                with performance_timer() as timer:
                    quality_report = quality_engine.assess_quality(f"performance_test_{level['rows']}")
                
                assert timer.duration <= level['max_time'], \
                    f"Quality assessment took {timer.duration:.2f}s for {level['rows']} rows, expected <= {level['max_time']}s"
                
                assert quality_report is not None
                assert 'overall_score' in quality_report