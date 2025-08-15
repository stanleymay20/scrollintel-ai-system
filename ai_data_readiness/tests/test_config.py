"""
Comprehensive test configuration for AI Data Readiness Platform testing framework.
"""

import os
from typing import Dict, List, Any


class TestConfig:
    """Centralized test configuration."""
    
    # Database Configuration
    TEST_DATABASE_URL = os.getenv('TEST_DATABASE_URL', 'sqlite:///:memory:')
    
    # Test Data Configuration
    DEFAULT_DATASET_SIZES = [1000, 5000, 10000]
    LARGE_DATASET_SIZES = [25000, 50000, 100000]  # For performance testing
    
    # Quality Assessment Thresholds
    QUALITY_THRESHOLDS = {
        'completeness_accuracy_threshold': 0.95,
        'outlier_detection_sensitivity': 0.8,
        'duplicate_detection_accuracy': 0.9,
        'consistency_assessment_accuracy': 0.85,
        'validity_assessment_accuracy': 0.9,
        'overall_quality_correlation': 0.8
    }
    
    # Bias Detection Thresholds
    BIAS_DETECTION_THRESHOLDS = {
        'demographic_parity_accuracy': 0.9,
        'equalized_odds_accuracy': 0.85,
        'individual_fairness_accuracy': 0.8,
        'intersectional_bias_detection': 0.85,
        'bias_amplification_detection': 0.9
    }
    
    # Performance Thresholds (per 1000 rows)
    PERFORMANCE_THRESHOLDS = {
        'max_ingestion_time_per_1k_rows': 1.0,  # seconds
        'max_quality_assessment_time_per_1k_rows': 2.0,
        'max_bias_analysis_time_per_1k_rows': 1.5,
        'max_feature_engineering_time_per_1k_rows': 3.0,
        'max_drift_monitoring_time_per_1k_rows': 1.0,
        'min_throughput_rows_per_second': 500
    }
    
    # Memory Usage Thresholds
    MEMORY_THRESHOLDS = {
        'max_memory_per_row_kb': 10,  # KB per row
        'max_memory_growth_factor': 1.5,  # Max growth factor between dataset sizes
        'max_peak_memory_mb': 2000  # MB
    }
    
    # Test Data Generation Configuration
    SYNTHETIC_DATA_CONFIG = {
        'feature_types': {
            'numerical_ratio': 0.4,
            'categorical_ratio': 0.3,
            'boolean_ratio': 0.2,
            'datetime_ratio': 0.1
        },
        'quality_issue_severities': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        'bias_strengths': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
        'outlier_strengths': [2.0, 3.0, 4.0, 5.0],
        'missing_value_patterns': ['random', 'clustered', 'periodic']
    }
    
    # Bias Test Scenarios
    BIAS_TEST_SCENARIOS = [
        {
            'name': 'gender_bias',
            'protected_attribute': 'gender',
            'bias_type': 'demographic_parity',
            'strengths': [0.1, 0.2, 0.3],
            'expected_detection_rate': 0.9
        },
        {
            'name': 'racial_bias',
            'protected_attribute': 'race',
            'bias_type': 'demographic_parity',
            'strengths': [0.15, 0.25, 0.35],
            'expected_detection_rate': 0.85
        },
        {
            'name': 'age_bias',
            'protected_attribute': 'age_group',
            'bias_type': 'demographic_parity',
            'strengths': [0.2, 0.3],
            'expected_detection_rate': 0.8
        },
        {
            'name': 'intersectional_bias',
            'protected_attributes': ['gender', 'race'],
            'bias_type': 'intersectional',
            'strengths': [0.2, 0.3],
            'expected_detection_rate': 0.85
        },
        {
            'name': 'proxy_bias',
            'protected_attribute': 'race',
            'proxy_attribute': 'zip_code',
            'bias_type': 'proxy',
            'strengths': [0.25, 0.35],
            'expected_detection_rate': 0.75
        }
    ]
    
    # Quality Issue Test Scenarios
    QUALITY_ISSUE_SCENARIOS = [
        {
            'issue_type': 'missing_values',
            'severities': [0.05, 0.1, 0.2, 0.3],
            'patterns': ['random', 'clustered', 'periodic'],
            'expected_detection_rate': 0.95
        },
        {
            'issue_type': 'outliers',
            'severities': [0.01, 0.02, 0.05, 0.1],
            'strengths': [2.0, 3.0, 4.0, 5.0],
            'expected_detection_rate': 0.8
        },
        {
            'issue_type': 'duplicates',
            'severities': [0.02, 0.05, 0.1, 0.15],
            'types': ['exact', 'near'],
            'expected_detection_rate': 0.9
        },
        {
            'issue_type': 'inconsistent_formats',
            'severities': [0.05, 0.1, 0.2],
            'format_types': ['case', 'spacing', 'punctuation'],
            'expected_detection_rate': 0.85
        },
        {
            'issue_type': 'invalid_values',
            'severities': [0.02, 0.05, 0.1],
            'violation_types': ['range', 'type', 'format'],
            'expected_detection_rate': 0.9
        }
    ]
    
    # Integration Test Scenarios
    INTEGRATION_TEST_SCENARIOS = [
        {
            'name': 'ml_model_development_workflow',
            'dataset_size': 5000,
            'quality_issues': ['missing_values', 'outliers'],
            'bias_issues': ['gender_bias'],
            'expected_ai_readiness_improvement': 0.15
        },
        {
            'name': 'production_monitoring_workflow',
            'dataset_size': 10000,
            'drift_scenarios': ['feature_drift', 'target_drift'],
            'expected_drift_detection_rate': 0.9
        },
        {
            'name': 'compliance_validation_workflow',
            'dataset_size': 3000,
            'pii_types': ['ssn', 'email', 'phone'],
            'regulations': ['GDPR', 'CCPA', 'HIPAA'],
            'expected_compliance_detection_rate': 0.95
        },
        {
            'name': 'multi_dataset_comparison_workflow',
            'dataset_count': 5,
            'quality_levels': ['high', 'medium', 'low'],
            'expected_ranking_accuracy': 0.9
        }
    ]
    
    # Test Output Configuration
    OUTPUT_CONFIG = {
        'base_directory': 'test_results',
        'generate_html_reports': True,
        'generate_json_reports': True,
        'save_test_datasets': False,  # Set to True for debugging
        'save_performance_metrics': True,
        'create_visualizations': True
    }
    
    # Logging Configuration
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'log_to_file': True,
        'log_file': 'test_results/test_execution.log'
    }
    
    # Parallel Testing Configuration
    PARALLEL_CONFIG = {
        'max_workers': 4,
        'enable_parallel_execution': True,
        'parallel_test_categories': [
            'unit_tests',
            'integration_tests',
            'performance_tests'
        ]
    }
    
    # Validation Rules
    VALIDATION_RULES = {
        'numerical_columns': {
            'age': {'min': 0, 'max': 120},
            'income': {'min': 0, 'max': 1000000},
            'score': {'min': 0, 'max': 100}
        },
        'categorical_columns': {
            'gender': {'allowed_values': ['M', 'F', 'Other']},
            'education': {'allowed_values': ['High School', 'Bachelor', 'Master', 'PhD']},
            'category': {'allowed_values': ['A', 'B', 'C', 'D', 'E']}
        },
        'format_rules': {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^\d{3}-\d{3}-\d{4}$',
            'ssn': r'^\d{3}-\d{2}-\d{4}$'
        }
    }
    
    @classmethod
    def get_test_config(cls, test_type: str = 'default') -> Dict[str, Any]:
        """Get configuration for specific test type."""
        base_config = {
            'database_url': cls.TEST_DATABASE_URL,
            'dataset_sizes': cls.DEFAULT_DATASET_SIZES,
            'quality_thresholds': cls.QUALITY_THRESHOLDS,
            'bias_thresholds': cls.BIAS_DETECTION_THRESHOLDS,
            'performance_thresholds': cls.PERFORMANCE_THRESHOLDS,
            'output_config': cls.OUTPUT_CONFIG,
            'logging_config': cls.LOGGING_CONFIG
        }
        
        if test_type == 'performance':
            base_config.update({
                'dataset_sizes': cls.LARGE_DATASET_SIZES,
                'memory_thresholds': cls.MEMORY_THRESHOLDS,
                'parallel_config': cls.PARALLEL_CONFIG
            })
        elif test_type == 'validation':
            base_config.update({
                'synthetic_data_config': cls.SYNTHETIC_DATA_CONFIG,
                'bias_scenarios': cls.BIAS_TEST_SCENARIOS,
                'quality_scenarios': cls.QUALITY_ISSUE_SCENARIOS,
                'validation_rules': cls.VALIDATION_RULES
            })
        elif test_type == 'integration':
            base_config.update({
                'integration_scenarios': cls.INTEGRATION_TEST_SCENARIOS,
                'parallel_config': cls.PARALLEL_CONFIG
            })
        
        return base_config
    
    @classmethod
    def get_quality_issue_config(cls, issue_type: str, severity: float) -> Dict[str, Any]:
        """Get configuration for specific quality issue type."""
        scenarios = {s['issue_type']: s for s in cls.QUALITY_ISSUE_SCENARIOS}
        
        if issue_type not in scenarios:
            raise ValueError(f"Unknown quality issue type: {issue_type}")
        
        scenario = scenarios[issue_type]
        
        return {
            'issue_type': issue_type,
            'severity': severity,
            'expected_detection_rate': scenario['expected_detection_rate'],
            'additional_params': {
                k: v for k, v in scenario.items() 
                if k not in ['issue_type', 'severities', 'expected_detection_rate']
            }
        }
    
    @classmethod
    def get_bias_scenario_config(cls, scenario_name: str, strength: float) -> Dict[str, Any]:
        """Get configuration for specific bias scenario."""
        scenarios = {s['name']: s for s in cls.BIAS_TEST_SCENARIOS}
        
        if scenario_name not in scenarios:
            raise ValueError(f"Unknown bias scenario: {scenario_name}")
        
        scenario = scenarios[scenario_name]
        
        return {
            'name': scenario_name,
            'protected_attribute': scenario.get('protected_attribute'),
            'protected_attributes': scenario.get('protected_attributes'),
            'bias_type': scenario['bias_type'],
            'strength': strength,
            'expected_detection_rate': scenario['expected_detection_rate'],
            'proxy_attribute': scenario.get('proxy_attribute')
        }
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> bool:
        """Validate test configuration."""
        required_keys = [
            'database_url', 'dataset_sizes', 'quality_thresholds',
            'performance_thresholds', 'output_config'
        ]
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        # Validate dataset sizes
        if not all(isinstance(size, int) and size > 0 for size in config['dataset_sizes']):
            raise ValueError("Dataset sizes must be positive integers")
        
        # Validate thresholds
        for threshold_dict in [config['quality_thresholds'], config['performance_thresholds']]:
            for key, value in threshold_dict.items():
                if not isinstance(value, (int, float)) or value <= 0:
                    raise ValueError(f"Threshold {key} must be a positive number")
        
        return True


# Global test configuration instance
TEST_CONFIG = TestConfig()