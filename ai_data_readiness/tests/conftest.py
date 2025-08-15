"""
Pytest configuration and fixtures for AI Data Readiness Platform tests.
"""

import pytest
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ai_data_readiness.core.config import Config
from ai_data_readiness.models.database import Base
from ai_data_readiness.core.data_ingestion_service import DataIngestionService
from ai_data_readiness.engines.quality_assessment_engine import QualityAssessmentEngine
from ai_data_readiness.engines.bias_analysis_engine import BiasAnalysisEngine
from ai_data_readiness.engines.feature_engineering_engine import FeatureEngineeringEngine


@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture."""
    config = Config()
    config.DATABASE_URL = "sqlite:///:memory:"
    config.TESTING = True
    config.LOG_LEVEL = "DEBUG"
    return config


@pytest.fixture(scope="session")
def test_engine(test_config):
    """Test database engine fixture."""
    engine = create_engine(test_config.DATABASE_URL, echo=False)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def test_session(test_engine):
    """Test database session fixture."""
    Session = sessionmaker(bind=test_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def temp_directory():
    """Temporary directory fixture for file operations."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing."""
    return pd.DataFrame({
        'id': range(1, 101),
        'name': [f'User_{i}' for i in range(1, 101)],
        'age': np.random.randint(18, 80, 100),
        'income': np.random.normal(50000, 15000, 100),
        'gender': np.random.choice(['M', 'F'], 100),
        'score': np.random.uniform(0, 100, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'is_active': np.random.choice([True, False], 100)
    })


@pytest.fixture
def sample_biased_data():
    """Sample biased dataset for bias testing."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create biased data where gender affects income
    gender = np.random.choice(['M', 'F'], n_samples, p=[0.6, 0.4])
    age = np.random.randint(22, 65, n_samples)
    
    # Introduce bias: males tend to have higher income
    income = np.where(
        gender == 'M',
        np.random.normal(60000, 15000, n_samples),
        np.random.normal(45000, 12000, n_samples)
    )
    
    # Target variable also biased
    approved = np.where(
        (gender == 'M') & (income > 50000),
        np.random.choice([True, False], n_samples, p=[0.8, 0.2]),
        np.random.choice([True, False], n_samples, p=[0.4, 0.6])
    )
    
    return pd.DataFrame({
        'gender': gender,
        'age': age,
        'income': income,
        'approved': approved
    })


@pytest.fixture
def sample_time_series_data():
    """Sample time series data for temporal feature testing."""
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    return pd.DataFrame({
        'date': dates,
        'value': np.random.normal(100, 15, 365) + np.sin(np.arange(365) * 2 * np.pi / 365) * 10,
        'category': np.random.choice(['A', 'B', 'C'], 365),
        'trend': np.arange(365) * 0.1 + np.random.normal(0, 5, 365)
    })


@pytest.fixture
def sample_missing_data():
    """Sample data with missing values for quality testing."""
    data = pd.DataFrame({
        'complete_col': range(100),
        'partial_missing': [i if i % 5 != 0 else None for i in range(100)],
        'mostly_missing': [i if i % 10 == 0 else None for i in range(100)],
        'all_missing': [None] * 100,
        'categorical': np.random.choice(['A', 'B', 'C', None], 100, p=[0.3, 0.3, 0.3, 0.1])
    })
    return data


@pytest.fixture
def mock_data_ingestion_service():
    """Mock data ingestion service for testing."""
    service = Mock(spec=DataIngestionService)
    service.ingest_batch_data.return_value = {
        'dataset_id': 'test_dataset_123',
        'rows': 1000,
        'columns': 10,
        'schema': {'col1': 'int64', 'col2': 'object'}
    }
    return service


@pytest.fixture
def mock_quality_engine():
    """Mock quality assessment engine for testing."""
    engine = Mock(spec=QualityAssessmentEngine)
    engine.assess_quality.return_value = {
        'overall_score': 0.85,
        'completeness_score': 0.90,
        'accuracy_score': 0.80,
        'consistency_score': 0.85,
        'validity_score': 0.85
    }
    return engine


@pytest.fixture
def mock_bias_engine():
    """Mock bias analysis engine for testing."""
    engine = Mock(spec=BiasAnalysisEngine)
    engine.detect_bias.return_value = {
        'bias_detected': True,
        'protected_attributes': ['gender', 'age'],
        'bias_score': 0.3,
        'fairness_metrics': {
            'demographic_parity': 0.7,
            'equalized_odds': 0.6
        }
    }
    return engine


@pytest.fixture
def mock_feature_engine():
    """Mock feature engineering engine for testing."""
    engine = Mock(spec=FeatureEngineeringEngine)
    engine.recommend_features.return_value = {
        'recommended_features': ['feature1_encoded', 'feature2_scaled'],
        'transformations': ['one_hot_encoding', 'standard_scaling'],
        'feature_importance': {'feature1': 0.8, 'feature2': 0.6}
    }
    return engine


@pytest.fixture
def performance_test_data():
    """Large dataset for performance testing."""
    np.random.seed(42)
    n_samples = 10000
    
    return pd.DataFrame({
        'id': range(n_samples),
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.exponential(2, n_samples),
        'feature3': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'feature4': np.random.uniform(0, 100, n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })


@pytest.fixture
def scalability_test_config():
    """Configuration for scalability testing."""
    return {
        'small_dataset_size': 1000,
        'medium_dataset_size': 10000,
        'large_dataset_size': 100000,
        'performance_thresholds': {
            'ingestion_time_per_row': 0.001,  # seconds
            'quality_assessment_time': 5.0,   # seconds
            'bias_analysis_time': 10.0,       # seconds
            'feature_engineering_time': 15.0  # seconds
        }
    }


# Test data generators
def generate_synthetic_dataset(n_rows=1000, n_features=10, missing_rate=0.1, bias_factor=0.0):
    """Generate synthetic dataset with configurable characteristics."""
    np.random.seed(42)
    
    data = {}
    for i in range(n_features):
        if i % 3 == 0:  # Numerical features
            data[f'num_feature_{i}'] = np.random.normal(0, 1, n_rows)
        elif i % 3 == 1:  # Categorical features
            data[f'cat_feature_{i}'] = np.random.choice(['A', 'B', 'C', 'D'], n_rows)
        else:  # Boolean features
            data[f'bool_feature_{i}'] = np.random.choice([True, False], n_rows)
    
    # Add target variable with optional bias
    if bias_factor > 0:
        # Introduce bias based on first categorical feature
        bias_mask = data['cat_feature_1'] == 'A'
        target_prob = np.where(bias_mask, 0.8, 0.2)
        data['target'] = np.random.binomial(1, target_prob)
    else:
        data['target'] = np.random.choice([0, 1], n_rows)
    
    df = pd.DataFrame(data)
    
    # Introduce missing values
    if missing_rate > 0:
        for col in df.columns:
            if col != 'target':  # Don't add missing values to target
                missing_mask = np.random.random(n_rows) < missing_rate
                df.loc[missing_mask, col] = None
    
    return df


# Assertion helpers
def assert_quality_report_structure(report):
    """Assert that a quality report has the expected structure."""
    required_fields = [
        'overall_score', 'completeness_score', 'accuracy_score',
        'consistency_score', 'validity_score'
    ]
    for field in required_fields:
        assert field in report, f"Missing required field: {field}"
        assert 0 <= report[field] <= 1, f"Score {field} should be between 0 and 1"


def assert_bias_report_structure(report):
    """Assert that a bias report has the expected structure."""
    required_fields = ['bias_detected', 'protected_attributes', 'bias_score']
    for field in required_fields:
        assert field in report, f"Missing required field: {field}"
    
    assert isinstance(report['bias_detected'], bool)
    assert isinstance(report['protected_attributes'], list)
    assert 0 <= report['bias_score'] <= 1


def assert_feature_recommendations_structure(recommendations):
    """Assert that feature recommendations have the expected structure."""
    required_fields = ['recommended_features', 'transformations']
    for field in required_fields:
        assert field in recommendations, f"Missing required field: {field}"
        assert isinstance(recommendations[field], list)


# Performance testing utilities
class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time


@pytest.fixture
def performance_timer():
    """Performance timer fixture."""
    return PerformanceTimer