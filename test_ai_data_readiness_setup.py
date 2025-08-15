#!/usr/bin/env python3
"""Test script to verify AI Data Readiness Platform setup."""

import sys
import unittest
from pathlib import Path
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from ai_data_readiness.core.config import Config
from ai_data_readiness.models.base_models import (
    Dataset, DatasetMetadata, Schema, QualityReport, 
    BiasReport, AIReadinessScore, DatasetStatus
)
from ai_data_readiness.models.drift_models import DriftReport, DriftAlert, AlertSeverity, DriftType
from ai_data_readiness.models.feature_models import FeatureRecommendations, ModelType


class TestAIDataReadinessSetup(unittest.TestCase):
    """Test cases for AI Data Readiness Platform setup."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
    
    def test_config_creation(self):
        """Test configuration creation and validation."""
        self.assertIsInstance(self.config, Config)
        self.assertTrue(self.config.validate())
        
        # Test database config
        self.assertIsNotNone(self.config.database.connection_string)
        self.assertGreater(self.config.database.port, 0)
        
        # Test processing config
        self.assertGreater(self.config.processing.max_workers, 0)
        self.assertGreater(self.config.processing.batch_size, 0)
    
    def test_schema_model(self):
        """Test Schema model functionality."""
        schema = Schema(
            columns={"id": "integer", "name": "string", "score": "float"},
            primary_key="id"
        )
        
        self.assertTrue(schema.validate_data_types())
        self.assertEqual(len(schema.columns), 3)
        self.assertEqual(schema.primary_key, "id")
    
    def test_dataset_model(self):
        """Test Dataset model functionality."""
        metadata = DatasetMetadata(
            name="test_dataset",
            description="Test dataset for validation",
            row_count=1000,
            column_count=5
        )
        
        dataset = Dataset(
            name="test_dataset",
            description="Test dataset",
            metadata=metadata,
            status=DatasetStatus.PENDING
        )
        
        self.assertIsNotNone(dataset.id)
        self.assertEqual(dataset.status, DatasetStatus.PENDING)
        self.assertFalse(dataset.is_ai_ready())
        
        # Test status update
        dataset.update_status(DatasetStatus.READY)
        self.assertEqual(dataset.status, DatasetStatus.READY)
    
    def test_quality_report_model(self):
        """Test QualityReport model functionality."""
        report = QualityReport(
            dataset_id="test-dataset-id",
            overall_score=0.85,
            completeness_score=0.90,
            accuracy_score=0.80,
            consistency_score=0.85,
            validity_score=0.90,
            uniqueness_score=0.95,
            timeliness_score=0.75
        )
        
        self.assertEqual(report.overall_score, 0.85)
        self.assertIsInstance(report.generated_at, datetime)
        
        # Test dimension score retrieval
        from ai_data_readiness.models.base_models import QualityDimension
        completeness = report.get_dimension_score(QualityDimension.COMPLETENESS)
        self.assertEqual(completeness, 0.90)
    
    def test_ai_readiness_score_model(self):
        """Test AIReadinessScore model functionality."""
        score = AIReadinessScore(
            overall_score=0.82,
            data_quality_score=0.85,
            feature_quality_score=0.80,
            bias_score=0.90,
            compliance_score=0.75,
            scalability_score=0.85
        )
        
        self.assertEqual(score.overall_score, 0.82)
        self.assertIsInstance(score.generated_at, datetime)
    
    def test_bias_report_model(self):
        """Test BiasReport model functionality."""
        report = BiasReport(
            dataset_id="test-dataset-id",
            protected_attributes=["gender", "age"],
            bias_metrics={"demographic_parity": 0.15, "equalized_odds": 0.12}
        )
        
        self.assertEqual(len(report.protected_attributes), 2)
        self.assertIn("demographic_parity", report.bias_metrics)
    
    def test_drift_report_model(self):
        """Test DriftReport model functionality."""
        alert = DriftAlert(
            id="alert-1",
            dataset_id="test-dataset-id",
            drift_type=DriftType.COVARIATE_SHIFT,
            severity=AlertSeverity.MEDIUM,
            message="Significant drift detected",
            affected_features=["feature1", "feature2"],
            drift_score=0.35,
            threshold=0.3
        )
        
        report = DriftReport(
            dataset_id="test-dataset-id",
            reference_dataset_id="reference-dataset-id",
            drift_score=0.35,
            feature_drift_scores={"feature1": 0.4, "feature2": 0.3},
            statistical_tests={},
            alerts=[alert]
        )
        
        self.assertEqual(report.drift_score, 0.35)
        self.assertEqual(len(report.alerts), 1)
        self.assertEqual(report.get_severity_level(), AlertSeverity.MEDIUM)
        self.assertTrue(report.has_significant_drift(threshold=0.3))
    
    def test_feature_recommendations_model(self):
        """Test FeatureRecommendations model functionality."""
        recommendations = FeatureRecommendations(
            dataset_id="test-dataset-id",
            model_type=ModelType.RANDOM_FOREST,
            target_column="target"
        )
        
        self.assertEqual(recommendations.model_type, ModelType.RANDOM_FOREST)
        self.assertEqual(recommendations.target_column, "target")
        self.assertIsInstance(recommendations.generated_at, datetime)
    
    def test_directory_structure(self):
        """Test that all required directories exist."""
        required_dirs = [
            "ai_data_readiness",
            "ai_data_readiness/core",
            "ai_data_readiness/models",
            "ai_data_readiness/engines",
            "ai_data_readiness/api",
            "ai_data_readiness/storage",
            "ai_data_readiness/migrations"
        ]
        
        for directory in required_dirs:
            self.assertTrue(Path(directory).exists(), f"Directory {directory} does not exist")
            self.assertTrue(Path(directory, "__init__.py").exists(), f"__init__.py missing in {directory}")


def run_tests():
    """Run all tests."""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    print("Running AI Data Readiness Platform setup tests...")
    run_tests()