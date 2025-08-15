"""
Unit tests for EthicsEngine - AI bias detection and fairness evaluation
"""

import pytest
import pytest_asyncio
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from scrollintel.engines.ethics_engine import (
    EthicsEngine, 
    BiasType, 
    FairnessMetric, 
    ComplianceFramework
)
from scrollintel.engines.base_engine import EngineStatus, EngineCapability

class TestEthicsEngine:
    """Test suite for EthicsEngine"""
    
    @pytest_asyncio.fixture
    async def ethics_engine(self):
        """Create and initialize EthicsEngine for testing"""
        engine = EthicsEngine()
        await engine.start()
        yield engine
        await engine.stop()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing"""
        np.random.seed(42)
        n_samples = 100
        
        data = pd.DataFrame({
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'race': np.random.choice(['White', 'Black', 'Hispanic'], n_samples),
            'age': np.random.randint(18, 65, n_samples),
            'income': np.random.normal(50000, 15000, n_samples),
            'education': np.random.choice(['High School', 'Bachelor', 'Master'], n_samples)
        })
        
        # Create biased predictions
        predictions = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
        # Introduce bias for gender
        male_mask = data['gender'] == 'Male'
        predictions[male_mask] = np.random.choice([0, 1], np.sum(male_mask), p=[0.3, 0.7])
        
        true_labels = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
        pred_probs = np.random.uniform(0, 1, n_samples)
        
        return data, predictions, true_labels, pred_probs
    
    def test_engine_initialization(self):
        """Test EthicsEngine initialization"""
        engine = EthicsEngine()
        
        assert engine.engine_id == "ethics_engine"
        assert engine.name == "EthicsEngine"
        assert EngineCapability.BIAS_DETECTION in engine.capabilities
        assert EngineCapability.EXPLANATION in engine.capabilities
        assert engine.version == "1.0.0"
        assert len(engine.ethical_principles) == 8
        assert len(engine.fairness_thresholds) == 4
    
    @pytest.mark.asyncio
    async def test_engine_lifecycle(self):
        """Test engine start/stop lifecycle"""
        engine = EthicsEngine()
        
        # Test initialization
        assert engine.status == EngineStatus.INITIALIZING
        
        # Test start
        await engine.start()
        assert engine.status == EngineStatus.READY
        
        # Test stop
        await engine.stop()
        assert engine.status == EngineStatus.MAINTENANCE
    
    @pytest.mark.asyncio
    async def test_get_status(self, ethics_engine):
        """Test getting engine status"""
        status = ethics_engine.get_status()
        
        assert status["engine_id"] == "ethics_engine"
        assert status["name"] == "EthicsEngine"
        assert status["version"] == "1.0.0"
        assert "audit_entries" in status
        assert "supported_metrics" in status
        assert "compliance_frameworks" in status
        assert status["healthy"] is True
    
    @pytest.mark.asyncio
    async def test_health_check(self, ethics_engine):
        """Test health check functionality"""
        is_healthy = await ethics_engine.health_check()
        assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_detect_bias_basic(self, ethics_engine, sample_data):
        """Test basic bias detection functionality"""
        data, predictions, true_labels, pred_probs = sample_data
        
        result = await ethics_engine.detect_bias(
            data=data,
            predictions=predictions,
            protected_attributes=['gender', 'race'],
            true_labels=true_labels,
            prediction_probabilities=pred_probs
        )
        
        assert result["status"] == "success"
        assert "results" in result
        
        results = result["results"]
        assert "timestamp" in results
        assert "protected_attributes" in results
        assert "total_samples" in results
        assert "bias_detected" in results
        assert "fairness_metrics" in results
        assert "group_statistics" in results
        assert "recommendations" in results
        
        # Check protected attributes
        assert results["protected_attributes"] == ['gender', 'race']
        assert results["total_samples"] == len(data)
        
        # Check fairness metrics structure
        assert 'gender' in results["fairness_metrics"]
        assert 'race' in results["fairness_metrics"]
    
    @pytest.mark.asyncio
    async def test_detect_bias_missing_attributes(self, ethics_engine, sample_data):
        """Test bias detection with missing protected attributes"""
        data, predictions, _, _ = sample_data
        
        result = await ethics_engine.detect_bias(
            data=data,
            predictions=predictions,
            protected_attributes=['gender', 'nonexistent_attr']
        )
        
        assert result["status"] == "error"
        assert "not found in data" in result["message"]
    
    @pytest.mark.asyncio
    async def test_demographic_parity_calculation(self, ethics_engine):
        """Test demographic parity calculation"""
        # Create simple test data with clear bias
        data = pd.DataFrame({
            'gender': ['Male'] * 50 + ['Female'] * 50
        })
        
        # Biased predictions: 80% positive for males, 40% positive for females
        predictions = np.array([1] * 40 + [0] * 10 + [1] * 20 + [0] * 30)
        
        result = await ethics_engine.detect_bias(
            data=data,
            predictions=predictions,
            protected_attributes=['gender']
        )
        
        assert result["status"] == "success"
        results = result["results"]
        
        # Should detect bias due to large difference in positive rates
        assert results["bias_detected"] is True
        
        gender_metrics = results["fairness_metrics"]["gender"]
        assert gender_metrics["bias_detected"] is True
        
        # Check demographic parity metrics
        demo_parity = gender_metrics["metrics"]["demographic_parity"]
        assert demo_parity["bias_detected"] is True
        assert demo_parity["parity_difference"] > 0.1  # Should exceed threshold
    
    @pytest.mark.asyncio
    async def test_equalized_odds_calculation(self, ethics_engine):
        """Test equalized odds calculation"""
        data = pd.DataFrame({
            'gender': ['Male'] * 50 + ['Female'] * 50
        })
        
        predictions = np.array([1] * 40 + [0] * 10 + [1] * 20 + [0] * 30)
        true_labels = np.array([1] * 25 + [0] * 25 + [1] * 25 + [0] * 25)
        
        result = await ethics_engine.detect_bias(
            data=data,
            predictions=predictions,
            protected_attributes=['gender'],
            true_labels=true_labels
        )
        
        assert result["status"] == "success"
        results = result["results"]
        
        gender_metrics = results["fairness_metrics"]["gender"]
        assert "equalized_odds" in gender_metrics["metrics"]
        
        eq_odds = gender_metrics["metrics"]["equalized_odds"]
        assert "group_metrics" in eq_odds
        assert "tpr_difference" in eq_odds
        assert "fpr_difference" in eq_odds
        assert "equalized_odds_difference" in eq_odds
    
    @pytest.mark.asyncio
    async def test_equal_opportunity_calculation(self, ethics_engine):
        """Test equal opportunity calculation"""
        data = pd.DataFrame({
            'gender': ['Male'] * 50 + ['Female'] * 50
        })
        
        predictions = np.array([1] * 40 + [0] * 10 + [1] * 20 + [0] * 30)
        true_labels = np.array([1] * 25 + [0] * 25 + [1] * 25 + [0] * 25)
        
        result = await ethics_engine.detect_bias(
            data=data,
            predictions=predictions,
            protected_attributes=['gender'],
            true_labels=true_labels
        )
        
        assert result["status"] == "success"
        results = result["results"]
        
        gender_metrics = results["fairness_metrics"]["gender"]
        assert "equal_opportunity" in gender_metrics["metrics"]
        
        eq_opp = gender_metrics["metrics"]["equal_opportunity"]
        assert "group_tprs" in eq_opp
        assert "equal_opportunity_difference" in eq_opp
    
    @pytest.mark.asyncio
    async def test_calibration_calculation(self, ethics_engine):
        """Test calibration calculation"""
        data = pd.DataFrame({
            'gender': ['Male'] * 50 + ['Female'] * 50
        })
        
        predictions = np.array([1] * 40 + [0] * 10 + [1] * 20 + [0] * 30)
        true_labels = np.array([1] * 25 + [0] * 25 + [1] * 25 + [0] * 25)
        pred_probs = np.random.uniform(0, 1, 100)
        
        result = await ethics_engine.detect_bias(
            data=data,
            predictions=predictions,
            protected_attributes=['gender'],
            true_labels=true_labels,
            prediction_probabilities=pred_probs
        )
        
        assert result["status"] == "success"
        results = result["results"]
        
        gender_metrics = results["fairness_metrics"]["gender"]
        assert "calibration" in gender_metrics["metrics"]
        
        calibration = gender_metrics["metrics"]["calibration"]
        assert "group_calibration" in calibration
        assert "calibration_difference" in calibration
    
    @pytest.mark.asyncio
    async def test_generate_transparency_report(self, ethics_engine, sample_data):
        """Test transparency report generation"""
        data, predictions, true_labels, pred_probs = sample_data
        
        # First detect bias
        bias_result = await ethics_engine.detect_bias(
            data=data,
            predictions=predictions,
            protected_attributes=['gender'],
            true_labels=true_labels,
            prediction_probabilities=pred_probs
        )
        
        model_info = {
            "model_type": "Random Forest",
            "training_date": "2024-01-01",
            "features": list(data.columns),
            "training_size": len(data),
            "version": "1.0"
        }
        
        performance_metrics = {
            "accuracy": 0.85,
            "precision": 0.80,
            "recall": 0.75,
            "f1_score": 0.77
        }
        
        result = await ethics_engine.generate_transparency_report(
            model_info=model_info,
            bias_results=bias_result["results"],
            performance_metrics=performance_metrics
        )
        
        assert result["status"] == "success"
        assert "report" in result
        
        report = result["report"]
        assert "report_id" in report
        assert "timestamp" in report
        assert "model_information" in report
        assert "fairness_assessment" in report
        assert "performance_metrics" in report
        assert "ethical_compliance" in report
        assert "risk_assessment" in report
        assert "recommendations" in report
        assert "limitations" in report
        assert "monitoring_plan" in report
    
    @pytest.mark.asyncio
    async def test_check_gdpr_compliance(self, ethics_engine, sample_data):
        """Test GDPR compliance checking"""
        data, predictions, true_labels, _ = sample_data
        
        bias_result = await ethics_engine.detect_bias(
            data=data,
            predictions=predictions,
            protected_attributes=['gender'],
            true_labels=true_labels
        )
        
        model_info = {
            "automated_decisions": True,
            "explainable": False,
            "consent_mechanism": False
        }
        
        result = await ethics_engine.check_regulatory_compliance(
            framework=ComplianceFramework.GDPR,
            model_info=model_info,
            bias_results=bias_result["results"]
        )
        
        assert result["status"] == "success"
        assert "compliance" in result
        
        compliance = result["compliance"]
        assert compliance["framework"] == "GDPR"
        assert "compliant" in compliance
        assert "issues" in compliance
        assert "recommendations" in compliance
    
    @pytest.mark.asyncio
    async def test_check_nist_compliance(self, ethics_engine, sample_data):
        """Test NIST AI RMF compliance checking"""
        data, predictions, true_labels, _ = sample_data
        
        bias_result = await ethics_engine.detect_bias(
            data=data,
            predictions=predictions,
            protected_attributes=['gender'],
            true_labels=true_labels
        )
        
        model_info = {
            "risk_assessment": False,
            "monitoring_plan": False
        }
        
        result = await ethics_engine.check_regulatory_compliance(
            framework=ComplianceFramework.NIST_AI_RMF,
            model_info=model_info,
            bias_results=bias_result["results"]
        )
        
        assert result["status"] == "success"
        compliance = result["compliance"]
        assert compliance["framework"] == "NIST AI RMF"
        assert len(compliance["issues"]) > 0  # Should have issues due to missing components
    
    @pytest.mark.asyncio
    async def test_check_eu_ai_act_compliance(self, ethics_engine, sample_data):
        """Test EU AI Act compliance checking"""
        data, predictions, true_labels, _ = sample_data
        
        bias_result = await ethics_engine.detect_bias(
            data=data,
            predictions=predictions,
            protected_attributes=['gender'],
            true_labels=true_labels
        )
        
        model_info = {
            "risk_category": "high",
            "human_oversight": False,
            "documentation": False
        }
        
        result = await ethics_engine.check_regulatory_compliance(
            framework=ComplianceFramework.EU_AI_ACT,
            model_info=model_info,
            bias_results=bias_result["results"]
        )
        
        assert result["status"] == "success"
        compliance = result["compliance"]
        assert compliance["framework"] == "EU AI Act"
    
    @pytest.mark.asyncio
    async def test_unsupported_compliance_framework(self, ethics_engine):
        """Test handling of unsupported compliance framework"""
        # This test would need to be adjusted based on actual implementation
        # For now, all frameworks in the enum are supported
        pass
    
    @pytest.mark.asyncio
    async def test_get_ethical_guidelines(self, ethics_engine):
        """Test getting ethical guidelines"""
        result = await ethics_engine.get_ethical_guidelines()
        
        assert result["status"] == "success"
        assert "ethical_principles" in result
        assert "fairness_thresholds" in result
        assert "supported_bias_types" in result
        assert "supported_metrics" in result
        assert "compliance_frameworks" in result
        
        # Check that all expected principles are present
        principles = result["ethical_principles"]
        expected_principles = ["fairness", "transparency", "accountability", "privacy"]
        for principle in expected_principles:
            assert principle in principles
    
    @pytest.mark.asyncio
    async def test_update_fairness_thresholds(self, ethics_engine):
        """Test updating fairness thresholds"""
        new_thresholds = {
            "demographic_parity_difference": 0.05,
            "equalized_odds_difference": 0.08
        }
        
        result = await ethics_engine.update_fairness_thresholds(new_thresholds)
        
        assert result["status"] == "success"
        assert "updated_thresholds" in result
        assert result["updated_thresholds"]["demographic_parity_difference"] == 0.05
        assert result["updated_thresholds"]["equalized_odds_difference"] == 0.08
    
    @pytest.mark.asyncio
    async def test_update_invalid_fairness_thresholds(self, ethics_engine):
        """Test updating fairness thresholds with invalid values"""
        # Test invalid threshold value (> 1)
        invalid_thresholds = {
            "demographic_parity_difference": 1.5
        }
        
        result = await ethics_engine.update_fairness_thresholds(invalid_thresholds)
        assert result["status"] == "error"
        assert "must be between 0 and 1" in result["message"]
        
        # Test invalid metric name
        invalid_metric = {
            "nonexistent_metric": 0.1
        }
        
        result = await ethics_engine.update_fairness_thresholds(invalid_metric)
        assert result["status"] == "error"
        assert "Unknown fairness metric" in result["message"]
    
    @pytest.mark.asyncio
    async def test_audit_trail_logging(self, ethics_engine, sample_data):
        """Test audit trail logging functionality"""
        data, predictions, _, _ = sample_data
        
        # Perform bias detection to generate audit entries
        await ethics_engine.detect_bias(
            data=data,
            predictions=predictions,
            protected_attributes=['gender']
        )
        
        # Get audit trail
        result = await ethics_engine.get_audit_trail()
        
        assert result["status"] == "success"
        assert "audit_trail" in result
        assert "total_entries" in result
        
        # Should have at least one entry from bias detection
        assert result["total_entries"] > 0
        
        # Check audit entry structure
        if result["audit_trail"]:
            entry = result["audit_trail"][0]
            assert "timestamp" in entry
            assert "event_type" in entry
            assert "details" in entry
            assert "engine_id" in entry
    
    @pytest.mark.asyncio
    async def test_audit_trail_filtering(self, ethics_engine, sample_data):
        """Test audit trail filtering functionality"""
        data, predictions, _, _ = sample_data
        
        # Generate some audit entries
        await ethics_engine.detect_bias(
            data=data,
            predictions=predictions,
            protected_attributes=['gender']
        )
        
        # Test filtering by event type
        result = await ethics_engine.get_audit_trail(event_type="bias_detection")
        
        assert result["status"] == "success"
        assert "filters_applied" in result
        assert result["filters_applied"]["event_type"] == "bias_detection"
    
    @pytest.mark.asyncio
    async def test_group_statistics_calculation(self, ethics_engine):
        """Test group statistics calculation"""
        data = pd.DataFrame({
            'gender': ['Male'] * 50 + ['Female'] * 50,
            'age': list(range(25, 75)) + list(range(20, 70)),
            'income': [60000] * 50 + [45000] * 50
        })
        
        predictions = np.array([1] * 40 + [0] * 10 + [1] * 20 + [0] * 30)
        
        result = await ethics_engine.detect_bias(
            data=data,
            predictions=predictions,
            protected_attributes=['gender']
        )
        
        assert result["status"] == "success"
        results = result["results"]
        
        group_stats = results["group_statistics"]
        assert "gender" in group_stats
        
        gender_stats = group_stats["gender"]
        assert "Male" in gender_stats
        assert "Female" in gender_stats
        
        # Check male group statistics
        male_stats = gender_stats["Male"]
        assert male_stats["sample_size"] == 50
        assert male_stats["percentage"] == 50.0
        assert "positive_prediction_rate" in male_stats
        assert "feature_statistics" in male_stats
    
    @pytest.mark.asyncio
    async def test_bias_recommendations_generation(self, ethics_engine):
        """Test bias recommendations generation"""
        data = pd.DataFrame({
            'gender': ['Male'] * 50 + ['Female'] * 50
        })
        
        # Create clearly biased predictions
        predictions = np.array([1] * 45 + [0] * 5 + [1] * 10 + [0] * 40)
        
        result = await ethics_engine.detect_bias(
            data=data,
            predictions=predictions,
            protected_attributes=['gender']
        )
        
        assert result["status"] == "success"
        results = result["results"]
        
        # Should detect bias and generate recommendations
        assert results["bias_detected"] is True
        assert len(results["recommendations"]) > 0
        
        # Check that recommendations contain relevant advice
        recommendations_text = " ".join(results["recommendations"])
        assert any(keyword in recommendations_text.lower() for keyword in 
                  ["bias", "fairness", "rebalance", "mitigation"])
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_data(self, ethics_engine):
        """Test error handling with invalid data"""
        # Test with mismatched data and predictions length
        data = pd.DataFrame({'gender': ['Male', 'Female']})
        predictions = np.array([1, 0, 1])  # Different length
        
        result = await ethics_engine.detect_bias(
            data=data,
            predictions=predictions,
            protected_attributes=['gender']
        )
        
        # Should handle gracefully (implementation dependent)
        # This test verifies the engine doesn't crash with invalid input
        assert "status" in result
    
    def test_fairness_threshold_initialization(self):
        """Test that fairness thresholds are properly initialized"""
        engine = EthicsEngine()
        
        # Check that default thresholds are set
        assert FairnessMetric.DEMOGRAPHIC_PARITY_DIFFERENCE in engine.fairness_thresholds
        assert FairnessMetric.EQUALIZED_ODDS_DIFFERENCE in engine.fairness_thresholds
        assert FairnessMetric.EQUAL_OPPORTUNITY_DIFFERENCE in engine.fairness_thresholds
        assert FairnessMetric.CALIBRATION_ERROR in engine.fairness_thresholds
        
        # Check threshold values are reasonable
        for threshold in engine.fairness_thresholds.values():
            assert 0 <= threshold <= 1
    
    def test_ethical_principles_initialization(self):
        """Test that ethical principles are properly initialized"""
        engine = EthicsEngine()
        
        expected_principles = [
            "fairness", "transparency", "accountability", "privacy",
            "beneficence", "non_maleficence", "autonomy", "justice"
        ]
        
        for principle in expected_principles:
            assert principle in engine.ethical_principles
            assert isinstance(engine.ethical_principles[principle], str)
            assert len(engine.ethical_principles[principle]) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_bias_detection(self, ethics_engine, sample_data):
        """Test concurrent bias detection operations"""
        data, predictions, true_labels, pred_probs = sample_data
        
        # Run multiple bias detection operations concurrently
        tasks = []
        for i in range(3):
            task = ethics_engine.detect_bias(
                data=data,
                predictions=predictions,
                protected_attributes=['gender'],
                true_labels=true_labels,
                prediction_probabilities=pred_probs
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # All operations should succeed
        for result in results:
            assert result["status"] == "success"
            assert "results" in result
    
    @pytest.mark.asyncio
    async def test_memory_management(self, ethics_engine):
        """Test that audit trail doesn't grow indefinitely"""
        # Add many audit entries
        for i in range(1200):  # More than the 1000 limit
            ethics_engine._log_audit_event(f"test_event_{i}", {"test": i})
        
        # Check that audit trail is limited
        assert len(ethics_engine.audit_trail) <= 1000
        
        # Check that most recent entries are kept
        last_entry = ethics_engine.audit_trail[-1]
        assert "test_event_1199" in last_entry["event_type"]

if __name__ == "__main__":
    pytest.main([__file__])