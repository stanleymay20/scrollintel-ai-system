"""
Integration tests for EthicsEngine - AI bias detection and fairness evaluation
"""

import pytest
import pytest_asyncio
import pandas as pd
import numpy as np
from datetime import datetime

from scrollintel.engines.ethics_engine import EthicsEngine, ComplianceFramework

class TestEthicsEngineIntegration:
    """Integration test suite for EthicsEngine"""
    
    @pytest_asyncio.fixture
    async def ethics_engine(self):
        """Create and initialize EthicsEngine for testing"""
        engine = EthicsEngine()
        await engine.start()
        yield engine
        await engine.stop()
    
    @pytest.fixture
    def biased_dataset(self):
        """Create a biased dataset for testing"""
        np.random.seed(42)
        n_samples = 200
        
        # Create biased data
        gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4])
        race = np.random.choice(['White', 'Black', 'Hispanic'], n_samples, p=[0.6, 0.2, 0.2])
        
        data = pd.DataFrame({
            'gender': gender,
            'race': race,
            'age': np.random.randint(18, 65, n_samples),
            'income': np.random.normal(50000, 15000, n_samples)
        })
        
        # Create biased predictions (favor males and whites)
        predictions = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
        male_mask = data['gender'] == 'Male'
        white_mask = data['race'] == 'White'
        
        # Introduce bias
        predictions[male_mask] = np.random.choice([0, 1], np.sum(male_mask), p=[0.2, 0.8])
        predictions[white_mask] = np.random.choice([0, 1], np.sum(white_mask), p=[0.3, 0.7])
        
        true_labels = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
        pred_probs = np.random.uniform(0, 1, n_samples)
        
        return data, predictions, true_labels, pred_probs
    
    @pytest.mark.asyncio
    async def test_complete_bias_detection_workflow(self, ethics_engine, biased_dataset):
        """Test complete bias detection workflow"""
        data, predictions, true_labels, pred_probs = biased_dataset
        
        # 1. Detect bias
        result = await ethics_engine.detect_bias(
            data=data,
            predictions=predictions,
            protected_attributes=['gender', 'race'],
            true_labels=true_labels,
            prediction_probabilities=pred_probs
        )
        
        assert result["status"] == "success"
        assert result["results"]["bias_detected"] is True
        assert len(result["results"]["protected_attributes"]) == 2
        assert "gender" in result["results"]["fairness_metrics"]
        assert "race" in result["results"]["fairness_metrics"]
        
        # 2. Generate transparency report
        model_info = {
            "model_type": "Test Model",
            "training_date": "2024-01-01",
            "features": list(data.columns),
            "training_size": len(data),
            "version": "1.0",
            "automated_decisions": True,
            "explainable": True
        }
        
        performance_metrics = {
            "accuracy": 0.85,
            "precision": 0.80,
            "recall": 0.75
        }
        
        report_result = await ethics_engine.generate_transparency_report(
            model_info=model_info,
            bias_results=result["results"],
            performance_metrics=performance_metrics
        )
        
        assert report_result["status"] == "success"
        assert "report_id" in report_result["report"]
        assert report_result["report"]["fairness_assessment"]["bias_detected"] is True
        
        # 3. Check compliance
        compliance_result = await ethics_engine.check_regulatory_compliance(
            framework=ComplianceFramework.GDPR,
            model_info=model_info,
            bias_results=result["results"]
        )
        
        assert compliance_result["status"] == "success"
        assert compliance_result["compliance"]["framework"] == "GDPR"
        assert compliance_result["compliance"]["compliant"] is False  # Should fail due to bias
        
        # 4. Get audit trail
        audit_result = await ethics_engine.get_audit_trail()
        assert audit_result["status"] == "success"
        assert audit_result["total_entries"] >= 3  # bias_detection, transparency_report, compliance_check
    
    @pytest.mark.asyncio
    async def test_fairness_metrics_calculation(self, ethics_engine, biased_dataset):
        """Test fairness metrics calculation accuracy"""
        data, predictions, true_labels, pred_probs = biased_dataset
        
        result = await ethics_engine.detect_bias(
            data=data,
            predictions=predictions,
            protected_attributes=['gender'],
            true_labels=true_labels,
            prediction_probabilities=pred_probs
        )
        
        assert result["status"] == "success"
        gender_metrics = result["results"]["fairness_metrics"]["gender"]
        
        # Check that all expected metrics are calculated
        assert "demographic_parity" in gender_metrics["metrics"]
        assert "equalized_odds" in gender_metrics["metrics"]
        assert "equal_opportunity" in gender_metrics["metrics"]
        assert "calibration" in gender_metrics["metrics"]
        
        # Verify demographic parity structure
        dp = gender_metrics["metrics"]["demographic_parity"]
        assert "group_rates" in dp
        assert "parity_difference" in dp
        assert "bias_detected" in dp
        assert isinstance(dp["parity_difference"], float)
    
    @pytest.mark.asyncio
    async def test_ethical_guidelines_and_thresholds(self, ethics_engine):
        """Test ethical guidelines and threshold management"""
        # Get initial guidelines
        guidelines_result = await ethics_engine.get_ethical_guidelines()
        assert guidelines_result["status"] == "success"
        assert len(guidelines_result["ethical_principles"]) == 8
        
        # Update thresholds
        new_thresholds = {
            "demographic_parity_difference": 0.05,
            "equalized_odds_difference": 0.08
        }
        
        update_result = await ethics_engine.update_fairness_thresholds(new_thresholds)
        assert update_result["status"] == "success"
        assert update_result["updated_thresholds"]["demographic_parity_difference"] == 0.05
        
        # Verify thresholds were updated
        updated_guidelines = await ethics_engine.get_ethical_guidelines()
        assert updated_guidelines["fairness_thresholds"]["demographic_parity_difference"] == 0.05
    
    @pytest.mark.asyncio
    async def test_compliance_frameworks(self, ethics_engine, biased_dataset):
        """Test different compliance frameworks"""
        data, predictions, true_labels, _ = biased_dataset
        
        bias_result = await ethics_engine.detect_bias(
            data=data,
            predictions=predictions,
            protected_attributes=['gender'],
            true_labels=true_labels
        )
        
        model_info = {
            "model_type": "Test Model",
            "risk_assessment": False,
            "monitoring_plan": False,
            "human_oversight": False,
            "documentation": False,
            "risk_category": "high"
        }
        
        # Test GDPR
        gdpr_result = await ethics_engine.check_regulatory_compliance(
            framework=ComplianceFramework.GDPR,
            model_info=model_info,
            bias_results=bias_result["results"]
        )
        assert gdpr_result["status"] == "success"
        assert gdpr_result["compliance"]["framework"] == "GDPR"
        
        # Test NIST AI RMF
        nist_result = await ethics_engine.check_regulatory_compliance(
            framework=ComplianceFramework.NIST_AI_RMF,
            model_info=model_info,
            bias_results=bias_result["results"]
        )
        assert nist_result["status"] == "success"
        assert nist_result["compliance"]["framework"] == "NIST AI RMF"
        
        # Test EU AI Act
        eu_result = await ethics_engine.check_regulatory_compliance(
            framework=ComplianceFramework.EU_AI_ACT,
            model_info=model_info,
            bias_results=bias_result["results"]
        )
        assert eu_result["status"] == "success"
        assert eu_result["compliance"]["framework"] == "EU AI Act"
    
    @pytest.mark.asyncio
    async def test_engine_performance(self, ethics_engine, biased_dataset):
        """Test engine performance with larger dataset"""
        data, predictions, true_labels, pred_probs = biased_dataset
        
        # Duplicate data to make it larger
        large_data = pd.concat([data] * 5, ignore_index=True)
        large_predictions = np.tile(predictions, 5)
        large_true_labels = np.tile(true_labels, 5)
        large_pred_probs = np.tile(pred_probs, 5)
        
        start_time = datetime.now()
        
        result = await ethics_engine.detect_bias(
            data=large_data,
            predictions=large_predictions,
            protected_attributes=['gender', 'race'],
            true_labels=large_true_labels,
            prediction_probabilities=large_pred_probs
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        assert result["status"] == "success"
        assert result["results"]["total_samples"] == 1000
        assert processing_time < 10  # Should complete within 10 seconds
        
        print(f"Processing time for 1000 samples: {processing_time:.2f} seconds")

if __name__ == "__main__":
    pytest.main([__file__])