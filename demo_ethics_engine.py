"""
Demo script for EthicsEngine - AI bias detection and fairness evaluation
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import json

from scrollintel.engines.ethics_engine import EthicsEngine, ComplianceFramework

async def demo_ethics_engine():
    """Demonstrate EthicsEngine capabilities"""
    print("🔍 ScrollIntel EthicsEngine Demo")
    print("=" * 50)
    
    # Initialize EthicsEngine
    ethics_engine = EthicsEngine()
    await ethics_engine.start()
    
    try:
        # 1. Create synthetic dataset with potential bias
        print("\n1. Creating synthetic dataset with potential bias...")
        np.random.seed(42)
        
        # Generate synthetic data
        n_samples = 1000
        
        # Protected attributes
        gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4])
        race = np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], n_samples, p=[0.5, 0.2, 0.2, 0.1])
        age_group = np.random.choice(['Young', 'Middle', 'Senior'], n_samples, p=[0.3, 0.5, 0.2])
        
        # Features (with some correlation to protected attributes to introduce bias)
        education_score = np.random.normal(0.7, 0.2, n_samples)
        experience_years = np.random.normal(8, 4, n_samples)
        
        # Introduce bias: higher scores for certain groups
        bias_factor = np.where(gender == 'Male', 0.1, 0) + np.where(race == 'White', 0.15, 0)
        education_score += bias_factor
        experience_years += bias_factor * 2
        
        # Create biased predictions (loan approval scenario)
        base_score = 0.3 * education_score + 0.2 * (experience_years / 10) + bias_factor
        predictions = (base_score + np.random.normal(0, 0.1, n_samples)) > 0.5
        prediction_probs = np.clip(base_score + np.random.normal(0, 0.1, n_samples), 0, 1)
        
        # True labels (less biased ground truth)
        true_labels = (0.4 * education_score + 0.3 * (experience_years / 10) + np.random.normal(0, 0.1, n_samples)) > 0.5
        
        # Create DataFrame
        data = pd.DataFrame({
            'gender': gender,
            'race': race,
            'age_group': age_group,
            'education_score': education_score,
            'experience_years': experience_years
        })
        
        print(f"Dataset created: {len(data)} samples")
        print(f"Protected attributes: gender, race, age_group")
        print(f"Prediction rate: {np.mean(predictions):.3f}")
        
        # 2. Detect bias
        print("\n2. Detecting bias across protected attributes...")
        
        bias_result = await ethics_engine.detect_bias(
            data=data,
            predictions=predictions.astype(int),
            protected_attributes=['gender', 'race', 'age_group'],
            true_labels=true_labels.astype(int),
            prediction_probabilities=prediction_probs
        )
        
        if bias_result["status"] == "success":
            results = bias_result["results"]
            print(f"Bias detected: {results['bias_detected']}")
            print(f"Total samples analyzed: {results['total_samples']}")
            
            # Show fairness metrics for each attribute
            for attr, metrics in results["fairness_metrics"].items():
                print(f"\n--- {attr.upper()} ---")
                print(f"Bias detected: {metrics.get('bias_detected', False)}")
                
                if "metrics" in metrics:
                    for metric_name, metric_data in metrics["metrics"].items():
                        if isinstance(metric_data, dict) and "bias_detected" in metric_data:
                            print(f"{metric_name}: {metric_data.get('bias_detected', False)}")
                            if metric_name == "demographic_parity":
                                print(f"  Parity difference: {metric_data.get('parity_difference', 0):.3f}")
                            elif metric_name == "equalized_odds":
                                print(f"  Equalized odds difference: {metric_data.get('equalized_odds_difference', 0):.3f}")
            
            # Show recommendations
            print(f"\nRecommendations ({len(results['recommendations'])}):")
            for i, rec in enumerate(results["recommendations"][:5], 1):
                print(f"{i}. {rec}")
        
        # 3. Generate transparency report
        print("\n3. Generating AI transparency report...")
        
        model_info = {
            "model_type": "Logistic Regression",
            "training_date": "2024-01-15",
            "features": list(data.columns),
            "training_size": len(data),
            "version": "1.0",
            "automated_decisions": True,
            "explainable": True,
            "risk_assessment": True,
            "monitoring_plan": True,
            "human_oversight": True,
            "documentation": True,
            "risk_category": "high"
        }
        
        performance_metrics = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.78,
            "f1_score": 0.80,
            "auc_roc": 0.88
        }
        
        transparency_result = await ethics_engine.generate_transparency_report(
            model_info=model_info,
            bias_results=results,
            performance_metrics=performance_metrics
        )
        
        if transparency_result["status"] == "success":
            report = transparency_result["report"]
            print(f"Transparency report generated: {report['report_id']}")
            print(f"Model type: {report['model_information']['model_type']}")
            print(f"Overall bias detected: {report['fairness_assessment']['bias_detected']}")
            
            # Show ethical compliance
            print("\nEthical Compliance Assessment:")
            for principle, assessment in report["ethical_compliance"].items():
                status = "✅" if assessment["compliant"] else "❌"
                print(f"{status} {principle.title()}: {assessment['assessment']}")
            
            # Show risk assessment
            print("\nRisk Assessment:")
            for risk_type, risk_data in report["risk_assessment"].items():
                print(f"- {risk_type.replace('_', ' ').title()}: {risk_data['level']}")
        
        # 4. Check regulatory compliance
        print("\n4. Checking regulatory compliance...")
        
        # Check GDPR compliance
        gdpr_result = await ethics_engine.check_regulatory_compliance(
            framework=ComplianceFramework.GDPR,
            model_info=model_info,
            bias_results=results
        )
        
        if gdpr_result["status"] == "success":
            compliance = gdpr_result["compliance"]
            status = "✅ COMPLIANT" if compliance["compliant"] else "❌ NON-COMPLIANT"
            print(f"GDPR: {status}")
            if compliance["issues"]:
                print("Issues found:")
                for issue in compliance["issues"]:
                    print(f"  - {issue}")
        
        # Check NIST AI RMF compliance
        nist_result = await ethics_engine.check_regulatory_compliance(
            framework=ComplianceFramework.NIST_AI_RMF,
            model_info=model_info,
            bias_results=results
        )
        
        if nist_result["status"] == "success":
            compliance = nist_result["compliance"]
            status = "✅ COMPLIANT" if compliance["compliant"] else "❌ NON-COMPLIANT"
            print(f"NIST AI RMF: {status}")
            if compliance["issues"]:
                print("Issues found:")
                for issue in compliance["issues"]:
                    print(f"  - {issue}")
        
        # 5. Get ethical guidelines
        print("\n5. Retrieving ethical guidelines...")
        
        guidelines_result = await ethics_engine.get_ethical_guidelines()
        if guidelines_result["status"] == "success":
            principles = guidelines_result["ethical_principles"]
            print(f"Ethical principles ({len(principles)}):")
            for principle, description in list(principles.items())[:4]:
                print(f"- {principle.title()}: {description}")
            
            thresholds = guidelines_result["fairness_thresholds"]
            print(f"\nFairness thresholds:")
            for metric, threshold in thresholds.items():
                print(f"- {metric}: {threshold}")
        
        # 6. Update fairness thresholds
        print("\n6. Updating fairness thresholds...")
        
        new_thresholds = {
            "demographic_parity_difference": 0.05,  # More strict
            "equalized_odds_difference": 0.08
        }
        
        update_result = await ethics_engine.update_fairness_thresholds(new_thresholds)
        if update_result["status"] == "success":
            print("✅ Fairness thresholds updated successfully")
            for metric, threshold in update_result["updated_thresholds"].items():
                print(f"- {metric}: {threshold}")
        
        # 7. Get audit trail
        print("\n7. Retrieving audit trail...")
        
        audit_result = await ethics_engine.get_audit_trail()
        if audit_result["status"] == "success":
            trail = audit_result["audit_trail"]
            print(f"Audit entries: {len(trail)}")
            
            # Show recent entries
            for entry in trail[-3:]:
                print(f"- {entry['timestamp']}: {entry['event_type']}")
        
        # 8. Engine status
        print("\n8. Engine status...")
        status = ethics_engine.get_status()
        print(f"Engine: {status['name']} v{status['version']}")
        print(f"Status: {status['status']}")
        print(f"Audit entries: {status['audit_entries']}")
        print(f"Supported metrics: {len(status['supported_metrics'])}")
        print(f"Compliance frameworks: {len(status['compliance_frameworks'])}")
        
        print("\n✅ EthicsEngine demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await ethics_engine.stop()

if __name__ == "__main__":
    asyncio.run(demo_ethics_engine())