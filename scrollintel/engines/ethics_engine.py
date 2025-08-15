"""
EthicsEngine - AI Ethics and Bias Detection Engine for ScrollIntel
Provides comprehensive fairness auditing and ethical AI compliance capabilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import logging
from datetime import datetime
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from .base_engine import BaseEngine, EngineCapability

class BiasType(str, Enum):
    """Types of bias that can be detected."""
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    EQUALITY_OF_OPPORTUNITY = "equality_of_opportunity"
    CALIBRATION = "calibration"
    INDIVIDUAL_FAIRNESS = "individual_fairness"

class FairnessMetric(str, Enum):
    """Fairness metrics for evaluation."""
    DEMOGRAPHIC_PARITY_DIFFERENCE = "demographic_parity_difference"
    DEMOGRAPHIC_PARITY_RATIO = "demographic_parity_ratio"
    EQUALIZED_ODDS_DIFFERENCE = "equalized_odds_difference"
    EQUALIZED_ODDS_RATIO = "equalized_odds_ratio"
    EQUAL_OPPORTUNITY_DIFFERENCE = "equal_opportunity_difference"
    CALIBRATION_ERROR = "calibration_error"
    STATISTICAL_PARITY = "statistical_parity"

class ComplianceFramework(str, Enum):
    """Regulatory compliance frameworks."""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOX = "sox"
    ISO_27001 = "iso_27001"
    NIST_AI_RMF = "nist_ai_rmf"
    EU_AI_ACT = "eu_ai_act"

class EthicsEngine(BaseEngine):
    """
    Advanced AI ethics and bias detection engine providing comprehensive fairness auditing
    """
    
    def __init__(self):
        super().__init__(
            engine_id="ethics_engine",
            name="EthicsEngine",
            capabilities=[
                EngineCapability.BIAS_DETECTION,
                EngineCapability.EXPLANATION
            ]
        )
        self.version = "1.0.0"
        self.logger = logging.getLogger(__name__)
        
        # Initialize bias detection components
        self.protected_attributes = []
        self.fairness_thresholds = {
            FairnessMetric.DEMOGRAPHIC_PARITY_DIFFERENCE: 0.1,
            FairnessMetric.EQUALIZED_ODDS_DIFFERENCE: 0.1,
            FairnessMetric.EQUAL_OPPORTUNITY_DIFFERENCE: 0.1,
            FairnessMetric.CALIBRATION_ERROR: 0.05
        }
        
        # Audit trail storage
        self.audit_trail = []
        
        # Compliance frameworks
        self.compliance_checks = {
            ComplianceFramework.GDPR: self._check_gdpr_compliance,
            ComplianceFramework.NIST_AI_RMF: self._check_nist_compliance,
            ComplianceFramework.EU_AI_ACT: self._check_eu_ai_act_compliance
        }
        
        # Ethical guidelines
        self.ethical_principles = {
            "fairness": "Ensure equal treatment across all demographic groups",
            "transparency": "Provide clear explanations for AI decisions",
            "accountability": "Maintain human oversight and responsibility",
            "privacy": "Protect personal and sensitive information",
            "beneficence": "Ensure AI benefits humanity and avoids harm",
            "non_maleficence": "Do no harm through AI systems",
            "autonomy": "Respect human agency and decision-making",
            "justice": "Ensure fair distribution of AI benefits and risks"
        }
        
    async def initialize(self) -> None:
        """Initialize the Ethics engine with configuration"""
        try:
            # Initialize bias detection algorithms
            self.audit_trail = []
            self.logger.info("EthicsEngine initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize EthicsEngine: {str(e)}")
            raise
    
    async def process(self, input_data: Any, parameters: Dict[str, Any] = None) -> Any:
        """Process input data and return results"""
        # This is a generic process method - specific methods are called directly
        return {"status": "success", "message": "Use specific ethics analysis methods"}
    
    async def cleanup(self) -> None:
        """Clean up resources used by the engine"""
        self.audit_trail = []
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the engine"""
        return {
            "engine_id": self.engine_id,
            "name": self.name,
            "status": self.status.value,
            "version": self.version,
            "audit_entries": len(self.audit_trail),
            "protected_attributes": len(self.protected_attributes),
            "supported_metrics": [metric.value for metric in FairnessMetric],
            "compliance_frameworks": [framework.value for framework in ComplianceFramework],
            "healthy": True
        }
    
    async def detect_bias(self, 
                         data: pd.DataFrame,
                         predictions: np.ndarray,
                         protected_attributes: List[str],
                         true_labels: Optional[np.ndarray] = None,
                         prediction_probabilities: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Comprehensive bias detection across multiple fairness metrics
        
        Args:
            data: Input dataset with features
            predictions: Model predictions
            protected_attributes: List of protected attribute column names
            true_labels: Ground truth labels (optional, for supervised metrics)
            prediction_probabilities: Prediction probabilities (optional)
        
        Returns:
            Dictionary containing bias analysis results
        """
        try:
            self.protected_attributes = protected_attributes
            bias_results = {
                "timestamp": datetime.utcnow().isoformat(),
                "protected_attributes": protected_attributes,
                "total_samples": len(data),
                "bias_detected": False,
                "fairness_metrics": {},
                "group_statistics": {},
                "recommendations": []
            }
            
            # Validate protected attributes exist in data
            missing_attrs = [attr for attr in protected_attributes if attr not in data.columns]
            if missing_attrs:
                return {
                    "status": "error",
                    "message": f"Protected attributes not found in data: {missing_attrs}"
                }
            
            # Calculate bias metrics for each protected attribute
            for attr in protected_attributes:
                attr_results = await self._analyze_attribute_bias(
                    data, predictions, attr, true_labels, prediction_probabilities
                )
                bias_results["fairness_metrics"][attr] = attr_results
                
                # Check if bias is detected
                if attr_results.get("bias_detected", False):
                    bias_results["bias_detected"] = True
            
            # Generate group statistics
            bias_results["group_statistics"] = self._calculate_group_statistics(
                data, predictions, protected_attributes
            )
            
            # Generate recommendations
            bias_results["recommendations"] = self._generate_bias_recommendations(
                bias_results["fairness_metrics"]
            )
            
            # Log to audit trail
            self._log_audit_event("bias_detection", {
                "protected_attributes": protected_attributes,
                "bias_detected": bias_results["bias_detected"],
                "metrics_calculated": list(bias_results["fairness_metrics"].keys())
            })
            
            return {"status": "success", "results": bias_results}
            
        except Exception as e:
            self.logger.error(f"Failed to detect bias: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _analyze_attribute_bias(self,
                                    data: pd.DataFrame,
                                    predictions: np.ndarray,
                                    attribute: str,
                                    true_labels: Optional[np.ndarray] = None,
                                    prediction_probabilities: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Analyze bias for a specific protected attribute"""
        try:
            attr_values = data[attribute]
            unique_groups = attr_values.unique()
            
            results = {
                "attribute": attribute,
                "groups": unique_groups.tolist(),
                "metrics": {},
                "bias_detected": False
            }
            
            # Calculate demographic parity
            demo_parity = self._calculate_demographic_parity(
                predictions, attr_values, unique_groups
            )
            results["metrics"]["demographic_parity"] = demo_parity
            
            # Calculate equalized odds (if true labels available)
            if true_labels is not None:
                eq_odds = self._calculate_equalized_odds(
                    predictions, true_labels, attr_values, unique_groups
                )
                results["metrics"]["equalized_odds"] = eq_odds
                
                # Calculate equal opportunity
                eq_opp = self._calculate_equal_opportunity(
                    predictions, true_labels, attr_values, unique_groups
                )
                results["metrics"]["equal_opportunity"] = eq_opp
            
            # Calculate calibration (if probabilities available)
            if prediction_probabilities is not None and true_labels is not None:
                calibration = self._calculate_calibration(
                    prediction_probabilities, true_labels, attr_values, unique_groups
                )
                results["metrics"]["calibration"] = calibration
            
            # Check if any metric indicates bias
            for metric_name, metric_data in results["metrics"].items():
                if metric_data.get("bias_detected", False):
                    results["bias_detected"] = True
                    break
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to analyze attribute bias for {attribute}: {str(e)}")
            return {"attribute": attribute, "error": str(e)}
    
    def _calculate_demographic_parity(self,
                                    predictions: np.ndarray,
                                    attr_values: pd.Series,
                                    groups: np.ndarray) -> Dict[str, Any]:
        """Calculate demographic parity metrics"""
        try:
            group_rates = {}
            
            for group in groups:
                group_mask = attr_values == group
                group_predictions = predictions[group_mask]
                
                if len(group_predictions) > 0:
                    positive_rate = np.mean(group_predictions)
                    group_rates[str(group)] = {
                        "positive_rate": float(positive_rate),
                        "sample_size": int(np.sum(group_mask))
                    }
            
            # Calculate parity difference and ratio
            rates = [data["positive_rate"] for data in group_rates.values()]
            parity_difference = max(rates) - min(rates) if rates else 0
            parity_ratio = min(rates) / max(rates) if rates and max(rates) > 0 else 1
            
            bias_detected = parity_difference > self.fairness_thresholds[
                FairnessMetric.DEMOGRAPHIC_PARITY_DIFFERENCE
            ]
            
            return {
                "group_rates": group_rates,
                "parity_difference": float(parity_difference),
                "parity_ratio": float(parity_ratio),
                "bias_detected": bias_detected,
                "threshold": self.fairness_thresholds[FairnessMetric.DEMOGRAPHIC_PARITY_DIFFERENCE]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate demographic parity: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_equalized_odds(self,
                                predictions: np.ndarray,
                                true_labels: np.ndarray,
                                attr_values: pd.Series,
                                groups: np.ndarray) -> Dict[str, Any]:
        """Calculate equalized odds metrics"""
        try:
            group_metrics = {}
            
            for group in groups:
                group_mask = attr_values == group
                group_predictions = predictions[group_mask]
                group_labels = true_labels[group_mask]
                
                if len(group_predictions) > 0:
                    # True positive rate (sensitivity)
                    positive_mask = group_labels == 1
                    tpr = np.mean(group_predictions[positive_mask]) if np.sum(positive_mask) > 0 else 0
                    
                    # False positive rate (1 - specificity)
                    negative_mask = group_labels == 0
                    fpr = np.mean(group_predictions[negative_mask]) if np.sum(negative_mask) > 0 else 0
                    
                    group_metrics[str(group)] = {
                        "true_positive_rate": float(tpr),
                        "false_positive_rate": float(fpr),
                        "sample_size": int(np.sum(group_mask))
                    }
            
            # Calculate equalized odds difference
            tprs = [data["true_positive_rate"] for data in group_metrics.values()]
            fprs = [data["false_positive_rate"] for data in group_metrics.values()]
            
            tpr_difference = max(tprs) - min(tprs) if tprs else 0
            fpr_difference = max(fprs) - min(fprs) if fprs else 0
            
            # Equalized odds requires both TPR and FPR to be similar across groups
            eq_odds_difference = max(tpr_difference, fpr_difference)
            
            bias_detected = eq_odds_difference > self.fairness_thresholds[
                FairnessMetric.EQUALIZED_ODDS_DIFFERENCE
            ]
            
            return {
                "group_metrics": group_metrics,
                "tpr_difference": float(tpr_difference),
                "fpr_difference": float(fpr_difference),
                "equalized_odds_difference": float(eq_odds_difference),
                "bias_detected": bias_detected,
                "threshold": self.fairness_thresholds[FairnessMetric.EQUALIZED_ODDS_DIFFERENCE]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate equalized odds: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_equal_opportunity(self,
                                   predictions: np.ndarray,
                                   true_labels: np.ndarray,
                                   attr_values: pd.Series,
                                   groups: np.ndarray) -> Dict[str, Any]:
        """Calculate equal opportunity metrics"""
        try:
            group_tprs = {}
            
            for group in groups:
                group_mask = attr_values == group
                group_predictions = predictions[group_mask]
                group_labels = true_labels[group_mask]
                
                if len(group_predictions) > 0:
                    # True positive rate for positive class only
                    positive_mask = group_labels == 1
                    tpr = np.mean(group_predictions[positive_mask]) if np.sum(positive_mask) > 0 else 0
                    
                    group_tprs[str(group)] = {
                        "true_positive_rate": float(tpr),
                        "positive_samples": int(np.sum(positive_mask)),
                        "total_samples": int(np.sum(group_mask))
                    }
            
            # Calculate equal opportunity difference
            tprs = [data["true_positive_rate"] for data in group_tprs.values()]
            eq_opp_difference = max(tprs) - min(tprs) if tprs else 0
            
            bias_detected = eq_opp_difference > self.fairness_thresholds[
                FairnessMetric.EQUAL_OPPORTUNITY_DIFFERENCE
            ]
            
            return {
                "group_tprs": group_tprs,
                "equal_opportunity_difference": float(eq_opp_difference),
                "bias_detected": bias_detected,
                "threshold": self.fairness_thresholds[FairnessMetric.EQUAL_OPPORTUNITY_DIFFERENCE]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate equal opportunity: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_calibration(self,
                             prediction_probabilities: np.ndarray,
                             true_labels: np.ndarray,
                             attr_values: pd.Series,
                             groups: np.ndarray) -> Dict[str, Any]:
        """Calculate calibration metrics"""
        try:
            group_calibration = {}
            
            for group in groups:
                group_mask = attr_values == group
                group_probs = prediction_probabilities[group_mask]
                group_labels = true_labels[group_mask]
                
                if len(group_probs) > 0:
                    # Calculate calibration error using binning
                    n_bins = min(10, len(group_probs) // 10 + 1)
                    bin_boundaries = np.linspace(0, 1, n_bins + 1)
                    
                    calibration_error = 0
                    bin_data = []
                    
                    for i in range(n_bins):
                        bin_lower = bin_boundaries[i]
                        bin_upper = bin_boundaries[i + 1]
                        
                        in_bin = (group_probs >= bin_lower) & (group_probs < bin_upper)
                        if i == n_bins - 1:  # Include upper boundary for last bin
                            in_bin = (group_probs >= bin_lower) & (group_probs <= bin_upper)
                        
                        if np.sum(in_bin) > 0:
                            bin_accuracy = np.mean(group_labels[in_bin])
                            bin_confidence = np.mean(group_probs[in_bin])
                            bin_size = np.sum(in_bin)
                            
                            bin_error = abs(bin_accuracy - bin_confidence)
                            calibration_error += (bin_size / len(group_probs)) * bin_error
                            
                            bin_data.append({
                                "bin_range": [float(bin_lower), float(bin_upper)],
                                "accuracy": float(bin_accuracy),
                                "confidence": float(bin_confidence),
                                "size": int(bin_size),
                                "error": float(bin_error)
                            })
                    
                    group_calibration[str(group)] = {
                        "calibration_error": float(calibration_error),
                        "bin_data": bin_data,
                        "sample_size": int(np.sum(group_mask))
                    }
            
            # Calculate calibration difference across groups
            errors = [data["calibration_error"] for data in group_calibration.values()]
            calibration_difference = max(errors) - min(errors) if errors else 0
            
            bias_detected = calibration_difference > self.fairness_thresholds[
                FairnessMetric.CALIBRATION_ERROR
            ]
            
            return {
                "group_calibration": group_calibration,
                "calibration_difference": float(calibration_difference),
                "bias_detected": bias_detected,
                "threshold": self.fairness_thresholds[FairnessMetric.CALIBRATION_ERROR]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate calibration: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_group_statistics(self,
                                  data: pd.DataFrame,
                                  predictions: np.ndarray,
                                  protected_attributes: List[str]) -> Dict[str, Any]:
        """Calculate comprehensive group statistics"""
        try:
            group_stats = {}
            
            for attr in protected_attributes:
                attr_values = data[attr]
                unique_groups = attr_values.unique()
                
                attr_stats = {}
                for group in unique_groups:
                    group_mask = attr_values == group
                    group_data = data[group_mask]
                    group_predictions = predictions[group_mask]
                    
                    attr_stats[str(group)] = {
                        "sample_size": int(np.sum(group_mask)),
                        "percentage": float(np.sum(group_mask) / len(data) * 100),
                        "positive_prediction_rate": float(np.mean(group_predictions)),
                        "feature_statistics": {
                            col: {
                                "mean": float(group_data[col].mean()) if pd.api.types.is_numeric_dtype(group_data[col]) else None,
                                "std": float(group_data[col].std()) if pd.api.types.is_numeric_dtype(group_data[col]) else None
                            }
                            for col in group_data.select_dtypes(include=[np.number]).columns
                        }
                    }
                
                group_stats[attr] = attr_stats
            
            return group_stats
            
        except Exception as e:
            self.logger.error(f"Failed to calculate group statistics: {str(e)}")
            return {"error": str(e)}
    
    def _generate_bias_recommendations(self, fairness_metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on bias detection results"""
        recommendations = []
        
        for attr, metrics in fairness_metrics.items():
            if metrics.get("bias_detected", False):
                recommendations.append(f"Bias detected in protected attribute '{attr}'. Consider:")
                
                # Demographic parity recommendations
                if "demographic_parity" in metrics.get("metrics", {}):
                    dp_metrics = metrics["metrics"]["demographic_parity"]
                    if dp_metrics.get("bias_detected", False):
                        recommendations.append(
                            f"- Rebalance training data for attribute '{attr}' "
                            f"(parity difference: {dp_metrics.get('parity_difference', 0):.3f})"
                        )
                
                # Equalized odds recommendations
                if "equalized_odds" in metrics.get("metrics", {}):
                    eo_metrics = metrics["metrics"]["equalized_odds"]
                    if eo_metrics.get("bias_detected", False):
                        recommendations.append(
                            f"- Apply post-processing fairness constraints for attribute '{attr}' "
                            f"(equalized odds difference: {eo_metrics.get('equalized_odds_difference', 0):.3f})"
                        )
                
                # General recommendations
                recommendations.extend([
                    f"- Consider fairness-aware machine learning algorithms",
                    f"- Implement bias mitigation techniques during preprocessing",
                    f"- Regular monitoring and auditing of model predictions",
                    f"- Stakeholder consultation on fairness definitions"
                ])
        
        if not any(metrics.get("bias_detected", False) for metrics in fairness_metrics.values()):
            recommendations.append("No significant bias detected. Continue regular monitoring.")
        
        return recommendations
    
    async def generate_transparency_report(self,
                                         model_info: Dict[str, Any],
                                         bias_results: Dict[str, Any],
                                         performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive AI transparency report"""
        try:
            report = {
                "report_id": f"transparency_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.utcnow().isoformat(),
                "model_information": {
                    "model_type": model_info.get("model_type", "Unknown"),
                    "training_date": model_info.get("training_date"),
                    "features_used": model_info.get("features", []),
                    "training_data_size": model_info.get("training_size"),
                    "model_version": model_info.get("version", "1.0")
                },
                "fairness_assessment": bias_results,
                "performance_metrics": performance_metrics,
                "ethical_compliance": await self._assess_ethical_compliance(bias_results),
                "risk_assessment": self._assess_model_risks(bias_results, performance_metrics),
                "recommendations": self._generate_transparency_recommendations(bias_results),
                "limitations": self._identify_model_limitations(model_info, bias_results),
                "monitoring_plan": self._create_monitoring_plan()
            }
            
            # Log to audit trail
            self._log_audit_event("transparency_report_generated", {
                "report_id": report["report_id"],
                "model_type": model_info.get("model_type"),
                "bias_detected": bias_results.get("bias_detected", False)
            })
            
            return {"status": "success", "report": report}
            
        except Exception as e:
            self.logger.error(f"Failed to generate transparency report: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _assess_ethical_compliance(self, bias_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess compliance with ethical principles"""
        compliance = {}
        
        for principle, description in self.ethical_principles.items():
            if principle == "fairness":
                compliance[principle] = {
                    "compliant": not bias_results.get("bias_detected", False),
                    "description": description,
                    "assessment": "PASS" if not bias_results.get("bias_detected", False) else "FAIL",
                    "details": "No significant bias detected" if not bias_results.get("bias_detected", False) 
                             else "Bias detected in model predictions"
                }
            else:
                # For other principles, provide general assessment
                compliance[principle] = {
                    "compliant": True,  # Default to compliant, would need specific checks
                    "description": description,
                    "assessment": "REVIEW_REQUIRED",
                    "details": f"Manual review required for {principle} compliance"
                }
        
        return compliance
    
    def _assess_model_risks(self,
                          bias_results: Dict[str, Any],
                          performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess model risks based on bias and performance"""
        risks = {
            "bias_risk": {
                "level": "HIGH" if bias_results.get("bias_detected", False) else "LOW",
                "description": "Risk of discriminatory outcomes",
                "mitigation": "Implement bias mitigation techniques" if bias_results.get("bias_detected", False) 
                            else "Continue monitoring"
            },
            "performance_risk": {
                "level": "MEDIUM",  # Would need actual performance thresholds
                "description": "Risk of poor model performance",
                "mitigation": "Regular model retraining and validation"
            },
            "regulatory_risk": {
                "level": "MEDIUM",
                "description": "Risk of regulatory non-compliance",
                "mitigation": "Regular compliance audits and documentation"
            }
        }
        
        return risks
    
    def _generate_transparency_recommendations(self, bias_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for transparency improvement"""
        recommendations = [
            "Implement regular bias monitoring and reporting",
            "Provide clear explanations for model decisions",
            "Establish human oversight and appeal processes",
            "Document model limitations and appropriate use cases",
            "Conduct regular stakeholder consultations on fairness"
        ]
        
        if bias_results.get("bias_detected", False):
            recommendations.extend([
                "Immediate bias mitigation required",
                "Consider model retraining with fairness constraints",
                "Implement additional monitoring for affected groups"
            ])
        
        return recommendations
    
    def _identify_model_limitations(self,
                                  model_info: Dict[str, Any],
                                  bias_results: Dict[str, Any]) -> List[str]:
        """Identify and document model limitations"""
        limitations = [
            "Model performance may vary across different populations",
            "Predictions are based on historical data patterns",
            "Model may not generalize to significantly different contexts",
            "Regular retraining required to maintain performance"
        ]
        
        if bias_results.get("bias_detected", False):
            limitations.append("Model shows bias against certain demographic groups")
        
        # Add feature-specific limitations
        features = model_info.get("features", [])
        if len(features) < 5:
            limitations.append("Limited feature set may affect prediction accuracy")
        
        return limitations
    
    def _create_monitoring_plan(self) -> Dict[str, Any]:
        """Create ongoing monitoring plan"""
        return {
            "bias_monitoring": {
                "frequency": "monthly",
                "metrics": ["demographic_parity", "equalized_odds"],
                "alert_thresholds": self.fairness_thresholds
            },
            "performance_monitoring": {
                "frequency": "weekly",
                "metrics": ["accuracy", "precision", "recall", "f1_score"]
            },
            "compliance_review": {
                "frequency": "quarterly",
                "frameworks": ["GDPR", "NIST_AI_RMF"]
            },
            "stakeholder_review": {
                "frequency": "semi_annually",
                "participants": ["data_scientists", "legal_team", "affected_communities"]
            }
        }
    
    async def check_regulatory_compliance(self,
                                        framework: ComplianceFramework,
                                        model_info: Dict[str, Any],
                                        bias_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance with specific regulatory framework"""
        try:
            if framework not in self.compliance_checks:
                return {
                    "status": "error",
                    "message": f"Compliance framework {framework} not supported"
                }
            
            compliance_result = await self.compliance_checks[framework](model_info, bias_results)
            
            # Log to audit trail
            self._log_audit_event("compliance_check", {
                "framework": framework.value,
                "compliant": compliance_result.get("compliant", False),
                "issues_found": len(compliance_result.get("issues", []))
            })
            
            return {"status": "success", "compliance": compliance_result}
            
        except Exception as e:
            self.logger.error(f"Failed to check {framework} compliance: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _check_gdpr_compliance(self,
                                   model_info: Dict[str, Any],
                                   bias_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check GDPR compliance"""
        compliance = {
            "framework": "GDPR",
            "compliant": True,
            "issues": [],
            "recommendations": []
        }
        
        # Check for automated decision-making
        if model_info.get("automated_decisions", True):
            compliance["issues"].append("Automated decision-making requires explicit consent")
            compliance["recommendations"].append("Implement consent mechanism for automated decisions")
        
        # Check for bias (fairness requirement)
        if bias_results.get("bias_detected", False):
            compliance["issues"].append("Bias detected may violate non-discrimination principles")
            compliance["recommendations"].append("Implement bias mitigation measures")
        
        # Check for transparency
        if not model_info.get("explainable", False):
            compliance["issues"].append("Lack of explainability may violate transparency requirements")
            compliance["recommendations"].append("Implement model explanation capabilities")
        
        compliance["compliant"] = len(compliance["issues"]) == 0
        
        return compliance
    
    async def _check_nist_compliance(self,
                                   model_info: Dict[str, Any],
                                   bias_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check NIST AI Risk Management Framework compliance"""
        compliance = {
            "framework": "NIST AI RMF",
            "compliant": True,
            "issues": [],
            "recommendations": []
        }
        
        # Check risk management
        if not model_info.get("risk_assessment", False):
            compliance["issues"].append("Missing comprehensive risk assessment")
            compliance["recommendations"].append("Conduct thorough AI risk assessment")
        
        # Check bias and fairness
        if bias_results.get("bias_detected", False):
            compliance["issues"].append("Bias detected requires mitigation")
            compliance["recommendations"].append("Implement fairness-aware ML techniques")
        
        # Check monitoring
        if not model_info.get("monitoring_plan", False):
            compliance["issues"].append("Missing ongoing monitoring plan")
            compliance["recommendations"].append("Establish continuous monitoring procedures")
        
        compliance["compliant"] = len(compliance["issues"]) == 0
        
        return compliance
    
    async def _check_eu_ai_act_compliance(self,
                                        model_info: Dict[str, Any],
                                        bias_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check EU AI Act compliance"""
        compliance = {
            "framework": "EU AI Act",
            "compliant": True,
            "issues": [],
            "recommendations": []
        }
        
        # Determine risk category
        risk_category = model_info.get("risk_category", "limited")
        
        if risk_category in ["high", "unacceptable"]:
            # High-risk AI systems requirements
            if bias_results.get("bias_detected", False):
                compliance["issues"].append("High-risk AI system shows bias")
                compliance["recommendations"].append("Mandatory bias testing and mitigation")
            
            if not model_info.get("human_oversight", False):
                compliance["issues"].append("High-risk AI system requires human oversight")
                compliance["recommendations"].append("Implement human oversight mechanisms")
            
            if not model_info.get("documentation", False):
                compliance["issues"].append("High-risk AI system requires comprehensive documentation")
                compliance["recommendations"].append("Create detailed technical documentation")
        
        compliance["compliant"] = len(compliance["issues"]) == 0
        
        return compliance
    
    def _log_audit_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log event to audit trail"""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "details": details,
            "engine_id": self.engine_id
        }
        
        self.audit_trail.append(audit_entry)
        
        # Keep only last 1000 entries to prevent memory issues
        if len(self.audit_trail) > 1000:
            self.audit_trail = self.audit_trail[-1000:]
    
    async def get_audit_trail(self,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None,
                            event_type: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve audit trail with optional filtering"""
        try:
            filtered_trail = self.audit_trail.copy()
            
            # Filter by date range
            if start_date:
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                filtered_trail = [
                    entry for entry in filtered_trail
                    if datetime.fromisoformat(entry["timestamp"].replace('Z', '+00:00')) >= start_dt
                ]
            
            if end_date:
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                filtered_trail = [
                    entry for entry in filtered_trail
                    if datetime.fromisoformat(entry["timestamp"].replace('Z', '+00:00')) <= end_dt
                ]
            
            # Filter by event type
            if event_type:
                filtered_trail = [
                    entry for entry in filtered_trail
                    if entry["event_type"] == event_type
                ]
            
            return {
                "status": "success",
                "audit_trail": filtered_trail,
                "total_entries": len(filtered_trail),
                "filters_applied": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "event_type": event_type
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve audit trail: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def get_ethical_guidelines(self) -> Dict[str, Any]:
        """Get ethical guidelines and principles"""
        return {
            "status": "success",
            "ethical_principles": self.ethical_principles,
            "fairness_thresholds": {
                metric.value: threshold 
                for metric, threshold in self.fairness_thresholds.items()
            },
            "supported_bias_types": [bias_type.value for bias_type in BiasType],
            "supported_metrics": [metric.value for metric in FairnessMetric],
            "compliance_frameworks": [framework.value for framework in ComplianceFramework]
        }
    
    async def update_fairness_thresholds(self, new_thresholds: Dict[str, float]) -> Dict[str, Any]:
        """Update fairness thresholds"""
        try:
            updated_thresholds = {}
            
            for metric_name, threshold in new_thresholds.items():
                try:
                    metric = FairnessMetric(metric_name)
                    if 0 <= threshold <= 1:
                        self.fairness_thresholds[metric] = threshold
                        updated_thresholds[metric_name] = threshold
                    else:
                        return {
                            "status": "error",
                            "message": f"Threshold for {metric_name} must be between 0 and 1"
                        }
                except ValueError:
                    return {
                        "status": "error",
                        "message": f"Unknown fairness metric: {metric_name}"
                    }
            
            # Log to audit trail
            self._log_audit_event("thresholds_updated", {
                "updated_thresholds": updated_thresholds
            })
            
            return {
                "status": "success",
                "message": "Fairness thresholds updated successfully",
                "updated_thresholds": updated_thresholds
            }
            
        except Exception as e:
            self.logger.error(f"Failed to update fairness thresholds: {str(e)}")
            return {"status": "error", "message": str(e)}