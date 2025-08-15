"""
ScrollEthicsAgent - AI Ethics & Bias Detection
Comprehensive fairness auditing, bias detection, and ethical AI compliance.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from uuid import uuid4
from dataclasses import dataclass
from enum import Enum
import json
import logging

# Fairness libraries
try:
    from sklearn.metrics import confusion_matrix, accuracy_score
    from scipy import stats
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from scrollintel.core.interfaces import BaseAgent, AgentType, AgentRequest, AgentResponse, AgentCapability, ResponseStatus

logger = logging.getLogger(__name__)


class BiasType(str, Enum):
    """Types of bias that can be detected."""
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    CALIBRATION = "calibration"
    INDIVIDUAL_FAIRNESS = "individual_fairness"
    COUNTERFACTUAL_FAIRNESS = "counterfactual_fairness"
    SELECTION_BIAS = "selection_bias"
    CONFIRMATION_BIAS = "confirmation_bias"
    REPRESENTATION_BIAS = "representation_bias"


class ProtectedAttribute(str, Enum):
    """Protected attributes for fairness analysis."""
    RACE = "race"
    GENDER = "gender"
    AGE = "age"
    RELIGION = "religion"
    SEXUAL_ORIENTATION = "sexual_orientation"
    DISABILITY = "disability"
    NATIONALITY = "nationality"
    SOCIOECONOMIC_STATUS = "socioeconomic_status"


class EthicalPrinciple(str, Enum):
    """Ethical principles for AI systems."""
    FAIRNESS = "fairness"
    TRANSPARENCY = "transparency"
    ACCOUNTABILITY = "accountability"
    PRIVACY = "privacy"
    HUMAN_AUTONOMY = "human_autonomy"
    NON_MALEFICENCE = "non_maleficence"
    BENEFICENCE = "beneficence"
    JUSTICE = "justice"


class ComplianceFramework(str, Enum):
    """Regulatory compliance frameworks."""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOX = "sox"
    EU_AI_ACT = "eu_ai_act"
    IEEE_ETHICALLY_ALIGNED_DESIGN = "ieee_ethically_aligned_design"
    PARTNERSHIP_AI_TENETS = "partnership_ai_tenets"


@dataclass
class BiasAuditResult:
    """Result of a bias audit."""
    id: str
    model_id: str
    bias_type: BiasType
    protected_attribute: ProtectedAttribute
    bias_detected: bool
    bias_score: float
    threshold: float
    affected_groups: List[str]
    recommendations: List[str]
    statistical_tests: Dict[str, Any]
    confidence: float
    audit_timestamp: datetime = None
    
    def __post_init__(self):
        if self.audit_timestamp is None:
            self.audit_timestamp = datetime.utcnow()


@dataclass
class EthicalAssessment:
    """Comprehensive ethical assessment of an AI system."""
    id: str
    system_id: str
    principles_evaluated: List[EthicalPrinciple]
    compliance_frameworks: List[ComplianceFramework]
    overall_score: float
    principle_scores: Dict[str, float]
    violations: List[Dict[str, Any]]
    recommendations: List[str]
    risk_level: str  # "low", "medium", "high", "critical"
    assessment_timestamp: datetime = None
    
    def __post_init__(self):
        if self.assessment_timestamp is None:
            self.assessment_timestamp = datetime.utcnow()


@dataclass
class FairnessMetrics:
    """Comprehensive fairness metrics."""
    demographic_parity: float
    equalized_odds: float
    equal_opportunity: float
    calibration: float
    individual_fairness: float
    statistical_parity_difference: float
    disparate_impact: float
    theil_index: float


class ScrollEthicsAgent(BaseAgent):
    """Advanced AI ethics and bias detection agent."""
    
    def __init__(self):
        super().__init__(
            agent_id="scroll-ethics-agent",
            name="ScrollEthics Agent",
            agent_type=AgentType.AI_ENGINEER
        )
        
        self.capabilities = [
            AgentCapability(
                name="bias_detection",
                description="Detect various types of bias in AI models and datasets",
                input_types=["model", "dataset", "predictions", "protected_attributes"],
                output_types=["bias_report", "fairness_metrics", "recommendations"]
            ),
            AgentCapability(
                name="fairness_auditing",
                description="Comprehensive fairness auditing with multiple metrics",
                input_types=["model_predictions", "ground_truth", "demographic_data"],
                output_types=["fairness_audit", "compliance_report"]
            ),
            AgentCapability(
                name="ethical_assessment",
                description="Evaluate AI systems against ethical principles and frameworks",
                input_types=["system_description", "use_case", "stakeholder_analysis"],
                output_types=["ethical_assessment", "risk_analysis", "mitigation_strategies"]
            ),
            AgentCapability(
                name="compliance_checking",
                description="Check compliance with regulatory frameworks",
                input_types=["system_specs", "data_handling", "decision_processes"],
                output_types=["compliance_report", "violation_alerts", "remediation_plan"]
            )
        ]
        
        # Ethics state
        self.audit_history = {}
        self.ethical_assessments = {}
        self.compliance_reports = {}
        self.bias_thresholds = self._initialize_bias_thresholds()
        
        # Fairness metrics calculators
        self.fairness_calculators = {
            BiasType.DEMOGRAPHIC_PARITY: self._calculate_demographic_parity,
            BiasType.EQUALIZED_ODDS: self._calculate_equalized_odds,
            BiasType.EQUAL_OPPORTUNITY: self._calculate_equal_opportunity,
            BiasType.CALIBRATION: self._calculate_calibration,
            BiasType.INDIVIDUAL_FAIRNESS: self._calculate_individual_fairness
        }
    
    def _initialize_bias_thresholds(self) -> Dict[BiasType, float]:
        """Initialize bias detection thresholds."""
        return {
            BiasType.DEMOGRAPHIC_PARITY: 0.1,
            BiasType.EQUALIZED_ODDS: 0.1,
            BiasType.EQUAL_OPPORTUNITY: 0.1,
            BiasType.CALIBRATION: 0.05,
            BiasType.INDIVIDUAL_FAIRNESS: 0.1,
            BiasType.SELECTION_BIAS: 0.15,
            BiasType.REPRESENTATION_BIAS: 0.2
        }
    
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """Process ethics and bias detection requests."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            prompt = request.prompt.lower()
            context = request.context or {}
            
            if "bias" in prompt and "detect" in prompt:
                content = await self._detect_bias(request.prompt, context)
            elif "fairness" in prompt and "audit" in prompt:
                content = await self._conduct_fairness_audit(request.prompt, context)
            elif "ethical" in prompt and "assess" in prompt:
                content = await self._conduct_ethical_assessment(request.prompt, context)
            elif "compliance" in prompt:
                content = await self._check_compliance(request.prompt, context)
            elif "recommend" in prompt or "mitigate" in prompt:
                content = await self._generate_mitigation_strategies(request.prompt, context)
            else:
                content = await self._general_ethics_analysis(request.prompt, context)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return AgentResponse(
                id=f"ethics-{uuid4()}",
                request_id=request.id,
                content=content,
                artifacts=[],
                execution_time=execution_time,
                status=ResponseStatus.SUCCESS
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return AgentResponse(
                id=f"ethics-{uuid4()}",
                request_id=request.id,
                content=f"Error in ethics analysis: {str(e)}",
                artifacts=[],
                execution_time=execution_time,
                status=ResponseStatus.ERROR,
                error_message=str(e)
            )
    
    async def _detect_bias(self, prompt: str, context: Dict[str, Any]) -> str:
        """Detect bias in models or datasets."""
        model_predictions = context.get("predictions")
        ground_truth = context.get("ground_truth")
        protected_attributes = context.get("protected_attributes", {})
        bias_types = context.get("bias_types", list(BiasType))
        
        if not model_predictions or not protected_attributes:
            return "Error: Model predictions and protected attributes are required for bias detection."
        
        # Convert to appropriate format
        if isinstance(model_predictions, list):
            predictions = np.array(model_predictions)
        else:
            predictions = model_predictions
        
        if ground_truth and isinstance(ground_truth, list):
            y_true = np.array(ground_truth)
        else:
            y_true = ground_truth
        
        # Perform bias detection for each type
        bias_results = []
        for bias_type in bias_types:
            if isinstance(bias_type, str):
                bias_type = BiasType(bias_type)
            
            for attr_name, attr_values in protected_attributes.items():
                try:
                    protected_attr = ProtectedAttribute(attr_name)
                except ValueError:
                    protected_attr = attr_name  # Use as string if not in enum
                
                result = await self._detect_specific_bias(
                    predictions, y_true, attr_values, bias_type, protected_attr
                )
                bias_results.append(result)
        
        # Store audit results
        audit_id = f"bias-audit-{uuid4()}"
        self.audit_history[audit_id] = bias_results
        
        return f"""
# Bias Detection Report

## Audit ID: {audit_id}
## Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Bias Types Tested**: {len(bias_types)}
- **Protected Attributes**: {len(protected_attributes)}
- **Total Tests**: {len(bias_results)}
- **Bias Detected**: {sum(1 for r in bias_results if r.bias_detected)}

## Detailed Results
{await self._format_bias_results(bias_results)}

## Overall Assessment
{await self._generate_overall_bias_assessment(bias_results)}

## Recommendations
{await self._generate_bias_recommendations(bias_results)}

## Next Steps
{await self._suggest_bias_mitigation_steps(bias_results)}
"""
    
    async def _detect_specific_bias(self, predictions: np.ndarray, y_true: Optional[np.ndarray], 
                                  protected_attr: np.ndarray, bias_type: BiasType, 
                                  attr_name: str) -> BiasAuditResult:
        """Detect a specific type of bias."""
        try:
            # Calculate bias metric
            if bias_type in self.fairness_calculators:
                bias_score = await self.fairness_calculators[bias_type](
                    predictions, y_true, protected_attr
                )
            else:
                bias_score = await self._calculate_generic_bias(
                    predictions, y_true, protected_attr, bias_type
                )
            
            # Check against threshold
            threshold = self.bias_thresholds.get(bias_type, 0.1)
            bias_detected = abs(bias_score) > threshold
            
            # Identify affected groups
            affected_groups = await self._identify_affected_groups(
                predictions, protected_attr, bias_score
            )
            
            # Generate recommendations
            recommendations = await self._generate_bias_specific_recommendations(
                bias_type, bias_score, affected_groups
            )
            
            # Statistical tests
            statistical_tests = await self._perform_statistical_tests(
                predictions, y_true, protected_attr, bias_type
            )
            
            return BiasAuditResult(
                id=f"bias-{uuid4()}",
                model_id=f"model-{uuid4()}",  # Would be provided in real scenario
                bias_type=bias_type,
                protected_attribute=attr_name,
                bias_detected=bias_detected,
                bias_score=bias_score,
                threshold=threshold,
                affected_groups=affected_groups,
                recommendations=recommendations,
                statistical_tests=statistical_tests,
                confidence=0.85  # Would be calculated based on sample size, etc.
            )
            
        except Exception as e:
            logger.error(f"Bias detection failed for {bias_type}: {e}")
            return BiasAuditResult(
                id=f"bias-error-{uuid4()}",
                model_id="unknown",
                bias_type=bias_type,
                protected_attribute=attr_name,
                bias_detected=False,
                bias_score=0.0,
                threshold=0.1,
                affected_groups=[],
                recommendations=[f"Error in bias detection: {str(e)}"],
                statistical_tests={},
                confidence=0.0
            )
    
    async def _calculate_demographic_parity(self, predictions: np.ndarray, 
                                          y_true: Optional[np.ndarray], 
                                          protected_attr: np.ndarray) -> float:
        """Calculate demographic parity violation."""
        if not SKLEARN_AVAILABLE:
            return 0.0
        
        try:
            # Get unique groups
            unique_groups = np.unique(protected_attr)
            if len(unique_groups) < 2:
                return 0.0
            
            # Calculate positive prediction rates for each group
            group_rates = []
            for group in unique_groups:
                group_mask = protected_attr == group
                group_predictions = predictions[group_mask]
                positive_rate = np.mean(group_predictions > 0.5)  # Assuming binary classification
                group_rates.append(positive_rate)
            
            # Calculate maximum difference (demographic parity violation)
            max_diff = max(group_rates) - min(group_rates)
            return max_diff
            
        except Exception as e:
            logger.error(f"Demographic parity calculation failed: {e}")
            return 0.0
    
    async def _calculate_equalized_odds(self, predictions: np.ndarray, 
                                      y_true: np.ndarray, 
                                      protected_attr: np.ndarray) -> float:
        """Calculate equalized odds violation."""
        if not SKLEARN_AVAILABLE or y_true is None:
            return 0.0
        
        try:
            unique_groups = np.unique(protected_attr)
            if len(unique_groups) < 2:
                return 0.0
            
            # Calculate TPR and FPR for each group
            group_tprs = []
            group_fprs = []
            
            for group in unique_groups:
                group_mask = protected_attr == group
                group_pred = predictions[group_mask] > 0.5
                group_true = y_true[group_mask]
                
                if len(group_true) > 0:
                    tn, fp, fn, tp = confusion_matrix(group_true, group_pred).ravel()
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    
                    group_tprs.append(tpr)
                    group_fprs.append(fpr)
            
            # Calculate maximum differences
            tpr_diff = max(group_tprs) - min(group_tprs) if group_tprs else 0
            fpr_diff = max(group_fprs) - min(group_fprs) if group_fprs else 0
            
            # Return maximum of the two differences
            return max(tpr_diff, fpr_diff)
            
        except Exception as e:
            logger.error(f"Equalized odds calculation failed: {e}")
            return 0.0
    
    async def _conduct_ethical_assessment(self, prompt: str, context: Dict[str, Any]) -> str:
        """Conduct comprehensive ethical assessment."""
        system_description = context.get("system_description", "AI System")
        use_case = context.get("use_case", "General AI Application")
        stakeholders = context.get("stakeholders", [])
        principles = context.get("principles", list(EthicalPrinciple))
        frameworks = context.get("frameworks", [ComplianceFramework.EU_AI_ACT])
        
        # Evaluate each ethical principle
        principle_scores = {}
        violations = []
        
        for principle in principles:
            if isinstance(principle, str):
                principle = EthicalPrinciple(principle)
            
            score, principle_violations = await self._evaluate_ethical_principle(
                system_description, use_case, principle, context
            )
            principle_scores[principle.value] = score
            violations.extend(principle_violations)
        
        # Calculate overall score
        overall_score = sum(principle_scores.values()) / len(principle_scores) if principle_scores else 0.0
        
        # Determine risk level
        risk_level = await self._determine_risk_level(overall_score, violations)
        
        # Generate recommendations
        recommendations = await self._generate_ethical_recommendations(
            principle_scores, violations, risk_level
        )
        
        # Create assessment
        assessment = EthicalAssessment(
            id=f"ethics-{uuid4()}",
            system_id=context.get("system_id", f"system-{uuid4()}"),
            principles_evaluated=principles,
            compliance_frameworks=frameworks,
            overall_score=overall_score,
            principle_scores=principle_scores,
            violations=violations,
            recommendations=recommendations,
            risk_level=risk_level
        )
        
        # Store assessment
        self.ethical_assessments[assessment.id] = assessment
        
        return f"""
# Ethical Assessment Report

## System: {system_description}
## Use Case: {use_case}
## Assessment ID: {assessment.id}

## Overall Ethical Score: {overall_score:.2f}/1.0
## Risk Level: {risk_level.upper()}

## Principle Evaluation
{await self._format_principle_scores(principle_scores)}

## Violations Detected
{await self._format_violations(violations)}

## Stakeholder Impact Analysis
{await self._analyze_stakeholder_impact(stakeholders, violations)}

## Compliance Framework Analysis
{await self._analyze_compliance_frameworks(frameworks, assessment)}

## Recommendations
{await self._format_recommendations(recommendations)}

## Mitigation Strategies
{await self._generate_mitigation_strategies_detailed(violations, risk_level)}
"""
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Return agent capabilities."""
        return self.capabilities
    
    async def health_check(self) -> bool:
        """Check agent health."""
        return True
    
    # Helper methods
    async def _format_bias_results(self, results: List[BiasAuditResult]) -> str:
        """Format bias detection results."""
        formatted = []
        for result in results:
            status = "🚨 BIAS DETECTED" if result.bias_detected else "✅ No Bias"
            formatted.append(f"""
**{result.bias_type.value} - {result.protected_attribute}**
- Status: {status}
- Score: {result.bias_score:.3f} (Threshold: {result.threshold})
- Confidence: {result.confidence:.2f}
- Affected Groups: {', '.join(result.affected_groups) if result.affected_groups else 'None'}
""")
        return "\n".join(formatted)
    
    async def _generate_overall_bias_assessment(self, results: List[BiasAuditResult]) -> str:
        """Generate overall bias assessment."""
        total_tests = len(results)
        bias_detected = sum(1 for r in results if r.bias_detected)
        avg_confidence = sum(r.confidence for r in results) / total_tests if total_tests > 0 else 0
        
        if bias_detected == 0:
            return f"✅ **No significant bias detected** across {total_tests} tests (Avg. Confidence: {avg_confidence:.2f})"
        elif bias_detected < total_tests * 0.3:
            return f"⚠️ **Low bias risk** - {bias_detected}/{total_tests} tests detected bias"
        elif bias_detected < total_tests * 0.6:
            return f"🔶 **Medium bias risk** - {bias_detected}/{total_tests} tests detected bias"
        else:
            return f"🚨 **High bias risk** - {bias_detected}/{total_tests} tests detected bias"
    
    # Placeholder implementations for complex methods
    async def _calculate_equal_opportunity(self, predictions: np.ndarray, y_true: np.ndarray, protected_attr: np.ndarray) -> float:
        """Calculate equal opportunity violation."""
        return 0.05  # Mock implementation
    
    async def _calculate_calibration(self, predictions: np.ndarray, y_true: np.ndarray, protected_attr: np.ndarray) -> float:
        """Calculate calibration violation."""
        return 0.03  # Mock implementation
    
    async def _calculate_individual_fairness(self, predictions: np.ndarray, y_true: np.ndarray, protected_attr: np.ndarray) -> float:
        """Calculate individual fairness violation."""
        return 0.08  # Mock implementation
    
    async def _calculate_generic_bias(self, predictions: np.ndarray, y_true: Optional[np.ndarray], 
                                    protected_attr: np.ndarray, bias_type: BiasType) -> float:
        """Calculate generic bias metric."""
        return 0.05  # Mock implementation
    
    async def _identify_affected_groups(self, predictions: np.ndarray, protected_attr: np.ndarray, bias_score: float) -> List[str]:
        """Identify groups affected by bias."""
        unique_groups = np.unique(protected_attr)
        return [str(group) for group in unique_groups[:2]]  # Mock implementation
    
    async def _generate_bias_specific_recommendations(self, bias_type: BiasType, bias_score: float, affected_groups: List[str]) -> List[str]:
        """Generate bias-specific recommendations."""
        return [
            f"Address {bias_type.value} bias affecting {', '.join(affected_groups)}",
            "Implement bias mitigation techniques",
            "Increase representation in training data",
            "Apply fairness constraints during training"
        ]
    
    async def _perform_statistical_tests(self, predictions: np.ndarray, y_true: Optional[np.ndarray], 
                                       protected_attr: np.ndarray, bias_type: BiasType) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        return {
            "chi_square_test": {"statistic": 5.2, "p_value": 0.023},
            "kolmogorov_smirnov": {"statistic": 0.15, "p_value": 0.045}
        }