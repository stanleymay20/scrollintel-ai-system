"""
A/B Testing Framework for Feature Experiments
Implements experiment management, user assignment, and statistical analysis
"""

import json
import uuid
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import asyncio
import logging
from enum import Enum
import math

logger = logging.getLogger(__name__)

class ExperimentStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class VariantType(Enum):
    CONTROL = "control"
    TREATMENT = "treatment"

@dataclass
class ExperimentVariant:
    variant_id: str
    name: str
    description: str
    variant_type: VariantType
    traffic_allocation: float  # Percentage of traffic (0-100)
    configuration: Dict[str, Any]
    is_active: bool = True

@dataclass
class Experiment:
    experiment_id: str
    name: str
    description: str
    hypothesis: str
    success_metrics: List[str]
    variants: List[ExperimentVariant]
    status: ExperimentStatus
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    target_sample_size: int
    confidence_level: float
    created_by: str
    created_at: datetime
    metadata: Dict[str, Any]

@dataclass
class UserAssignment:
    assignment_id: str
    user_id: str
    experiment_id: str
    variant_id: str
    assigned_at: datetime
    session_id: str
    user_properties: Dict[str, Any]

@dataclass
class ExperimentResult:
    result_id: str
    experiment_id: str
    variant_id: str
    metric_name: str
    metric_value: float
    user_count: int
    conversion_rate: Optional[float]
    recorded_at: datetime

@dataclass
class StatisticalAnalysis:
    experiment_id: str
    analysis_id: str
    metric_name: str
    control_mean: float
    treatment_mean: float
    control_std: float
    treatment_std: float
    control_count: int
    treatment_count: int
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    is_significant: bool
    power: float
    recommendation: str
    analyzed_at: datetime

class ABTestingFramework:
    """A/B Testing framework for feature experiments"""
    
    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
        self.user_assignments: Dict[str, UserAssignment] = {}
        self.experiment_results: List[ExperimentResult] = []
        self.statistical_analyses: List[StatisticalAnalysis] = []
        
        # Configuration
        self.default_confidence_level = 0.95
        self.minimum_sample_size = 100
        self.maximum_experiment_duration = 30  # days
    
    async def create_experiment(self,
                               name: str,
                               description: str,
                               hypothesis: str,
                               success_metrics: List[str],
                               variants: List[Dict[str, Any]],
                               target_sample_size: int = 1000,
                               confidence_level: float = 0.95,
                               created_by: str = "system") -> str:
        """Create a new A/B test experiment"""
        try:
            experiment_id = str(uuid.uuid4())
            
            # Create experiment variants
            experiment_variants = []
            total_allocation = 0
            
            for variant_data in variants:
                variant = ExperimentVariant(
                    variant_id=str(uuid.uuid4()),
                    name=variant_data["name"],
                    description=variant_data.get("description", ""),
                    variant_type=VariantType(variant_data.get("type", "treatment")),
                    traffic_allocation=variant_data.get("traffic_allocation", 50.0),
                    configuration=variant_data.get("configuration", {}),
                    is_active=True
                )
                experiment_variants.append(variant)
                total_allocation += variant.traffic_allocation
            
            # Validate traffic allocation
            if abs(total_allocation - 100.0) > 0.1:
                raise ValueError(f"Traffic allocation must sum to 100%, got {total_allocation}%")
            
            # Create experiment
            experiment = Experiment(
                experiment_id=experiment_id,
                name=name,
                description=description,
                hypothesis=hypothesis,
                success_metrics=success_metrics,
                variants=experiment_variants,
                status=ExperimentStatus.DRAFT,
                start_date=None,
                end_date=None,
                target_sample_size=target_sample_size,
                confidence_level=confidence_level,
                created_by=created_by,
                created_at=datetime.utcnow(),
                metadata={}
            )
            
            self.experiments[experiment_id] = experiment
            logger.info(f"Created experiment: {name} with {len(variants)} variants")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Error creating experiment: {str(e)}")
            raise
    
    async def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment"""
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = self.experiments[experiment_id]
            
            if experiment.status != ExperimentStatus.DRAFT:
                raise ValueError(f"Can only start experiments in DRAFT status, current: {experiment.status}")
            
            # Validate experiment setup
            if not experiment.variants:
                raise ValueError("Experiment must have at least one variant")
            
            if not experiment.success_metrics:
                raise ValueError("Experiment must have at least one success metric")
            
            # Start experiment
            experiment.status = ExperimentStatus.RUNNING
            experiment.start_date = datetime.utcnow()
            experiment.end_date = datetime.utcnow() + timedelta(days=self.maximum_experiment_duration)
            
            logger.info(f"Started experiment: {experiment.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting experiment: {str(e)}")
            raise
    
    async def assign_user_to_experiment(self,
                                       user_id: str,
                                       experiment_id: str,
                                       session_id: str,
                                       user_properties: Dict[str, Any] = None) -> Optional[str]:
        """Assign user to experiment variant"""
        try:
            if experiment_id not in self.experiments:
                return None
            
            experiment = self.experiments[experiment_id]
            
            if experiment.status != ExperimentStatus.RUNNING:
                return None
            
            # Check if user already assigned
            assignment_key = f"{user_id}_{experiment_id}"
            if assignment_key in self.user_assignments:
                return self.user_assignments[assignment_key].variant_id
            
            # Assign user to variant based on traffic allocation
            variant = self._select_variant_for_user(user_id, experiment.variants)
            
            if not variant:
                return None
            
            # Create assignment
            assignment = UserAssignment(
                assignment_id=str(uuid.uuid4()),
                user_id=user_id,
                experiment_id=experiment_id,
                variant_id=variant.variant_id,
                assigned_at=datetime.utcnow(),
                session_id=session_id,
                user_properties=user_properties or {}
            )
            
            self.user_assignments[assignment_key] = assignment
            logger.info(f"Assigned user {user_id} to variant {variant.name} in experiment {experiment.name}")
            return variant.variant_id
            
        except Exception as e:
            logger.error(f"Error assigning user to experiment: {str(e)}")
            return None
    
    def _select_variant_for_user(self, user_id: str, variants: List[ExperimentVariant]) -> Optional[ExperimentVariant]:
        """Select variant for user based on traffic allocation"""
        # Use user ID as seed for consistent assignment
        random.seed(hash(user_id))
        
        # Create cumulative distribution
        cumulative_allocation = 0
        allocation_ranges = []
        
        for variant in variants:
            if variant.is_active:
                start = cumulative_allocation
                end = cumulative_allocation + variant.traffic_allocation
                allocation_ranges.append((start, end, variant))
                cumulative_allocation = end
        
        # Select variant based on random number
        rand_value = random.uniform(0, 100)
        
        for start, end, variant in allocation_ranges:
            if start <= rand_value < end:
                return variant
        
        return None
    
    async def record_experiment_result(self,
                                      user_id: str,
                                      experiment_id: str,
                                      metric_name: str,
                                      metric_value: float) -> str:
        """Record experiment result for user"""
        try:
            # Get user assignment
            assignment_key = f"{user_id}_{experiment_id}"
            if assignment_key not in self.user_assignments:
                raise ValueError(f"User {user_id} not assigned to experiment {experiment_id}")
            
            assignment = self.user_assignments[assignment_key]
            
            # Record result
            result = ExperimentResult(
                result_id=str(uuid.uuid4()),
                experiment_id=experiment_id,
                variant_id=assignment.variant_id,
                metric_name=metric_name,
                metric_value=metric_value,
                user_count=1,
                conversion_rate=metric_value if metric_name.endswith("_rate") else None,
                recorded_at=datetime.utcnow()
            )
            
            self.experiment_results.append(result)
            logger.info(f"Recorded result for experiment {experiment_id}: {metric_name}={metric_value}")
            return result.result_id
            
        except Exception as e:
            logger.error(f"Error recording experiment result: {str(e)}")
            raise
    
    async def analyze_experiment(self, experiment_id: str) -> Dict[str, StatisticalAnalysis]:
        """Perform statistical analysis of experiment results"""
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = self.experiments[experiment_id]
            analyses = {}
            
            # Get experiment results
            experiment_results = [
                r for r in self.experiment_results
                if r.experiment_id == experiment_id
            ]
            
            if not experiment_results:
                return {}
            
            # Analyze each success metric
            for metric_name in experiment.success_metrics:
                metric_results = [r for r in experiment_results if r.metric_name == metric_name]
                
                if len(metric_results) < self.minimum_sample_size:
                    continue
                
                # Group results by variant
                variant_results = {}
                for result in metric_results:
                    if result.variant_id not in variant_results:
                        variant_results[result.variant_id] = []
                    variant_results[result.variant_id].append(result.metric_value)
                
                # Find control and treatment variants
                control_variant = next((v for v in experiment.variants if v.variant_type == VariantType.CONTROL), None)
                treatment_variants = [v for v in experiment.variants if v.variant_type == VariantType.TREATMENT]
                
                if not control_variant or not treatment_variants:
                    continue
                
                control_values = variant_results.get(control_variant.variant_id, [])
                
                # Analyze each treatment vs control
                for treatment_variant in treatment_variants:
                    treatment_values = variant_results.get(treatment_variant.variant_id, [])
                    
                    if len(control_values) < 10 or len(treatment_values) < 10:
                        continue
                    
                    analysis = await self._perform_statistical_test(
                        experiment_id, metric_name, control_values, treatment_values
                    )
                    
                    analyses[f"{metric_name}_{treatment_variant.variant_id}"] = analysis
            
            return analyses
            
        except Exception as e:
            logger.error(f"Error analyzing experiment: {str(e)}")
            raise
    
    async def _perform_statistical_test(self,
                                       experiment_id: str,
                                       metric_name: str,
                                       control_values: List[float],
                                       treatment_values: List[float]) -> StatisticalAnalysis:
        """Perform statistical test (t-test) on experiment data"""
        try:
            # Calculate basic statistics
            control_mean = sum(control_values) / len(control_values)
            treatment_mean = sum(treatment_values) / len(treatment_values)
            
            control_variance = sum((x - control_mean) ** 2 for x in control_values) / (len(control_values) - 1)
            treatment_variance = sum((x - treatment_mean) ** 2 for x in treatment_values) / (len(treatment_values) - 1)
            
            control_std = math.sqrt(control_variance)
            treatment_std = math.sqrt(treatment_variance)
            
            # Perform two-sample t-test
            pooled_std = math.sqrt(
                ((len(control_values) - 1) * control_variance + (len(treatment_values) - 1) * treatment_variance) /
                (len(control_values) + len(treatment_values) - 2)
            )
            
            standard_error = pooled_std * math.sqrt(1/len(control_values) + 1/len(treatment_values))
            t_statistic = (treatment_mean - control_mean) / standard_error if standard_error > 0 else 0
            
            # Calculate degrees of freedom
            df = len(control_values) + len(treatment_values) - 2
            
            # Approximate p-value (simplified)
            p_value = self._approximate_p_value(abs(t_statistic), df)
            
            # Calculate effect size (Cohen's d)
            effect_size = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
            
            # Calculate confidence interval
            margin_of_error = 1.96 * standard_error  # 95% CI
            ci_lower = (treatment_mean - control_mean) - margin_of_error
            ci_upper = (treatment_mean - control_mean) + margin_of_error
            
            # Determine significance
            is_significant = p_value < (1 - self.default_confidence_level)
            
            # Calculate statistical power (simplified)
            power = self._calculate_power(effect_size, len(control_values), len(treatment_values))
            
            # Generate recommendation
            recommendation = self._generate_recommendation(
                is_significant, effect_size, power, treatment_mean, control_mean
            )
            
            analysis = StatisticalAnalysis(
                experiment_id=experiment_id,
                analysis_id=str(uuid.uuid4()),
                metric_name=metric_name,
                control_mean=control_mean,
                treatment_mean=treatment_mean,
                control_std=control_std,
                treatment_std=treatment_std,
                control_count=len(control_values),
                treatment_count=len(treatment_values),
                p_value=p_value,
                confidence_interval=(ci_lower, ci_upper),
                effect_size=effect_size,
                is_significant=is_significant,
                power=power,
                recommendation=recommendation,
                analyzed_at=datetime.utcnow()
            )
            
            self.statistical_analyses.append(analysis)
            return analysis
            
        except Exception as e:
            logger.error(f"Error performing statistical test: {str(e)}")
            raise
    
    def _approximate_p_value(self, t_stat: float, df: int) -> float:
        """Approximate p-value for t-test (simplified)"""
        # This is a very simplified approximation
        # In production, use scipy.stats.t.sf or similar
        if t_stat < 1.0:
            return 0.5
        elif t_stat < 1.96:
            return 0.1
        elif t_stat < 2.58:
            return 0.05
        else:
            return 0.01
    
    def _calculate_power(self, effect_size: float, n1: int, n2: int) -> float:
        """Calculate statistical power (simplified)"""
        # Simplified power calculation
        total_n = n1 + n2
        if total_n < 100:
            return 0.5
        elif abs(effect_size) > 0.5 and total_n > 200:
            return 0.8
        elif abs(effect_size) > 0.3 and total_n > 500:
            return 0.8
        else:
            return 0.6
    
    def _generate_recommendation(self,
                                is_significant: bool,
                                effect_size: float,
                                power: float,
                                treatment_mean: float,
                                control_mean: float) -> str:
        """Generate recommendation based on analysis"""
        if not is_significant:
            if power < 0.8:
                return "No significant difference detected. Consider increasing sample size for higher statistical power."
            else:
                return "No significant difference detected with adequate power. Treatment likely has no meaningful effect."
        
        improvement = ((treatment_mean - control_mean) / control_mean * 100) if control_mean != 0 else 0
        
        if abs(effect_size) < 0.2:
            return f"Statistically significant but small effect size ({effect_size:.3f}). Consider practical significance."
        elif abs(effect_size) < 0.5:
            return f"Moderate effect detected ({improvement:+.1f}% change). Consider implementing if cost-effective."
        else:
            return f"Large effect detected ({improvement:+.1f}% change). Strong candidate for implementation."
    
    async def get_experiment_dashboard(self) -> Dict[str, Any]:
        """Get experiment dashboard data"""
        try:
            running_experiments = [e for e in self.experiments.values() if e.status == ExperimentStatus.RUNNING]
            completed_experiments = [e for e in self.experiments.values() if e.status == ExperimentStatus.COMPLETED]
            
            # Calculate summary metrics
            total_users_in_experiments = len(set(a.user_id for a in self.user_assignments.values()))
            
            # Get recent results
            recent_results = [
                r for r in self.experiment_results
                if r.recorded_at >= datetime.utcnow() - timedelta(days=7)
            ]
            
            return {
                "total_experiments": len(self.experiments),
                "running_experiments": len(running_experiments),
                "completed_experiments": len(completed_experiments),
                "total_users_in_experiments": total_users_in_experiments,
                "recent_results_count": len(recent_results),
                "experiments": [
                    {
                        "experiment_id": exp.experiment_id,
                        "name": exp.name,
                        "status": exp.status.value,
                        "variants_count": len(exp.variants),
                        "start_date": exp.start_date.isoformat() if exp.start_date else None,
                        "target_sample_size": exp.target_sample_size,
                        "current_sample_size": len([a for a in self.user_assignments.values() if a.experiment_id == exp.experiment_id])
                    }
                    for exp in self.experiments.values()
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting experiment dashboard: {str(e)}")
            raise

# Global A/B testing framework instance
ab_testing_framework = ABTestingFramework()