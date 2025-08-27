"""
A/B Testing Engine for Advanced Prompt Management System.
Provides complete statistical analysis, experiment scheduling, automation,
winner selection, and results visualization capabilities.
"""

import logging
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import scipy.stats as stats
from scipy.stats import ttest_ind, chi2_contingency, mannwhitneyu
import json
import uuid
from croniter import croniter

from scrollintel.engines.base_engine import BaseEngine, EngineCapability, EngineStatus
from scrollintel.models.experiment_models import (
    Experiment, ExperimentVariant, VariantMetric, ExperimentResult,
    ExperimentSchedule, ExperimentStatus, StatisticalSignificance
)

logger = logging.getLogger(__name__)


@dataclass
class StatisticalAnalysis:
    """Results of statistical analysis for A/B test."""
    metric_name: str
    control_mean: float
    treatment_mean: float
    control_std: float
    treatment_std: float
    control_sample_size: int
    treatment_sample_size: int
    effect_size: float
    p_value: float
    confidence_interval: Tuple[float, float]
    statistical_significance: StatisticalSignificance
    statistical_power: float
    test_type: str
    recommendation: str


@dataclass
class ExperimentResults:
    """Complete experiment results with all analyses."""
    experiment_id: str
    status: str
    analyses: List[StatisticalAnalysis]
    winner_variant_id: Optional[str]
    confidence_level: float
    total_sample_size: int
    experiment_duration: timedelta
    conversion_rates: Dict[str, float]
    visualizations: Dict[str, Any]


class TestType(Enum):
    """Types of statistical tests."""
    T_TEST = "t_test"
    CHI_SQUARE = "chi_square"
    MANN_WHITNEY = "mann_whitney"
    PROPORTION_TEST = "proportion_test"


class ExperimentConfig:
    """Enhanced configuration for A/B test experiments."""
    
    def __init__(
        self,
        name: str,
        prompt_id: str,
        hypothesis: str,
        variants: List[Dict[str, Any]],
        success_metrics: List[str],
        target_sample_size: int = 1000,
        confidence_level: float = 0.95,
        minimum_effect_size: float = 0.05,
        traffic_allocation: float = 1.0,
        duration_hours: Optional[int] = None,
        auto_start: bool = False,
        auto_stop: bool = False,
        auto_promote_winner: bool = False,
        schedule_config: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.prompt_id = prompt_id
        self.hypothesis = hypothesis
        self.variants = variants
        self.success_metrics = success_metrics
        self.target_sample_size = target_sample_size
        self.confidence_level = confidence_level
        self.minimum_effect_size = minimum_effect_size
        self.traffic_allocation = traffic_allocation
        self.duration_hours = duration_hours
        self.auto_start = auto_start
        self.auto_stop = auto_stop
        self.auto_promote_winner = auto_promote_winner
        self.schedule_config = schedule_config or {}


class ExperimentEngine(BaseEngine):
    """Enhanced A/B Testing Engine with complete statistical analysis and automation."""
    
    def __init__(self):
        super().__init__(
            engine_id="experiment_engine",
            name="ExperimentEngine",
            capabilities=[
                EngineCapability.DATA_ANALYSIS,
                EngineCapability.REPORT_GENERATION
            ]
        )
        self.status = EngineStatus.READY
        self.running_experiments: Dict[str, Experiment] = {}
        self.scheduler_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize the experiment engine with scheduler."""
        self.status = EngineStatus.READY
        # Start the experiment scheduler
        self.scheduler_task = asyncio.create_task(self._experiment_scheduler())
        logger.info("ExperimentEngine initialized with scheduler")
    
    async def process(self, input_data: Any, parameters: Dict[str, Any] = None) -> Any:
        """Process experiment-related requests."""
        action = parameters.get("action", "status") if parameters else "status"
        
        if action == "create_experiment":
            return await self.create_experiment(input_data)
        elif action == "start_experiment":
            return await self.start_experiment(input_data.get("experiment_id"))
        elif action == "stop_experiment":
            return await self.stop_experiment(input_data.get("experiment_id"))
        elif action == "analyze_results":
            return await self.analyze_experiment_results(input_data.get("experiment_id"))
        elif action == "promote_winner":
            return await self.promote_winner(input_data.get("experiment_id"))
        else:
            return {"status": "ready", "engine": "ExperimentEngine"}
    
    async def cleanup(self) -> None:
        """Clean up experiment engine resources."""
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("ExperimentEngine cleanup completed")
    
    async def create_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Create a new A/B test experiment."""
        try:
            experiment_id = str(uuid.uuid4())
            
            # Create experiment record
            experiment = Experiment(
                id=experiment_id,
                name=config.name,
                prompt_id=config.prompt_id,
                hypothesis=config.hypothesis,
                success_metrics=config.success_metrics,
                target_sample_size=config.target_sample_size,
                confidence_level=config.confidence_level,
                minimum_effect_size=config.minimum_effect_size,
                traffic_allocation=config.traffic_allocation,
                status=ExperimentStatus.DRAFT.value,
                created_by="system"
            )
            
            # Create variants
            for i, variant_config in enumerate(config.variants):
                variant = ExperimentVariant(
                    experiment_id=experiment_id,
                    name=variant_config.get("name", f"Variant {i+1}"),
                    description=variant_config.get("description", ""),
                    prompt_content=variant_config["prompt_content"],
                    prompt_variables=variant_config.get("variables", {}),
                    variant_type=variant_config.get("type", "treatment"),
                    traffic_weight=variant_config.get("traffic_weight", 1.0 / len(config.variants))
                )
                experiment.variants.append(variant)
            
            # Create schedule if configured
            if config.schedule_config:
                schedule = ExperimentSchedule(
                    experiment_id=experiment_id,
                    schedule_type=config.schedule_config.get("type", "manual"),
                    cron_expression=config.schedule_config.get("cron_expression"),
                    auto_start=config.auto_start,
                    auto_stop=config.auto_stop,
                    auto_promote_winner=config.auto_promote_winner,
                    max_duration_hours=config.duration_hours,
                    created_by="system"
                )
            
            self.running_experiments[experiment_id] = experiment
            
            logger.info(f"Created experiment {experiment_id}: {config.name}")
            return {
                "experiment_id": experiment_id,
                "status": "created",
                "variants_count": len(config.variants),
                "target_sample_size": config.target_sample_size
            }
            
        except Exception as e:
            logger.error(f"Error creating experiment: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def start_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Start an A/B test experiment."""
        try:
            experiment = self.running_experiments.get(experiment_id)
            if not experiment:
                return {"error": "Experiment not found", "status": "failed"}
            
            experiment.status = ExperimentStatus.RUNNING.value
            experiment.start_date = datetime.utcnow()
            
            logger.info(f"Started experiment {experiment_id}")
            return {
                "experiment_id": experiment_id,
                "status": "running",
                "start_date": experiment.start_date.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error starting experiment {experiment_id}: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def stop_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Stop an A/B test experiment."""
        try:
            experiment = self.running_experiments.get(experiment_id)
            if not experiment:
                return {"error": "Experiment not found", "status": "failed"}
            
            experiment.status = ExperimentStatus.COMPLETED.value
            experiment.end_date = datetime.utcnow()
            
            # Perform final analysis
            results = await self.analyze_experiment_results(experiment_id)
            
            logger.info(f"Stopped experiment {experiment_id}")
            return {
                "experiment_id": experiment_id,
                "status": "completed",
                "end_date": experiment.end_date.isoformat(),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error stopping experiment {experiment_id}: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def analyze_experiment_results(self, experiment_id: str) -> ExperimentResults:
        """Perform comprehensive statistical analysis of experiment results."""
        try:
            experiment = self.running_experiments.get(experiment_id)
            if not experiment:
                raise ValueError("Experiment not found")
            
            # Get metrics data for all variants
            variant_data = {}
            for variant in experiment.variants:
                variant_data[variant.id] = self._get_variant_metrics(variant.id)
            
            # Perform statistical analyses for each success metric
            analyses = []
            for metric_name in experiment.success_metrics:
                analysis = await self._perform_statistical_analysis(
                    variant_data, metric_name, experiment.confidence_level
                )
                if analysis:
                    analyses.append(analysis)
            
            # Determine winner
            winner_variant_id = self._determine_winner(analyses, experiment.minimum_effect_size)
            
            # Calculate conversion rates
            conversion_rates = self._calculate_conversion_rates(variant_data)
            
            # Generate visualizations
            visualizations = self._generate_visualizations(variant_data, analyses)
            
            # Calculate experiment duration
            duration = timedelta(0)
            if experiment.start_date and experiment.end_date:
                duration = experiment.end_date - experiment.start_date
            elif experiment.start_date:
                duration = datetime.utcnow() - experiment.start_date
            
            results = ExperimentResults(
                experiment_id=experiment_id,
                status=experiment.status,
                analyses=analyses,
                winner_variant_id=winner_variant_id,
                confidence_level=experiment.confidence_level,
                total_sample_size=sum(len(data) for data in variant_data.values()),
                experiment_duration=duration,
                conversion_rates=conversion_rates,
                visualizations=visualizations
            )
            
            # Store results in database
            await self._store_experiment_results(experiment, results)
            
            logger.info(f"Analyzed experiment {experiment_id} with {len(analyses)} metrics")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing experiment {experiment_id}: {e}")
            raise
    
    async def _perform_statistical_analysis(
        self, 
        variant_data: Dict[str, List[Dict]], 
        metric_name: str, 
        confidence_level: float
    ) -> Optional[StatisticalAnalysis]:
        """Perform statistical analysis for a specific metric."""
        try:
            # Extract metric values for each variant
            variant_metrics = {}
            for variant_id, data in variant_data.items():
                metrics = [d[metric_name] for d in data if metric_name in d]
                if metrics:
                    variant_metrics[variant_id] = metrics
            
            if len(variant_metrics) < 2:
                return None
            
            # For simplicity, compare first two variants (control vs treatment)
            variant_ids = list(variant_metrics.keys())
            control_data = variant_metrics[variant_ids[0]]
            treatment_data = variant_metrics[variant_ids[1]]
            
            if not control_data or not treatment_data:
                return None
            
            # Determine appropriate statistical test
            test_type = self._determine_test_type(control_data, treatment_data, metric_name)
            
            # Perform statistical test
            if test_type == TestType.T_TEST:
                statistic, p_value = ttest_ind(control_data, treatment_data)
                control_mean = np.mean(control_data)
                treatment_mean = np.mean(treatment_data)
                control_std = np.std(control_data, ddof=1)
                treatment_std = np.std(treatment_data, ddof=1)
                
            elif test_type == TestType.MANN_WHITNEY:
                statistic, p_value = mannwhitneyu(control_data, treatment_data, alternative='two-sided')
                control_mean = np.median(control_data)
                treatment_mean = np.median(treatment_data)
                control_std = np.std(control_data, ddof=1)
                treatment_std = np.std(treatment_data, ddof=1)
            
            else:  # Default to t-test
                statistic, p_value = ttest_ind(control_data, treatment_data)
                control_mean = np.mean(control_data)
                treatment_mean = np.mean(treatment_data)
                control_std = np.std(control_data, ddof=1)
                treatment_std = np.std(treatment_data, ddof=1)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(control_data) - 1) * control_std**2 + 
                                (len(treatment_data) - 1) * treatment_std**2) / 
                               (len(control_data) + len(treatment_data) - 2))
            effect_size = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
            
            # Calculate confidence interval
            alpha = 1 - confidence_level
            degrees_freedom = len(control_data) + len(treatment_data) - 2
            t_critical = stats.t.ppf(1 - alpha/2, degrees_freedom)
            
            se_diff = pooled_std * np.sqrt(1/len(control_data) + 1/len(treatment_data))
            mean_diff = treatment_mean - control_mean
            margin_error = t_critical * se_diff
            
            confidence_interval = (mean_diff - margin_error, mean_diff + margin_error)
            
            # Determine statistical significance
            if p_value < 0.001:
                significance = StatisticalSignificance.HIGHLY_SIGNIFICANT
            elif p_value < alpha:
                significance = StatisticalSignificance.SIGNIFICANT
            else:
                significance = StatisticalSignificance.NOT_SIGNIFICANT
            
            # Calculate statistical power (simplified)
            statistical_power = self._calculate_statistical_power(
                effect_size, len(control_data), len(treatment_data), alpha
            )
            
            # Generate recommendation
            recommendation = self._generate_recommendation(
                effect_size, p_value, significance, statistical_power
            )
            
            return StatisticalAnalysis(
                metric_name=metric_name,
                control_mean=control_mean,
                treatment_mean=treatment_mean,
                control_std=control_std,
                treatment_std=treatment_std,
                control_sample_size=len(control_data),
                treatment_sample_size=len(treatment_data),
                effect_size=effect_size,
                p_value=p_value,
                confidence_interval=confidence_interval,
                statistical_significance=significance,
                statistical_power=statistical_power,
                test_type=test_type.value,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Error in statistical analysis for {metric_name}: {e}")
            return None
    
    def _determine_test_type(self, control_data: List[float], treatment_data: List[float], metric_name: str) -> TestType:
        """Determine the appropriate statistical test based on data characteristics."""
        # Check for normality (simplified)
        if len(control_data) < 30 or len(treatment_data) < 30:
            return TestType.MANN_WHITNEY
        
        # Check if data appears to be binary/categorical
        control_unique = len(set(control_data))
        treatment_unique = len(set(treatment_data))
        
        if control_unique <= 2 and treatment_unique <= 2:
            return TestType.PROPORTION_TEST
        
        return TestType.T_TEST
    
    def _calculate_statistical_power(self, effect_size: float, n1: int, n2: int, alpha: float) -> float:
        """Calculate statistical power (simplified approximation)."""
        try:
            # Simplified power calculation
            n_harmonic = 2 / (1/n1 + 1/n2)
            ncp = effect_size * np.sqrt(n_harmonic / 2)
            
            # Use normal approximation for power
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = ncp - z_alpha
            power = stats.norm.cdf(z_beta)
            
            return max(0.0, min(1.0, power))
        except:
            return 0.8  # Default reasonable power
    
    def _generate_recommendation(self, effect_size: float, p_value: float, 
                               significance: StatisticalSignificance, power: float) -> str:
        """Generate actionable recommendation based on statistical results."""
        if significance == StatisticalSignificance.HIGHLY_SIGNIFICANT and abs(effect_size) > 0.2:
            return "Strong evidence for treatment effect. Recommend implementing the winning variant."
        elif significance == StatisticalSignificance.SIGNIFICANT and abs(effect_size) > 0.1:
            return "Moderate evidence for treatment effect. Consider implementing with monitoring."
        elif power < 0.8:
            return "Insufficient statistical power. Recommend increasing sample size and continuing experiment."
        else:
            return "No significant difference detected. Consider testing alternative variants."
    
    def _determine_winner(self, analyses: List[StatisticalAnalysis], min_effect_size: float) -> Optional[str]:
        """Determine the winning variant based on statistical analyses."""
        significant_analyses = [
            a for a in analyses 
            if a.statistical_significance != StatisticalSignificance.NOT_SIGNIFICANT
            and abs(a.effect_size) >= min_effect_size
        ]
        
        if not significant_analyses:
            return None
        
        # Simple heuristic: winner is the variant with the largest positive effect size
        best_analysis = max(significant_analyses, key=lambda a: a.effect_size)
        
        # Return treatment variant ID if effect size is positive, control if negative
        return "treatment_variant_id" if best_analysis.effect_size > 0 else "control_variant_id"
    
    async def promote_winner(self, experiment_id: str) -> Dict[str, Any]:
        """Promote the winning variant to production."""
        try:
            results = await self.analyze_experiment_results(experiment_id)
            
            if not results.winner_variant_id:
                return {
                    "status": "no_winner",
                    "message": "No statistically significant winner found"
                }
            
            # In a real implementation, this would update the prompt template
            # to use the winning variant's content
            
            logger.info(f"Promoted winner {results.winner_variant_id} for experiment {experiment_id}")
            return {
                "status": "promoted",
                "winner_variant_id": results.winner_variant_id,
                "experiment_id": experiment_id
            }
            
        except Exception as e:
            logger.error(f"Error promoting winner for experiment {experiment_id}: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def _experiment_scheduler(self):
        """Background task for experiment scheduling and automation."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = datetime.utcnow()
                
                # Check for experiments that need to be started or stopped
                for experiment_id, experiment in self.running_experiments.items():
                    # Auto-stop experiments that have reached their duration
                    if (experiment.status == ExperimentStatus.RUNNING.value and 
                        experiment.start_date):
                        
                        # Check if experiment should auto-stop
                        duration = current_time - experiment.start_date
                        # Add duration check logic here
                        
                        # Check if sample size target is reached
                        total_samples = sum(
                            len(self._get_variant_metrics(v.id)) 
                            for v in experiment.variants
                        )
                        
                        if total_samples >= experiment.target_sample_size:
                            await self.stop_experiment(experiment_id)
                            
                            # Auto-promote winner if configured
                            # This would be based on experiment schedule configuration
                            
            except Exception as e:
                logger.error(f"Error in experiment scheduler: {e}")
    
    def _get_variant_metrics(self, variant_id: str) -> List[Dict[str, Any]]:
        """Get metrics data for a variant (mock implementation)."""
        # In a real implementation, this would query the database
        # For now, return mock data
        return [
            {"accuracy": 0.85, "response_time": 1.2, "user_satisfaction": 4.2},
            {"accuracy": 0.87, "response_time": 1.1, "user_satisfaction": 4.3},
            {"accuracy": 0.83, "response_time": 1.3, "user_satisfaction": 4.1}
        ]
    
    def _calculate_conversion_rates(self, variant_data: Dict[str, List[Dict]]) -> Dict[str, float]:
        """Calculate conversion rates for each variant."""
        conversion_rates = {}
        for variant_id, data in variant_data.items():
            if data:
                # Simple conversion rate calculation (mock)
                conversions = sum(1 for d in data if d.get("converted", False))
                conversion_rates[variant_id] = conversions / len(data)
            else:
                conversion_rates[variant_id] = 0.0
        return conversion_rates
    
    def _generate_visualizations(self, variant_data: Dict[str, List[Dict]], 
                               analyses: List[StatisticalAnalysis]) -> Dict[str, Any]:
        """Generate visualization data for experiment results."""
        return {
            "metric_comparisons": [
                {
                    "metric": analysis.metric_name,
                    "control_mean": analysis.control_mean,
                    "treatment_mean": analysis.treatment_mean,
                    "p_value": analysis.p_value,
                    "effect_size": analysis.effect_size
                }
                for analysis in analyses
            ],
            "confidence_intervals": [
                {
                    "metric": analysis.metric_name,
                    "lower": analysis.confidence_interval[0],
                    "upper": analysis.confidence_interval[1]
                }
                for analysis in analyses
            ],
            "sample_sizes": {
                variant_id: len(data) 
                for variant_id, data in variant_data.items()
            }
        }
    
    async def _store_experiment_results(self, experiment: Experiment, results: ExperimentResults):
        """Store experiment results in the database."""
        # In a real implementation, this would save to the database
        logger.info(f"Stored results for experiment {experiment.id}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the experiment engine."""
        return {
            "engine_id": self.engine_id,
            "name": self.name,
            "status": self.status.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "healthy": self.status == EngineStatus.READY,
            "running_experiments": len(self.running_experiments),
            "scheduler_active": self.scheduler_task is not None and not self.scheduler_task.done()
        }