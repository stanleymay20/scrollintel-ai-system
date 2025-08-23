"""
Continuous Improvement Engine

This module implements the core continuous improvement system that orchestrates
feedback collection, A/B testing, model retraining, and feature enhancement processes.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func

from ..models.continuous_improvement_models import (
    UserFeedback, ABTest, ABTestResult, ModelRetrainingJob, FeatureEnhancement,
    FeedbackType, FeedbackPriority, ABTestStatus, ModelRetrainingStatus,
    FeatureEnhancementStatus, ImprovementMetrics, ImprovementRecommendation
)
from ..core.database import get_db_session
from ..core.config import get_settings

logger = logging.getLogger(__name__)

class ContinuousImprovementEngine:
    """
    Enterprise-grade continuous improvement engine that processes real business data
    to drive system enhancements and optimizations.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.improvement_threshold = 0.05  # 5% minimum improvement
        self.confidence_threshold = 0.95
        self.min_sample_size = 1000
        
    async def collect_user_feedback(
        self,
        user_id: str,
        feedback_data: Dict[str, Any],
        db: Session
    ) -> UserFeedback:
        """
        Collect and process user feedback with real-time analysis.
        
        Args:
            user_id: ID of the user providing feedback
            feedback_data: Feedback content and metadata
            db: Database session
            
        Returns:
            Created feedback record
        """
        try:
            # Calculate business impact score based on feedback content
            business_impact_score = await self._calculate_business_impact(
                feedback_data, db
            )
            
            # Create feedback record
            feedback = UserFeedback(
                user_id=user_id,
                feedback_type=feedback_data["feedback_type"],
                priority=feedback_data["priority"],
                title=feedback_data["title"],
                description=feedback_data["description"],
                context=feedback_data.get("context", {}),
                business_impact_score=business_impact_score,
                satisfaction_rating=feedback_data.get("satisfaction_rating"),
                agent_id=feedback_data.get("agent_id"),
                feature_area=feedback_data.get("feature_area")
            )
            
            db.add(feedback)
            db.commit()
            db.refresh(feedback)
            
            # Trigger automatic improvement analysis
            await self._analyze_feedback_patterns(feedback, db)
            
            logger.info(f"Collected feedback {feedback.id} from user {user_id}")
            return feedback
            
        except Exception as e:
            logger.error(f"Error collecting feedback: {str(e)}")
            db.rollback()
            raise
    
    async def create_ab_test(
        self,
        test_config: Dict[str, Any],
        db: Session
    ) -> ABTest:
        """
        Create and configure A/B test for system improvements.
        
        Args:
            test_config: A/B test configuration
            db: Database session
            
        Returns:
            Created A/B test
        """
        try:
            # Validate test configuration
            await self._validate_ab_test_config(test_config)
            
            # Create A/B test
            ab_test = ABTest(
                name=test_config["name"],
                description=test_config.get("description"),
                hypothesis=test_config["hypothesis"],
                feature_area=test_config["feature_area"],
                control_config=test_config["control_config"],
                variant_configs=test_config["variant_configs"],
                traffic_allocation=test_config["traffic_allocation"],
                primary_metric=test_config["primary_metric"],
                secondary_metrics=test_config.get("secondary_metrics", []),
                minimum_sample_size=test_config.get("minimum_sample_size", 1000),
                confidence_level=test_config.get("confidence_level", 0.95)
            )
            
            db.add(ab_test)
            db.commit()
            db.refresh(ab_test)
            
            logger.info(f"Created A/B test {ab_test.id}: {ab_test.name}")
            return ab_test
            
        except Exception as e:
            logger.error(f"Error creating A/B test: {str(e)}")
            db.rollback()
            raise    
 
   async def start_ab_test(self, test_id: int, db: Session) -> bool:
        """
        Start an A/B test and begin collecting results.
        
        Args:
            test_id: ID of the test to start
            db: Database session
            
        Returns:
            Success status
        """
        try:
            ab_test = db.query(ABTest).filter(ABTest.id == test_id).first()
            if not ab_test:
                raise ValueError(f"A/B test {test_id} not found")
            
            if ab_test.status != ABTestStatus.DRAFT:
                raise ValueError(f"Cannot start test in status {ab_test.status}")
            
            # Update test status and start date
            ab_test.status = ABTestStatus.RUNNING
            ab_test.start_date = datetime.utcnow()
            
            # Calculate end date based on minimum sample size and expected traffic
            expected_daily_users = await self._estimate_daily_users(
                ab_test.feature_area, db
            )
            days_needed = max(7, ab_test.minimum_sample_size / expected_daily_users)
            ab_test.end_date = ab_test.start_date + timedelta(days=days_needed)
            
            db.commit()
            
            # Initialize test monitoring
            await self._initialize_ab_test_monitoring(ab_test)
            
            logger.info(f"Started A/B test {test_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting A/B test {test_id}: {str(e)}")
            db.rollback()
            return False
    
    async def record_ab_test_result(
        self,
        test_id: int,
        user_id: str,
        variant_name: str,
        metrics: Dict[str, Any],
        db: Session
    ) -> ABTestResult:
        """
        Record A/B test result for analysis.
        
        Args:
            test_id: ID of the A/B test
            user_id: ID of the user
            variant_name: Name of the test variant
            metrics: Performance metrics
            db: Database session
            
        Returns:
            Created test result
        """
        try:
            # Validate test is running
            ab_test = db.query(ABTest).filter(
                and_(ABTest.id == test_id, ABTest.status == ABTestStatus.RUNNING)
            ).first()
            
            if not ab_test:
                raise ValueError(f"Active A/B test {test_id} not found")
            
            # Create test result
            result = ABTestResult(
                test_id=test_id,
                variant_name=variant_name,
                user_id=user_id,
                primary_metric_value=metrics.get(ab_test.primary_metric),
                secondary_metrics=metrics.get("secondary_metrics", {}),
                conversion_event=metrics.get("conversion_event", False),
                session_duration=metrics.get("session_duration"),
                user_satisfaction=metrics.get("user_satisfaction"),
                business_value_generated=metrics.get("business_value_generated", 0.0),
                user_segment=metrics.get("user_segment"),
                device_type=metrics.get("device_type")
            )
            
            db.add(result)
            db.commit()
            db.refresh(result)
            
            # Check if test has enough data for analysis
            await self._check_ab_test_completion(ab_test, db)
            
            return result
            
        except Exception as e:
            logger.error(f"Error recording A/B test result: {str(e)}")
            db.rollback()
            raise
    
    async def analyze_ab_test_results(
        self,
        test_id: int,
        db: Session
    ) -> Dict[str, Any]:
        """
        Analyze A/B test results and determine statistical significance.
        
        Args:
            test_id: ID of the A/B test
            db: Database session
            
        Returns:
            Analysis results with statistical significance
        """
        try:
            ab_test = db.query(ABTest).filter(ABTest.id == test_id).first()
            if not ab_test:
                raise ValueError(f"A/B test {test_id} not found")
            
            # Get all test results
            results = db.query(ABTestResult).filter(
                ABTestResult.test_id == test_id
            ).all()
            
            if len(results) < ab_test.minimum_sample_size:
                return {
                    "status": "insufficient_data",
                    "sample_size": len(results),
                    "required_size": ab_test.minimum_sample_size
                }
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame([{
                "variant": r.variant_name,
                "primary_metric": r.primary_metric_value,
                "conversion": r.conversion_event,
                "satisfaction": r.user_satisfaction,
                "business_value": r.business_value_generated
            } for r in results])
            
            # Perform statistical analysis
            analysis_results = await self._perform_statistical_analysis(
                df, ab_test.primary_metric, ab_test.confidence_level
            )
            
            # Calculate business impact
            business_impact = await self._calculate_ab_test_business_impact(
                df, analysis_results
            )
            
            # Generate recommendations
            recommendations = await self._generate_ab_test_recommendations(
                ab_test, analysis_results, business_impact
            )
            
            return {
                "status": "completed",
                "sample_size": len(results),
                "statistical_results": analysis_results,
                "business_impact": business_impact,
                "recommendations": recommendations,
                "confidence_level": ab_test.confidence_level
            }
            
        except Exception as e:
            logger.error(f"Error analyzing A/B test {test_id}: {str(e)}")
            raise    

    async def schedule_model_retraining(
        self,
        model_config: Dict[str, Any],
        db: Session
    ) -> ModelRetrainingJob:
        """
        Schedule model retraining based on business outcomes.
        
        Args:
            model_config: Model retraining configuration
            db: Database session
            
        Returns:
            Created retraining job
        """
        try:
            # Validate model configuration
            await self._validate_model_config(model_config)
            
            # Get baseline performance metrics
            baseline_metrics = await self._get_model_baseline_metrics(
                model_config["model_name"], db
            )
            
            # Create retraining job
            job = ModelRetrainingJob(
                model_name=model_config["model_name"],
                model_version=model_config["model_version"],
                agent_type=model_config.get("agent_type"),
                training_config=model_config["training_config"],
                data_sources=model_config["data_sources"],
                performance_threshold=model_config.get("performance_threshold", 0.8),
                baseline_metrics=baseline_metrics,
                scheduled_at=model_config["scheduled_at"]
            )
            
            db.add(job)
            db.commit()
            db.refresh(job)
            
            # Schedule background execution
            await self._schedule_retraining_execution(job)
            
            logger.info(f"Scheduled model retraining job {job.id}")
            return job
            
        except Exception as e:
            logger.error(f"Error scheduling model retraining: {str(e)}")
            db.rollback()
            raise
    
    async def execute_model_retraining(
        self,
        job_id: int,
        db: Session
    ) -> Dict[str, Any]:
        """
        Execute model retraining with real business data.
        
        Args:
            job_id: ID of the retraining job
            db: Database session
            
        Returns:
            Retraining results and performance metrics
        """
        try:
            job = db.query(ModelRetrainingJob).filter(
                ModelRetrainingJob.id == job_id
            ).first()
            
            if not job:
                raise ValueError(f"Retraining job {job_id} not found")
            
            # Update job status
            job.status = ModelRetrainingStatus.RUNNING
            job.started_at = datetime.utcnow()
            job.current_step = "data_preparation"
            db.commit()
            
            # Prepare training data from real business outcomes
            training_data = await self._prepare_training_data(job, db)
            
            # Update progress
            job.progress_percentage = 25.0
            job.current_step = "model_training"
            db.commit()
            
            # Train model with business outcome feedback
            training_results = await self._train_model_with_business_feedback(
                job, training_data
            )
            
            # Update progress
            job.progress_percentage = 75.0
            job.current_step = "model_evaluation"
            db.commit()
            
            # Evaluate model performance
            evaluation_results = await self._evaluate_model_performance(
                job, training_results
            )
            
            # Calculate improvement metrics
            improvement_percentage = await self._calculate_improvement_percentage(
                job.baseline_metrics, evaluation_results["metrics"]
            )
            
            # Update job with results
            job.status = ModelRetrainingStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.progress_percentage = 100.0
            job.current_metrics = evaluation_results["metrics"]
            job.improvement_percentage = improvement_percentage
            job.business_impact_metrics = evaluation_results["business_impact"]
            job.model_artifacts_path = training_results["artifacts_path"]
            
            db.commit()
            
            logger.info(f"Completed model retraining job {job_id}")
            
            return {
                "job_id": job_id,
                "improvement_percentage": improvement_percentage,
                "performance_metrics": evaluation_results["metrics"],
                "business_impact": evaluation_results["business_impact"],
                "artifacts_path": training_results["artifacts_path"]
            }
            
        except Exception as e:
            logger.error(f"Error executing model retraining {job_id}: {str(e)}")
            
            # Update job status on failure
            job.status = ModelRetrainingStatus.FAILED
            job.error_message = str(e)
            db.commit()
            
            raise
    
    async def create_feature_enhancement(
        self,
        requester_id: str,
        enhancement_data: Dict[str, Any],
        db: Session
    ) -> FeatureEnhancement:
        """
        Create feature enhancement request based on user requirements.
        
        Args:
            requester_id: ID of the user requesting enhancement
            enhancement_data: Enhancement details and requirements
            db: Database session
            
        Returns:
            Created feature enhancement
        """
        try:
            # Calculate priority score based on business value and user impact
            priority_score = await self._calculate_enhancement_priority(
                enhancement_data, db
            )
            
            # Create enhancement request
            enhancement = FeatureEnhancement(
                title=enhancement_data["title"],
                description=enhancement_data["description"],
                requester_id=requester_id,
                feature_area=enhancement_data["feature_area"],
                enhancement_type=enhancement_data["enhancement_type"],
                priority=enhancement_data["priority"],
                complexity_score=enhancement_data["complexity_score"],
                business_value_score=enhancement_data["business_value_score"],
                user_impact_score=enhancement_data["user_impact_score"],
                technical_feasibility_score=enhancement_data["technical_feasibility_score"],
                estimated_effort_hours=enhancement_data.get("estimated_effort_hours"),
                expected_roi=enhancement_data.get("expected_roi"),
                requirements=enhancement_data["requirements"],
                acceptance_criteria=enhancement_data["acceptance_criteria"],
                technical_specifications=enhancement_data.get("technical_specifications")
            )
            
            db.add(enhancement)
            db.commit()
            db.refresh(enhancement)
            
            # Trigger automatic review process
            await self._trigger_enhancement_review(enhancement, db)
            
            logger.info(f"Created feature enhancement {enhancement.id}")
            return enhancement
            
        except Exception as e:
            logger.error(f"Error creating feature enhancement: {str(e)}")
            db.rollback()
            raise    

    async def generate_improvement_recommendations(
        self,
        db: Session,
        time_window_days: int = 30
    ) -> List[ImprovementRecommendation]:
        """
        Generate data-driven improvement recommendations based on real business outcomes.
        
        Args:
            db: Database session
            time_window_days: Analysis time window in days
            
        Returns:
            List of improvement recommendations
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)
            
            # Analyze feedback patterns
            feedback_analysis = await self._analyze_feedback_patterns_comprehensive(
                db, cutoff_date
            )
            
            # Analyze A/B test results
            ab_test_analysis = await self._analyze_ab_test_trends(db, cutoff_date)
            
            # Analyze model performance trends
            model_analysis = await self._analyze_model_performance_trends(
                db, cutoff_date
            )
            
            # Analyze feature adoption and usage
            feature_analysis = await self._analyze_feature_adoption(db, cutoff_date)
            
            # Generate recommendations based on analysis
            recommendations = []
            
            # Performance-based recommendations
            perf_recommendations = await self._generate_performance_recommendations(
                model_analysis, ab_test_analysis
            )
            recommendations.extend(perf_recommendations)
            
            # User experience recommendations
            ux_recommendations = await self._generate_ux_recommendations(
                feedback_analysis, feature_analysis
            )
            recommendations.extend(ux_recommendations)
            
            # Business value recommendations
            business_recommendations = await self._generate_business_recommendations(
                feedback_analysis, ab_test_analysis, model_analysis
            )
            recommendations.extend(business_recommendations)
            
            # Prioritize recommendations by expected impact
            recommendations.sort(
                key=lambda x: x.expected_impact.get("business_value", 0),
                reverse=True
            )
            
            logger.info(f"Generated {len(recommendations)} improvement recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise
    
    async def get_improvement_metrics(
        self,
        db: Session,
        time_window_days: int = 30
    ) -> ImprovementMetrics:
        """
        Get comprehensive improvement metrics and analytics.
        
        Args:
            db: Database session
            time_window_days: Analysis time window in days
            
        Returns:
            Improvement metrics and trends
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)
            
            # Feedback metrics
            feedback_metrics = await self._calculate_feedback_metrics(db, cutoff_date)
            
            # A/B test metrics
            ab_test_metrics = await self._calculate_ab_test_metrics(db, cutoff_date)
            
            # Model performance metrics
            model_metrics = await self._calculate_model_performance_metrics(
                db, cutoff_date
            )
            
            # Feature adoption metrics
            feature_metrics = await self._calculate_feature_adoption_metrics(
                db, cutoff_date
            )
            
            # Business impact metrics
            business_metrics = await self._calculate_business_impact_metrics(
                db, cutoff_date
            )
            
            # User satisfaction trends
            satisfaction_trends = await self._calculate_satisfaction_trends(
                db, cutoff_date
            )
            
            # System reliability trends
            reliability_trends = await self._calculate_reliability_trends(
                db, cutoff_date
            )
            
            return ImprovementMetrics(
                feedback_metrics=feedback_metrics,
                ab_test_metrics=ab_test_metrics,
                model_performance_metrics=model_metrics,
                feature_adoption_metrics=feature_metrics,
                business_impact_metrics=business_metrics,
                user_satisfaction_trends=satisfaction_trends,
                system_reliability_trends=reliability_trends
            )
            
        except Exception as e:
            logger.error(f"Error calculating improvement metrics: {str(e)}")
            raise
    
    # Private helper methods
    async def _calculate_business_impact(
        self,
        feedback_data: Dict[str, Any],
        db: Session
    ) -> float:
        """Calculate business impact score for feedback."""
        try:
            # Base score from satisfaction rating
            base_score = feedback_data.get("satisfaction_rating", 5) / 10.0
            
            # Adjust based on feedback type
            type_multipliers = {
                FeedbackType.BUSINESS_IMPACT: 2.0,
                FeedbackType.PERFORMANCE_ISSUE: 1.5,
                FeedbackType.FEATURE_REQUEST: 1.2,
                FeedbackType.USER_SATISFACTION: 1.0,
                FeedbackType.BUG_REPORT: 0.8,
                FeedbackType.SYSTEM_RELIABILITY: 1.8
            }
            
            multiplier = type_multipliers.get(
                feedback_data["feedback_type"], 1.0
            )
            
            # Adjust based on user segment and historical impact
            user_impact = await self._get_user_impact_factor(
                feedback_data.get("user_id"), db
            )
            
            return min(10.0, base_score * multiplier * user_impact)
            
        except Exception as e:
            logger.error(f"Error calculating business impact: {str(e)}")
            return 5.0  # Default neutral score
    
    async def _analyze_feedback_patterns(
        self,
        feedback: UserFeedback,
        db: Session
    ) -> None:
        """Analyze feedback patterns for automatic improvement triggers."""
        try:
            # Check for recurring issues in the same feature area
            recent_feedback = db.query(UserFeedback).filter(
                and_(
                    UserFeedback.feature_area == feedback.feature_area,
                    UserFeedback.created_at >= datetime.utcnow() - timedelta(days=7),
                    UserFeedback.feedback_type == feedback.feedback_type
                )
            ).count()
            
            # Trigger automatic A/B test if threshold exceeded
            if recent_feedback >= 5:
                await self._trigger_automatic_ab_test(feedback, db)
            
            # Trigger model retraining if performance issues detected
            if (feedback.feedback_type == FeedbackType.PERFORMANCE_ISSUE and
                feedback.business_impact_score > 7.0):
                await self._trigger_automatic_model_retraining(feedback, db)
                
        except Exception as e:
            logger.error(f"Error analyzing feedback patterns: {str(e)}")
    
    async def _validate_ab_test_config(self, config: Dict[str, Any]) -> None:
        """Validate A/B test configuration."""
        required_fields = [
            "name", "hypothesis", "feature_area", "control_config",
            "variant_configs", "traffic_allocation", "primary_metric"
        ]
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate traffic allocation sums to 1.0
        total_allocation = sum(config["traffic_allocation"].values())
        if abs(total_allocation - 1.0) > 0.01:
            raise ValueError("Traffic allocation must sum to 1.0")
    
    async def _perform_statistical_analysis(
        self,
        df: pd.DataFrame,
        primary_metric: str,
        confidence_level: float
    ) -> Dict[str, Any]:
        """Perform statistical analysis on A/B test results."""
        try:
            from scipy import stats
            
            # Group by variant
            variants = df.groupby('variant')
            
            results = {}
            control_data = None
            
            for variant_name, variant_data in variants:
                if variant_name == 'control':
                    control_data = variant_data
                    continue
                
                if control_data is None:
                    continue
                
                # Perform t-test for primary metric
                control_values = control_data[primary_metric].dropna()
                variant_values = variant_data[primary_metric].dropna()
                
                if len(control_values) > 0 and len(variant_values) > 0:
                    t_stat, p_value = stats.ttest_ind(
                        variant_values, control_values
                    )
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(
                        ((len(control_values) - 1) * control_values.var() +
                         (len(variant_values) - 1) * variant_values.var()) /
                        (len(control_values) + len(variant_values) - 2)
                    )
                    
                    effect_size = (variant_values.mean() - control_values.mean()) / pooled_std
                    
                    results[variant_name] = {
                        "sample_size": len(variant_values),
                        "mean": variant_values.mean(),
                        "std": variant_values.std(),
                        "control_mean": control_values.mean(),
                        "lift": (variant_values.mean() - control_values.mean()) / control_values.mean(),
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "effect_size": effect_size,
                        "significant": p_value < (1 - confidence_level),
                        "confidence_level": confidence_level
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in statistical analysis: {str(e)}")
            return {}
    
    async def _get_user_impact_factor(self, user_id: str, db: Session) -> float:
        """Get user impact factor based on historical data."""
        try:
            # Simple implementation - can be enhanced with user segmentation
            user_feedback_count = db.query(UserFeedback).filter(
                UserFeedback.user_id == user_id
            ).count()
            
            # Users with more feedback history have higher impact
            return min(2.0, 1.0 + (user_feedback_count * 0.1))
            
        except Exception as e:
            logger.error(f"Error getting user impact factor: {str(e)}")
            return 1.0