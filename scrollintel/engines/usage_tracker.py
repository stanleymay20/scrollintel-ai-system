"""
Usage tracking engine for visual content generation billing system.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import statistics

from ..models.usage_tracking_models import (
    GenerationType, ResourceType, ResourceUsage, GenerationUsage,
    UserUsageSummary, BudgetAlert, UsageForecast, CostOptimizationRecommendation
)


class UsageTracker:
    """
    Comprehensive usage tracking system for visual content generation.
    Monitors resource consumption, costs, and provides analytics.
    """
    
    def __init__(self, storage_backend=None):
        self.logger = logging.getLogger(__name__)
        self.storage = storage_backend or {}  # In-memory storage for demo
        self.active_sessions = {}
        self.cost_rates = self._initialize_cost_rates()
        
    def _initialize_cost_rates(self) -> Dict[ResourceType, float]:
        """Initialize cost rates for different resource types."""
        return {
            ResourceType.GPU_SECONDS: 0.05,  # $0.05 per GPU second
            ResourceType.CPU_SECONDS: 0.001,  # $0.001 per CPU second
            ResourceType.STORAGE_GB: 0.02,   # $0.02 per GB stored
            ResourceType.BANDWIDTH_GB: 0.01, # $0.01 per GB bandwidth
            ResourceType.API_CALLS: 0.001    # $0.001 per API call
        }
    
    async def start_generation_tracking(
        self,
        user_id: str,
        generation_type: GenerationType,
        model_used: str,
        prompt: str,
        parameters: Dict[str, Any] = None
    ) -> str:
        """Start tracking a new generation request."""
        usage = GenerationUsage(
            user_id=user_id,
            generation_type=generation_type,
            model_used=model_used,
            prompt=prompt,
            parameters=parameters or {},
            start_time=datetime.utcnow()
        )
        
        self.active_sessions[usage.id] = usage
        
        self.logger.info(f"Started tracking generation {usage.id} for user {user_id}")
        return usage.id
    
    async def track_resource_usage(
        self,
        session_id: str,
        resource_type: ResourceType,
        amount: float,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Track resource usage for an active generation session."""
        if session_id not in self.active_sessions:
            self.logger.warning(f"Session {session_id} not found for resource tracking")
            return
        
        unit_cost = self.cost_rates.get(resource_type, 0.0)
        total_cost = amount * unit_cost
        
        resource_usage = ResourceUsage(
            resource_type=resource_type,
            amount=amount,
            unit_cost=unit_cost,
            total_cost=total_cost,
            metadata=metadata or {}
        )
        
        session = self.active_sessions[session_id]
        session.resources.append(resource_usage)
        session.total_cost += total_cost
        
        self.logger.debug(
            f"Tracked {amount} {resource_type.value} "
            f"(${total_cost:.4f}) for session {session_id}"
        )
    
    async def end_generation_tracking(
        self,
        session_id: str,
        success: bool = True,
        quality_score: Optional[float] = None,
        error_message: Optional[str] = None
    ) -> GenerationUsage:
        """End tracking for a generation session."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        session.end_time = datetime.utcnow()
        session.duration_seconds = (session.end_time - session.start_time).total_seconds()
        session.success = success
        session.quality_score = quality_score
        session.error_message = error_message
        
        # Store the completed session
        if session.user_id not in self.storage:
            self.storage[session.user_id] = []
        self.storage[session.user_id].append(session)
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        self.logger.info(
            f"Completed tracking for session {session_id}. "
            f"Duration: {session.duration_seconds:.2f}s, Cost: ${session.total_cost:.4f}"
        )
        
        return session
    
    async def get_user_usage_summary(
        self,
        user_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> UserUsageSummary:
        """Get comprehensive usage summary for a user in a time period."""
        user_sessions = self.storage.get(user_id, [])
        
        # Filter sessions by date range
        filtered_sessions = [
            session for session in user_sessions
            if start_date <= session.start_time <= end_date
        ]
        
        if not filtered_sessions:
            return UserUsageSummary(
                user_id=user_id,
                period_start=start_date,
                period_end=end_date
            )
        
        summary = UserUsageSummary(
            user_id=user_id,
            period_start=start_date,
            period_end=end_date
        )
        
        # Calculate totals
        summary.total_generations = len(filtered_sessions)
        summary.successful_generations = sum(1 for s in filtered_sessions if s.success)
        summary.failed_generations = summary.total_generations - summary.successful_generations
        
        # Count by type
        for session in filtered_sessions:
            if session.generation_type == GenerationType.IMAGE:
                summary.image_generations += 1
            elif session.generation_type == GenerationType.VIDEO:
                summary.video_generations += 1
            elif session.generation_type == GenerationType.ENHANCEMENT:
                summary.enhancement_operations += 1
            elif session.generation_type == GenerationType.BATCH:
                summary.batch_operations += 1
        
        # Calculate resource usage
        for session in filtered_sessions:
            summary.total_cost += session.total_cost
            
            for resource in session.resources:
                if resource.resource_type == ResourceType.GPU_SECONDS:
                    summary.total_gpu_seconds += resource.amount
                elif resource.resource_type == ResourceType.CPU_SECONDS:
                    summary.total_cpu_seconds += resource.amount
                elif resource.resource_type == ResourceType.STORAGE_GB:
                    summary.total_storage_gb += resource.amount
                elif resource.resource_type == ResourceType.BANDWIDTH_GB:
                    summary.total_bandwidth_gb += resource.amount
                elif resource.resource_type == ResourceType.API_CALLS:
                    summary.total_api_calls += int(resource.amount)
        
        # Calculate averages
        if summary.total_generations > 0:
            summary.average_cost_per_generation = summary.total_cost / summary.total_generations
            
            quality_scores = [s.quality_score for s in filtered_sessions if s.quality_score is not None]
            if quality_scores:
                summary.average_quality_score = statistics.mean(quality_scores)
            
            generation_times = [s.duration_seconds for s in filtered_sessions if s.duration_seconds > 0]
            if generation_times:
                summary.average_generation_time = statistics.mean(generation_times)
        
        return summary
    
    async def check_budget_alerts(
        self,
        user_id: str,
        budget_limit: float,
        period_days: int = 30
    ) -> List[BudgetAlert]:
        """Check for budget threshold alerts."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)
        
        summary = await self.get_user_usage_summary(user_id, start_date, end_date)
        
        alerts = []
        
        # Check various thresholds
        thresholds = [50.0, 75.0, 90.0, 100.0]
        
        for threshold in thresholds:
            usage_percentage = (summary.total_cost / budget_limit) * 100
            
            if usage_percentage >= threshold:
                alert = BudgetAlert(
                    user_id=user_id,
                    alert_type=f"budget_{threshold}percent",
                    threshold_percentage=threshold,
                    current_usage=summary.total_cost,
                    budget_limit=budget_limit,
                    period_start=start_date,
                    period_end=end_date
                )
                alerts.append(alert)
        
        return alerts
    
    async def generate_usage_forecast(
        self,
        user_id: str,
        forecast_days: int = 30,
        historical_days: int = 90
    ) -> UsageForecast:
        """Generate usage forecast based on historical data."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=historical_days)
        
        user_sessions = self.storage.get(user_id, [])
        historical_sessions = [
            session for session in user_sessions
            if start_date <= session.start_time <= end_date
        ]
        
        if not historical_sessions:
            return UsageForecast(
                user_id=user_id,
                forecast_period_days=forecast_days
            )
        
        # Group by day and calculate daily usage
        daily_usage = defaultdict(float)
        daily_costs = defaultdict(float)
        
        for session in historical_sessions:
            day = session.start_time.date()
            daily_usage[day] += 1
            daily_costs[day] += session.total_cost
        
        # Calculate trend
        dates = sorted(daily_costs.keys())
        costs = [daily_costs[date] for date in dates]
        
        if len(costs) >= 7:  # Need at least a week of data
            # Simple linear trend calculation
            recent_avg = statistics.mean(costs[-7:])  # Last week
            older_avg = statistics.mean(costs[:-7]) if len(costs) > 7 else recent_avg
            
            if recent_avg > older_avg * 1.1:
                trend = "increasing"
            elif recent_avg < older_avg * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
            
            # Simple forecast: project recent average
            daily_forecast = recent_avg
            predicted_cost = daily_forecast * forecast_days
            
            # Confidence interval (simple approach)
            std_dev = statistics.stdev(costs[-14:]) if len(costs) >= 14 else recent_avg * 0.2
            confidence_interval = (
                max(0, predicted_cost - std_dev * forecast_days),
                predicted_cost + std_dev * forecast_days
            )
        else:
            trend = "insufficient_data"
            predicted_cost = 0.0
            confidence_interval = (0.0, 0.0)
        
        return UsageForecast(
            user_id=user_id,
            forecast_period_days=forecast_days,
            historical_usage=costs,
            historical_dates=[datetime.combine(date, datetime.min.time()) for date in dates],
            predicted_usage=len(costs),
            predicted_cost=predicted_cost,
            confidence_interval=confidence_interval,
            usage_trend=trend
        )
    
    async def generate_cost_optimization_recommendations(
        self,
        user_id: str,
        analysis_days: int = 30
    ) -> List[CostOptimizationRecommendation]:
        """Generate cost optimization recommendations."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=analysis_days)
        
        summary = await self.get_user_usage_summary(user_id, start_date, end_date)
        user_sessions = self.storage.get(user_id, [])
        recent_sessions = [
            session for session in user_sessions
            if start_date <= session.start_time <= end_date
        ]
        
        recommendations = []
        
        # Analyze failed generations
        failed_sessions = [s for s in recent_sessions if not s.success]
        if len(failed_sessions) > len(recent_sessions) * 0.1:  # >10% failure rate
            failed_cost = sum(s.total_cost for s in failed_sessions)
            recommendations.append(CostOptimizationRecommendation(
                user_id=user_id,
                recommendation_type="reduce_failures",
                title="Reduce Failed Generations",
                description=f"You have {len(failed_sessions)} failed generations costing ${failed_cost:.2f}. "
                           "Consider improving prompt quality or using more reliable models.",
                potential_savings=failed_cost,
                current_cost=summary.total_cost,
                optimized_cost=summary.total_cost - failed_cost,
                priority="high"
            ))
        
        # Analyze model usage efficiency
        model_costs = defaultdict(float)
        model_counts = defaultdict(int)
        model_quality = defaultdict(list)
        
        for session in recent_sessions:
            if session.success:
                model_costs[session.model_used] += session.total_cost
                model_counts[session.model_used] += 1
                if session.quality_score:
                    model_quality[session.model_used].append(session.quality_score)
        
        # Find expensive models with low quality
        for model, cost in model_costs.items():
            if model_counts[model] >= 5:  # Enough data
                avg_cost = cost / model_counts[model]
                avg_quality = statistics.mean(model_quality[model]) if model_quality[model] else 0.5
                
                if avg_cost > summary.average_cost_per_generation * 1.5 and avg_quality < 0.7:
                    potential_savings = cost * 0.3  # Assume 30% savings with better model
                    recommendations.append(CostOptimizationRecommendation(
                        user_id=user_id,
                        recommendation_type="optimize_model_selection",
                        title=f"Optimize {model} Usage",
                        description=f"Model {model} has high cost (${avg_cost:.3f}/generation) "
                                   f"and low quality ({avg_quality:.2f}). Consider alternatives.",
                        potential_savings=potential_savings,
                        current_cost=cost,
                        optimized_cost=cost - potential_savings,
                        affected_operations=[model],
                        priority="medium"
                    ))
        
        # Analyze resource usage patterns
        if summary.total_gpu_seconds > 0:
            gpu_cost = summary.total_gpu_seconds * self.cost_rates[ResourceType.GPU_SECONDS]
            if gpu_cost > summary.total_cost * 0.7:  # GPU is >70% of cost
                recommendations.append(CostOptimizationRecommendation(
                    user_id=user_id,
                    recommendation_type="optimize_gpu_usage",
                    title="Optimize GPU Usage",
                    description="GPU usage represents a large portion of your costs. "
                               "Consider batch processing or off-peak scheduling.",
                    potential_savings=gpu_cost * 0.2,  # 20% potential savings
                    current_cost=gpu_cost,
                    optimized_cost=gpu_cost * 0.8,
                    priority="medium"
                ))
        
        return recommendations
    
    async def get_real_time_cost_calculation(
        self,
        generation_type: GenerationType,
        model_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate estimated cost for a generation request in real-time."""
        # Base cost estimates by type and model
        base_costs = {
            GenerationType.IMAGE: {
                "stable_diffusion_xl": 0.02,
                "dalle3": 0.04,
                "midjourney": 0.08
            },
            GenerationType.VIDEO: {
                "runway_ml": 0.50,
                "pika_labs": 0.40,
                "custom_video": 1.00
            },
            GenerationType.ENHANCEMENT: {
                "real_esrgan": 0.01,
                "gfpgan": 0.015,
                "codeformer": 0.02
            }
        }
        
        base_cost = base_costs.get(generation_type, {}).get(model_name, 0.05)
        
        # Adjust for parameters
        multiplier = 1.0
        
        if generation_type == GenerationType.IMAGE:
            resolution = parameters.get("resolution", (1024, 1024))
            pixels = resolution[0] * resolution[1]
            multiplier *= (pixels / (1024 * 1024))  # Scale by resolution
            
            steps = parameters.get("steps", 50)
            multiplier *= (steps / 50)  # Scale by inference steps
            
        elif generation_type == GenerationType.VIDEO:
            duration = parameters.get("duration", 5.0)
            fps = parameters.get("fps", 24)
            resolution = parameters.get("resolution", (1280, 720))
            
            frames = duration * fps
            pixels_per_frame = resolution[0] * resolution[1]
            total_pixels = frames * pixels_per_frame
            
            # Scale by total computation
            multiplier *= (total_pixels / (5 * 24 * 1280 * 720))
        
        estimated_cost = base_cost * multiplier
        
        return {
            "base_cost": base_cost,
            "multiplier": multiplier,
            "estimated_cost": estimated_cost,
            "currency": "USD"
        }
    
    async def get_usage_analytics(
        self,
        user_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get comprehensive usage analytics."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        summary = await self.get_user_usage_summary(user_id, start_date, end_date)
        forecast = await self.generate_usage_forecast(user_id)
        recommendations = await self.generate_cost_optimization_recommendations(user_id)
        
        return {
            "summary": summary,
            "forecast": forecast,
            "recommendations": recommendations,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": days
            }
        }