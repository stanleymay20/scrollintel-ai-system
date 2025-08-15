"""Capacity planning and resource optimization recommendations."""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

from .platform_monitor import get_platform_monitor
from .resource_optimizer import get_resource_optimizer
from ..models.monitoring_models import CapacityPlan


class ResourceComponent(Enum):
    """Resource components for capacity planning."""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    PROCESSING_CAPACITY = "processing_capacity"


class ForecastMethod(Enum):
    """Forecasting methods."""
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    EXPONENTIAL = "exponential"
    SEASONAL = "seasonal"


@dataclass
class ResourceForecast:
    """Resource utilization forecast."""
    component: ResourceComponent
    current_utilization: float
    forecasted_values: List[float]
    forecast_dates: List[datetime]
    confidence_intervals: List[Tuple[float, float]]
    method_used: ForecastMethod
    accuracy_score: float
    trend_direction: str  # increasing, decreasing, stable
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'component': self.component.value,
            'current_utilization': self.current_utilization,
            'forecasted_values': self.forecasted_values,
            'forecast_dates': [d.isoformat() for d in self.forecast_dates],
            'confidence_intervals': self.confidence_intervals,
            'method_used': self.method_used.value,
            'accuracy_score': self.accuracy_score,
            'trend_direction': self.trend_direction
        }


@dataclass
class CapacityRecommendation:
    """Capacity planning recommendation."""
    component: ResourceComponent
    current_capacity: float
    recommended_capacity: float
    capacity_increase_percent: float
    justification: str
    priority: str  # high, medium, low
    estimated_cost: Optional[str] = None
    implementation_timeline: Optional[str] = None
    risk_level: str = "medium"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'component': self.component.value,
            'current_capacity': self.current_capacity,
            'recommended_capacity': self.recommended_capacity,
            'capacity_increase_percent': self.capacity_increase_percent,
            'justification': self.justification,
            'priority': self.priority,
            'estimated_cost': self.estimated_cost,
            'implementation_timeline': self.implementation_timeline,
            'risk_level': self.risk_level
        }


@dataclass
class CapacityPlanningReport:
    """Comprehensive capacity planning report."""
    generated_at: datetime = field(default_factory=datetime.utcnow)
    time_horizon_days: int = 30
    forecasts: List[ResourceForecast] = field(default_factory=list)
    recommendations: List[CapacityRecommendation] = field(default_factory=list)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    cost_analysis: Dict[str, Any] = field(default_factory=dict)
    executive_summary: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'generated_at': self.generated_at.isoformat(),
            'time_horizon_days': self.time_horizon_days,
            'forecasts': [f.to_dict() for f in self.forecasts],
            'recommendations': [r.to_dict() for r in self.recommendations],
            'risk_assessment': self.risk_assessment,
            'cost_analysis': self.cost_analysis,
            'executive_summary': self.executive_summary
        }


class CapacityPlanner:
    """Advanced capacity planning system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.monitor = get_platform_monitor()
        self.optimizer = get_resource_optimizer()
        
        # Capacity thresholds
        self.capacity_thresholds = {
            ResourceComponent.CPU: {'warning': 0.7, 'critical': 0.85},
            ResourceComponent.MEMORY: {'warning': 0.75, 'critical': 0.9},
            ResourceComponent.STORAGE: {'warning': 0.8, 'critical': 0.95},
            ResourceComponent.NETWORK: {'warning': 0.6, 'critical': 0.8},
            ResourceComponent.PROCESSING_CAPACITY: {'warning': 0.7, 'critical': 0.85}
        }
        
        # Cost estimates (per unit increase)
        self.cost_estimates = {
            ResourceComponent.CPU: {'unit': 'vCPU', 'cost_per_unit': 50},
            ResourceComponent.MEMORY: {'unit': 'GB', 'cost_per_unit': 10},
            ResourceComponent.STORAGE: {'unit': 'GB', 'cost_per_unit': 0.5},
            ResourceComponent.NETWORK: {'unit': 'Mbps', 'cost_per_unit': 5},
            ResourceComponent.PROCESSING_CAPACITY: {'unit': 'worker', 'cost_per_unit': 200}
        }
    
    def generate_capacity_plan(self, time_horizon_days: int = 30) -> CapacityPlanningReport:
        """Generate comprehensive capacity planning report."""
        try:
            self.logger.info(f"Generating capacity plan for {time_horizon_days} days")
            
            # Get historical data
            historical_data = self._collect_historical_data(days=min(time_horizon_days * 2, 90))
            
            # Generate forecasts for each component
            forecasts = []
            for component in ResourceComponent:
                forecast = self._generate_forecast(component, historical_data, time_horizon_days)
                if forecast:
                    forecasts.append(forecast)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(forecasts)
            
            # Perform risk assessment
            risk_assessment = self._assess_capacity_risks(forecasts, recommendations)
            
            # Calculate cost analysis
            cost_analysis = self._calculate_cost_analysis(recommendations)
            
            # Generate executive summary
            executive_summary = self._generate_executive_summary(forecasts, recommendations, risk_assessment)
            
            report = CapacityPlanningReport(
                time_horizon_days=time_horizon_days,
                forecasts=forecasts,
                recommendations=recommendations,
                risk_assessment=risk_assessment,
                cost_analysis=cost_analysis,
                executive_summary=executive_summary
            )
            
            self.logger.info("Capacity planning report generated successfully")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating capacity plan: {e}")
            raise
    
    def _collect_historical_data(self, days: int = 30) -> Dict[str, Any]:
        """Collect historical resource utilization data."""
        try:
            # Get metrics history
            metrics_history = self.monitor.get_metrics_history(hours=days * 24)
            resource_history = self.optimizer.get_resource_history(hours=days * 24)
            
            # Organize data by component
            historical_data = {}
            
            # System metrics
            system_metrics = metrics_history.get('system_metrics', [])
            if system_metrics:
                df_system = pd.DataFrame(system_metrics)
                df_system['timestamp'] = pd.to_datetime(df_system['timestamp'])
                
                historical_data[ResourceComponent.CPU] = {
                    'timestamps': df_system['timestamp'].tolist(),
                    'values': df_system['cpu_percent'].tolist()
                }
                
                historical_data[ResourceComponent.MEMORY] = {
                    'timestamps': df_system['timestamp'].tolist(),
                    'values': df_system['memory_percent'].tolist()
                }
                
                historical_data[ResourceComponent.STORAGE] = {
                    'timestamps': df_system['timestamp'].tolist(),
                    'values': df_system['disk_usage_percent'].tolist()
                }
            
            # Platform metrics
            platform_metrics = metrics_history.get('platform_metrics', [])
            if platform_metrics:
                df_platform = pd.DataFrame(platform_metrics)
                df_platform['timestamp'] = pd.to_datetime(df_platform['timestamp'])
                
                # Calculate processing capacity utilization
                max_processing_capacity = 100  # Assume 100 concurrent operations max
                processing_utilization = (df_platform['processing_datasets'] / max_processing_capacity * 100).tolist()
                
                historical_data[ResourceComponent.PROCESSING_CAPACITY] = {
                    'timestamps': df_platform['timestamp'].tolist(),
                    'values': processing_utilization
                }
            
            # Network utilization (placeholder)
            if system_metrics:
                # Calculate network utilization as percentage of max bandwidth
                max_bandwidth_mbps = 1000  # Assume 1Gbps max
                network_utilization = []
                
                for metric in system_metrics:
                    bytes_per_sec = (metric.get('network_bytes_sent', 0) + metric.get('network_bytes_recv', 0)) / 60
                    mbps = (bytes_per_sec * 8) / (1024 * 1024)
                    utilization = min(100, (mbps / max_bandwidth_mbps) * 100)
                    network_utilization.append(utilization)
                
                historical_data[ResourceComponent.NETWORK] = {
                    'timestamps': df_system['timestamp'].tolist(),
                    'values': network_utilization
                }
            
            return historical_data
            
        except Exception as e:
            self.logger.error(f"Error collecting historical data: {e}")
            return {}
    
    def _generate_forecast(self, component: ResourceComponent, 
                          historical_data: Dict[str, Any], 
                          time_horizon_days: int) -> Optional[ResourceForecast]:
        """Generate forecast for a specific resource component."""
        try:
            if component not in historical_data:
                return None
            
            data = historical_data[component]
            if len(data['values']) < 10:  # Need minimum data points
                return None
            
            # Prepare data
            timestamps = pd.to_datetime(data['timestamps'])
            values = np.array(data['values'])
            
            # Create time series features
            df = pd.DataFrame({
                'timestamp': timestamps,
                'value': values
            })
            df = df.sort_values('timestamp')
            
            # Create numerical time features
            df['time_numeric'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 3600  # hours
            
            # Try different forecasting methods
            best_forecast = None
            best_score = -np.inf
            
            for method in ForecastMethod:
                try:
                    forecast = self._apply_forecast_method(df, method, time_horizon_days)
                    if forecast and forecast.accuracy_score > best_score:
                        best_forecast = forecast
                        best_score = forecast.accuracy_score
                except Exception as e:
                    self.logger.warning(f"Error with {method.value} forecast for {component.value}: {e}")
            
            return best_forecast
            
        except Exception as e:
            self.logger.error(f"Error generating forecast for {component.value}: {e}")
            return None
    
    def _apply_forecast_method(self, df: pd.DataFrame, method: ForecastMethod, 
                              time_horizon_days: int) -> Optional[ResourceForecast]:
        """Apply specific forecasting method."""
        try:
            # Split data for validation
            split_point = int(len(df) * 0.8)
            train_df = df.iloc[:split_point]
            test_df = df.iloc[split_point:]
            
            if len(train_df) < 5 or len(test_df) < 2:
                return None
            
            X_train = train_df[['time_numeric']].values
            y_train = train_df['value'].values
            X_test = test_df[['time_numeric']].values
            y_test = test_df['value'].values
            
            # Apply forecasting method
            if method == ForecastMethod.LINEAR:
                model = LinearRegression()
                model.fit(X_train, y_train)
                
            elif method == ForecastMethod.POLYNOMIAL:
                poly_features = PolynomialFeatures(degree=2)
                X_train_poly = poly_features.fit_transform(X_train)
                X_test_poly = poly_features.transform(X_test)
                
                model = LinearRegression()
                model.fit(X_train_poly, y_train)
                
                # Update test predictions
                y_pred_test = model.predict(X_test_poly)
                
            else:
                # Default to linear for other methods
                model = LinearRegression()
                model.fit(X_train, y_train)
            
            # Make predictions on test set
            if method != ForecastMethod.POLYNOMIAL:
                y_pred_test = model.predict(X_test)
            
            # Calculate accuracy
            r2 = r2_score(y_test, y_pred_test)
            
            # Generate future predictions
            last_time = df['time_numeric'].max()
            future_times = []
            forecast_dates = []
            
            for i in range(1, time_horizon_days + 1):
                future_time = last_time + (i * 24)  # 24 hours per day
                future_times.append([future_time])
                
                future_date = df['timestamp'].max() + timedelta(days=i)
                forecast_dates.append(future_date)
            
            future_times = np.array(future_times)
            
            # Make future predictions
            if method == ForecastMethod.POLYNOMIAL:
                future_times_poly = poly_features.transform(future_times)
                forecasted_values = model.predict(future_times_poly).tolist()
            else:
                forecasted_values = model.predict(future_times).tolist()
            
            # Ensure values are within reasonable bounds (0-100%)
            forecasted_values = [max(0, min(100, v)) for v in forecasted_values]
            
            # Calculate confidence intervals (simplified)
            mse = mean_squared_error(y_test, y_pred_test)
            std_error = np.sqrt(mse)
            confidence_intervals = [
                (max(0, v - 1.96 * std_error), min(100, v + 1.96 * std_error))
                for v in forecasted_values
            ]
            
            # Determine trend direction
            if len(forecasted_values) >= 2:
                trend_slope = (forecasted_values[-1] - forecasted_values[0]) / len(forecasted_values)
                if trend_slope > 1:
                    trend_direction = "increasing"
                elif trend_slope < -1:
                    trend_direction = "decreasing"
                else:
                    trend_direction = "stable"
            else:
                trend_direction = "stable"
            
            current_utilization = df['value'].iloc[-1] / 100.0  # Convert to fraction
            
            return ResourceForecast(
                component=ResourceComponent(df.name if hasattr(df, 'name') else 'cpu'),
                current_utilization=current_utilization,
                forecasted_values=forecasted_values,
                forecast_dates=forecast_dates,
                confidence_intervals=confidence_intervals,
                method_used=method,
                accuracy_score=r2,
                trend_direction=trend_direction
            )
            
        except Exception as e:
            self.logger.error(f"Error applying {method.value} forecast: {e}")
            return None
    
    def _generate_recommendations(self, forecasts: List[ResourceForecast]) -> List[CapacityRecommendation]:
        """Generate capacity recommendations based on forecasts."""
        recommendations = []
        
        for forecast in forecasts:
            try:
                # Get thresholds for this component
                thresholds = self.capacity_thresholds.get(forecast.component, {'warning': 0.7, 'critical': 0.85})
                
                # Find maximum forecasted utilization
                max_forecasted = max(forecast.forecasted_values) / 100.0  # Convert to fraction
                
                # Determine if action is needed
                if max_forecasted > thresholds['critical']:
                    priority = "high"
                    risk_level = "high"
                    justification = f"Forecasted utilization ({max_forecasted:.1%}) exceeds critical threshold ({thresholds['critical']:.1%})"
                elif max_forecasted > thresholds['warning']:
                    priority = "medium"
                    risk_level = "medium"
                    justification = f"Forecasted utilization ({max_forecasted:.1%}) exceeds warning threshold ({thresholds['warning']:.1%})"
                else:
                    continue  # No action needed
                
                # Calculate recommended capacity increase
                target_utilization = 0.7  # Target 70% utilization
                current_capacity = 100  # Assume current capacity is 100 units
                required_capacity = (max_forecasted / target_utilization) * current_capacity
                capacity_increase_percent = ((required_capacity - current_capacity) / current_capacity) * 100
                
                # Estimate cost
                cost_info = self.cost_estimates.get(forecast.component, {'unit': 'unit', 'cost_per_unit': 100})
                estimated_monthly_cost = (required_capacity - current_capacity) * cost_info['cost_per_unit']
                estimated_cost = f"${estimated_monthly_cost:.0f}/month"
                
                # Implementation timeline
                if priority == "high":
                    implementation_timeline = "1-2 weeks"
                elif priority == "medium":
                    implementation_timeline = "2-4 weeks"
                else:
                    implementation_timeline = "1-2 months"
                
                recommendation = CapacityRecommendation(
                    component=forecast.component,
                    current_capacity=current_capacity,
                    recommended_capacity=required_capacity,
                    capacity_increase_percent=capacity_increase_percent,
                    justification=justification,
                    priority=priority,
                    estimated_cost=estimated_cost,
                    implementation_timeline=implementation_timeline,
                    risk_level=risk_level
                )
                
                recommendations.append(recommendation)
                
            except Exception as e:
                self.logger.error(f"Error generating recommendation for {forecast.component.value}: {e}")
        
        return recommendations
    
    def _assess_capacity_risks(self, forecasts: List[ResourceForecast], 
                              recommendations: List[CapacityRecommendation]) -> Dict[str, Any]:
        """Assess capacity-related risks."""
        try:
            risks = {
                'overall_risk_level': 'low',
                'risk_factors': [],
                'mitigation_strategies': [],
                'business_impact': []
            }
            
            high_risk_components = []
            medium_risk_components = []
            
            # Analyze each forecast for risks
            for forecast in forecasts:
                max_utilization = max(forecast.forecasted_values) / 100.0
                
                if max_utilization > 0.9:
                    high_risk_components.append(forecast.component.value)
                    risks['risk_factors'].append(
                        f"{forecast.component.value} utilization may exceed 90% ({max_utilization:.1%})"
                    )
                elif max_utilization > 0.75:
                    medium_risk_components.append(forecast.component.value)
                    risks['risk_factors'].append(
                        f"{forecast.component.value} utilization may exceed 75% ({max_utilization:.1%})"
                    )
            
            # Determine overall risk level
            if high_risk_components:
                risks['overall_risk_level'] = 'high'
            elif medium_risk_components:
                risks['overall_risk_level'] = 'medium'
            
            # Add mitigation strategies
            if high_risk_components:
                risks['mitigation_strategies'].extend([
                    "Implement immediate capacity scaling for high-risk components",
                    "Set up automated scaling policies",
                    "Establish emergency capacity procurement procedures"
                ])
            
            if medium_risk_components:
                risks['mitigation_strategies'].extend([
                    "Plan capacity upgrades for medium-risk components",
                    "Implement enhanced monitoring and alerting",
                    "Optimize resource utilization through performance tuning"
                ])
            
            # Business impact assessment
            if risks['overall_risk_level'] == 'high':
                risks['business_impact'].extend([
                    "Potential service degradation or outages",
                    "Reduced data processing throughput",
                    "Increased response times for AI model training",
                    "Possible SLA violations"
                ])
            elif risks['overall_risk_level'] == 'medium':
                risks['business_impact'].extend([
                    "Reduced system performance during peak loads",
                    "Longer processing times for large datasets",
                    "Limited capacity for handling traffic spikes"
                ])
            
            return risks
            
        except Exception as e:
            self.logger.error(f"Error assessing capacity risks: {e}")
            return {'overall_risk_level': 'unknown', 'risk_factors': [], 'mitigation_strategies': [], 'business_impact': []}
    
    def _calculate_cost_analysis(self, recommendations: List[CapacityRecommendation]) -> Dict[str, Any]:
        """Calculate cost analysis for capacity recommendations."""
        try:
            total_monthly_cost = 0
            total_annual_cost = 0
            cost_breakdown = {}
            
            for rec in recommendations:
                if rec.estimated_cost:
                    # Extract cost from string (assumes format like "$500/month")
                    cost_str = rec.estimated_cost.replace('$', '').replace('/month', '').replace(',', '')
                    try:
                        monthly_cost = float(cost_str)
                        total_monthly_cost += monthly_cost
                        cost_breakdown[rec.component.value] = monthly_cost
                    except ValueError:
                        pass
            
            total_annual_cost = total_monthly_cost * 12
            
            # Calculate ROI estimates
            # Assume capacity constraints cost 2x the upgrade cost in lost productivity
            potential_savings = total_monthly_cost * 2
            roi_months = 6  # Assume 6 months to break even
            
            return {
                'total_monthly_cost': total_monthly_cost,
                'total_annual_cost': total_annual_cost,
                'cost_breakdown': cost_breakdown,
                'potential_monthly_savings': potential_savings,
                'roi_break_even_months': roi_months,
                'cost_benefit_ratio': potential_savings / max(total_monthly_cost, 1)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating cost analysis: {e}")
            return {}
    
    def _generate_executive_summary(self, forecasts: List[ResourceForecast], 
                                   recommendations: List[CapacityRecommendation],
                                   risk_assessment: Dict[str, Any]) -> str:
        """Generate executive summary of capacity planning."""
        try:
            summary_parts = []
            
            # Overall assessment
            risk_level = risk_assessment.get('overall_risk_level', 'unknown')
            summary_parts.append(f"Overall capacity risk level: {risk_level.upper()}")
            
            # Key findings
            if recommendations:
                high_priority = [r for r in recommendations if r.priority == 'high']
                medium_priority = [r for r in recommendations if r.priority == 'medium']
                
                if high_priority:
                    components = [r.component.value for r in high_priority]
                    summary_parts.append(f"URGENT: {len(high_priority)} components require immediate attention: {', '.join(components)}")
                
                if medium_priority:
                    components = [r.component.value for r in medium_priority]
                    summary_parts.append(f"PLANNED: {len(medium_priority)} components need capacity planning: {', '.join(components)}")
            
            # Trend analysis
            increasing_trends = [f for f in forecasts if f.trend_direction == 'increasing']
            if increasing_trends:
                components = [f.component.value for f in increasing_trends]
                summary_parts.append(f"Growing demand detected for: {', '.join(components)}")
            
            # Cost impact
            total_cost = sum(
                float(r.estimated_cost.replace('$', '').replace('/month', '').replace(',', ''))
                for r in recommendations if r.estimated_cost
            )
            if total_cost > 0:
                summary_parts.append(f"Estimated monthly investment required: ${total_cost:,.0f}")
            
            # Business impact
            business_impacts = risk_assessment.get('business_impact', [])
            if business_impacts:
                summary_parts.append(f"Key business risks: {business_impacts[0]}")
            
            # Recommendations
            if recommendations:
                summary_parts.append(f"Recommended actions: {len(recommendations)} capacity adjustments needed")
            else:
                summary_parts.append("No immediate capacity adjustments required")
            
            return ". ".join(summary_parts) + "."
            
        except Exception as e:
            self.logger.error(f"Error generating executive summary: {e}")
            return "Executive summary generation failed."
    
    def export_capacity_plan(self, report: CapacityPlanningReport, filepath: str):
        """Export capacity planning report to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
            
            self.logger.info(f"Capacity planning report exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting capacity plan: {e}")
            raise
    
    def get_current_capacity_status(self) -> Dict[str, Any]:
        """Get current capacity status across all components."""
        try:
            current_system = self.monitor.get_current_system_metrics()
            current_platform = self.monitor.get_current_platform_metrics()
            
            if not current_system or not current_platform:
                return {'status': 'unknown', 'message': 'Insufficient metrics data'}
            
            status = {}
            
            # CPU status
            cpu_util = current_system.cpu_percent / 100.0
            cpu_threshold = self.capacity_thresholds[ResourceComponent.CPU]
            status['cpu'] = {
                'utilization': cpu_util,
                'status': self._get_utilization_status(cpu_util, cpu_threshold),
                'headroom_percent': (1 - cpu_util) * 100
            }
            
            # Memory status
            memory_util = current_system.memory_percent / 100.0
            memory_threshold = self.capacity_thresholds[ResourceComponent.MEMORY]
            status['memory'] = {
                'utilization': memory_util,
                'status': self._get_utilization_status(memory_util, memory_threshold),
                'headroom_percent': (1 - memory_util) * 100
            }
            
            # Storage status
            storage_util = current_system.disk_usage_percent / 100.0
            storage_threshold = self.capacity_thresholds[ResourceComponent.STORAGE]
            status['storage'] = {
                'utilization': storage_util,
                'status': self._get_utilization_status(storage_util, storage_threshold),
                'headroom_percent': (1 - storage_util) * 100
            }
            
            # Processing capacity
            max_processing = 100  # Assume max 100 concurrent operations
            processing_util = current_platform.processing_datasets / max_processing
            processing_threshold = self.capacity_thresholds[ResourceComponent.PROCESSING_CAPACITY]
            status['processing'] = {
                'utilization': processing_util,
                'status': self._get_utilization_status(processing_util, processing_threshold),
                'headroom_percent': (1 - processing_util) * 100
            }
            
            # Overall status
            all_statuses = [comp['status'] for comp in status.values()]
            if 'critical' in all_statuses:
                overall_status = 'critical'
            elif 'warning' in all_statuses:
                overall_status = 'warning'
            else:
                overall_status = 'healthy'
            
            status['overall'] = {
                'status': overall_status,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting current capacity status: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _get_utilization_status(self, utilization: float, thresholds: Dict[str, float]) -> str:
        """Get status based on utilization and thresholds."""
        if utilization >= thresholds['critical']:
            return 'critical'
        elif utilization >= thresholds['warning']:
            return 'warning'
        else:
            return 'healthy'


# Global capacity planner instance
_capacity_planner = None


def get_capacity_planner() -> CapacityPlanner:
    """Get global capacity planner instance."""
    global _capacity_planner
    if _capacity_planner is None:
        _capacity_planner = CapacityPlanner()
    return _capacity_planner