"""
Hyperscale Auto-Scaler

Real-time auto-scaling across multiple cloud regions for billion-user capacity.
Implements intelligent scaling algorithms with predictive capabilities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import asdict
import math

from ..models.hyperscale_models import (
    HyperscaleMetrics, RegionalMetrics, ScalingEvent,
    CloudProvider, ResourceType, ScalingDirection
)


class HyperscaleAutoScaler:
    """
    Real-time auto-scaling system for hyperscale infrastructure
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaling_history: List[ScalingEvent] = []
        self.scaling_policies: Dict[str, Dict] = {}
        self.predictive_models: Dict[str, Any] = {}
        
        # Scaling thresholds for billion-user scale
        self.SCALING_THRESHOLDS = {
            'cpu_scale_up': 70.0,
            'cpu_scale_down': 30.0,
            'memory_scale_up': 80.0,
            'memory_scale_down': 40.0,
            'rps_scale_up': 1000.0,  # per instance
            'rps_scale_down': 200.0,
            'latency_scale_up': 100.0,  # ms
            'error_rate_scale_up': 1.0,  # %
            'min_instances': 100,  # Minimum for billion-user scale
            'max_instances': 100000,  # Maximum per region
            'scale_cooldown': 300  # 5 minutes
        }
    
    async def auto_scale_resources(
        self,
        metrics: HyperscaleMetrics,
        demand_forecast: Optional[Dict[str, float]] = None
    ) -> List[ScalingEvent]:
        """
        Perform real-time auto-scaling based on metrics and forecasts
        """
        self.logger.info("Performing auto-scaling analysis")
        
        scaling_events = []
        
        # Scale each region independently
        for region, regional_metrics in metrics.regional_distribution.items():
            region_events = await self._scale_region(
                region, regional_metrics, demand_forecast
            )
            scaling_events.extend(region_events)
        
        # Global coordination scaling
        global_events = await self._coordinate_global_scaling(
            metrics, scaling_events
        )
        scaling_events.extend(global_events)
        
        # Apply predictive scaling
        predictive_events = await self._apply_predictive_scaling(
            metrics, demand_forecast
        )
        scaling_events.extend(predictive_events)
        
        # Execute scaling events
        for event in scaling_events:
            await self._execute_scaling_event(event)
            self.scaling_history.append(event)
        
        return scaling_events
    
    async def handle_traffic_surge(
        self,
        region: str,
        surge_magnitude: float,
        duration_estimate: int
    ) -> List[ScalingEvent]:
        """
        Handle sudden traffic surges with emergency scaling
        """
        self.logger.warning(f"Handling traffic surge in {region}: {surge_magnitude}x for {duration_estimate}s")
        
        scaling_events = []
        
        # Calculate required scaling
        current_capacity = await self._get_current_capacity(region)
        required_capacity = current_capacity * surge_magnitude
        
        # Emergency compute scaling
        compute_scaling = await self._calculate_emergency_scaling(
            region, ResourceType.COMPUTE, required_capacity
        )
        if compute_scaling:
            scaling_events.append(compute_scaling)
        
        # Network scaling
        network_scaling = await self._calculate_emergency_scaling(
            region, ResourceType.NETWORK, required_capacity * 1.5
        )
        if network_scaling:
            scaling_events.append(network_scaling)
        
        # Cache scaling for reduced database load
        cache_scaling = await self._calculate_emergency_scaling(
            region, ResourceType.CACHE, required_capacity * 0.5
        )
        if cache_scaling:
            scaling_events.append(cache_scaling)
        
        # Execute emergency scaling
        for event in scaling_events:
            event.trigger_metric = "traffic_surge"
            event.trigger_value = surge_magnitude
            await self._execute_scaling_event(event)
            self.scaling_history.append(event)
        
        return scaling_events
    
    async def optimize_scaling_policies(
        self,
        performance_history: List[HyperscaleMetrics]
    ) -> Dict[str, Dict]:
        """
        Optimize scaling policies based on historical performance
        """
        self.logger.info("Optimizing scaling policies")
        
        optimized_policies = {}
        
        # Analyze scaling effectiveness
        effectiveness_analysis = await self._analyze_scaling_effectiveness(
            performance_history
        )
        
        # Optimize thresholds
        for metric_type in ['cpu', 'memory', 'rps', 'latency']:
            optimal_thresholds = await self._optimize_metric_thresholds(
                metric_type, effectiveness_analysis
            )
            optimized_policies[metric_type] = optimal_thresholds
        
        # Update scaling policies
        self.scaling_policies.update(optimized_policies)
        
        return optimized_policies
    
    async def predict_scaling_needs(
        self,
        current_metrics: HyperscaleMetrics,
        forecast_horizon: int = 3600  # 1 hour
    ) -> Dict[str, List[ScalingEvent]]:
        """
        Predict future scaling needs using ML models
        """
        self.logger.info(f"Predicting scaling needs for {forecast_horizon}s horizon")
        
        predictions = {}
        
        for region, regional_metrics in current_metrics.regional_distribution.items():
            # Predict resource demand
            demand_prediction = await self._predict_regional_demand(
                region, regional_metrics, forecast_horizon
            )
            
            # Calculate required scaling events
            required_events = await self._calculate_predictive_scaling(
                region, demand_prediction
            )
            
            predictions[region] = required_events
        
        return predictions
    
    async def _scale_region(
        self,
        region: str,
        metrics: RegionalMetrics,
        demand_forecast: Optional[Dict[str, float]]
    ) -> List[ScalingEvent]:
        """Scale resources in a specific region"""
        
        scaling_events = []
        
        # CPU-based scaling
        if metrics.cpu_utilization > self.SCALING_THRESHOLDS['cpu_scale_up']:
            event = await self._create_scaling_event(
                region, ResourceType.COMPUTE, ScalingDirection.OUT,
                'cpu_utilization', metrics.cpu_utilization,
                self.SCALING_THRESHOLDS['cpu_scale_up']
            )
            scaling_events.append(event)
        elif metrics.cpu_utilization < self.SCALING_THRESHOLDS['cpu_scale_down']:
            event = await self._create_scaling_event(
                region, ResourceType.COMPUTE, ScalingDirection.IN,
                'cpu_utilization', metrics.cpu_utilization,
                self.SCALING_THRESHOLDS['cpu_scale_down']
            )
            scaling_events.append(event)
        
        # Memory-based scaling
        if metrics.memory_utilization > self.SCALING_THRESHOLDS['memory_scale_up']:
            event = await self._create_scaling_event(
                region, ResourceType.COMPUTE, ScalingDirection.UP,
                'memory_utilization', metrics.memory_utilization,
                self.SCALING_THRESHOLDS['memory_scale_up']
            )
            scaling_events.append(event)
        
        # RPS-based scaling
        current_rps_per_instance = await self._calculate_rps_per_instance(region)
        if current_rps_per_instance > self.SCALING_THRESHOLDS['rps_scale_up']:
            event = await self._create_scaling_event(
                region, ResourceType.COMPUTE, ScalingDirection.OUT,
                'rps_per_instance', current_rps_per_instance,
                self.SCALING_THRESHOLDS['rps_scale_up']
            )
            scaling_events.append(event)
        
        # Latency-based scaling
        if metrics.latency_p95 > self.SCALING_THRESHOLDS['latency_scale_up']:
            event = await self._create_scaling_event(
                region, ResourceType.COMPUTE, ScalingDirection.OUT,
                'latency_p95', metrics.latency_p95,
                self.SCALING_THRESHOLDS['latency_scale_up']
            )
            scaling_events.append(event)
        
        return scaling_events
    
    async def _coordinate_global_scaling(
        self,
        metrics: HyperscaleMetrics,
        regional_events: List[ScalingEvent]
    ) -> List[ScalingEvent]:
        """Coordinate scaling across regions"""
        
        global_events = []
        
        # Check if multiple regions are scaling up simultaneously
        scale_up_regions = [
            event.region for event in regional_events
            if event.direction in [ScalingDirection.UP, ScalingDirection.OUT]
        ]
        
        if len(scale_up_regions) > 5:  # More than 5 regions scaling up
            # Consider global load redistribution
            redistribution_event = await self._create_load_redistribution_event(
                scale_up_regions, metrics
            )
            if redistribution_event:
                global_events.append(redistribution_event)
        
        # Global capacity management
        total_capacity_utilization = await self._calculate_global_capacity_utilization(metrics)
        if total_capacity_utilization > 80.0:
            # Activate reserve capacity
            reserve_events = await self._activate_global_reserves(metrics)
            global_events.extend(reserve_events)
        
        return global_events
    
    async def _apply_predictive_scaling(
        self,
        metrics: HyperscaleMetrics,
        demand_forecast: Optional[Dict[str, float]]
    ) -> List[ScalingEvent]:
        """Apply predictive scaling based on forecasts"""
        
        if not demand_forecast:
            return []
        
        predictive_events = []
        
        for region, regional_metrics in metrics.regional_distribution.items():
            forecast_key = f"{region}_demand"
            if forecast_key in demand_forecast:
                predicted_demand = demand_forecast[forecast_key]
                current_demand = regional_metrics.requests_per_second
                
                demand_ratio = predicted_demand / max(current_demand, 1)
                
                if demand_ratio > 1.5:  # 50% increase predicted
                    event = await self._create_predictive_scaling_event(
                        region, demand_ratio
                    )
                    predictive_events.append(event)
        
        return predictive_events
    
    async def _create_scaling_event(
        self,
        region: str,
        resource_type: ResourceType,
        direction: ScalingDirection,
        trigger_metric: str,
        trigger_value: float,
        threshold: float
    ) -> ScalingEvent:
        """Create a scaling event"""
        
        # Calculate scale factor
        if direction in [ScalingDirection.UP, ScalingDirection.OUT]:
            scale_factor = min(2.0, trigger_value / threshold)
        else:
            scale_factor = max(0.5, threshold / trigger_value)
        
        # Get current instances
        current_instances = await self._get_current_instances(region, resource_type)
        
        # Calculate new instances
        if direction == ScalingDirection.OUT:
            new_instances = min(
                int(current_instances * scale_factor),
                self.SCALING_THRESHOLDS['max_instances']
            )
        elif direction == ScalingDirection.IN:
            new_instances = max(
                int(current_instances / scale_factor),
                self.SCALING_THRESHOLDS['min_instances']
            )
        else:
            new_instances = current_instances
        
        # Calculate cost impact
        cost_impact = await self._calculate_scaling_cost_impact(
            resource_type, current_instances, new_instances
        )
        
        return ScalingEvent(
            id="",
            timestamp=datetime.now(),
            region=region,
            resource_type=resource_type,
            direction=direction,
            scale_factor=scale_factor,
            trigger_metric=trigger_metric,
            trigger_value=trigger_value,
            threshold=threshold,
            instances_before=current_instances,
            instances_after=new_instances,
            cost_impact=cost_impact,
            performance_impact=await self._estimate_performance_impact(
                resource_type, current_instances, new_instances
            )
        )
    
    async def _execute_scaling_event(self, event: ScalingEvent) -> bool:
        """Execute a scaling event"""
        
        self.logger.info(f"Executing scaling event: {event.region} {event.resource_type} {event.direction}")
        
        try:
            # Check cooldown period
            if not await self._check_scaling_cooldown(event.region, event.resource_type):
                self.logger.warning(f"Scaling cooldown active for {event.region}")
                return False
            
            # Execute the scaling operation
            success = await self._perform_scaling_operation(event)
            
            if success:
                self.logger.info(f"Scaling successful: {event.instances_before} -> {event.instances_after}")
            else:
                self.logger.error(f"Scaling failed for {event.region}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error executing scaling event: {e}")
            return False
    
    async def _get_current_capacity(self, region: str) -> float:
        """Get current capacity for region"""
        # Mock implementation
        return 10000.0  # Current RPS capacity
    
    async def _calculate_emergency_scaling(
        self,
        region: str,
        resource_type: ResourceType,
        required_capacity: float
    ) -> Optional[ScalingEvent]:
        """Calculate emergency scaling requirements"""
        
        current_instances = await self._get_current_instances(region, resource_type)
        current_capacity = current_instances * 100  # Assume 100 RPS per instance
        
        if required_capacity > current_capacity:
            required_instances = math.ceil(required_capacity / 100)
            scale_factor = required_instances / current_instances
            
            return ScalingEvent(
                id="",
                timestamp=datetime.now(),
                region=region,
                resource_type=resource_type,
                direction=ScalingDirection.OUT,
                scale_factor=scale_factor,
                trigger_metric="emergency_scaling",
                trigger_value=required_capacity,
                threshold=current_capacity,
                instances_before=current_instances,
                instances_after=required_instances,
                cost_impact=await self._calculate_scaling_cost_impact(
                    resource_type, current_instances, required_instances
                ),
                performance_impact={}
            )
        
        return None
    
    async def _get_current_instances(self, region: str, resource_type: ResourceType) -> int:
        """Get current number of instances"""
        # Mock implementation
        base_instances = {
            ResourceType.COMPUTE: 1000,
            ResourceType.STORAGE: 100,
            ResourceType.NETWORK: 50,
            ResourceType.DATABASE: 10,
            ResourceType.CACHE: 20,
            ResourceType.CDN: 5
        }
        return base_instances.get(resource_type, 100)
    
    async def _calculate_rps_per_instance(self, region: str) -> float:
        """Calculate requests per second per instance"""
        # Mock implementation
        return 500.0  # 500 RPS per instance
    
    async def _calculate_scaling_cost_impact(
        self,
        resource_type: ResourceType,
        current_instances: int,
        new_instances: int
    ) -> float:
        """Calculate cost impact of scaling"""
        
        cost_per_instance = {
            ResourceType.COMPUTE: 100,  # $100/month
            ResourceType.STORAGE: 50,
            ResourceType.NETWORK: 200,
            ResourceType.DATABASE: 500,
            ResourceType.CACHE: 150,
            ResourceType.CDN: 300
        }
        
        instance_cost = cost_per_instance.get(resource_type, 100)
        cost_difference = (new_instances - current_instances) * instance_cost
        
        return cost_difference
    
    async def _estimate_performance_impact(
        self,
        resource_type: ResourceType,
        current_instances: int,
        new_instances: int
    ) -> Dict[str, float]:
        """Estimate performance impact of scaling"""
        
        scale_ratio = new_instances / max(current_instances, 1)
        
        return {
            'throughput_change': (scale_ratio - 1) * 100,  # Percentage change
            'latency_change': -(scale_ratio - 1) * 20,  # Inverse relationship
            'availability_change': min(5, (scale_ratio - 1) * 10)
        }
    
    async def _check_scaling_cooldown(self, region: str, resource_type: ResourceType) -> bool:
        """Check if scaling cooldown period has passed"""
        
        cooldown_key = f"{region}_{resource_type}"
        cooldown_seconds = self.SCALING_THRESHOLDS['scale_cooldown']
        
        # Check recent scaling events
        recent_events = [
            event for event in self.scaling_history[-100:]  # Last 100 events
            if (event.region == region and 
                event.resource_type == resource_type and
                (datetime.now() - event.timestamp).total_seconds() < cooldown_seconds)
        ]
        
        return len(recent_events) == 0
    
    async def _perform_scaling_operation(self, event: ScalingEvent) -> bool:
        """Perform the actual scaling operation"""
        # Mock implementation - would integrate with cloud APIs
        self.logger.info(f"Scaling {event.resource_type} in {event.region} from {event.instances_before} to {event.instances_after}")
        return True
    
    async def _calculate_global_capacity_utilization(self, metrics: HyperscaleMetrics) -> float:
        """Calculate global capacity utilization"""
        if not metrics.regional_distribution:
            return 0.0
        
        total_cpu = sum(m.cpu_utilization for m in metrics.regional_distribution.values())
        avg_cpu = total_cpu / len(metrics.regional_distribution)
        return avg_cpu
    
    async def _create_load_redistribution_event(
        self, 
        scale_up_regions: List[str], 
        metrics: HyperscaleMetrics
    ) -> Optional[ScalingEvent]:
        """Create load redistribution event"""
        if len(scale_up_regions) < 3:
            return None
        
        return ScalingEvent(
            id="",
            timestamp=datetime.now(),
            region="global",
            resource_type=ResourceType.NETWORK,
            direction=ScalingDirection.OUT,
            scale_factor=1.2,
            trigger_metric="global_load_redistribution",
            trigger_value=len(scale_up_regions),
            threshold=5.0,
            instances_before=0,
            instances_after=0,
            cost_impact=0.0,
            performance_impact={"load_distribution": 20.0}
        )
    
    async def _activate_global_reserves(self, metrics: HyperscaleMetrics) -> List[ScalingEvent]:
        """Activate global reserve capacity"""
        reserve_events = []
        
        for region in list(metrics.regional_distribution.keys())[:3]:  # Top 3 regions
            reserve_event = ScalingEvent(
                id="",
                timestamp=datetime.now(),
                region=region,
                resource_type=ResourceType.COMPUTE,
                direction=ScalingDirection.OUT,
                scale_factor=1.5,
                trigger_metric="global_reserve_activation",
                trigger_value=80.0,
                threshold=80.0,
                instances_before=1000,
                instances_after=1500,
                cost_impact=50000.0,
                performance_impact={"capacity_increase": 50.0}
            )
            reserve_events.append(reserve_event)
        
        return reserve_events
    
    async def _create_predictive_scaling_event(
        self, 
        region: str, 
        demand_ratio: float
    ) -> ScalingEvent:
        """Create predictive scaling event"""
        return ScalingEvent(
            id="",
            timestamp=datetime.now(),
            region=region,
            resource_type=ResourceType.COMPUTE,
            direction=ScalingDirection.OUT,
            scale_factor=demand_ratio,
            trigger_metric="predictive_scaling",
            trigger_value=demand_ratio,
            threshold=1.5,
            instances_before=1000,
            instances_after=int(1000 * demand_ratio),
            cost_impact=1000.0 * (demand_ratio - 1),
            performance_impact={"predicted_capacity": (demand_ratio - 1) * 100}
        )