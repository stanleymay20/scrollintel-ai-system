"""
Innovation Pipeline Optimizer

This module implements the core optimization engine for managing innovation pipelines,
including flow optimization, resource allocation, and performance monitoring.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
import numpy as np
from dataclasses import asdict

from ..models.innovation_pipeline_models import (
    InnovationPipelineItem, PipelineStage, InnovationPriority, ResourceType,
    PipelineStatus, ResourceAllocation, ResourceRequirement, PipelineMetrics,
    PipelineOptimizationConfig, PipelineOptimizationResult, PipelinePerformanceReport
)


class InnovationPipelineOptimizer:
    """
    Core engine for optimizing innovation pipeline flow and resource allocation
    """
    
    def __init__(self, config: Optional[PipelineOptimizationConfig] = None):
        self.config = config or PipelineOptimizationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Pipeline state
        self.pipeline_items: Dict[str, InnovationPipelineItem] = {}
        self.resource_pool: Dict[ResourceType, float] = {}
        self.stage_capacities: Dict[PipelineStage, int] = {}
        
        # Performance tracking
        self.historical_metrics: List[PipelinePerformanceReport] = []
        self.optimization_history: List[PipelineOptimizationResult] = []
        
        # Optimization state
        self.last_optimization: Optional[datetime] = None
        self._optimization_lock = asyncio.Lock()
        
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize pipeline with default configurations"""
        # Set default stage capacities
        self.stage_capacities = {
            PipelineStage.IDEATION: 50,
            PipelineStage.RESEARCH: 30,
            PipelineStage.EXPERIMENTATION: 20,
            PipelineStage.PROTOTYPING: 15,
            PipelineStage.VALIDATION: 10,
            PipelineStage.OPTIMIZATION: 8,
            PipelineStage.DEPLOYMENT: 5,
            PipelineStage.MONITORING: 100
        }
        
        # Initialize resource pool
        self.resource_pool = {
            ResourceType.COMPUTE: 1000.0,
            ResourceType.STORAGE: 10000.0,
            ResourceType.BANDWIDTH: 1000.0,
            ResourceType.RESEARCH_TIME: 2000.0,
            ResourceType.DEVELOPMENT_TIME: 1500.0,
            ResourceType.TESTING_TIME: 1000.0,
            ResourceType.BUDGET: 1000000.0
        }
    
    async def add_innovation_to_pipeline(self, innovation_item: InnovationPipelineItem) -> bool:
        """Add a new innovation to the pipeline"""
        try:
            # Check capacity constraints
            if not self._check_capacity_constraints(innovation_item):
                self.logger.warning(f"Cannot add innovation {innovation_item.id} - capacity constraints")
                return False
            
            # Add to pipeline
            self.pipeline_items[innovation_item.id] = innovation_item
            
            # Allocate initial resources
            await self._allocate_resources(innovation_item)
            
            self.logger.info(f"Added innovation {innovation_item.id} to pipeline at stage {innovation_item.current_stage}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding innovation to pipeline: {e}")
            return False
    
    async def optimize_pipeline_flow(self) -> PipelineOptimizationResult:
        """Optimize the entire pipeline flow for maximum efficiency"""
        # Try to acquire lock without blocking
        try:
            # Use asyncio.wait_for with timeout=0 to check if lock is available
            await asyncio.wait_for(self._optimization_lock.acquire(), timeout=0.001)
        except asyncio.TimeoutError:
            self.logger.warning("Optimization already running")
            return PipelineOptimizationResult(
                optimization_score=0.0,
                confidence_level=0.0,
                recommendations=["Optimization already in progress"],
                warnings=["Cannot run concurrent optimizations"]
            )
        
        try:
            optimization_start = datetime.utcnow()
            
            try:
                result = PipelineOptimizationResult()
                
                # Analyze current pipeline state
                bottlenecks = await self._identify_bottlenecks()
                resource_utilization = await self._calculate_resource_utilization()
                
                # Optimize resource allocation
                resource_optimizations = await self._optimize_resource_allocation()
                result.resource_reallocations = resource_optimizations
                
                # Optimize priority assignments
                priority_optimizations = await self._optimize_priorities()
                result.priority_adjustments = priority_optimizations
                
                # Optimize stage transitions
                stage_optimizations = await self._optimize_stage_transitions()
                result.stage_transitions = stage_optimizations
                
                # Calculate expected improvements
                result.expected_throughput_improvement = await self._calculate_throughput_improvement(result)
                result.expected_cycle_time_reduction = await self._calculate_cycle_time_reduction(result)
                result.expected_resource_savings = await self._calculate_resource_savings(result)
                
                # Generate optimization score
                result.optimization_score = await self._calculate_optimization_score(result)
                result.confidence_level = await self._calculate_confidence_level(result)
                
                # Generate recommendations
                result.recommendations = await self._generate_recommendations(bottlenecks, resource_utilization)
                result.warnings = await self._generate_warnings(bottlenecks, resource_utilization)
                
                # Apply optimizations
                await self._apply_optimizations(result)
                
                # Update optimization history
                self.optimization_history.append(result)
                self.last_optimization = optimization_start
                
                self.logger.info(f"Pipeline optimization completed with score: {result.optimization_score}")
                return result
                
            except Exception as e:
                self.logger.error(f"Error during pipeline optimization: {e}")
                return PipelineOptimizationResult()
        finally:
            self._optimization_lock.release()
    
    async def monitor_pipeline_performance(self) -> PipelinePerformanceReport:
        """Generate comprehensive pipeline performance report"""
        try:
            report = PipelinePerformanceReport()
            report.period_start = datetime.utcnow() - timedelta(hours=24)
            report.period_end = datetime.utcnow()
            
            # Calculate overall metrics
            active_items = [item for item in self.pipeline_items.values() 
                          if item.status == PipelineStatus.ACTIVE]
            
            report.total_innovations = len(self.pipeline_items)
            report.active_innovations = len(active_items)
            report.completed_innovations = len([item for item in self.pipeline_items.values() 
                                              if item.status == PipelineStatus.COMPLETED])
            report.failed_innovations = len([item for item in self.pipeline_items.values() 
                                           if item.status == PipelineStatus.FAILED])
            
            # Calculate stage-wise metrics
            report.stage_metrics = await self._calculate_stage_metrics()
            
            # Calculate resource metrics
            report.total_resources_allocated = await self._calculate_total_resource_allocation()
            report.total_resources_used = await self._calculate_total_resource_usage()
            report.resource_efficiency = await self._calculate_resource_efficiency()
            
            # Calculate performance indicators
            report.overall_throughput = await self._calculate_overall_throughput()
            report.average_cycle_time = await self._calculate_average_cycle_time()
            report.overall_success_rate = await self._calculate_overall_success_rate()
            report.cost_per_innovation = await self._calculate_cost_per_innovation()
            
            # Identify bottlenecks
            bottlenecks = await self._identify_bottlenecks()
            report.identified_bottlenecks = list(bottlenecks.keys())
            report.bottleneck_severity = bottlenecks
            
            # Calculate trends
            report.throughput_trend = await self._calculate_throughput_trend()
            report.quality_trend = await self._calculate_quality_trend()
            report.efficiency_trend = await self._calculate_efficiency_trend()
            
            # Generate recommendations
            report.optimization_recommendations = await self._generate_optimization_recommendations()
            report.capacity_recommendations = await self._generate_capacity_recommendations()
            report.process_improvements = await self._generate_process_improvements()
            
            # Store report
            self.historical_metrics.append(report)
            
            self.logger.info(f"Generated pipeline performance report with {report.total_innovations} innovations")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return PipelinePerformanceReport()
    
    async def prioritize_innovations(self, criteria: Dict[str, float]) -> Dict[str, InnovationPriority]:
        """Prioritize innovations based on multiple criteria"""
        try:
            prioritization_results = {}
            
            for innovation_id, item in self.pipeline_items.items():
                if item.status != PipelineStatus.ACTIVE:
                    continue
                
                # Calculate priority score
                priority_score = 0.0
                
                # Impact score weight
                if 'impact' in criteria:
                    priority_score += item.impact_score * criteria['impact']
                
                # Success probability weight
                if 'success_probability' in criteria:
                    priority_score += item.success_probability * criteria['success_probability']
                
                # Risk score weight (inverse)
                if 'risk' in criteria:
                    priority_score += (1.0 - item.risk_score) * criteria['risk']
                
                # Resource efficiency weight
                if 'resource_efficiency' in criteria:
                    efficiency = await self._calculate_innovation_resource_efficiency(item)
                    priority_score += efficiency * criteria['resource_efficiency']
                
                # Time sensitivity weight
                if 'time_sensitivity' in criteria:
                    time_factor = await self._calculate_time_sensitivity(item)
                    priority_score += time_factor * criteria['time_sensitivity']
                
                # Convert score to priority level
                if priority_score >= 0.8:
                    prioritization_results[innovation_id] = InnovationPriority.CRITICAL
                elif priority_score >= 0.6:
                    prioritization_results[innovation_id] = InnovationPriority.HIGH
                elif priority_score >= 0.4:
                    prioritization_results[innovation_id] = InnovationPriority.MEDIUM
                else:
                    prioritization_results[innovation_id] = InnovationPriority.LOW
            
            self.logger.info(f"Prioritized {len(prioritization_results)} innovations")
            return prioritization_results
            
        except Exception as e:
            self.logger.error(f"Error prioritizing innovations: {e}")
            return {}
    
    async def allocate_pipeline_resources(self, allocation_strategy: str = "balanced") -> Dict[str, List[ResourceAllocation]]:
        """Allocate resources across pipeline innovations"""
        try:
            allocation_results = defaultdict(list)
            
            # Get active innovations sorted by priority
            active_items = sorted(
                [item for item in self.pipeline_items.values() if item.status == PipelineStatus.ACTIVE],
                key=lambda x: self._get_priority_weight(x.priority),
                reverse=True
            )
            
            # Calculate available resources
            available_resources = await self._calculate_available_resources()
            
            for item in active_items:
                for requirement in item.resource_requirements:
                    if available_resources[requirement.resource_type] >= requirement.amount:
                        # Create allocation
                        allocation = ResourceAllocation(
                            innovation_id=item.id,
                            resource_type=requirement.resource_type,
                            allocated_amount=requirement.amount,
                            allocation_time=datetime.utcnow(),
                            expected_completion=datetime.utcnow() + timedelta(hours=requirement.duration or 24)
                        )
                        
                        # Apply allocation strategy
                        if allocation_strategy == "aggressive":
                            allocation.allocated_amount *= 1.2
                        elif allocation_strategy == "conservative":
                            allocation.allocated_amount *= 0.8
                        
                        # Update available resources
                        available_resources[requirement.resource_type] -= allocation.allocated_amount
                        
                        # Add to results
                        allocation_results[item.id].append(allocation)
                        item.resource_allocations.append(allocation)
            
            self.logger.info(f"Allocated resources to {len(allocation_results)} innovations")
            return dict(allocation_results)
            
        except Exception as e:
            self.logger.error(f"Error allocating pipeline resources: {e}")
            return {}
    
    # Helper methods
    
    def _check_capacity_constraints(self, item: InnovationPipelineItem) -> bool:
        """Check if pipeline has capacity for new innovation"""
        stage_count = len([i for i in self.pipeline_items.values() 
                          if i.current_stage == item.current_stage and i.status == PipelineStatus.ACTIVE])
        return stage_count < self.stage_capacities.get(item.current_stage, 10)
    
    async def _allocate_resources(self, item: InnovationPipelineItem):
        """Allocate initial resources to innovation"""
        for requirement in item.resource_requirements:
            if self.resource_pool.get(requirement.resource_type, 0) >= requirement.amount:
                allocation = ResourceAllocation(
                    innovation_id=item.id,
                    resource_type=requirement.resource_type,
                    allocated_amount=requirement.amount,
                    allocation_time=datetime.utcnow()
                )
                item.resource_allocations.append(allocation)
                self.resource_pool[requirement.resource_type] -= requirement.amount
    
    async def _identify_bottlenecks(self) -> Dict[PipelineStage, float]:
        """Identify pipeline bottlenecks"""
        bottlenecks = {}
        
        for stage in PipelineStage:
            stage_items = [item for item in self.pipeline_items.values() 
                          if item.current_stage == stage and item.status == PipelineStatus.ACTIVE]
            
            capacity = self.stage_capacities.get(stage, 10)
            utilization = len(stage_items) / capacity if capacity > 0 else 0
            
            if utilization >= self.config.bottleneck_threshold:
                bottlenecks[stage] = utilization
        
        return bottlenecks
    
    async def _calculate_resource_utilization(self) -> Dict[ResourceType, float]:
        """Calculate current resource utilization"""
        utilization = {}
        
        for resource_type, total in self.resource_pool.items():
            used = sum(
                allocation.allocated_amount 
                for item in self.pipeline_items.values()
                for allocation in item.resource_allocations
                if allocation.resource_type == resource_type
            )
            utilization[resource_type] = used / total if total > 0 else 0
        
        return utilization
    
    async def _optimize_resource_allocation(self) -> List[ResourceAllocation]:
        """Optimize resource allocation across innovations"""
        optimizations = []
        
        # Identify underutilized resources
        utilization = await self._calculate_resource_utilization()
        
        for resource_type, util in utilization.items():
            if util < self.config.resource_utilization_target:
                # Find innovations that could use more resources
                for item in self.pipeline_items.values():
                    if item.status == PipelineStatus.ACTIVE:
                        for allocation in item.resource_allocations:
                            if allocation.resource_type == resource_type and allocation.efficiency_score < 0.8:
                                # Increase allocation
                                new_allocation = ResourceAllocation(
                                    innovation_id=item.id,
                                    resource_type=resource_type,
                                    allocated_amount=allocation.allocated_amount * 1.2,
                                    allocation_time=datetime.utcnow()
                                )
                                optimizations.append(new_allocation)
        
        return optimizations
    
    async def _optimize_priorities(self) -> Dict[str, InnovationPriority]:
        """Optimize innovation priorities"""
        criteria = {
            'impact': 0.3,
            'success_probability': 0.3,
            'risk': 0.2,
            'resource_efficiency': 0.2
        }
        return await self.prioritize_innovations(criteria)
    
    async def _optimize_stage_transitions(self) -> Dict[str, PipelineStage]:
        """Optimize stage transitions for innovations"""
        transitions = {}
        
        for item in self.pipeline_items.values():
            if item.status == PipelineStatus.ACTIVE:
                # Check if innovation is ready for next stage
                if await self._is_ready_for_next_stage(item):
                    next_stage = self._get_next_stage(item.current_stage)
                    if next_stage and self._check_stage_capacity(next_stage):
                        transitions[item.id] = next_stage
        
        return transitions
    
    def _get_priority_weight(self, priority: InnovationPriority) -> float:
        """Get numeric weight for priority level"""
        weights = {
            InnovationPriority.CRITICAL: self.config.priority_weight_critical,
            InnovationPriority.HIGH: self.config.priority_weight_high,
            InnovationPriority.MEDIUM: self.config.priority_weight_medium,
            InnovationPriority.LOW: self.config.priority_weight_low
        }
        return weights.get(priority, 0.5)
    
    async def _calculate_available_resources(self) -> Dict[ResourceType, float]:
        """Calculate currently available resources"""
        available = self.resource_pool.copy()
        
        for item in self.pipeline_items.values():
            for allocation in item.resource_allocations:
                if allocation.expected_completion and allocation.expected_completion > datetime.utcnow():
                    available[allocation.resource_type] -= allocation.allocated_amount
        
        return available
    
    async def _calculate_throughput_improvement(self, result: PipelineOptimizationResult) -> float:
        """Calculate expected throughput improvement"""
        # Simplified calculation based on resource reallocations and priority adjustments
        base_improvement = len(result.resource_reallocations) * 0.05
        priority_improvement = len(result.priority_adjustments) * 0.03
        stage_improvement = len(result.stage_transitions) * 0.02
        
        return min(base_improvement + priority_improvement + stage_improvement, 0.5)
    
    async def _calculate_cycle_time_reduction(self, result: PipelineOptimizationResult) -> float:
        """Calculate expected cycle time reduction"""
        # Simplified calculation
        return len(result.stage_transitions) * 0.1
    
    async def _calculate_resource_savings(self, result: PipelineOptimizationResult) -> float:
        """Calculate expected resource savings"""
        # Simplified calculation
        return len(result.resource_reallocations) * 0.05
    
    async def _calculate_optimization_score(self, result: PipelineOptimizationResult) -> float:
        """Calculate overall optimization score"""
        score = (
            result.expected_throughput_improvement * 0.4 +
            result.expected_cycle_time_reduction * 0.3 +
            result.expected_resource_savings * 0.3
        )
        return min(score, 1.0)
    
    async def _calculate_confidence_level(self, result: PipelineOptimizationResult) -> float:
        """Calculate confidence level for optimization"""
        # Based on historical optimization success
        if len(self.optimization_history) < 5:
            return 0.7
        
        recent_scores = [opt.optimization_score for opt in self.optimization_history[-5:]]
        return min(np.mean(recent_scores), 0.95)
    
    async def _generate_recommendations(self, bottlenecks: Dict[PipelineStage, float], 
                                      utilization: Dict[ResourceType, float]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Bottleneck recommendations
        for stage, severity in bottlenecks.items():
            if severity > 0.9:
                recommendations.append(f"Critical bottleneck at {stage.value} stage - consider increasing capacity")
            elif severity > 0.8:
                recommendations.append(f"High utilization at {stage.value} stage - monitor closely")
        
        # Resource recommendations
        for resource_type, util in utilization.items():
            if util > 0.95:
                recommendations.append(f"Very high {resource_type.value} utilization - consider scaling up")
            elif util < 0.5:
                recommendations.append(f"Low {resource_type.value} utilization - consider reallocation")
        
        return recommendations
    
    async def _generate_warnings(self, bottlenecks: Dict[PipelineStage, float], 
                                utilization: Dict[ResourceType, float]) -> List[str]:
        """Generate optimization warnings"""
        warnings = []
        
        # Critical bottlenecks
        critical_bottlenecks = [stage for stage, severity in bottlenecks.items() if severity > 0.95]
        if critical_bottlenecks:
            warnings.append(f"Critical bottlenecks detected: {[s.value for s in critical_bottlenecks]}")
        
        # Resource exhaustion
        exhausted_resources = [rt for rt, util in utilization.items() if util > 0.98]
        if exhausted_resources:
            warnings.append(f"Near resource exhaustion: {[r.value for r in exhausted_resources]}")
        
        return warnings
    
    async def _apply_optimizations(self, result: PipelineOptimizationResult):
        """Apply optimization results to pipeline"""
        # Apply resource reallocations
        for allocation in result.resource_reallocations:
            if allocation.innovation_id in self.pipeline_items:
                self.pipeline_items[allocation.innovation_id].resource_allocations.append(allocation)
        
        # Apply priority adjustments
        for innovation_id, new_priority in result.priority_adjustments.items():
            if innovation_id in self.pipeline_items:
                self.pipeline_items[innovation_id].priority = new_priority
        
        # Apply stage transitions
        for innovation_id, new_stage in result.stage_transitions.items():
            if innovation_id in self.pipeline_items:
                item = self.pipeline_items[innovation_id]
                item.current_stage = new_stage
                item.stage_entered_at = datetime.utcnow()
    
    # Additional helper methods for metrics calculation
    
    async def _calculate_stage_metrics(self) -> Dict[PipelineStage, PipelineMetrics]:
        """Calculate metrics for each pipeline stage"""
        metrics = {}
        
        for stage in PipelineStage:
            stage_items = [item for item in self.pipeline_items.values() 
                          if item.current_stage == stage]
            
            if stage_items:
                metrics[stage] = PipelineMetrics(
                    stage=stage,
                    throughput=len(stage_items) / 24.0,  # per hour
                    cycle_time=np.mean([
                        (datetime.utcnow() - item.stage_entered_at).total_seconds() / 3600
                        for item in stage_items
                    ]),
                    success_rate=len([item for item in stage_items if item.success_probability > 0.7]) / len(stage_items),
                    resource_utilization=0.8,  # Simplified
                    bottleneck_score=len(stage_items) / self.stage_capacities.get(stage, 10),
                    quality_score=np.mean([item.impact_score for item in stage_items]),
                    cost_efficiency=0.75  # Simplified
                )
        
        return metrics
    
    async def _calculate_total_resource_allocation(self) -> Dict[ResourceType, float]:
        """Calculate total resource allocation by type"""
        totals = defaultdict(float)
        
        for item in self.pipeline_items.values():
            for allocation in item.resource_allocations:
                totals[allocation.resource_type] += allocation.allocated_amount
        
        return dict(totals)
    
    async def _calculate_total_resource_usage(self) -> Dict[ResourceType, float]:
        """Calculate total resource usage by type"""
        totals = defaultdict(float)
        
        for item in self.pipeline_items.values():
            for allocation in item.resource_allocations:
                totals[allocation.resource_type] += allocation.used_amount
        
        return dict(totals)
    
    async def _calculate_resource_efficiency(self) -> Dict[ResourceType, float]:
        """Calculate resource efficiency by type"""
        allocated = await self._calculate_total_resource_allocation()
        used = await self._calculate_total_resource_usage()
        
        efficiency = {}
        for resource_type in ResourceType:
            if allocated.get(resource_type, 0) > 0:
                efficiency[resource_type] = used.get(resource_type, 0) / allocated[resource_type]
            else:
                efficiency[resource_type] = 0.0
        
        return efficiency
    
    async def _calculate_overall_throughput(self) -> float:
        """Calculate overall pipeline throughput"""
        completed_last_24h = len([
            item for item in self.pipeline_items.values()
            if item.status == PipelineStatus.COMPLETED and
            item.actual_completion and
            item.actual_completion > datetime.utcnow() - timedelta(hours=24)
        ])
        return completed_last_24h
    
    async def _calculate_average_cycle_time(self) -> float:
        """Calculate average cycle time across all innovations"""
        completed_items = [
            item for item in self.pipeline_items.values()
            if item.status == PipelineStatus.COMPLETED and item.actual_completion
        ]
        
        if not completed_items:
            return 0.0
        
        cycle_times = [
            (item.actual_completion - item.created_at).total_seconds() / 3600
            for item in completed_items
        ]
        
        return np.mean(cycle_times)
    
    async def _calculate_overall_success_rate(self) -> float:
        """Calculate overall pipeline success rate"""
        completed_items = [
            item for item in self.pipeline_items.values()
            if item.status in [PipelineStatus.COMPLETED, PipelineStatus.FAILED]
        ]
        
        if not completed_items:
            return 0.0
        
        successful_items = [item for item in completed_items if item.status == PipelineStatus.COMPLETED]
        return len(successful_items) / len(completed_items)
    
    async def _calculate_cost_per_innovation(self) -> float:
        """Calculate average cost per innovation"""
        total_budget_used = sum(
            allocation.used_amount
            for item in self.pipeline_items.values()
            for allocation in item.resource_allocations
            if allocation.resource_type == ResourceType.BUDGET
        )
        
        completed_innovations = len([
            item for item in self.pipeline_items.values()
            if item.status == PipelineStatus.COMPLETED
        ])
        
        if completed_innovations == 0:
            return 0.0
        
        return total_budget_used / completed_innovations
    
    async def _calculate_throughput_trend(self) -> List[float]:
        """Calculate throughput trend over time"""
        # Simplified - return last 10 data points
        return [float(i) for i in range(10)]
    
    async def _calculate_quality_trend(self) -> List[float]:
        """Calculate quality trend over time"""
        # Simplified - return last 10 data points
        return [0.8 + i * 0.01 for i in range(10)]
    
    async def _calculate_efficiency_trend(self) -> List[float]:
        """Calculate efficiency trend over time"""
        # Simplified - return last 10 data points
        return [0.75 + i * 0.02 for i in range(10)]
    
    async def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        return [
            "Consider increasing compute resources for experimentation stage",
            "Implement parallel processing for validation stage",
            "Optimize resource allocation algorithm for better efficiency"
        ]
    
    async def _generate_capacity_recommendations(self) -> List[str]:
        """Generate capacity recommendations"""
        return [
            "Increase prototyping stage capacity by 20%",
            "Add more research time allocation",
            "Consider expanding testing infrastructure"
        ]
    
    async def _generate_process_improvements(self) -> List[str]:
        """Generate process improvement recommendations"""
        return [
            "Implement automated stage transition criteria",
            "Add real-time bottleneck detection",
            "Enhance resource utilization monitoring"
        ]
    
    async def _calculate_innovation_resource_efficiency(self, item: InnovationPipelineItem) -> float:
        """Calculate resource efficiency for specific innovation"""
        if not item.resource_allocations:
            return 0.0
        
        total_allocated = sum(allocation.allocated_amount for allocation in item.resource_allocations)
        total_used = sum(allocation.used_amount for allocation in item.resource_allocations)
        
        if total_allocated == 0:
            return 0.0
        
        return total_used / total_allocated
    
    async def _calculate_time_sensitivity(self, item: InnovationPipelineItem) -> float:
        """Calculate time sensitivity factor for innovation"""
        if not item.estimated_completion:
            return 0.5
        
        time_remaining = (item.estimated_completion - datetime.utcnow()).total_seconds()
        time_total = (item.estimated_completion - item.created_at).total_seconds()
        
        if time_total <= 0:
            return 1.0
        
        return max(0.0, min(1.0, 1.0 - (time_remaining / time_total)))
    
    async def _is_ready_for_next_stage(self, item: InnovationPipelineItem) -> bool:
        """Check if innovation is ready for next stage"""
        # Simplified logic - check if been in current stage for minimum time
        time_in_stage = (datetime.utcnow() - item.stage_entered_at).total_seconds() / 3600
        min_time_requirements = {
            PipelineStage.IDEATION: 2,
            PipelineStage.RESEARCH: 24,
            PipelineStage.EXPERIMENTATION: 48,
            PipelineStage.PROTOTYPING: 72,
            PipelineStage.VALIDATION: 24,
            PipelineStage.OPTIMIZATION: 12,
            PipelineStage.DEPLOYMENT: 6
        }
        
        min_time = min_time_requirements.get(item.current_stage, 12)
        return time_in_stage >= min_time and item.success_probability > 0.6
    
    def _get_next_stage(self, current_stage: PipelineStage) -> Optional[PipelineStage]:
        """Get next stage in pipeline"""
        stage_order = [
            PipelineStage.IDEATION,
            PipelineStage.RESEARCH,
            PipelineStage.EXPERIMENTATION,
            PipelineStage.PROTOTYPING,
            PipelineStage.VALIDATION,
            PipelineStage.OPTIMIZATION,
            PipelineStage.DEPLOYMENT,
            PipelineStage.MONITORING
        ]
        
        try:
            current_index = stage_order.index(current_stage)
            if current_index < len(stage_order) - 1:
                return stage_order[current_index + 1]
        except ValueError:
            pass
        
        return None
    
    def _check_stage_capacity(self, stage: PipelineStage) -> bool:
        """Check if stage has available capacity"""
        current_count = len([
            item for item in self.pipeline_items.values()
            if item.current_stage == stage and item.status == PipelineStatus.ACTIVE
        ])
        return current_count < self.stage_capacities.get(stage, 10)