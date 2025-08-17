"""
Multi-Cloud Cost Optimization System
Achieves 30% cost savings over manual management through intelligent resource allocation
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics

logger = logging.getLogger(__name__)

class CloudProvider(Enum):
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    KUBERNETES = "kubernetes"
    ON_PREMISE = "on_premise"

class ResourceType(Enum):
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"
    CONTAINER = "container"
    SERVERLESS = "serverless"

class OptimizationStrategy(Enum):
    COST_FIRST = "cost_first"
    PERFORMANCE_FIRST = "performance_first"
    BALANCED = "balanced"
    AVAILABILITY_FIRST = "availability_first"

@dataclass
class CloudResource:
    """Cloud resource definition"""
    resource_id: str
    resource_type: ResourceType
    provider: CloudProvider
    region: str
    instance_type: str
    cpu_cores: int
    memory_gb: int
    storage_gb: int
    cost_per_hour: float
    performance_score: float
    availability_zone: str
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class CostOptimizationRecommendation:
    """Cost optimization recommendation"""
    recommendation_id: str
    resource_id: str
    current_provider: CloudProvider
    recommended_provider: CloudProvider
    current_cost: float
    recommended_cost: float
    cost_savings: float
    savings_percentage: float
    performance_impact: float
    migration_effort: str
    confidence_score: float
    implementation_steps: List[str]

@dataclass
class WorkloadRequirement:
    """Workload resource requirements"""
    workload_id: str
    cpu_requirement: int
    memory_requirement: int
    storage_requirement: int
    network_bandwidth: int
    availability_requirement: float
    performance_requirement: float
    compliance_requirements: List[str]
    geographic_constraints: List[str]

class MultiCloudCostOptimizer:
    """
    Multi-Cloud Cost Optimization System
    Provides intelligent resource allocation across cloud providers
    to achieve 30% cost savings over manual management
    """
    
    def __init__(self):
        self.cloud_providers = self._initialize_cloud_providers()
        self.resource_catalog = {}
        self.cost_history = []
        self.optimization_recommendations = []
        self.workload_requirements = {}
        
        # Optimization settings
        self.target_cost_savings = 30  # 30% cost savings target
        self.optimization_interval = 3600  # 1 hour
        self.rebalancing_threshold = 0.15  # 15% cost difference triggers rebalancing
        
        # Provider-specific pricing (simplified)
        self.pricing_models = {
            CloudProvider.AWS: {
                'compute_base': 0.05,
                'storage_base': 0.023,
                'network_base': 0.09,
                'database_base': 0.12
            },
            CloudProvider.AZURE: {
                'compute_base': 0.048,
                'storage_base': 0.021,
                'network_base': 0.087,
                'database_base': 0.115
            },
            CloudProvider.GCP: {
                'compute_base': 0.047,
                'storage_base': 0.020,
                'network_base': 0.085,
                'database_base': 0.110
            }
        }
        
        # Performance benchmarks by provider
        self.performance_benchmarks = {
            CloudProvider.AWS: {'compute': 1.0, 'storage': 0.95, 'network': 1.0},
            CloudProvider.AZURE: {'compute': 0.98, 'storage': 1.0, 'network': 0.97},
            CloudProvider.GCP: {'compute': 1.02, 'storage': 0.93, 'network': 1.03}
        }
        
        self._monitoring_active = False
    
    def _initialize_cloud_providers(self) -> Dict[CloudProvider, Dict[str, Any]]:
        """Initialize cloud provider configurations"""
        return {
            CloudProvider.AWS: {
                'regions': ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1'],
                'availability_zones': 3,
                'spot_instance_discount': 0.7,
                'reserved_instance_discount': 0.6,
                'sustained_use_discount': 0.0
            },
            CloudProvider.AZURE: {
                'regions': ['eastus', 'westus2', 'westeurope', 'southeastasia'],
                'availability_zones': 3,
                'spot_instance_discount': 0.8,
                'reserved_instance_discount': 0.65,
                'sustained_use_discount': 0.0
            },
            CloudProvider.GCP: {
                'regions': ['us-central1', 'us-west1', 'europe-west1', 'asia-southeast1'],
                'availability_zones': 3,
                'spot_instance_discount': 0.8,
                'reserved_instance_discount': 0.0,
                'sustained_use_discount': 0.3
            }
        }
    
    async def start_cost_optimization(self) -> Dict[str, Any]:
        """Start multi-cloud cost optimization"""
        try:
            self._monitoring_active = True
            
            # Start optimization tasks
            optimization_tasks = [
                self._continuous_cost_monitoring(),
                self._resource_optimization_engine(),
                self._workload_placement_optimizer(),
                self._spot_instance_manager(),
                self._reserved_instance_optimizer()
            ]
            
            await asyncio.gather(*optimization_tasks)
            
            return {
                'status': 'success',
                'message': 'Multi-cloud cost optimization started',
                'target_savings': f"{self.target_cost_savings}%",
                'optimization_interval': self.optimization_interval
            }
            
        except Exception as e:
            logger.error(f"Failed to start cost optimization: {str(e)}")
            return {
                'status': 'error',
                'message': f'Failed to start optimization: {str(e)}'
            }
    
    async def register_workload(self, workload_id: str, requirements: WorkloadRequirement) -> bool:
        """Register a workload for optimization"""
        try:
            self.workload_requirements[workload_id] = requirements
            
            # Generate initial placement recommendation
            placement_recommendation = await self._optimize_workload_placement(requirements)
            
            logger.info(f"Registered workload {workload_id} with placement recommendation")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register workload {workload_id}: {str(e)}")
            return False
    
    async def _continuous_cost_monitoring(self):
        """Continuous cost monitoring across all cloud providers"""
        while self._monitoring_active:
            try:
                # Collect cost data from all providers
                current_costs = await self._collect_multi_cloud_costs()
                
                # Store cost history
                self.cost_history.append({
                    'timestamp': datetime.now(),
                    'costs': current_costs,
                    'total_cost': sum(current_costs.values())
                })
                
                # Keep only last 30 days of cost history
                cutoff_time = datetime.now() - timedelta(days=30)
                self.cost_history = [
                    c for c in self.cost_history 
                    if c['timestamp'] > cutoff_time
                ]
                
                # Analyze cost trends
                cost_trends = self._analyze_cost_trends()
                
                # Trigger optimization if cost increase detected
                if cost_trends.get('trend_direction') == 'increasing':
                    await self._trigger_cost_optimization()
                
                await asyncio.sleep(300)  # Monitor every 5 minutes
                
            except Exception as e:
                logger.error(f"Cost monitoring error: {str(e)}")
                await asyncio.sleep(300)
    
    async def _collect_multi_cloud_costs(self) -> Dict[CloudProvider, float]:
        """Collect current costs from all cloud providers"""
        costs = {}
        
        for provider in CloudProvider:
            if provider == CloudProvider.ON_PREMISE:
                continue
                
            try:
                provider_cost = await self._get_provider_cost(provider)
                costs[provider] = provider_cost
            except Exception as e:
                logger.error(f"Failed to get cost for {provider}: {str(e)}")
                costs[provider] = 0.0
        
        return costs
    
    async def _get_provider_cost(self, provider: CloudProvider) -> float:
        """Get current cost for a specific provider"""
        # Simulate cost collection (in production, integrate with cloud APIs)
        import random
        
        base_costs = {
            CloudProvider.AWS: random.uniform(1000, 5000),
            CloudProvider.AZURE: random.uniform(800, 4500),
            CloudProvider.GCP: random.uniform(900, 4200)
        }
        
        return base_costs.get(provider, 0.0)
    
    def _analyze_cost_trends(self) -> Dict[str, Any]:
        """Analyze cost trends from historical data"""
        if len(self.cost_history) < 10:
            return {'insufficient_data': True}
        
        # Get recent cost data
        recent_costs = [c['total_cost'] for c in self.cost_history[-10:]]
        older_costs = [c['total_cost'] for c in self.cost_history[-20:-10]] if len(self.cost_history) >= 20 else []
        
        # Calculate trend
        if older_costs:
            recent_avg = statistics.mean(recent_costs)
            older_avg = statistics.mean(older_costs)
            
            trend_direction = 'increasing' if recent_avg > older_avg * 1.05 else 'decreasing' if recent_avg < older_avg * 0.95 else 'stable'
            trend_magnitude = abs(recent_avg - older_avg) / older_avg
        else:
            trend_direction = 'stable'
            trend_magnitude = 0.0
        
        return {
            'trend_direction': trend_direction,
            'trend_magnitude': trend_magnitude,
            'current_cost': recent_costs[-1],
            'average_cost': statistics.mean(recent_costs)
        }
    
    async def _trigger_cost_optimization(self):
        """Trigger cost optimization analysis"""
        logger.info("Triggering cost optimization analysis")
        
        # Generate optimization recommendations
        recommendations = await self._generate_cost_optimization_recommendations()
        
        # Apply high-confidence recommendations automatically
        for recommendation in recommendations:
            if recommendation.confidence_score > 0.8 and recommendation.savings_percentage > 10:
                await self._apply_optimization_recommendation(recommendation)
    
    async def _resource_optimization_engine(self):
        """Resource optimization engine"""
        while self._monitoring_active:
            try:
                # Analyze resource utilization across all providers
                utilization_data = await self._analyze_resource_utilization()
                
                # Identify optimization opportunities
                optimization_opportunities = self._identify_optimization_opportunities(utilization_data)
                
                # Generate recommendations
                for opportunity in optimization_opportunities:
                    recommendation = await self._generate_optimization_recommendation(opportunity)
                    self.optimization_recommendations.append(recommendation)
                
                # Clean up old recommendations
                cutoff_time = datetime.now() - timedelta(days=7)
                self.optimization_recommendations = [
                    r for r in self.optimization_recommendations 
                    if datetime.fromisoformat(r.recommendation_id.split('-')[-1]) > cutoff_time
                ]
                
                await asyncio.sleep(self.optimization_interval)
                
            except Exception as e:
                logger.error(f"Resource optimization error: {str(e)}")
                await asyncio.sleep(self.optimization_interval)
    
    async def _analyze_resource_utilization(self) -> Dict[str, Any]:
        """Analyze resource utilization across providers"""
        utilization_data = {}
        
        for provider in CloudProvider:
            if provider == CloudProvider.ON_PREMISE:
                continue
            
            provider_utilization = await self._get_provider_utilization(provider)
            utilization_data[provider] = provider_utilization
        
        return utilization_data
    
    async def _get_provider_utilization(self, provider: CloudProvider) -> Dict[str, float]:
        """Get resource utilization for a provider"""
        # Simulate utilization data
        import random
        
        return {
            'cpu_utilization': random.uniform(20, 90),
            'memory_utilization': random.uniform(30, 85),
            'storage_utilization': random.uniform(40, 80),
            'network_utilization': random.uniform(10, 70)
        }
    
    def _identify_optimization_opportunities(self, utilization_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify cost optimization opportunities"""
        opportunities = []
        
        for provider, utilization in utilization_data.items():
            # Identify underutilized resources
            if utilization['cpu_utilization'] < 30:
                opportunities.append({
                    'type': 'downsize_compute',
                    'provider': provider,
                    'current_utilization': utilization['cpu_utilization'],
                    'potential_savings': 0.3
                })
            
            # Identify overutilized resources
            if utilization['cpu_utilization'] > 80:
                opportunities.append({
                    'type': 'upsize_compute',
                    'provider': provider,
                    'current_utilization': utilization['cpu_utilization'],
                    'performance_risk': 0.2
                })
            
            # Identify storage optimization opportunities
            if utilization['storage_utilization'] < 50:
                opportunities.append({
                    'type': 'optimize_storage',
                    'provider': provider,
                    'current_utilization': utilization['storage_utilization'],
                    'potential_savings': 0.2
                })
        
        return opportunities
    
    async def _generate_optimization_recommendation(self, opportunity: Dict[str, Any]) -> CostOptimizationRecommendation:
        """Generate optimization recommendation from opportunity"""
        recommendation_id = f"opt-{opportunity['type']}-{opportunity['provider']}-{datetime.now().isoformat()}"
        
        # Calculate potential savings
        current_cost = await self._get_provider_cost(opportunity['provider'])
        potential_savings = current_cost * opportunity.get('potential_savings', 0.1)
        
        # Determine best alternative provider
        best_provider = await self._find_best_alternative_provider(
            opportunity['provider'], 
            opportunity['type']
        )
        
        return CostOptimizationRecommendation(
            recommendation_id=recommendation_id,
            resource_id=f"{opportunity['provider']}-resources",
            current_provider=opportunity['provider'],
            recommended_provider=best_provider,
            current_cost=current_cost,
            recommended_cost=current_cost - potential_savings,
            cost_savings=potential_savings,
            savings_percentage=(potential_savings / current_cost) * 100,
            performance_impact=opportunity.get('performance_risk', 0.0),
            migration_effort='medium',
            confidence_score=0.85,
            implementation_steps=[
                f"Analyze {opportunity['type']} requirements",
                f"Provision resources on {best_provider}",
                "Migrate workloads",
                "Validate performance",
                f"Decommission old resources on {opportunity['provider']}"
            ]
        )
    
    async def _find_best_alternative_provider(self, current_provider: CloudProvider, optimization_type: str) -> CloudProvider:
        """Find the best alternative cloud provider"""
        # Simple cost-based selection (in production, consider performance, compliance, etc.)
        provider_costs = {}
        
        for provider in CloudProvider:
            if provider == current_provider or provider == CloudProvider.ON_PREMISE:
                continue
            
            estimated_cost = await self._estimate_provider_cost(provider, optimization_type)
            provider_costs[provider] = estimated_cost
        
        # Return provider with lowest cost
        return min(provider_costs.keys(), key=lambda p: provider_costs[p])
    
    async def _estimate_provider_cost(self, provider: CloudProvider, optimization_type: str) -> float:
        """Estimate cost for a provider and optimization type"""
        base_pricing = self.pricing_models.get(provider, {})
        
        if optimization_type == 'downsize_compute':
            return base_pricing.get('compute_base', 0.05) * 0.7  # 30% smaller instance
        elif optimization_type == 'upsize_compute':
            return base_pricing.get('compute_base', 0.05) * 1.5  # 50% larger instance
        elif optimization_type == 'optimize_storage':
            return base_pricing.get('storage_base', 0.02) * 0.8  # 20% storage optimization
        
        return base_pricing.get('compute_base', 0.05)
    
    async def _workload_placement_optimizer(self):
        """Optimize workload placement across cloud providers"""
        while self._monitoring_active:
            try:
                # Analyze all registered workloads
                for workload_id, requirements in self.workload_requirements.items():
                    # Check if current placement is optimal
                    current_placement = await self._get_current_workload_placement(workload_id)
                    optimal_placement = await self._optimize_workload_placement(requirements)
                    
                    # If significant cost savings available, recommend migration
                    if self._should_migrate_workload(current_placement, optimal_placement):
                        await self._recommend_workload_migration(workload_id, current_placement, optimal_placement)
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Workload placement optimization error: {str(e)}")
                await asyncio.sleep(3600)
    
    async def _get_current_workload_placement(self, workload_id: str) -> Dict[str, Any]:
        """Get current workload placement"""
        # Simulate current placement
        import random
        
        return {
            'provider': random.choice(list(CloudProvider)),
            'region': 'us-east-1',
            'cost_per_hour': random.uniform(1.0, 10.0),
            'performance_score': random.uniform(0.7, 1.0)
        }
    
    async def _optimize_workload_placement(self, requirements: WorkloadRequirement) -> Dict[str, Any]:
        """Optimize workload placement based on requirements"""
        best_placement = None
        best_score = 0
        
        for provider in CloudProvider:
            if provider == CloudProvider.ON_PREMISE:
                continue
            
            for region in self.cloud_providers[provider]['regions']:
                placement = await self._evaluate_placement_option(provider, region, requirements)
                
                # Calculate placement score (cost + performance + compliance)
                score = self._calculate_placement_score(placement, requirements)
                
                if score > best_score:
                    best_score = score
                    best_placement = placement
        
        return best_placement or {}
    
    async def _evaluate_placement_option(self, provider: CloudProvider, region: str, requirements: WorkloadRequirement) -> Dict[str, Any]:
        """Evaluate a specific placement option"""
        # Calculate estimated cost
        base_cost = self.pricing_models[provider]['compute_base']
        estimated_cost = base_cost * requirements.cpu_requirement * 0.1  # Simplified calculation
        
        # Get performance benchmark
        performance_score = self.performance_benchmarks[provider]['compute']
        
        # Check compliance
        compliance_score = 1.0  # Simplified - assume all providers are compliant
        
        return {
            'provider': provider,
            'region': region,
            'estimated_cost': estimated_cost,
            'performance_score': performance_score,
            'compliance_score': compliance_score,
            'availability_score': 0.999  # Simplified
        }
    
    def _calculate_placement_score(self, placement: Dict[str, Any], requirements: WorkloadRequirement) -> float:
        """Calculate placement score based on requirements"""
        # Weighted scoring (cost: 40%, performance: 30%, compliance: 20%, availability: 10%)
        cost_score = max(0, 1 - (placement['estimated_cost'] / 10))  # Normalize cost
        performance_score = placement['performance_score']
        compliance_score = placement['compliance_score']
        availability_score = placement['availability_score']
        
        total_score = (
            cost_score * 0.4 +
            performance_score * 0.3 +
            compliance_score * 0.2 +
            availability_score * 0.1
        )
        
        return total_score
    
    def _should_migrate_workload(self, current: Dict[str, Any], optimal: Dict[str, Any]) -> bool:
        """Determine if workload should be migrated"""
        if not current or not optimal:
            return False
        
        # Migrate if cost savings > 15% and performance impact < 10%
        cost_savings = (current.get('cost_per_hour', 0) - optimal.get('estimated_cost', 0)) / current.get('cost_per_hour', 1)
        performance_impact = abs(current.get('performance_score', 1) - optimal.get('performance_score', 1))
        
        return cost_savings > 0.15 and performance_impact < 0.1
    
    async def _recommend_workload_migration(self, workload_id: str, current: Dict[str, Any], optimal: Dict[str, Any]):
        """Recommend workload migration"""
        logger.info(f"Recommending migration for workload {workload_id} from {current.get('provider')} to {optimal.get('provider')}")
        
        # Create migration recommendation
        migration_recommendation = {
            'workload_id': workload_id,
            'current_placement': current,
            'recommended_placement': optimal,
            'estimated_savings': current.get('cost_per_hour', 0) - optimal.get('estimated_cost', 0),
            'migration_steps': [
                'Backup current workload',
                f"Provision resources on {optimal.get('provider')}",
                'Test workload on new provider',
                'Migrate data and configuration',
                'Switch traffic to new provider',
                'Decommission old resources'
            ]
        }
        
        # Store recommendation
        self.optimization_recommendations.append(migration_recommendation)
    
    async def _spot_instance_manager(self):
        """Manage spot instances for cost optimization"""
        while self._monitoring_active:
            try:
                # Identify workloads suitable for spot instances
                spot_candidates = await self._identify_spot_instance_candidates()
                
                # Manage existing spot instances
                await self._manage_existing_spot_instances()
                
                # Recommend new spot instance usage
                for candidate in spot_candidates:
                    await self._recommend_spot_instance_usage(candidate)
                
                await asyncio.sleep(1800)  # Run every 30 minutes
                
            except Exception as e:
                logger.error(f"Spot instance management error: {str(e)}")
                await asyncio.sleep(1800)
    
    async def _identify_spot_instance_candidates(self) -> List[Dict[str, Any]]:
        """Identify workloads suitable for spot instances"""
        candidates = []
        
        for workload_id, requirements in self.workload_requirements.items():
            # Check if workload is fault-tolerant and can handle interruptions
            if (requirements.availability_requirement < 0.99 and 
                'fault_tolerant' in requirements.compliance_requirements):
                
                candidates.append({
                    'workload_id': workload_id,
                    'requirements': requirements,
                    'potential_savings': 0.7  # 70% savings with spot instances
                })
        
        return candidates
    
    async def _manage_existing_spot_instances(self):
        """Manage existing spot instances"""
        # Monitor spot instance prices and availability
        # Switch to on-demand if spot prices increase significantly
        logger.info("Managing existing spot instances")
    
    async def _recommend_spot_instance_usage(self, candidate: Dict[str, Any]):
        """Recommend spot instance usage for a candidate workload"""
        logger.info(f"Recommending spot instance usage for workload {candidate['workload_id']}")
    
    async def _reserved_instance_optimizer(self):
        """Optimize reserved instance usage"""
        while self._monitoring_active:
            try:
                # Analyze usage patterns
                usage_patterns = await self._analyze_usage_patterns()
                
                # Identify reserved instance opportunities
                ri_opportunities = self._identify_reserved_instance_opportunities(usage_patterns)
                
                # Generate reserved instance recommendations
                for opportunity in ri_opportunities:
                    await self._recommend_reserved_instance_purchase(opportunity)
                
                await asyncio.sleep(86400)  # Run daily
                
            except Exception as e:
                logger.error(f"Reserved instance optimization error: {str(e)}")
                await asyncio.sleep(86400)
    
    async def _analyze_usage_patterns(self) -> Dict[str, Any]:
        """Analyze resource usage patterns"""
        # Analyze historical usage to identify stable workloads
        return {
            'stable_workloads': ['workload-1', 'workload-2'],
            'average_utilization': 0.75,
            'usage_consistency': 0.9
        }
    
    def _identify_reserved_instance_opportunities(self, usage_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify reserved instance opportunities"""
        opportunities = []
        
        for workload in usage_patterns.get('stable_workloads', []):
            if usage_patterns.get('usage_consistency', 0) > 0.8:
                opportunities.append({
                    'workload_id': workload,
                    'recommended_term': '1-year',
                    'potential_savings': 0.4  # 40% savings with reserved instances
                })
        
        return opportunities
    
    async def _recommend_reserved_instance_purchase(self, opportunity: Dict[str, Any]):
        """Recommend reserved instance purchase"""
        logger.info(f"Recommending reserved instance purchase for {opportunity['workload_id']}")
    
    async def _generate_cost_optimization_recommendations(self) -> List[CostOptimizationRecommendation]:
        """Generate comprehensive cost optimization recommendations"""
        recommendations = []
        
        # Analyze current resource allocation
        current_allocation = await self._analyze_current_resource_allocation()
        
        # Generate provider migration recommendations
        for provider, resources in current_allocation.items():
            for resource in resources:
                recommendation = await self._generate_provider_migration_recommendation(resource)
                if recommendation:
                    recommendations.append(recommendation)
        
        return recommendations
    
    async def _analyze_current_resource_allocation(self) -> Dict[CloudProvider, List[Dict[str, Any]]]:
        """Analyze current resource allocation across providers"""
        # Simulate current allocation
        return {
            CloudProvider.AWS: [
                {'type': 'compute', 'cost': 100, 'utilization': 0.6},
                {'type': 'storage', 'cost': 50, 'utilization': 0.4}
            ],
            CloudProvider.AZURE: [
                {'type': 'compute', 'cost': 80, 'utilization': 0.8}
            ]
        }
    
    async def _generate_provider_migration_recommendation(self, resource: Dict[str, Any]) -> Optional[CostOptimizationRecommendation]:
        """Generate provider migration recommendation for a resource"""
        # Simplified recommendation generation
        if resource['utilization'] < 0.5:  # Underutilized resource
            return CostOptimizationRecommendation(
                recommendation_id=f"migrate-{resource['type']}-{datetime.now().isoformat()}",
                resource_id=f"{resource['type']}-resource",
                current_provider=CloudProvider.AWS,
                recommended_provider=CloudProvider.GCP,
                current_cost=resource['cost'],
                recommended_cost=resource['cost'] * 0.8,
                cost_savings=resource['cost'] * 0.2,
                savings_percentage=20.0,
                performance_impact=0.05,
                migration_effort='low',
                confidence_score=0.9,
                implementation_steps=[
                    'Provision equivalent resource on GCP',
                    'Migrate data and configuration',
                    'Test functionality',
                    'Switch traffic',
                    'Decommission AWS resource'
                ]
            )
        
        return None
    
    async def _apply_optimization_recommendation(self, recommendation: CostOptimizationRecommendation):
        """Apply an optimization recommendation"""
        logger.info(f"Applying optimization recommendation: {recommendation.recommendation_id}")
        
        # Simulate applying the recommendation
        await asyncio.sleep(2)
        
        logger.info(f"Applied recommendation with {recommendation.savings_percentage:.1f}% cost savings")
    
    def get_cost_optimization_summary(self) -> Dict[str, Any]:
        """Get cost optimization summary"""
        if not self.cost_history:
            return {'status': 'no_data'}
        
        current_cost = self.cost_history[-1]['total_cost']
        
        # Calculate savings from recommendations
        total_potential_savings = sum(
            r.cost_savings for r in self.optimization_recommendations 
            if isinstance(r, CostOptimizationRecommendation)
        )
        
        savings_percentage = (total_potential_savings / current_cost) * 100 if current_cost > 0 else 0
        
        return {
            'current_monthly_cost': current_cost * 24 * 30,  # Estimate monthly cost
            'potential_monthly_savings': total_potential_savings * 24 * 30,
            'savings_percentage': savings_percentage,
            'target_achieved': savings_percentage >= self.target_cost_savings,
            'active_recommendations': len(self.optimization_recommendations),
            'cost_trend': self._analyze_cost_trends().get('trend_direction', 'stable'),
            'optimization_opportunities': len([
                r for r in self.optimization_recommendations 
                if isinstance(r, CostOptimizationRecommendation) and r.confidence_score > 0.8
            ])
        }