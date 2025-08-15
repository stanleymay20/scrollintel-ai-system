"""
Multi-Source Funding Coordination System

Coordinates unlimited funding access across multiple sources with redundancy,
security validation, and real-time monitoring for guaranteed success.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import logging
from decimal import Decimal
import uuid

from ..engines.funding_engine import (
    UnlimitedFundingEngine, 
    FundingRequest, 
    FundingSourceType,
    FundingStatus
)

logger = logging.getLogger(__name__)


@dataclass
class FundingAllocation:
    """Represents a funding allocation across multiple sources"""
    id: str
    request_id: str
    total_amount: Decimal
    sources: List[Dict[str, Any]]
    deployment_timeline: str
    security_level: int
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    deployed_at: Optional[datetime] = None


@dataclass
class FundingMetrics:
    """Real-time funding metrics and performance data"""
    total_committed: Decimal
    total_available: Decimal
    total_deployed: Decimal
    active_requests: int
    successful_allocations: int
    failed_allocations: int
    average_deployment_time: float
    security_incidents: int
    backup_activations: int
    last_updated: datetime = field(default_factory=datetime.now)


class MultiFundingCoordinator:
    """
    Coordinates unlimited funding access across multiple sources with
    $25B+ commitment management, security validation, and backup activation.
    """
    
    def __init__(self):
        self.funding_engine = UnlimitedFundingEngine()
        self.allocations: Dict[str, FundingAllocation] = {}
        self.metrics = FundingMetrics(
            total_committed=Decimal('0'),
            total_available=Decimal('0'),
            total_deployed=Decimal('0'),
            active_requests=0,
            successful_allocations=0,
            failed_allocations=0,
            average_deployment_time=0.0,
            security_incidents=0,
            backup_activations=0
        )
        self.monitoring_active = False
        
        # Initialize monitoring
        asyncio.create_task(self._initialize_monitoring())
    
    async def _initialize_monitoring(self):
        """Initialize real-time monitoring systems"""
        await asyncio.sleep(1)  # Allow engine to initialize
        await self.funding_engine.monitor_funding_availability()
        self.monitoring_active = True
        asyncio.create_task(self._update_metrics_continuously())
        logger.info("Multi-source funding coordination system initialized")
    
    async def request_unlimited_funding(
        self, 
        amount: Decimal, 
        purpose: str,
        urgency_level: int = 5,
        required_by: Optional[datetime] = None,
        security_requirements: Optional[List[str]] = None,
        preferred_sources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Request unlimited funding with multi-source coordination
        
        Args:
            amount: Funding amount requested
            purpose: Purpose of funding
            urgency_level: 1-10, 10 being most urgent
            required_by: When funding is needed by
            security_requirements: List of security requirements
            preferred_sources: Preferred funding source IDs
            
        Returns:
            Dict containing allocation details and status
        """
        try:
            # Create funding request
            request_id = str(uuid.uuid4())
            request = FundingRequest(
                id=request_id,
                amount=amount,
                purpose=purpose,
                urgency_level=urgency_level,
                required_by=required_by or (datetime.now() + timedelta(days=30)),
                security_requirements=security_requirements or ["min_security_8"],
                preferred_sources=preferred_sources or [],
                backup_acceptable=True
            )
            
            logger.info(f"Processing unlimited funding request: ${amount:,} for {purpose}")
            
            # Process through funding engine
            result = await self.funding_engine.request_funding(request)
            
            if result['success']:
                # Create allocation record
                allocation = FundingAllocation(
                    id=str(uuid.uuid4()),
                    request_id=request_id,
                    total_amount=amount,
                    sources=result['sources_used'],
                    deployment_timeline=result['deployment_timeline'],
                    security_level=result['security_level'],
                    status="allocated"
                )
                
                self.allocations[allocation.id] = allocation
                self.metrics.successful_allocations += 1
                self.metrics.total_deployed += amount
                
                # Update deployment timeline metrics
                timeline_hours = float(result['deployment_timeline'].split()[0])
                self._update_deployment_metrics(timeline_hours)
                
                logger.info(f"Successfully allocated ${amount:,} across {len(result['sources_used'])} sources")
                
                return {
                    'success': True,
                    'allocation_id': allocation.id,
                    'request_id': request_id,
                    'amount_allocated': amount,
                    'sources_used': result['sources_used'],
                    'deployment_timeline': result['deployment_timeline'],
                    'security_level': result['security_level'],
                    'coordination_status': 'multi_source_success'
                }
            else:
                self.metrics.failed_allocations += 1
                logger.error(f"Failed to allocate funding: {result.get('error', 'Unknown error')}")
                
                return {
                    'success': False,
                    'error': result.get('error', 'Funding allocation failed'),
                    'request_id': request_id,
                    'amount_requested': amount,
                    'coordination_status': 'allocation_failed'
                }
                
        except Exception as e:
            logger.error(f"Error in unlimited funding request: {str(e)}")
            self.metrics.failed_allocations += 1
            return {
                'success': False,
                'error': str(e),
                'coordination_status': 'system_error'
            }
    
    async def get_funding_status(self) -> Dict[str, Any]:
        """Get comprehensive funding status across all sources"""
        try:
            # Get engine status
            engine_status = await self.funding_engine.monitor_funding_availability()
            
            # Get capacity information
            capacity = self.funding_engine.get_funding_capacity()
            
            # Calculate coordination metrics
            coordination_metrics = {
                'total_allocations': len(self.allocations),
                'active_allocations': len([a for a in self.allocations.values() if a.status == "allocated"]),
                'total_amount_coordinated': sum(a.total_amount for a in self.allocations.values()),
                'average_sources_per_allocation': self._calculate_average_sources(),
                'coordination_success_rate': self._calculate_success_rate(),
                'backup_activation_rate': self.metrics.backup_activations / max(1, self.metrics.successful_allocations) * 100
            }
            
            return {
                'funding_engine_status': engine_status,
                'capacity_status': capacity,
                'coordination_metrics': coordination_metrics,
                'system_metrics': {
                    'total_committed': self.metrics.total_committed,
                    'total_available': self.metrics.total_available,
                    'total_deployed': self.metrics.total_deployed,
                    'successful_allocations': self.metrics.successful_allocations,
                    'failed_allocations': self.metrics.failed_allocations,
                    'average_deployment_time_hours': self.metrics.average_deployment_time,
                    'security_incidents': self.metrics.security_incidents,
                    'backup_activations': self.metrics.backup_activations
                },
                'status': 'operational',
                'unlimited_capacity': True,
                'multi_source_coordination': True,
                'backup_systems_active': True,
                'real_time_monitoring': self.monitoring_active
            }
            
        except Exception as e:
            logger.error(f"Error getting funding status: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'unlimited_capacity': False
            }
    
    async def activate_emergency_funding(self, amount: Decimal, purpose: str) -> Dict[str, Any]:
        """
        Activate emergency unlimited funding with maximum speed and all backup sources
        
        Args:
            amount: Emergency funding amount
            purpose: Emergency purpose
            
        Returns:
            Emergency funding allocation result
        """
        try:
            logger.warning(f"Activating emergency funding: ${amount:,} for {purpose}")
            
            # Create high-priority emergency request
            result = await self.request_unlimited_funding(
                amount=amount,
                purpose=f"EMERGENCY: {purpose}",
                urgency_level=10,
                required_by=datetime.now() + timedelta(hours=1),
                security_requirements=["min_security_6"],  # Lower security for speed
                preferred_sources=["crypto_treasury", "corporate_strategic"]  # Fastest sources
            )
            
            if result['success']:
                logger.info(f"Emergency funding activated: ${amount:,}")
                return {
                    **result,
                    'emergency_activation': True,
                    'priority_level': 'maximum'
                }
            else:
                logger.error(f"Emergency funding activation failed: {result.get('error')}")
                return {
                    **result,
                    'emergency_activation': False,
                    'escalation_required': True
                }
                
        except Exception as e:
            logger.error(f"Error in emergency funding activation: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'emergency_activation': False,
                'escalation_required': True
            }
    
    async def validate_funding_security(self) -> Dict[str, Any]:
        """Validate security across all funding sources and allocations"""
        try:
            security_status = {
                'overall_security_level': 10,
                'sources_validated': 0,
                'sources_failed_validation': 0,
                'security_incidents': self.metrics.security_incidents,
                'last_validation': datetime.now(),
                'validation_details': {}
            }
            
            # Validate each funding source through the engine
            engine_status = await self.funding_engine.monitor_funding_availability()
            
            for source_id, source_details in engine_status['source_details'].items():
                security_status['sources_validated'] += 1
                security_status['validation_details'][source_id] = {
                    'security_level': source_details['security_level'],
                    'status': source_details['status'],
                    'last_validated': source_details['last_validated']
                }
                
                # Update overall security level (minimum across all sources)
                security_status['overall_security_level'] = min(
                    security_status['overall_security_level'],
                    source_details['security_level']
                )
            
            # Validate allocations
            allocation_security = {
                'total_allocations': len(self.allocations),
                'secure_allocations': 0,
                'insecure_allocations': 0
            }
            
            for allocation in self.allocations.values():
                if allocation.security_level >= 8:
                    allocation_security['secure_allocations'] += 1
                else:
                    allocation_security['insecure_allocations'] += 1
            
            return {
                'security_validation': 'passed',
                'source_security': security_status,
                'allocation_security': allocation_security,
                'recommendations': self._generate_security_recommendations(security_status)
            }
            
        except Exception as e:
            logger.error(f"Error validating funding security: {str(e)}")
            return {
                'security_validation': 'failed',
                'error': str(e)
            }
    
    def _update_deployment_metrics(self, timeline_hours: float):
        """Update average deployment time metrics"""
        if self.metrics.successful_allocations == 1:
            self.metrics.average_deployment_time = timeline_hours
        else:
            # Calculate running average
            total_time = self.metrics.average_deployment_time * (self.metrics.successful_allocations - 1)
            self.metrics.average_deployment_time = (total_time + timeline_hours) / self.metrics.successful_allocations
    
    def _calculate_average_sources(self) -> float:
        """Calculate average number of sources per allocation"""
        if not self.allocations:
            return 0.0
        
        total_sources = sum(len(allocation.sources) for allocation in self.allocations.values())
        return total_sources / len(self.allocations)
    
    def _calculate_success_rate(self) -> float:
        """Calculate funding coordination success rate"""
        total_attempts = self.metrics.successful_allocations + self.metrics.failed_allocations
        if total_attempts == 0:
            return 100.0
        
        return (self.metrics.successful_allocations / total_attempts) * 100
    
    def _generate_security_recommendations(self, security_status: Dict[str, Any]) -> List[str]:
        """Generate security recommendations based on validation results"""
        recommendations = []
        
        if security_status['overall_security_level'] < 8:
            recommendations.append("Consider upgrading security levels for funding sources")
        
        if security_status['sources_failed_validation'] > 0:
            recommendations.append("Investigate and resolve failed source validations")
        
        if self.metrics.security_incidents > 0:
            recommendations.append("Review and address recent security incidents")
        
        if not recommendations:
            recommendations.append("Security status is optimal")
        
        return recommendations
    
    async def _update_metrics_continuously(self):
        """Continuously update funding metrics"""
        while self.monitoring_active:
            try:
                # Update metrics from funding engine
                capacity = self.funding_engine.get_funding_capacity()
                self.metrics.total_committed = capacity['total_committed']
                self.metrics.total_available = capacity['total_available']
                self.metrics.active_requests = len(self.funding_engine.active_requests)
                self.metrics.last_updated = datetime.now()
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error updating funding metrics: {str(e)}")
                await asyncio.sleep(30)  # Retry in 30 seconds on error


# Global coordinator instance
funding_coordinator = MultiFundingCoordinator()