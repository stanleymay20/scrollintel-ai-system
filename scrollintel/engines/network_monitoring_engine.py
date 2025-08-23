"""
Real-time Network Monitoring Engine for Global Influence Network System

This engine provides real-time monitoring of influence network changes,
shift detection, and adaptive strategies.
"""

import logging
import asyncio
from typing import List, Dict, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json

from ..models.influence_network_models import (
    InfluenceNetwork, InfluenceNode, InfluenceEdge, NetworkGap
)


@dataclass
class NetworkChange:
    """Represents a change in the influence network"""
    change_id: str
    network_id: str
    change_type: str  # 'node_added', 'node_removed', 'edge_added', 'edge_removed', 'influence_shift'
    affected_entities: List[str]
    change_magnitude: float
    change_description: str
    detected_at: datetime
    change_data: Dict[str, Any]
    impact_assessment: Dict[str, float]


@dataclass
class InfluenceShift:
    """Represents a shift in influence within the network"""
    shift_id: str
    network_id: str
    node_id: str
    previous_influence: float
    current_influence: float
    shift_magnitude: float
    shift_direction: str  # 'increase', 'decrease'
    contributing_factors: List[str]
    detected_at: datetime
    confidence_score: float


@dataclass
class NetworkAlert:
    """Represents an alert about network changes"""
    alert_id: str
    network_id: str
    alert_type: str  # 'influence_shift', 'new_gap', 'competitor_move', 'opportunity'
    severity: str  # 'low', 'medium', 'high', 'critical'
    title: str
    description: str
    affected_nodes: List[str]
    recommended_actions: List[str]
    created_at: datetime
    expires_at: Optional[datetime]
    acknowledged: bool


class NetworkMonitoringEngine:
    """Engine for real-time monitoring of influence network changes"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.monitored_networks = {}
        self.network_snapshots = {}
        self.change_history = defaultdict(list)
        self.active_alerts = {}
        self.monitoring_config = {
            'influence_threshold': 0.1,  # Minimum change to trigger alert
            'monitoring_interval': 300,  # 5 minutes
            'alert_retention_days': 7,
            'max_alerts_per_network': 50
        }
    
    async def start_monitoring(
        self,
        network: InfluenceNetwork,
        monitoring_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start monitoring an influence network"""
        try:
            monitoring_id = f"monitor_{network.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Update config if provided
            config = self.monitoring_config.copy()
            if monitoring_config:
                config.update(monitoring_config)
            
            # Store network for monitoring
            self.monitored_networks[monitoring_id] = {
                'network': network,
                'config': config,
                'started_at': datetime.now(),
                'last_check': datetime.now(),
                'status': 'active'
            }
            
            # Create initial snapshot
            await self._create_network_snapshot(network.id, network)
            
            self.logger.info(f"Started monitoring network {network.id} with ID {monitoring_id}")
            return monitoring_id
            
        except Exception as e:
            self.logger.error(f"Error starting network monitoring: {str(e)}")
            raise
    
    async def detect_influence_shifts(
        self,
        network: InfluenceNetwork,
        previous_snapshot: Optional[Dict[str, Any]] = None
    ) -> List[InfluenceShift]:
        """Detect shifts in influence within the network"""
        try:
            shifts = []
            
            if not previous_snapshot:
                # Get the most recent snapshot
                snapshots = self.network_snapshots.get(network.id, [])
                if not snapshots:
                    return shifts
                previous_snapshot = snapshots[-1]
            
            # Compare current network with previous snapshot
            previous_nodes = {node['id']: node for node in previous_snapshot.get('nodes', [])}
            
            for current_node in network.nodes:
                if current_node.id in previous_nodes:
                    previous_node = previous_nodes[current_node.id]
                    previous_influence = previous_node.get('influence_score', 0)
                    current_influence = current_node.influence_score
                    
                    # Calculate shift magnitude
                    shift_magnitude = abs(current_influence - previous_influence)
                    
                    # Check if shift is significant
                    if shift_magnitude >= self.monitoring_config['influence_threshold']:
                        shift_direction = 'increase' if current_influence > previous_influence else 'decrease'
                        
                        # Analyze contributing factors
                        contributing_factors = await self._analyze_shift_factors(
                            current_node, previous_node, network
                        )
                        
                        shift = InfluenceShift(
                            shift_id=f"shift_{current_node.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            network_id=network.id,
                            node_id=current_node.id,
                            previous_influence=previous_influence,
                            current_influence=current_influence,
                            shift_magnitude=shift_magnitude,
                            shift_direction=shift_direction,
                            contributing_factors=contributing_factors,
                            detected_at=datetime.now(),
                            confidence_score=min(shift_magnitude * 2, 1.0)
                        )
                        shifts.append(shift)
            
            self.logger.info(f"Detected {len(shifts)} influence shifts in network {network.id}")
            return shifts
            
        except Exception as e:
            self.logger.error(f"Error detecting influence shifts: {str(e)}")
            raise  
  
    async def identify_network_gaps_changes(
        self,
        network: InfluenceNetwork,
        previous_gaps: List[NetworkGap]
    ) -> List[NetworkChange]:
        """Identify changes in network gaps"""
        try:
            changes = []
            
            # This would integrate with the influence mapping engine
            # For now, we'll simulate gap detection
            current_gaps = await self._simulate_gap_detection(network)
            
            # Compare with previous gaps
            previous_gap_ids = {gap.gap_id for gap in previous_gaps}
            current_gap_ids = {gap.gap_id for gap in current_gaps}
            
            # New gaps
            new_gap_ids = current_gap_ids - previous_gap_ids
            for gap_id in new_gap_ids:
                gap = next(g for g in current_gaps if g.gap_id == gap_id)
                change = NetworkChange(
                    change_id=f"gap_new_{gap_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    network_id=network.id,
                    change_type='new_gap',
                    affected_entities=[gap_id],
                    change_magnitude=gap.potential_impact,
                    change_description=f"New network gap identified: {gap.description}",
                    detected_at=datetime.now(),
                    change_data={'gap': asdict(gap)},
                    impact_assessment={'priority': gap.priority, 'impact': gap.potential_impact}
                )
                changes.append(change)
            
            # Resolved gaps
            resolved_gap_ids = previous_gap_ids - current_gap_ids
            for gap_id in resolved_gap_ids:
                change = NetworkChange(
                    change_id=f"gap_resolved_{gap_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    network_id=network.id,
                    change_type='gap_resolved',
                    affected_entities=[gap_id],
                    change_magnitude=0.5,
                    change_description=f"Network gap resolved: {gap_id}",
                    detected_at=datetime.now(),
                    change_data={'gap_id': gap_id},
                    impact_assessment={'positive_impact': 0.7}
                )
                changes.append(change)
            
            return changes
            
        except Exception as e:
            self.logger.error(f"Error identifying gap changes: {str(e)}")
            raise
    
    async def generate_adaptive_strategies(
        self,
        network: InfluenceNetwork,
        detected_changes: List[NetworkChange],
        influence_shifts: List[InfluenceShift]
    ) -> List[Dict[str, Any]]:
        """Generate adaptive strategies based on detected changes"""
        try:
            strategies = []
            
            # Analyze influence shifts
            for shift in influence_shifts:
                if shift.shift_direction == 'decrease' and shift.shift_magnitude > 0.2:
                    # Significant influence decrease - need intervention
                    strategy = {
                        'strategy_id': f"intervention_{shift.node_id}",
                        'type': 'influence_recovery',
                        'target_node': shift.node_id,
                        'priority': 'high',
                        'actions': [
                            'Increase engagement frequency',
                            'Provide additional value',
                            'Strengthen relationship'
                        ],
                        'timeline': '2-4 weeks',
                        'expected_impact': 0.3
                    }
                    strategies.append(strategy)
                
                elif shift.shift_direction == 'increase' and shift.shift_magnitude > 0.15:
                    # Significant influence increase - leverage opportunity
                    strategy = {
                        'strategy_id': f"leverage_{shift.node_id}",
                        'type': 'influence_leverage',
                        'target_node': shift.node_id,
                        'priority': 'medium',
                        'actions': [
                            'Expand collaboration opportunities',
                            'Introduce to other network members',
                            'Leverage for strategic initiatives'
                        ],
                        'timeline': '1-2 weeks',
                        'expected_impact': 0.4
                    }
                    strategies.append(strategy)
            
            # Analyze network changes
            for change in detected_changes:
                if change.change_type == 'new_gap' and change.change_magnitude > 0.6:
                    # High-impact new gap - immediate action needed
                    strategy = {
                        'strategy_id': f"gap_response_{change.change_id}",
                        'type': 'gap_mitigation',
                        'target_gap': change.affected_entities[0],
                        'priority': 'high',
                        'actions': [
                            'Identify potential connections',
                            'Develop relationship building plan',
                            'Allocate resources for gap closure'
                        ],
                        'timeline': '1-3 months',
                        'expected_impact': change.change_magnitude
                    }
                    strategies.append(strategy)
            
            self.logger.info(f"Generated {len(strategies)} adaptive strategies")
            return strategies
            
        except Exception as e:
            self.logger.error(f"Error generating adaptive strategies: {str(e)}")
            raise
    
    async def create_network_alerts(
        self,
        network: InfluenceNetwork,
        changes: List[NetworkChange],
        shifts: List[InfluenceShift]
    ) -> List[NetworkAlert]:
        """Create alerts based on network changes and shifts"""
        try:
            alerts = []
            
            # Create alerts for significant influence shifts
            for shift in shifts:
                if shift.shift_magnitude > 0.2:
                    severity = 'high' if shift.shift_magnitude > 0.3 else 'medium'
                    
                    alert = NetworkAlert(
                        alert_id=f"alert_shift_{shift.shift_id}",
                        network_id=network.id,
                        alert_type='influence_shift',
                        severity=severity,
                        title=f"Significant Influence {shift.shift_direction.title()}",
                        description=f"Node {shift.node_id} experienced {shift.shift_direction} of {shift.shift_magnitude:.2f}",
                        affected_nodes=[shift.node_id],
                        recommended_actions=[
                            'Review relationship status',
                            'Analyze contributing factors',
                            'Implement adaptive strategy'
                        ],
                        created_at=datetime.now(),
                        expires_at=datetime.now() + timedelta(days=7),
                        acknowledged=False
                    )
                    alerts.append(alert)
            
            # Create alerts for network changes
            for change in changes:
                if change.change_type == 'new_gap' and change.change_magnitude > 0.5:
                    alert = NetworkAlert(
                        alert_id=f"alert_gap_{change.change_id}",
                        network_id=network.id,
                        alert_type='new_gap',
                        severity='medium',
                        title="New Network Gap Identified",
                        description=change.change_description,
                        affected_nodes=change.affected_entities,
                        recommended_actions=[
                            'Assess gap impact',
                            'Develop mitigation plan',
                            'Allocate resources'
                        ],
                        created_at=datetime.now(),
                        expires_at=datetime.now() + timedelta(days=14),
                        acknowledged=False
                    )
                    alerts.append(alert)
            
            # Store alerts
            for alert in alerts:
                self.active_alerts[alert.alert_id] = alert
            
            self.logger.info(f"Created {len(alerts)} network alerts")
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error creating network alerts: {str(e)}")
            raise
    
    async def _create_network_snapshot(self, network_id: str, network: InfluenceNetwork):
        """Create a snapshot of the current network state"""
        snapshot = {
            'network_id': network_id,
            'timestamp': datetime.now().isoformat(),
            'nodes': [
                {
                    'id': node.id,
                    'name': node.name,
                    'influence_score': node.influence_score,
                    'centrality_score': node.centrality_score,
                    'connections': node.connections.copy(),
                    'influence_type': node.influence_type
                }
                for node in network.nodes
            ],
            'edges': [
                {
                    'source': edge.source_id,
                    'target': edge.target_id,
                    'strength': edge.strength,
                    'influence_flow': edge.influence_flow
                }
                for edge in network.edges
            ],
            'metrics': network.network_metrics.copy()
        }
        
        if network_id not in self.network_snapshots:
            self.network_snapshots[network_id] = []
        
        self.network_snapshots[network_id].append(snapshot)
        
        # Keep only last 10 snapshots
        if len(self.network_snapshots[network_id]) > 10:
            self.network_snapshots[network_id] = self.network_snapshots[network_id][-10:]
    
    async def _analyze_shift_factors(
        self,
        current_node: InfluenceNode,
        previous_node: Dict[str, Any],
        network: InfluenceNetwork
    ) -> List[str]:
        """Analyze factors contributing to influence shift"""
        factors = []
        
        # Connection changes
        prev_connections = set(previous_node.get('connections', []))
        curr_connections = set(current_node.connections)
        
        if len(curr_connections) > len(prev_connections):
            factors.append('increased_connections')
        elif len(curr_connections) < len(prev_connections):
            factors.append('decreased_connections')
        
        # Centrality changes
        prev_centrality = previous_node.get('centrality_score', 0)
        if current_node.centrality_score > prev_centrality + 0.1:
            factors.append('improved_centrality')
        elif current_node.centrality_score < prev_centrality - 0.1:
            factors.append('reduced_centrality')
        
        # External factors (would be determined by external data)
        factors.append('external_events')
        
        return factors
    
    async def _simulate_gap_detection(self, network: InfluenceNetwork) -> List[NetworkGap]:
        """Simulate gap detection (would integrate with influence mapping engine)"""
        gaps = []
        
        # Simulate some gaps based on network structure
        isolated_nodes = [node for node in network.nodes if len(node.connections) < 2]
        
        for node in isolated_nodes:
            gap = NetworkGap(
                gap_id=f"isolation_{node.id}",
                gap_type="missing_connection",
                title=f"Isolated Node: {node.name}",
                description=f"Node {node.name} has limited connections",
                priority='medium',
                target_nodes=[node.id],
                affected_areas=['connectivity'],
                recommended_actions=['Identify potential connections'],
                potential_impact=0.6,
                effort_required=0.4,
                timeline_estimate='2-4 weeks',
                success_criteria=['Increase connections'],
                created_at=datetime.now()
            )
            gaps.append(gap)
        
        return gaps
    
    def get_monitoring_status(self, monitoring_id: str) -> Optional[Dict[str, Any]]:
        """Get monitoring status for a specific monitoring session"""
        if monitoring_id not in self.monitored_networks:
            return None
        
        monitoring_data = self.monitored_networks[monitoring_id]
        network = monitoring_data['network']
        
        return {
            'monitoring_id': monitoring_id,
            'network_id': network.id,
            'network_name': network.name,
            'status': monitoring_data['status'],
            'started_at': monitoring_data['started_at'].isoformat(),
            'last_check': monitoring_data['last_check'].isoformat(),
            'total_changes': len(self.change_history.get(network.id, [])),
            'active_alerts': len([a for a in self.active_alerts.values() 
                                if a.network_id == network.id and not a.acknowledged])
        }
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            return True
        return False
    
    def get_network_alerts(self, network_id: str, include_acknowledged: bool = False) -> List[NetworkAlert]:
        """Get alerts for a specific network"""
        alerts = [
            alert for alert in self.active_alerts.values()
            if alert.network_id == network_id
        ]
        
        if not include_acknowledged:
            alerts = [alert for alert in alerts if not alert.acknowledged]
        
        return sorted(alerts, key=lambda a: a.created_at, reverse=True)
    
    async def stop_monitoring(self, monitoring_id: str) -> bool:
        """Stop monitoring a network"""
        if monitoring_id in self.monitored_networks:
            self.monitored_networks[monitoring_id]['status'] = 'stopped'
            self.logger.info(f"Stopped monitoring session {monitoring_id}")
            return True
        return False