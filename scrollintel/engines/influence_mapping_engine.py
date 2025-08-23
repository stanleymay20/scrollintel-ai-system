"""
Influence Mapping Engine for Global Influence Network System

This engine provides industry hierarchy mapping, network centrality analysis,
and competitive positioning for building global influence networks.
"""

import logging
import asyncio
from typing import List, Dict, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict, deque

from ..models.influence_strategy_models import InfluenceTarget
from ..models.relationship_models import RelationshipProfile, RelationshipType
from ..models.ecosystem_models import PartnershipOpportunity


@dataclass
class InfluenceNode:
    """Represents a node in the influence network"""
    id: str
    name: str
    title: str
    organization: str
    industry: str
    influence_score: float
    centrality_score: float
    connections: List[str]
    influence_type: str  # 'thought_leader', 'decision_maker', 'connector', 'gatekeeper'
    geographic_reach: List[str]
    expertise_areas: List[str]
    last_updated: datetime


@dataclass
class InfluenceEdge:
    """Represents a connection between influence nodes"""
    source_id: str
    target_id: str
    relationship_type: str
    strength: float
    direction: str  # 'bidirectional', 'source_to_target', 'target_to_source'
    interaction_frequency: float
    last_interaction: Optional[datetime]
    influence_flow: float  # How much influence flows through this edge


@dataclass
class InfluenceNetwork:
    """Represents the complete influence network"""
    id: str
    name: str
    industry: str
    nodes: List[InfluenceNode]
    edges: List[InfluenceEdge]
    network_metrics: Dict[str, float]
    competitive_position: Dict[str, Any]
    created_at: datetime
    last_updated: datetime


@dataclass
class NetworkGap:
    """Represents a gap in the influence network"""
    gap_id: str
    gap_type: str  # 'missing_connection', 'weak_influence', 'competitor_advantage'
    description: str
    priority: str  # 'high', 'medium', 'low'
    target_nodes: List[str]
    recommended_actions: List[str]
    potential_impact: float
    effort_required: float


@dataclass
class CompetitivePosition:
    """Represents competitive positioning in influence networks"""
    position_id: str
    industry: str
    our_influence_score: float
    competitor_scores: Dict[str, float]
    market_share_influence: float
    key_advantages: List[str]
    vulnerabilities: List[str]
    strategic_recommendations: List[str]
    trend_analysis: Dict[str, float]


class InfluenceMappingEngine:
    """Engine for mapping and analyzing global influence networks"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.networks = {}
        self.node_cache = {}
        self.centrality_algorithms = {
            'betweenness': self._calculate_betweenness_centrality,
            'closeness': self._calculate_closeness_centrality,
            'eigenvector': self._calculate_eigenvector_centrality,
            'pagerank': self._calculate_pagerank_centrality
        }
    
    async def create_influence_network(
        self,
        industry: str,
        stakeholders: List[RelationshipProfile],
        partnerships: List[PartnershipOpportunity],
        external_data: Optional[Dict[str, Any]] = None
    ) -> InfluenceNetwork:
        """Create comprehensive influence network mapping"""
        try:
            self.logger.info(f"Creating influence network for industry: {industry}")
            
            # Convert stakeholders to influence nodes
            nodes = await self._create_influence_nodes(stakeholders, external_data)
            
            # Create edges from relationships and partnerships
            edges = await self._create_influence_edges(nodes, partnerships, external_data)
            
            # Calculate network metrics
            network_metrics = await self._calculate_network_metrics(nodes, edges)
            
            # Analyze competitive position
            competitive_position = await self._analyze_competitive_position(
                industry, nodes, edges, external_data
            )
            
            # Create network object
            network = InfluenceNetwork(
                id=f"influence_network_{industry}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name=f"{industry.title()} Influence Network",
                industry=industry,
                nodes=nodes,
                edges=edges,
                network_metrics=network_metrics,
                competitive_position=competitive_position,
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            
            # Cache the network
            self.networks[network.id] = network
            
            self.logger.info(f"Created influence network with {len(nodes)} nodes and {len(edges)} edges")
            return network
            
        except Exception as e:
            self.logger.error(f"Error creating influence network: {str(e)}")
            raise
    
    async def analyze_network_centrality(
        self, 
        network: InfluenceNetwork,
        algorithms: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Analyze network centrality using multiple algorithms"""
        try:
            if algorithms is None:
                algorithms = ['betweenness', 'closeness', 'eigenvector', 'pagerank']
            
            centrality_results = {}
            
            for algorithm in algorithms:
                if algorithm in self.centrality_algorithms:
                    self.logger.info(f"Calculating {algorithm} centrality")
                    centrality_scores = await self.centrality_algorithms[algorithm](network)
                    centrality_results[algorithm] = centrality_scores
            
            # Update node centrality scores with combined metric
            await self._update_node_centrality_scores(network, centrality_results)
            
            return centrality_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing network centrality: {str(e)}")
            raise
    
    async def identify_network_gaps(
        self, 
        network: InfluenceNetwork,
        target_objectives: List[str]
    ) -> List[NetworkGap]:
        """Identify gaps in the influence network"""
        try:
            gaps = []
            
            # Identify missing key connections
            missing_connections = await self._identify_missing_connections(network)
            gaps.extend(missing_connections)
            
            # Identify weak influence areas
            weak_areas = await self._identify_weak_influence_areas(network, target_objectives)
            gaps.extend(weak_areas)
            
            # Identify competitor advantages
            competitor_advantages = await self._identify_competitor_advantages(network)
            gaps.extend(competitor_advantages)
            
            # Prioritize gaps
            prioritized_gaps = await self._prioritize_gaps(gaps)
            
            self.logger.info(f"Identified {len(prioritized_gaps)} network gaps")
            return prioritized_gaps
            
        except Exception as e:
            self.logger.error(f"Error identifying network gaps: {str(e)}")
            raise
    
    async def generate_network_expansion_strategy(
        self,
        network: InfluenceNetwork,
        gaps: List[NetworkGap],
        resources: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate strategy for expanding influence network"""
        try:
            strategy = {
                'expansion_targets': [],
                'relationship_building_plan': [],
                'partnership_opportunities': [],
                'content_strategy': [],
                'timeline': {},
                'resource_allocation': {},
                'success_metrics': []
            }
            
            # Analyze high-priority gaps
            high_priority_gaps = [gap for gap in gaps if gap.priority == 'high']
            
            # Generate expansion targets
            expansion_targets = await self._generate_expansion_targets(
                network, high_priority_gaps, resources
            )
            strategy['expansion_targets'] = expansion_targets
            
            # Create relationship building plan
            relationship_plan = await self._create_relationship_building_plan(
                network, expansion_targets, resources
            )
            strategy['relationship_building_plan'] = relationship_plan
            
            # Identify partnership opportunities
            partnership_opportunities = await self._identify_partnership_opportunities(
                network, expansion_targets
            )
            strategy['partnership_opportunities'] = partnership_opportunities
            
            # Generate content strategy
            content_strategy = await self._generate_content_strategy(
                network, expansion_targets
            )
            strategy['content_strategy'] = content_strategy
            
            # Create implementation timeline
            timeline = await self._create_implementation_timeline(
                expansion_targets, relationship_plan, resources
            )
            strategy['timeline'] = timeline
            
            # Calculate resource allocation
            resource_allocation = await self._calculate_resource_allocation(
                expansion_targets, relationship_plan, resources
            )
            strategy['resource_allocation'] = resource_allocation
            
            # Define success metrics
            success_metrics = await self._define_success_metrics(
                network, expansion_targets, gaps
            )
            strategy['success_metrics'] = success_metrics
            
            self.logger.info("Generated comprehensive network expansion strategy")
            return strategy
            
        except Exception as e:
            self.logger.error(f"Error generating expansion strategy: {str(e)}")
            raise
    
    async def _create_influence_nodes(
        self,
        stakeholders: List[RelationshipProfile],
        external_data: Optional[Dict[str, Any]]
    ) -> List[InfluenceNode]:
        """Create influence nodes from stakeholders and external data"""
        nodes = []
        
        for stakeholder in stakeholders:
            # Calculate influence score
            influence_score = await self._calculate_influence_score(stakeholder, external_data)
            
            # Determine influence type
            influence_type = await self._determine_influence_type(stakeholder)
            
            # Extract geographic reach
            geographic_reach = await self._extract_geographic_reach(stakeholder, external_data)
            
            # Extract expertise areas
            expertise_areas = await self._extract_expertise_areas(stakeholder)
            
            node = InfluenceNode(
                id=stakeholder.stakeholder_id,
                name=stakeholder.name,
                title=stakeholder.title,
                organization=stakeholder.organization,
                industry=getattr(stakeholder, 'industry', 'technology'),
                influence_score=influence_score,
                centrality_score=0.0,  # Will be calculated later
                connections=[],  # Will be populated when creating edges
                influence_type=influence_type,
                geographic_reach=geographic_reach,
                expertise_areas=expertise_areas,
                last_updated=datetime.now()
            )
            
            nodes.append(node)
            self.node_cache[node.id] = node
        
        return nodes
    
    async def _create_influence_edges(
        self,
        nodes: List[InfluenceNode],
        partnerships: List[PartnershipOpportunity],
        external_data: Optional[Dict[str, Any]]
    ) -> List[InfluenceEdge]:
        """Create influence edges from relationships and partnerships"""
        edges = []
        node_dict = {node.id: node for node in nodes}
        
        # Create edges from existing relationships
        for node in nodes:
            # Get connections from external data or infer from partnerships
            connections = await self._infer_connections(node, nodes, partnerships, external_data)
            
            for connection_id, connection_data in connections.items():
                if connection_id in node_dict and connection_id != node.id:
                    edge = InfluenceEdge(
                        source_id=node.id,
                        target_id=connection_id,
                        relationship_type=connection_data.get('type', 'professional'),
                        strength=connection_data.get('strength', 0.5),
                        direction=connection_data.get('direction', 'bidirectional'),
                        interaction_frequency=connection_data.get('frequency', 0.3),
                        last_interaction=connection_data.get('last_interaction'),
                        influence_flow=connection_data.get('influence_flow', 0.4)
                    )
                    edges.append(edge)
                    
                    # Update node connections
                    if connection_id not in node.connections:
                        node.connections.append(connection_id)
        
        return edges
    
    async def _calculate_network_metrics(
        self,
        nodes: List[InfluenceNode],
        edges: List[InfluenceEdge]
    ) -> Dict[str, float]:
        """Calculate comprehensive network metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['node_count'] = len(nodes)
        metrics['edge_count'] = len(edges)
        metrics['density'] = len(edges) / (len(nodes) * (len(nodes) - 1)) if len(nodes) > 1 else 0
        
        # Influence distribution
        influence_scores = [node.influence_score for node in nodes]
        metrics['avg_influence_score'] = np.mean(influence_scores)
        metrics['influence_std'] = np.std(influence_scores)
        metrics['max_influence_score'] = np.max(influence_scores)
        
        # Connectivity metrics
        connection_counts = [len(node.connections) for node in nodes]
        metrics['avg_connections'] = np.mean(connection_counts)
        metrics['max_connections'] = np.max(connection_counts) if connection_counts else 0
        
        # Network strength
        edge_strengths = [edge.strength for edge in edges]
        metrics['avg_edge_strength'] = np.mean(edge_strengths) if edge_strengths else 0
        metrics['network_cohesion'] = metrics['avg_edge_strength'] * metrics['density']
        
        return metrics
    
    async def _analyze_competitive_position(
        self,
        industry: str,
        nodes: List[InfluenceNode],
        edges: List[InfluenceEdge],
        external_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze competitive position in the influence network"""
        position = {
            'industry': industry,
            'our_network_strength': 0.0,
            'competitor_analysis': {},
            'market_position': 'unknown',
            'key_advantages': [],
            'vulnerabilities': [],
            'opportunities': []
        }
        
        # Calculate our network strength
        our_influence = sum(node.influence_score for node in nodes)
        our_connections = sum(len(node.connections) for node in nodes)
        position['our_network_strength'] = our_influence * (our_connections / len(nodes)) if nodes else 0
        
        # Analyze competitor presence (simplified - would use external data in practice)
        if external_data and 'competitors' in external_data:
            competitor_data = external_data['competitors']
            for competitor, data in competitor_data.items():
                position['competitor_analysis'][competitor] = {
                    'influence_score': data.get('influence_score', 0.5),
                    'network_size': data.get('network_size', 100),
                    'market_share': data.get('market_share', 0.1)
                }
        
        # Determine market position
        if position['our_network_strength'] > 0.8:
            position['market_position'] = 'dominant'
        elif position['our_network_strength'] > 0.6:
            position['market_position'] = 'strong'
        elif position['our_network_strength'] > 0.4:
            position['market_position'] = 'competitive'
        else:
            position['market_position'] = 'emerging'
        
        # Identify advantages and vulnerabilities
        high_influence_nodes = [n for n in nodes if n.influence_score > 0.7]
        if high_influence_nodes:
            position['key_advantages'].append(f"Strong relationships with {len(high_influence_nodes)} high-influence individuals")
        
        isolated_nodes = [n for n in nodes if len(n.connections) < 2]
        if isolated_nodes:
            position['vulnerabilities'].append(f"{len(isolated_nodes)} stakeholders with limited connections")
        
        return position
    
    async def _calculate_betweenness_centrality(self, network: InfluenceNetwork) -> Dict[str, float]:
        """Calculate betweenness centrality for all nodes"""
        centrality = {}
        nodes = {node.id: node for node in network.nodes}
        
        # Build adjacency list
        adj_list = defaultdict(list)
        for edge in network.edges:
            adj_list[edge.source_id].append(edge.target_id)
            if edge.direction == 'bidirectional':
                adj_list[edge.target_id].append(edge.source_id)
        
        # Calculate betweenness centrality using Brandes algorithm (simplified)
        for node_id in nodes:
            centrality[node_id] = 0.0
        
        for source in nodes:
            # BFS to find shortest paths
            stack = []
            paths = defaultdict(list)
            sigma = defaultdict(int)
            sigma[source] = 1
            distances = {source: 0}
            queue = deque([source])
            
            while queue:
                current = queue.popleft()
                stack.append(current)
                
                for neighbor in adj_list[current]:
                    if neighbor not in distances:
                        distances[neighbor] = distances[current] + 1
                        queue.append(neighbor)
                    
                    if distances[neighbor] == distances[current] + 1:
                        sigma[neighbor] += sigma[current]
                        paths[neighbor].append(current)
            
            # Accumulate betweenness values
            delta = defaultdict(float)
            while stack:
                current = stack.pop()
                for predecessor in paths[current]:
                    delta[predecessor] += (sigma[predecessor] / sigma[current]) * (1 + delta[current])
                
                if current != source:
                    centrality[current] += delta[current]
        
        # Normalize
        n = len(nodes)
        if n > 2:
            norm = 2.0 / ((n - 1) * (n - 2))
            for node_id in centrality:
                centrality[node_id] *= norm
        
        return centrality
    
    async def _calculate_closeness_centrality(self, network: InfluenceNetwork) -> Dict[str, float]:
        """Calculate closeness centrality for all nodes"""
        centrality = {}
        nodes = {node.id: node for node in network.nodes}
        
        # Build adjacency list
        adj_list = defaultdict(list)
        for edge in network.edges:
            adj_list[edge.source_id].append(edge.target_id)
            if edge.direction == 'bidirectional':
                adj_list[edge.target_id].append(edge.source_id)
        
        for source in nodes:
            # BFS to calculate shortest path distances
            distances = {source: 0}
            queue = deque([source])
            
            while queue:
                current = queue.popleft()
                for neighbor in adj_list[current]:
                    if neighbor not in distances:
                        distances[neighbor] = distances[current] + 1
                        queue.append(neighbor)
            
            # Calculate closeness centrality
            total_distance = sum(distances.values())
            if total_distance > 0:
                centrality[source] = (len(distances) - 1) / total_distance
            else:
                centrality[source] = 0.0
        
        return centrality
    
    async def _calculate_eigenvector_centrality(self, network: InfluenceNetwork) -> Dict[str, float]:
        """Calculate eigenvector centrality for all nodes"""
        nodes = {node.id: i for i, node in enumerate(network.nodes)}
        n = len(nodes)
        
        if n == 0:
            return {}
        
        # Build adjacency matrix
        adj_matrix = np.zeros((n, n))
        for edge in network.edges:
            if edge.source_id in nodes and edge.target_id in nodes:
                i, j = nodes[edge.source_id], nodes[edge.target_id]
                adj_matrix[i][j] = edge.strength
                if edge.direction == 'bidirectional':
                    adj_matrix[j][i] = edge.strength
        
        # Power iteration to find dominant eigenvector
        centrality_vector = np.ones(n) / n
        for _ in range(100):  # Max iterations
            new_vector = adj_matrix.T @ centrality_vector
            norm = np.linalg.norm(new_vector)
            if norm > 0:
                new_vector = new_vector / norm
            
            if np.allclose(centrality_vector, new_vector, atol=1e-6):
                break
            centrality_vector = new_vector
        
        # Convert back to node IDs
        centrality = {}
        for node_id, index in nodes.items():
            centrality[node_id] = float(centrality_vector[index])
        
        return centrality
    
    async def _calculate_pagerank_centrality(self, network: InfluenceNetwork) -> Dict[str, float]:
        """Calculate PageRank centrality for all nodes"""
        nodes = {node.id: node for node in network.nodes}
        centrality = {node_id: 1.0 / len(nodes) for node_id in nodes}
        
        # Build adjacency information
        outgoing = defaultdict(list)
        incoming = defaultdict(list)
        
        for edge in network.edges:
            outgoing[edge.source_id].append((edge.target_id, edge.strength))
            incoming[edge.target_id].append((edge.source_id, edge.strength))
            
            if edge.direction == 'bidirectional':
                outgoing[edge.target_id].append((edge.source_id, edge.strength))
                incoming[edge.source_id].append((edge.target_id, edge.strength))
        
        # PageRank iteration
        damping = 0.85
        for _ in range(100):  # Max iterations
            new_centrality = {}
            
            for node_id in nodes:
                rank = (1 - damping) / len(nodes)
                
                for source_id, weight in incoming[node_id]:
                    out_degree = sum(w for _, w in outgoing[source_id])
                    if out_degree > 0:
                        rank += damping * centrality[source_id] * (weight / out_degree)
                
                new_centrality[node_id] = rank
            
            # Check convergence
            if all(abs(new_centrality[node_id] - centrality[node_id]) < 1e-6 
                   for node_id in nodes):
                break
            
            centrality = new_centrality
        
        return centrality
    
    async def _update_node_centrality_scores(
        self,
        network: InfluenceNetwork,
        centrality_results: Dict[str, Dict[str, float]]
    ):
        """Update node centrality scores with combined metric"""
        for node in network.nodes:
            scores = []
            for algorithm, results in centrality_results.items():
                if node.id in results:
                    scores.append(results[node.id])
            
            # Combined centrality score (average of all algorithms)
            node.centrality_score = np.mean(scores) if scores else 0.0
    
    async def _calculate_influence_score(
        self,
        stakeholder: RelationshipProfile,
        external_data: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate influence score for a stakeholder"""
        base_score = 0.5
        
        # Factor in relationship strength
        base_score += stakeholder.relationship_strength * 0.3
        
        # Factor in trust metrics
        if stakeholder.trust_metrics:
            base_score += stakeholder.trust_metrics.overall_trust_score * 0.2
        
        # Factor in influence level
        base_score += stakeholder.influence_level * 0.3
        
        # Factor in decision making power
        base_score += stakeholder.decision_making_power * 0.2
        
        # External data adjustments
        if external_data and 'influence_multipliers' in external_data:
            multipliers = external_data['influence_multipliers']
            if stakeholder.stakeholder_id in multipliers:
                base_score *= multipliers[stakeholder.stakeholder_id]
        
        return min(base_score, 1.0)
    
    async def _determine_influence_type(self, stakeholder: RelationshipProfile) -> str:
        """Determine the type of influence a stakeholder has"""
        if stakeholder.relationship_type == RelationshipType.BOARD_MEMBER:
            return 'decision_maker'
        elif stakeholder.influence_level > 0.8:
            return 'thought_leader'
        elif len(stakeholder.network_connections) > 10:
            return 'connector'
        else:
            return 'gatekeeper'
    
    async def _extract_geographic_reach(
        self,
        stakeholder: RelationshipProfile,
        external_data: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Extract geographic reach for a stakeholder"""
        # Default based on organization or external data
        if external_data and 'geographic_data' in external_data:
            geo_data = external_data['geographic_data']
            if stakeholder.stakeholder_id in geo_data:
                return geo_data[stakeholder.stakeholder_id]
        
        # Default geographic reach
        return ['north_america', 'global']
    
    async def _extract_expertise_areas(self, stakeholder: RelationshipProfile) -> List[str]:
        """Extract expertise areas for a stakeholder"""
        expertise = []
        
        # Extract from interests and priorities
        expertise.extend(stakeholder.key_interests[:3])
        expertise.extend([priority.name for priority in stakeholder.business_priorities[:3]])
        
        # Add default based on relationship type
        if stakeholder.relationship_type == RelationshipType.BOARD_MEMBER:
            expertise.append('governance')
        elif stakeholder.relationship_type == RelationshipType.INVESTOR:
            expertise.append('investment')
        elif stakeholder.relationship_type == RelationshipType.EXECUTIVE:
            expertise.append('leadership')
        
        return list(set(expertise))  # Remove duplicates
    
    async def _infer_connections(
        self,
        node: InfluenceNode,
        all_nodes: List[InfluenceNode],
        partnerships: List[PartnershipOpportunity],
        external_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Infer connections between nodes"""
        connections = {}
        
        # Connect nodes in same organization
        for other_node in all_nodes:
            if (other_node.id != node.id and 
                other_node.organization == node.organization):
                connections[other_node.id] = {
                    'type': 'organizational',
                    'strength': 0.7,
                    'direction': 'bidirectional',
                    'frequency': 0.8,
                    'influence_flow': 0.6
                }
        
        # Connect nodes with similar expertise
        for other_node in all_nodes:
            if (other_node.id != node.id and 
                set(node.expertise_areas) & set(other_node.expertise_areas)):
                if other_node.id not in connections:
                    connections[other_node.id] = {
                        'type': 'professional',
                        'strength': 0.5,
                        'direction': 'bidirectional',
                        'frequency': 0.4,
                        'influence_flow': 0.4
                    }
        
        # Connect through partnerships
        for partnership in partnerships:
            if partnership.partner_name in [n.organization for n in all_nodes]:
                partner_nodes = [n for n in all_nodes if n.organization == partnership.partner_name]
                for partner_node in partner_nodes:
                    if partner_node.id not in connections:
                        connections[partner_node.id] = {
                            'type': 'partnership',
                            'strength': partnership.strategic_value,
                            'direction': 'bidirectional',
                            'frequency': 0.3,
                            'influence_flow': partnership.strategic_value * 0.8
                        }
        
        return connections
    
    async def _identify_missing_connections(self, network: InfluenceNetwork) -> List[NetworkGap]:
        """Identify missing key connections in the network"""
        gaps = []
        
        # Find high-influence nodes with few connections
        for node in network.nodes:
            if node.influence_score > 0.7 and len(node.connections) < 3:
                gap = NetworkGap(
                    gap_id=f"missing_conn_{node.id}",
                    gap_type='missing_connection',
                    description=f"High-influence node {node.name} has limited connections",
                    priority='high',
                    target_nodes=[node.id],
                    recommended_actions=[
                        f"Identify potential connections for {node.name}",
                        "Develop relationship building strategy",
                        "Leverage existing mutual connections"
                    ],
                    potential_impact=node.influence_score * 0.8,
                    effort_required=0.6
                )
                gaps.append(gap)
        
        return gaps
    
    async def _identify_weak_influence_areas(
        self,
        network: InfluenceNetwork,
        target_objectives: List[str]
    ) -> List[NetworkGap]:
        """Identify areas where influence is weak"""
        gaps = []
        
        # Analyze influence by expertise area
        expertise_influence = defaultdict(list)
        for node in network.nodes:
            for expertise in node.expertise_areas:
                expertise_influence[expertise].append(node.influence_score)
        
        # Find weak areas
        for expertise, scores in expertise_influence.items():
            avg_influence = np.mean(scores)
            if avg_influence < 0.5 and expertise in target_objectives:
                gap = NetworkGap(
                    gap_id=f"weak_influence_{expertise}",
                    gap_type='weak_influence',
                    description=f"Weak influence in {expertise} area",
                    priority='medium',
                    target_nodes=[],
                    recommended_actions=[
                        f"Identify thought leaders in {expertise}",
                        "Develop content strategy for this area",
                        "Build partnerships with domain experts"
                    ],
                    potential_impact=0.7,
                    effort_required=0.8
                )
                gaps.append(gap)
        
        return gaps
    
    async def _identify_competitor_advantages(self, network: InfluenceNetwork) -> List[NetworkGap]:
        """Identify areas where competitors have advantages"""
        gaps = []
        
        # Analyze competitive position
        competitive_data = network.competitive_position
        if isinstance(competitive_data, dict) and 'competitor_analysis' in competitive_data:
            for competitor, data in competitive_data['competitor_analysis'].items():
                if data.get('influence_score', 0) > network.network_metrics.get('avg_influence_score', 0):
                    gap = NetworkGap(
                        gap_id=f"competitor_advantage_{competitor}",
                        gap_type='competitor_advantage',
                        description=f"{competitor} has stronger influence network",
                        priority='high',
                        target_nodes=[],
                        recommended_actions=[
                            f"Analyze {competitor}'s key relationships",
                            "Identify opportunities to compete for influence",
                            "Develop counter-strategies"
                        ],
                        potential_impact=0.8,
                        effort_required=0.9
                    )
                    gaps.append(gap)
        
        return gaps
    
    async def _prioritize_gaps(self, gaps: List[NetworkGap]) -> List[NetworkGap]:
        """Prioritize network gaps by impact and effort"""
        # Calculate priority score (impact / effort)
        for gap in gaps:
            gap.priority_score = gap.potential_impact / max(gap.effort_required, 0.1)
        
        # Sort by priority score and existing priority
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        gaps.sort(key=lambda g: (priority_order.get(g.priority, 0), g.priority_score), reverse=True)
        
        return gaps
    
    async def _generate_expansion_targets(
        self,
        network: InfluenceNetwork,
        gaps: List[NetworkGap],
        resources: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate specific targets for network expansion"""
        targets = []
        
        for gap in gaps[:5]:  # Top 5 gaps
            if gap.gap_type == 'missing_connection':
                # Find potential connections for isolated high-influence nodes
                target_node = next((n for n in network.nodes if n.id in gap.target_nodes), None)
                if target_node:
                    targets.append({
                        'type': 'direct_connection',
                        'target_node': target_node.id,
                        'target_name': target_node.name,
                        'strategy': 'relationship_building',
                        'priority': gap.priority,
                        'expected_impact': gap.potential_impact
                    })
            
            elif gap.gap_type == 'weak_influence':
                # Find thought leaders in weak areas
                expertise_area = gap.gap_id.replace('weak_influence_', '')
                targets.append({
                    'type': 'expertise_expansion',
                    'expertise_area': expertise_area,
                    'strategy': 'thought_leadership',
                    'priority': gap.priority,
                    'expected_impact': gap.potential_impact
                })
        
        return targets
    
    async def _create_relationship_building_plan(
        self,
        network: InfluenceNetwork,
        targets: List[Dict[str, Any]],
        resources: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create detailed relationship building plan"""
        plan = []
        
        for target in targets:
            if target['type'] == 'direct_connection':
                plan.append({
                    'target_id': target['target_node'],
                    'target_name': target['target_name'],
                    'approach': 'direct_outreach',
                    'timeline': '2-4 weeks',
                    'resources_needed': ['research_time', 'meeting_coordination'],
                    'success_metrics': ['initial_meeting_scheduled', 'positive_response_rate'],
                    'next_steps': [
                        'Research target background and interests',
                        'Identify mutual connections for warm introduction',
                        'Prepare value proposition for initial meeting'
                    ]
                })
            
            elif target['type'] == 'expertise_expansion':
                plan.append({
                    'expertise_area': target['expertise_area'],
                    'approach': 'content_and_events',
                    'timeline': '1-3 months',
                    'resources_needed': ['content_creation', 'event_participation'],
                    'success_metrics': ['thought_leadership_recognition', 'expert_connections'],
                    'next_steps': [
                        f'Develop content strategy for {target["expertise_area"]}',
                        'Identify key events and conferences',
                        'Build relationships with domain experts'
                    ]
                })
        
        return plan
    
    async def _identify_partnership_opportunities(
        self,
        network: InfluenceNetwork,
        targets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify partnership opportunities for network expansion"""
        opportunities = []
        
        # Analyze organizations in the network
        organizations = set(node.organization for node in network.nodes)
        
        for org in organizations:
            org_nodes = [n for n in network.nodes if n.organization == org]
            avg_influence = np.mean([n.influence_score for n in org_nodes])
            
            if avg_influence > 0.6:
                opportunities.append({
                    'organization': org,
                    'partnership_type': 'strategic_alliance',
                    'key_contacts': [n.name for n in org_nodes[:3]],
                    'potential_value': avg_influence,
                    'recommended_approach': 'formal_partnership_proposal'
                })
        
        return opportunities
    
    async def _generate_content_strategy(
        self,
        network: InfluenceNetwork,
        targets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate content strategy for influence building"""
        strategy = []
        
        # Identify key expertise areas
        expertise_areas = set()
        for node in network.nodes:
            expertise_areas.update(node.expertise_areas)
        
        for area in list(expertise_areas)[:5]:  # Top 5 areas
            strategy.append({
                'expertise_area': area,
                'content_types': ['thought_leadership_articles', 'industry_insights', 'trend_analysis'],
                'distribution_channels': ['linkedin', 'industry_publications', 'conferences'],
                'target_audience': f'{area}_professionals',
                'success_metrics': ['engagement_rate', 'share_count', 'expert_recognition']
            })
        
        return strategy
    
    async def _create_implementation_timeline(
        self,
        targets: List[Dict[str, Any]],
        plan: List[Dict[str, Any]],
        resources: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create implementation timeline"""
        timeline = {
            'phase_1': {
                'duration': '1-2 months',
                'activities': ['Research and preparation', 'Initial outreach'],
                'deliverables': ['Target research completed', 'Initial meetings scheduled']
            },
            'phase_2': {
                'duration': '2-4 months',
                'activities': ['Relationship building', 'Content creation'],
                'deliverables': ['Key relationships established', 'Thought leadership content published']
            },
            'phase_3': {
                'duration': '4-6 months',
                'activities': ['Partnership development', 'Network expansion'],
                'deliverables': ['Strategic partnerships formed', 'Network influence increased']
            }
        }
        
        return timeline
    
    async def _calculate_resource_allocation(
        self,
        targets: List[Dict[str, Any]],
        plan: List[Dict[str, Any]],
        resources: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate resource allocation for network expansion"""
        allocation = {
            'time_investment': {
                'research': '20%',
                'relationship_building': '40%',
                'content_creation': '25%',
                'partnership_development': '15%'
            },
            'budget_allocation': {
                'events_and_conferences': '30%',
                'content_production': '25%',
                'travel_and_meetings': '25%',
                'tools_and_platforms': '20%'
            },
            'team_allocation': {
                'relationship_manager': '1 FTE',
                'content_strategist': '0.5 FTE',
                'partnership_manager': '0.5 FTE'
            }
        }
        
        return allocation
    
    async def _define_success_metrics(
        self,
        network: InfluenceNetwork,
        targets: List[Dict[str, Any]],
        gaps: List[NetworkGap]
    ) -> List[Dict[str, str]]:
        """Define success metrics for network expansion"""
        metrics = [
            {
                'metric': 'Network Growth Rate',
                'target': '25% increase in nodes per quarter',
                'measurement': 'Count of new high-value connections'
            },
            {
                'metric': 'Influence Score Improvement',
                'target': '15% increase in average influence score',
                'measurement': 'Weighted average of node influence scores'
            },
            {
                'metric': 'Network Density',
                'target': '20% increase in network density',
                'measurement': 'Ratio of actual to possible connections'
            },
            {
                'metric': 'Gap Closure Rate',
                'target': '80% of high-priority gaps addressed',
                'measurement': 'Percentage of gaps with implemented solutions'
            },
            {
                'metric': 'Partnership Conversion',
                'target': '30% of identified opportunities converted',
                'measurement': 'Number of formal partnerships established'
            }
        ]
        
        return metrics