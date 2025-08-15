"""
Data Lineage Tracking System for Advanced Analytics Dashboard

Provides comprehensive data lineage tracking, audit trails,
and compliance reporting for multi-source data integration.
"""

from typing import Dict, List, Any, Optional, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
import uuid
from collections import defaultdict, deque
import networkx as nx

logger = logging.getLogger(__name__)


class LineageEventType(Enum):
    DATA_INGESTION = "data_ingestion"
    DATA_TRANSFORMATION = "data_transformation"
    DATA_VALIDATION = "data_validation"
    DATA_RECONCILIATION = "data_reconciliation"
    DATA_EXPORT = "data_export"
    SCHEMA_CHANGE = "schema_change"
    QUALITY_CHECK = "quality_check"


class DataClassification(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class DataAsset:
    """Represents a data asset in the lineage graph"""
    id: str
    name: str
    type: str  # table, file, api, etc.
    source_system: str
    schema_info: Dict[str, Any]
    classification: DataClassification
    owner: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LineageEvent:
    """Represents a single event in data lineage"""
    id: str
    event_type: LineageEventType
    timestamp: datetime
    source_assets: List[str]
    target_assets: List[str]
    transformation_details: Dict[str, Any]
    user_id: str
    system_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataTransformation:
    """Represents a data transformation operation"""
    id: str
    name: str
    description: str
    transformation_type: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    business_rules: List[str]
    created_by: str
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ComplianceRule:
    """Represents a compliance rule for data handling"""
    id: str
    name: str
    description: str
    regulation: str  # GDPR, CCPA, SOX, etc.
    data_classifications: List[DataClassification]
    retention_period_days: Optional[int]
    access_restrictions: List[str]
    audit_requirements: Dict[str, Any]


@dataclass
class LineageQuery:
    """Query parameters for lineage analysis"""
    asset_id: Optional[str] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    event_types: Optional[List[LineageEventType]] = None
    max_depth: int = 10
    direction: str = "both"  # upstream, downstream, both


class DataLineageTracker:
    """
    Comprehensive data lineage tracking and audit system
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.assets: Dict[str, DataAsset] = {}
        self.events: List[LineageEvent] = []
        self.transformations: Dict[str, DataTransformation] = {}
        self.compliance_rules: Dict[str, ComplianceRule] = {}
        self.lineage_graph = nx.DiGraph()
        
    def register_data_asset(self, asset: DataAsset) -> bool:
        """Register a data asset for lineage tracking"""
        try:
            self.assets[asset.id] = asset
            self.lineage_graph.add_node(asset.id, **{
                "name": asset.name,
                "type": asset.type,
                "source_system": asset.source_system,
                "classification": asset.classification.value,
                "owner": asset.owner,
                "created_at": asset.created_at.isoformat()
            })
            
            logger.info(f"Registered data asset: {asset.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register asset {asset.name}: {str(e)}")
            return False
    
    def register_transformation(self, transformation: DataTransformation) -> bool:
        """Register a data transformation for lineage tracking"""
        try:
            self.transformations[transformation.id] = transformation
            logger.info(f"Registered transformation: {transformation.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register transformation {transformation.name}: {str(e)}")
            return False
    
    def register_compliance_rule(self, rule: ComplianceRule) -> bool:
        """Register a compliance rule"""
        try:
            self.compliance_rules[rule.id] = rule
            logger.info(f"Registered compliance rule: {rule.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register compliance rule {rule.name}: {str(e)}")
            return False
    
    def track_lineage_event(self, event_type: LineageEventType,
                          source_assets: List[str],
                          target_assets: List[str],
                          transformation_details: Dict[str, Any],
                          user_id: str,
                          system_id: str,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """Track a lineage event"""
        try:
            event_id = str(uuid.uuid4())
            
            event = LineageEvent(
                id=event_id,
                event_type=event_type,
                timestamp=datetime.utcnow(),
                source_assets=source_assets,
                target_assets=target_assets,
                transformation_details=transformation_details,
                user_id=user_id,
                system_id=system_id,
                metadata=metadata or {}
            )
            
            self.events.append(event)
            
            # Update lineage graph
            self._update_lineage_graph(event)
            
            logger.info(f"Tracked lineage event: {event_type.value}")
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to track lineage event: {str(e)}")
            return ""
    
    def _update_lineage_graph(self, event: LineageEvent) -> None:
        """Update the lineage graph with a new event"""
        try:
            # Add edges from source to target assets
            for source_id in event.source_assets:
                for target_id in event.target_assets:
                    self.lineage_graph.add_edge(source_id, target_id, **{
                        "event_id": event.id,
                        "event_type": event.event_type.value,
                        "timestamp": event.timestamp.isoformat(),
                        "transformation": event.transformation_details,
                        "user_id": event.user_id,
                        "system_id": event.system_id
                    })
                    
        except Exception as e:
            logger.error(f"Failed to update lineage graph: {str(e)}")
    
    def get_upstream_lineage(self, asset_id: str, max_depth: int = 10) -> Dict[str, Any]:
        """Get upstream lineage for a data asset"""
        try:
            if asset_id not in self.lineage_graph:
                return {"error": f"Asset {asset_id} not found in lineage graph"}
            
            # Find all upstream nodes
            upstream_nodes = set()
            queue = deque([(asset_id, 0)])
            visited = {asset_id}
            
            while queue:
                current_node, depth = queue.popleft()
                
                if depth >= max_depth:
                    continue
                
                # Get predecessors (upstream nodes)
                predecessors = list(self.lineage_graph.predecessors(current_node))
                
                for pred in predecessors:
                    if pred not in visited:
                        visited.add(pred)
                        upstream_nodes.add(pred)
                        queue.append((pred, depth + 1))
            
            # Build lineage information
            lineage_info = {
                "asset_id": asset_id,
                "upstream_assets": [],
                "lineage_paths": [],
                "total_upstream_count": len(upstream_nodes)
            }
            
            # Get asset details for upstream nodes
            for node_id in upstream_nodes:
                asset = self.assets.get(node_id)
                if asset:
                    lineage_info["upstream_assets"].append({
                        "id": asset.id,
                        "name": asset.name,
                        "type": asset.type,
                        "source_system": asset.source_system,
                        "classification": asset.classification.value
                    })
            
            # Get lineage paths
            lineage_info["lineage_paths"] = self._get_lineage_paths(
                upstream_nodes, asset_id, "upstream"
            )
            
            return lineage_info
            
        except Exception as e:
            logger.error(f"Failed to get upstream lineage: {str(e)}")
            return {"error": str(e)}
    
    def get_downstream_lineage(self, asset_id: str, max_depth: int = 10) -> Dict[str, Any]:
        """Get downstream lineage for a data asset"""
        try:
            if asset_id not in self.lineage_graph:
                return {"error": f"Asset {asset_id} not found in lineage graph"}
            
            # Find all downstream nodes
            downstream_nodes = set()
            queue = deque([(asset_id, 0)])
            visited = {asset_id}
            
            while queue:
                current_node, depth = queue.popleft()
                
                if depth >= max_depth:
                    continue
                
                # Get successors (downstream nodes)
                successors = list(self.lineage_graph.successors(current_node))
                
                for succ in successors:
                    if succ not in visited:
                        visited.add(succ)
                        downstream_nodes.add(succ)
                        queue.append((succ, depth + 1))
            
            # Build lineage information
            lineage_info = {
                "asset_id": asset_id,
                "downstream_assets": [],
                "lineage_paths": [],
                "total_downstream_count": len(downstream_nodes)
            }
            
            # Get asset details for downstream nodes
            for node_id in downstream_nodes:
                asset = self.assets.get(node_id)
                if asset:
                    lineage_info["downstream_assets"].append({
                        "id": asset.id,
                        "name": asset.name,
                        "type": asset.type,
                        "source_system": asset.source_system,
                        "classification": asset.classification.value
                    })
            
            # Get lineage paths
            lineage_info["lineage_paths"] = self._get_lineage_paths(
                downstream_nodes, asset_id, "downstream"
            )
            
            return lineage_info
            
        except Exception as e:
            logger.error(f"Failed to get downstream lineage: {str(e)}")
            return {"error": str(e)}
    
    def _get_lineage_paths(self, target_nodes: Set[str], 
                          source_node: str, direction: str) -> List[Dict[str, Any]]:
        """Get detailed lineage paths between nodes"""
        paths = []
        
        try:
            for target_node in target_nodes:
                try:
                    if direction == "upstream":
                        # Find path from target to source (reverse for upstream)
                        if nx.has_path(self.lineage_graph, target_node, source_node):
                            path = nx.shortest_path(self.lineage_graph, target_node, source_node)
                            path.reverse()  # Reverse to show upstream flow
                        else:
                            continue
                    else:
                        # Find path from source to target (downstream)
                        if nx.has_path(self.lineage_graph, source_node, target_node):
                            path = nx.shortest_path(self.lineage_graph, source_node, target_node)
                        else:
                            continue
                    
                    # Build path details
                    path_info = {
                        "path_nodes": path,
                        "path_length": len(path) - 1,
                        "transformations": []
                    }
                    
                    # Get transformation details for each edge in path
                    for i in range(len(path) - 1):
                        edge_data = self.lineage_graph.get_edge_data(path[i], path[i + 1])
                        if edge_data:
                            path_info["transformations"].append({
                                "from_asset": path[i],
                                "to_asset": path[i + 1],
                                "event_type": edge_data.get("event_type"),
                                "timestamp": edge_data.get("timestamp"),
                                "transformation_details": edge_data.get("transformation", {})
                            })
                    
                    paths.append(path_info)
                    
                except nx.NetworkXNoPath:
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to get lineage paths: {str(e)}")
        
        return paths
    
    def get_impact_analysis(self, asset_id: str) -> Dict[str, Any]:
        """Analyze the impact of changes to a data asset"""
        try:
            downstream_info = self.get_downstream_lineage(asset_id)
            
            if "error" in downstream_info:
                return downstream_info
            
            # Analyze impact by classification and system
            impact_analysis = {
                "asset_id": asset_id,
                "total_impacted_assets": downstream_info["total_downstream_count"],
                "impact_by_classification": defaultdict(int),
                "impact_by_system": defaultdict(int),
                "critical_paths": [],
                "compliance_considerations": []
            }
            
            # Analyze downstream assets
            for asset_info in downstream_info["downstream_assets"]:
                classification = asset_info["classification"]
                system = asset_info["source_system"]
                
                impact_analysis["impact_by_classification"][classification] += 1
                impact_analysis["impact_by_system"][system] += 1
                
                # Check for critical assets (restricted or confidential)
                if classification in ["restricted", "confidential"]:
                    impact_analysis["critical_paths"].append({
                        "asset_id": asset_info["id"],
                        "asset_name": asset_info["name"],
                        "classification": classification,
                        "system": system
                    })
            
            # Check compliance considerations
            source_asset = self.assets.get(asset_id)
            if source_asset:
                for rule_id, rule in self.compliance_rules.items():
                    if source_asset.classification in rule.data_classifications:
                        impact_analysis["compliance_considerations"].append({
                            "rule_name": rule.name,
                            "regulation": rule.regulation,
                            "requirements": rule.audit_requirements
                        })
            
            # Convert defaultdicts to regular dicts
            impact_analysis["impact_by_classification"] = dict(impact_analysis["impact_by_classification"])
            impact_analysis["impact_by_system"] = dict(impact_analysis["impact_by_system"])
            
            return impact_analysis
            
        except Exception as e:
            logger.error(f"Failed to perform impact analysis: {str(e)}")
            return {"error": str(e)}
    
    def generate_audit_report(self, time_range: Optional[Tuple[datetime, datetime]] = None,
                            asset_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        try:
            # Default to last 30 days if no time range specified
            if not time_range:
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(days=30)
                time_range = (start_time, end_time)
            
            start_time, end_time = time_range
            
            # Filter events by time range and assets
            relevant_events = []
            for event in self.events:
                if start_time <= event.timestamp <= end_time:
                    if not asset_ids or any(asset_id in event.source_assets + event.target_assets 
                                          for asset_id in asset_ids):
                        relevant_events.append(event)
            
            # Generate audit report
            audit_report = {
                "report_generated": datetime.utcnow().isoformat(),
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "summary": {
                    "total_events": len(relevant_events),
                    "unique_assets_involved": len(set(
                        asset_id for event in relevant_events 
                        for asset_id in event.source_assets + event.target_assets
                    )),
                    "unique_users": len(set(event.user_id for event in relevant_events)),
                    "unique_systems": len(set(event.system_id for event in relevant_events))
                },
                "event_breakdown": {},
                "user_activity": {},
                "system_activity": {},
                "compliance_events": [],
                "data_movements": []
            }
            
            # Event breakdown by type
            event_counts = defaultdict(int)
            for event in relevant_events:
                event_counts[event.event_type.value] += 1
            audit_report["event_breakdown"] = dict(event_counts)
            
            # User activity analysis
            user_activity = defaultdict(lambda: {"event_count": 0, "event_types": set()})
            for event in relevant_events:
                user_activity[event.user_id]["event_count"] += 1
                user_activity[event.user_id]["event_types"].add(event.event_type.value)
            
            # Convert sets to lists for JSON serialization
            for user_id, activity in user_activity.items():
                audit_report["user_activity"][user_id] = {
                    "event_count": activity["event_count"],
                    "event_types": list(activity["event_types"])
                }
            
            # System activity analysis
            system_activity = defaultdict(lambda: {"event_count": 0, "event_types": set()})
            for event in relevant_events:
                system_activity[event.system_id]["event_count"] += 1
                system_activity[event.system_id]["event_types"].add(event.event_type.value)
            
            for system_id, activity in system_activity.items():
                audit_report["system_activity"][system_id] = {
                    "event_count": activity["event_count"],
                    "event_types": list(activity["event_types"])
                }
            
            # Compliance-relevant events
            for event in relevant_events:
                for asset_id in event.source_assets + event.target_assets:
                    asset = self.assets.get(asset_id)
                    if asset and asset.classification in [DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED]:
                        audit_report["compliance_events"].append({
                            "event_id": event.id,
                            "event_type": event.event_type.value,
                            "timestamp": event.timestamp.isoformat(),
                            "asset_id": asset_id,
                            "asset_name": asset.name,
                            "classification": asset.classification.value,
                            "user_id": event.user_id,
                            "system_id": event.system_id
                        })
            
            # Data movement tracking
            for event in relevant_events:
                if event.event_type in [LineageEventType.DATA_INGESTION, LineageEventType.DATA_EXPORT]:
                    audit_report["data_movements"].append({
                        "event_id": event.id,
                        "event_type": event.event_type.value,
                        "timestamp": event.timestamp.isoformat(),
                        "source_assets": event.source_assets,
                        "target_assets": event.target_assets,
                        "user_id": event.user_id,
                        "system_id": event.system_id
                    })
            
            return audit_report
            
        except Exception as e:
            logger.error(f"Failed to generate audit report: {str(e)}")
            return {"error": str(e)}
    
    def check_compliance_violations(self) -> List[Dict[str, Any]]:
        """Check for potential compliance violations"""
        violations = []
        
        try:
            current_time = datetime.utcnow()
            
            for asset_id, asset in self.assets.items():
                # Check retention period violations
                for rule_id, rule in self.compliance_rules.items():
                    if asset.classification in rule.data_classifications:
                        if rule.retention_period_days:
                            retention_cutoff = current_time - timedelta(days=rule.retention_period_days)
                            
                            if asset.created_at < retention_cutoff:
                                violations.append({
                                    "type": "retention_violation",
                                    "asset_id": asset_id,
                                    "asset_name": asset.name,
                                    "rule_name": rule.name,
                                    "regulation": rule.regulation,
                                    "violation_details": f"Asset exceeds retention period of {rule.retention_period_days} days",
                                    "created_at": asset.created_at.isoformat(),
                                    "retention_cutoff": retention_cutoff.isoformat()
                                })
                
                # Check access pattern violations
                recent_events = [
                    event for event in self.events
                    if asset_id in event.source_assets + event.target_assets
                    and event.timestamp >= current_time - timedelta(days=7)
                ]
                
                # Check for unusual access patterns
                unique_users = set(event.user_id for event in recent_events)
                if len(unique_users) > 10 and asset.classification == DataClassification.RESTRICTED:
                    violations.append({
                        "type": "access_pattern_violation",
                        "asset_id": asset_id,
                        "asset_name": asset.name,
                        "violation_details": f"Restricted asset accessed by {len(unique_users)} users in past 7 days",
                        "unique_users_count": len(unique_users),
                        "classification": asset.classification.value
                    })
            
            return violations
            
        except Exception as e:
            logger.error(f"Failed to check compliance violations: {str(e)}")
            return [{"error": str(e)}]
    
    def export_lineage_graph(self, format: str = "json") -> str:
        """Export the lineage graph in specified format"""
        try:
            if format.lower() == "json":
                # Convert NetworkX graph to JSON-serializable format
                graph_data = {
                    "nodes": [],
                    "edges": [],
                    "metadata": {
                        "total_nodes": self.lineage_graph.number_of_nodes(),
                        "total_edges": self.lineage_graph.number_of_edges(),
                        "exported_at": datetime.utcnow().isoformat()
                    }
                }
                
                # Add nodes
                for node_id in self.lineage_graph.nodes():
                    node_data = self.lineage_graph.nodes[node_id].copy()
                    node_data["id"] = node_id
                    graph_data["nodes"].append(node_data)
                
                # Add edges
                for source, target in self.lineage_graph.edges():
                    edge_data = self.lineage_graph.get_edge_data(source, target).copy()
                    edge_data["source"] = source
                    edge_data["target"] = target
                    graph_data["edges"].append(edge_data)
                
                return json.dumps(graph_data, indent=2)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Failed to export lineage graph: {str(e)}")
            return f"Export failed: {str(e)}"
    
    def get_lineage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive lineage statistics"""
        try:
            current_time = datetime.utcnow()
            
            stats = {
                "assets": {
                    "total_count": len(self.assets),
                    "by_type": defaultdict(int),
                    "by_classification": defaultdict(int),
                    "by_system": defaultdict(int)
                },
                "events": {
                    "total_count": len(self.events),
                    "by_type": defaultdict(int),
                    "recent_activity": {
                        "last_24h": 0,
                        "last_7d": 0,
                        "last_30d": 0
                    }
                },
                "lineage_graph": {
                    "nodes": self.lineage_graph.number_of_nodes(),
                    "edges": self.lineage_graph.number_of_edges(),
                    "connected_components": nx.number_connected_components(self.lineage_graph.to_undirected()),
                    "average_degree": sum(dict(self.lineage_graph.degree()).values()) / self.lineage_graph.number_of_nodes() if self.lineage_graph.number_of_nodes() > 0 else 0
                },
                "compliance": {
                    "total_rules": len(self.compliance_rules),
                    "violations_found": len(self.check_compliance_violations())
                },
                "generated_at": current_time.isoformat()
            }
            
            # Asset statistics
            for asset in self.assets.values():
                stats["assets"]["by_type"][asset.type] += 1
                stats["assets"]["by_classification"][asset.classification.value] += 1
                stats["assets"]["by_system"][asset.source_system] += 1
            
            # Event statistics
            cutoff_24h = current_time - timedelta(hours=24)
            cutoff_7d = current_time - timedelta(days=7)
            cutoff_30d = current_time - timedelta(days=30)
            
            for event in self.events:
                stats["events"]["by_type"][event.event_type.value] += 1
                
                if event.timestamp >= cutoff_24h:
                    stats["events"]["recent_activity"]["last_24h"] += 1
                if event.timestamp >= cutoff_7d:
                    stats["events"]["recent_activity"]["last_7d"] += 1
                if event.timestamp >= cutoff_30d:
                    stats["events"]["recent_activity"]["last_30d"] += 1
            
            # Convert defaultdicts to regular dicts
            stats["assets"]["by_type"] = dict(stats["assets"]["by_type"])
            stats["assets"]["by_classification"] = dict(stats["assets"]["by_classification"])
            stats["assets"]["by_system"] = dict(stats["assets"]["by_system"])
            stats["events"]["by_type"] = dict(stats["events"]["by_type"])
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get lineage statistics: {str(e)}")
            return {"error": str(e)}