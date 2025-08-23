"""
Data Lineage Tracking Engine

This module provides comprehensive data lineage tracking throughout pipelines,
enabling complete visibility into data transformations and dependencies.
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
import networkx as nx
import json
import logging

from scrollintel.models.lineage_models import (
    DataLineage, LineageEventType, LineageEventRequest,
    LineageQueryRequest, AuditTrail, AuditTrailRequest
)
from scrollintel.models.database_utils import get_sync_db

logger = logging.getLogger(__name__)


class LineageTracker:
    """Comprehensive data lineage tracking system"""
    
    def __init__(self):
        self.session = next(get_sync_db())
        self.lineage_graph = nx.DiGraph()
        self._load_existing_lineage()
    
    def track_lineage_event(
        self,
        request: LineageEventRequest,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """
        Track a data lineage event
        
        Args:
            request: Lineage event details
            user_id: User performing the operation
            session_id: Session identifier
            
        Returns:
            str: Lineage event ID
        """
        try:
            # Create lineage record
            lineage_id = str(uuid.uuid4())
            lineage = DataLineage(
                id=lineage_id,
                pipeline_id=request.pipeline_id,
                source_dataset_id=request.source_dataset_id,
                target_dataset_id=request.target_dataset_id,
                transformation_id=request.transformation_id,
                event_type=request.event_type.value,
                source_schema=request.source_schema,
                target_schema=request.target_schema,
                transformation_details=request.transformation_details,
                data_volume=request.data_volume,
                processing_duration=request.processing_duration,
                user_id=user_id,
                session_id=session_id,
                event_metadata=request.event_metadata
            )
            
            self.session.add(lineage)
            self.session.commit()
            
            # Update lineage graph
            self._update_lineage_graph(lineage)
            
            # Create audit trail
            self._create_audit_trail(
                entity_type="lineage",
                entity_id=lineage_id,
                action="create",
                user_id=user_id or "system",
                new_values=request.model_dump(),
                session_id=session_id
            )
            
            logger.info(f"Tracked lineage event: {lineage_id}")
            return lineage_id
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error tracking lineage event: {str(e)}")
            raise
    
    def get_lineage_history(
        self,
        query: LineageQueryRequest
    ) -> List[Dict[str, Any]]:
        """
        Get lineage history based on query parameters
        
        Args:
            query: Lineage query parameters
            
        Returns:
            List[Dict]: Lineage events
        """
        try:
            # Build query
            db_query = self.session.query(DataLineage)
            
            if query.dataset_id:
                db_query = db_query.filter(
                    or_(
                        DataLineage.source_dataset_id == query.dataset_id,
                        DataLineage.target_dataset_id == query.dataset_id
                    )
                )
            
            if query.pipeline_id:
                db_query = db_query.filter(DataLineage.pipeline_id == query.pipeline_id)
            
            if query.start_date:
                db_query = db_query.filter(DataLineage.event_timestamp >= query.start_date)
            
            if query.end_date:
                db_query = db_query.filter(DataLineage.event_timestamp <= query.end_date)
            
            if query.event_types:
                event_type_values = [et.value for et in query.event_types]
                db_query = db_query.filter(DataLineage.event_type.in_(event_type_values))
            
            # Execute query
            lineage_events = db_query.order_by(desc(DataLineage.event_timestamp)).all()
            
            # Convert to response format
            result = []
            for event in lineage_events:
                event_data = {
                    "id": event.id,
                    "pipeline_id": event.pipeline_id,
                    "source_dataset_id": event.source_dataset_id,
                    "target_dataset_id": event.target_dataset_id,
                    "transformation_id": event.transformation_id,
                    "event_type": event.event_type,
                    "event_timestamp": event.event_timestamp.isoformat(),
                    "data_volume": event.data_volume,
                    "processing_duration": event.processing_duration,
                    "user_id": event.user_id,
                    "event_metadata": event.event_metadata
                }
                
                if query.include_transformations:
                    event_data.update({
                        "source_schema": event.source_schema,
                        "target_schema": event.target_schema,
                        "transformation_details": event.transformation_details
                    })
                
                result.append(event_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting lineage history: {str(e)}")
            raise
    
    def get_data_lineage_graph(
        self,
        dataset_id: str,
        depth: int = 5,
        direction: str = "both"
    ) -> Dict[str, Any]:
        """
        Get data lineage graph for a dataset
        
        Args:
            dataset_id: Dataset to trace lineage for
            depth: Maximum depth to traverse
            direction: Direction to traverse (upstream, downstream, both)
            
        Returns:
            Dict: Lineage graph data
        """
        try:
            nodes = set()
            edges = []
            
            if direction in ["upstream", "both"]:
                # Get upstream lineage
                upstream_nodes, upstream_edges = self._get_upstream_lineage(
                    dataset_id, depth
                )
                nodes.update(upstream_nodes)
                edges.extend(upstream_edges)
            
            if direction in ["downstream", "both"]:
                # Get downstream lineage
                downstream_nodes, downstream_edges = self._get_downstream_lineage(
                    dataset_id, depth
                )
                nodes.update(downstream_nodes)
                edges.extend(downstream_edges)
            
            # Add the root dataset
            nodes.add(dataset_id)
            
            # Get node details
            node_details = self._get_node_details(list(nodes))
            
            return {
                "nodes": node_details,
                "edges": edges,
                "root_dataset": dataset_id,
                "depth": depth,
                "direction": direction
            }
            
        except Exception as e:
            logger.error(f"Error getting lineage graph: {str(e)}")
            raise
    
    def get_impact_analysis(
        self,
        dataset_id: str,
        change_type: str = "schema_change"
    ) -> Dict[str, Any]:
        """
        Analyze impact of changes to a dataset
        
        Args:
            dataset_id: Dataset being changed
            change_type: Type of change being made
            
        Returns:
            Dict: Impact analysis results
        """
        try:
            # Get downstream dependencies
            downstream_nodes, downstream_edges = self._get_downstream_lineage(
                dataset_id, depth=10
            )
            
            # Analyze impact
            impact_summary = {
                "affected_datasets": len(downstream_nodes),
                "affected_pipelines": set(),
                "critical_dependencies": [],
                "recommendations": []
            }
            
            # Get pipeline information
            for edge in downstream_edges:
                pipeline_id = edge.get("pipeline_id")
                if pipeline_id:
                    impact_summary["affected_pipelines"].add(pipeline_id)
            
            impact_summary["affected_pipelines"] = list(impact_summary["affected_pipelines"])
            
            # Identify critical dependencies
            for node in downstream_nodes:
                node_info = self._get_dataset_info(node)
                if node_info.get("is_critical", False):
                    impact_summary["critical_dependencies"].append(node)
            
            # Generate recommendations
            if impact_summary["affected_datasets"] > 10:
                impact_summary["recommendations"].append(
                    "High impact change - consider phased rollout"
                )
            
            if impact_summary["critical_dependencies"]:
                impact_summary["recommendations"].append(
                    "Critical systems affected - require additional testing"
                )
            
            return impact_summary
            
        except Exception as e:
            logger.error(f"Error performing impact analysis: {str(e)}")
            raise
    
    def _load_existing_lineage(self):
        """Load existing lineage data into graph"""
        try:
            # Load recent lineage events (last 30 days)
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
            lineage_events = self.session.query(DataLineage).filter(
                DataLineage.event_timestamp >= cutoff_date
            ).all()
            
            for event in lineage_events:
                self._update_lineage_graph(event)
                
        except Exception as e:
            logger.error(f"Error loading existing lineage: {str(e)}")
    
    def _update_lineage_graph(self, lineage: DataLineage):
        """Update the in-memory lineage graph"""
        try:
            # Add nodes
            self.lineage_graph.add_node(
                lineage.source_dataset_id,
                dataset_id=lineage.source_dataset_id,
                last_updated=lineage.event_timestamp
            )
            self.lineage_graph.add_node(
                lineage.target_dataset_id,
                dataset_id=lineage.target_dataset_id,
                last_updated=lineage.event_timestamp
            )
            
            # Add edge
            self.lineage_graph.add_edge(
                lineage.source_dataset_id,
                lineage.target_dataset_id,
                lineage_id=lineage.id,
                pipeline_id=lineage.pipeline_id,
                transformation_id=lineage.transformation_id,
                event_type=lineage.event_type,
                timestamp=lineage.event_timestamp
            )
            
        except Exception as e:
            logger.error(f"Error updating lineage graph: {str(e)}")
    
    def _get_upstream_lineage(
        self,
        dataset_id: str,
        depth: int,
        visited: Optional[set] = None
    ) -> Tuple[set, List[Dict]]:
        """Get upstream lineage for a dataset"""
        if visited is None:
            visited = set()
        
        if depth <= 0 or dataset_id in visited:
            return set(), []
        
        visited.add(dataset_id)
        nodes = set()
        edges = []
        
        # Get direct upstream dependencies
        upstream_events = self.session.query(DataLineage).filter(
            DataLineage.target_dataset_id == dataset_id
        ).all()
        
        for event in upstream_events:
            source_id = event.source_dataset_id
            nodes.add(source_id)
            
            edges.append({
                "source": source_id,
                "target": dataset_id,
                "pipeline_id": event.pipeline_id,
                "transformation_id": event.transformation_id,
                "event_type": event.event_type,
                "timestamp": event.event_timestamp.isoformat()
            })
            
            # Recursively get upstream dependencies
            upstream_nodes, upstream_edges = self._get_upstream_lineage(
                source_id, depth - 1, visited.copy()
            )
            nodes.update(upstream_nodes)
            edges.extend(upstream_edges)
        
        return nodes, edges
    
    def _get_downstream_lineage(
        self,
        dataset_id: str,
        depth: int,
        visited: Optional[set] = None
    ) -> Tuple[set, List[Dict]]:
        """Get downstream lineage for a dataset"""
        if visited is None:
            visited = set()
        
        if depth <= 0 or dataset_id in visited:
            return set(), []
        
        visited.add(dataset_id)
        nodes = set()
        edges = []
        
        # Get direct downstream dependencies
        downstream_events = self.session.query(DataLineage).filter(
            DataLineage.source_dataset_id == dataset_id
        ).all()
        
        for event in downstream_events:
            target_id = event.target_dataset_id
            nodes.add(target_id)
            
            edges.append({
                "source": dataset_id,
                "target": target_id,
                "pipeline_id": event.pipeline_id,
                "transformation_id": event.transformation_id,
                "event_type": event.event_type,
                "timestamp": event.event_timestamp.isoformat()
            })
            
            # Recursively get downstream dependencies
            downstream_nodes, downstream_edges = self._get_downstream_lineage(
                target_id, depth - 1, visited.copy()
            )
            nodes.update(downstream_nodes)
            edges.extend(downstream_edges)
        
        return nodes, edges
    
    def _get_node_details(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """Get detailed information for nodes"""
        node_details = []
        
        for node_id in node_ids:
            # Get basic node info
            node_info = {
                "id": node_id,
                "type": "dataset",
                "name": node_id,
                "metadata": {}
            }
            
            # Get additional details from recent lineage events
            recent_events = self.session.query(DataLineage).filter(
                or_(
                    DataLineage.source_dataset_id == node_id,
                    DataLineage.target_dataset_id == node_id
                )
            ).order_by(desc(DataLineage.event_timestamp)).limit(1).all()
            
            if recent_events:
                event = recent_events[0]
                node_info["metadata"] = {
                    "last_updated": event.event_timestamp.isoformat(),
                    "last_pipeline": event.pipeline_id,
                    "data_volume": event.data_volume
                }
            
            node_details.append(node_info)
        
        return node_details
    
    def _get_dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        """Get dataset information"""
        # This would typically query a dataset registry
        # For now, return basic info
        return {
            "id": dataset_id,
            "name": dataset_id,
            "is_critical": False  # Would be determined by business rules
        }
    
    def _create_audit_trail(
        self,
        entity_type: str,
        entity_id: str,
        action: str,
        user_id: str,
        old_values: Optional[Dict] = None,
        new_values: Optional[Dict] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Create audit trail entry"""
        try:
            audit_id = str(uuid.uuid4())
            audit = AuditTrail(
                id=audit_id,
                entity_type=entity_type,
                entity_id=entity_id,
                action=action,
                user_id=user_id,
                session_id=session_id,
                old_values=old_values,
                new_values=new_values,
                audit_metadata=metadata
            )
            
            self.session.add(audit)
            self.session.commit()
            
        except Exception as e:
            logger.error(f"Error creating audit trail: {str(e)}")
            # Don't raise - audit trail failures shouldn't break main functionality
    
    def get_audit_trail(
        self,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get audit trail entries"""
        try:
            query = self.session.query(AuditTrail)
            
            if entity_type:
                query = query.filter(AuditTrail.entity_type == entity_type)
            
            if entity_id:
                query = query.filter(AuditTrail.entity_id == entity_id)
            
            if user_id:
                query = query.filter(AuditTrail.user_id == user_id)
            
            if start_date:
                query = query.filter(AuditTrail.timestamp >= start_date)
            
            if end_date:
                query = query.filter(AuditTrail.timestamp <= end_date)
            
            audit_entries = query.order_by(desc(AuditTrail.timestamp)).limit(limit).all()
            
            return [
                {
                    "id": entry.id,
                    "entity_type": entry.entity_type,
                    "entity_id": entry.entity_id,
                    "action": entry.action,
                    "user_id": entry.user_id,
                    "session_id": entry.session_id,
                    "old_values": entry.old_values,
                    "new_values": entry.new_values,
                    "change_summary": entry.change_summary,
                    "timestamp": entry.timestamp.isoformat(),
                    "audit_metadata": entry.audit_metadata
                }
                for entry in audit_entries
            ]
            
        except Exception as e:
            logger.error(f"Error getting audit trail: {str(e)}")
            raise