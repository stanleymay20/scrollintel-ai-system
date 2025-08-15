"""
Lineage Visualization and Reporting Engine for AI Data Readiness Platform

This module provides lineage graph generation, visualization, query capabilities,
and impact analysis for data lineage tracking.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

from .lineage_engine import LineageEngine, LineageNode, LineageEdge, TransformationType, LineageNodeType

logger = logging.getLogger(__name__)


class VisualizationType(Enum):
    """Types of lineage visualizations"""
    GRAPH = "graph"
    TREE = "tree"
    TIMELINE = "timeline"
    IMPACT_MAP = "impact_map"
    DEPENDENCY_MATRIX = "dependency_matrix"


class QueryType(Enum):
    """Types of lineage queries"""
    UPSTREAM = "upstream"
    DOWNSTREAM = "downstream"
    FULL_LINEAGE = "full_lineage"
    IMPACT_ANALYSIS = "impact_analysis"
    SHORTEST_PATH = "shortest_path"
    COMMON_ANCESTORS = "common_ancestors"


@dataclass
class LineageQuery:
    """Represents a lineage query with parameters and results"""
    query_id: str
    query_type: QueryType
    target_nodes: List[str]
    parameters: Dict[str, Any]
    results: Optional[Dict[str, Any]] = None
    execution_time_ms: Optional[int] = None
    created_at: Optional[datetime] = None


@dataclass
class ImpactAnalysisResult:
    """Results of impact analysis for a node or change"""
    target_node_id: str
    affected_nodes: List[str]
    affected_models: List[str]
    impact_score: float
    risk_level: str
    recommendations: List[str]
    analysis_details: Dict[str, Any]


class LineageVisualizer:
    """
    Engine for lineage visualization, querying, and impact analysis
    """
    
    def __init__(self, lineage_engine: LineageEngine):
        """
        Initialize the LineageVisualizer
        
        Args:
            lineage_engine: Instance of LineageEngine for data access
        """
        self.lineage_engine = lineage_engine
        self.query_cache: Dict[str, LineageQuery] = {}
        self.visualization_cache: Dict[str, Any] = {}
        
    def generate_lineage_graph(
        self,
        target_node_id: str,
        include_upstream: bool = True,
        include_downstream: bool = True,
        max_depth: Optional[int] = None
    ) -> nx.DiGraph:
        """
        Generate a NetworkX graph for lineage visualization
        
        Args:
            target_node_id: ID of the target node
            include_upstream: Whether to include upstream nodes
            include_downstream: Whether to include downstream nodes
            max_depth: Maximum depth to traverse (None for unlimited)
            
        Returns:
            NetworkX directed graph representing the lineage
        """
        try:
            graph = nx.DiGraph()
            
            # Get target node
            target_node = self.lineage_engine.lineage_graph.get(target_node_id)
            if not target_node:
                raise ValueError(f"Node {target_node_id} not found")
            
            # Add target node
            graph.add_node(
                target_node_id,
                **self._node_to_graph_attributes(target_node)
            )
            
            visited = set()
            
            # Add upstream nodes and edges
            if include_upstream:
                self._add_upstream_to_graph(
                    graph, target_node_id, visited, max_depth, 0
                )
            
            # Reset visited for downstream traversal
            visited = set()
            
            # Add downstream nodes and edges
            if include_downstream:
                self._add_downstream_to_graph(
                    graph, target_node_id, visited, max_depth, 0
                )
            
            logger.info(f"Generated lineage graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            return graph
            
        except Exception as e:
            logger.error(f"Error generating lineage graph for {target_node_id}: {str(e)}")
            raise
    
    def create_interactive_visualization(
        self,
        target_node_id: str,
        visualization_type: VisualizationType = VisualizationType.GRAPH,
        **kwargs
    ) -> go.Figure:
        """
        Create an interactive Plotly visualization of the lineage
        
        Args:
            target_node_id: ID of the target node
            visualization_type: Type of visualization to create
            **kwargs: Additional parameters for visualization
            
        Returns:
            Plotly figure object
        """
        try:
            if visualization_type == VisualizationType.GRAPH:
                return self._create_graph_visualization(target_node_id, **kwargs)
            elif visualization_type == VisualizationType.TREE:
                return self._create_tree_visualization(target_node_id, **kwargs)
            elif visualization_type == VisualizationType.TIMELINE:
                return self._create_timeline_visualization(target_node_id, **kwargs)
            elif visualization_type == VisualizationType.IMPACT_MAP:
                return self._create_impact_map_visualization(target_node_id, **kwargs)
            elif visualization_type == VisualizationType.DEPENDENCY_MATRIX:
                return self._create_dependency_matrix_visualization(target_node_id, **kwargs)
            else:
                raise ValueError(f"Unsupported visualization type: {visualization_type}")
                
        except Exception as e:
            logger.error(f"Error creating visualization for {target_node_id}: {str(e)}")
            raise
    
    def query_lineage(
        self,
        query_type: QueryType,
        target_nodes: List[str],
        parameters: Optional[Dict[str, Any]] = None
    ) -> LineageQuery:
        """
        Execute a lineage query
        
        Args:
            query_type: Type of query to execute
            target_nodes: List of target node IDs
            parameters: Additional query parameters
            
        Returns:
            LineageQuery object with results
        """
        try:
            start_time = datetime.now()
            query_id = f"{query_type.value}_{hash(tuple(target_nodes))}_{int(start_time.timestamp())}"
            
            parameters = parameters or {}
            
            query = LineageQuery(
                query_id=query_id,
                query_type=query_type,
                target_nodes=target_nodes,
                parameters=parameters,
                created_at=start_time
            )
            
            # Execute query based on type
            if query_type == QueryType.UPSTREAM:
                results = self._query_upstream(target_nodes, parameters)
            elif query_type == QueryType.DOWNSTREAM:
                results = self._query_downstream(target_nodes, parameters)
            elif query_type == QueryType.FULL_LINEAGE:
                results = self._query_full_lineage(target_nodes, parameters)
            elif query_type == QueryType.IMPACT_ANALYSIS:
                results = self._query_impact_analysis(target_nodes, parameters)
            elif query_type == QueryType.SHORTEST_PATH:
                results = self._query_shortest_path(target_nodes, parameters)
            elif query_type == QueryType.COMMON_ANCESTORS:
                results = self._query_common_ancestors(target_nodes, parameters)
            else:
                raise ValueError(f"Unsupported query type: {query_type}")
            
            # Calculate execution time
            end_time = datetime.now()
            execution_time = int((end_time - start_time).total_seconds() * 1000)
            
            query.results = results
            query.execution_time_ms = execution_time
            
            # Cache query
            self.query_cache[query_id] = query
            
            logger.info(f"Executed lineage query {query_id} in {execution_time}ms")
            return query
            
        except Exception as e:
            logger.error(f"Error executing lineage query: {str(e)}")
            raise
    
    def analyze_impact(
        self,
        target_node_id: str,
        change_type: str = "modification",
        change_details: Optional[Dict[str, Any]] = None
    ) -> ImpactAnalysisResult:
        """
        Analyze the impact of changes to a specific node
        
        Args:
            target_node_id: ID of the node being changed
            change_type: Type of change (modification, deletion, schema_change, etc.)
            change_details: Details about the change
            
        Returns:
            ImpactAnalysisResult with analysis details
        """
        try:
            change_details = change_details or {}
            
            # Get all downstream nodes
            downstream_nodes = self.lineage_engine._get_downstream_nodes(target_node_id)
            affected_node_ids = [node.id for node in downstream_nodes]
            
            # Get affected models
            affected_models = []
            for model_id, links in self.lineage_engine.model_dataset_links.items():
                for link in links:
                    if link.dataset_id == target_node_id or link.dataset_id in affected_node_ids:
                        affected_models.append(model_id)
            
            # Calculate impact score
            impact_score = self._calculate_impact_score(
                target_node_id, affected_node_ids, affected_models, change_type
            )
            
            # Determine risk level
            risk_level = self._determine_risk_level(impact_score, len(affected_node_ids), len(affected_models))
            
            # Generate recommendations
            recommendations = self._generate_impact_recommendations(
                target_node_id, affected_node_ids, affected_models, change_type, risk_level
            )
            
            # Compile analysis details
            analysis_details = {
                "change_type": change_type,
                "change_details": change_details,
                "downstream_depth": self._calculate_max_depth(target_node_id, "downstream"),
                "upstream_depth": self._calculate_max_depth(target_node_id, "upstream"),
                "transformation_types": self._get_transformation_types_in_path(target_node_id),
                "critical_paths": self._identify_critical_paths(target_node_id, affected_models)
            }
            
            result = ImpactAnalysisResult(
                target_node_id=target_node_id,
                affected_nodes=affected_node_ids,
                affected_models=affected_models,
                impact_score=impact_score,
                risk_level=risk_level,
                recommendations=recommendations,
                analysis_details=analysis_details
            )
            
            logger.info(f"Completed impact analysis for {target_node_id}: {risk_level} risk")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing impact for {target_node_id}: {str(e)}")
            raise
    
    def generate_lineage_report(
        self,
        target_node_id: str,
        report_format: str = "html",
        include_visualizations: bool = True
    ) -> str:
        """
        Generate a comprehensive lineage report
        
        Args:
            target_node_id: ID of the target node
            report_format: Format of the report (html, json, markdown)
            include_visualizations: Whether to include visualizations
            
        Returns:
            Report content as string
        """
        try:
            # Get lineage data
            lineage = self.lineage_engine.get_dataset_lineage(target_node_id)
            
            # Perform impact analysis
            impact_analysis = self.analyze_impact(target_node_id)
            
            # Get statistics
            stats = self._get_node_statistics(target_node_id)
            
            # Generate report based on format
            if report_format == "html":
                return self._generate_html_report(lineage, impact_analysis, stats, include_visualizations)
            elif report_format == "json":
                return self._generate_json_report(lineage, impact_analysis, stats)
            elif report_format == "markdown":
                return self._generate_markdown_report(lineage, impact_analysis, stats)
            else:
                raise ValueError(f"Unsupported report format: {report_format}")
                
        except Exception as e:
            logger.error(f"Error generating lineage report for {target_node_id}: {str(e)}")
            raise
    
    def search_lineage(
        self,
        search_term: str,
        search_fields: List[str] = None,
        node_types: List[LineageNodeType] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for nodes in the lineage graph
        
        Args:
            search_term: Term to search for
            search_fields: Fields to search in (name, metadata, etc.)
            node_types: Types of nodes to include in search
            
        Returns:
            List of matching nodes with their details
        """
        try:
            search_fields = search_fields or ["name", "metadata"]
            node_types = node_types or list(LineageNodeType)
            
            results = []
            search_term_lower = search_term.lower()
            
            for node_id, node in self.lineage_engine.lineage_graph.items():
                if node.node_type not in node_types:
                    continue
                
                match_found = False
                
                # Search in specified fields
                for field in search_fields:
                    if field == "name" and search_term_lower in node.name.lower():
                        match_found = True
                        break
                    elif field == "metadata":
                        metadata_str = json.dumps(node.metadata, default=str).lower()
                        if search_term_lower in metadata_str:
                            match_found = True
                            break
                    elif field == "id" and search_term_lower in node_id.lower():
                        match_found = True
                        break
                
                if match_found:
                    # Get additional context
                    upstream_count = len(self.lineage_engine._get_upstream_nodes(node_id))
                    downstream_count = len(self.lineage_engine._get_downstream_nodes(node_id))
                    
                    result = {
                        "node": node.to_dict(),
                        "upstream_count": upstream_count,
                        "downstream_count": downstream_count,
                        "versions": len(self.lineage_engine.dataset_versions.get(node_id, [])),
                        "model_links": len([
                            link for links in self.lineage_engine.model_dataset_links.values()
                            for link in links if link.dataset_id == node_id
                        ])
                    }
                    results.append(result)
            
            logger.info(f"Found {len(results)} nodes matching '{search_term}'")
            return results
            
        except Exception as e:
            logger.error(f"Error searching lineage: {str(e)}")
            raise
    
    # Private helper methods
    
    def _node_to_graph_attributes(self, node: LineageNode) -> Dict[str, Any]:
        """Convert LineageNode to graph node attributes"""
        return {
            "name": node.name,
            "type": node.node_type.value,
            "version": node.version,
            "created_at": node.created_at.isoformat(),
            "metadata": node.metadata
        }
    
    def _add_upstream_to_graph(
        self,
        graph: nx.DiGraph,
        node_id: str,
        visited: Set[str],
        max_depth: Optional[int],
        current_depth: int
    ):
        """Recursively add upstream nodes to graph"""
        if node_id in visited or (max_depth and current_depth >= max_depth):
            return
        
        visited.add(node_id)
        
        for edge in self.lineage_engine.lineage_edges.values():
            if edge.target_node_id == node_id:
                source_node = self.lineage_engine.lineage_graph.get(edge.source_node_id)
                if source_node:
                    graph.add_node(
                        edge.source_node_id,
                        **self._node_to_graph_attributes(source_node)
                    )
                    graph.add_edge(
                        edge.source_node_id,
                        edge.target_node_id,
                        transformation_type=edge.transformation_type.value,
                        details=edge.transformation_details,
                        created_by=edge.created_by
                    )
                    
                    self._add_upstream_to_graph(
                        graph, edge.source_node_id, visited, max_depth, current_depth + 1
                    )
    
    def _add_downstream_to_graph(
        self,
        graph: nx.DiGraph,
        node_id: str,
        visited: Set[str],
        max_depth: Optional[int],
        current_depth: int
    ):
        """Recursively add downstream nodes to graph"""
        if node_id in visited or (max_depth and current_depth >= max_depth):
            return
        
        visited.add(node_id)
        
        for edge in self.lineage_engine.lineage_edges.values():
            if edge.source_node_id == node_id:
                target_node = self.lineage_engine.lineage_graph.get(edge.target_node_id)
                if target_node:
                    graph.add_node(
                        edge.target_node_id,
                        **self._node_to_graph_attributes(target_node)
                    )
                    graph.add_edge(
                        edge.source_node_id,
                        edge.target_node_id,
                        transformation_type=edge.transformation_type.value,
                        details=edge.transformation_details,
                        created_by=edge.created_by
                    )
                    
                    self._add_downstream_to_graph(
                        graph, edge.target_node_id, visited, max_depth, current_depth + 1
                    )
    
    def _create_graph_visualization(self, target_node_id: str, **kwargs) -> go.Figure:
        """Create interactive graph visualization"""
        graph = self.generate_lineage_graph(target_node_id)
        
        # Use spring layout for positioning
        pos = nx.spring_layout(graph, k=3, iterations=50)
        
        # Create edge traces
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.append(f"{edge[0]} → {edge[1]}<br>Type: {edge[2].get('transformation_type', 'Unknown')}")
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node traces
        node_x = []
        node_y = []
        node_info = []
        node_colors = []
        
        color_map = {
            'dataset': '#1f77b4',
            'transformation': '#ff7f0e',
            'model': '#2ca02c',
            'feature': '#d62728',
            'source': '#9467bd'
        }
        
        for node in graph.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)
            
            node_type = node[1].get('type', 'unknown')
            node_colors.append(color_map.get(node_type, '#888888'))
            
            info = f"<b>{node[1].get('name', node[0])}</b><br>"
            info += f"Type: {node_type}<br>"
            info += f"Version: {node[1].get('version', 'Unknown')}<br>"
            info += f"Created: {node[1].get('created_at', 'Unknown')}"
            node_info.append(info)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node[1].get('name', node[0]) for node in graph.nodes(data=True)],
            textposition="middle center",
            hovertext=node_info,
            marker=dict(
                size=20,
                color=node_colors,
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=f'Data Lineage Graph for {target_node_id}',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Hover over nodes for details",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor="left", yanchor="bottom",
                    font=dict(color="#888", size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        return fig
    
    def _create_timeline_visualization(self, target_node_id: str, **kwargs) -> go.Figure:
        """Create timeline visualization of dataset versions"""
        versions = self.lineage_engine.dataset_versions.get(target_node_id, [])
        
        if not versions:
            # Create empty timeline
            fig = go.Figure()
            fig.add_annotation(
                text="No version history available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Sort versions by creation time
        versions_sorted = sorted(versions, key=lambda v: v.created_at)
        
        # Create timeline data
        dates = [v.created_at for v in versions_sorted]
        versions_text = [v.version_number for v in versions_sorted]
        commit_messages = [v.commit_message for v in versions_sorted]
        created_by = [v.created_by for v in versions_sorted]
        
        fig = go.Figure()
        
        # Add timeline trace
        fig.add_trace(go.Scatter(
            x=dates,
            y=[1] * len(dates),
            mode='markers+text',
            text=versions_text,
            textposition="top center",
            hovertemplate="<b>Version %{text}</b><br>" +
                         "Date: %{x}<br>" +
                         "Author: %{customdata[0]}<br>" +
                         "Message: %{customdata[1]}<extra></extra>",
            customdata=list(zip(created_by, commit_messages)),
            marker=dict(size=15, color='blue')
        ))
        
        fig.update_layout(
            title=f'Version Timeline for {target_node_id}',
            xaxis_title='Date',
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            height=400
        )
        
        return fig
    
    def _create_impact_map_visualization(self, target_node_id: str, **kwargs) -> go.Figure:
        """Create impact map visualization"""
        impact_analysis = self.analyze_impact(target_node_id)
        
        # Create network graph showing impact
        graph = nx.DiGraph()
        
        # Add target node
        graph.add_node(target_node_id, type='target', impact_score=impact_analysis.impact_score)
        
        # Add affected nodes
        for node_id in impact_analysis.affected_nodes:
            node = self.lineage_engine.lineage_graph.get(node_id)
            if node:
                graph.add_node(node_id, type='affected', impact_score=0.5)
                graph.add_edge(target_node_id, node_id)
        
        # Add affected models
        for model_id in impact_analysis.affected_models:
            graph.add_node(model_id, type='model', impact_score=0.8)
            # Connect to datasets used by this model
            for link in self.lineage_engine.model_dataset_links.get(model_id, []):
                if link.dataset_id in graph.nodes:
                    graph.add_edge(link.dataset_id, model_id)
        
        # Create visualization similar to graph visualization but with impact coloring
        pos = nx.spring_layout(graph)
        
        # Create traces with impact-based coloring
        edge_x, edge_y = [], []
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'),
                               hoverinfo='none', mode='lines')
        
        node_x, node_y, node_colors, node_info = [], [], [], []
        color_map = {'target': 'red', 'affected': 'orange', 'model': 'green'}
        
        for node in graph.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)
            node_colors.append(color_map.get(node[1]['type'], 'gray'))
            
            info = f"<b>{node[0]}</b><br>Type: {node[1]['type']}<br>Impact Score: {node[1]['impact_score']:.2f}"
            node_info.append(info)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text',
            text=[node[0] for node in graph.nodes()],
            textposition="middle center",
            hovertext=node_info, hoverinfo='text',
            marker=dict(size=15, color=node_colors)
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(title=f'Impact Analysis for {target_node_id}',
                                       showlegend=False, hovermode='closest'))
        
        return fig
    
    def _create_tree_visualization(self, target_node_id: str, **kwargs) -> go.Figure:
        """Create tree visualization (placeholder)"""
        # This would create a hierarchical tree view
        # For now, return a simple message
        fig = go.Figure()
        fig.add_annotation(
            text="Tree visualization not yet implemented",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    def _create_dependency_matrix_visualization(self, target_node_id: str, **kwargs) -> go.Figure:
        """Create dependency matrix visualization (placeholder)"""
        # This would create a matrix showing dependencies
        fig = go.Figure()
        fig.add_annotation(
            text="Dependency matrix visualization not yet implemented",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Query methods
    
    def _query_upstream(self, target_nodes: List[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute upstream query"""
        results = {}
        for node_id in target_nodes:
            upstream_nodes = self.lineage_engine._get_upstream_nodes(node_id)
            results[node_id] = [node.to_dict() for node in upstream_nodes]
        return results
    
    def _query_downstream(self, target_nodes: List[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute downstream query"""
        results = {}
        for node_id in target_nodes:
            downstream_nodes = self.lineage_engine._get_downstream_nodes(node_id)
            results[node_id] = [node.to_dict() for node in downstream_nodes]
        return results
    
    def _query_full_lineage(self, target_nodes: List[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute full lineage query"""
        results = {}
        for node_id in target_nodes:
            results[node_id] = self.lineage_engine.get_dataset_lineage(node_id)
        return results
    
    def _query_impact_analysis(self, target_nodes: List[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute impact analysis query"""
        results = {}
        for node_id in target_nodes:
            impact_result = self.analyze_impact(node_id)
            results[node_id] = asdict(impact_result)
        return results
    
    def _query_shortest_path(self, target_nodes: List[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute shortest path query"""
        if len(target_nodes) != 2:
            raise ValueError("Shortest path query requires exactly 2 target nodes")
        
        # Create graph and find shortest path
        graph = self.generate_lineage_graph(target_nodes[0], include_upstream=True, include_downstream=True)
        
        try:
            path = nx.shortest_path(graph, target_nodes[0], target_nodes[1])
            return {"path": path, "length": len(path) - 1}
        except nx.NetworkXNoPath:
            return {"path": [], "length": -1, "message": "No path found"}
    
    def _query_common_ancestors(self, target_nodes: List[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute common ancestors query"""
        if len(target_nodes) < 2:
            raise ValueError("Common ancestors query requires at least 2 target nodes")
        
        # Get upstream nodes for each target
        upstream_sets = []
        for node_id in target_nodes:
            upstream_nodes = self.lineage_engine._get_upstream_nodes(node_id)
            upstream_sets.append(set(node.id for node in upstream_nodes))
        
        # Find intersection
        common_ancestors = set.intersection(*upstream_sets) if upstream_sets else set()
        
        return {"common_ancestors": list(common_ancestors), "count": len(common_ancestors)}
    
    # Impact analysis helper methods
    
    def _calculate_impact_score(
        self,
        target_node_id: str,
        affected_nodes: List[str],
        affected_models: List[str],
        change_type: str
    ) -> float:
        """Calculate impact score based on various factors"""
        base_score = 0.0
        
        # Factor in number of affected nodes
        base_score += len(affected_nodes) * 0.1
        
        # Factor in number of affected models (higher weight)
        base_score += len(affected_models) * 0.3
        
        # Factor in change type severity
        change_severity = {
            "modification": 0.2,
            "schema_change": 0.6,
            "deletion": 1.0,
            "corruption": 0.8
        }
        base_score += change_severity.get(change_type, 0.5)
        
        # Normalize to 0-1 scale
        return min(base_score, 1.0)
    
    def _determine_risk_level(self, impact_score: float, affected_nodes_count: int, affected_models_count: int) -> str:
        """Determine risk level based on impact metrics"""
        if impact_score >= 0.8 or affected_models_count >= 5:
            return "HIGH"
        elif impact_score >= 0.5 or affected_models_count >= 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_impact_recommendations(
        self,
        target_node_id: str,
        affected_nodes: List[str],
        affected_models: List[str],
        change_type: str,
        risk_level: str
    ) -> List[str]:
        """Generate recommendations based on impact analysis"""
        recommendations = []
        
        if risk_level == "HIGH":
            recommendations.append("Consider staging this change in a test environment first")
            recommendations.append("Notify all stakeholders before implementing changes")
            recommendations.append("Prepare rollback procedures")
        
        if affected_models:
            recommendations.append(f"Retrain and validate {len(affected_models)} affected models")
            recommendations.append("Monitor model performance after changes")
        
        if change_type == "schema_change":
            recommendations.append("Update data validation rules")
            recommendations.append("Check downstream transformations for compatibility")
        
        if len(affected_nodes) > 10:
            recommendations.append("Consider implementing changes in phases")
        
        return recommendations
    
    def _calculate_max_depth(self, node_id: str, direction: str) -> int:
        """Calculate maximum depth in upstream or downstream direction"""
        if direction == "upstream":
            nodes = self.lineage_engine._get_upstream_nodes(node_id)
        else:
            nodes = self.lineage_engine._get_downstream_nodes(node_id)
        
        # This is a simplified depth calculation
        # In a real implementation, you'd traverse the graph to find the actual maximum depth
        return len(nodes)
    
    def _get_transformation_types_in_path(self, node_id: str) -> List[str]:
        """Get all transformation types in the lineage path"""
        related_edges = self.lineage_engine._get_related_edges(node_id)
        return list(set(edge.transformation_type.value for edge in related_edges))
    
    def _identify_critical_paths(self, node_id: str, affected_models: List[str]) -> List[str]:
        """Identify critical paths to affected models"""
        # Simplified implementation - would need more sophisticated path analysis
        return [f"Path to {model_id}" for model_id in affected_models[:3]]
    
    def _get_node_statistics(self, node_id: str) -> Dict[str, Any]:
        """Get statistics for a specific node"""
        upstream_nodes = self.lineage_engine._get_upstream_nodes(node_id)
        downstream_nodes = self.lineage_engine._get_downstream_nodes(node_id)
        versions = self.lineage_engine.dataset_versions.get(node_id, [])
        
        return {
            "upstream_count": len(upstream_nodes),
            "downstream_count": len(downstream_nodes),
            "version_count": len(versions),
            "last_updated": max([v.created_at for v in versions]).isoformat() if versions else None
        }
    
    # Report generation methods
    
    def _generate_html_report(
        self,
        lineage: Dict[str, Any],
        impact_analysis: ImpactAnalysisResult,
        stats: Dict[str, Any],
        include_visualizations: bool
    ) -> str:
        """Generate HTML report"""
        html = f"""
        <html>
        <head><title>Lineage Report for {lineage['dataset']['id']}</title></head>
        <body>
        <h1>Data Lineage Report</h1>
        <h2>Dataset: {lineage['dataset']['name']}</h2>
        <p>Generated on: {datetime.now().isoformat()}</p>
        
        <h3>Statistics</h3>
        <ul>
        <li>Upstream nodes: {stats['upstream_count']}</li>
        <li>Downstream nodes: {stats['downstream_count']}</li>
        <li>Versions: {stats['version_count']}</li>
        </ul>
        
        <h3>Impact Analysis</h3>
        <p>Risk Level: <strong>{impact_analysis.risk_level}</strong></p>
        <p>Impact Score: {impact_analysis.impact_score:.2f}</p>
        <p>Affected Nodes: {len(impact_analysis.affected_nodes)}</p>
        <p>Affected Models: {len(impact_analysis.affected_models)}</p>
        
        <h4>Recommendations</h4>
        <ul>
        {"".join(f"<li>{rec}</li>" for rec in impact_analysis.recommendations)}
        </ul>
        
        </body>
        </html>
        """
        return html
    
    def _generate_json_report(
        self,
        lineage: Dict[str, Any],
        impact_analysis: ImpactAnalysisResult,
        stats: Dict[str, Any]
    ) -> str:
        """Generate JSON report"""
        report = {
            "lineage": lineage,
            "impact_analysis": asdict(impact_analysis),
            "statistics": stats,
            "generated_at": datetime.now().isoformat()
        }
        return json.dumps(report, indent=2, default=str)
    
    def _generate_markdown_report(
        self,
        lineage: Dict[str, Any],
        impact_analysis: ImpactAnalysisResult,
        stats: Dict[str, Any]
    ) -> str:
        """Generate Markdown report"""
        report = f"""# Data Lineage Report

## Dataset: {lineage['dataset']['name']}
Generated on: {datetime.now().isoformat()}

### Statistics
- Upstream nodes: {stats['upstream_count']}
- Downstream nodes: {stats['downstream_count']}
- Versions: {stats['version_count']}

### Impact Analysis
- **Risk Level**: {impact_analysis.risk_level}
- **Impact Score**: {impact_analysis.impact_score:.2f}
- **Affected Nodes**: {len(impact_analysis.affected_nodes)}
- **Affected Models**: {len(impact_analysis.affected_models)}

#### Recommendations
{chr(10).join(f"- {rec}" for rec in impact_analysis.recommendations)}
"""
        return report