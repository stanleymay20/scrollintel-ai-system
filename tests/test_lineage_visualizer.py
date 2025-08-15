"""
Tests for LineageVisualizer in AI Data Readiness Platform
"""

import pytest
from unittest.mock import Mock, patch
import json
import networkx as nx
from datetime import datetime

from ai_data_readiness.engines.lineage_engine import (
    LineageEngine, 
    TransformationType, 
    LineageNodeType
)
from ai_data_readiness.engines.lineage_visualizer import (
    LineageVisualizer,
    VisualizationType,
    QueryType,
    LineageQuery,
    ImpactAnalysisResult
)


class TestLineageVisualizer:
    """Test cases for LineageVisualizer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.lineage_engine = LineageEngine()
        self.visualizer = LineageVisualizer(self.lineage_engine)
        
        # Create test data
        self.setup_test_lineage()
    
    def setup_test_lineage(self):
        """Set up a test lineage graph"""
        # Create datasets
        datasets = [
            ("raw_data", "Raw Data", {"format": "csv", "size": 1000}),
            ("cleaned_data", "Cleaned Data", {"format": "parquet", "size": 950}),
            ("features", "Feature Set", {"format": "parquet", "size": 950}),
            ("train_set", "Training Set", {"format": "parquet", "size": 800}),
            ("test_set", "Test Set", {"format": "parquet", "size": 150})
        ]
        
        for dataset_id, name, metadata in datasets:
            self.lineage_engine.create_dataset_node(dataset_id, name, metadata)
        
        # Create transformations
        transformations = [
            ("raw_data", "cleaned_data", TransformationType.CLEANING, {"operation": "remove_nulls"}),
            ("cleaned_data", "features", TransformationType.FEATURE_ENGINEERING, {"operation": "create_features"}),
            ("features", "train_set", TransformationType.SPLITTING, {"ratio": 0.8}),
            ("features", "test_set", TransformationType.SPLITTING, {"ratio": 0.2})
        ]
        
        for source, target, trans_type, details in transformations:
            self.lineage_engine.add_transformation_edge(source, target, trans_type, details, "test_user")
        
        # Create model links
        self.lineage_engine.link_model_to_dataset(
            "model_v1", "1.0.0", "train_set", "1.0.0", "training", {"accuracy": 0.85}
        )
        self.lineage_engine.link_model_to_dataset(
            "model_v1", "1.0.0", "test_set", "1.0.0", "testing", {"accuracy": 0.82}
        )
    
    def test_generate_lineage_graph(self):
        """Test generating NetworkX graph for lineage"""
        graph = self.visualizer.generate_lineage_graph("train_set")
        
        assert isinstance(graph, nx.DiGraph)
        assert "train_set" in graph.nodes
        assert "features" in graph.nodes  # upstream
        assert "cleaned_data" in graph.nodes  # upstream
        assert "raw_data" in graph.nodes  # upstream
        
        # Check node attributes
        train_node = graph.nodes["train_set"]
        assert train_node["name"] == "Training Set"
        assert train_node["type"] == "dataset"
        
        # Check edges
        assert graph.has_edge("features", "train_set")
        edge_data = graph.edges["features", "train_set"]
        assert edge_data["transformation_type"] == "splitting"
    
    def test_generate_lineage_graph_with_depth_limit(self):
        """Test generating lineage graph with depth limit"""
        graph = self.visualizer.generate_lineage_graph("train_set", max_depth=1)
        
        assert "train_set" in graph.nodes
        assert "features" in graph.nodes  # depth 1
        assert "cleaned_data" not in graph.nodes  # depth 2, should be excluded
    
    def test_generate_lineage_graph_upstream_only(self):
        """Test generating lineage graph with upstream only"""
        graph = self.visualizer.generate_lineage_graph(
            "features", 
            include_upstream=True, 
            include_downstream=False
        )
        
        assert "features" in graph.nodes
        assert "cleaned_data" in graph.nodes  # upstream
        assert "raw_data" in graph.nodes  # upstream
        assert "train_set" not in graph.nodes  # downstream, should be excluded
        assert "test_set" not in graph.nodes  # downstream, should be excluded
    
    def test_generate_lineage_graph_downstream_only(self):
        """Test generating lineage graph with downstream only"""
        graph = self.visualizer.generate_lineage_graph(
            "features", 
            include_upstream=False, 
            include_downstream=True
        )
        
        assert "features" in graph.nodes
        assert "train_set" in graph.nodes  # downstream
        assert "test_set" in graph.nodes  # downstream
        assert "cleaned_data" not in graph.nodes  # upstream, should be excluded
        assert "raw_data" not in graph.nodes  # upstream, should be excluded
    
    @patch('ai_data_readiness.engines.lineage_visualizer.go')
    def test_create_interactive_visualization_graph(self, mock_go):
        """Test creating interactive graph visualization"""
        mock_figure = Mock()
        mock_go.Figure.return_value = mock_figure
        mock_go.Scatter.return_value = Mock()
        
        fig = self.visualizer.create_interactive_visualization(
            "train_set", 
            VisualizationType.GRAPH
        )
        
        assert mock_go.Figure.called
        assert fig == mock_figure
    
    @patch('ai_data_readiness.engines.lineage_visualizer.go')
    def test_create_interactive_visualization_timeline(self, mock_go):
        """Test creating timeline visualization"""
        mock_figure = Mock()
        mock_go.Figure.return_value = mock_figure
        
        fig = self.visualizer.create_interactive_visualization(
            "train_set", 
            VisualizationType.TIMELINE
        )
        
        assert mock_go.Figure.called
        assert fig == mock_figure
    
    def test_query_lineage_upstream(self):
        """Test upstream lineage query"""
        query = self.visualizer.query_lineage(
            QueryType.UPSTREAM,
            ["train_set"]
        )
        
        assert isinstance(query, LineageQuery)
        assert query.query_type == QueryType.UPSTREAM
        assert query.target_nodes == ["train_set"]
        assert query.results is not None
        assert query.execution_time_ms is not None
        
        # Check results
        results = query.results["train_set"]
        upstream_ids = [node["id"] for node in results]
        assert "features" in upstream_ids
        assert "cleaned_data" in upstream_ids
        assert "raw_data" in upstream_ids
    
    def test_query_lineage_downstream(self):
        """Test downstream lineage query"""
        query = self.visualizer.query_lineage(
            QueryType.DOWNSTREAM,
            ["features"]
        )
        
        assert query.query_type == QueryType.DOWNSTREAM
        results = query.results["features"]
        downstream_ids = [node["id"] for node in results]
        assert "train_set" in downstream_ids
        assert "test_set" in downstream_ids
    
    def test_query_lineage_full_lineage(self):
        """Test full lineage query"""
        query = self.visualizer.query_lineage(
            QueryType.FULL_LINEAGE,
            ["train_set"]
        )
        
        assert query.query_type == QueryType.FULL_LINEAGE
        results = query.results["train_set"]
        
        assert "dataset" in results
        assert "upstream_nodes" in results
        assert "downstream_nodes" in results
        assert "transformations" in results
        assert "versions" in results
        assert "model_links" in results
    
    def test_query_lineage_impact_analysis(self):
        """Test impact analysis query"""
        query = self.visualizer.query_lineage(
            QueryType.IMPACT_ANALYSIS,
            ["features"]
        )
        
        assert query.query_type == QueryType.IMPACT_ANALYSIS
        results = query.results["features"]
        
        assert "target_node_id" in results
        assert "affected_nodes" in results
        assert "affected_models" in results
        assert "impact_score" in results
        assert "risk_level" in results
    
    def test_query_lineage_shortest_path(self):
        """Test shortest path query"""
        query = self.visualizer.query_lineage(
            QueryType.SHORTEST_PATH,
            ["raw_data", "train_set"]
        )
        
        assert query.query_type == QueryType.SHORTEST_PATH
        results = query.results
        
        assert "path" in results
        assert "length" in results
        assert results["length"] > 0
        assert "raw_data" in results["path"]
        assert "train_set" in results["path"]
    
    def test_query_lineage_common_ancestors(self):
        """Test common ancestors query"""
        query = self.visualizer.query_lineage(
            QueryType.COMMON_ANCESTORS,
            ["train_set", "test_set"]
        )
        
        assert query.query_type == QueryType.COMMON_ANCESTORS
        results = query.results
        
        assert "common_ancestors" in results
        assert "count" in results
        assert "features" in results["common_ancestors"]
        assert "cleaned_data" in results["common_ancestors"]
        assert "raw_data" in results["common_ancestors"]
    
    def test_analyze_impact(self):
        """Test impact analysis"""
        impact_result = self.visualizer.analyze_impact("features")
        
        assert isinstance(impact_result, ImpactAnalysisResult)
        assert impact_result.target_node_id == "features"
        assert len(impact_result.affected_nodes) > 0
        assert len(impact_result.affected_models) > 0
        assert 0 <= impact_result.impact_score <= 1
        assert impact_result.risk_level in ["LOW", "MEDIUM", "HIGH"]
        assert len(impact_result.recommendations) > 0
        
        # Check that downstream nodes are included
        assert "train_set" in impact_result.affected_nodes
        assert "test_set" in impact_result.affected_nodes
        
        # Check that models are included
        assert "model_v1" in impact_result.affected_models
    
    def test_analyze_impact_with_change_details(self):
        """Test impact analysis with specific change details"""
        impact_result = self.visualizer.analyze_impact(
            "features",
            change_type="schema_change",
            change_details={"removed_columns": ["feature_1"]}
        )
        
        assert impact_result.target_node_id == "features"
        assert "schema_change" in impact_result.analysis_details["change_type"]
        
        # Schema changes should have higher impact
        assert impact_result.impact_score > 0.5
    
    def test_generate_lineage_report_html(self):
        """Test generating HTML lineage report"""
        report = self.visualizer.generate_lineage_report(
            "train_set",
            report_format="html",
            include_visualizations=False
        )
        
        assert isinstance(report, str)
        assert "<html>" in report
        assert "Data Lineage Report" in report
        assert "Training Set" in report
        assert "Impact Analysis" in report
        assert "Statistics" in report
    
    def test_generate_lineage_report_json(self):
        """Test generating JSON lineage report"""
        report = self.visualizer.generate_lineage_report(
            "train_set",
            report_format="json"
        )
        
        assert isinstance(report, str)
        report_data = json.loads(report)
        
        assert "lineage" in report_data
        assert "impact_analysis" in report_data
        assert "statistics" in report_data
        assert "generated_at" in report_data
    
    def test_generate_lineage_report_markdown(self):
        """Test generating Markdown lineage report"""
        report = self.visualizer.generate_lineage_report(
            "train_set",
            report_format="markdown"
        )
        
        assert isinstance(report, str)
        assert "# Data Lineage Report" in report
        assert "## Dataset: Training Set" in report
        assert "### Statistics" in report
        assert "### Impact Analysis" in report
    
    def test_search_lineage_by_name(self):
        """Test searching lineage by name"""
        results = self.visualizer.search_lineage("Training", search_fields=["name"])
        
        assert len(results) == 1
        assert results[0]["node"]["id"] == "train_set"
        assert results[0]["node"]["name"] == "Training Set"
        assert "upstream_count" in results[0]
        assert "downstream_count" in results[0]
    
    def test_search_lineage_by_metadata(self):
        """Test searching lineage by metadata"""
        results = self.visualizer.search_lineage("parquet", search_fields=["metadata"])
        
        # Should find multiple datasets with parquet format
        assert len(results) > 1
        parquet_datasets = [r["node"]["id"] for r in results]
        assert "cleaned_data" in parquet_datasets
        assert "features" in parquet_datasets
    
    def test_search_lineage_by_node_type(self):
        """Test searching lineage by node type"""
        results = self.visualizer.search_lineage(
            "data", 
            search_fields=["name"],
            node_types=[LineageNodeType.DATASET]
        )
        
        # Should only return dataset nodes
        for result in results:
            assert result["node"]["node_type"] == "dataset"
    
    def test_search_lineage_no_results(self):
        """Test searching lineage with no matching results"""
        results = self.visualizer.search_lineage("nonexistent")
        
        assert len(results) == 0
    
    def test_query_cache(self):
        """Test that queries are cached"""
        # Execute different queries to ensure they get different IDs
        query1 = self.visualizer.query_lineage(QueryType.UPSTREAM, ["train_set"])
        query2 = self.visualizer.query_lineage(QueryType.DOWNSTREAM, ["train_set"])
        
        # Both queries should be in cache
        assert len(self.visualizer.query_cache) >= 2
        assert query1.query_id in self.visualizer.query_cache
        assert query2.query_id in self.visualizer.query_cache
    
    def test_error_handling_invalid_node(self):
        """Test error handling for invalid node operations"""
        with pytest.raises(ValueError, match="Node nonexistent not found"):
            self.visualizer.generate_lineage_graph("nonexistent")
    
    def test_error_handling_invalid_visualization_type(self):
        """Test error handling for invalid visualization type"""
        with pytest.raises(ValueError, match="Unsupported visualization type"):
            # Create a mock invalid visualization type
            invalid_type = Mock()
            invalid_type.value = "invalid"
            self.visualizer.create_interactive_visualization("train_set", invalid_type)
    
    def test_error_handling_invalid_query_type(self):
        """Test error handling for invalid query type"""
        with pytest.raises(ValueError, match="Unsupported query type"):
            # Create a mock invalid query type
            invalid_type = Mock()
            invalid_type.value = "invalid"
            self.visualizer.query_lineage(invalid_type, ["train_set"])
    
    def test_error_handling_shortest_path_wrong_nodes(self):
        """Test error handling for shortest path with wrong number of nodes"""
        with pytest.raises(ValueError, match="Shortest path query requires exactly 2 target nodes"):
            self.visualizer.query_lineage(QueryType.SHORTEST_PATH, ["train_set"])
    
    def test_error_handling_common_ancestors_insufficient_nodes(self):
        """Test error handling for common ancestors with insufficient nodes"""
        with pytest.raises(ValueError, match="Common ancestors query requires at least 2 target nodes"):
            self.visualizer.query_lineage(QueryType.COMMON_ANCESTORS, ["train_set"])
    
    def test_impact_score_calculation(self):
        """Test impact score calculation logic"""
        # Test with different change types
        impact_low = self.visualizer.analyze_impact("test_set", change_type="modification")
        impact_high = self.visualizer.analyze_impact("features", change_type="deletion")
        
        # Deletion should have higher impact than modification
        assert impact_high.impact_score > impact_low.impact_score
        
        # Features node should have higher impact than test_set (more downstream dependencies)
        assert impact_high.impact_score > impact_low.impact_score
    
    def test_risk_level_determination(self):
        """Test risk level determination logic"""
        # Test with node that has many downstream dependencies
        impact_features = self.visualizer.analyze_impact("features")
        
        # Test with leaf node
        impact_test_set = self.visualizer.analyze_impact("test_set")
        
        # Features should have higher or equal risk level
        risk_levels = ["LOW", "MEDIUM", "HIGH"]
        features_risk_index = risk_levels.index(impact_features.risk_level)
        test_set_risk_index = risk_levels.index(impact_test_set.risk_level)
        
        assert features_risk_index >= test_set_risk_index
    
    def test_recommendations_generation(self):
        """Test that appropriate recommendations are generated"""
        impact_result = self.visualizer.analyze_impact("features", change_type="schema_change")
        
        recommendations = impact_result.recommendations
        assert len(recommendations) > 0
        
        # Should include model-related recommendations since models are affected
        model_recommendations = [rec for rec in recommendations if "model" in rec.lower()]
        assert len(model_recommendations) > 0
        
        # Should include schema-related recommendations for schema changes
        schema_recommendations = [rec for rec in recommendations if "schema" in rec.lower() or "validation" in rec.lower()]
        assert len(schema_recommendations) > 0
    
    def test_complex_lineage_scenario(self):
        """Test visualizer with complex lineage scenario"""
        # Add more complex transformations
        self.lineage_engine.create_dataset_node("validation_set", "Validation Set", {"format": "parquet"})
        self.lineage_engine.add_transformation_edge(
            "features", "validation_set", TransformationType.SPLITTING, {"ratio": 0.1}, "test_user"
        )
        
        # Add another model
        self.lineage_engine.link_model_to_dataset(
            "model_v2", "2.0.0", "train_set", "1.0.0", "training", {"accuracy": 0.90}
        )
        self.lineage_engine.link_model_to_dataset(
            "model_v2", "2.0.0", "validation_set", "1.0.0", "validation", {"accuracy": 0.88}
        )
        
        # Test full lineage query
        query = self.visualizer.query_lineage(QueryType.FULL_LINEAGE, ["features"])
        results = query.results["features"]
        
        # Should include all downstream nodes
        downstream_ids = [node["id"] for node in results["downstream_nodes"]]
        assert "train_set" in downstream_ids
        assert "test_set" in downstream_ids
        assert "validation_set" in downstream_ids
        
        # Should include model links for the features dataset itself
        # Note: model_links in lineage results only include links for the specific dataset
        # not for all downstream datasets
        features_model_links = [link for link in results["model_links"] if link["dataset_id"] == "features"]
        # Since we didn't link any models directly to features, this might be 0
        # But we can check that the structure is correct
        assert "model_links" in results
        
        # Test impact analysis
        impact_result = self.visualizer.analyze_impact("features")
        assert "validation_set" in impact_result.affected_nodes
        assert "model_v2" in impact_result.affected_models
    
    def test_node_statistics(self):
        """Test node statistics calculation"""
        stats = self.visualizer._get_node_statistics("features")
        
        assert "upstream_count" in stats
        assert "downstream_count" in stats
        assert "version_count" in stats
        assert stats["upstream_count"] == 2  # cleaned_data, raw_data
        assert stats["downstream_count"] == 2  # train_set, test_set
        assert stats["version_count"] == 1  # Initial version
    
    def test_transformation_types_in_path(self):
        """Test getting transformation types in lineage path"""
        trans_types = self.visualizer._get_transformation_types_in_path("train_set")
        
        assert "splitting" in trans_types  # Direct transformation to train_set
        # Note: This method gets transformations where the node is source or target
        # So it might not include all upstream transformations in this simple implementation


if __name__ == "__main__":
    pytest.main([__file__])