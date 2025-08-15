"""
Demo script for AI Data Readiness Platform Lineage Visualization

This script demonstrates the comprehensive lineage tracking and visualization
capabilities of the AI Data Readiness Platform.
"""

import json
from datetime import datetime
from ai_data_readiness.engines.lineage_engine import (
    LineageEngine, 
    TransformationType, 
    LineageNodeType
)
from ai_data_readiness.engines.lineage_visualizer import (
    LineageVisualizer,
    VisualizationType,
    QueryType
)


def create_sample_lineage():
    """Create a comprehensive sample lineage for demonstration"""
    engine = LineageEngine()
    
    print("🔧 Creating sample data lineage...")
    
    # Create raw data sources
    raw_datasets = [
        ("customer_data", "Customer Demographics", {
            "source": "CRM System",
            "format": "csv",
            "size_mb": 150,
            "columns": ["customer_id", "age", "gender", "location", "income"],
            "quality_score": 0.85
        }),
        ("transaction_data", "Transaction History", {
            "source": "Payment System",
            "format": "json",
            "size_mb": 500,
            "columns": ["transaction_id", "customer_id", "amount", "timestamp", "merchant"],
            "quality_score": 0.92
        }),
        ("product_data", "Product Catalog", {
            "source": "Inventory System",
            "format": "parquet",
            "size_mb": 75,
            "columns": ["product_id", "category", "price", "description"],
            "quality_score": 0.95
        })
    ]
    
    for dataset_id, name, metadata in raw_datasets:
        engine.create_dataset_node(dataset_id, name, metadata)
    
    # Create cleaned datasets
    cleaned_datasets = [
        ("customer_cleaned", "Cleaned Customer Data", {
            "format": "parquet",
            "size_mb": 140,
            "quality_score": 0.95,
            "transformations_applied": ["null_removal", "data_validation", "standardization"]
        }),
        ("transaction_cleaned", "Cleaned Transaction Data", {
            "format": "parquet", 
            "size_mb": 480,
            "quality_score": 0.98,
            "transformations_applied": ["duplicate_removal", "outlier_detection", "currency_normalization"]
        })
    ]
    
    for dataset_id, name, metadata in cleaned_datasets:
        engine.create_dataset_node(dataset_id, name, metadata)
    
    # Create feature datasets
    feature_datasets = [
        ("customer_features", "Customer Feature Set", {
            "format": "parquet",
            "size_mb": 160,
            "feature_count": 25,
            "features": ["age_group", "spending_score", "loyalty_index", "location_cluster"]
        }),
        ("transaction_features", "Transaction Feature Set", {
            "format": "parquet",
            "size_mb": 520,
            "feature_count": 35,
            "features": ["avg_transaction_amount", "transaction_frequency", "merchant_diversity"]
        }),
        ("combined_features", "Combined Feature Set", {
            "format": "parquet",
            "size_mb": 680,
            "feature_count": 60,
            "ai_readiness_score": 0.92
        })
    ]
    
    for dataset_id, name, metadata in feature_datasets:
        engine.create_dataset_node(dataset_id, name, metadata)
    
    # Create ML datasets
    ml_datasets = [
        ("train_set", "Training Dataset", {
            "format": "parquet",
            "size_mb": 544,  # 80% of combined features
            "sample_count": 80000,
            "split_ratio": 0.8
        }),
        ("validation_set", "Validation Dataset", {
            "format": "parquet",
            "size_mb": 68,   # 10% of combined features
            "sample_count": 10000,
            "split_ratio": 0.1
        }),
        ("test_set", "Test Dataset", {
            "format": "parquet",
            "size_mb": 68,   # 10% of combined features
            "sample_count": 10000,
            "split_ratio": 0.1
        })
    ]
    
    for dataset_id, name, metadata in ml_datasets:
        engine.create_dataset_node(dataset_id, name, metadata)
    
    # Create transformations
    transformations = [
        # Data cleaning transformations
        ("customer_data", "customer_cleaned", TransformationType.CLEANING, {
            "operations": ["remove_nulls", "standardize_formats", "validate_ranges"],
            "rows_removed": 1200,
            "quality_improvement": 0.10
        }),
        ("transaction_data", "transaction_cleaned", TransformationType.CLEANING, {
            "operations": ["remove_duplicates", "detect_outliers", "normalize_currency"],
            "rows_removed": 5000,
            "quality_improvement": 0.06
        }),
        
        # Feature engineering transformations
        ("customer_cleaned", "customer_features", TransformationType.FEATURE_ENGINEERING, {
            "operations": ["create_age_groups", "calculate_spending_score", "cluster_locations"],
            "features_created": 15,
            "encoding_methods": ["one_hot", "label_encoding", "target_encoding"]
        }),
        ("transaction_cleaned", "transaction_features", TransformationType.FEATURE_ENGINEERING, {
            "operations": ["aggregate_by_customer", "calculate_frequencies", "create_time_features"],
            "features_created": 20,
            "aggregation_windows": ["7d", "30d", "90d"]
        }),
        
        # Data joining
        ("customer_features", "combined_features", TransformationType.JOINING, {
            "join_type": "inner",
            "join_key": "customer_id",
            "left_table": "customer_features",
            "right_table": "transaction_features"
        }),
        ("transaction_features", "combined_features", TransformationType.JOINING, {
            "join_type": "inner",
            "join_key": "customer_id",
            "left_table": "customer_features",
            "right_table": "transaction_features"
        }),
        
        # Data splitting
        ("combined_features", "train_set", TransformationType.SPLITTING, {
            "split_method": "stratified",
            "target_column": "churn_risk",
            "random_seed": 42
        }),
        ("combined_features", "validation_set", TransformationType.SPLITTING, {
            "split_method": "stratified",
            "target_column": "churn_risk",
            "random_seed": 42
        }),
        ("combined_features", "test_set", TransformationType.SPLITTING, {
            "split_method": "stratified",
            "target_column": "churn_risk",
            "random_seed": 42
        })
    ]
    
    for source, target, trans_type, details in transformations:
        engine.add_transformation_edge(source, target, trans_type, details, "data_engineer")
    
    # Create model links
    models = [
        ("churn_model_v1", "1.0.0", [
            ("train_set", "training", {"accuracy": 0.85, "precision": 0.82, "recall": 0.88}),
            ("validation_set", "validation", {"accuracy": 0.83, "precision": 0.80, "recall": 0.86}),
            ("test_set", "testing", {"accuracy": 0.84, "precision": 0.81, "recall": 0.87})
        ]),
        ("churn_model_v2", "2.0.0", [
            ("train_set", "training", {"accuracy": 0.89, "precision": 0.87, "recall": 0.91}),
            ("validation_set", "validation", {"accuracy": 0.87, "precision": 0.85, "recall": 0.89}),
            ("test_set", "testing", {"accuracy": 0.88, "precision": 0.86, "recall": 0.90})
        ]),
        ("recommendation_model", "1.0.0", [
            ("train_set", "training", {"ndcg": 0.75, "map": 0.68, "recall_at_10": 0.82}),
            ("test_set", "testing", {"ndcg": 0.73, "map": 0.66, "recall_at_10": 0.80})
        ])
    ]
    
    for model_id, model_version, dataset_links in models:
        for dataset_id, usage_type, metrics in dataset_links:
            engine.link_model_to_dataset(
                model_id, model_version, dataset_id, "1.0.0", usage_type, metrics
            )
    
    # Create some dataset versions to show evolution
    version_changes = [
        ("combined_features", [
            {"type": "feature_addition", "features": ["seasonal_trend", "weekend_behavior"], "description": "Added temporal features"}
        ], "Added seasonal and weekend behavior features"),
        ("train_set", [
            {"type": "data_refresh", "new_samples": 5000, "description": "Added recent customer data"}
        ], "Refreshed with latest customer data"),
    ]
    
    for dataset_id, changes, message in version_changes:
        engine.create_dataset_version(dataset_id, changes, "ml_engineer", message)
    
    print(f"✅ Created lineage with {len(engine.lineage_graph)} nodes and {len(engine.lineage_edges)} edges")
    return engine


def demonstrate_lineage_queries(visualizer):
    """Demonstrate various lineage query capabilities"""
    print("\n🔍 Demonstrating Lineage Queries...")
    
    # 1. Upstream query
    print("\n1. Upstream Analysis for Training Set:")
    upstream_query = visualizer.query_lineage(QueryType.UPSTREAM, ["train_set"])
    upstream_nodes = upstream_query.results["train_set"]
    print(f"   Found {len(upstream_nodes)} upstream dependencies:")
    for node in upstream_nodes[:3]:  # Show first 3
        print(f"   - {node['name']} ({node['id']})")
    print(f"   Query executed in {upstream_query.execution_time_ms}ms")
    
    # 2. Downstream query
    print("\n2. Downstream Impact Analysis for Combined Features:")
    downstream_query = visualizer.query_lineage(QueryType.DOWNSTREAM, ["combined_features"])
    downstream_nodes = downstream_query.results["combined_features"]
    print(f"   Found {len(downstream_nodes)} downstream dependencies:")
    for node in downstream_nodes:
        print(f"   - {node['name']} ({node['id']})")
    
    # 3. Common ancestors
    print("\n3. Common Ancestors of Train and Test Sets:")
    ancestors_query = visualizer.query_lineage(
        QueryType.COMMON_ANCESTORS, 
        ["train_set", "test_set"]
    )
    common_ancestors = ancestors_query.results["common_ancestors"]
    print(f"   Found {len(common_ancestors)} common ancestors:")
    for ancestor in common_ancestors[:3]:
        print(f"   - {ancestor}")
    
    # 4. Shortest path
    print("\n4. Shortest Path from Raw Data to Model:")
    try:
        path_query = visualizer.query_lineage(
            QueryType.SHORTEST_PATH,
            ["customer_data", "train_set"]
        )
        path_info = path_query.results
        print(f"   Path length: {path_info['length']} steps")
        print(f"   Path: {' → '.join(path_info['path'])}")
    except Exception as e:
        print(f"   Path analysis: {str(e)}")


def demonstrate_impact_analysis(visualizer):
    """Demonstrate impact analysis capabilities"""
    print("\n💥 Demonstrating Impact Analysis...")
    
    # Analyze impact of changes to different nodes
    test_scenarios = [
        ("customer_data", "schema_change", "Critical source data schema modification"),
        ("combined_features", "modification", "Feature set update"),
        ("test_set", "corruption", "Test data corruption incident")
    ]
    
    for node_id, change_type, description in test_scenarios:
        print(f"\n📊 Scenario: {description}")
        impact_result = visualizer.analyze_impact(
            node_id, 
            change_type=change_type,
            change_details={"scenario": description}
        )
        
        print(f"   Target Node: {impact_result.target_node_id}")
        print(f"   Risk Level: {impact_result.risk_level}")
        print(f"   Impact Score: {impact_result.impact_score:.3f}")
        print(f"   Affected Nodes: {len(impact_result.affected_nodes)}")
        print(f"   Affected Models: {len(impact_result.affected_models)}")
        
        if impact_result.recommendations:
            print("   Top Recommendations:")
            for i, rec in enumerate(impact_result.recommendations[:2], 1):
                print(f"   {i}. {rec}")


def demonstrate_search_capabilities(visualizer):
    """Demonstrate lineage search capabilities"""
    print("\n🔎 Demonstrating Search Capabilities...")
    
    search_scenarios = [
        ("customer", ["name"], "Search by name containing 'customer'"),
        ("parquet", ["metadata"], "Search by format in metadata"),
        ("churn", ["name", "metadata"], "Search for churn-related assets")
    ]
    
    for search_term, fields, description in search_scenarios:
        print(f"\n🔍 {description}:")
        results = visualizer.search_lineage(search_term, search_fields=fields)
        print(f"   Found {len(results)} matching nodes:")
        
        for result in results[:3]:  # Show first 3 results
            node = result["node"]
            print(f"   - {node['name']} ({node['id']})")
            print(f"     Type: {node['node_type']}, Upstream: {result['upstream_count']}, Downstream: {result['downstream_count']}")


def demonstrate_reporting(visualizer):
    """Demonstrate report generation capabilities"""
    print("\n📋 Demonstrating Report Generation...")
    
    # Generate different types of reports
    target_node = "combined_features"
    
    print(f"\n📄 Generating reports for {target_node}...")
    
    # JSON Report
    json_report = visualizer.generate_lineage_report(target_node, "json")
    json_data = json.loads(json_report)
    print(f"   JSON Report: {len(json_report)} characters")
    print(f"   Contains: lineage, impact_analysis, statistics")
    
    # HTML Report
    html_report = visualizer.generate_lineage_report(target_node, "html", include_visualizations=False)
    print(f"   HTML Report: {len(html_report)} characters")
    
    # Markdown Report
    md_report = visualizer.generate_lineage_report(target_node, "markdown")
    print(f"   Markdown Report: {len(md_report)} characters")
    
    # Show sample of markdown report
    print("\n📝 Sample Markdown Report:")
    print("   " + "\n   ".join(md_report.split('\n')[:10]))
    print("   ...")


def demonstrate_statistics(engine, visualizer):
    """Demonstrate lineage statistics"""
    print("\n📈 Demonstrating Lineage Statistics...")
    
    # Overall statistics
    stats = engine.get_lineage_statistics()
    print(f"\n📊 Overall Lineage Statistics:")
    print(f"   Total Nodes: {stats['total_nodes']}")
    print(f"   Total Edges: {stats['total_edges']}")
    print(f"   Total Dataset Versions: {stats['total_dataset_versions']}")
    print(f"   Total Model Links: {stats['total_model_links']}")
    
    print(f"\n🏷️  Node Types:")
    for node_type, count in stats['node_types'].items():
        print(f"   - {node_type}: {count}")
    
    print(f"\n🔄 Transformation Types:")
    for trans_type, count in stats['transformation_types'].items():
        print(f"   - {trans_type}: {count}")
    
    # Node-specific statistics
    key_nodes = ["customer_data", "combined_features", "train_set"]
    print(f"\n📋 Key Node Statistics:")
    
    for node_id in key_nodes:
        node_stats = visualizer._get_node_statistics(node_id)
        print(f"   {node_id}:")
        print(f"     Upstream: {node_stats['upstream_count']}, Downstream: {node_stats['downstream_count']}")
        print(f"     Versions: {node_stats['version_count']}")


def main():
    """Main demonstration function"""
    print("🚀 AI Data Readiness Platform - Lineage Visualization Demo")
    print("=" * 60)
    
    # Create sample lineage
    engine = create_sample_lineage()
    visualizer = LineageVisualizer(engine)
    
    # Demonstrate capabilities
    demonstrate_lineage_queries(visualizer)
    demonstrate_impact_analysis(visualizer)
    demonstrate_search_capabilities(visualizer)
    demonstrate_reporting(visualizer)
    demonstrate_statistics(engine, visualizer)
    
    print("\n" + "=" * 60)
    print("✅ Demo completed successfully!")
    print("\nKey Features Demonstrated:")
    print("• Complete data lineage tracking with versioning")
    print("• Interactive graph generation and visualization")
    print("• Advanced querying (upstream, downstream, paths, ancestors)")
    print("• Impact analysis with risk assessment")
    print("• Comprehensive search capabilities")
    print("• Multi-format report generation (HTML, JSON, Markdown)")
    print("• Model-to-dataset linking and tracking")
    print("• Statistical analysis and metrics")
    
    print(f"\n📊 Final Statistics:")
    final_stats = engine.get_lineage_statistics()
    print(f"   Tracked {final_stats['total_nodes']} nodes with {final_stats['total_edges']} transformations")
    print(f"   {final_stats['total_model_links']} model-dataset links established")
    print(f"   {final_stats['total_dataset_versions']} dataset versions maintained")


if __name__ == "__main__":
    main()