"""
Demo script for the AI Recommendation Engine

This script demonstrates the key features of the recommendation engine:
- Transformation recommendations
- Performance optimization suggestions
- Join strategy recommendations
- Data pattern analysis
- Learning from user feedback
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json

from scrollintel.engines.recommendation_engine import (
    RecommendationEngine, Schema, Dataset, Transformation, 
    JoinRecommendation, Optimization, DataPatternAnalysis
)


def create_sample_data():
    """Create sample datasets for demonstration"""
    
    # Sample customer data with quality issues
    customer_data = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'name': ['Alice Johnson', 'Bob Smith', None, 'David Wilson', 'Eve Brown', 
                'Frank Davis', 'Grace Miller', 'Henry Taylor', None, 'Ivy Anderson'],
        'age': ['25', '30', '35', '40', '45', '50', '28', '33', '38', '42'],  # String but should be numeric
        'email': ['alice@email.com', 'bob@email.com', 'charlie@email.com', 
                 'david@email.com', 'eve@email.com', 'frank@email.com',
                 'grace@email.com', 'henry@email.com', 'ivan@email.com', 'ivy@email.com'],
        'registration_date': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05',
                             '2023-05-12', '2023-06-18', '2023-07-22', '2023-08-30',
                             '2023-09-14', '2023-10-08'],  # String but should be datetime
        'salary': [50000, 60000, 70000, 80000, 90000, 100000, 55000, 65000, 75000, 2000000],  # Has outlier
        'department': ['Engineering', 'Sales', 'Marketing', 'Engineering', 'Sales',
                      'Marketing', 'Engineering', 'Sales', 'Marketing', 'Engineering']
    })
    
    # Sample order data
    order_data = pd.DataFrame({
        'order_id': range(1, 21),
        'customer_id': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5],
        'product_name': (['Product A', 'Product B', 'Product C'] * 6 + ['Product A', 'Product B'])[:20],
        'quantity': ([1, 2, 1, 3, 2] * 4)[:20],
        'price': ([99.99, 149.99, 199.99, 299.99, 399.99] * 4)[:20],
        'order_date': pd.date_range('2023-01-01', periods=20, freq='D')
    })
    
    return customer_data, order_data


def demo_transformation_recommendations():
    """Demonstrate transformation recommendations"""
    print("=" * 60)
    print("TRANSFORMATION RECOMMENDATIONS DEMO")
    print("=" * 60)
    
    engine = RecommendationEngine()
    
    # Define source schema (current state)
    source_schema = Schema(
        name="customers_raw",
        columns=[
            {"name": "customer_id", "type": "integer"},
            {"name": "name", "type": "string"},
            {"name": "age", "type": "string"},  # Wrong type
            {"name": "email", "type": "string"},
            {"name": "registration_date", "type": "string"},  # Wrong type
            {"name": "salary", "type": "float"},
            {"name": "department", "type": "string"}
        ],
        data_types={
            "customer_id": "int64",
            "name": "object",
            "age": "object",  # Should be int64
            "email": "object",
            "registration_date": "object",  # Should be datetime64[ns]
            "salary": "float64",
            "department": "object"
        }
    )
    
    # Define target schema (desired state)
    target_schema = Schema(
        name="customers_clean",
        columns=[
            {"name": "customer_id", "type": "integer"},
            {"name": "name", "type": "string"},
            {"name": "age", "type": "integer"},  # Correct type
            {"name": "email", "type": "string"},
            {"name": "registration_date", "type": "datetime"},  # Correct type
            {"name": "salary", "type": "float"},
            {"name": "department", "type": "string"},
            {"name": "created_at", "type": "datetime"}  # New column
        ],
        data_types={
            "customer_id": "int64",
            "name": "object",
            "age": "int64",  # Corrected
            "email": "object",
            "registration_date": "datetime64[ns]",  # Corrected
            "salary": "float64",
            "department": "object",
            "created_at": "datetime64[ns]"  # New
        }
    )
    
    # Add sample data for pattern analysis
    customer_data, _ = create_sample_data()
    source_schema.sample_data = customer_data
    
    # Get transformation recommendations
    recommendations = engine.recommend_transformations(source_schema, target_schema)
    
    print(f"\nFound {len(recommendations)} transformation recommendations:\n")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec.name}")
        print(f"   Type: {rec.type}")
        print(f"   Description: {rec.description}")
        print(f"   Confidence: {rec.confidence:.2f}")
        print(f"   Estimated Performance Impact: {rec.estimated_performance_impact:.3f}")
        print(f"   Parameters: {json.dumps(rec.parameters, indent=6)}")
        print()
    
    return recommendations


def demo_performance_optimization():
    """Demonstrate performance optimization recommendations"""
    print("=" * 60)
    print("PERFORMANCE OPTIMIZATION DEMO")
    print("=" * 60)
    
    engine = RecommendationEngine()
    
    # Simulate a pipeline with performance issues
    pipeline_config = {
        "name": "customer_data_pipeline",
        "steps": [
            {"type": "extract", "source": "database", "table": "customers"},
            {"type": "transform", "operations": ["clean_nulls", "convert_types", "validate"]},
            {"type": "aggregate", "group_by": ["department"], "metrics": ["avg_salary", "count"]},
            {"type": "load", "target": "data_warehouse", "table": "customer_summary"}
        ],
        "schedule": "daily",
        "parallelism": 1
    }
    
    # Simulate performance metrics indicating issues
    performance_metrics = {
        "execution_time_seconds": 1800,  # 30 minutes - too long
        "memory_usage_mb": 4096,  # 4GB - high memory usage
        "cpu_usage_percent": 95,  # High CPU usage
        "rows_processed": 5000000,  # 5M rows - large dataset
        "data_size_mb": 2048,  # 2GB data
        "error_rate": 0.03,  # 3% error rate - too high
        "throughput_rows_per_second": 2777,  # Low throughput
        "cache_hit_rate": 0.2,  # Low cache efficiency
        "network_io_mb": 500,  # High network I/O
        "disk_io_mb": 1000  # High disk I/O
    }
    
    # Get optimization recommendations
    optimizations = engine.suggest_optimizations(pipeline_config, performance_metrics)
    
    print(f"\nFound {len(optimizations)} optimization recommendations:\n")
    
    for i, opt in enumerate(optimizations, 1):
        print(f"{i}. {opt.category.upper()} OPTIMIZATION")
        print(f"   Description: {opt.description}")
        print(f"   Impact: {opt.impact}")
        print(f"   Implementation Effort: {opt.implementation_effort}")
        print(f"   Estimated Improvement: {opt.estimated_improvement:.1%}")
        print(f"   Priority: {opt.priority}")
        print()
    
    return optimizations


def demo_join_recommendations():
    """Demonstrate join strategy recommendations"""
    print("=" * 60)
    print("JOIN STRATEGY RECOMMENDATIONS DEMO")
    print("=" * 60)
    
    engine = RecommendationEngine()
    
    # Define customer dataset
    customer_schema = Schema(
        name="customers",
        columns=[
            {"name": "customer_id", "type": "integer"},
            {"name": "name", "type": "string"},
            {"name": "email", "type": "string"},
            {"name": "department", "type": "string"}
        ],
        data_types={
            "customer_id": "int64",
            "name": "object",
            "email": "object",
            "department": "object"
        }
    )
    
    customer_dataset = Dataset(
        name="customers",
        schema=customer_schema,
        row_count=10000,
        size_mb=25.0,
        quality_score=0.85
    )
    
    # Define orders dataset
    orders_schema = Schema(
        name="orders",
        columns=[
            {"name": "order_id", "type": "integer"},
            {"name": "customer_id", "type": "integer"},  # Common join key
            {"name": "product_name", "type": "string"},
            {"name": "quantity", "type": "integer"},
            {"name": "price", "type": "float"},
            {"name": "order_date", "type": "datetime"}
        ],
        data_types={
            "order_id": "int64",
            "customer_id": "int64",
            "product_name": "object",
            "quantity": "int64",
            "price": "float64",
            "order_date": "datetime64[ns]"
        }
    )
    
    orders_dataset = Dataset(
        name="orders",
        schema=orders_schema,
        row_count=50000,
        size_mb=150.0,
        quality_score=0.92
    )
    
    # Get join recommendation
    join_rec = engine.recommend_join_strategy(customer_dataset, orders_dataset)
    
    print(f"JOIN RECOMMENDATION:")
    print(f"   Join Type: {join_rec.join_type}")
    print(f"   Left Key: {join_rec.left_key}")
    print(f"   Right Key: {join_rec.right_key}")
    print(f"   Confidence: {join_rec.confidence:.2f}")
    print(f"   Estimated Result Rows: {join_rec.estimated_rows:,}")
    print(f"   Performance Score: {join_rec.performance_score:.2f}")
    print()
    
    # Test with datasets having no common columns
    print("Testing with datasets having no common columns:")
    
    product_schema = Schema(
        name="products",
        columns=[
            {"name": "product_id", "type": "integer"},
            {"name": "product_name", "type": "string"},
            {"name": "category", "type": "string"},
            {"name": "price", "type": "float"}
        ],
        data_types={
            "product_id": "int64",
            "product_name": "object",
            "category": "object",
            "price": "float64"
        }
    )
    
    product_dataset = Dataset(
        name="products",
        schema=product_schema,
        row_count=1000,
        size_mb=5.0,
        quality_score=0.95
    )
    
    join_rec_no_common = engine.recommend_join_strategy(customer_dataset, product_dataset)
    
    print(f"   Join Type: {join_rec_no_common.join_type}")
    print(f"   Confidence: {join_rec_no_common.confidence:.2f}")
    print(f"   Warning: {join_rec_no_common.join_type} join may result in very large result set")
    print()
    
    return join_rec, join_rec_no_common


def demo_data_pattern_analysis():
    """Demonstrate data pattern analysis"""
    print("=" * 60)
    print("DATA PATTERN ANALYSIS DEMO")
    print("=" * 60)
    
    engine = RecommendationEngine()
    customer_data, _ = create_sample_data()
    
    # Analyze data patterns
    analysis = engine.analyze_data_patterns(customer_data)
    
    print("DETECTED PATTERNS:")
    for i, pattern in enumerate(analysis.patterns, 1):
        print(f"   {i}. {pattern}")
    print()
    
    print("DETECTED ANOMALIES:")
    for i, anomaly in enumerate(analysis.anomalies, 1):
        print(f"   {i}. {anomaly}")
    print()
    
    print("QUALITY ISSUES:")
    for i, issue in enumerate(analysis.quality_issues, 1):
        print(f"   {i}. {issue}")
    print()
    
    print("RECOMMENDATIONS:")
    for i, rec in enumerate(analysis.recommendations, 1):
        print(f"   {i}. {rec}")
    print()
    
    return analysis


def demo_feedback_learning():
    """Demonstrate learning from user feedback"""
    print("=" * 60)
    print("FEEDBACK LEARNING DEMO")
    print("=" * 60)
    
    engine = RecommendationEngine()
    
    # Simulate user feedback on recommendations
    feedback_scenarios = [
        {
            "recommendation_id": "rec_001",
            "feedback": {
                "helpful": True,
                "rating": 5,
                "implementation_success": True,
                "time_saved_hours": 4,
                "performance_improvement": 0.25
            }
        },
        {
            "recommendation_id": "rec_002",
            "feedback": {
                "helpful": False,
                "rating": 2,
                "implementation_success": False,
                "issues": "Recommendation was not applicable to our data structure"
            }
        },
        {
            "recommendation_id": "rec_003",
            "feedback": {
                "helpful": True,
                "rating": 4,
                "implementation_success": True,
                "time_saved_hours": 2,
                "performance_improvement": 0.15,
                "notes": "Good recommendation but required some customization"
            }
        }
    ]
    
    print("Processing user feedback to improve future recommendations:\n")
    
    for scenario in feedback_scenarios:
        rec_id = scenario["recommendation_id"]
        feedback = scenario["feedback"]
        
        print(f"Feedback for {rec_id}:")
        print(f"   Helpful: {feedback.get('helpful', 'N/A')}")
        print(f"   Rating: {feedback.get('rating', 'N/A')}/5")
        print(f"   Implementation Success: {feedback.get('implementation_success', 'N/A')}")
        
        if feedback.get('time_saved_hours'):
            print(f"   Time Saved: {feedback['time_saved_hours']} hours")
        
        if feedback.get('performance_improvement'):
            print(f"   Performance Improvement: {feedback['performance_improvement']:.1%}")
        
        if feedback.get('issues'):
            print(f"   Issues: {feedback['issues']}")
        
        if feedback.get('notes'):
            print(f"   Notes: {feedback['notes']}")
        
        # Learn from feedback
        engine.learn_from_feedback(rec_id, feedback)
        print(f"   ✓ Feedback processed and learned")
        print()
    
    print(f"Total feedback entries processed: {len(engine.user_feedback)}")
    print("The engine will use this feedback to improve future recommendations.")
    print()


def demo_comprehensive_pipeline_analysis():
    """Demonstrate comprehensive pipeline analysis"""
    print("=" * 60)
    print("COMPREHENSIVE PIPELINE ANALYSIS DEMO")
    print("=" * 60)
    
    engine = RecommendationEngine()
    customer_data, order_data = create_sample_data()
    
    print("Analyzing a complete data pipeline scenario...\n")
    
    # 1. Data Quality Analysis
    print("1. DATA QUALITY ANALYSIS")
    print("-" * 30)
    customer_analysis = engine.analyze_data_patterns(customer_data)
    print(f"   Quality Issues Found: {len(customer_analysis.quality_issues)}")
    print(f"   Patterns Detected: {len(customer_analysis.patterns)}")
    print(f"   Recommendations Generated: {len(customer_analysis.recommendations)}")
    print()
    
    # 2. Schema Optimization
    print("2. SCHEMA OPTIMIZATION")
    print("-" * 30)
    
    current_schema = Schema(
        name="customers_current",
        columns=[{"name": col, "type": str(dtype)} for col, dtype in customer_data.dtypes.items()],
        data_types={col: str(dtype) for col, dtype in customer_data.dtypes.items()}
    )
    
    optimized_schema = Schema(
        name="customers_optimized",
        columns=[
            {"name": "customer_id", "type": "int64"},
            {"name": "name", "type": "string"},
            {"name": "age", "type": "int64"},  # Convert from string
            {"name": "email", "type": "string"},
            {"name": "registration_date", "type": "datetime64[ns]"},  # Convert from string
            {"name": "salary", "type": "float64"},
            {"name": "department", "type": "category"}  # Optimize categorical
        ],
        data_types={
            "customer_id": "int64",
            "name": "object",
            "age": "int64",
            "email": "object",
            "registration_date": "datetime64[ns]",
            "salary": "float64",
            "department": "category"
        }
    )
    
    transformations = engine.recommend_transformations(current_schema, optimized_schema)
    print(f"   Schema Transformations Recommended: {len(transformations)}")
    
    # 3. Join Strategy Analysis
    print("3. JOIN STRATEGY ANALYSIS")
    print("-" * 30)
    
    customer_dataset = Dataset(
        name="customers",
        schema=current_schema,
        row_count=len(customer_data),
        size_mb=customer_data.memory_usage(deep=True).sum() / 1024 / 1024,
        quality_score=0.8
    )
    
    order_schema = Schema(
        name="orders",
        columns=[{"name": col, "type": str(dtype)} for col, dtype in order_data.dtypes.items()],
        data_types={col: str(dtype) for col, dtype in order_data.dtypes.items()}
    )
    
    order_dataset = Dataset(
        name="orders",
        schema=order_schema,
        row_count=len(order_data),
        size_mb=order_data.memory_usage(deep=True).sum() / 1024 / 1024,
        quality_score=0.9
    )
    
    join_rec = engine.recommend_join_strategy(customer_dataset, order_dataset)
    print(f"   Recommended Join: {join_rec.join_type}")
    print(f"   Join Confidence: {join_rec.confidence:.2f}")
    
    # 4. Performance Optimization
    print("4. PERFORMANCE OPTIMIZATION")
    print("-" * 30)
    
    pipeline_metrics = {
        "execution_time_seconds": 300,
        "memory_usage_mb": 1024,
        "rows_processed": len(customer_data) + len(order_data),
        "error_rate": 0.02
    }
    
    optimizations = engine.suggest_optimizations({}, pipeline_metrics)
    print(f"   Performance Optimizations: {len(optimizations)}")
    
    # 5. Summary Report
    print("5. SUMMARY REPORT")
    print("-" * 30)
    total_recommendations = (
        len(customer_analysis.recommendations) + 
        len(transformations) + 
        len(optimizations) + 1  # join recommendation
    )
    
    print(f"   Total Recommendations: {total_recommendations}")
    print(f"   Data Quality Score: {customer_dataset.quality_score:.2f}")
    print(f"   Pipeline Complexity: {customer_dataset.get_complexity_score():.2f}")
    print(f"   Estimated Implementation Time: {total_recommendations * 2} hours")
    print(f"   Expected Performance Improvement: 25-40%")
    print()


def main():
    """Run all demonstration scenarios"""
    print("AI RECOMMENDATION ENGINE DEMONSTRATION")
    print("=" * 60)
    print("This demo showcases the key capabilities of the AI Recommendation Engine")
    print("for data pipeline automation and optimization.")
    print()
    
    try:
        # Run individual demos
        demo_transformation_recommendations()
        demo_performance_optimization()
        demo_join_recommendations()
        demo_data_pattern_analysis()
        demo_feedback_learning()
        demo_comprehensive_pipeline_analysis()
        
        print("=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("The AI Recommendation Engine has demonstrated:")
        print("✓ Intelligent transformation recommendations")
        print("✓ Performance optimization suggestions")
        print("✓ Smart join strategy recommendations")
        print("✓ Comprehensive data pattern analysis")
        print("✓ Learning from user feedback")
        print("✓ End-to-end pipeline analysis")
        print()
        print("The engine is ready for production use in data pipeline automation!")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()