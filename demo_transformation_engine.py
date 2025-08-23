"""
Demo script for the Data Pipeline Transformation Engine.

This script demonstrates the key capabilities of the transformation engine including:
- Common transformations (filter, map, aggregate, join)
- Data type conversion and validation
- Custom transformation framework
- Performance optimization
- Intelligent recommendations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

from scrollintel.engines.transformation_engine import (
    TransformationEngine,
    TransformationConfig,
    TransformationType,
    DataType
)


def create_sample_datasets():
    """Create sample datasets for demonstration."""
    print("🔧 Creating sample datasets...")
    
    # Main employee dataset
    np.random.seed(42)
    employees = pd.DataFrame({
        'employee_id': range(1, 1001),
        'first_name': [f'Employee_{i}' for i in range(1, 1001)],
        'last_name': [f'Lastname_{i}' for i in range(1, 1001)],
        'age': np.random.randint(22, 65, 1000),
        'salary': np.random.normal(75000, 20000, 1000),
        'department_id': np.random.randint(1, 6, 1000),
        'hire_date': [
            (datetime.now() - timedelta(days=np.random.randint(30, 2000))).strftime('%Y-%m-%d')
            for _ in range(1000)
        ],
        'is_active': np.random.choice([True, False], 1000, p=[0.85, 0.15]),
        'performance_score': np.random.uniform(1.0, 5.0, 1000),
        'overtime_hours': np.random.poisson(5, 1000)
    })
    
    # Department lookup dataset
    departments = pd.DataFrame({
        'department_id': [1, 2, 3, 4, 5],
        'department_name': ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'],
        'budget': [2000000, 1500000, 800000, 600000, 1200000],
        'manager_id': [101, 102, 103, 104, 105]
    })
    
    # Dataset with data quality issues
    dirty_data = pd.DataFrame({
        'id': [1, 2, 3, None, 5, 6, 7, 8],
        'name': ['Alice', 'Bob', '', 'David', None, 'Frank', 'Grace', 'Henry'],
        'age_str': ['25', '30', 'invalid', '40', '45', '50', '55', '60'],
        'salary_str': ['50000', None, '70000', '80000', '90000', '100k', '110000', '120000'],
        'email': ['alice@company.com', 'bob@invalid', 'charlie@company.com', None,
                 'eve@company.com', 'frank@company.com', 'grace@company.com', 'henry@company.com'],
        'join_date': ['2023-01-15', '2023-02-20', 'invalid_date', '2023-04-10',
                     '2023-05-05', '2023-06-12', '2023-07-08', '2023-08-22']
    })
    
    print(f"✅ Created datasets:")
    print(f"   - Employees: {len(employees)} rows, {len(employees.columns)} columns")
    print(f"   - Departments: {len(departments)} rows, {len(departments.columns)} columns")
    print(f"   - Dirty data: {len(dirty_data)} rows, {len(dirty_data.columns)} columns")
    
    return employees, departments, dirty_data


def demo_basic_transformations(engine, employees, departments):
    """Demonstrate basic transformation operations."""
    print("\n🔄 Demonstrating Basic Transformations")
    print("=" * 50)
    
    # 1. Filter Transformation
    print("\n1. Filter Transformation - Active employees with high performance")
    filter_config = TransformationConfig(
        name='filter_high_performers',
        type=TransformationType.FILTER,
        parameters={'condition': 'is_active == True and performance_score >= 4.0'}
    )
    
    filter_result = engine.execute_transformation(employees, filter_config)
    print(f"   Original rows: {len(employees)}")
    print(f"   Filtered rows: {filter_result.rows_output}")
    print(f"   Execution time: {filter_result.execution_time:.3f}s")
    print(f"   Filter selectivity: {filter_result.performance_metrics.get('filter_selectivity', 0):.2%}")
    
    # 2. Map Transformation
    print("\n2. Map Transformation - Calculate derived columns")
    map_config = TransformationConfig(
        name='calculate_derived_fields',
        type=TransformationType.MAP,
        parameters={
            'mappings': {
                'full_name': 'first_name + " " + last_name',
                'salary_k': 'salary / 1000',
                'years_of_service': '(pd.Timestamp.now() - pd.to_datetime(hire_date)).dt.days / 365.25',
                'performance_category': 'pd.cut(performance_score, bins=[0, 2, 3, 4, 5], labels=["Poor", "Fair", "Good", "Excellent"])'
            }
        }
    )
    
    map_result = engine.execute_transformation(employees.head(100), map_config)
    print(f"   Rows processed: {map_result.rows_processed}")
    print(f"   New columns added: {len(map_config.parameters['mappings'])}")
    print(f"   Execution time: {map_result.execution_time:.3f}s")
    if map_result.success:
        print(f"   Sample full name: {map_result.data['full_name'].iloc[0]}")
        print(f"   Sample salary (K): ${map_result.data['salary_k'].iloc[0]:.1f}K")
    
    # 3. Aggregate Transformation
    print("\n3. Aggregate Transformation - Department statistics")
    agg_config = TransformationConfig(
        name='department_statistics',
        type=TransformationType.AGGREGATE,
        parameters={
            'group_by': ['department_id'],
            'aggregations': {
                'salary': ['mean', 'median', 'std'],
                'age': ['mean', 'min', 'max'],
                'performance_score': 'mean',
                'employee_id': 'count'
            }
        }
    )
    
    agg_result = engine.execute_transformation(employees, agg_config)
    print(f"   Original rows: {agg_result.rows_processed}")
    print(f"   Aggregated rows: {agg_result.rows_output}")
    print(f"   Reduction ratio: {agg_result.performance_metrics.get('reduction_ratio', 0):.3f}")
    print(f"   Execution time: {agg_result.execution_time:.3f}s")
    
    # 4. Join Transformation
    print("\n4. Join Transformation - Enrich with department information")
    join_config = TransformationConfig(
        name='enrich_with_departments',
        type=TransformationType.JOIN,
        parameters={
            'right_data': departments,
            'join_keys': ['department_id'],
            'join_type': 'left'
        }
    )
    
    join_result = engine.execute_transformation(employees.head(100), join_config)
    print(f"   Left dataset rows: {join_result.rows_processed}")
    print(f"   Joined rows: {join_result.rows_output}")
    print(f"   Join ratio: {join_result.performance_metrics.get('join_ratio', 0):.2f}")
    print(f"   Execution time: {join_result.execution_time:.3f}s")
    if join_result.success:
        print(f"   Sample department: {join_result.data['department_name'].iloc[0]}")


def demo_data_type_conversion(engine, dirty_data):
    """Demonstrate data type conversion and validation."""
    print("\n🔧 Demonstrating Data Type Conversion")
    print("=" * 50)
    
    print("\nOriginal data types:")
    for col, dtype in dirty_data.dtypes.items():
        print(f"   {col}: {dtype}")
    
    # Test conversion validation first
    print("\n1. Validating conversions...")
    conversions_to_test = {
        'age_str': DataType.INTEGER,
        'salary_str': DataType.FLOAT,
        'join_date': DataType.DATETIME
    }
    
    for column, target_type in conversions_to_test.items():
        validation = engine.converter.validate_conversion(dirty_data[column], target_type)
        print(f"   {column} -> {target_type.value}:")
        print(f"     Convertible: {validation['convertible']}")
        print(f"     Success rate: {validation['success_rate']:.2%}")
        print(f"     Null count: {validation['null_count']}")
    
    # Perform actual conversions
    print("\n2. Performing conversions...")
    conversion_config = TransformationConfig(
        name='convert_data_types',
        type=TransformationType.CONVERT,
        parameters={
            'conversions': {
                'age_str': 'integer',
                'join_date': 'datetime'
            }
        }
    )
    
    conversion_result = engine.execute_transformation(dirty_data, conversion_config)
    print(f"   Conversion success: {conversion_result.success}")
    print(f"   Execution time: {conversion_result.execution_time:.3f}s")
    
    if conversion_result.success:
        print("\n   Converted data types:")
        for col, dtype in conversion_result.data.dtypes.items():
            print(f"     {col}: {dtype}")
        
        conversion_stats = conversion_result.performance_metrics.get('conversion_stats', {})
        for col, stats in conversion_stats.items():
            print(f"   {col} conversion: {stats['success_rate']:.2%} success rate")


def demo_custom_transformations(engine, employees):
    """Demonstrate custom transformation framework."""
    print("\n🛠️ Demonstrating Custom Transformations")
    print("=" * 50)
    
    # Register custom transformation functions
    def calculate_bonus(data, base_multiplier=0.1, performance_bonus=0.05, **kwargs):
        """Calculate employee bonus based on salary and performance."""
        result = data.copy()
        result['base_bonus'] = result['salary'] * base_multiplier
        result['performance_bonus'] = result['salary'] * performance_bonus * (result['performance_score'] - 3.0).clip(0)
        result['total_bonus'] = result['base_bonus'] + result['performance_bonus']
        return result
    
    def categorize_employees(data, **kwargs):
        """Categorize employees based on various factors."""
        result = data.copy()
        
        # Age categories
        result['age_category'] = pd.cut(
            result['age'], 
            bins=[0, 30, 40, 50, 100], 
            labels=['Young', 'Mid-Career', 'Senior', 'Veteran']
        )
        
        # Salary categories
        salary_percentiles = result['salary'].quantile([0.25, 0.5, 0.75])
        result['salary_tier'] = pd.cut(
            result['salary'],
            bins=[0, salary_percentiles[0.25], salary_percentiles[0.5], salary_percentiles[0.75], float('inf')],
            labels=['Entry', 'Mid', 'Senior', 'Executive']
        )
        
        return result
    
    def detect_outliers(data, columns=None, method='iqr', **kwargs):
        """Detect outliers in specified columns."""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
        
        result = data.copy()
        outlier_flags = pd.DataFrame(index=data.index)
        
        for col in columns:
            if col in data.columns:
                if method == 'iqr':
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outlier_flags[f'{col}_outlier'] = (data[col] < lower_bound) | (data[col] > upper_bound)
                elif method == 'zscore':
                    z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                    outlier_flags[f'{col}_outlier'] = z_scores > 3
        
        result = pd.concat([result, outlier_flags], axis=1)
        return result
    
    # Register custom transformations
    engine.register_custom_transformation('calculate_bonus', calculate_bonus)
    engine.register_custom_transformation('categorize_employees', categorize_employees)
    engine.register_custom_transformation('detect_outliers', detect_outliers)
    
    print("✅ Registered 3 custom transformations")
    
    # Test custom transformations
    sample_data = employees.head(200)
    
    # 1. Calculate bonus
    print("\n1. Custom Transformation: Calculate Bonus")
    bonus_config = TransformationConfig(
        name='calculate_employee_bonus',
        type=TransformationType.CUSTOM,
        parameters={
            'custom_name': 'calculate_bonus',
            'custom_params': {
                'base_multiplier': 0.12,
                'performance_bonus': 0.08
            }
        }
    )
    
    bonus_result = engine.execute_transformation(sample_data, bonus_config)
    print(f"   Success: {bonus_result.success}")
    print(f"   Execution time: {bonus_result.execution_time:.3f}s")
    if bonus_result.success:
        avg_bonus = bonus_result.data['total_bonus'].mean()
        print(f"   Average total bonus: ${avg_bonus:,.2f}")
    
    # 2. Categorize employees
    print("\n2. Custom Transformation: Categorize Employees")
    categorize_config = TransformationConfig(
        name='categorize_workforce',
        type=TransformationType.CUSTOM,
        parameters={
            'custom_name': 'categorize_employees',
            'custom_params': {}
        }
    )
    
    categorize_result = engine.execute_transformation(sample_data, categorize_config)
    print(f"   Success: {categorize_result.success}")
    print(f"   Execution time: {categorize_result.execution_time:.3f}s")
    if categorize_result.success:
        age_dist = categorize_result.data['age_category'].value_counts()
        print(f"   Age distribution: {dict(age_dist)}")
    
    # 3. Detect outliers
    print("\n3. Custom Transformation: Detect Outliers")
    outlier_config = TransformationConfig(
        name='detect_salary_outliers',
        type=TransformationType.CUSTOM,
        parameters={
            'custom_name': 'detect_outliers',
            'custom_params': {
                'columns': ['salary', 'performance_score'],
                'method': 'iqr'
            }
        }
    )
    
    outlier_result = engine.execute_transformation(sample_data, outlier_config)
    print(f"   Success: {outlier_result.success}")
    print(f"   Execution time: {outlier_result.execution_time:.3f}s")
    if outlier_result.success:
        salary_outliers = outlier_result.data['salary_outlier'].sum()
        performance_outliers = outlier_result.data['performance_score_outlier'].sum()
        print(f"   Salary outliers detected: {salary_outliers}")
        print(f"   Performance outliers detected: {performance_outliers}")


def demo_pipeline_execution(engine, employees, departments):
    """Demonstrate end-to-end pipeline execution."""
    print("\n🔄 Demonstrating Pipeline Execution")
    print("=" * 50)
    
    # Define comprehensive data pipeline
    pipeline = [
        # Step 1: Filter active employees
        TransformationConfig(
            name='filter_active_employees',
            type=TransformationType.FILTER,
            parameters={'condition': 'is_active == True'}
        ),
        
        # Step 2: Join with department information
        TransformationConfig(
            name='enrich_with_departments',
            type=TransformationType.JOIN,
            parameters={
                'right_data': departments,
                'join_keys': ['department_id'],
                'join_type': 'left'
            }
        ),
        
        # Step 3: Calculate derived metrics
        TransformationConfig(
            name='calculate_metrics',
            type=TransformationType.MAP,
            parameters={
                'mappings': {
                    'salary_k': 'salary / 1000',
                    'years_service': '(pd.Timestamp.now() - pd.to_datetime(hire_date)).dt.days / 365.25',
                    'efficiency_score': 'performance_score / (overtime_hours + 1)',
                    'compensation_ratio': 'salary / budget * 1000'
                }
            }
        ),
        
        # Step 4: Aggregate by department
        TransformationConfig(
            name='department_summary',
            type=TransformationType.AGGREGATE,
            parameters={
                'group_by': ['department_name'],
                'aggregations': {
                    'salary_k': ['mean', 'median', 'std'],
                    'years_service': 'mean',
                    'efficiency_score': 'mean',
                    'performance_score': 'mean',
                    'employee_id': 'count'
                }
            }
        )
    ]
    
    print(f"📋 Executing pipeline with {len(pipeline)} steps...")
    
    # Execute pipeline
    start_time = datetime.now()
    results = engine.execute_transformation_pipeline(employees, pipeline)
    total_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\n📊 Pipeline Results:")
    print(f"   Total execution time: {total_time:.3f}s")
    print(f"   Steps completed: {len(results)}")
    print(f"   Overall success: {all(r.success for r in results)}")
    
    # Detailed step results
    for i, (step, result) in enumerate(zip(pipeline, results), 1):
        status = "✅" if result.success else "❌"
        print(f"\n   Step {i}: {step.name} {status}")
        print(f"     Type: {step.type.value}")
        print(f"     Rows in: {result.rows_processed}")
        print(f"     Rows out: {result.rows_output}")
        print(f"     Time: {result.execution_time:.3f}s")
        
        if not result.success:
            print(f"     Error: {result.error_message}")
        elif result.performance_metrics:
            key_metrics = {k: v for k, v in result.performance_metrics.items() 
                          if k in ['filter_selectivity', 'join_ratio', 'reduction_ratio']}
            if key_metrics:
                print(f"     Metrics: {key_metrics}")
    
    # Show final results
    if results and results[-1].success:
        final_data = results[-1].data
        print(f"\n📈 Final Department Summary:")
        print(final_data.round(2).to_string(index=False))


def demo_performance_optimization(engine, employees):
    """Demonstrate performance optimization features."""
    print("\n⚡ Demonstrating Performance Optimization")
    print("=" * 50)
    
    # Create larger dataset for performance testing
    large_dataset = pd.concat([employees] * 5, ignore_index=True)  # 5x larger
    large_dataset['employee_id'] = range(len(large_dataset))
    
    print(f"📊 Testing with dataset: {len(large_dataset)} rows, {len(large_dataset.columns)} columns")
    print(f"   Memory usage: {large_dataset.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Test performance optimization recommendations
    should_parallel = engine.optimizer.should_use_parallel_processing(len(large_dataset), 'medium')
    chunk_size = engine.optimizer.get_optimal_chunk_size(len(large_dataset))
    
    print(f"\n🔧 Performance Recommendations:")
    print(f"   Use parallel processing: {should_parallel}")
    print(f"   Optimal chunk size: {chunk_size:,}")
    
    # Test DataFrame optimization
    print(f"\n🚀 DataFrame Optimization:")
    original_memory = large_dataset.memory_usage(deep=True).sum()
    optimized_dataset = engine.optimizer.optimize_dataframe_operations(large_dataset.copy())
    optimized_memory = optimized_dataset.memory_usage(deep=True).sum()
    
    memory_savings = (original_memory - optimized_memory) / original_memory * 100
    print(f"   Original memory: {original_memory / 1024 / 1024:.2f} MB")
    print(f"   Optimized memory: {optimized_memory / 1024 / 1024:.2f} MB")
    print(f"   Memory savings: {memory_savings:.1f}%")
    
    # Performance comparison
    print(f"\n⏱️ Performance Comparison:")
    
    # Test with original dataset
    test_config = TransformationConfig(
        name='performance_test',
        type=TransformationType.AGGREGATE,
        parameters={
            'group_by': ['department_id'],
            'aggregations': {
                'salary': ['mean', 'std'],
                'performance_score': 'mean',
                'employee_id': 'count'
            }
        }
    )
    
    # Original dataset
    start_time = datetime.now()
    result_original = engine.execute_transformation(large_dataset, test_config)
    time_original = (datetime.now() - start_time).total_seconds()
    
    # Optimized dataset
    start_time = datetime.now()
    result_optimized = engine.execute_transformation(optimized_dataset, test_config)
    time_optimized = (datetime.now() - start_time).total_seconds()
    
    speedup = (time_original - time_optimized) / time_original * 100 if time_original > 0 else 0
    
    print(f"   Original dataset: {time_original:.3f}s")
    print(f"   Optimized dataset: {time_optimized:.3f}s")
    print(f"   Performance improvement: {speedup:.1f}%")


def demo_intelligent_recommendations(engine, employees, dirty_data):
    """Demonstrate intelligent transformation recommendations."""
    print("\n🧠 Demonstrating Intelligent Recommendations")
    print("=" * 50)
    
    # Get recommendations for clean dataset
    print("\n1. Recommendations for Clean Dataset:")
    clean_recommendations = engine.get_transformation_recommendations(employees.head(500))
    
    print(f"   Total recommendations: {len(clean_recommendations)}")
    for i, rec in enumerate(clean_recommendations[:3], 1):  # Show first 3
        print(f"   {i}. {rec['type'].title()}: {rec.get('suggestion', rec.get('reason', 'N/A'))}")
    
    # Get recommendations for dirty dataset with target schema
    print("\n2. Recommendations for Dirty Dataset:")
    target_schema = {
        'id': 'integer',
        'name': 'string',
        'age_str': 'integer',
        'salary_str': 'float',
        'email': 'string',
        'join_date': 'datetime'
    }
    
    dirty_recommendations = engine.get_transformation_recommendations(dirty_data, target_schema)
    
    print(f"   Total recommendations: {len(dirty_recommendations)}")
    for i, rec in enumerate(dirty_recommendations, 1):
        if rec['type'] == 'conversion':
            print(f"   {i}. Convert {rec['column']}: {rec['from_type']} → {rec['to_type']} "
                  f"(confidence: {rec['confidence']:.2%})")
        elif rec['type'] == 'filter':
            print(f"   {i}. Filter: {rec['suggestion']} ({rec['reason']})")
        else:
            print(f"   {i}. {rec['type'].title()}: {rec.get('suggestion', rec.get('reason', 'N/A'))}")
    
    # Data profiling
    print("\n3. Data Profiling:")
    profile = engine._profile_data(dirty_data)
    
    print(f"   Dataset size: {profile['row_count']} rows, {profile['column_count']} columns")
    print(f"   Memory usage: {profile['memory_usage_mb']:.2f} MB")
    print(f"   Column types: {profile['column_types']}")
    
    print(f"\n   Data quality issues:")
    for col, stats in profile['column_stats'].items():
        if stats['null_percentage'] > 0:
            print(f"     {col}: {stats['null_percentage']:.1%} null values")
        if stats['unique_percentage'] < 0.1:
            print(f"     {col}: Low diversity ({stats['unique_percentage']:.1%} unique)")


def demo_performance_metrics(engine):
    """Demonstrate performance metrics and monitoring."""
    print("\n📊 Performance Metrics and Monitoring")
    print("=" * 50)
    
    # Get current performance metrics
    metrics = engine.get_performance_metrics()
    
    if not metrics:
        print("   No execution history available yet.")
        return
    
    print(f"📈 Execution Statistics:")
    print(f"   Total executions: {metrics['total_executions']}")
    print(f"   Successful executions: {metrics['successful_executions']}")
    print(f"   Failed executions: {metrics['failed_executions']}")
    print(f"   Success rate: {metrics['success_rate']:.2%}")
    print(f"   Average execution time: {metrics['average_execution_time']:.3f}s")
    print(f"   Total rows processed: {metrics['total_rows_processed']:,}")
    
    print(f"\n🔧 Transformation Types Used:")
    for transform_type in metrics['transformation_types']:
        count = sum(1 for h in engine.execution_history if h['transformation_type'] == transform_type)
        print(f"   {transform_type}: {count} executions")
    
    if metrics['recent_failures']:
        print(f"\n❌ Recent Failures:")
        for error in metrics['recent_failures']:
            print(f"   - {error}")
    
    # Show execution timeline
    print(f"\n⏰ Recent Execution Timeline:")
    recent_executions = engine.execution_history[-10:]  # Last 10 executions
    for execution in recent_executions:
        status = "✅" if execution['success'] else "❌"
        print(f"   {execution['timestamp'].strftime('%H:%M:%S')} - "
              f"{execution['transformation_name']} ({execution['transformation_type']}) "
              f"{status} - {execution['execution_time']:.3f}s")


def main():
    """Main demo function."""
    print("🚀 Data Pipeline Transformation Engine Demo")
    print("=" * 60)
    
    # Initialize transformation engine
    engine = TransformationEngine()
    
    # Create sample datasets
    employees, departments, dirty_data = create_sample_datasets()
    
    # Run demonstrations
    demo_basic_transformations(engine, employees, departments)
    demo_data_type_conversion(engine, dirty_data)
    demo_custom_transformations(engine, employees)
    demo_pipeline_execution(engine, employees, departments)
    demo_performance_optimization(engine, employees)
    demo_intelligent_recommendations(engine, employees, dirty_data)
    demo_performance_metrics(engine)
    
    print("\n🎉 Demo completed successfully!")
    print("\nKey Features Demonstrated:")
    print("✅ Filter, Map, Aggregate, and Join transformations")
    print("✅ Data type conversion and validation")
    print("✅ Custom transformation framework")
    print("✅ End-to-end pipeline execution")
    print("✅ Performance optimization")
    print("✅ Intelligent recommendations")
    print("✅ Performance monitoring and metrics")
    
    print(f"\nFinal Statistics:")
    final_metrics = engine.get_performance_metrics()
    if final_metrics:
        print(f"   Total transformations executed: {final_metrics['total_executions']}")
        print(f"   Overall success rate: {final_metrics['success_rate']:.2%}")
        print(f"   Total processing time: {sum(h['execution_time'] for h in engine.execution_history):.3f}s")
        print(f"   Total rows processed: {final_metrics['total_rows_processed']:,}")


if __name__ == "__main__":
    main()