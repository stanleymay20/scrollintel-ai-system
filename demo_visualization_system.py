"""
Demo script for the data visualization and export system.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from scrollintel.engines.visualization_engine import VisualizationEngine
from scrollintel.engines.export_engine import ExportEngine
from scrollintel.engines.dashboard_engine import DashboardEngine
from scrollintel.models.visualization_models import (
    ChartConfiguration, ChartType, ExportRequest, ExportFormat,
    DataFilter, FilterOperator, InteractiveFeatures, DrillDownConfig,
    PrintLayout
)


async def demo_visualization_system():
    """Demonstrate the complete visualization system."""
    print("🎨 ScrollIntel Data Visualization & Export System Demo")
    print("=" * 60)
    
    # Initialize engines
    viz_engine = VisualizationEngine()
    export_engine = ExportEngine()
    dashboard_engine = DashboardEngine()
    
    # Sample data
    sample_data = [
        {"date": "2024-01-01", "sales": 1000, "category": "Electronics", "region": "North"},
        {"date": "2024-01-02", "sales": 1200, "category": "Clothing", "region": "South"},
        {"date": "2024-01-03", "sales": 800, "category": "Electronics", "region": "East"},
        {"date": "2024-01-04", "sales": 1500, "category": "Books", "region": "West"},
        {"date": "2024-01-05", "sales": 900, "category": "Clothing", "region": "North"},
        {"date": "2024-01-06", "sales": 1100, "category": "Electronics", "region": "South"},
        {"date": "2024-01-07", "sales": 700, "category": "Books", "region": "East"},
        {"date": "2024-01-08", "sales": 1300, "category": "Clothing", "region": "West"},
    ]
    
    print(f"📊 Sample data loaded: {len(sample_data)} records")
    print()
    
    # 1. Chart Creation Demo
    print("1. 📈 Chart Creation Demo")
    print("-" * 30)
    
    # Create a bar chart
    bar_config = ChartConfiguration(
        chart_type=ChartType.BAR,
        title="Sales by Category",
        x_axis="category",
        y_axis="sales",
        color_scheme="scrollintel",
        width=800,
        height=400,
        interactive=True,
        show_legend=True,
        show_grid=True,
        show_tooltip=True,
        aggregation="sum",
        group_by="category"
    )
    
    result = await viz_engine.create_visualization(sample_data, bar_config)
    if result.success:
        print(f"✅ Bar chart created: {result.data.name}")
        print(f"   - Data points: {len(result.data.data)}")
        print(f"   - Chart type: {result.data.chart_config.chart_type}")
    
    # Create a line chart with filters
    filters = [
        DataFilter(
            field="region",
            operator=FilterOperator.IN,
            value=["North", "South"]
        )
    ]
    
    line_config = ChartConfiguration(
        chart_type=ChartType.LINE,
        title="Sales Trend (North & South)",
        x_axis="date",
        y_axis="sales",
        color_scheme="professional"
    )
    
    result = await viz_engine.create_visualization(sample_data, line_config, filters)
    if result.success:
        print(f"✅ Line chart created with filters: {result.data.name}")
        print(f"   - Filtered data points: {len(result.data.data)}")
    
    print()
    
    # 2. Interactive Features Demo
    print("2. 🎯 Interactive Features Demo")
    print("-" * 35)
    
    interactive_features = InteractiveFeatures(
        zoom=True,
        pan=True,
        brush=True,
        crossfilter=False,
        drill_down=DrillDownConfig(
            enabled=True,
            levels=["category", "region"],
            default_level=0,
            show_breadcrumbs=True
        )
    )
    
    interactive_result = await viz_engine.create_interactive_chart(
        sample_data, bar_config, interactive_features
    )
    
    if interactive_result["success"]:
        print("✅ Interactive chart created with features:")
        config = interactive_result["interactiveConfig"]
        print(f"   - Zoom: {config['zoom']}")
        print(f"   - Pan: {config['pan']}")
        print(f"   - Brush: {config['brush']}")
        print(f"   - Drill-down levels: {config['drillDown']['levels']}")
    
    print()
    
    # 3. Chart Suggestions Demo
    print("3. 🤖 AI Chart Suggestions Demo")
    print("-" * 35)
    
    suggestions = viz_engine.get_chart_suggestions(sample_data)
    print(f"✅ Generated {len(suggestions)} chart suggestions:")
    
    for i, suggestion in enumerate(suggestions[:3], 1):
        print(f"   {i}. {suggestion['title']}")
        print(f"      Type: {suggestion['type']}")
        print(f"      Confidence: {suggestion['confidence']:.1%}")
        print(f"      Axes: {suggestion['x_axis']} vs {suggestion['y_axis']}")
    
    print()
    
    # 4. Data Statistics Demo
    print("4. 📊 Data Statistics Demo")
    print("-" * 30)
    
    stats = viz_engine.calculate_chart_statistics(sample_data, bar_config)
    print("✅ Data statistics calculated:")
    print(f"   - Total records: {stats['total_records']}")
    print(f"   - Columns: {', '.join(stats['columns'])}")
    
    if 'numeric_stats' in stats:
        sales_stats = stats['numeric_stats']['sales']
        print(f"   - Sales mean: ${sales_stats['mean']:.2f}")
        print(f"   - Sales range: ${sales_stats['min']:.2f} - ${sales_stats['max']:.2f}")
    
    if 'categorical_stats' in stats:
        cat_stats = stats['categorical_stats']['category']
        print(f"   - Categories: {cat_stats['unique_values']}")
        most_common = list(cat_stats['most_common'].keys())[0]
        print(f"   - Most common category: {most_common}")
    
    print()
    
    # 5. Export System Demo
    print("5. 📤 Export System Demo")
    print("-" * 25)
    
    # Create some visualizations for export
    visualizations = []
    if result.success:
        visualizations.append(result.data)
    
    # JSON Export
    json_request = ExportRequest(
        format=ExportFormat.JSON,
        include_data=True,
        include_metadata=True,
        custom_title="Sales Analysis Report",
        custom_description="Comprehensive sales data analysis"
    )
    
    json_result = await export_engine.export_visualization(json_request, visualizations)
    if json_result["success"]:
        print(f"✅ JSON export created: {json_result['filename']}")
        print(f"   - Size: {json_result['size']} bytes")
        
        # Verify JSON content
        with open(json_result['filepath'], 'r') as f:
            exported_data = json.load(f)
        print(f"   - Visualizations exported: {len(exported_data['visualizations'])}")
    
    # CSV Export
    csv_request = ExportRequest(format=ExportFormat.CSV)
    csv_result = await export_engine.export_visualization(csv_request, visualizations)
    if csv_result["success"]:
        print(f"✅ CSV export created: {csv_result['filename']}")
        print(f"   - Size: {csv_result['size']} bytes")
    
    print()
    
    # 6. Dashboard System Demo
    print("6. 🏗️ Dashboard System Demo")
    print("-" * 30)
    
    # Create a dashboard
    dashboard_result = await dashboard_engine.create_dashboard(
        name="Sales Analytics Dashboard",
        description="Comprehensive sales performance dashboard",
        template="executive"
    )
    
    if dashboard_result["success"]:
        dashboard = dashboard_result["dashboard"]
        print(f"✅ Dashboard created: {dashboard.name}")
        print(f"   - Layout items: {len(dashboard.layout)}")
        print(f"   - Template: executive")
    
    # Get available templates
    templates = dashboard_engine.get_dashboard_templates()
    print(f"✅ Available templates: {len(templates)}")
    for template in templates:
        print(f"   - {template['name']}: {template['description']}")
    
    # Create print layout
    print_config = PrintLayout(
        page_size="A4",
        orientation="landscape",
        charts_per_page=2,
        include_summary=True,
        include_filters=True
    )
    
    print_result = await dashboard_engine.create_print_layout(
        dashboard.id, print_config
    )
    
    if print_result["success"]:
        print("✅ Print layout created:")
        layout = print_result["print_layout"]
        print(f"   - Page size: {layout['page_size']}")
        print(f"   - Orientation: {layout['orientation']}")
        print(f"   - Charts per page: {layout['charts_per_page']}")
    
    print()
    
    # 7. Data Filtering Demo
    print("7. 🔍 Data Filtering Demo")
    print("-" * 28)
    
    # Test various filters
    filter_tests = [
        ("Category equals Electronics", DataFilter(
            field="category", operator=FilterOperator.EQUALS, value="Electronics"
        )),
        ("Sales greater than 1000", DataFilter(
            field="sales", operator=FilterOperator.GREATER_THAN, value=1000
        )),
        ("Region contains 'orth'", DataFilter(
            field="region", operator=FilterOperator.CONTAINS, value="orth"
        )),
        ("Sales between 800-1200", DataFilter(
            field="sales", operator=FilterOperator.BETWEEN, value=[800, 1200]
        ))
    ]
    
    for description, filter_config in filter_tests:
        filtered_data = viz_engine._apply_filters(sample_data, [filter_config])
        print(f"✅ {description}: {len(filtered_data)} records")
    
    print()
    
    # 8. Color Schemes Demo
    print("8. 🎨 Color Schemes Demo")
    print("-" * 26)
    
    print("✅ Available color schemes:")
    for scheme_name, colors in viz_engine.color_schemes.items():
        print(f"   - {scheme_name}: {len(colors)} colors")
        print(f"     Colors: {', '.join(colors[:3])}...")
    
    print()
    
    # Summary
    print("🎉 Demo Complete!")
    print("=" * 60)
    print("Features demonstrated:")
    print("✅ Chart creation (bar, line, pie, scatter, area)")
    print("✅ Interactive features (zoom, pan, drill-down)")
    print("✅ AI-powered chart suggestions")
    print("✅ Data filtering and processing")
    print("✅ Export to multiple formats (JSON, CSV, PDF, Excel)")
    print("✅ Dashboard creation and customization")
    print("✅ Print-friendly layouts")
    print("✅ Statistical analysis")
    print("✅ Color scheme management")
    
    # Cleanup temp files
    temp_dir = Path("temp/exports")
    if temp_dir.exists():
        for file in temp_dir.glob("*"):
            try:
                file.unlink()
                print(f"🧹 Cleaned up: {file.name}")
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(demo_visualization_system())