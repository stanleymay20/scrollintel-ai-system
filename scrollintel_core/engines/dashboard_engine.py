"""
Dashboard Engine - Interactive visualization and dashboard creation
"""
import time
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DashboardEngine:
    """Dashboard Engine for creating interactive visualizations and dashboards"""
    
    def __init__(self):
        self.name = "Dashboard Engine"
        self.supported_chart_types = [
            "line", "bar", "area", "pie", "scatter", "heatmap", 
            "histogram", "box", "violin", "treemap", "gauge"
        ]
        self.supported_export_formats = ["png", "pdf", "svg", "json", "excel"]
    
    def create_dashboard(self, data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a complete dashboard with multiple visualizations"""
        try:
            start_time = time.time()
            
            # Convert data to DataFrame if needed
            df = self._prepare_data(data)
            
            if df is None or df.empty:
                return self._get_empty_dashboard_response()
            
            # Generate dashboard components
            dashboard_config = self._generate_dashboard_config(df, config)
            charts = self._create_charts(df, dashboard_config)
            metrics = self._calculate_key_metrics(df)
            insights = self._generate_insights(df)
            
            return {
                "success": True,
                "dashboard": {
                    "id": f"dashboard_{int(time.time())}",
                    "title": config.get("title", "Data Dashboard"),
                    "created_at": datetime.now().isoformat(),
                    "charts": charts,
                    "metrics": metrics,
                    "insights": insights,
                    "config": dashboard_config,
                    "data_summary": self._get_data_summary(df)
                },
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Dashboard creation error: {e}")
            return {
                "success": False,
                "error": str(e),
                "dashboard": None
            }
    
    def create_chart(self, data: Any, chart_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a single chart visualization"""
        try:
            df = self._prepare_data(data)
            
            if df is None or df.empty:
                return self._get_empty_chart_response()
            
            chart_type = chart_config.get("type", "line")
            
            if chart_type not in self.supported_chart_types:
                chart_type = self._recommend_chart_type(df)
            
            chart_data = self._generate_chart_data(df, chart_type, chart_config)
            
            return {
                "success": True,
                "chart": {
                    "id": f"chart_{int(time.time())}",
                    "type": chart_type,
                    "data": chart_data,
                    "config": self._get_chart_config(chart_type, chart_config),
                    "metadata": {
                        "data_points": len(df),
                        "columns": list(df.columns),
                        "created_at": datetime.now().isoformat()
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Chart creation error: {e}")
            return {
                "success": False,
                "error": str(e),
                "chart": None
            }
    
    def update_dashboard_realtime(self, dashboard_id: str, new_data: Any) -> Dict[str, Any]:
        """Update dashboard with new data for real-time updates"""
        try:
            df = self._prepare_data(new_data)
            
            if df is None or df.empty:
                return {"success": False, "error": "No valid data provided"}
            
            # Generate updated metrics and insights
            updated_metrics = self._calculate_key_metrics(df)
            updated_insights = self._generate_insights(df)
            
            return {
                "success": True,
                "dashboard_id": dashboard_id,
                "updates": {
                    "metrics": updated_metrics,
                    "insights": updated_insights,
                    "last_updated": datetime.now().isoformat(),
                    "data_points": len(df)
                }
            }
            
        except Exception as e:
            logger.error(f"Dashboard update error: {e}")
            return {"success": False, "error": str(e)}
    
    def export_dashboard(self, dashboard_data: Dict[str, Any], format: str = "json") -> Dict[str, Any]:
        """Export dashboard in specified format"""
        try:
            if format not in self.supported_export_formats:
                format = "json"
            
            if format == "json":
                return {
                    "success": True,
                    "export_data": dashboard_data,
                    "format": format,
                    "filename": f"dashboard_export_{int(time.time())}.json"
                }
            elif format == "excel":
                return self._export_to_excel(dashboard_data)
            else:
                return {
                    "success": True,
                    "message": f"Export to {format} format prepared",
                    "format": format,
                    "filename": f"dashboard_export_{int(time.time())}.{format}"
                }
                
        except Exception as e:
            logger.error(f"Dashboard export error: {e}")
            return {"success": False, "error": str(e)}
    
    def _prepare_data(self, data: Any) -> Optional[pd.DataFrame]:
        """Prepare data for dashboard creation"""
        try:
            if data is None:
                return None
            
            if isinstance(data, pd.DataFrame):
                return data
            elif isinstance(data, dict):
                return pd.DataFrame(data)
            elif isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], dict):
                    return pd.DataFrame(data)
                else:
                    return pd.DataFrame({"values": data})
            else:
                return pd.DataFrame({"values": [data]})
                
        except Exception as e:
            logger.error(f"Data preparation error: {e}")
            return None
    
    def _generate_dashboard_config(self, df: pd.DataFrame, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate dashboard configuration based on data characteristics"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        config = {
            "layout": user_config.get("layout", "grid"),
            "theme": user_config.get("theme", "light"),
            "auto_refresh": user_config.get("auto_refresh", False),
            "refresh_interval": user_config.get("refresh_interval", 30),
            "columns": {
                "numeric": numeric_columns,
                "categorical": categorical_columns,
                "datetime": datetime_columns
            },
            "recommended_charts": self._recommend_charts(df)
        }
        
        return config
    
    def _create_charts(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create multiple charts for the dashboard"""
        charts = []
        numeric_cols = config["columns"]["numeric"]
        categorical_cols = config["columns"]["categorical"]
        datetime_cols = config["columns"]["datetime"]
        
        # Time series chart if datetime column exists
        if datetime_cols and numeric_cols:
            charts.append(self._create_time_series_chart(df, datetime_cols[0], numeric_cols[0]))
        
        # Distribution charts for numeric columns
        for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            charts.append(self._create_distribution_chart(df, col))
        
        # Category analysis if categorical columns exist
        if categorical_cols and numeric_cols:
            charts.append(self._create_category_chart(df, categorical_cols[0], numeric_cols[0]))
        
        # Correlation heatmap if multiple numeric columns
        if len(numeric_cols) > 1:
            charts.append(self._create_correlation_chart(df, numeric_cols))
        
        return charts
    
    def _create_time_series_chart(self, df: pd.DataFrame, date_col: str, value_col: str) -> Dict[str, Any]:
        """Create time series line chart"""
        chart_data = []
        
        # Sort by date and prepare data
        df_sorted = df.sort_values(date_col)
        
        for _, row in df_sorted.iterrows():
            chart_data.append({
                "x": row[date_col].isoformat() if hasattr(row[date_col], 'isoformat') else str(row[date_col]),
                "y": float(row[value_col]) if pd.notna(row[value_col]) else 0
            })
        
        return {
            "id": f"timeseries_{int(time.time())}",
            "type": "line",
            "title": f"{value_col} Over Time",
            "data": chart_data,
            "config": {
                "xAxis": {"label": date_col, "type": "datetime"},
                "yAxis": {"label": value_col, "type": "numeric"},
                "responsive": True
            }
        }
    
    def _create_distribution_chart(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Create histogram for numeric column distribution"""
        values = df[column].dropna()
        
        # Create histogram bins
        hist, bin_edges = np.histogram(values, bins=20)
        
        chart_data = []
        for i in range(len(hist)):
            chart_data.append({
                "x": f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}",
                "y": int(hist[i])
            })
        
        return {
            "id": f"histogram_{column}_{int(time.time())}",
            "type": "bar",
            "title": f"Distribution of {column}",
            "data": chart_data,
            "config": {
                "xAxis": {"label": "Value Range", "type": "category"},
                "yAxis": {"label": "Frequency", "type": "numeric"},
                "responsive": True
            }
        }
    
    def _create_category_chart(self, df: pd.DataFrame, cat_col: str, num_col: str) -> Dict[str, Any]:
        """Create bar chart for categorical analysis"""
        grouped = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
        
        chart_data = []
        for category, value in grouped.items():
            chart_data.append({
                "x": str(category),
                "y": float(value)
            })
        
        return {
            "id": f"category_{cat_col}_{int(time.time())}",
            "type": "bar",
            "title": f"Average {num_col} by {cat_col}",
            "data": chart_data,
            "config": {
                "xAxis": {"label": cat_col, "type": "category"},
                "yAxis": {"label": f"Average {num_col}", "type": "numeric"},
                "responsive": True
            }
        }
    
    def _create_correlation_chart(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
        """Create correlation heatmap"""
        corr_matrix = df[numeric_cols].corr()
        
        chart_data = []
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                chart_data.append({
                    "x": col1,
                    "y": col2,
                    "value": float(corr_matrix.iloc[i, j])
                })
        
        return {
            "id": f"correlation_{int(time.time())}",
            "type": "heatmap",
            "title": "Correlation Matrix",
            "data": chart_data,
            "config": {
                "xAxis": {"label": "Variables", "type": "category"},
                "yAxis": {"label": "Variables", "type": "category"},
                "colorScale": {"min": -1, "max": 1},
                "responsive": True
            }
        }
    
    def _calculate_key_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate key metrics for the dashboard"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        metrics = {
            "total_records": len(df),
            "columns": len(df.columns),
            "numeric_columns": len(numeric_cols),
            "missing_values": int(df.isnull().sum().sum()),
            "data_quality_score": float((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100)
        }
        
        # Add metrics for numeric columns
        if len(numeric_cols) > 0:
            for col in numeric_cols[:5]:  # Limit to first 5 columns
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    metrics[f"{col}_mean"] = float(col_data.mean())
                    metrics[f"{col}_std"] = float(col_data.std())
                    metrics[f"{col}_min"] = float(col_data.min())
                    metrics[f"{col}_max"] = float(col_data.max())
        
        return metrics
    
    def _generate_insights(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate insights from the data"""
        insights = []
        
        # Data quality insights
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct > 10:
            insights.append({
                "type": "warning",
                "title": "Data Quality Issue",
                "message": f"Dataset has {missing_pct:.1f}% missing values. Consider data cleaning.",
                "priority": "high"
            })
        
        # Numeric column insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:3]:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                cv = col_data.std() / col_data.mean() if col_data.mean() != 0 else 0
                if cv > 1:
                    insights.append({
                        "type": "info",
                        "title": f"High Variability in {col}",
                        "message": f"Column {col} shows high variability (CV: {cv:.2f}). Consider investigating outliers.",
                        "priority": "medium"
                    })
        
        # Size insights
        if len(df) < 100:
            insights.append({
                "type": "warning",
                "title": "Small Dataset",
                "message": f"Dataset has only {len(df)} records. Consider collecting more data for robust analysis.",
                "priority": "medium"
            })
        elif len(df) > 10000:
            insights.append({
                "type": "success",
                "title": "Large Dataset",
                "message": f"Dataset has {len(df)} records. Good sample size for analysis.",
                "priority": "low"
            })
        
        return insights
    
    def _recommend_chart_type(self, df: pd.DataFrame) -> str:
        """Recommend chart type based on data characteristics"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        if len(datetime_cols) > 0 and len(numeric_cols) > 0:
            return "line"  # Time series
        elif len(categorical_cols) > 0 and len(numeric_cols) > 0:
            return "bar"   # Category analysis
        elif len(numeric_cols) >= 2:
            return "scatter"  # Correlation
        elif len(numeric_cols) == 1:
            return "histogram"  # Distribution
        else:
            return "bar"  # Default
    
    def _recommend_charts(self, df: pd.DataFrame) -> List[str]:
        """Recommend chart types for the dashboard"""
        recommendations = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        if len(datetime_cols) > 0 and len(numeric_cols) > 0:
            recommendations.append("line")
        
        if len(numeric_cols) > 0:
            recommendations.append("histogram")
        
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            recommendations.append("bar")
        
        if len(numeric_cols) >= 2:
            recommendations.extend(["scatter", "heatmap"])
        
        return recommendations
    
    def _get_chart_config(self, chart_type: str, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get chart configuration"""
        base_config = {
            "responsive": True,
            "maintainAspectRatio": False,
            "animation": {"duration": 1000}
        }
        
        # Merge with user config
        base_config.update(user_config.get("chart_config", {}))
        
        return base_config
    
    def _get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of the data"""
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "column_types": {
                "numeric": len(df.select_dtypes(include=[np.number]).columns),
                "categorical": len(df.select_dtypes(include=['object', 'category']).columns),
                "datetime": len(df.select_dtypes(include=['datetime64']).columns)
            },
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"
        }
    
    def _export_to_excel(self, dashboard_data: Dict[str, Any]) -> Dict[str, Any]:
        """Export dashboard data to Excel format"""
        try:
            # This would typically create an actual Excel file
            # For now, return the structure
            return {
                "success": True,
                "message": "Excel export prepared",
                "format": "excel",
                "filename": f"dashboard_export_{int(time.time())}.xlsx",
                "sheets": {
                    "Dashboard_Summary": "Dashboard metadata and metrics",
                    "Chart_Data": "All chart data combined",
                    "Insights": "Generated insights and recommendations"
                }
            }
        except Exception as e:
            logger.error(f"Excel export error: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_empty_dashboard_response(self) -> Dict[str, Any]:
        """Return empty dashboard response"""
        return {
            "success": False,
            "error": "No valid data provided",
            "dashboard": {
                "charts": [],
                "metrics": {},
                "insights": [{
                    "type": "warning",
                    "title": "No Data",
                    "message": "Please provide valid data to create dashboard",
                    "priority": "high"
                }]
            }
        }
    
    def _get_empty_chart_response(self) -> Dict[str, Any]:
        """Return empty chart response"""
        return {
            "success": False,
            "error": "No valid data provided",
            "chart": None
        }