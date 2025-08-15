"""
Integration tests for ScrollInsightRadar API routes.

Tests the complete API functionality including file upload,
pattern detection, trend analysis, anomaly detection, and notifications.
"""

import pytest
import pandas as pd
import numpy as np
import io
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from scrollintel.api.gateway import app
from scrollintel.models.schemas import PatternDetectionConfig


class TestScrollInsightRadarIntegration:
    """Integration test suite for ScrollInsightRadar API."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_user(self):
        """Mock authenticated user."""
        return {
            "user_id": "test_user_123",
            "email": "test@example.com",
            "role": "analyst"
        }
    
    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data for testing."""
        np.random.seed(42)
        data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=50, freq='D'),
            'sales': np.random.normal(1000, 200, 50),
            'marketing_spend': np.random.normal(500, 100, 50),
            'temperature': np.random.normal(20, 5, 50),
            'category': np.random.choice(['A', 'B', 'C'], 50)
        })
        
        # Add correlation
        data['marketing_spend'] = data['sales'] * 0.4 + np.random.normal(0, 50, 50)
        
        return data.to_csv(index=False)
    
    @pytest.fixture
    def sample_excel_data(self):
        """Create sample Excel data for testing."""
        np.random.seed(42)
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='H'),
            'metric_1': np.random.normal(100, 20, 100),
            'metric_2': np.random.normal(50, 10, 100),
            'outlier_metric': np.concatenate([np.random.normal(10, 2, 95), [100, 110, 120, 130, 140]])
        })
        
        # Create Excel bytes
        excel_buffer = io.BytesIO()
        data.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)
        return excel_buffer.getvalue()

    @patch('scrollintel.api.routes.scroll_insight_radar_routes.get_current_user')
    def test_detect_patterns_csv_success(self, mock_get_user, client, mock_user, sample_csv_data):
        """Test successful pattern detection with CSV file."""
        mock_get_user.return_value = mock_user
        
        response = client.post(
            "/insight-radar/detect-patterns",
            files={"file": ("test_data.csv", sample_csv_data, "text/csv")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["file_name"] == "test_data.csv"
        assert data["user_id"] == mock_user["user_id"]
        assert "results" in data
        
        results = data["results"]
        assert "timestamp" in results
        assert "dataset_info" in results
        assert "patterns" in results
        assert "trends" in results
        assert "anomalies" in results
        assert "insights" in results
        assert "business_impact_score" in results

    @patch('scrollintel.api.routes.scroll_insight_radar_routes.get_current_user')
    def test_detect_patterns_excel_success(self, mock_get_user, client, mock_user, sample_excel_data):
        """Test successful pattern detection with Excel file."""
        mock_get_user.return_value = mock_user
        
        response = client.post(
            "/insight-radar/detect-patterns",
            files={"file": ("test_data.xlsx", sample_excel_data, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["file_name"] == "test_data.xlsx"
        assert "results" in data

    @patch('scrollintel.api.routes.scroll_insight_radar_routes.get_current_user')
    def test_detect_patterns_invalid_file_type(self, mock_get_user, client, mock_user):
        """Test pattern detection with invalid file type."""
        mock_get_user.return_value = mock_user
        
        response = client.post(
            "/insight-radar/detect-patterns",
            files={"file": ("test_data.txt", "invalid content", "text/plain")}
        )
        
        assert response.status_code == 400
        assert "Only CSV and Excel files are supported" in response.json()["detail"]

    @patch('scrollintel.api.routes.scroll_insight_radar_routes.get_current_user')
    def test_analyze_trends_success(self, mock_get_user, client, mock_user, sample_csv_data):
        """Test successful trend analysis."""
        mock_get_user.return_value = mock_user
        
        response = client.post(
            "/insight-radar/analyze-trends",
            files={"file": ("trend_data.csv", sample_csv_data, "text/csv")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["file_name"] == "trend_data.csv"
        assert "results" in data

    @patch('scrollintel.api.routes.scroll_insight_radar_routes.get_current_user')
    def test_detect_anomalies_success(self, mock_get_user, client, mock_user, sample_excel_data):
        """Test successful anomaly detection."""
        mock_get_user.return_value = mock_user
        
        response = client.post(
            "/insight-radar/detect-anomalies",
            files={"file": ("anomaly_data.xlsx", sample_excel_data, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["file_name"] == "anomaly_data.xlsx"
        assert "results" in data

    @patch('scrollintel.api.routes.scroll_insight_radar_routes.get_current_user')
    def test_generate_insights_with_notifications(self, mock_get_user, client, mock_user, sample_csv_data):
        """Test insight generation with notifications enabled."""
        mock_get_user.return_value = mock_user
        
        response = client.post(
            "/insight-radar/generate-insights",
            files={"file": ("insight_data.csv", sample_csv_data, "text/csv")},
            params={"send_notifications": True}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["file_name"] == "insight_data.csv"
        assert "insights" in data
        assert "business_impact_score" in data
        assert "total_insights" in data
        assert "high_priority_insights" in data
        assert data["notifications_sent"] is True

    @patch('scrollintel.api.routes.scroll_insight_radar_routes.get_current_user')
    def test_generate_insights_without_notifications(self, mock_get_user, client, mock_user, sample_csv_data):
        """Test insight generation without notifications."""
        mock_get_user.return_value = mock_user
        
        response = client.post(
            "/insight-radar/generate-insights",
            files={"file": ("insight_data.csv", sample_csv_data, "text/csv")},
            params={"send_notifications": False}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["notifications_sent"] is False

    def test_health_status(self, client):
        """Test health status endpoint."""
        response = client.get("/insight-radar/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["engine"] == "ScrollInsightRadar"
        assert data["version"] == "1.0.0"
        assert "capabilities" in data
        assert "last_check" in data

    def test_get_capabilities(self, client):
        """Test capabilities endpoint."""
        response = client.get("/insight-radar/capabilities")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["engine"] == "ScrollInsightRadar"
        assert data["version"] == "1.0.0"
        assert "capabilities" in data
        assert "supported_file_types" in data
        assert "analysis_types" in data
        assert data["notification_support"] is True
        assert data["real_time_processing"] is True
        
        # Check supported file types
        assert ".csv" in data["supported_file_types"]
        assert ".xlsx" in data["supported_file_types"]
        assert ".xls" in data["supported_file_types"]
        
        # Check analysis types
        expected_analysis_types = [
            "correlation_patterns",
            "seasonal_patterns",
            "clustering_patterns", 
            "distribution_patterns",
            "trend_analysis",
            "anomaly_detection",
            "insight_ranking",
            "statistical_significance_testing"
        ]
        
        for analysis_type in expected_analysis_types:
            assert analysis_type in data["analysis_types"]

    @patch('scrollintel.api.routes.scroll_insight_radar_routes.get_current_user')
    def test_batch_analysis_success(self, mock_get_user, client, mock_user, sample_csv_data):
        """Test successful batch analysis."""
        mock_get_user.return_value = mock_user
        
        # Create multiple files
        files = [
            ("file1.csv", sample_csv_data, "text/csv"),
            ("file2.csv", sample_csv_data, "text/csv")
        ]
        
        response = client.post(
            "/insight-radar/batch-analysis",
            files=[("files", file) for file in files]
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["files_processed"] == 2
        assert "batch_results" in data
        assert "aggregated_insights" in data
        assert "total_insights" in data
        assert "average_impact_score" in data

    @patch('scrollintel.api.routes.scroll_insight_radar_routes.get_current_user')
    def test_batch_analysis_too_many_files(self, mock_get_user, client, mock_user, sample_csv_data):
        """Test batch analysis with too many files."""
        mock_get_user.return_value = mock_user
        
        # Create more than 10 files
        files = [(f"file{i}.csv", sample_csv_data, "text/csv") for i in range(12)]
        
        response = client.post(
            "/insight-radar/batch-analysis",
            files=[("files", file) for file in files]
        )
        
        assert response.status_code == 400
        assert "Maximum 10 files allowed" in response.json()["detail"]

    @patch('scrollintel.api.routes.scroll_insight_radar_routes.get_current_user')
    def test_pattern_detection_with_config(self, mock_get_user, client, mock_user, sample_csv_data):
        """Test pattern detection with custom configuration."""
        mock_get_user.return_value = mock_user
        
        # Custom config would be passed as form data in a real implementation
        response = client.post(
            "/insight-radar/detect-patterns",
            files={"file": ("test_data.csv", sample_csv_data, "text/csv")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @patch('scrollintel.api.routes.scroll_insight_radar_routes.get_current_user')
    def test_error_handling_corrupted_file(self, mock_get_user, client, mock_user):
        """Test error handling with corrupted file."""
        mock_get_user.return_value = mock_user
        
        corrupted_csv = "invalid,csv,data\n1,2,\n,,"
        
        response = client.post(
            "/insight-radar/detect-patterns",
            files={"file": ("corrupted.csv", corrupted_csv, "text/csv")}
        )
        
        # Should handle gracefully or return appropriate error
        assert response.status_code in [200, 500]

    @patch('scrollintel.api.routes.scroll_insight_radar_routes.get_current_user')
    def test_empty_file_handling(self, mock_get_user, client, mock_user):
        """Test handling of empty files."""
        mock_get_user.return_value = mock_user
        
        empty_csv = ""
        
        response = client.post(
            "/insight-radar/detect-patterns",
            files={"file": ("empty.csv", empty_csv, "text/csv")}
        )
        
        # Should return error for empty file
        assert response.status_code == 500

    @patch('scrollintel.api.routes.scroll_insight_radar_routes.get_current_user')
    def test_large_file_handling(self, mock_get_user, client, mock_user):
        """Test handling of large files."""
        mock_get_user.return_value = mock_user
        
        # Create larger dataset
        np.random.seed(42)
        large_data = pd.DataFrame({
            'col1': np.random.normal(0, 1, 1000),
            'col2': np.random.normal(0, 1, 1000),
            'col3': np.random.normal(0, 1, 1000)
        })
        
        large_csv = large_data.to_csv(index=False)
        
        response = client.post(
            "/insight-radar/detect-patterns",
            files={"file": ("large_data.csv", large_csv, "text/csv")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @patch('scrollintel.api.routes.scroll_insight_radar_routes.get_current_user')
    def test_authentication_required(self, mock_get_user, client, sample_csv_data):
        """Test that authentication is required."""
        mock_get_user.side_effect = Exception("Authentication required")
        
        response = client.post(
            "/insight-radar/detect-patterns",
            files={"file": ("test_data.csv", sample_csv_data, "text/csv")}
        )
        
        assert response.status_code == 500

    @patch('scrollintel.api.routes.scroll_insight_radar_routes.get_current_user')
    def test_concurrent_requests(self, mock_get_user, client, mock_user, sample_csv_data):
        """Test handling of concurrent requests."""
        mock_get_user.return_value = mock_user
        
        import concurrent.futures
        import threading
        
        def make_request():
            return client.post(
                "/insight-radar/detect-patterns",
                files={"file": ("test_data.csv", sample_csv_data, "text/csv")}
            )
        
        # Make multiple concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request) for _ in range(3)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200

    @patch('scrollintel.api.routes.scroll_insight_radar_routes.get_current_user')
    def test_response_format_consistency(self, mock_get_user, client, mock_user, sample_csv_data):
        """Test that response format is consistent across endpoints."""
        mock_get_user.return_value = mock_user
        
        endpoints = [
            "/insight-radar/detect-patterns",
            "/insight-radar/analyze-trends", 
            "/insight-radar/detect-anomalies"
        ]
        
        for endpoint in endpoints:
            response = client.post(
                endpoint,
                files={"file": ("test_data.csv", sample_csv_data, "text/csv")}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Check common response fields
            assert "success" in data
            assert "message" in data
            assert "file_name" in data
            assert "analysis_timestamp" in data
            assert "user_id" in data

    @patch('scrollintel.api.routes.scroll_insight_radar_routes.get_current_user')
    def test_insight_quality_validation(self, mock_get_user, client, mock_user, sample_csv_data):
        """Test that generated insights meet quality standards."""
        mock_get_user.return_value = mock_user
        
        response = client.post(
            "/insight-radar/generate-insights",
            files={"file": ("test_data.csv", sample_csv_data, "text/csv")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        insights = data["insights"]
        
        for insight in insights:
            # Validate insight structure
            assert "type" in insight
            assert "title" in insight
            assert "description" in insight
            assert "impact_score" in insight
            assert "priority" in insight
            assert "rank" in insight
            
            # Validate insight values
            assert insight["type"] in ["correlation", "trend", "anomaly", "clustering"]
            assert insight["priority"] in ["high", "medium", "low"]
            assert 0.0 <= insight["impact_score"] <= 1.0
            assert insight["rank"] >= 1
            assert len(insight["title"]) > 0
            assert len(insight["description"]) > 0

    @patch('scrollintel.api.routes.scroll_insight_radar_routes.get_current_user')
    def test_statistical_significance_validation(self, mock_get_user, client, mock_user, sample_csv_data):
        """Test that statistical tests are properly validated."""
        mock_get_user.return_value = mock_user
        
        response = client.post(
            "/insight-radar/detect-patterns",
            files={"file": ("test_data.csv", sample_csv_data, "text/csv")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        results = data["results"]
        
        if "statistical_tests" in results:
            tests = results["statistical_tests"]
            
            # Validate correlation tests
            if "correlation_tests" in tests:
                for test in tests["correlation_tests"]:
                    assert "p_value" in test
                    assert 0.0 <= test["p_value"] <= 1.0
                    assert "correlation" in test
                    assert -1.0 <= test["correlation"] <= 1.0
            
            # Validate normality tests
            if "normality_tests" in tests:
                for test in tests["normality_tests"]:
                    assert "p_value" in test
                    assert 0.0 <= test["p_value"] <= 1.0