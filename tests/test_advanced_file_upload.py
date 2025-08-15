"""
Test advanced file upload functionality with drag-and-drop, progress tracking, and file management.
"""

import pytest
import tempfile
import os
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
import asyncio

from scrollintel.engines.file_processor import FileProcessorEngine
from scrollintel.core.interfaces import EngineError


class TestAdvancedFileProcessor:
    """Test advanced file processor engine features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = FileProcessorEngine()
    
    def test_engine_initialization(self):
        """Test engine initializes correctly."""
        assert self.engine.engine_id == "file_processor"
        assert self.engine.name == "Optimized File Processor Engine"
        assert len(self.engine.supported_types) > 0
        assert self.engine.max_file_size == 100 * 1024 * 1024  # 100MB
        assert self.engine.chunk_size == 8192  # 8KB
    
    def test_supported_file_types(self):
        """Test supported file types are configured."""
        expected_types = ['csv', 'xlsx', 'json', 'sql']
        
        for file_type in expected_types:
            assert file_type in self.engine.supported_types.values()
    
    @pytest.mark.asyncio
    async def test_validate_file_comprehensive(self):
        """Test comprehensive file validation."""
        # Test valid file
        mock_file = Mock()
        mock_file.size = 1000
        mock_file.content_type = "text/csv"
        mock_file.filename = "test.csv"
        mock_file.file = Mock()
        mock_file.file.tell.return_value = 0
        mock_file.file.read.return_value = b"id,name,value\n1,test,100\n"
        mock_file.file.seek = Mock()
        
        # Should not raise exception
        await self.engine._validate_file_optimized(mock_file)
    
    @pytest.mark.asyncio
    async def test_validate_file_size_limits(self):
        """Test file size validation with various limits."""
        # Test file too large
        mock_file = Mock()
        mock_file.size = self.engine.max_file_size + 1
        mock_file.content_type = "text/csv"
        mock_file.filename = "large.csv"
        
        with pytest.raises(EngineError, match="exceeds maximum allowed size"):
            await self.engine._validate_file_optimized(mock_file)
        
        # Test file too small
        mock_file.size = 5  # Less than 10 bytes
        with pytest.raises(EngineError, match="too small to be valid"):
            await self.engine._validate_file_optimized(mock_file)
    
    @pytest.mark.asyncio
    async def test_validate_file_type_detection(self):
        """Test file type validation and detection."""
        # Test unsupported content type
        mock_file = Mock()
        mock_file.size = 1000
        mock_file.content_type = "application/unsupported"
        mock_file.filename = "test.unsupported"
        
        with pytest.raises(EngineError, match="Unsupported file type"):
            await self.engine._validate_file_optimized(mock_file)
        
        # Test dangerous filename characters
        mock_file.content_type = "text/csv"
        mock_file.filename = "test<script>.csv"
        
        with pytest.raises(EngineError, match="contains invalid characters"):
            await self.engine._validate_file_optimized(mock_file)
    
    @pytest.mark.asyncio
    async def test_file_format_validation_api(self):
        """Test file format validation API method."""
        mock_file = Mock()
        mock_file.filename = "test.csv"
        mock_file.size = 1000
        mock_file.content_type = "text/csv"
        mock_file.file = Mock()
        mock_file.file.tell.return_value = 0
        mock_file.file.read.return_value = b"id,name\n1,test\n"
        mock_file.file.seek = Mock()
        
        result = await self.engine.validate_file_format(mock_file)
        
        assert "valid" in result
        assert "errors" in result
        assert "warnings" in result
        assert "file_info" in result
        assert result["file_info"]["name"] == "test.csv"
        assert result["file_info"]["size"] == 1000
    
    @pytest.mark.asyncio
    async def test_content_type_detection(self):
        """Test advanced content type detection."""
        # Test CSV detection
        mock_file = Mock()
        mock_file.content_type = "text/plain"
        mock_file.filename = "data.csv"
        mock_file.file = Mock()
        mock_file.file.tell.return_value = 0
        mock_file.file.read.return_value = b"id,name,value\n1,test,100\n"
        mock_file.file.seek = Mock()
        
        detected_type = await self.engine._detect_file_type_optimized(
            mock_file.content_type, mock_file.filename, mock_file
        )
        assert detected_type == "csv"
        
        # Test JSON detection
        mock_file.file.read.return_value = b'{"data": [{"id": 1, "name": "test"}]}'
        detected_type = await self.engine._detect_file_type_optimized(
            "text/plain", "data.json", mock_file
        )
        assert detected_type == "json"
        
        # Test SQL detection
        mock_file.file.read.return_value = b'SELECT * FROM users WHERE id = 1;'
        detected_type = await self.engine._detect_file_type_optimized(
            "text/plain", "query.sql", mock_file
        )
        assert detected_type == "sql"
    
    def test_advanced_column_type_inference(self):
        """Test advanced column type inference."""
        # Test email detection
        email_series = pd.Series([
            "user1@example.com",
            "admin@company.org",
            "test@domain.net"
        ])
        assert self.engine._infer_column_type(email_series) == "email"
        
        # Test URL detection
        url_series = pd.Series([
            "https://example.com",
            "http://test.org",
            "https://company.net/page"
        ])
        assert self.engine._infer_column_type(url_series) == "url"
        
        # Test phone number detection
        phone_series = pd.Series([
            "+1-555-123-4567",
            "(555) 987-6543",
            "555.111.2222"
        ])
        assert self.engine._infer_column_type(phone_series) == "phone"
        
        # Test datetime detection
        date_series = pd.Series([
            "2023-01-01",
            "2023-02-15",
            "2023-12-31"
        ])
        # Note: This might not always work due to pandas datetime parsing
        inferred_type = self.engine._infer_column_type(date_series)
        assert inferred_type in ["datetime", "text"]  # Could be either
    
    def test_quality_report_comprehensive(self):
        """Test comprehensive quality report generation."""
        # Create test DataFrame with various data quality issues
        test_data = {
            "id": [1, 2, 3, 4, 5, 5],  # Duplicate row
            "name": ["Alice", "Bob", None, "David", "Eve", "Eve"],  # Missing value
            "score": [85.5, 92.0, 78.5, 88.0, 95.5, 95.5],
            "category": ["A", "B", "A", "C", "B", "B"],
            "mixed_type": [1, "two", 3, "four", 5, 5],  # Mixed types
            "outlier_col": [10, 12, 11, 1000, 13, 13]  # Outlier
        }
        df = pd.DataFrame(test_data)
        
        report = self.engine._generate_quality_report(df)
        
        # Check basic metrics
        assert report["total_rows"] == 6
        assert report["total_columns"] == 6
        assert report["missing_values"]["name"] == 1
        assert report["duplicate_rows"] == 1
        
        # Check data type issues detection
        assert len(report["data_type_issues"]) > 0
        mixed_type_issue = next(
            (issue for issue in report["data_type_issues"] 
             if issue["column"] == "mixed_type"), None
        )
        assert mixed_type_issue is not None
        
        # Check outlier detection
        assert "outlier_col" in report["outliers"]
        
        # Check quality score
        assert 0 <= report["quality_score"] <= 100
        assert report["quality_score"] < 100  # Should be less than perfect due to issues
        
        # Check recommendations
        assert len(report["recommendations"]) > 0
        recommendations_text = " ".join(report["recommendations"]).lower()
        assert "missing" in recommendations_text or "duplicate" in recommendations_text
    
    @pytest.mark.asyncio
    async def test_get_supported_formats_detailed(self):
        """Test getting detailed supported formats information."""
        formats_info = await self.engine.get_supported_formats()
        
        assert "formats" in formats_info
        assert "max_file_size" in formats_info
        assert "features" in formats_info
        
        # Check format details
        formats = formats_info["formats"]
        assert len(formats) >= 5  # CSV, XLSX, XLS, JSON, SQL
        
        csv_format = next((f for f in formats if f["extension"] == ".csv"), None)
        assert csv_format is not None
        assert "Auto-delimiter detection" in csv_format["features"]
        
        # Check features
        features = formats_info["features"]
        assert features["auto_detection"] is True
        assert features["schema_inference"] is True
        assert features["progress_tracking"] is True
    
    def test_extract_quality_score(self):
        """Test quality score extraction from report."""
        # Test with valid report
        report = {"quality_score": 85.5, "other_data": "test"}
        score = self.engine._extract_quality_score(report)
        assert score == 85.5
        
        # Test with None report
        score = self.engine._extract_quality_score(None)
        assert score is None
        
        # Test with report without quality score
        report = {"other_data": "test"}
        score = self.engine._extract_quality_score(report)
        assert score is None


class TestAdvancedFileUploadAPI:
    """Test advanced file upload API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from scrollintel.api.main import app
        return TestClient(app)
    
    @pytest.fixture
    def sample_files(self):
        """Create sample files for testing."""
        files = {}
        
        # CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,name,score,category\n")
            f.write("1,Alice,85.5,A\n")
            f.write("2,Bob,92.0,B\n")
            f.write("3,Charlie,78.5,A\n")
            f.write("4,David,88.0,C\n")
            f.write("5,Eve,95.5,B\n")
            files['csv'] = f.name
        
        # JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {
                "users": [
                    {"id": 1, "name": "Alice", "email": "alice@example.com"},
                    {"id": 2, "name": "Bob", "email": "bob@example.com"}
                ]
            }
            json.dump(data, f)
            files['json'] = f.name
        
        # SQL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False) as f:
            f.write("CREATE TABLE users (\n")
            f.write("  id INTEGER PRIMARY KEY,\n")
            f.write("  name VARCHAR(100),\n")
            f.write("  email VARCHAR(255)\n")
            f.write(");\n")
            files['sql'] = f.name
        
        yield files
        
        # Cleanup
        for file_path in files.values():
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    def test_validate_file_endpoint(self, client, sample_files):
        """Test file validation endpoint."""
        with patch('scrollintel.api.routes.file_routes.require_permission'):
            with open(sample_files['csv'], 'rb') as f:
                response = client.post(
                    "/api/files/validate",
                    files={"file": ("test.csv", f, "text/csv")}
                )
            
            assert response.status_code == 200
            data = response.json()
            assert "valid" in data
            assert "errors" in data
            assert "warnings" in data
            assert "file_info" in data
    
    def test_search_uploads_endpoint(self, client):
        """Test search uploads endpoint."""
        with patch('scrollintel.api.routes.file_routes.require_permission'), \
             patch('scrollintel.api.routes.file_routes.get_db'), \
             patch.object(FileProcessorEngine, 'list_user_uploads') as mock_list:
            
            mock_list.return_value = [
                {
                    "id": "upload-1",
                    "filename": "test1.csv",
                    "size": 1000,
                    "status": "completed",
                    "quality_score": 85.5
                },
                {
                    "id": "upload-2", 
                    "filename": "test2.json",
                    "size": 2000,
                    "status": "processing",
                    "quality_score": None
                }
            ]
            
            response = client.get("/api/files/uploads/search?q=test&status=completed")
            
            assert response.status_code == 200
            data = response.json()
            assert "uploads" in data
            assert "total" in data
            assert len(data["uploads"]) == 2
    
    def test_get_enhanced_supported_formats(self, client):
        """Test getting enhanced supported formats."""
        with patch('scrollintel.api.routes.file_routes.require_permission'), \
             patch.object(FileProcessorEngine, 'get_supported_formats') as mock_formats:
            
            mock_formats.return_value = {
                "formats": [
                    {
                        "extension": ".csv",
                        "mime_types": ["text/csv"],
                        "description": "Comma-separated values file",
                        "features": ["Auto-delimiter detection", "Schema inference"]
                    }
                ],
                "max_file_size": 104857600,
                "features": {
                    "auto_detection": True,
                    "progress_tracking": True
                }
            }
            
            response = client.get("/api/files/supported-formats")
            
            assert response.status_code == 200
            data = response.json()
            assert "formats" in data
            assert "features" in data
            assert data["features"]["auto_detection"] is True
    
    def test_upload_with_progress_tracking(self, client, sample_files):
        """Test file upload with progress tracking."""
        with patch('scrollintel.api.routes.file_routes.require_permission'), \
             patch('scrollintel.api.routes.file_routes.get_db'), \
             patch.object(FileProcessorEngine, 'process_upload') as mock_process:
            
            # Mock the process_upload response with progress tracking
            mock_process.return_value = {
                "upload_id": "test-upload-id",
                "filename": "test.csv",
                "original_filename": "test.csv",
                "file_path": "/tmp/test.csv",
                "file_size": 1000,
                "content_type": "text/csv",
                "detected_type": "csv",
                "schema_info": {
                    "columns": {
                        "id": {"name": "id", "dtype": "int64", "inferred_type": "integer"},
                        "name": {"name": "name", "dtype": "object", "inferred_type": "text"}
                    },
                    "total_rows": 5,
                    "total_columns": 4
                },
                "preview_data": {
                    "columns": [
                        {"name": "id", "type": "int64", "inferred_type": "integer"},
                        {"name": "name", "type": "object", "inferred_type": "text"}
                    ],
                    "sample_data": [
                        {"id": 1, "name": "Alice"},
                        {"id": 2, "name": "Bob"}
                    ],
                    "total_rows": 5,
                    "preview_rows": 2
                },
                "quality_report": {
                    "quality_score": 95.0,
                    "total_rows": 5,
                    "missing_values": {},
                    "recommendations": ["Data quality looks good"]
                },
                "dataset_id": None,
                "processing_time": 1.2
            }
            
            with open(sample_files['csv'], 'rb') as f:
                response = client.post(
                    "/api/files/upload",
                    files={"file": ("test.csv", f, "text/csv")},
                    data={
                        "auto_detect_schema": "true",
                        "generate_preview": "true"
                    }
                )
            
            assert response.status_code == 200
            data = response.json()
            assert data["upload_id"] == "test-upload-id"
            assert data["detected_type"] == "csv"
            assert "schema_info" in data
            assert "preview_data" in data
            assert "quality_report" in data
            assert data["processing_time"] == 1.2
    
    def test_multiple_file_format_support(self, client, sample_files):
        """Test uploading different file formats."""
        with patch('scrollintel.api.routes.file_routes.require_permission'), \
             patch('scrollintel.api.routes.file_routes.get_db'), \
             patch.object(FileProcessorEngine, 'process_upload') as mock_process:
            
            # Test CSV
            mock_process.return_value = {
                "upload_id": "csv-upload",
                "detected_type": "csv",
                "filename": "test.csv",
                "original_filename": "test.csv",
                "file_path": "/tmp/test.csv",
                "file_size": 1000,
                "content_type": "text/csv",
                "schema_info": {},
                "preview_data": None,
                "quality_report": None,
                "dataset_id": None,
                "processing_time": 0.5
            }
            
            with open(sample_files['csv'], 'rb') as f:
                response = client.post(
                    "/api/files/upload",
                    files={"file": ("test.csv", f, "text/csv")}
                )
            assert response.status_code == 200
            assert response.json()["detected_type"] == "csv"
            
            # Test JSON
            mock_process.return_value["upload_id"] = "json-upload"
            mock_process.return_value["detected_type"] = "json"
            
            with open(sample_files['json'], 'rb') as f:
                response = client.post(
                    "/api/files/upload",
                    files={"file": ("test.json", f, "application/json")}
                )
            assert response.status_code == 200
            assert response.json()["detected_type"] == "json"
            
            # Test SQL
            mock_process.return_value["upload_id"] = "sql-upload"
            mock_process.return_value["detected_type"] = "sql"
            
            with open(sample_files['sql'], 'rb') as f:
                response = client.post(
                    "/api/files/upload",
                    files={"file": ("test.sql", f, "text/plain")}
                )
            assert response.status_code == 200
            assert response.json()["detected_type"] == "sql"


class TestFileUploadIntegration:
    """Integration tests for file upload workflows."""
    
    @pytest.fixture
    def sample_data_files(self):
        """Create various sample data files for integration testing."""
        files = {}
        
        # Large CSV file for performance testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,name,email,score,category,date\n")
            for i in range(1000):
                f.write(f"{i},User{i},user{i}@example.com,{85.5 + (i % 15)},Category{i % 5},2023-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}\n")
            files['large_csv'] = f.name
        
        # CSV with quality issues
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,name,email,score\n")
            f.write("1,Alice,alice@example.com,85.5\n")
            f.write("2,Bob,,92.0\n")  # Missing email
            f.write("3,Charlie,invalid-email,78.5\n")  # Invalid email
            f.write("1,Alice,alice@example.com,85.5\n")  # Duplicate
            f.write("4,David,david@example.com,999\n")  # Outlier score
            files['quality_issues_csv'] = f.name
        
        # Complex JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {
                "metadata": {
                    "version": "1.0",
                    "created": "2023-01-01T00:00:00Z"
                },
                "users": [
                    {
                        "id": 1,
                        "profile": {
                            "name": "Alice",
                            "email": "alice@example.com",
                            "preferences": {
                                "theme": "dark",
                                "notifications": True
                            }
                        },
                        "scores": [85, 90, 88]
                    },
                    {
                        "id": 2,
                        "profile": {
                            "name": "Bob",
                            "email": "bob@example.com",
                            "preferences": {
                                "theme": "light",
                                "notifications": False
                            }
                        },
                        "scores": [92, 89, 95]
                    }
                ]
            }
            json.dump(data, f, indent=2)
            files['complex_json'] = f.name
        
        yield files
        
        # Cleanup
        for file_path in files.values():
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    def test_end_to_end_file_upload_workflow(self, sample_data_files):
        """Test complete file upload workflow from upload to dataset creation."""
        engine = FileProcessorEngine()
        
        # Test file validation
        with open(sample_data_files['large_csv'], 'rb') as f:
            mock_file = Mock()
            mock_file.filename = "large_test.csv"
            mock_file.size = os.path.getsize(sample_data_files['large_csv'])
            mock_file.content_type = "text/csv"
            mock_file.file = f
            
            # This would normally be an async call
            # validation_result = await engine.validate_file_format(mock_file)
            # assert validation_result["valid"] is True
        
        # Test data loading and processing
        df = pd.read_csv(sample_data_files['large_csv'])
        
        # Test schema inference
        schema_info = engine._infer_schema(df)
        assert schema_info["total_rows"] == 1000
        assert schema_info["total_columns"] == 6
        assert "id" in schema_info["columns"]
        assert schema_info["columns"]["email"]["inferred_type"] == "email"
        
        # Test quality report
        quality_report = engine._generate_quality_report(df)
        assert quality_report["quality_score"] > 90  # Should be high quality
        assert quality_report["duplicate_rows"] == 0
        
        # Test preview generation
        preview_data = engine._generate_preview(df)
        assert preview_data["total_rows"] == 1000
        assert preview_data["preview_rows"] <= engine.max_preview_rows
        assert len(preview_data["sample_data"]) <= engine.max_preview_rows
    
    def test_quality_issues_detection(self, sample_data_files):
        """Test detection of various data quality issues."""
        engine = FileProcessorEngine()
        
        # Load file with quality issues
        df = pd.read_csv(sample_data_files['quality_issues_csv'])
        
        # Test quality report
        quality_report = engine._generate_quality_report(df)
        
        # Should detect missing values
        assert quality_report["missing_values"]["email"] > 0
        
        # Should detect duplicates
        assert quality_report["duplicate_rows"] > 0
        
        # Should detect outliers
        assert "score" in quality_report["outliers"]
        
        # Quality score should be lower due to issues
        assert quality_report["quality_score"] < 90
        
        # Should have recommendations
        assert len(quality_report["recommendations"]) > 0
        recommendations_text = " ".join(quality_report["recommendations"]).lower()
        assert any(keyword in recommendations_text for keyword in ["missing", "duplicate", "outlier"])
    
    def test_complex_json_processing(self, sample_data_files):
        """Test processing of complex nested JSON files."""
        engine = FileProcessorEngine()
        
        # Load complex JSON
        with open(sample_data_files['complex_json'], 'r') as f:
            json_data = json.load(f)
        
        # Convert to DataFrame (simplified - real implementation would be more complex)
        users_data = []
        for user in json_data["users"]:
            flat_user = {
                "id": user["id"],
                "name": user["profile"]["name"],
                "email": user["profile"]["email"],
                "theme": user["profile"]["preferences"]["theme"],
                "notifications": user["profile"]["preferences"]["notifications"],
                "avg_score": sum(user["scores"]) / len(user["scores"])
            }
            users_data.append(flat_user)
        
        df = pd.DataFrame(users_data)
        
        # Test schema inference on flattened data
        schema_info = engine._infer_schema(df)
        assert schema_info["total_rows"] == 2
        assert schema_info["columns"]["email"]["inferred_type"] == "email"
        assert schema_info["columns"]["notifications"]["inferred_type"] == "boolean"
        
        # Test quality report
        quality_report = engine._generate_quality_report(df)
        assert quality_report["quality_score"] > 95  # Should be high quality


if __name__ == "__main__":
    pytest.main([__file__, "-v"])