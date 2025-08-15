"""
Integration tests for file upload and processing functionality.
Tests file upload endpoints, auto-detection, schema inference, and data quality validation.
"""

import os
import tempfile
import pytest
import pandas as pd
import json
from io import BytesIO
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from scrollintel.api.gateway import app
from scrollintel.models.database import FileUpload, Dataset, User
from scrollintel.models.database_utils import get_test_db_manager
from scrollintel.engines.file_processor import FileProcessorEngine
from scrollintel.security.auth import PasswordManager


@pytest.fixture
async def db_session():
    """Get test database session."""
    manager = await get_test_db_manager()
    async with manager.get_async_session() as session:
        yield session


@pytest.fixture
def client():
    """Get test client."""
    return TestClient(app)


@pytest.fixture
async def test_user(db_session: Session):
    """Create a test user."""
    password_manager = PasswordManager()
    hashed_password = password_manager.hash_password("testpassword123")
    
    user = User(
        email="test@example.com",
        hashed_password=hashed_password,
        full_name="Test User",
        is_active=True,
        is_verified=True
    )
    
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    
    return user


@pytest.fixture
def auth_headers(client: TestClient, test_user: User):
    """Get authentication headers for test user."""
    # Login to get token
    response = client.post("/auth/login", json={
        "email": "test@example.com",
        "password": "testpassword123"
    })
    
    assert response.status_code == 200
    token_data = response.json()
    
    return {"Authorization": f"Bearer {token_data['access_token']}"}


@pytest.fixture
def sample_csv_file():
    """Create a sample CSV file for testing."""
    data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com', 'diana@example.com', 'eve@example.com'],
        'salary': [50000, 60000, 70000, 55000, 65000],
        'active': [True, True, False, True, True]
    }
    
    df = pd.DataFrame(data)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        return f.name


@pytest.fixture
def sample_xlsx_file():
    """Create a sample Excel file for testing."""
    data = {
        'product_id': [1, 2, 3, 4, 5],
        'product_name': ['Widget A', 'Widget B', 'Widget C', 'Widget D', 'Widget E'],
        'price': [19.99, 29.99, 39.99, 24.99, 34.99],
        'category': ['Electronics', 'Electronics', 'Home', 'Electronics', 'Home'],
        'in_stock': [True, False, True, True, False],
        'rating': [4.5, 3.8, 4.2, 4.0, 3.9]
    }
    
    df = pd.DataFrame(data)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
        df.to_excel(f.name, index=False)
        return f.name


@pytest.fixture
def sample_json_file():
    """Create a sample JSON file for testing."""
    data = [
        {"user_id": 1, "username": "alice", "score": 95, "level": "advanced"},
        {"user_id": 2, "username": "bob", "score": 87, "level": "intermediate"},
        {"user_id": 3, "username": "charlie", "score": 92, "level": "advanced"},
        {"user_id": 4, "username": "diana", "score": 78, "level": "beginner"},
        {"user_id": 5, "username": "eve", "score": 89, "level": "intermediate"}
    ]
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        return f.name


@pytest.fixture
def sample_sql_file():
    """Create a sample SQL file for testing."""
    sql_content = """
    CREATE TABLE employees (
        id INTEGER PRIMARY KEY,
        first_name VARCHAR(50) NOT NULL,
        last_name VARCHAR(50) NOT NULL,
        email VARCHAR(100) UNIQUE NOT NULL,
        department VARCHAR(50),
        salary DECIMAL(10,2),
        hire_date DATE,
        is_active BOOLEAN DEFAULT TRUE
    );
    
    CREATE TABLE departments (
        id INTEGER PRIMARY KEY,
        name VARCHAR(50) NOT NULL,
        budget DECIMAL(12,2),
        manager_id INTEGER
    );
    """
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False) as f:
        f.write(sql_content)
        return f.name


class TestFileUpload:
    """Test file upload functionality."""
    
    def test_upload_csv_file(self, client: TestClient, auth_headers: dict, sample_csv_file: str):
        """Test uploading a CSV file."""
        with open(sample_csv_file, 'rb') as f:
            response = client.post(
                "/files/upload",
                headers=auth_headers,
                files={"file": ("test.csv", f, "text/csv")},
                data={
                    "auto_detect_schema": "true",
                    "generate_preview": "true"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "upload_id" in data
        assert data["detected_type"] == "csv"
        assert data["original_filename"] == "test.csv"
        assert data["file_size"] > 0
        assert data["content_type"] == "text/csv"
        
        # Clean up
        os.unlink(sample_csv_file)
    
    def test_upload_xlsx_file(self, client: TestClient, auth_headers: dict, sample_xlsx_file: str):
        """Test uploading an Excel file."""
        with open(sample_xlsx_file, 'rb') as f:
            response = client.post(
                "/files/upload",
                headers=auth_headers,
                files={"file": ("test.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
                data={
                    "auto_detect_schema": "true",
                    "generate_preview": "true"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["detected_type"] == "xlsx"
        assert data["original_filename"] == "test.xlsx"
        
        # Clean up
        os.unlink(sample_xlsx_file)
    
    def test_upload_json_file(self, client: TestClient, auth_headers: dict, sample_json_file: str):
        """Test uploading a JSON file."""
        with open(sample_json_file, 'rb') as f:
            response = client.post(
                "/files/upload",
                headers=auth_headers,
                files={"file": ("test.json", f, "application/json")},
                data={
                    "auto_detect_schema": "true",
                    "generate_preview": "true"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["detected_type"] == "json"
        assert data["original_filename"] == "test.json"
        
        # Clean up
        os.unlink(sample_json_file)
    
    def test_upload_sql_file(self, client: TestClient, auth_headers: dict, sample_sql_file: str):
        """Test uploading a SQL file."""
        with open(sample_sql_file, 'rb') as f:
            response = client.post(
                "/files/upload",
                headers=auth_headers,
                files={"file": ("test.sql", f, "text/plain")},
                data={
                    "auto_detect_schema": "true",
                    "generate_preview": "true"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["detected_type"] == "sql"
        assert data["original_filename"] == "test.sql"
        
        # Clean up
        os.unlink(sample_sql_file)
    
    def test_upload_unsupported_file(self, client: TestClient, auth_headers: dict):
        """Test uploading an unsupported file type."""
        # Create a fake image file
        fake_image = BytesIO(b"fake image content")
        
        response = client.post(
            "/files/upload",
            headers=auth_headers,
            files={"file": ("test.png", fake_image, "image/png")}
        )
        
        assert response.status_code == 400
        assert "unsupported file type" in response.json()["detail"].lower()
    
    def test_upload_large_file(self, client: TestClient, auth_headers: dict):
        """Test uploading a file that exceeds size limit."""
        # Create a large fake file (simulate > 100MB)
        large_content = b"x" * (101 * 1024 * 1024)  # 101MB
        large_file = BytesIO(large_content)
        
        response = client.post(
            "/files/upload",
            headers=auth_headers,
            files={"file": ("large.csv", large_file, "text/csv")}
        )
        
        assert response.status_code == 400
        assert "file size" in response.json()["detail"].lower()
    
    def test_upload_without_authentication(self, client: TestClient, sample_csv_file: str):
        """Test uploading without authentication."""
        with open(sample_csv_file, 'rb') as f:
            response = client.post(
                "/files/upload",
                files={"file": ("test.csv", f, "text/csv")}
            )
        
        assert response.status_code == 401
        
        # Clean up
        os.unlink(sample_csv_file)


class TestFileProcessing:
    """Test file processing functionality."""
    
    @pytest.mark.asyncio
    async def test_file_processor_csv(self, sample_csv_file: str):
        """Test file processor with CSV file."""
        processor = FileProcessorEngine()
        
        # Load and process CSV
        df = await processor._load_data(sample_csv_file, "csv")
        
        assert len(df) == 5
        assert "name" in df.columns
        assert "age" in df.columns
        assert "email" in df.columns
        
        # Test schema inference
        schema_info = processor._infer_schema(df)
        
        assert schema_info["total_rows"] == 5
        assert schema_info["total_columns"] == 6
        assert "columns" in schema_info
        
        # Check column types
        columns = schema_info["columns"]
        assert columns["age"]["inferred_type"] == "integer"
        assert columns["email"]["inferred_type"] == "email"
        assert columns["active"]["inferred_type"] == "boolean"
        
        # Test preview generation
        preview = processor._generate_preview(df)
        
        assert len(preview["sample_data"]) == 5
        assert len(preview["columns"]) == 6
        assert preview["total_rows"] == 5
        
        # Test quality report
        quality_report = processor._generate_quality_report(df)
        
        assert quality_report["total_rows"] == 5
        assert quality_report["total_columns"] == 6
        assert quality_report["duplicate_rows"] == 0
        assert quality_report["quality_score"] > 80  # Should be high quality
        
        # Clean up
        os.unlink(sample_csv_file)
    
    @pytest.mark.asyncio
    async def test_file_processor_json(self, sample_json_file: str):
        """Test file processor with JSON file."""
        processor = FileProcessorEngine()
        
        # Load and process JSON
        df = await processor._load_data(sample_json_file, "json")
        
        assert len(df) == 5
        assert "username" in df.columns
        assert "score" in df.columns
        assert "level" in df.columns
        
        # Test schema inference
        schema_info = processor._infer_schema(df)
        
        assert schema_info["total_rows"] == 5
        assert "columns" in schema_info
        
        # Check column types
        columns = schema_info["columns"]
        assert columns["score"]["inferred_type"] == "integer"
        assert columns["level"]["inferred_type"] == "categorical"
        
        # Clean up
        os.unlink(sample_json_file)
    
    @pytest.mark.asyncio
    async def test_file_processor_sql(self, sample_sql_file: str):
        """Test file processor with SQL file."""
        processor = FileProcessorEngine()
        
        # Load and process SQL
        df = await processor._load_data(sample_sql_file, "sql")
        
        # SQL files create empty DataFrames with column info
        assert len(df.columns) > 0 or len(df) == 0
        
        # Clean up
        os.unlink(sample_sql_file)
    
    def test_column_type_inference(self):
        """Test column type inference."""
        processor = FileProcessorEngine()
        
        # Test email detection
        email_series = pd.Series(["alice@example.com", "bob@test.org", "charlie@domain.net"])
        assert processor._infer_column_type(email_series) == "email"
        
        # Test URL detection
        url_series = pd.Series(["https://example.com", "http://test.org", "https://domain.net"])
        assert processor._infer_column_type(url_series) == "url"
        
        # Test categorical detection
        category_series = pd.Series(["A", "B", "A", "C", "B", "A", "C"])
        assert processor._infer_column_type(category_series) == "categorical"
        
        # Test numeric detection
        int_series = pd.Series([1, 2, 3, 4, 5])
        assert processor._infer_column_type(int_series) == "integer"
        
        float_series = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5])
        assert processor._infer_column_type(float_series) == "float"
    
    def test_quality_score_calculation(self):
        """Test data quality score calculation."""
        processor = FileProcessorEngine()
        
        # Perfect data
        perfect_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['A', 'B', 'C', 'D', 'E'],
            'value': [10, 20, 30, 40, 50]
        })
        
        quality_report = processor._generate_quality_report(perfect_data)
        assert quality_report["quality_score"] > 90
        
        # Data with issues
        problematic_data = pd.DataFrame({
            'id': [1, 2, 2, 4, 5],  # Duplicate
            'name': ['A', None, 'C', 'D', None],  # Missing values
            'value': [10, 20, 1000, 40, 50]  # Outlier
        })
        
        quality_report = processor._generate_quality_report(problematic_data)
        assert quality_report["quality_score"] < 90
        assert quality_report["duplicate_rows"] > 0
        assert sum(quality_report["missing_values"].values()) > 0


class TestFileAPI:
    """Test file API endpoints."""
    
    def test_get_supported_formats(self, client: TestClient, auth_headers: dict):
        """Test getting supported file formats."""
        response = client.get("/files/supported-formats", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "supported_formats" in data
        assert "max_file_size" in data
        assert "max_file_size_mb" in data
        
        formats = data["supported_formats"]
        extensions = [fmt["extension"] for fmt in formats]
        
        assert ".csv" in extensions
        assert ".xlsx" in extensions
        assert ".json" in extensions
        assert ".sql" in extensions
    
    def test_list_uploads(self, client: TestClient, auth_headers: dict):
        """Test listing user uploads."""
        response = client.get("/files/uploads", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
    
    def test_get_upload_status_not_found(self, client: TestClient, auth_headers: dict):
        """Test getting status for non-existent upload."""
        response = client.get("/files/upload/nonexistent-id/status", headers=auth_headers)
        
        assert response.status_code == 404
    
    def test_get_preview_not_found(self, client: TestClient, auth_headers: dict):
        """Test getting preview for non-existent upload."""
        response = client.get("/files/upload/nonexistent-id/preview", headers=auth_headers)
        
        assert response.status_code == 404
    
    def test_get_quality_report_not_found(self, client: TestClient, auth_headers: dict):
        """Test getting quality report for non-existent upload."""
        response = client.get("/files/upload/nonexistent-id/quality", headers=auth_headers)
        
        assert response.status_code == 404


class TestDatasetCreation:
    """Test dataset creation from uploads."""
    
    def test_create_dataset_from_upload_not_found(self, client: TestClient, auth_headers: dict):
        """Test creating dataset from non-existent upload."""
        response = client.post(
            "/files/upload/nonexistent-id/create-dataset",
            headers=auth_headers,
            data={
                "dataset_name": "Test Dataset",
                "dataset_description": "Test description"
            }
        )
        
        assert response.status_code == 400


class TestErrorHandling:
    """Test error handling in file processing."""
    
    def test_invalid_csv_file(self):
        """Test handling of invalid CSV file."""
        processor = FileProcessorEngine()
        
        # Create invalid CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("invalid,csv,content\nwith,mismatched\ncolumns")
            invalid_csv = f.name
        
        try:
            # This should handle the error gracefully
            import asyncio
            df = asyncio.run(processor._load_data(invalid_csv, "csv"))
            # Should still create a DataFrame, even if malformed
            assert isinstance(df, pd.DataFrame)
        except Exception as e:
            # Should raise EngineError with descriptive message
            assert "Failed to load data" in str(e)
        finally:
            os.unlink(invalid_csv)
    
    def test_empty_file(self):
        """Test handling of empty file."""
        processor = FileProcessorEngine()
        
        # Create empty CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("")
            empty_csv = f.name
        
        try:
            import asyncio
            df = asyncio.run(processor._load_data(empty_csv, "csv"))
            
            # Should create empty DataFrame
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0
            
            # Quality report should handle empty data
            quality_report = processor._generate_quality_report(df)
            assert quality_report["quality_score"] == 0.0
            assert "empty" in quality_report["recommendations"][0].lower()
            
        finally:
            os.unlink(empty_csv)


if __name__ == "__main__":
    pytest.main([__file__])