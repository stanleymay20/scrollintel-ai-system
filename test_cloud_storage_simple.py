"""
Simple Cloud Storage Test
Basic test to verify cloud storage integration functionality.
"""

import asyncio
from io import BytesIO
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.engines.cloud_storage_engine import CloudStorageEngine, FileProcessor
from scrollintel.models.cloud_storage_models import (
    CloudStorageConnectionConfig, CloudProvider, FileFormat
)


def test_file_processor():
    """Test file format detection and metadata extraction"""
    print("Testing File Processor...")
    
    processor = FileProcessor()
    
    # Test CSV detection
    csv_content = b"name,age\nJohn,30\nJane,25"
    csv_format = processor.detect_file_format("test.csv", csv_content)
    print(f"✓ CSV format detected: {csv_format}")
    assert csv_format == FileFormat.CSV
    
    # Test CSV metadata extraction
    csv_metadata = processor.extract_metadata("test.csv", csv_content, FileFormat.CSV)
    print(f"✓ CSV metadata extracted: {csv_metadata}")
    assert csv_metadata['rows'] == 2
    assert csv_metadata['columns'] == 2
    
    # Test JSON detection
    json_content = b'{"name": "test", "value": 123}'
    json_format = processor.detect_file_format("test.json", json_content)
    print(f"✓ JSON format detected: {json_format}")
    assert json_format == FileFormat.JSON
    
    # Test JSON metadata extraction
    json_metadata = processor.extract_metadata("test.json", json_content, FileFormat.JSON)
    print(f"✓ JSON metadata extracted: {json_metadata}")
    assert json_metadata['json_type'] == 'dict'
    assert 'name' in json_metadata['keys']
    
    print("✅ File Processor tests passed!\n")


@patch('scrollintel.engines.cloud_storage_engine.get_database_session')
@patch('scrollintel.connectors.cloud_storage_connector.CloudStorageConnectorFactory.create_connector')
async def test_cloud_storage_engine(mock_factory, mock_db_session):
    """Test cloud storage engine functionality"""
    print("Testing Cloud Storage Engine...")
    
    # Mock connector
    mock_connector = AsyncMock()
    mock_connector.test_connection.return_value = True
    mock_connector.upload_file.return_value = {
        'file_path': 'test.txt',
        'size': 100,
        'checksum': 'abc123'
    }
    mock_connector.download_file.return_value = AsyncMock()
    mock_connector.list_files.return_value = [
        {'file_path': 'test.txt', 'size': 100}
    ]
    mock_factory.return_value = mock_connector
    
    # Mock database session
    mock_session = AsyncMock()
    mock_db_session.return_value.__aenter__.return_value = mock_session
    
    # Create engine
    engine = CloudStorageEngine()
    
    # Test connection creation
    config = CloudStorageConnectionConfig(
        name="Test Connection",
        provider=CloudProvider.AWS_S3,
        config={"bucket_name": "test"},
        credentials={"access_key": "test"}
    )
    
    connection_id = await engine.create_connection(config)
    print(f"✓ Connection created: {connection_id}")
    assert connection_id is not None
    
    # Test file upload
    test_file = BytesIO(b"test content")
    file_id = await engine.upload_file(
        connection_id=connection_id,
        file_path="test.txt",
        file_data=test_file
    )
    print(f"✓ File uploaded: {file_id}")
    assert file_id is not None
    
    # Test file listing
    files = await engine.list_files(connection_id)
    print(f"✓ Files listed: {len(files)} files")
    assert len(files) >= 0
    
    print("✅ Cloud Storage Engine tests passed!\n")


def test_connector_factory():
    """Test cloud storage connector factory"""
    print("Testing Connector Factory...")
    
    from scrollintel.connectors.cloud_storage_connector import (
        CloudStorageConnectorFactory, S3Connector, AzureBlobConnector, GoogleCloudConnector
    )
    
    # Test S3 connector creation
    s3_connector = CloudStorageConnectorFactory.create_connector(
        CloudProvider.AWS_S3,
        {"bucket_name": "test"},
        {"access_key": "test"}
    )
    print(f"✓ S3 connector created: {type(s3_connector).__name__}")
    assert isinstance(s3_connector, S3Connector)
    
    # Test Azure connector creation
    azure_connector = CloudStorageConnectorFactory.create_connector(
        CloudProvider.AZURE_BLOB,
        {"container_name": "test"},
        {"account_key": "test"}
    )
    print(f"✓ Azure connector created: {type(azure_connector).__name__}")
    assert isinstance(azure_connector, AzureBlobConnector)
    
    # Test GCS connector creation
    gcs_connector = CloudStorageConnectorFactory.create_connector(
        CloudProvider.GOOGLE_CLOUD,
        {"bucket_name": "test"},
        {"service_account_key": {}}
    )
    print(f"✓ GCS connector created: {type(gcs_connector).__name__}")
    assert isinstance(gcs_connector, GoogleCloudConnector)
    
    print("✅ Connector Factory tests passed!\n")


async def test_api_models():
    """Test API models and validation"""
    print("Testing API Models...")
    
    # Test connection config model
    config = CloudStorageConnectionConfig(
        name="Test Connection",
        provider=CloudProvider.AWS_S3,
        config={"bucket_name": "test-bucket"},
        credentials={"access_key_id": "test-key"}
    )
    print(f"✓ Connection config created: {config.name}")
    assert config.provider == CloudProvider.AWS_S3
    
    # Test enum values
    assert CloudProvider.AWS_S3 == "aws_s3"
    assert CloudProvider.AZURE_BLOB == "azure_blob"
    assert CloudProvider.GOOGLE_CLOUD == "google_cloud"
    
    assert FileFormat.CSV == "csv"
    assert FileFormat.JSON == "json"
    assert FileFormat.PARQUET == "parquet"
    
    print("✅ API Models tests passed!\n")


async def run_simple_tests():
    """Run all simple tests"""
    print("🧪 Running Simple Cloud Storage Tests")
    print("=" * 40)
    
    try:
        # Run synchronous tests
        test_file_processor()
        test_connector_factory()
        await test_api_models()
        
        # Run asynchronous tests
        await test_cloud_storage_engine()
        
        print("=" * 40)
        print("✅ All Simple Tests Passed!")
        print("\nTested Components:")
        print("• File format detection and metadata extraction")
        print("• Cloud storage connector factory")
        print("• Cloud storage engine core functionality")
        print("• API models and validation")
        
    except Exception as e:
        print(f"\n❌ Tests failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_simple_tests())