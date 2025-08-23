"""
Cloud Storage Integration Tests
Comprehensive tests for cloud storage functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from io import BytesIO
import json
from datetime import datetime

from scrollintel.engines.cloud_storage_engine import CloudStorageEngine, FileProcessor
from scrollintel.connectors.cloud_storage_connector import (
    CloudStorageConnectorFactory, S3Connector, AzureBlobConnector, 
    GoogleCloudConnector, CloudStorageError
)
from scrollintel.models.cloud_storage_models import (
    CloudProvider, FileFormat, ConnectionStatus, CloudStorageConnectionConfig
)


class TestFileProcessor:
    """Test file format detection and metadata extraction"""
    
    def test_detect_file_format_csv(self):
        """Test CSV file format detection"""
        processor = FileProcessor()
        
        # Test with CSV content
        csv_content = b"name,age,city\nJohn,30,NYC\nJane,25,LA"
        format_detected = processor.detect_file_format("test.csv", csv_content)
        
        assert format_detected == FileFormat.CSV
    
    def test_detect_file_format_json(self):
        """Test JSON file format detection"""
        processor = FileProcessor()
        
        # Test with JSON content
        json_content = b'{"name": "John", "age": 30}'
        format_detected = processor.detect_file_format("test.json", json_content)
        
        assert format_detected == FileFormat.JSON
    
    def test_extract_csv_metadata(self):
        """Test CSV metadata extraction"""
        processor = FileProcessor()
        
        csv_content = b"name,age,city\nJohn,30,NYC\nJane,25,LA"
        metadata = processor.extract_metadata("test.csv", csv_content, FileFormat.CSV)
        
        assert metadata['rows'] == 2
        assert metadata['columns'] == 3
        assert 'name' in metadata['column_names']
        assert 'age' in metadata['column_names']
        assert 'city' in metadata['column_names']
    
    def test_extract_json_metadata(self):
        """Test JSON metadata extraction"""
        processor = FileProcessor()
        
        json_content = b'{"name": "John", "age": 30, "skills": ["Python", "AI"]}'
        metadata = processor.extract_metadata("test.json", json_content, FileFormat.JSON)
        
        assert metadata['json_type'] == 'dict'
        assert 'name' in metadata['keys']
        assert 'age' in metadata['keys']
        assert 'skills' in metadata['keys']
    
    def test_extract_text_metadata(self):
        """Test text metadata extraction"""
        processor = FileProcessor()
        
        text_content = b"Hello world!\nThis is a test file.\nIt has multiple lines."
        metadata = processor.extract_metadata("test.txt", text_content, FileFormat.TEXT)
        
        assert metadata['line_count'] == 3
        assert metadata['character_count'] > 0
        assert metadata['word_count'] > 0


class TestCloudStorageConnectors:
    """Test cloud storage connector implementations"""
    
    @pytest.fixture
    def s3_config(self):
        return {
            'bucket_name': 'test-bucket',
            'region': 'us-east-1'
        }
    
    @pytest.fixture
    def s3_credentials(self):
        return {
            'access_key_id': 'test-access-key',
            'secret_access_key': 'test-secret-key'
        }
    
    @pytest.fixture
    def azure_config(self):
        return {
            'container_name': 'test-container',
            'account_name': 'testaccount'
        }
    
    @pytest.fixture
    def azure_credentials(self):
        return {
            'account_key': 'test-account-key'
        }
    
    @pytest.fixture
    def gcs_config(self):
        return {
            'bucket_name': 'test-bucket',
            'project_id': 'test-project'
        }
    
    @pytest.fixture
    def gcs_credentials(self):
        return {
            'service_account_key': {
                'type': 'service_account',
                'project_id': 'test-project'
            }
        }
    
    def test_connector_factory_s3(self, s3_config, s3_credentials):
        """Test S3 connector creation"""
        connector = CloudStorageConnectorFactory.create_connector(
            CloudProvider.AWS_S3, s3_config, s3_credentials
        )
        
        assert isinstance(connector, S3Connector)
        assert connector.bucket_name == 'test-bucket'
        assert connector.region == 'us-east-1'
    
    def test_connector_factory_azure(self, azure_config, azure_credentials):
        """Test Azure Blob connector creation"""
        connector = CloudStorageConnectorFactory.create_connector(
            CloudProvider.AZURE_BLOB, azure_config, azure_credentials
        )
        
        assert isinstance(connector, AzureBlobConnector)
        assert connector.container_name == 'test-container'
        assert connector.account_name == 'testaccount'
    
    def test_connector_factory_gcs(self, gcs_config, gcs_credentials):
        """Test Google Cloud Storage connector creation"""
        connector = CloudStorageConnectorFactory.create_connector(
            CloudProvider.GOOGLE_CLOUD, gcs_config, gcs_credentials
        )
        
        assert isinstance(connector, GoogleCloudConnector)
        assert connector.bucket_name == 'test-bucket'
        assert connector.project_id == 'test-project'
    
    def test_connector_factory_invalid_provider(self):
        """Test invalid provider handling"""
        with pytest.raises(ValueError):
            CloudStorageConnectorFactory.create_connector(
                "invalid_provider", {}, {}
            )
    
    @patch('boto3.client')
    async def test_s3_connector_test_connection(self, mock_boto_client, s3_config, s3_credentials):
        """Test S3 connection testing"""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        connector = S3Connector(s3_config, s3_credentials)
        
        # Test successful connection
        mock_client.head_bucket.return_value = {}
        result = await connector.test_connection()
        assert result is True
        
        # Test failed connection
        mock_client.head_bucket.side_effect = Exception("Connection failed")
        with pytest.raises(Exception):
            await connector.test_connection()
    
    @patch('boto3.client')
    async def test_s3_connector_upload_file(self, mock_boto_client, s3_config, s3_credentials):
        """Test S3 file upload"""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        connector = S3Connector(s3_config, s3_credentials)
        
        # Test file upload
        test_data = BytesIO(b"test file content")
        result = await connector.upload_file("test/file.txt", test_data)
        
        assert result['file_path'] == "test/file.txt"
        assert result['size'] > 0
        assert 'checksum' in result
        mock_client.put_object.assert_called_once()
    
    @patch('boto3.client')
    async def test_s3_connector_download_file(self, mock_boto_client, s3_config, s3_credentials):
        """Test S3 file download"""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        # Mock S3 response
        mock_body = Mock()
        mock_body.read.return_value = b"test content"
        mock_client.get_object.return_value = {'Body': mock_body}
        
        connector = S3Connector(s3_config, s3_credentials)
        
        # Test file download
        chunks = []
        async for chunk in connector.download_file("test/file.txt"):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        mock_client.get_object.assert_called_once()
    
    @patch('boto3.client')
    async def test_s3_connector_list_files(self, mock_boto_client, s3_config, s3_credentials):
        """Test S3 file listing"""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        # Mock paginator
        mock_paginator = Mock()
        mock_client.get_paginator.return_value = mock_paginator
        
        mock_page_iterator = [
            {
                'Contents': [
                    {
                        'Key': 'file1.txt',
                        'Size': 100,
                        'LastModified': datetime.utcnow(),
                        'ETag': 'etag1'
                    },
                    {
                        'Key': 'file2.txt',
                        'Size': 200,
                        'LastModified': datetime.utcnow(),
                        'ETag': 'etag2'
                    }
                ]
            }
        ]
        mock_paginator.paginate.return_value = mock_page_iterator
        
        connector = S3Connector(s3_config, s3_credentials)
        
        # Test file listing
        files = await connector.list_files()
        
        assert len(files) == 2
        assert files[0]['file_path'] == 'file1.txt'
        assert files[1]['file_path'] == 'file2.txt'


class TestCloudStorageEngine:
    """Test cloud storage engine functionality"""
    
    @pytest.fixture
    def engine(self):
        return CloudStorageEngine()
    
    @pytest.fixture
    def connection_config(self):
        return CloudStorageConnectionConfig(
            name="Test S3 Connection",
            provider=CloudProvider.AWS_S3,
            config={
                'bucket_name': 'test-bucket',
                'region': 'us-east-1'
            },
            credentials={
                'access_key_id': 'test-access-key',
                'secret_access_key': 'test-secret-key'
            }
        )
    
    @patch('scrollintel.engines.cloud_storage_engine.get_database_session')
    @patch('scrollintel.connectors.cloud_storage_connector.CloudStorageConnectorFactory.create_connector')
    async def test_create_connection(self, mock_factory, mock_db_session, engine, connection_config):
        """Test connection creation"""
        # Mock connector
        mock_connector = AsyncMock()
        mock_connector.test_connection.return_value = True
        mock_factory.return_value = mock_connector
        
        # Mock database session
        mock_session = AsyncMock()
        mock_db_session.return_value.__aenter__.return_value = mock_session
        
        # Test connection creation
        connection_id = await engine.create_connection(connection_config)
        
        assert connection_id is not None
        assert len(connection_id) > 0
        mock_connector.test_connection.assert_called_once()
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
    
    @patch('scrollintel.engines.cloud_storage_engine.get_database_session')
    async def test_get_connection(self, mock_db_session, engine):
        """Test getting connection by ID"""
        mock_session = AsyncMock()
        mock_db_session.return_value.__aenter__.return_value = mock_session
        
        # Mock connection object
        mock_connection = Mock()
        mock_connection.id = "test-connection-id"
        mock_connection.name = "Test Connection"
        mock_session.get.return_value = mock_connection
        
        # Test getting connection
        connection = await engine.get_connection("test-connection-id")
        
        assert connection is not None
        assert connection.id == "test-connection-id"
        mock_session.get.assert_called_once()
    
    @patch('scrollintel.engines.cloud_storage_engine.get_database_session')
    @patch.object(CloudStorageEngine, '_get_connector')
    async def test_upload_file(self, mock_get_connector, mock_db_session, engine):
        """Test file upload"""
        # Mock connector
        mock_connector = AsyncMock()
        mock_connector.upload_file.return_value = {
            'file_path': 'test/file.txt',
            'size': 100,
            'checksum': 'abc123'
        }
        mock_get_connector.return_value = mock_connector
        
        # Mock database session
        mock_session = AsyncMock()
        mock_db_session.return_value.__aenter__.return_value = mock_session
        
        # Test file upload
        test_file = BytesIO(b"test content")
        file_id = await engine.upload_file(
            connection_id="test-connection",
            file_path="test/file.txt",
            file_data=test_file,
            metadata={"test": "metadata"},
            tags=["tag1", "tag2"]
        )
        
        assert file_id is not None
        mock_connector.upload_file.assert_called_once()
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
    
    @patch.object(CloudStorageEngine, '_get_connector')
    async def test_download_file(self, mock_get_connector, engine):
        """Test file download"""
        # Mock connector
        mock_connector = AsyncMock()
        
        async def mock_download_generator():
            yield b"chunk1"
            yield b"chunk2"
        
        mock_connector.download_file.return_value = mock_download_generator()
        mock_get_connector.return_value = mock_connector
        
        # Test file download
        chunks = []
        async for chunk in engine.download_file("test-connection", "test/file.txt"):
            chunks.append(chunk)
        
        assert len(chunks) == 2
        assert chunks[0] == b"chunk1"
        assert chunks[1] == b"chunk2"
    
    @patch('scrollintel.engines.cloud_storage_engine.get_database_session')
    @patch.object(CloudStorageEngine, '_get_connector')
    async def test_delete_file(self, mock_get_connector, mock_db_session, engine):
        """Test file deletion"""
        # Mock connector
        mock_connector = AsyncMock()
        mock_connector.delete_file.return_value = True
        mock_get_connector.return_value = mock_connector
        
        # Mock database session
        mock_session = AsyncMock()
        mock_db_session.return_value.__aenter__.return_value = mock_session
        
        # Test file deletion
        result = await engine.delete_file("test-connection", "test/file.txt")
        
        assert result is True
        mock_connector.delete_file.assert_called_once()
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()
    
    @patch.object(CloudStorageEngine, '_get_connector')
    async def test_list_files(self, mock_get_connector, engine):
        """Test file listing"""
        # Mock connector
        mock_connector = AsyncMock()
        mock_connector.list_files.return_value = [
            {'file_path': 'file1.txt', 'size': 100},
            {'file_path': 'file2.txt', 'size': 200}
        ]
        mock_get_connector.return_value = mock_connector
        
        # Test file listing
        files = await engine.list_files("test-connection")
        
        assert len(files) == 2
        assert files[0]['file_path'] == 'file1.txt'
        assert files[1]['file_path'] == 'file2.txt'
    
    @patch('scrollintel.engines.cloud_storage_engine.get_database_session')
    async def test_search_files(self, mock_db_session, engine):
        """Test file search"""
        mock_session = AsyncMock()
        mock_db_session.return_value.__aenter__.return_value = mock_session
        
        # Mock search results
        mock_result = Mock()
        mock_file = Mock()
        mock_file.id = "file1"
        mock_file.connection_id = "conn1"
        mock_file.file_path = "test.txt"
        mock_file.file_name = "test.txt"
        mock_file.file_size = 100
        mock_file.file_format = "text"
        mock_file.mime_type = "text/plain"
        mock_file.checksum = "abc123"
        mock_file.metadata = {}
        mock_file.tags = ["tag1"]
        mock_file.created_at = datetime.utcnow()
        mock_file.last_modified = None
        mock_file.indexed_at = datetime.utcnow()
        
        mock_result.fetchall.return_value = [mock_file]
        mock_session.execute.return_value = mock_result
        
        # Test file search
        files = await engine.search_files(
            connection_id="conn1",
            file_format=FileFormat.TEXT,
            tags=["tag1"]
        )
        
        assert len(files) == 1
        assert files[0].id == "file1"
        assert files[0].file_name == "test.txt"
    
    @patch('scrollintel.engines.cloud_storage_engine.get_database_session')
    async def test_get_storage_stats(self, mock_db_session, engine):
        """Test storage statistics"""
        mock_session = AsyncMock()
        mock_db_session.return_value.__aenter__.return_value = mock_session
        
        # Mock stats results
        mock_conn_result = Mock()
        mock_conn_result.fetchall.return_value = [
            Mock(total=2, provider='aws_s3'),
            Mock(total=1, provider='azure_blob')
        ]
        
        mock_file_result = Mock()
        mock_file_result.fetchall.return_value = [
            Mock(total=10, total_size=1000, file_format='csv'),
            Mock(total=5, total_size=500, file_format='json')
        ]
        
        mock_session.execute.side_effect = [mock_conn_result, mock_file_result]
        
        # Test stats retrieval
        stats = await engine.get_storage_stats()
        
        assert stats['total_connections'] == 3
        assert stats['total_files'] == 15
        assert stats['total_size'] == 1500
        assert 'aws_s3' in stats['providers']
        assert 'csv' in stats['file_formats']


class TestCloudStorageAPI:
    """Test cloud storage API endpoints"""
    
    @pytest.fixture
    def mock_engine(self):
        with patch('scrollintel.api.routes.cloud_storage_routes.cloud_storage_engine') as mock:
            yield mock
    
    @pytest.fixture
    def mock_auth(self):
        with patch('scrollintel.api.routes.cloud_storage_routes.get_current_user') as mock:
            mock.return_value = {"user_id": "test-user"}
            yield mock
    
    async def test_create_connection_endpoint(self, mock_engine, mock_auth):
        """Test connection creation endpoint"""
        from scrollintel.api.routes.cloud_storage_routes import create_connection
        
        mock_engine.create_connection.return_value = "test-connection-id"
        
        config = CloudStorageConnectionConfig(
            name="Test Connection",
            provider=CloudProvider.AWS_S3,
            config={"bucket_name": "test"},
            credentials={"access_key": "test"}
        )
        
        result = await create_connection(config, {"user_id": "test-user"})
        
        assert result["success"] is True
        assert result["connection_id"] == "test-connection-id"
        mock_engine.create_connection.assert_called_once_with(config)
    
    async def test_list_connections_endpoint(self, mock_engine, mock_auth):
        """Test list connections endpoint"""
        from scrollintel.api.routes.cloud_storage_routes import list_connections
        
        # Mock connection objects
        mock_connection = Mock()
        mock_connection.id = "conn1"
        mock_connection.name = "Test Connection"
        mock_connection.provider = "aws_s3"
        mock_connection.status = "active"
        mock_connection.last_sync = None
        mock_connection.created_at = datetime.utcnow()
        
        mock_engine.list_connections.return_value = [mock_connection]
        
        result = await list_connections({"user_id": "test-user"})
        
        assert len(result) == 1
        assert result[0].id == "conn1"
        assert result[0].name == "Test Connection"
    
    async def test_upload_file_endpoint(self, mock_engine, mock_auth):
        """Test file upload endpoint"""
        from scrollintel.api.routes.cloud_storage_routes import upload_file
        from fastapi import UploadFile
        
        mock_engine.upload_file.return_value = "test-file-id"
        
        # Mock upload file
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "test.txt"
        mock_file.file = BytesIO(b"test content")
        
        result = await upload_file(
            connection_id="test-connection",
            file=mock_file,
            file_path="test/file.txt",
            tags="tag1,tag2",
            encrypt=True,
            current_user={"user_id": "test-user"}
        )
        
        assert result["success"] is True
        assert result["file_id"] == "test-file-id"
        mock_engine.upload_file.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])