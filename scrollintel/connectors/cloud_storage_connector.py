"""
Cloud Storage Connector
Provides unified interface for AWS S3, Azure Blob, and Google Cloud Storage.
"""

import asyncio
import hashlib
import mimetypes
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, List, AsyncGenerator, BinaryIO
from io import BytesIO

# Optional cloud provider dependencies
try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

try:
    from azure.storage.blob import BlobServiceClient
    HAS_AZURE = True
except ImportError:
    HAS_AZURE = False

try:
    from google.cloud import storage as gcs
    HAS_GCS = True
except ImportError:
    HAS_GCS = False

try:
    from cryptography.fernet import Fernet
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

from ..models.cloud_storage_models import (
    CloudProvider, FileFormat, ConnectionStatus, FileMetadata
)


class CloudStorageError(Exception):
    """Base exception for cloud storage operations"""
    pass


class ConnectionError(CloudStorageError):
    """Connection-related errors"""
    pass


class UploadError(CloudStorageError):
    """Upload operation errors"""
    pass


class DownloadError(CloudStorageError):
    """Download operation errors"""
    pass


class BaseCloudConnector(ABC):
    """Abstract base class for cloud storage connectors"""
    
    def __init__(self, config: Dict[str, Any], credentials: Dict[str, Any]):
        self.config = config
        self.credentials = credentials
        self._client = None
        self._encryption_key = self._get_encryption_key()
    
    def _get_encryption_key(self):
        """Get encryption key for secure operations"""
        if not HAS_CRYPTOGRAPHY:
            return None
        key = os.environ.get('CLOUD_STORAGE_ENCRYPTION_KEY')
        if key:
            return Fernet(key.encode())
        return None
    
    def _encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data if encryption is enabled"""
        if self._encryption_key:
            return self._encryption_key.encrypt(data)
        return data
    
    def _decrypt_data(self, data: bytes) -> bytes:
        """Decrypt data if encryption is enabled"""
        if self._encryption_key:
            return self._encryption_key.decrypt(data)
        return data
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate MD5 checksum for data integrity"""
        return hashlib.md5(data).hexdigest()
    
    def _detect_file_format(self, file_path: str, content: bytes = None) -> FileFormat:
        """Detect file format from path and content"""
        mime_type, _ = mimetypes.guess_type(file_path)
        
        if mime_type:
            if mime_type.startswith('image/'):
                return FileFormat.IMAGE
            elif mime_type.startswith('video/'):
                return FileFormat.VIDEO
            elif mime_type.startswith('audio/'):
                return FileFormat.AUDIO
            elif mime_type == 'application/json':
                return FileFormat.JSON
            elif mime_type == 'text/csv':
                return FileFormat.CSV
            elif mime_type == 'application/pdf':
                return FileFormat.PDF
            elif mime_type.startswith('text/'):
                return FileFormat.TEXT
        
        # Check file extension
        ext = os.path.splitext(file_path)[1].lower()
        format_map = {
            '.csv': FileFormat.CSV,
            '.json': FileFormat.JSON,
            '.parquet': FileFormat.PARQUET,
            '.xlsx': FileFormat.EXCEL,
            '.xls': FileFormat.EXCEL,
            '.pdf': FileFormat.PDF,
            '.txt': FileFormat.TEXT,
            '.md': FileFormat.TEXT,
        }
        
        return format_map.get(ext, FileFormat.BINARY)
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test connection to cloud storage"""
        pass
    
    @abstractmethod
    async def upload_file(self, file_path: str, data: BinaryIO, 
                         metadata: Dict[str, Any] = None, encrypt: bool = True) -> Dict[str, Any]:
        """Upload file to cloud storage"""
        pass
    
    @abstractmethod
    async def download_file(self, file_path: str, stream: bool = True) -> AsyncGenerator[bytes, None]:
        """Download file from cloud storage"""
        pass
    
    @abstractmethod
    async def delete_file(self, file_path: str) -> bool:
        """Delete file from cloud storage"""
        pass
    
    @abstractmethod
    async def list_files(self, prefix: str = "", limit: int = 1000) -> List[Dict[str, Any]]:
        """List files in cloud storage"""
        pass
    
    @abstractmethod
    async def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Get file metadata"""
        pass


class S3Connector(BaseCloudConnector):
    """AWS S3 connector implementation"""
    
    def __init__(self, config: Dict[str, Any], credentials: Dict[str, Any]):
        super().__init__(config, credentials)
        self.bucket_name = config.get('bucket_name')
        self.region = config.get('region', 'us-east-1')
        
    def _get_client(self):
        """Get S3 client"""
        if not HAS_BOTO3:
            raise CloudStorageError("boto3 library not installed. Install with: pip install boto3")
        
        if not self._client:
            self._client = boto3.client(
                's3',
                aws_access_key_id=self.credentials.get('access_key_id'),
                aws_secret_access_key=self.credentials.get('secret_access_key'),
                region_name=self.region
            )
        return self._client
    
    async def test_connection(self) -> bool:
        """Test S3 connection"""
        try:
            client = self._get_client()
            client.head_bucket(Bucket=self.bucket_name)
            return True
        except Exception as e:
            raise ConnectionError(f"S3 connection failed: {str(e)}")
    
    async def upload_file(self, file_path: str, data: BinaryIO, 
                         metadata: Dict[str, Any] = None, encrypt: bool = True) -> Dict[str, Any]:
        """Upload file to S3"""
        try:
            client = self._get_client()
            
            # Read and optionally encrypt data
            content = data.read()
            if encrypt:
                content = self._encrypt_data(content)
            
            # Calculate checksum
            checksum = self._calculate_checksum(content)
            
            # Prepare metadata
            s3_metadata = metadata or {}
            s3_metadata.update({
                'checksum': checksum,
                'encrypted': str(encrypt),
                'upload_time': datetime.utcnow().isoformat()
            })
            
            # Upload to S3
            client.put_object(
                Bucket=self.bucket_name,
                Key=file_path,
                Body=content,
                Metadata=s3_metadata
            )
            
            return {
                'file_path': file_path,
                'size': len(content),
                'checksum': checksum,
                'metadata': s3_metadata
            }
            
        except Exception as e:
            raise UploadError(f"S3 upload failed: {str(e)}")
    
    async def download_file(self, file_path: str, stream: bool = True) -> AsyncGenerator[bytes, None]:
        """Download file from S3"""
        try:
            client = self._get_client()
            
            if stream:
                # Streaming download
                response = client.get_object(Bucket=self.bucket_name, Key=file_path)
                body = response['Body']
                
                chunk_size = 8192
                while True:
                    chunk = body.read(chunk_size)
                    if not chunk:
                        break
                    
                    # Decrypt if needed
                    if self._encryption_key:
                        chunk = self._decrypt_data(chunk)
                    
                    yield chunk
            else:
                # Full download
                response = client.get_object(Bucket=self.bucket_name, Key=file_path)
                content = response['Body'].read()
                
                # Decrypt if needed
                if self._encryption_key:
                    content = self._decrypt_data(content)
                
                yield content
                
        except Exception as e:
            raise DownloadError(f"S3 download failed: {str(e)}")
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete file from S3"""
        try:
            client = self._get_client()
            client.delete_object(Bucket=self.bucket_name, Key=file_path)
            return True
        except Exception as e:
            raise CloudStorageError(f"S3 delete failed: {str(e)}")
    
    async def list_files(self, prefix: str = "", limit: int = 1000) -> List[Dict[str, Any]]:
        """List files in S3"""
        try:
            client = self._get_client()
            
            paginator = client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=prefix,
                PaginationConfig={'MaxItems': limit}
            )
            
            files = []
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        files.append({
                            'file_path': obj['Key'],
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'],
                            'etag': obj['ETag']
                        })
            
            return files
            
        except Exception as e:
            raise CloudStorageError(f"S3 list failed: {str(e)}")
    
    async def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Get S3 file metadata"""
        try:
            client = self._get_client()
            response = client.head_object(Bucket=self.bucket_name, Key=file_path)
            
            return {
                'file_path': file_path,
                'size': response.get('ContentLength'),
                'last_modified': response.get('LastModified'),
                'content_type': response.get('ContentType'),
                'metadata': response.get('Metadata', {}),
                'etag': response.get('ETag')
            }
            
        except Exception as e:
            raise CloudStorageError(f"S3 metadata retrieval failed: {str(e)}")


class AzureBlobConnector(BaseCloudConnector):
    """Azure Blob Storage connector implementation"""
    
    def __init__(self, config: Dict[str, Any], credentials: Dict[str, Any]):
        super().__init__(config, credentials)
        self.container_name = config.get('container_name')
        self.account_name = config.get('account_name')
    
    def _get_client(self):
        """Get Azure Blob client"""
        if not HAS_AZURE:
            raise CloudStorageError("azure-storage-blob library not installed. Install with: pip install azure-storage-blob")
        
        if not self._client:
            connection_string = (
                f"DefaultEndpointsProtocol=https;"
                f"AccountName={self.account_name};"
                f"AccountKey={self.credentials.get('account_key')};"
                f"EndpointSuffix=core.windows.net"
            )
            self._client = BlobServiceClient.from_connection_string(connection_string)
        return self._client
    
    async def test_connection(self) -> bool:
        """Test Azure Blob connection"""
        try:
            client = self._get_client()
            container_client = client.get_container_client(self.container_name)
            container_client.get_container_properties()
            return True
        except Exception as e:
            raise ConnectionError(f"Azure Blob connection failed: {str(e)}")
    
    async def upload_file(self, file_path: str, data: BinaryIO, 
                         metadata: Dict[str, Any] = None, encrypt: bool = True) -> Dict[str, Any]:
        """Upload file to Azure Blob"""
        try:
            client = self._get_client()
            blob_client = client.get_blob_client(
                container=self.container_name, 
                blob=file_path
            )
            
            # Read and optionally encrypt data
            content = data.read()
            if encrypt:
                content = self._encrypt_data(content)
            
            # Calculate checksum
            checksum = self._calculate_checksum(content)
            
            # Prepare metadata
            blob_metadata = metadata or {}
            blob_metadata.update({
                'checksum': checksum,
                'encrypted': str(encrypt),
                'upload_time': datetime.utcnow().isoformat()
            })
            
            # Upload to Azure Blob
            blob_client.upload_blob(
                content,
                metadata=blob_metadata,
                overwrite=True
            )
            
            return {
                'file_path': file_path,
                'size': len(content),
                'checksum': checksum,
                'metadata': blob_metadata
            }
            
        except Exception as e:
            raise UploadError(f"Azure Blob upload failed: {str(e)}")
    
    async def download_file(self, file_path: str, stream: bool = True) -> AsyncGenerator[bytes, None]:
        """Download file from Azure Blob"""
        try:
            client = self._get_client()
            blob_client = client.get_blob_client(
                container=self.container_name, 
                blob=file_path
            )
            
            if stream:
                # Streaming download
                stream_downloader = blob_client.download_blob()
                
                for chunk in stream_downloader.chunks():
                    # Decrypt if needed
                    if self._encryption_key:
                        chunk = self._decrypt_data(chunk)
                    yield chunk
            else:
                # Full download
                content = blob_client.download_blob().readall()
                
                # Decrypt if needed
                if self._encryption_key:
                    content = self._decrypt_data(content)
                
                yield content
                
        except Exception as e:
            raise DownloadError(f"Azure Blob download failed: {str(e)}")
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete file from Azure Blob"""
        try:
            client = self._get_client()
            blob_client = client.get_blob_client(
                container=self.container_name, 
                blob=file_path
            )
            blob_client.delete_blob()
            return True
        except Exception as e:
            raise CloudStorageError(f"Azure Blob delete failed: {str(e)}")
    
    async def list_files(self, prefix: str = "", limit: int = 1000) -> List[Dict[str, Any]]:
        """List files in Azure Blob"""
        try:
            client = self._get_client()
            container_client = client.get_container_client(self.container_name)
            
            blobs = container_client.list_blobs(name_starts_with=prefix)
            
            files = []
            count = 0
            for blob in blobs:
                if count >= limit:
                    break
                
                files.append({
                    'file_path': blob.name,
                    'size': blob.size,
                    'last_modified': blob.last_modified,
                    'etag': blob.etag
                })
                count += 1
            
            return files
            
        except Exception as e:
            raise CloudStorageError(f"Azure Blob list failed: {str(e)}")
    
    async def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Get Azure Blob file metadata"""
        try:
            client = self._get_client()
            blob_client = client.get_blob_client(
                container=self.container_name, 
                blob=file_path
            )
            
            properties = blob_client.get_blob_properties()
            
            return {
                'file_path': file_path,
                'size': properties.size,
                'last_modified': properties.last_modified,
                'content_type': properties.content_settings.content_type,
                'metadata': properties.metadata,
                'etag': properties.etag
            }
            
        except Exception as e:
            raise CloudStorageError(f"Azure Blob metadata retrieval failed: {str(e)}")


class GoogleCloudConnector(BaseCloudConnector):
    """Google Cloud Storage connector implementation"""
    
    def __init__(self, config: Dict[str, Any], credentials: Dict[str, Any]):
        super().__init__(config, credentials)
        self.bucket_name = config.get('bucket_name')
        self.project_id = config.get('project_id')
    
    def _get_client(self):
        """Get Google Cloud Storage client"""
        if not HAS_GCS:
            raise CloudStorageError("google-cloud-storage library not installed. Install with: pip install google-cloud-storage")
        
        if not self._client:
            # Use service account key from credentials
            service_account_info = self.credentials.get('service_account_key')
            self._client = gcs.Client.from_service_account_info(
                service_account_info,
                project=self.project_id
            )
        return self._client
    
    async def test_connection(self) -> bool:
        """Test Google Cloud Storage connection"""
        try:
            client = self._get_client()
            bucket = client.bucket(self.bucket_name)
            bucket.reload()
            return True
        except Exception as e:
            raise ConnectionError(f"Google Cloud Storage connection failed: {str(e)}")
    
    async def upload_file(self, file_path: str, data: BinaryIO, 
                         metadata: Dict[str, Any] = None, encrypt: bool = True) -> Dict[str, Any]:
        """Upload file to Google Cloud Storage"""
        try:
            client = self._get_client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(file_path)
            
            # Read and optionally encrypt data
            content = data.read()
            if encrypt:
                content = self._encrypt_data(content)
            
            # Calculate checksum
            checksum = self._calculate_checksum(content)
            
            # Prepare metadata
            gcs_metadata = metadata or {}
            gcs_metadata.update({
                'checksum': checksum,
                'encrypted': str(encrypt),
                'upload_time': datetime.utcnow().isoformat()
            })
            
            # Set metadata
            blob.metadata = gcs_metadata
            
            # Upload to Google Cloud Storage
            blob.upload_from_string(content)
            
            return {
                'file_path': file_path,
                'size': len(content),
                'checksum': checksum,
                'metadata': gcs_metadata
            }
            
        except Exception as e:
            raise UploadError(f"Google Cloud Storage upload failed: {str(e)}")
    
    async def download_file(self, file_path: str, stream: bool = True) -> AsyncGenerator[bytes, None]:
        """Download file from Google Cloud Storage"""
        try:
            client = self._get_client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(file_path)
            
            if stream:
                # Streaming download (simulate with chunks)
                content = blob.download_as_bytes()
                
                # Decrypt if needed
                if self._encryption_key:
                    content = self._decrypt_data(content)
                
                # Yield in chunks
                chunk_size = 8192
                for i in range(0, len(content), chunk_size):
                    yield content[i:i + chunk_size]
            else:
                # Full download
                content = blob.download_as_bytes()
                
                # Decrypt if needed
                if self._encryption_key:
                    content = self._decrypt_data(content)
                
                yield content
                
        except Exception as e:
            raise DownloadError(f"Google Cloud Storage download failed: {str(e)}")
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete file from Google Cloud Storage"""
        try:
            client = self._get_client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(file_path)
            blob.delete()
            return True
        except Exception as e:
            raise CloudStorageError(f"Google Cloud Storage delete failed: {str(e)}")
    
    async def list_files(self, prefix: str = "", limit: int = 1000) -> List[Dict[str, Any]]:
        """List files in Google Cloud Storage"""
        try:
            client = self._get_client()
            bucket = client.bucket(self.bucket_name)
            
            blobs = bucket.list_blobs(prefix=prefix, max_results=limit)
            
            files = []
            for blob in blobs:
                files.append({
                    'file_path': blob.name,
                    'size': blob.size,
                    'last_modified': blob.time_created,
                    'etag': blob.etag
                })
            
            return files
            
        except Exception as e:
            raise CloudStorageError(f"Google Cloud Storage list failed: {str(e)}")
    
    async def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Get Google Cloud Storage file metadata"""
        try:
            client = self._get_client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(file_path)
            blob.reload()
            
            return {
                'file_path': file_path,
                'size': blob.size,
                'last_modified': blob.time_created,
                'content_type': blob.content_type,
                'metadata': blob.metadata or {},
                'etag': blob.etag
            }
            
        except Exception as e:
            raise CloudStorageError(f"Google Cloud Storage metadata retrieval failed: {str(e)}")


class CloudStorageConnectorFactory:
    """Factory for creating cloud storage connectors"""
    
    @staticmethod
    def create_connector(provider: CloudProvider, config: Dict[str, Any], 
                        credentials: Dict[str, Any]) -> BaseCloudConnector:
        """Create appropriate connector based on provider"""
        
        if provider == CloudProvider.AWS_S3:
            return S3Connector(config, credentials)
        elif provider == CloudProvider.AZURE_BLOB:
            return AzureBlobConnector(config, credentials)
        elif provider == CloudProvider.GOOGLE_CLOUD:
            return GoogleCloudConnector(config, credentials)
        else:
            raise ValueError(f"Unsupported cloud provider: {provider}")