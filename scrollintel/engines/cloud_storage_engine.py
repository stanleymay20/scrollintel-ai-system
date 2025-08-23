"""
Cloud Storage Engine
Main engine for cloud storage operations including file processing and metadata extraction.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, BinaryIO
from io import BytesIO
import json
import mimetypes

# Optional dependencies with fallbacks
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False

from ..models.cloud_storage_models import (
    CloudStorageConnection, FileMetadata, CloudProvider, FileFormat, 
    ConnectionStatus, CloudStorageConnectionConfig, FileMetadataResponse
)
from ..connectors.cloud_storage_connector import (
    CloudStorageConnectorFactory, BaseCloudConnector, CloudStorageError
)
try:
    from ..core.database import get_database_session
except ImportError:
    # Mock database session for testing
    from contextlib import asynccontextmanager
    
    @asynccontextmanager
    async def get_database_session():
        class MockSession:
            async def add(self, obj): pass
            async def commit(self): pass
            async def get(self, model, id): return None
            async def execute(self, query, params=None): 
                class MockResult:
                    def fetchall(self): return []
                return MockResult()
        yield MockSession()


class FileProcessor:
    """File format detection and processing utilities"""
    
    @staticmethod
    def detect_file_format(file_path: str, content: bytes = None) -> FileFormat:
        """Enhanced file format detection"""
        
        # First, try extension-based detection for common formats
        if '.' in file_path:
            ext = file_path.lower().split('.')[-1]
            ext_format_map = {
                'csv': FileFormat.CSV,
                'json': FileFormat.JSON,
                'parquet': FileFormat.PARQUET,
                'xlsx': FileFormat.EXCEL,
                'xls': FileFormat.EXCEL,
                'pdf': FileFormat.PDF,
                'txt': FileFormat.TEXT,
                'md': FileFormat.TEXT,
                'jpg': FileFormat.IMAGE,
                'jpeg': FileFormat.IMAGE,
                'png': FileFormat.IMAGE,
                'gif': FileFormat.IMAGE,
                'mp4': FileFormat.VIDEO,
                'avi': FileFormat.VIDEO,
                'mov': FileFormat.VIDEO,
                'mp3': FileFormat.AUDIO,
                'wav': FileFormat.AUDIO,
                'flac': FileFormat.AUDIO,
            }
            
            if ext in ext_format_map:
                return ext_format_map[ext]
        
        # Try python-magic if available
        mime_type = None
        if HAS_MAGIC:
            try:
                if content:
                    mime_type = magic.from_buffer(content, mime=True)
                else:
                    mime_type = magic.from_file(file_path, mime=True)
            except Exception:
                pass
        
        # Fallback to mimetypes module
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(file_path)
        
        # Map MIME types to FileFormat enum
        if mime_type:
            mime_format_map = {
                'text/csv': FileFormat.CSV,
                'application/json': FileFormat.JSON,
                'application/octet-stream': FileFormat.PARQUET,  # Parquet files
                'application/vnd.ms-excel': FileFormat.EXCEL,
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': FileFormat.EXCEL,
                'application/pdf': FileFormat.PDF,
                'text/plain': FileFormat.TEXT,
                'text/markdown': FileFormat.TEXT,
            }
            
            if mime_type in mime_format_map:
                return mime_format_map[mime_type]
            
            # Check for image, video, audio
            if mime_type.startswith('image/'):
                return FileFormat.IMAGE
            elif mime_type.startswith('video/'):
                return FileFormat.VIDEO
            elif mime_type.startswith('audio/'):
                return FileFormat.AUDIO
        
        # Final fallback - try to detect from content
        if content:
            try:
                # Try to decode as text and check for CSV patterns
                text = content.decode('utf-8')
                lines = text.strip().split('\n')
                if len(lines) >= 2:
                    # Check if it looks like CSV (comma-separated values)
                    first_line = lines[0]
                    if ',' in first_line and not first_line.startswith('{') and not first_line.startswith('['):
                        return FileFormat.CSV
                
                # Check for JSON
                if text.strip().startswith(('{', '[')):
                    return FileFormat.JSON
                
                return FileFormat.TEXT
            except:
                pass
        
        return FileFormat.BINARY
    
    @staticmethod
    def extract_metadata(file_path: str, content: bytes, file_format: FileFormat) -> Dict[str, Any]:
        """Extract metadata based on file format"""
        metadata = {
            'file_size': len(content),
            'extraction_time': datetime.utcnow().isoformat()
        }
        
        try:
            if file_format == FileFormat.CSV and HAS_PANDAS:
                # CSV metadata
                try:
                    df = pd.read_csv(BytesIO(content))
                    metadata.update({
                        'rows': len(df),
                        'columns': len(df.columns),
                        'column_names': df.columns.tolist(),
                        'data_types': df.dtypes.astype(str).to_dict()
                    })
                except Exception:
                    # Fallback CSV analysis
                    lines = content.decode('utf-8').split('\n')
                    if lines:
                        headers = lines[0].split(',')
                        metadata.update({
                            'rows': len(lines) - 1,
                            'columns': len(headers),
                            'column_names': [h.strip() for h in headers]
                        })
            
            elif file_format == FileFormat.JSON:
                # JSON metadata
                try:
                    data = json.loads(content.decode('utf-8'))
                    metadata.update({
                        'json_type': type(data).__name__,
                        'keys': list(data.keys()) if isinstance(data, dict) else None,
                        'length': len(data) if isinstance(data, (list, dict)) else None
                    })
                except Exception:
                    pass
            
            elif file_format == FileFormat.EXCEL and HAS_PANDAS:
                # Excel metadata
                try:
                    df = pd.read_excel(BytesIO(content))
                    metadata.update({
                        'rows': len(df),
                        'columns': len(df.columns),
                        'column_names': df.columns.tolist(),
                        'sheet_names': ['Sheet1']  # Simplified
                    })
                except Exception:
                    pass
            
            elif file_format == FileFormat.IMAGE and HAS_PIL:
                # Image metadata
                try:
                    img = Image.open(BytesIO(content))
                    metadata.update({
                        'width': img.width,
                        'height': img.height,
                        'mode': img.mode,
                        'format': img.format
                    })
                except Exception:
                    pass
            
            elif file_format == FileFormat.TEXT:
                # Text metadata
                try:
                    text = content.decode('utf-8')
                    metadata.update({
                        'character_count': len(text),
                        'line_count': text.count('\n') + 1,
                        'word_count': len(text.split())
                    })
                except Exception:
                    pass
        
        except Exception as e:
            metadata['extraction_error'] = str(e)
        
        return metadata


class CloudStorageEngine:
    """Main cloud storage engine for managing connections and operations"""
    
    def __init__(self):
        self.file_processor = FileProcessor()
        self._connectors: Dict[str, BaseCloudConnector] = {}
    
    async def create_connection(self, config: CloudStorageConnectionConfig) -> str:
        """Create a new cloud storage connection"""
        try:
            # Generate connection ID
            connection_id = str(uuid.uuid4())
            
            # Create connector to test connection
            connector = CloudStorageConnectorFactory.create_connector(
                config.provider, config.config, config.credentials
            )
            
            # Test connection
            await connector.test_connection()
            
            # Save connection to database
            async with get_database_session() as session:
                connection = CloudStorageConnection(
                    id=connection_id,
                    name=config.name,
                    provider=config.provider.value,
                    config=config.config,
                    credentials=config.credentials,  # Should be encrypted in production
                    status=ConnectionStatus.ACTIVE.value,
                    created_at=datetime.utcnow()
                )
                
                session.add(connection)
                await session.commit()
            
            # Cache connector
            self._connectors[connection_id] = connector
            
            return connection_id
            
        except Exception as e:
            raise CloudStorageError(f"Failed to create connection: {str(e)}")
    
    async def get_connection(self, connection_id: str) -> Optional[CloudStorageConnection]:
        """Get connection by ID"""
        async with get_database_session() as session:
            return await session.get(CloudStorageConnection, connection_id)
    
    async def list_connections(self) -> List[CloudStorageConnection]:
        """List all cloud storage connections"""
        async with get_database_session() as session:
            result = await session.execute(
                "SELECT * FROM cloud_storage_connections ORDER BY created_at DESC"
            )
            return result.fetchall()
    
    async def test_connection(self, connection_id: str) -> bool:
        """Test existing connection"""
        try:
            connector = await self._get_connector(connection_id)
            return await connector.test_connection()
        except Exception as e:
            raise CloudStorageError(f"Connection test failed: {str(e)}")
    
    async def upload_file(self, connection_id: str, file_path: str, 
                         file_data: BinaryIO, metadata: Dict[str, Any] = None,
                         tags: List[str] = None, encrypt: bool = True) -> str:
        """Upload file to cloud storage with metadata extraction"""
        try:
            connector = await self._get_connector(connection_id)
            
            # Read file content for processing
            content = file_data.read()
            file_data.seek(0)  # Reset for upload
            
            # Detect file format
            file_format = self.file_processor.detect_file_format(file_path, content)
            
            # Extract metadata
            extracted_metadata = self.file_processor.extract_metadata(
                file_path, content, file_format
            )
            
            # Merge with provided metadata
            if metadata:
                extracted_metadata.update(metadata)
            
            # Upload file
            upload_result = await connector.upload_file(
                file_path, file_data, extracted_metadata, encrypt
            )
            
            # Save file metadata to database
            file_id = str(uuid.uuid4())
            async with get_database_session() as session:
                file_metadata = FileMetadata(
                    id=file_id,
                    connection_id=connection_id,
                    file_path=file_path,
                    file_name=file_path.split('/')[-1],
                    file_size=upload_result['size'],
                    file_format=file_format.value,
                    mime_type=magic.from_buffer(content, mime=True) if HAS_MAGIC else mimetypes.guess_type(file_path)[0],
                    checksum=upload_result['checksum'],
                    file_metadata=extracted_metadata,
                    tags=tags or [],
                    created_at=datetime.utcnow(),
                    indexed_at=datetime.utcnow()
                )
                
                session.add(file_metadata)
                await session.commit()
            
            return file_id
            
        except Exception as e:
            raise CloudStorageError(f"File upload failed: {str(e)}")
    
    async def download_file(self, connection_id: str, file_path: str, 
                           stream: bool = True):
        """Download file from cloud storage"""
        try:
            connector = await self._get_connector(connection_id)
            
            async for chunk in connector.download_file(file_path, stream):
                yield chunk
                
        except Exception as e:
            raise CloudStorageError(f"File download failed: {str(e)}")
    
    async def delete_file(self, connection_id: str, file_path: str) -> bool:
        """Delete file from cloud storage"""
        try:
            connector = await self._get_connector(connection_id)
            
            # Delete from cloud storage
            success = await connector.delete_file(file_path)
            
            if success:
                # Remove from database
                async with get_database_session() as session:
                    await session.execute(
                        "DELETE FROM file_metadata WHERE connection_id = ? AND file_path = ?",
                        (connection_id, file_path)
                    )
                    await session.commit()
            
            return success
            
        except Exception as e:
            raise CloudStorageError(f"File deletion failed: {str(e)}")
    
    async def list_files(self, connection_id: str, prefix: str = "", 
                        limit: int = 1000) -> List[Dict[str, Any]]:
        """List files in cloud storage"""
        try:
            connector = await self._get_connector(connection_id)
            return await connector.list_files(prefix, limit)
        except Exception as e:
            raise CloudStorageError(f"File listing failed: {str(e)}")
    
    async def get_file_metadata(self, file_id: str) -> Optional[FileMetadataResponse]:
        """Get file metadata by ID"""
        async with get_database_session() as session:
            metadata = await session.get(FileMetadata, file_id)
            if metadata:
                return FileMetadataResponse(
                    id=metadata.id,
                    connection_id=metadata.connection_id,
                    file_path=metadata.file_path,
                    file_name=metadata.file_name,
                    file_size=metadata.file_size,
                    file_format=FileFormat(metadata.file_format) if metadata.file_format else None,
                    mime_type=metadata.mime_type,
                    checksum=metadata.checksum,
                    file_metadata=metadata.file_metadata or {},
                    tags=metadata.tags or [],
                    created_at=metadata.created_at,
                    last_modified=metadata.last_modified,
                    indexed_at=metadata.indexed_at
                )
            return None
    
    async def search_files(self, connection_id: str = None, file_format: FileFormat = None,
                          tags: List[str] = None, limit: int = 100) -> List[FileMetadataResponse]:
        """Search files by various criteria"""
        try:
            query = "SELECT * FROM file_metadata WHERE 1=1"
            params = []
            
            if connection_id:
                query += " AND connection_id = ?"
                params.append(connection_id)
            
            if file_format:
                query += " AND file_format = ?"
                params.append(file_format.value)
            
            if tags:
                # Simple tag search (in production, use proper JSON queries)
                for tag in tags:
                    query += " AND tags LIKE ?"
                    params.append(f'%{tag}%')
            
            query += f" ORDER BY indexed_at DESC LIMIT {limit}"
            
            async with get_database_session() as session:
                result = await session.execute(query, params)
                files = result.fetchall()
                
                return [
                    FileMetadataResponse(
                        id=file.id,
                        connection_id=file.connection_id,
                        file_path=file.file_path,
                        file_name=file.file_name,
                        file_size=file.file_size,
                        file_format=FileFormat(file.file_format) if file.file_format else None,
                        mime_type=file.mime_type,
                        checksum=file.checksum,
                        file_metadata=file.file_metadata or {},
                        tags=file.tags or [],
                        created_at=file.created_at,
                        last_modified=file.last_modified,
                        indexed_at=file.indexed_at
                    )
                    for file in files
                ]
                
        except Exception as e:
            raise CloudStorageError(f"File search failed: {str(e)}")
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get cloud storage statistics"""
        try:
            async with get_database_session() as session:
                # Total connections
                conn_result = await session.execute(
                    "SELECT COUNT(*) as total, provider FROM cloud_storage_connections GROUP BY provider"
                )
                connections = conn_result.fetchall()
                
                # Total files
                file_result = await session.execute(
                    "SELECT COUNT(*) as total, SUM(file_size) as total_size, file_format "
                    "FROM file_metadata GROUP BY file_format"
                )
                files = file_result.fetchall()
                
                return {
                    'total_connections': sum(c.total for c in connections),
                    'active_connections': len([c for c in connections if c.provider]),
                    'total_files': sum(f.total for f in files),
                    'total_size': sum(f.total_size or 0 for f in files),
                    'providers': {c.provider: c.total for c in connections},
                    'file_formats': {f.file_format: f.total for f in files}
                }
                
        except Exception as e:
            raise CloudStorageError(f"Stats retrieval failed: {str(e)}")
    
    async def _get_connector(self, connection_id: str) -> BaseCloudConnector:
        """Get or create connector for connection"""
        if connection_id not in self._connectors:
            connection = await self.get_connection(connection_id)
            if not connection:
                raise CloudStorageError(f"Connection not found: {connection_id}")
            
            connector = CloudStorageConnectorFactory.create_connector(
                CloudProvider(connection.provider),
                connection.config,
                connection.credentials
            )
            
            self._connectors[connection_id] = connector
        
        return self._connectors[connection_id]