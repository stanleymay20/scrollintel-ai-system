"""
Production-Grade Cloud Storage for Visual Generation Content
Handles AWS S3, Google Cloud Storage, and CDN integration
"""

import asyncio
import logging
import os
import hashlib
import mimetypes
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import boto3
from botocore.exceptions import ClientError
from google.cloud import storage as gcs
import aiofiles
import aiohttp

logger = logging.getLogger(__name__)

class StorageProvider(Enum):
    AWS_S3 = "aws_s3"
    GOOGLE_CLOUD = "google_cloud"
    AZURE_BLOB = "azure_blob"

class ContentType(Enum):
    IMAGE = "image"
    VIDEO = "video"
    THUMBNAIL = "thumbnail"
    METADATA = "metadata"

@dataclass
class StorageConfig:
    provider: StorageProvider
    bucket_name: str
    region: str
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    project_id: Optional[str] = None
    cdn_domain: Optional[str] = None
    encryption_enabled: bool = True

@dataclass
class StoredContent:
    content_id: str
    storage_key: str
    content_type: ContentType
    file_size: int
    mime_type: str
    storage_url: str
    cdn_url: Optional[str]
    created_at: datetime
    expires_at: Optional[datetime]
    metadata: Dict[str, Any]

class CloudStorageManager:
    """Manages cloud storage operations for visual generation content"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.s3_client = None
        self.gcs_client = None
        
        # Initialize storage clients
        self._initialize_clients()
        
        # Content organization
        self.folder_structure = {
            ContentType.IMAGE: "images",
            ContentType.VIDEO: "videos", 
            ContentType.THUMBNAIL: "thumbnails",
            ContentType.METADATA: "metadata"
        }
        
        # CDN configuration
        self.cdn_cache_control = {
            ContentType.IMAGE: "public, max-age=31536000",  # 1 year
            ContentType.VIDEO: "public, max-age=31536000",  # 1 year
            ContentType.THUMBNAIL: "public, max-age=86400",  # 1 day
            ContentType.METADATA: "private, max-age=3600"   # 1 hour
        }
    
    def _initialize_clients(self):
        """Initialize cloud storage clients"""
        try:
            if self.config.provider == StorageProvider.AWS_S3:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=self.config.access_key,
                    aws_secret_access_key=self.config.secret_key,
                    region_name=self.config.region
                )
                logger.info("AWS S3 client initialized")
                
            elif self.config.provider == StorageProvider.GOOGLE_CLOUD:
                if self.config.project_id:
                    self.gcs_client = gcs.Client(project=self.config.project_id)
                else:
                    self.gcs_client = gcs.Client()
                logger.info("Google Cloud Storage client initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize storage client: {str(e)}")
            raise
    
    def _generate_storage_key(self, content_id: str, content_type: ContentType, 
                            file_extension: str) -> str:
        """Generate organized storage key"""
        # Create date-based folder structure
        now = datetime.utcnow()
        date_path = f"{now.year}/{now.month:02d}/{now.day:02d}"
        
        # Add content type folder
        type_folder = self.folder_structure[content_type]
        
        # Generate unique filename
        filename = f"{content_id}.{file_extension}"
        
        return f"{type_folder}/{date_path}/{filename}"
    
    def _get_mime_type(self, filename: str) -> str:
        """Get MIME type for file"""
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type or "application/octet-stream"
    
    async def upload_content(self, content_id: str, file_path: str, 
                           content_type: ContentType, metadata: Dict[str, Any] = None) -> StoredContent:
        """Upload content to cloud storage"""
        try:
            # Get file info
            file_size = os.path.getsize(file_path)
            file_extension = os.path.splitext(file_path)[1][1:]  # Remove dot
            mime_type = self._get_mime_type(file_path)
            
            # Generate storage key
            storage_key = self._generate_storage_key(content_id, content_type, file_extension)
            
            # Prepare metadata
            upload_metadata = {
                "content_id": content_id,
                "content_type": content_type.value,
                "uploaded_at": datetime.utcnow().isoformat(),
                "file_size": str(file_size),
                "mime_type": mime_type
            }
            
            if metadata:
                upload_metadata.update(metadata)
            
            # Upload based on provider
            if self.config.provider == StorageProvider.AWS_S3:
                storage_url = await self._upload_to_s3(file_path, storage_key, mime_type, upload_metadata)
            elif self.config.provider == StorageProvider.GOOGLE_CLOUD:
                storage_url = await self._upload_to_gcs(file_path, storage_key, mime_type, upload_metadata)
            else:
                raise ValueError(f"Unsupported storage provider: {self.config.provider}")
            
            # Generate CDN URL if configured
            cdn_url = None
            if self.config.cdn_domain:
                cdn_url = f"https://{self.config.cdn_domain}/{storage_key}"
            
            # Create stored content record
            stored_content = StoredContent(
                content_id=content_id,
                storage_key=storage_key,
                content_type=content_type,
                file_size=file_size,
                mime_type=mime_type,
                storage_url=storage_url,
                cdn_url=cdn_url,
                created_at=datetime.utcnow(),
                expires_at=None,  # Set based on retention policy
                metadata=upload_metadata
            )
            
            logger.info(f"Successfully uploaded content {content_id} to {storage_key}")
            return stored_content
            
        except Exception as e:
            logger.error(f"Failed to upload content {content_id}: {str(e)}")
            raise
    
    async def _upload_to_s3(self, file_path: str, storage_key: str, 
                          mime_type: str, metadata: Dict[str, Any]) -> str:
        """Upload file to AWS S3"""
        try:
            # Prepare upload parameters
            upload_params = {
                'Bucket': self.config.bucket_name,
                'Key': storage_key,
                'ContentType': mime_type,
                'Metadata': {k: str(v) for k, v in metadata.items()},
                'CacheControl': self.cdn_cache_control.get(
                    ContentType(metadata.get('content_type', 'image')), 
                    'public, max-age=3600'
                )
            }
            
            # Add encryption if enabled
            if self.config.encryption_enabled:
                upload_params['ServerSideEncryption'] = 'AES256'
            
            # Upload file
            await asyncio.to_thread(
                self.s3_client.upload_file,
                file_path,
                self.config.bucket_name,
                storage_key,
                ExtraArgs=upload_params
            )
            
            # Generate storage URL
            storage_url = f"https://{self.config.bucket_name}.s3.{self.config.region}.amazonaws.com/{storage_key}"
            
            return storage_url
            
        except ClientError as e:
            logger.error(f"S3 upload failed: {str(e)}")
            raise
    
    async def _upload_to_gcs(self, file_path: str, storage_key: str, 
                           mime_type: str, metadata: Dict[str, Any]) -> str:
        """Upload file to Google Cloud Storage"""
        try:
            # Get bucket
            bucket = self.gcs_client.bucket(self.config.bucket_name)
            blob = bucket.blob(storage_key)
            
            # Set metadata
            blob.metadata = {k: str(v) for k, v in metadata.items()}
            blob.content_type = mime_type
            
            # Set cache control
            content_type_enum = ContentType(metadata.get('content_type', 'image'))
            blob.cache_control = self.cdn_cache_control.get(content_type_enum, 'public, max-age=3600')
            
            # Upload file
            await asyncio.to_thread(blob.upload_from_filename, file_path)
            
            # Generate storage URL
            storage_url = f"https://storage.googleapis.com/{self.config.bucket_name}/{storage_key}"
            
            return storage_url
            
        except Exception as e:
            logger.error(f"GCS upload failed: {str(e)}")
            raise
    
    async def download_content(self, storage_key: str, local_path: str) -> bool:
        """Download content from cloud storage"""
        try:
            if self.config.provider == StorageProvider.AWS_S3:
                await asyncio.to_thread(
                    self.s3_client.download_file,
                    self.config.bucket_name,
                    storage_key,
                    local_path
                )
            elif self.config.provider == StorageProvider.GOOGLE_CLOUD:
                bucket = self.gcs_client.bucket(self.config.bucket_name)
                blob = bucket.blob(storage_key)
                await asyncio.to_thread(blob.download_to_filename, local_path)
            
            logger.info(f"Successfully downloaded {storage_key} to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {storage_key}: {str(e)}")
            return False
    
    async def delete_content(self, storage_key: str) -> bool:
        """Delete content from cloud storage"""
        try:
            if self.config.provider == StorageProvider.AWS_S3:
                await asyncio.to_thread(
                    self.s3_client.delete_object,
                    Bucket=self.config.bucket_name,
                    Key=storage_key
                )
            elif self.config.provider == StorageProvider.GOOGLE_CLOUD:
                bucket = self.gcs_client.bucket(self.config.bucket_name)
                blob = bucket.blob(storage_key)
                await asyncio.to_thread(blob.delete)
            
            logger.info(f"Successfully deleted {storage_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete {storage_key}: {str(e)}")
            return False
    
    async def generate_presigned_url(self, storage_key: str, expiration: int = 3600) -> Optional[str]:
        """Generate presigned URL for temporary access"""
        try:
            if self.config.provider == StorageProvider.AWS_S3:
                url = await asyncio.to_thread(
                    self.s3_client.generate_presigned_url,
                    'get_object',
                    Params={'Bucket': self.config.bucket_name, 'Key': storage_key},
                    ExpiresIn=expiration
                )
                return url
            elif self.config.provider == StorageProvider.GOOGLE_CLOUD:
                bucket = self.gcs_client.bucket(self.config.bucket_name)
                blob = bucket.blob(storage_key)
                url = await asyncio.to_thread(
                    blob.generate_signed_url,
                    expiration=datetime.utcnow() + timedelta(seconds=expiration)
                )
                return url
            
        except Exception as e:
            logger.error(f"Failed to generate presigned URL for {storage_key}: {str(e)}")
            return None
    
    async def list_content(self, prefix: str = "", limit: int = 100) -> List[Dict[str, Any]]:
        """List content in storage"""
        try:
            content_list = []
            
            if self.config.provider == StorageProvider.AWS_S3:
                response = await asyncio.to_thread(
                    self.s3_client.list_objects_v2,
                    Bucket=self.config.bucket_name,
                    Prefix=prefix,
                    MaxKeys=limit
                )
                
                for obj in response.get('Contents', []):
                    content_list.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'],
                        'etag': obj['ETag']
                    })
                    
            elif self.config.provider == StorageProvider.GOOGLE_CLOUD:
                bucket = self.gcs_client.bucket(self.config.bucket_name)
                blobs = await asyncio.to_thread(
                    list,
                    bucket.list_blobs(prefix=prefix, max_results=limit)
                )
                
                for blob in blobs:
                    content_list.append({
                        'key': blob.name,
                        'size': blob.size,
                        'last_modified': blob.updated,
                        'etag': blob.etag
                    })
            
            return content_list
            
        except Exception as e:
            logger.error(f"Failed to list content: {str(e)}")
            return []
    
    async def get_storage_metrics(self) -> Dict[str, Any]:
        """Get storage usage metrics"""
        try:
            metrics = {
                'total_objects': 0,
                'total_size_bytes': 0,
                'content_types': {},
                'storage_cost_estimate': 0.0
            }
            
            # List all content
            all_content = await self.list_content(limit=10000)  # Adjust as needed
            
            for content in all_content:
                metrics['total_objects'] += 1
                metrics['total_size_bytes'] += content['size']
                
                # Categorize by content type based on path
                key_parts = content['key'].split('/')
                if len(key_parts) > 0:
                    content_type = key_parts[0]
                    if content_type not in metrics['content_types']:
                        metrics['content_types'][content_type] = {'count': 0, 'size_bytes': 0}
                    
                    metrics['content_types'][content_type]['count'] += 1
                    metrics['content_types'][content_type]['size_bytes'] += content['size']
            
            # Estimate storage costs (simplified)
            size_gb = metrics['total_size_bytes'] / (1024**3)
            if self.config.provider == StorageProvider.AWS_S3:
                metrics['storage_cost_estimate'] = size_gb * 0.023  # $0.023 per GB/month
            elif self.config.provider == StorageProvider.GOOGLE_CLOUD:
                metrics['storage_cost_estimate'] = size_gb * 0.020  # $0.020 per GB/month
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get storage metrics: {str(e)}")
            return {}

class ContentDeliveryNetwork:
    """Manages CDN configuration and cache invalidation"""
    
    def __init__(self, cdn_domain: str, provider: str = "cloudflare"):
        self.cdn_domain = cdn_domain
        self.provider = provider
        self.cloudflare_api_key = os.getenv("CLOUDFLARE_API_KEY")
        self.cloudflare_zone_id = os.getenv("CLOUDFLARE_ZONE_ID")
    
    async def invalidate_cache(self, urls: List[str]) -> bool:
        """Invalidate CDN cache for specific URLs"""
        try:
            if self.provider == "cloudflare" and self.cloudflare_api_key:
                return await self._invalidate_cloudflare_cache(urls)
            else:
                logger.warning("CDN cache invalidation not configured")
                return True
                
        except Exception as e:
            logger.error(f"Failed to invalidate CDN cache: {str(e)}")
            return False
    
    async def _invalidate_cloudflare_cache(self, urls: List[str]) -> bool:
        """Invalidate Cloudflare cache"""
        try:
            headers = {
                'Authorization': f'Bearer {self.cloudflare_api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'files': urls
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f'https://api.cloudflare.com/client/v4/zones/{self.cloudflare_zone_id}/purge_cache',
                    headers=headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        logger.info(f"Successfully invalidated {len(urls)} URLs from Cloudflare cache")
                        return True
                    else:
                        logger.error(f"Cloudflare cache invalidation failed: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Cloudflare cache invalidation error: {str(e)}")
            return False
    
    def get_cdn_url(self, storage_key: str) -> str:
        """Get CDN URL for storage key"""
        return f"https://{self.cdn_domain}/{storage_key}"

class ContentArchivalManager:
    """Manages content archival and cleanup policies"""
    
    def __init__(self, storage_manager: CloudStorageManager):
        self.storage_manager = storage_manager
        self.archival_policies = {
            ContentType.IMAGE: timedelta(days=365),  # Archive after 1 year
            ContentType.VIDEO: timedelta(days=180),  # Archive after 6 months
            ContentType.THUMBNAIL: timedelta(days=90),  # Archive after 3 months
            ContentType.METADATA: timedelta(days=30)   # Archive after 1 month
        }
    
    async def cleanup_expired_content(self) -> Dict[str, int]:
        """Clean up expired content based on policies"""
        try:
            cleanup_stats = {
                'scanned': 0,
                'archived': 0,
                'deleted': 0,
                'errors': 0
            }
            
            # List all content
            all_content = await self.storage_manager.list_content(limit=10000)
            cleanup_stats['scanned'] = len(all_content)
            
            current_time = datetime.utcnow()
            
            for content in all_content:
                try:
                    # Determine content type from path
                    content_type = self._get_content_type_from_path(content['key'])
                    if not content_type:
                        continue
                    
                    # Check if content should be archived
                    age = current_time - content['last_modified'].replace(tzinfo=None)
                    archival_age = self.archival_policies.get(content_type)
                    
                    if archival_age and age > archival_age:
                        # Archive or delete content
                        if await self._archive_content(content['key']):
                            cleanup_stats['archived'] += 1
                        else:
                            cleanup_stats['errors'] += 1
                
                except Exception as e:
                    logger.error(f"Error processing content {content['key']}: {str(e)}")
                    cleanup_stats['errors'] += 1
            
            logger.info(f"Cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Content cleanup failed: {str(e)}")
            return {'scanned': 0, 'archived': 0, 'deleted': 0, 'errors': 1}
    
    def _get_content_type_from_path(self, storage_key: str) -> Optional[ContentType]:
        """Extract content type from storage key path"""
        try:
            path_parts = storage_key.split('/')
            if len(path_parts) > 0:
                folder = path_parts[0]
                for content_type, folder_name in self.storage_manager.folder_structure.items():
                    if folder == folder_name:
                        return content_type
            return None
        except:
            return None
    
    async def _archive_content(self, storage_key: str) -> bool:
        """Archive content (move to archive storage class or delete)"""
        try:
            # For now, we'll delete old content
            # In production, you might move to cheaper storage class
            return await self.storage_manager.delete_content(storage_key)
        except Exception as e:
            logger.error(f"Failed to archive content {storage_key}: {str(e)}")
            return False

# Factory function to create storage manager
def create_storage_manager() -> CloudStorageManager:
    """Create storage manager with production configuration"""
    config = StorageConfig(
        provider=StorageProvider(os.getenv("STORAGE_PROVIDER", "aws_s3")),
        bucket_name=os.getenv("STORAGE_BUCKET", "scrollintel-visual-generation"),
        region=os.getenv("STORAGE_REGION", "us-east-1"),
        access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        project_id=os.getenv("GCP_PROJECT_ID"),
        cdn_domain=os.getenv("CDN_DOMAIN"),
        encryption_enabled=os.getenv("STORAGE_ENCRYPTION", "true").lower() == "true"
    )
    
    return CloudStorageManager(config)

# Global instances
storage_manager = create_storage_manager()
cdn_manager = ContentDeliveryNetwork(os.getenv("CDN_DOMAIN", "")) if os.getenv("CDN_DOMAIN") else None
archival_manager = ContentArchivalManager(storage_manager)