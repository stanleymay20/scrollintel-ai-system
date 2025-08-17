
import os
from minio import Minio
from typing import Optional
import asyncio
import aiofiles

class DistributedStorage:
    def __init__(self):
        self.minio_client = None
        self.bucket_name = os.getenv('OBJECT_STORAGE_BUCKET', 'scrollintel-data')
        self.setup_minio()
    
    def setup_minio(self):
        """Setup MinIO client for object storage"""
        endpoint = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
        access_key = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
        secret_key = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
        
        self.minio_client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False
        )
        
        # Create bucket if it doesn't exist
        if not self.minio_client.bucket_exists(self.bucket_name):
            self.minio_client.make_bucket(self.bucket_name)
    
    async def upload_file(self, local_path: str, object_name: str) -> bool:
        """Upload file to distributed storage"""
        try:
            self.minio_client.fput_object(
                self.bucket_name, 
                object_name, 
                local_path
            )
            return True
        except Exception as e:
            print(f"Upload failed: {e}")
            return False
    
    async def download_file(self, object_name: str, local_path: str) -> bool:
        """Download file from distributed storage"""
        try:
            self.minio_client.fget_object(
                self.bucket_name,
                object_name,
                local_path
            )
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False
