"""
Cloud Storage Integration Demo
Demonstrates cloud storage capabilities including AWS S3, Azure Blob, and Google Cloud Storage.
"""

import asyncio
import json
from io import BytesIO
from datetime import datetime

from scrollintel.engines.cloud_storage_engine import CloudStorageEngine
from scrollintel.models.cloud_storage_models import (
    CloudStorageConnectionConfig, CloudProvider, FileFormat
)


class CloudStorageDemo:
    """Demo class for cloud storage integration"""
    
    def __init__(self):
        self.engine = CloudStorageEngine()
        self.connections = {}
    
    async def demo_connection_management(self):
        """Demo connection creation and management"""
        print("=== Cloud Storage Connection Management Demo ===\n")
        
        # Demo AWS S3 connection
        print("1. Creating AWS S3 Connection...")
        s3_config = CloudStorageConnectionConfig(
            name="Demo S3 Connection",
            provider=CloudProvider.AWS_S3,
            config={
                "bucket_name": "scrollintel-demo-bucket",
                "region": "us-east-1"
            },
            credentials={
                "access_key_id": "demo_access_key",
                "secret_access_key": "demo_secret_key"
            }
        )
        
        try:
            s3_connection_id = await self.engine.create_connection(s3_config)
            self.connections['s3'] = s3_connection_id
            print(f"✓ S3 Connection created: {s3_connection_id}")
        except Exception as e:
            print(f"✗ S3 Connection failed: {e}")
        
        # Demo Azure Blob connection
        print("\n2. Creating Azure Blob Connection...")
        azure_config = CloudStorageConnectionConfig(
            name="Demo Azure Connection",
            provider=CloudProvider.AZURE_BLOB,
            config={
                "container_name": "scrollintel-demo",
                "account_name": "scrollinteldemo"
            },
            credentials={
                "account_key": "demo_account_key"
            }
        )
        
        try:
            azure_connection_id = await self.engine.create_connection(azure_config)
            self.connections['azure'] = azure_connection_id
            print(f"✓ Azure Connection created: {azure_connection_id}")
        except Exception as e:
            print(f"✗ Azure Connection failed: {e}")
        
        # Demo Google Cloud Storage connection
        print("\n3. Creating Google Cloud Storage Connection...")
        gcs_config = CloudStorageConnectionConfig(
            name="Demo GCS Connection",
            provider=CloudProvider.GOOGLE_CLOUD,
            config={
                "bucket_name": "scrollintel-demo-bucket",
                "project_id": "scrollintel-demo"
            },
            credentials={
                "service_account_key": {
                    "type": "service_account",
                    "project_id": "scrollintel-demo",
                    "private_key_id": "demo_key_id",
                    "private_key": "demo_private_key",
                    "client_email": "demo@scrollintel-demo.iam.gserviceaccount.com"
                }
            }
        )
        
        try:
            gcs_connection_id = await self.engine.create_connection(gcs_config)
            self.connections['gcs'] = gcs_connection_id
            print(f"✓ GCS Connection created: {gcs_connection_id}")
        except Exception as e:
            print(f"✗ GCS Connection failed: {e}")
        
        # List all connections
        print("\n4. Listing all connections...")
        try:
            connections = await self.engine.list_connections()
            for conn in connections:
                print(f"   - {conn.name} ({conn.provider}): {conn.status}")
        except Exception as e:
            print(f"✗ Failed to list connections: {e}")
    
    async def demo_file_operations(self):
        """Demo file upload, download, and management"""
        print("\n=== File Operations Demo ===\n")
        
        if not self.connections:
            print("No connections available for file operations demo")
            return
        
        # Use first available connection
        connection_id = list(self.connections.values())[0]
        
        # Demo file uploads with different formats
        print("1. Uploading different file types...")
        
        # CSV file
        csv_content = "name,age,city\nJohn Doe,30,New York\nJane Smith,25,Los Angeles\nBob Johnson,35,Chicago"
        csv_file = BytesIO(csv_content.encode())
        
        try:
            csv_file_id = await self.engine.upload_file(
                connection_id=connection_id,
                file_path="data/employees.csv",
                file_data=csv_file,
                metadata={"department": "HR", "year": "2024"},
                tags=["hr", "employees", "data"],
                encrypt=True
            )
            print(f"✓ CSV file uploaded: {csv_file_id}")
        except Exception as e:
            print(f"✗ CSV upload failed: {e}")
        
        # JSON file
        json_data = {
            "company": "ScrollIntel",
            "employees": [
                {"name": "Alice", "role": "Engineer", "skills": ["Python", "AI"]},
                {"name": "Bob", "role": "Designer", "skills": ["UI/UX", "Figma"]}
            ],
            "founded": "2024"
        }
        json_file = BytesIO(json.dumps(json_data, indent=2).encode())
        
        try:
            json_file_id = await self.engine.upload_file(
                connection_id=connection_id,
                file_path="config/company.json",
                file_data=json_file,
                metadata={"type": "configuration", "version": "1.0"},
                tags=["config", "company"],
                encrypt=True
            )
            print(f"✓ JSON file uploaded: {json_file_id}")
        except Exception as e:
            print(f"✗ JSON upload failed: {e}")
        
        # Text file
        text_content = """ScrollIntel Cloud Storage Integration
        
This is a demo text file showcasing the cloud storage capabilities.
Features include:
- Multi-cloud support (AWS S3, Azure Blob, Google Cloud)
- Automatic file format detection
- Metadata extraction
- Encryption and security
- Streaming uploads/downloads
"""
        text_file = BytesIO(text_content.encode())
        
        try:
            text_file_id = await self.engine.upload_file(
                connection_id=connection_id,
                file_path="docs/readme.txt",
                file_data=text_file,
                metadata={"author": "ScrollIntel", "category": "documentation"},
                tags=["docs", "readme"],
                encrypt=False
            )
            print(f"✓ Text file uploaded: {text_file_id}")
        except Exception as e:
            print(f"✗ Text upload failed: {e}")
        
        # List files
        print("\n2. Listing uploaded files...")
        try:
            files = await self.engine.list_files(connection_id, limit=10)
            for file_info in files:
                print(f"   - {file_info['file_path']} ({file_info['size']} bytes)")
        except Exception as e:
            print(f"✗ File listing failed: {e}")
        
        # Download file demo
        print("\n3. Downloading file...")
        try:
            chunks = []
            async for chunk in self.engine.download_file(connection_id, "data/employees.csv"):
                chunks.append(chunk)
            
            downloaded_content = b''.join(chunks).decode()
            print(f"✓ Downloaded file content preview:")
            print(f"   {downloaded_content[:100]}...")
        except Exception as e:
            print(f"✗ File download failed: {e}")
    
    async def demo_metadata_and_search(self):
        """Demo metadata extraction and file search"""
        print("\n=== Metadata and Search Demo ===\n")
        
        # Search files by format
        print("1. Searching files by format...")
        try:
            csv_files = await self.engine.search_files(
                file_format=FileFormat.CSV,
                limit=5
            )
            print(f"✓ Found {len(csv_files)} CSV files:")
            for file_meta in csv_files:
                print(f"   - {file_meta.file_name} ({file_meta.file_size} bytes)")
                if file_meta.metadata:
                    print(f"     Metadata: {json.dumps(file_meta.metadata, indent=6)}")
        except Exception as e:
            print(f"✗ CSV search failed: {e}")
        
        # Search files by tags
        print("\n2. Searching files by tags...")
        try:
            tagged_files = await self.engine.search_files(
                tags=["hr", "employees"],
                limit=5
            )
            print(f"✓ Found {len(tagged_files)} files with HR/employee tags:")
            for file_meta in tagged_files:
                print(f"   - {file_meta.file_name} (tags: {file_meta.tags})")
        except Exception as e:
            print(f"✗ Tag search failed: {e}")
        
        # Get detailed metadata
        print("\n3. Getting detailed file metadata...")
        try:
            # Search for any file to get metadata
            all_files = await self.engine.search_files(limit=1)
            if all_files:
                file_meta = all_files[0]
                print(f"✓ Detailed metadata for {file_meta.file_name}:")
                print(f"   File ID: {file_meta.id}")
                print(f"   Path: {file_meta.file_path}")
                print(f"   Format: {file_meta.file_format}")
                print(f"   MIME Type: {file_meta.mime_type}")
                print(f"   Size: {file_meta.file_size} bytes")
                print(f"   Checksum: {file_meta.checksum}")
                print(f"   Created: {file_meta.created_at}")
                print(f"   Tags: {file_meta.tags}")
                print(f"   Metadata: {json.dumps(file_meta.metadata, indent=6)}")
        except Exception as e:
            print(f"✗ Metadata retrieval failed: {e}")
    
    async def demo_storage_statistics(self):
        """Demo storage statistics and analytics"""
        print("\n=== Storage Statistics Demo ===\n")
        
        try:
            stats = await self.engine.get_storage_stats()
            
            print("📊 Storage Statistics:")
            print(f"   Total Connections: {stats['total_connections']}")
            print(f"   Active Connections: {stats['active_connections']}")
            print(f"   Total Files: {stats['total_files']}")
            print(f"   Total Storage Size: {stats['total_size']} bytes")
            
            print("\n📈 Provider Distribution:")
            for provider, count in stats['providers'].items():
                print(f"   {provider}: {count} connections")
            
            print("\n📁 File Format Distribution:")
            for format_type, count in stats['file_formats'].items():
                print(f"   {format_type}: {count} files")
            
        except Exception as e:
            print(f"✗ Statistics retrieval failed: {e}")
    
    async def demo_advanced_features(self):
        """Demo advanced features like batch operations and sync"""
        print("\n=== Advanced Features Demo ===\n")
        
        if not self.connections:
            print("No connections available for advanced features demo")
            return
        
        connection_id = list(self.connections.values())[0]
        
        # Demo batch upload simulation
        print("1. Batch Upload Simulation...")
        batch_files = [
            ("reports/q1_2024.csv", "quarter,revenue,expenses\nQ1,100000,75000"),
            ("reports/q2_2024.csv", "quarter,revenue,expenses\nQ2,120000,80000"),
            ("reports/summary.json", '{"total_revenue": 220000, "total_expenses": 155000}')
        ]
        
        uploaded_files = []
        for file_path, content in batch_files:
            try:
                file_data = BytesIO(content.encode())
                file_id = await self.engine.upload_file(
                    connection_id=connection_id,
                    file_path=file_path,
                    file_data=file_data,
                    metadata={"batch": "quarterly_reports", "year": "2024"},
                    tags=["reports", "2024", "financial"]
                )
                uploaded_files.append((file_path, file_id))
                print(f"   ✓ Uploaded: {file_path}")
            except Exception as e:
                print(f"   ✗ Failed to upload {file_path}: {e}")
        
        print(f"✓ Batch upload completed: {len(uploaded_files)} files")
        
        # Demo connection sync
        print("\n2. Connection Sync Demo...")
        try:
            # This would sync metadata with actual cloud storage
            print("   Syncing connection metadata...")
            files = await self.engine.list_files(connection_id, limit=100)
            print(f"   ✓ Sync completed: {len(files)} files in storage")
        except Exception as e:
            print(f"   ✗ Sync failed: {e}")
        
        # Demo file cleanup
        print("\n3. File Cleanup Demo...")
        cleanup_count = 0
        try:
            # Delete some demo files
            for file_path, _ in uploaded_files[:2]:  # Delete first 2 files
                success = await self.engine.delete_file(connection_id, file_path)
                if success:
                    cleanup_count += 1
                    print(f"   ✓ Deleted: {file_path}")
            
            print(f"✓ Cleanup completed: {cleanup_count} files deleted")
        except Exception as e:
            print(f"✗ Cleanup failed: {e}")
    
    async def run_full_demo(self):
        """Run the complete cloud storage demo"""
        print("🚀 ScrollIntel Cloud Storage Integration Demo")
        print("=" * 50)
        
        try:
            await self.demo_connection_management()
            await self.demo_file_operations()
            await self.demo_metadata_and_search()
            await self.demo_storage_statistics()
            await self.demo_advanced_features()
            
            print("\n" + "=" * 50)
            print("✅ Cloud Storage Demo Completed Successfully!")
            print("\nKey Features Demonstrated:")
            print("• Multi-cloud provider support (AWS S3, Azure Blob, Google Cloud)")
            print("• Automatic file format detection and metadata extraction")
            print("• Secure file upload/download with encryption")
            print("• Advanced search and filtering capabilities")
            print("• Comprehensive storage analytics")
            print("• Batch operations and connection management")
            
        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """Main demo function"""
    demo = CloudStorageDemo()
    await demo.run_full_demo()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())