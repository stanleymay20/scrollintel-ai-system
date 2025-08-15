#!/usr/bin/env python3
"""
Demo script for ScrollIntel Database and File Processing Performance Optimizations.
Demonstrates database connection pooling, query optimization, file processing with progress tracking,
and background job processing capabilities.
"""

import asyncio
import tempfile
import os
import pandas as pd
import time
from pathlib import Path
from io import BytesIO
from fastapi import UploadFile

from scrollintel.core.database_pool import OptimizedDatabasePool
from scrollintel.core.database_optimizer import DatabaseOptimizer
from scrollintel.core.background_jobs import BackgroundJobProcessor, JobPriority
from scrollintel.engines.file_processor import FileProcessorEngine
from scrollintel.core.config import get_settings


class PerformanceOptimizationDemo:
    """Comprehensive demo of performance optimization features."""
    
    def __init__(self):
        self.settings = get_settings()
        self.db_pool = None
        self.db_optimizer = None
        self.job_processor = None
        self.file_processor = FileProcessorEngine()
    
    async def initialize(self):
        """Initialize all components."""
        print("🚀 Initializing ScrollIntel Performance Optimization Demo...")
        
        # Initialize database pool
        print("📊 Setting up optimized database connection pool...")
        self.db_pool = OptimizedDatabasePool()
        await self.db_pool.initialize()
        
        # Initialize database optimizer
        print("⚡ Initializing database optimizer...")
        self.db_optimizer = DatabaseOptimizer()
        await self.db_optimizer.initialize()
        
        # Initialize background job processor
        print("🔄 Setting up background job processor...")
        self.job_processor = BackgroundJobProcessor(max_workers=2)
        await self.job_processor.initialize()
        
        print("✅ All components initialized successfully!\n")
    
    async def demo_database_performance(self):
        """Demonstrate database performance optimizations."""
        print("=" * 60)
        print("🗄️  DATABASE PERFORMANCE OPTIMIZATION DEMO")
        print("=" * 60)
        
        # 1. Connection Pool Status
        print("\n1️⃣  Database Connection Pool Status:")
        pool_status = self.db_pool.get_pool_status()
        for key, value in pool_status.items():
            print(f"   {key}: {value}")
        
        # 2. Performance Monitoring
        print("\n2️⃣  Database Performance Monitoring:")
        
        # Execute some test queries to generate metrics
        async with self.db_pool.get_async_session() as session:
            from sqlalchemy import text
            
            # Fast query
            await session.execute(text("SELECT 1 as test"))
            
            # Slower query (simulate with pg_sleep)
            await session.execute(text("SELECT pg_sleep(0.1), 'slow_query' as type"))
            
            # Another fast query
            await session.execute(text("SELECT COUNT(*) FROM information_schema.tables"))
        
        # Get performance stats
        performance_stats = self.db_pool.get_performance_stats()
        print(f"   Total queries executed: {performance_stats['performance_report']['query_stats']['total_queries']}")
        print(f"   Average query time: {performance_stats['performance_report']['query_stats']['avg_query_time']:.3f}s")
        
        # 3. Query Caching Demo
        print("\n3️⃣  Query Caching Demonstration:")
        if self.db_pool.query_cache:
            from sqlalchemy import text
            
            # Execute cached query twice
            query = text("SELECT COUNT(*) as table_count FROM information_schema.tables")
            
            # First execution (cache miss)
            start_time = time.time()
            result1 = await self.db_pool.execute_cached_query(query, cache_key="table_count_demo")
            first_duration = time.time() - start_time
            
            # Second execution (cache hit)
            start_time = time.time()
            result2 = await self.db_pool.execute_cached_query(query, cache_key="table_count_demo")
            second_duration = time.time() - start_time
            
            print(f"   First query (cache miss): {first_duration:.3f}s")
            print(f"   Second query (cache hit): {second_duration:.3f}s")
            print(f"   Speed improvement: {(first_duration / second_duration):.1f}x faster")
            
            # Cache stats
            cache_stats = self.db_pool.query_cache.get_stats()
            print(f"   Cache size: {cache_stats['size']}/{cache_stats['max_size']}")
        else:
            print("   Query caching is disabled")
        
        # 4. Database Health Report
        print("\n4️⃣  Database Health Analysis:")
        health_report = await self.db_optimizer.get_database_health_report()
        
        print(f"   Overall Health: {health_report['overall_health'].upper()}")
        print(f"   Health Score: {health_report.get('health_score', 'N/A')}/100")
        print(f"   Database Size: {health_report['metrics']['database_size_mb']} MB")
        print(f"   Total Tables: {health_report['metrics']['total_tables']}")
        print(f"   Active Connections: {health_report['metrics']['active_connections']}")
        
        # 5. Index Recommendations
        print("\n5️⃣  Database Optimization Recommendations:")
        table_stats = await self.db_optimizer._analyze_table_statistics()
        recommendations = await self.db_optimizer._generate_index_recommendations(table_stats)
        
        if recommendations:
            print(f"   Found {len(recommendations)} optimization recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"   {i}. {rec.table_name}: {rec.reason}")
                print(f"      Priority: {rec.priority.upper()}")
                print(f"      Benefit: {rec.estimated_benefit:.1f}%")
        else:
            print("   No optimization recommendations at this time")
    
    async def demo_file_processing_performance(self):
        """Demonstrate file processing performance optimizations."""
        print("\n" + "=" * 60)
        print("📁 FILE PROCESSING PERFORMANCE OPTIMIZATION DEMO")
        print("=" * 60)
        
        # Create sample data files
        print("\n1️⃣  Creating Sample Data Files:")
        
        # Small file for immediate processing
        small_data = {
            'id': range(1, 101),
            'name': [f'User {i}' for i in range(1, 101)],
            'email': [f'user{i}@example.com' for i in range(1, 101)],
            'age': [20 + (i % 50) for i in range(1, 101)],
            'created_at': ['2024-01-01'] * 100
        }
        small_df = pd.DataFrame(small_data)
        
        # Large file for background processing
        large_data = {
            'id': range(1, 5001),
            'name': [f'User {i}' for i in range(1, 5001)],
            'email': [f'user{i}@example.com' for i in range(1, 5001)],
            'age': [20 + (i % 50) for i in range(1, 5001)],
            'salary': [30000 + (i * 100) for i in range(1, 5001)],
            'department': [f'Dept {i % 10}' for i in range(1, 5001)],
            'created_at': ['2024-01-01'] * 5000
        }
        large_df = pd.DataFrame(large_data)
        
        # Save to temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as small_file:
            small_df.to_csv(small_file.name, index=False)
            small_file_path = small_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as large_file:
            large_df.to_csv(large_file.name, index=False)
            large_file_path = large_file.name
        
        print(f"   Small file: {len(small_df)} rows, {os.path.getsize(small_file_path) / 1024:.1f} KB")
        print(f"   Large file: {len(large_df)} rows, {os.path.getsize(large_file_path) / 1024:.1f} KB")
        
        try:
            # 2. File Validation and Type Detection
            print("\n2️⃣  File Validation and Type Detection:")
            
            def create_upload_file(file_path: str, filename: str) -> UploadFile:
                with open(file_path, 'rb') as f:
                    content = f.read()
                file_obj = BytesIO(content)
                return UploadFile(
                    filename=filename,
                    file=file_obj,
                    size=len(content),
                    headers={'content-type': 'text/csv'}
                )
            
            small_upload = create_upload_file(small_file_path, "small_data.csv")
            
            # Validate file
            await self.file_processor._validate_file_optimized(small_upload)
            print("   ✅ File validation passed")
            
            # Detect file type
            detected_type = await self.file_processor._detect_file_type_optimized(
                small_upload.content_type, small_upload.filename, small_upload
            )
            print(f"   📋 Detected file type: {detected_type}")
            
            # 3. Optimized File Saving
            print("\n3️⃣  Optimized File Saving with Streaming:")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                start_time = time.time()
                file_path, file_hash = await self.file_processor._save_file_optimized(
                    small_upload, "demo-upload-123", temp_dir
                )
                save_duration = time.time() - start_time
                
                print(f"   💾 File saved in {save_duration:.3f}s")
                print(f"   🔐 File hash: {file_hash[:16]}...")
                print(f"   📍 Saved to: {Path(file_path).name}")
            
            # 4. Schema Inference with Progress Tracking
            print("\n4️⃣  Schema Inference with Progress Tracking:")
            
            # Load data
            df = await self.file_processor._load_data(small_file_path, "csv")
            
            # Mock progress tracker for demo
            class MockProgressTracker:
                def __init__(self):
                    self.progress = 0.0
                    self.message = ""
                
                async def update(self, progress: float, message: str = ""):
                    self.progress = progress
                    self.message = message
                    print(f"   📊 Progress: {progress:.1f}% - {message}")
            
            progress_tracker = MockProgressTracker()
            
            start_time = time.time()
            schema_info = await self.file_processor._infer_schema_async(df, progress_tracker)
            schema_duration = time.time() - start_time
            
            print(f"   ✅ Schema inference completed in {schema_duration:.3f}s")
            print(f"   📋 Detected {len(schema_info['columns'])} columns")
            
            # Show some column info
            for col_name, col_info in list(schema_info['columns'].items())[:3]:
                print(f"      - {col_name}: {col_info['inferred_type']} ({col_info['non_null_count']} non-null)")
            
            # 5. Quality Analysis with Progress Tracking
            print("\n5️⃣  Data Quality Analysis:")
            
            progress_tracker = MockProgressTracker()
            
            start_time = time.time()
            quality_report = await self.file_processor._generate_quality_report_async(df, progress_tracker)
            quality_duration = time.time() - start_time
            
            print(f"   ✅ Quality analysis completed in {quality_duration:.3f}s")
            print(f"   🎯 Quality Score: {quality_report['quality_score']:.1f}/100")
            print(f"   📊 Total Rows: {quality_report['total_rows']:,}")
            print(f"   🔍 Missing Values: {sum(quality_report['missing_values'].values())} cells")
            print(f"   🔄 Duplicate Rows: {quality_report['duplicate_rows']}")
            
            if quality_report['recommendations']:
                print("   💡 Recommendations:")
                for rec in quality_report['recommendations'][:2]:
                    print(f"      - {rec}")
            
        finally:
            # Cleanup
            os.unlink(small_file_path)
            os.unlink(large_file_path)
    
    async def demo_background_job_processing(self):
        """Demonstrate background job processing capabilities."""
        print("\n" + "=" * 60)
        print("🔄 BACKGROUND JOB PROCESSING DEMO")
        print("=" * 60)
        
        # 1. Register demo job functions
        print("\n1️⃣  Registering Background Job Functions:")
        
        async def demo_data_processing_job(data_size: int, processing_time: float = 2.0, progress_tracker=None):
            """Demo job that simulates data processing with progress updates."""
            
            if progress_tracker:
                await progress_tracker.update(0.0, f"Starting processing of {data_size:,} records...")
                await asyncio.sleep(0.2)
                
                await progress_tracker.update(25.0, "Loading data...")
                await asyncio.sleep(processing_time * 0.25)
                
                await progress_tracker.update(50.0, "Processing data...")
                await asyncio.sleep(processing_time * 0.25)
                
                await progress_tracker.update(75.0, "Analyzing results...")
                await asyncio.sleep(processing_time * 0.25)
                
                await progress_tracker.update(100.0, "Processing completed!")
                await asyncio.sleep(processing_time * 0.25)
            else:
                await asyncio.sleep(processing_time)
            
            return {
                "processed_records": data_size,
                "processing_time": processing_time,
                "status": "completed",
                "results": {
                    "average_value": data_size / 2,
                    "total_sum": data_size * (data_size + 1) / 2
                }
            }
        
        self.job_processor.register_job_function("demo_data_processing_job", demo_data_processing_job)
        print("   ✅ Registered 'demo_data_processing_job'")
        
        # 2. Submit jobs with different priorities
        print("\n2️⃣  Submitting Background Jobs:")
        
        # High priority job
        high_priority_job = await self.job_processor.submit_job(
            job_type="data_processing",
            function_name="demo_data_processing_job",
            args=[1000],
            kwargs={"processing_time": 1.0},
            priority=JobPriority.HIGH,
            user_id="demo-user",
            metadata={"description": "High priority data processing"}
        )
        print(f"   🔥 High priority job submitted: {high_priority_job}")
        
        # Normal priority job
        normal_priority_job = await self.job_processor.submit_job(
            job_type="data_processing",
            function_name="demo_data_processing_job",
            args=[5000],
            kwargs={"processing_time": 2.0},
            priority=JobPriority.NORMAL,
            user_id="demo-user",
            metadata={"description": "Normal priority data processing"}
        )
        print(f"   📋 Normal priority job submitted: {normal_priority_job}")
        
        # 3. Start workers and monitor progress
        print("\n3️⃣  Starting Workers and Monitoring Progress:")
        
        await self.job_processor.start_workers()
        print("   🚀 Background workers started")
        
        # Monitor job progress
        jobs_to_monitor = [high_priority_job, normal_priority_job]
        completed_jobs = set()
        
        print("\n   📊 Job Progress Monitoring:")
        
        while len(completed_jobs) < len(jobs_to_monitor):
            for job_id in jobs_to_monitor:
                if job_id in completed_jobs:
                    continue
                
                # Check job result
                result = await self.job_processor.get_job_result(job_id)
                if result and result.status.value in ['completed', 'failed']:
                    completed_jobs.add(job_id)
                    
                    if result.status.value == 'completed':
                        print(f"   ✅ Job {job_id[:8]}... completed in {result.execution_time:.2f}s")
                        print(f"      Result: Processed {result.result['processed_records']:,} records")
                    else:
                        print(f"   ❌ Job {job_id[:8]}... failed: {result.error}")
                else:
                    # Show progress
                    progress = await self.job_processor.get_job_progress(job_id)
                    if progress.get('progress', 0) > 0:
                        print(f"   🔄 Job {job_id[:8]}... {progress['progress']:.1f}% - {progress.get('message', '')}")
            
            if len(completed_jobs) < len(jobs_to_monitor):
                await asyncio.sleep(0.5)
        
        # 4. Job processor statistics
        print("\n4️⃣  Background Job Processor Statistics:")
        
        stats = await self.job_processor.get_stats()
        print(f"   Workers: {stats['workers']}/{stats['max_workers']}")
        print(f"   Registered Functions: {len(stats['registered_functions'])}")
        
        if 'queue_stats' in stats:
            queue_stats = stats['queue_stats']
            print(f"   Queue Stats:")
            print(f"      - Pending: {queue_stats.get('pending', 0)}")
            print(f"      - Processing: {queue_stats.get('processing', 0)}")
            print(f"      - Completed: {queue_stats.get('completed', 0)}")
            print(f"      - Failed: {queue_stats.get('failed', 0)}")
        
        await self.job_processor.stop_workers()
        print("   🛑 Background workers stopped")
    
    async def demo_performance_monitoring(self):
        """Demonstrate performance monitoring capabilities."""
        print("\n" + "=" * 60)
        print("📈 PERFORMANCE MONITORING DEMO")
        print("=" * 60)
        
        # 1. Database Performance Metrics
        print("\n1️⃣  Database Performance Metrics:")
        
        # Execute various queries to generate metrics
        async with self.db_pool.get_async_session() as session:
            from sqlalchemy import text
            
            # Fast queries
            for i in range(5):
                await session.execute(text("SELECT 1"))
            
            # Medium queries
            for i in range(3):
                await session.execute(text("SELECT COUNT(*) FROM information_schema.columns"))
            
            # Slow query
            await session.execute(text("SELECT pg_sleep(0.2)"))
        
        # Get performance statistics
        performance_stats = self.db_pool.get_performance_stats()
        
        print(f"   📊 Query Statistics:")
        query_stats = performance_stats['performance_report']['query_stats']
        print(f"      - Total Queries: {query_stats['total_queries']}")
        print(f"      - Average Query Time: {query_stats['avg_query_time']:.3f}s")
        print(f"      - Slow Queries: {query_stats['slow_queries']}")
        
        # Show slow queries
        slow_queries = self.db_pool.monitor.get_slow_queries(limit=3)
        if slow_queries:
            print(f"   🐌 Slowest Queries:")
            for i, query in enumerate(slow_queries, 1):
                print(f"      {i}. {query['avg_execution_time']:.3f}s avg - {query['query_text'][:50]}...")
        
        # 2. Connection Pool Metrics
        print(f"\n   🔗 Connection Pool Status:")
        pool_status = self.db_pool.get_pool_status()
        print(f"      - Pool Size: {pool_status['pool_size']}")
        print(f"      - Active Connections: {pool_status['checked_out']}")
        print(f"      - Available Connections: {pool_status['checked_in']}")
        print(f"      - Total Connections: {pool_status['total_connections']}")
        
        # 3. Cache Performance
        if self.db_pool.query_cache:
            print(f"\n   💾 Cache Performance:")
            cache_stats = self.db_pool.query_cache.get_stats()
            print(f"      - Cache Size: {cache_stats['size']}/{cache_stats['max_size']}")
            print(f"      - TTL: {cache_stats['ttl_seconds']}s")
        
        print(f"\n   📈 Performance Report Generated Successfully!")
    
    async def cleanup(self):
        """Cleanup all resources."""
        print("\n🧹 Cleaning up resources...")
        
        if self.job_processor:
            await self.job_processor.cleanup()
        
        if self.db_pool:
            await self.db_pool.close()
        
        print("✅ Cleanup completed!")
    
    async def run_complete_demo(self):
        """Run the complete performance optimization demo."""
        try:
            await self.initialize()
            
            await self.demo_database_performance()
            await self.demo_file_processing_performance()
            await self.demo_background_job_processing()
            await self.demo_performance_monitoring()
            
            print("\n" + "=" * 60)
            print("🎉 PERFORMANCE OPTIMIZATION DEMO COMPLETED!")
            print("=" * 60)
            print("\n✨ Key Features Demonstrated:")
            print("   • Database connection pooling with monitoring")
            print("   • Query caching and performance tracking")
            print("   • Database health analysis and optimization")
            print("   • Optimized file processing with progress tracking")
            print("   • Background job processing with priority queues")
            print("   • Comprehensive performance monitoring")
            print("\n🚀 ScrollIntel is ready for high-performance production workloads!")
            
        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.cleanup()


async def main():
    """Run the performance optimization demo."""
    demo = PerformanceOptimizationDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())