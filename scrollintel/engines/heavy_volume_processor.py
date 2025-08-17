"""
Heavy Volume File Processor Engine for ScrollIntel
Implements chunked processing, memory management, and performance optimizations
for handling large datasets (1GB+ files, millions of rows).
"""

import os
import asyncio
import aiofiles
import pandas as pd
import numpy as np
import psutil
import gc
from typing import Dict, Any, List, Optional, AsyncGenerator, Tuple
from pathlib import Path
import tempfile
import logging
from datetime import datetime

from .base_engine import BaseEngine, EngineStatus
from ..core.interfaces import EngineError

logger = logging.getLogger(__name__)


class HeavyVolumeProcessor(BaseEngine):
    """Enhanced file processor for heavy volume datasets."""
    
    def __init__(self):
        from .base_engine import EngineCapability
        super().__init__(
            engine_id="heavy_volume_processor",
            name="ScrollIntel Heavy Volume File Processor",
            capabilities=[EngineCapability.DATA_ANALYSIS]
        )
        
        # Heavy volume settings - Updated limits
        self.max_file_size = 10 * 1024 * 1024 * 1024  # 10GB (was 100MB)
        self.max_memory_usage = 8 * 1024 * 1024 * 1024  # 8GB (was 512MB)
        self.chunk_size = 100000  # 100K rows per chunk (was 8KB)
        self.streaming_threshold = 100 * 1024 * 1024  # 100MB
        
        # Performance settings
        self.enable_parallel_processing = True
        self.max_workers = min(32, (os.cpu_count() or 1) + 4)
        self.memory_monitor_interval = 5.0  # seconds
        
        # Compression and optimization
        self.enable_compression = True
        self.compression_level = 6
        self.use_pyarrow = True  # Use PyArrow for better performance
        
        # Progress tracking
        self.progress_callbacks = {}
    
    async def process_large_file_chunked(
        self, 
        file_path: str, 
        chunk_size: int = None,
        progress_callback: callable = None
    ) -> AsyncGenerator[Tuple[pd.DataFrame, Dict[str, Any]], None]:
        """Process large files in chunks to manage memory usage."""
        
        chunk_size = chunk_size or self.chunk_size
        file_ext = Path(file_path).suffix.lower()
        
        # Get file size for progress tracking
        file_size = os.path.getsize(file_path)
        processed_bytes = 0
        
        logger.info(f"Starting chunked processing of {file_path} ({file_size / 1024**2:.1f}MB)")
        
        try:
            if file_ext == '.csv':
                async for chunk, metadata in self._process_csv_chunked(
                    file_path, chunk_size, progress_callback
                ):
                    yield chunk, metadata
            elif file_ext in ['.xlsx', '.xls']:
                async for chunk, metadata in self._process_excel_chunked(
                    file_path, chunk_size, progress_callback
                ):
                    yield chunk, metadata
            elif file_ext == '.json':
                async for chunk, metadata in self._process_json_chunked(
                    file_path, chunk_size, progress_callback
                ):
                    yield chunk, metadata
            else:
                raise EngineError(f"Unsupported file format for chunked processing: {file_ext}")
                
        except Exception as e:
            logger.error(f"Chunked processing failed for {file_path}: {str(e)}")
            raise EngineError(f"Chunked processing failed: {str(e)}")
    
    async def _process_csv_chunked(
        self, 
        file_path: str, 
        chunk_size: int,
        progress_callback: callable = None
    ) -> AsyncGenerator[Tuple[pd.DataFrame, Dict[str, Any]], None]:
        """Process CSV files in chunks with memory management."""
        
        try:
            # Configure pandas for optimal performance
            read_kwargs = {
                'chunksize': chunk_size,
                'low_memory': False,
                'engine': 'c',  # Use C engine for speed
            }
            
            # Use PyArrow backend if available
            if self.use_pyarrow:
                try:
                    read_kwargs['dtype_backend'] = 'pyarrow'
                except Exception:
                    logger.warning("PyArrow not available, using default backend")
            
            chunk_reader = pd.read_csv(file_path, **read_kwargs)
            
            chunk_num = 0
            total_rows_processed = 0
            
            for chunk in chunk_reader:
                chunk_num += 1
                
                # Memory monitoring before processing
                await self._monitor_memory_usage()
                
                # Optimize chunk data types for memory efficiency
                chunk = await self._optimize_chunk_dtypes(chunk)
                
                # Create chunk metadata
                metadata = {
                    'chunk_number': chunk_num,
                    'chunk_rows': len(chunk),
                    'chunk_columns': len(chunk.columns),
                    'total_rows_processed': total_rows_processed + len(chunk),
                    'memory_usage_mb': chunk.memory_usage(deep=True).sum() / 1024**2,
                    'processing_time': datetime.utcnow().isoformat()
                }
                
                total_rows_processed += len(chunk)
                
                # Progress callback
                if progress_callback:
                    await progress_callback(chunk_num, len(chunk), total_rows_processed)
                
                logger.debug(f"Processed chunk {chunk_num}: {len(chunk)} rows, "
                           f"{chunk.memory_usage(deep=True).sum() / 1024**2:.1f}MB")
                
                yield chunk, metadata
                
                # Force garbage collection after each chunk
                gc.collect()
                
        except Exception as e:
            logger.error(f"CSV chunked processing failed: {str(e)}")
            raise EngineError(f"CSV chunked processing failed: {str(e)}")
    
    async def _monitor_memory_usage(self) -> None:
        """Monitor and manage memory usage during processing."""
        
        memory_info = psutil.virtual_memory()
        memory_usage_gb = memory_info.used / (1024**3)
        memory_limit_gb = self.max_memory_usage / (1024**3)
        
        if memory_usage_gb > memory_limit_gb * 0.9:  # 90% threshold
            logger.warning(f"High memory usage: {memory_usage_gb:.1f}GB / {memory_limit_gb:.1f}GB")
            
            # Force garbage collection
            gc.collect()
            
            # Check again after GC
            memory_info = psutil.virtual_memory()
            memory_usage_gb = memory_info.used / (1024**3)
            
            if memory_usage_gb > memory_limit_gb:
                raise EngineError(
                    f"Memory usage {memory_usage_gb:.1f}GB exceeds "
                    f"limit {memory_limit_gb:.1f}GB"
                )
    
    async def _optimize_chunk_dtypes(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency."""
        
        for col in chunk.columns:
            if chunk[col].dtype == 'object':
                try:
                    # Try numeric conversion first
                    numeric = pd.to_numeric(chunk[col], errors='ignore')
                    if not numeric.equals(chunk[col]):
                        chunk[col] = numeric
                        continue
                    
                    # Try datetime conversion for date-like strings
                    if chunk[col].astype(str).str.match(r'\d{4}-\d{2}-\d{2}').any():
                        datetime_col = pd.to_datetime(chunk[col], errors='ignore')
                        if not datetime_col.equals(chunk[col]):
                            chunk[col] = datetime_col
                            continue
                    
                    # Use category for low cardinality strings (< 50% unique)
                    if chunk[col].nunique() / len(chunk) < 0.5:
                        chunk[col] = chunk[col].astype('category')
                        
                except Exception as e:
                    logger.debug(f"Could not optimize column {col}: {str(e)}")
                    pass  # Keep original type if conversion fails
        
        return chunk 
    
    async def process_file_streaming(
        self, 
        file_path: str,
        output_callback: callable = None
    ) -> Dict[str, Any]:
        """Process file using streaming approach for maximum efficiency."""
        
        start_time = datetime.utcnow()
        total_rows = 0
        total_chunks = 0
        processing_stats = {
            'start_time': start_time.isoformat(),
            'file_path': file_path,
            'file_size_mb': os.path.getsize(file_path) / 1024**2,
            'chunks_processed': 0,
            'total_rows': 0,
            'memory_peak_mb': 0,
            'processing_errors': []
        }
        
        try:
            async for chunk, metadata in self.process_large_file_chunked(file_path):
                total_chunks += 1
                total_rows += len(chunk)
                
                # Track peak memory usage
                current_memory_mb = psutil.virtual_memory().used / 1024**2
                processing_stats['memory_peak_mb'] = max(
                    processing_stats['memory_peak_mb'], 
                    current_memory_mb
                )
                
                # Process chunk (custom logic can be added here)
                if output_callback:
                    await output_callback(chunk, metadata)
                
                # Update stats
                processing_stats['chunks_processed'] = total_chunks
                processing_stats['total_rows'] = total_rows
                
                logger.info(f"Processed chunk {total_chunks}: {len(chunk)} rows "
                          f"(Total: {total_rows} rows)")
            
            # Final statistics
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            processing_stats.update({
                'end_time': end_time.isoformat(),
                'processing_time_seconds': processing_time,
                'rows_per_second': total_rows / processing_time if processing_time > 0 else 0,
                'status': 'completed'
            })
            
            logger.info(f"Streaming processing completed: {total_rows} rows in "
                       f"{processing_time:.1f}s ({total_rows/processing_time:.0f} rows/sec)")
            
            return processing_stats
            
        except Exception as e:
            processing_stats['status'] = 'failed'
            processing_stats['error'] = str(e)
            logger.error(f"Streaming processing failed: {str(e)}")
            raise EngineError(f"Streaming processing failed: {str(e)}")
    
    async def _process_excel_chunked(
        self, 
        file_path: str, 
        chunk_size: int,
        progress_callback: callable = None
    ) -> AsyncGenerator[Tuple[pd.DataFrame, Dict[str, Any]], None]:
        """Process Excel files in chunks (limited chunking support)."""
        
        try:
            # Excel files need to be loaded entirely first, then chunked
            logger.info(f"Loading Excel file: {file_path}")
            
            # Check file size before loading
            file_size_mb = os.path.getsize(file_path) / 1024**2
            if file_size_mb > 1000:  # 1GB limit for Excel
                raise EngineError(f"Excel file too large: {file_size_mb:.1f}MB (limit: 1GB)")
            
            # Load Excel file
            df = pd.read_excel(file_path, engine='openpyxl')
            
            # Process in chunks
            total_rows = len(df)
            chunk_num = 0
            
            for start_idx in range(0, total_rows, chunk_size):
                end_idx = min(start_idx + chunk_size, total_rows)
                chunk = df.iloc[start_idx:end_idx].copy()
                
                chunk_num += 1
                
                # Optimize chunk
                chunk = await self._optimize_chunk_dtypes(chunk)
                
                metadata = {
                    'chunk_number': chunk_num,
                    'chunk_rows': len(chunk),
                    'chunk_columns': len(chunk.columns),
                    'start_row': start_idx,
                    'end_row': end_idx,
                    'total_rows': total_rows,
                    'progress_percent': (end_idx / total_rows) * 100
                }
                
                if progress_callback:
                    await progress_callback(chunk_num, len(chunk), end_idx)
                
                yield chunk, metadata
                
                # Memory management
                await self._monitor_memory_usage()
                gc.collect()
            
            # Clean up the full DataFrame
            del df
            gc.collect()
            
        except Exception as e:
            logger.error(f"Excel chunked processing failed: {str(e)}")
            raise EngineError(f"Excel chunked processing failed: {str(e)}")
    
    async def _process_json_chunked(
        self, 
        file_path: str, 
        chunk_size: int,
        progress_callback: callable = None
    ) -> AsyncGenerator[Tuple[pd.DataFrame, Dict[str, Any]], None]:
        """Process JSON files in chunks."""
        
        try:
            # For JSON, we'll read line by line if it's JSONL format
            # or load and chunk if it's a single JSON array
            
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
            
            if first_line.startswith('['):
                # JSON array format - load and chunk
                async for chunk, metadata in self._process_json_array_chunked(
                    file_path, chunk_size, progress_callback
                ):
                    yield chunk, metadata
            else:
                # JSONL format - process line by line
                async for chunk, metadata in self._process_jsonl_chunked(
                    file_path, chunk_size, progress_callback
                ):
                    yield chunk, metadata
                    
        except Exception as e:
            logger.error(f"JSON chunked processing failed: {str(e)}")
            raise EngineError(f"JSON chunked processing failed: {str(e)}")
    
    async def _process_jsonl_chunked(
        self, 
        file_path: str, 
        chunk_size: int,
        progress_callback: callable = None
    ) -> AsyncGenerator[Tuple[pd.DataFrame, Dict[str, Any]], None]:
        """Process JSONL (JSON Lines) files in chunks."""
        
        import json
        
        chunk_data = []
        chunk_num = 0
        total_lines = 0
        
        async with aiofiles.open(file_path, 'r') as f:
            async for line in f:
                try:
                    json_obj = json.loads(line.strip())
                    chunk_data.append(json_obj)
                    total_lines += 1
                    
                    if len(chunk_data) >= chunk_size:
                        # Create DataFrame from chunk
                        chunk_df = pd.DataFrame(chunk_data)
                        chunk_df = await self._optimize_chunk_dtypes(chunk_df)
                        
                        chunk_num += 1
                        metadata = {
                            'chunk_number': chunk_num,
                            'chunk_rows': len(chunk_df),
                            'chunk_columns': len(chunk_df.columns),
                            'total_lines_processed': total_lines
                        }
                        
                        if progress_callback:
                            await progress_callback(chunk_num, len(chunk_df), total_lines)
                        
                        yield chunk_df, metadata
                        
                        # Reset for next chunk
                        chunk_data = []
                        await self._monitor_memory_usage()
                        gc.collect()
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line {total_lines}: {str(e)}")
                    continue
        
        # Process remaining data
        if chunk_data:
            chunk_df = pd.DataFrame(chunk_data)
            chunk_df = await self._optimize_chunk_dtypes(chunk_df)
            
            chunk_num += 1
            metadata = {
                'chunk_number': chunk_num,
                'chunk_rows': len(chunk_df),
                'chunk_columns': len(chunk_df.columns),
                'total_lines_processed': total_lines,
                'final_chunk': True
            }
            
            yield chunk_df, metadata
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            'memory_usage_gb': memory_info.used / (1024**3),
            'memory_available_gb': memory_info.available / (1024**3),
            'memory_percent': memory_info.percent,
            'cpu_percent': cpu_percent,
            'max_file_size_gb': self.max_file_size / (1024**3),
            'max_memory_usage_gb': self.max_memory_usage / (1024**3),
            'chunk_size': self.chunk_size,
            'parallel_processing': self.enable_parallel_processing,
            'max_workers': self.max_workers
        } 
   # Abstract method implementations
    async def initialize(self) -> None:
        """Initialize the heavy volume processor."""
        self.status = EngineStatus.READY
        logger.info(f"Heavy Volume Processor initialized with max file size: {self.max_file_size / (1024**3):.1f}GB")
    
    async def process(self, input_data: Any, parameters: Dict[str, Any] = None) -> Any:
        """Process heavy volume data."""
        if parameters is None:
            parameters = {}
        
        file_path = parameters.get('file_path')
        if not file_path:
            raise EngineError("file_path parameter is required")
        
        # Use streaming processing for large files
        if os.path.getsize(file_path) > self.streaming_threshold:
            return await self.process_file_streaming(file_path)
        else:
            return await self.process_file_chunked(file_path)
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        gc.collect()
        logger.info("Heavy Volume Processor cleaned up")
    
    def get_status(self) -> Dict[str, Any]:
        """Get processor status."""
        return {
            "engine_id": self.engine_id,
            "name": self.name,
            "status": self.status.value,
            "max_file_size_gb": self.max_file_size / (1024**3),
            "max_memory_gb": self.max_memory_usage / (1024**3),
            "chunk_size": self.chunk_size,
            "usage_count": self.usage_count,
            "error_count": self.error_count
        }