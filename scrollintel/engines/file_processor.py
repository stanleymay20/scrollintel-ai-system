"""
Optimized File Processing Engine for ScrollIntel.
Handles file upload, auto-detection, schema inference, and data quality validation
with performance optimizations, progress tracking, and background processing.
"""

import os
import uuid
import mimetypes
import pandas as pd
import json
import sqlite3
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import asyncio
import aiofiles
from fastapi import UploadFile
import numpy as np
from sqlalchemy.orm import Session
import time
import hashlib
import tempfile
import shutil

from .base_engine import BaseEngine
from ..core.interfaces import EngineError
from ..core.background_jobs import get_job_processor, JobPriority, background_job
from ..core.database_pool import get_optimized_db_pool


class ProgressTracker:
    """Simple progress tracker for file processing operations."""
    
    def __init__(self, operation_id: str):
        self.operation_id = operation_id
        self.current_progress = 0.0
        self.current_message = ""
        self.start_time = time.time()
    
    async def update_progress(self, progress: float, message: str = ""):
        """Update progress and message."""
        self.current_progress = progress
        self.current_message = message
        
        # Log progress for debugging
        elapsed = time.time() - self.start_time
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Progress [{self.operation_id}]: {progress:.1%} - {message} (elapsed: {elapsed:.1f}s)")
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information."""
        return {
            "operation_id": self.operation_id,
            "progress": self.current_progress,
            "message": self.current_message,
            "elapsed_time": time.time() - self.start_time
        }
from ..models.database import FileUpload, Dataset
from ..models.schemas import (
    FileUploadResponse, DataPreviewResponse, DataQualityReport,
    FileProcessingStatus
)


class FileProcessorEngine(BaseEngine):
    """Enhanced file processor engine for ScrollIntel Core Focus.
    
    Handles CSV, Excel, JSON uploads with automatic schema detection,
    data validation, preprocessing pipeline, security validation,
    and progress tracking.
    """
    
    def __init__(self):
        from .base_engine import EngineCapability
        super().__init__(
            engine_id="file_processor",
            name="ScrollIntel Core File Processor",
            capabilities=[EngineCapability.DATA_ANALYSIS]
        )
        
        # Supported file types for Core Focus
        self.supported_types = {
            'text/csv': 'csv',
            'application/vnd.ms-excel': 'xlsx',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
            'application/json': 'json',
            'text/plain': 'csv',  # Default text files to CSV
        }
        
        # Core Focus limits and settings
        self.max_preview_rows = 100
        self.max_file_size = 100 * 1024 * 1024  # 100MB limit
        self.chunk_size = 8192  # 8KB chunks for streaming
        self.temp_dir = tempfile.gettempdir()
        
        # Security settings
        self.allowed_extensions = ['.csv', '.xlsx', '.xls', '.json']
        self.dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\0', '..']
        self.max_filename_length = 255
        
        # Performance settings
        self.enable_parallel_processing = True
        self.max_memory_usage = 512 * 1024 * 1024  # 512MB
        self.enable_compression = True
        
        # Progress tracking
        self.progress_callbacks = {}
        
        # Data preprocessing settings
        self.enable_data_cleaning = True
        self.enable_normalization = True
        self.outlier_detection_threshold = 3.0  # Standard deviations
    
    async def process_upload(
        self, 
        file: UploadFile, 
        user_id: str,
        storage_path: str,
        db_session: Session,
        auto_detect_schema: bool = True,
        generate_preview: bool = True,
        use_background_processing: bool = True
    ) -> FileUploadResponse:
        """Process uploaded file with optimized auto-detection and validation."""
        
        # Generate unique upload ID
        upload_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Validate file with enhanced checks
            await self._validate_file_optimized(file)
            
            # Detect file type with content analysis
            detected_type = await self._detect_file_type_optimized(file.content_type, file.filename, file)
            
            # Save file with streaming and compression
            file_path, file_hash = await self._save_file_optimized(file, upload_id, storage_path)
            
            # Create file upload record with enhanced metadata
            file_upload = FileUpload(
                upload_id=upload_id,
                user_id=user_id,
                filename=f"{upload_id}_{file.filename}",
                original_filename=file.filename,
                file_path=file_path,
                file_size=file.size,
                content_type=file.content_type,
                detected_type=detected_type,
                processing_status="pending"
            )
            
            db_session.add(file_upload)
            db_session.commit()
            
            # Submit background job for processing if enabled
            if use_background_processing and file.size > 1024 * 1024:  # > 1MB
                job_processor = await get_job_processor()
                await job_processor.submit_job(
                    job_type="file_processing",
                    function_name="process_file_background",
                    args=[upload_id, file_path, detected_type],
                    kwargs={
                        'auto_detect_schema': auto_detect_schema,
                        'generate_preview': generate_preview,
                        'user_id': user_id,
                        'file_hash': file_hash
                    },
                    priority=JobPriority.HIGH,
                    user_id=user_id,
                    metadata={
                        'filename': file.filename,
                        'file_size': file.size,
                        'detected_type': detected_type
                    }
                )
                
                file_upload.processing_status = "queued"
                db_session.commit()
            else:
                # Process immediately for small files
                asyncio.create_task(
                    self._process_file_immediate(
                        upload_id, file_path, detected_type, 
                        auto_detect_schema, generate_preview, user_id
                    )
                )
                
                file_upload.processing_status = "processing"
                db_session.commit()
            
            processing_time = time.time() - start_time
            
            return FileUploadResponse(
                upload_id=upload_id,
                filename=file_upload.filename,
                original_filename=file_upload.original_filename,
                file_path=file_path,
                file_size=file.size,
                content_type=file.content_type,
                detected_type=detected_type,
                schema_info={},
                preview_data=None,
                quality_report=None,
                dataset_id=None,
                processing_time=processing_time
            )
            
        except Exception as e:
            # Update status to failed if record exists
            try:
                file_upload = db_session.query(FileUpload).filter_by(upload_id=upload_id).first()
                if file_upload:
                    file_upload.processing_status = "failed"
                    file_upload.error_details = {"error": str(e)}
                    db_session.commit()
            except:
                pass
            
            raise EngineError(f"File processing failed: {str(e)}")
    
    async def _validate_file_optimized(self, file: UploadFile) -> None:
        """Enhanced file validation with comprehensive security checks."""
        
        # Basic file checks
        if not file.filename:
            raise EngineError("Filename is required")
        
        if file.size is None:
            raise EngineError("File size could not be determined")
        
        # File size validation
        if file.size > self.max_file_size:
            size_mb = file.size / (1024 * 1024)
            max_mb = self.max_file_size / (1024 * 1024)
            raise EngineError(f"File size {size_mb:.1f}MB exceeds maximum allowed size {max_mb:.1f}MB")
        
        if file.size < 10:  # Less than 10 bytes
            raise EngineError("File is too small to be valid")
        
        # Filename security validation
        if len(file.filename) > self.max_filename_length:
            raise EngineError(f"Filename is too long (max {self.max_filename_length} characters)")
        
        # Check for dangerous characters and patterns
        if any(char in file.filename for char in self.dangerous_chars):
            raise EngineError("Filename contains invalid or dangerous characters")
        
        # Path traversal protection
        if '..' in file.filename or file.filename.startswith('/') or '\\' in file.filename:
            raise EngineError("Filename contains invalid path characters")
        
        # File extension validation
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in self.allowed_extensions:
            raise EngineError(f"Unsupported file extension: {file_ext}. Allowed: {', '.join(self.allowed_extensions)}")
        
        # Content type validation
        if file.content_type not in self.supported_types:
            # Try to infer from extension as fallback
            ext_to_content_type = {
                '.csv': 'text/csv',
                '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                '.xls': 'application/vnd.ms-excel',
                '.json': 'application/json'
            }
            
            expected_content_type = ext_to_content_type.get(file_ext)
            if not expected_content_type:
                raise EngineError(f"Unsupported file type: {file.content_type}")
        
        # File signature validation for security
        await self._validate_file_signature(file, file_ext)
        
        # Additional security checks
        await self._perform_security_scan(file)
    
    async def _validate_file_signature(self, file: UploadFile, file_ext: str) -> None:
        """Validate file signature (magic bytes) for security."""
        # Read first 512 bytes for signature validation
        current_pos = file.file.tell()
        file.file.seek(0)
        header = file.file.read(512)
        file.file.seek(current_pos)
        
        # File signature validation for Core Focus supported formats
        signatures = {
            '.xlsx': [b'PK\x03\x04'],  # ZIP-based format (Office Open XML)
            '.xls': [b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'],  # OLE2 format (Legacy Excel)
            '.csv': [],  # No specific signature for CSV - validated by content
            '.json': []  # No specific signature for JSON - validated by content
        }
        
        if file_ext in signatures and signatures[file_ext]:
            valid_signature = any(header.startswith(sig) for sig in signatures[file_ext])
            if not valid_signature:
                raise EngineError(f"File signature does not match expected format for {file_ext}")
        
        # Additional validation for text-based formats
        if file_ext in ['.csv', '.json']:
            try:
                # Try to decode as UTF-8 to ensure it's valid text
                header_text = header.decode('utf-8', errors='strict')
                
                # Basic content validation
                if file_ext == '.json':
                    # JSON should start with { or [
                    stripped = header_text.strip()
                    if stripped and not stripped.startswith(('{', '[')):
                        # Try to parse a small sample
                        try:
                            json.loads(stripped[:100])
                        except json.JSONDecodeError:
                            raise EngineError("File does not appear to contain valid JSON")
                
            except UnicodeDecodeError:
                raise EngineError(f"File does not appear to be valid text for {file_ext} format")
    
    async def _perform_security_scan(self, file: UploadFile) -> None:
        """Perform additional security scans on the uploaded file."""
        current_pos = file.file.tell()
        file.file.seek(0)
        
        # Read first 1KB for security scanning
        sample = file.file.read(1024)
        file.file.seek(current_pos)
        
        # Check for potentially malicious content
        malicious_patterns = [
            b'<script',  # JavaScript
            b'javascript:',  # JavaScript URLs
            b'vbscript:',  # VBScript
            b'data:text/html',  # Data URLs with HTML
            b'<?php',  # PHP code
            b'<%',  # ASP/JSP code
            b'eval(',  # Code evaluation
            b'exec(',  # Code execution
        ]
        
        sample_lower = sample.lower()
        for pattern in malicious_patterns:
            if pattern in sample_lower:
                raise EngineError("File contains potentially malicious content")
        
        # Check for excessive null bytes (potential binary injection)
        null_count = sample.count(b'\x00')
        if null_count > len(sample) * 0.1:  # More than 10% null bytes
            raise EngineError("File contains suspicious binary content")
    
    async def _detect_file_type_optimized(self, content_type: str, filename: str, file: UploadFile) -> str:
        """Detect file type with enhanced content analysis."""
        
        # First try content type
        if content_type in self.supported_types:
            detected_type = self.supported_types[content_type]
        else:
            # Fall back to file extension
            file_ext = Path(filename).suffix.lower()
            ext_mapping = {
                '.csv': 'csv',
                '.xlsx': 'xlsx',
                '.xls': 'xlsx',
                '.json': 'json',
                '.sql': 'sql'
            }
            detected_type = ext_mapping.get(file_ext, 'csv')  # Default to CSV
        
        # Additional content-based detection for ambiguous cases
        if detected_type == 'csv' or content_type == 'text/plain':
            detected_type = await self._analyze_text_content(file)
        
        return detected_type
    
    async def _analyze_text_content(self, file: UploadFile) -> str:
        """Analyze text content to determine if it's CSV, JSON, or SQL."""
        current_pos = file.file.tell()
        file.file.seek(0)
        
        # Read first 1KB for analysis
        sample = file.file.read(1024).decode('utf-8', errors='ignore')
        file.file.seek(current_pos)
        
        sample_lower = sample.lower().strip()
        
        # Check for SQL keywords
        sql_keywords = ['select', 'insert', 'update', 'delete', 'create', 'drop', 'alter']
        if any(keyword in sample_lower for keyword in sql_keywords):
            return 'sql'
        
        # Check for JSON structure
        if sample.strip().startswith(('{', '[')):
            try:
                json.loads(sample[:100])  # Try to parse first part
                return 'json'
            except:
                pass
        
        # Default to CSV for text files
        return 'csv'
    
    async def _save_file_optimized(self, file: UploadFile, upload_id: str, storage_path: str) -> Tuple[str, str]:
        """Save uploaded file with streaming and compression."""
        
        # Create storage directory if it doesn't exist
        storage_dir = Path(storage_path) / "uploads"
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        file_ext = Path(file.filename).suffix
        filename = f"{upload_id}_{file.filename}"
        file_path = storage_dir / filename
        
        # Calculate file hash while saving
        hasher = hashlib.sha256()
        total_size = 0
        
        # Save file with streaming to handle large files
        async with aiofiles.open(file_path, 'wb') as f:
            while True:
                chunk = await file.read(self.chunk_size)
                if not chunk:
                    break
                
                hasher.update(chunk)
                total_size += len(chunk)
                await f.write(chunk)
                
                # Memory usage check
                if total_size > self.max_memory_usage:
                    raise EngineError("File too large for processing")
        
        file_hash = hasher.hexdigest()
        
        # Verify file was saved correctly
        if not file_path.exists() or file_path.stat().st_size != file.size:
            raise EngineError("File save verification failed")
        
        return str(file_path), file_hash
    
    async def _process_file_immediate(
        self,
        upload_id: str,
        file_path: str,
        detected_type: str,
        auto_detect_schema: bool,
        generate_preview: bool,
        user_id: str
    ) -> None:
        """Process file immediately with enhanced progress tracking."""
        
        progress_tracker = ProgressTracker(upload_id)
        
        try:
            # Get database session
            db_pool = await get_optimized_db_pool()
            async with db_pool.get_async_session() as db_session:
                # Initialize progress tracking
                await progress_tracker.update_progress(0.0, "Initializing file processing...")
                await self._update_file_status(db_session, upload_id, "processing", 0.0, "Initializing...")
                
                # Step 1: Load and validate data (30% of progress)
                await progress_tracker.update_progress(0.1, "Loading file data...")
                df = await self._load_data(file_path, detected_type)
                
                if df.empty:
                    await progress_tracker.update_progress(1.0, "File is empty - processing completed")
                    await self._update_file_status(db_session, upload_id, "completed", 1.0, "File is empty")
                    return
                
                await progress_tracker.update_progress(0.2, f"Loaded {len(df)} rows with {len(df.columns)} columns")
                
                # Step 2: Data preprocessing (if enabled)
                if self.enable_data_cleaning:
                    await progress_tracker.update_progress(0.25, "Preprocessing data...")
                    df = await self.preprocess_data(df, {'optimize_dtypes': True, 'normalize_text': True})
                
                await progress_tracker.update_progress(0.3, "Data loading completed")
                await self._update_file_status(db_session, upload_id, "processing", 0.3, "Data loaded successfully")
                
                # Step 3: Schema inference (20% of progress)
                schema_info = {}
                if auto_detect_schema:
                    await progress_tracker.update_progress(0.35, "Analyzing data schema...")
                    schema_info = await self._infer_schema_with_progress(df, progress_tracker, 0.35, 0.5)
                    await progress_tracker.update_progress(0.5, "Schema analysis completed")
                
                await self._update_file_status(db_session, upload_id, "processing", 0.5, "Schema analysis completed")
                
                # Step 4: Generate preview (15% of progress)
                preview_data = None
                if generate_preview:
                    await progress_tracker.update_progress(0.55, "Generating data preview...")
                    preview_data = await self._generate_preview_with_progress(df, progress_tracker, 0.55, 0.65)
                    await progress_tracker.update_progress(0.65, "Preview generation completed")
                
                await self._update_file_status(db_session, upload_id, "processing", 0.65, "Preview generated")
                
                # Step 5: Quality analysis (25% of progress)
                await progress_tracker.update_progress(0.7, "Running data quality analysis...")
                quality_report = await self._generate_quality_report_with_progress(df, progress_tracker, 0.7, 0.9)
                await progress_tracker.update_progress(0.9, "Quality analysis completed")
                
                await self._update_file_status(db_session, upload_id, "processing", 0.9, "Quality analysis completed")
                
                # Step 6: Finalize (10% of progress)
                await progress_tracker.update_progress(0.95, "Finalizing processing...")
                
                # Update file upload record with all results
                file_upload = await db_session.get(FileUpload, upload_id)
                if file_upload:
                    file_upload.schema_info = schema_info
                    file_upload.preview_data = preview_data
                    file_upload.quality_report = quality_report
                    file_upload.processing_status = "completed"
                    file_upload.processing_progress = 1.0
                    file_upload.processing_message = "Processing completed successfully"
                    await db_session.commit()
                
                await progress_tracker.update_progress(1.0, "File processing completed successfully")
                
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            await progress_tracker.update_progress(-1, error_msg)
            
            # Update status to failed
            try:
                db_pool = await get_optimized_db_pool()
                async with db_pool.get_async_session() as db_session:
                    await self._update_file_status(
                        db_session, upload_id, "failed", -1, error_msg,
                        error_details={"error": str(e), "timestamp": datetime.utcnow().isoformat()}
                    )
            except Exception as update_error:
                logger.error(f"Failed to update error status: {update_error}")
    
    async def _update_file_status(
        self, 
        db_session, 
        upload_id: str, 
        status: str, 
        progress: float, 
        message: str,
        error_details: Dict[str, Any] = None
    ) -> None:
        """Update file processing status in database."""
        
        file_upload = await db_session.get(FileUpload, upload_id)
        if file_upload:
            file_upload.processing_status = status
            file_upload.processing_progress = progress
            file_upload.processing_message = message
            if error_details:
                file_upload.error_details = error_details
            await db_session.commit()
    
    async def _infer_schema_with_progress(
        self, 
        df: pd.DataFrame, 
        progress_tracker: ProgressTracker,
        start_progress: float,
        end_progress: float
    ) -> Dict[str, Any]:
        """Infer schema with progress updates."""
        
        total_columns = len(df.columns)
        progress_step = (end_progress - start_progress) / max(total_columns, 1)
        
        schema_info = {
            "columns": {},
            "total_rows": len(df),
            "total_columns": total_columns,
            "inferred_at": datetime.utcnow().isoformat(),
            "data_quality_summary": {}
        }
        
        for i, column in enumerate(df.columns):
            current_progress = start_progress + (i * progress_step)
            await progress_tracker.update_progress(
                current_progress, 
                f"Analyzing column {i+1}/{total_columns}: {column}"
            )
            
            col_info = await self._analyze_column_detailed(df[column], column)
            schema_info["columns"][column] = col_info
        
        # Add data quality summary
        schema_info["data_quality_summary"] = {
            "total_missing_values": int(df.isnull().sum().sum()),
            "columns_with_missing": int((df.isnull().sum() > 0).sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        }
        
        return schema_info
    
    async def _analyze_column_detailed(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Perform detailed column analysis."""
        
        col_info = {
            "name": column_name,
            "dtype": str(series.dtype),
            "non_null_count": int(series.count()),
            "null_count": int(series.isnull().sum()),
            "unique_count": int(series.nunique()),
            "inferred_type": self._infer_column_type(series),
            "memory_usage_bytes": int(series.memory_usage(deep=True))
        }
        
        # Add sample values (non-null only)
        sample_values = series.dropna().head(5).tolist()
        col_info["sample_values"] = [str(v) for v in sample_values]
        
        # Add statistics for numeric columns
        if pd.api.types.is_numeric_dtype(series):
            col_info["statistics"] = {
                "mean": float(series.mean()) if not series.empty else None,
                "median": float(series.median()) if not series.empty else None,
                "std": float(series.std()) if not series.empty else None,
                "min": float(series.min()) if not series.empty else None,
                "max": float(series.max()) if not series.empty else None,
                "q25": float(series.quantile(0.25)) if not series.empty else None,
                "q75": float(series.quantile(0.75)) if not series.empty else None
            }
        
        # Add frequency information for categorical columns
        if col_info["inferred_type"] == "categorical" or series.dtype == 'object':
            value_counts = series.value_counts().head(10)
            col_info["top_values"] = {
                str(k): int(v) for k, v in value_counts.items()
            }
        
        return col_info
    
    async def _generate_preview_with_progress(
        self, 
        df: pd.DataFrame, 
        progress_tracker: ProgressTracker,
        start_progress: float,
        end_progress: float
    ) -> Dict[str, Any]:
        """Generate preview with progress tracking."""
        
        await progress_tracker.update_progress(start_progress + 0.02, "Preparing data preview...")
        
        # Limit rows for preview
        preview_df = df.head(self.max_preview_rows)
        
        await progress_tracker.update_progress(start_progress + 0.05, "Formatting preview data...")
        
        # Convert to JSON-serializable format
        preview_data = {
            "columns": [
                {
                    "name": col,
                    "type": str(df[col].dtype),
                    "inferred_type": self._infer_column_type(df[col])
                }
                for col in df.columns
            ],
            "sample_data": preview_df.fillna("").to_dict('records'),
            "total_rows": len(df),
            "preview_rows": len(preview_df),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        await progress_tracker.update_progress(start_progress + 0.08, "Adding statistical summary...")
        
        # Add basic statistics for numeric columns
        if len(df) > 0:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                preview_data["statistics"] = df[numeric_cols].describe().to_dict()
        
        return preview_data
    
    async def _generate_quality_report_with_progress(
        self, 
        df: pd.DataFrame, 
        progress_tracker: ProgressTracker,
        start_progress: float,
        end_progress: float
    ) -> Dict[str, Any]:
        """Generate quality report with progress tracking."""
        
        if df.empty:
            return {
                "total_rows": 0,
                "total_columns": 0,
                "missing_values": {},
                "duplicate_rows": 0,
                "data_type_issues": [],
                "outliers": {},
                "quality_score": 0.0,
                "recommendations": ["File appears to be empty or could not be processed"]
            }
        
        progress_step = (end_progress - start_progress) / 6  # 6 analysis steps
        
        # Step 1: Basic metrics
        await progress_tracker.update_progress(start_progress + progress_step, "Calculating basic metrics...")
        total_rows = len(df)
        total_columns = len(df.columns)
        
        # Step 2: Missing values analysis
        await progress_tracker.update_progress(start_progress + 2*progress_step, "Analyzing missing values...")
        missing_values = df.isnull().sum().to_dict()
        missing_values = {k: int(v) for k, v in missing_values.items()}
        
        # Step 3: Duplicate analysis
        await progress_tracker.update_progress(start_progress + 3*progress_step, "Checking for duplicates...")
        duplicate_rows = int(df.duplicated().sum())
        
        # Step 4: Data type issues
        await progress_tracker.update_progress(start_progress + 4*progress_step, "Analyzing data types...")
        data_type_issues = await self._analyze_data_type_issues(df)
        
        # Step 5: Outlier detection
        await progress_tracker.update_progress(start_progress + 5*progress_step, "Detecting outliers...")
        outliers = await self._detect_outliers(df)
        
        # Step 6: Calculate quality score and recommendations
        await progress_tracker.update_progress(start_progress + 6*progress_step, "Calculating quality score...")
        quality_score = self._calculate_quality_score(
            total_rows, total_columns, missing_values, 
            duplicate_rows, data_type_issues, outliers
        )
        
        recommendations = self._generate_recommendations(
            missing_values, duplicate_rows, data_type_issues, outliers
        )
        
        return {
            "total_rows": total_rows,
            "total_columns": total_columns,
            "missing_values": missing_values,
            "duplicate_rows": duplicate_rows,
            "data_type_issues": data_type_issues,
            "outliers": outliers,
            "quality_score": quality_score,
            "recommendations": recommendations,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    async def _analyze_data_type_issues(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze data type inconsistencies."""
        
        data_type_issues = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for mixed types in object columns
                sample_values = df[col].dropna().head(100)
                if len(sample_values) > 0:
                    types_found = set()
                    for val in sample_values:
                        if isinstance(val, str):
                            if val.isdigit():
                                types_found.add('numeric_string')
                            elif val.lower() in ['true', 'false', 'yes', 'no']:
                                types_found.add('boolean_string')
                            else:
                                types_found.add('text')
                        else:
                            types_found.add(type(val).__name__)
                    
                    if len(types_found) > 1:
                        data_type_issues.append({
                            "column": col,
                            "issue": "mixed_types",
                            "types_found": list(types_found),
                            "recommendation": f"Consider standardizing data types in column '{col}'"
                        })
        
        return data_type_issues
    
    async def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, List[Any]]:
        """Detect outliers in numeric columns."""
        
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 10:  # Need sufficient data points
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_values = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                if len(outlier_values) > 0:
                    outliers[col] = {
                        "count": len(outlier_values),
                        "percentage": round(len(outlier_values) / len(col_data) * 100, 2),
                        "sample_values": outlier_values.head(10).tolist(),
                        "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)}
                    }
        
        return outliers
    
    async def _load_data(self, file_path: str, detected_type: str) -> pd.DataFrame:
        """Load data from file with enhanced error handling and preprocessing."""
        
        try:
            if detected_type == 'csv':
                return await self._load_csv_with_detection(file_path)
                
            elif detected_type == 'xlsx':
                return await self._load_excel_file(file_path)
                
            elif detected_type == 'json':
                return await self._load_json_file(file_path)
            
            else:
                raise EngineError(f"Unsupported file type: {detected_type}")
                
        except Exception as e:
            raise EngineError(f"Failed to load data from {detected_type} file: {str(e)}")
    
    async def _load_csv_with_detection(self, file_path: str) -> pd.DataFrame:
        """Load CSV with automatic delimiter and encoding detection."""
        
        # Try different encodings and separators
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        separators = [',', ';', '\t', '|']
        
        best_df = None
        best_score = 0
        
        for encoding in encodings:
            for sep in separators:
                try:
                    # Test with small sample first
                    df_sample = pd.read_csv(file_path, encoding=encoding, sep=sep, nrows=10)
                    
                    # Score based on number of columns and data quality
                    score = len(df_sample.columns)
                    if len(df_sample.columns) > 1 and len(df_sample) > 0:
                        # Bonus for having data in multiple columns
                        non_empty_cols = sum(1 for col in df_sample.columns 
                                           if not df_sample[col].isna().all())
                        score += non_empty_cols * 2
                        
                        if score > best_score:
                            best_score = score
                            # Load full file with best parameters
                            best_df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                            
                except Exception:
                    continue
        
        if best_df is None:
            # Fallback to default pandas behavior
            best_df = pd.read_csv(file_path)
        
        return best_df
    
    async def _load_excel_file(self, file_path: str) -> pd.DataFrame:
        """Load Excel file with error handling."""
        try:
            # Try to read the first sheet
            df = pd.read_excel(file_path, sheet_name=0)
            
            # If empty, try other sheets
            if df.empty:
                excel_file = pd.ExcelFile(file_path)
                for sheet_name in excel_file.sheet_names:
                    try:
                        df = pd.read_excel(file_path, sheet_name=sheet_name)
                        if not df.empty:
                            break
                    except Exception:
                        continue
            
            return df
            
        except Exception as e:
            raise EngineError(f"Failed to load Excel file: {str(e)}")
    
    async def _load_json_file(self, file_path: str) -> pd.DataFrame:
        """Load JSON file with format detection."""
        try:
            # First, try to determine JSON structure
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
            
            # Try JSON Lines format first
            if first_line.startswith('{'):
                try:
                    df = pd.read_json(file_path, lines=True)
                    if not df.empty:
                        return df
                except Exception:
                    pass
            
            # Try regular JSON array format
            try:
                df = pd.read_json(file_path)
                if not df.empty:
                    return df
            except Exception:
                pass
            
            # Try loading as raw JSON and normalizing
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                df = pd.json_normalize(data)
            elif isinstance(data, dict):
                df = pd.json_normalize([data])
            else:
                raise EngineError("JSON file format not supported")
            
            return df
            
        except Exception as e:
            raise EngineError(f"Failed to load JSON file: {str(e)}")
    
    async def preprocess_data(self, df: pd.DataFrame, options: Dict[str, Any] = None) -> pd.DataFrame:
        """Comprehensive data preprocessing pipeline."""
        
        if df.empty:
            return df
        
        options = options or {}
        processed_df = df.copy()
        
        # Step 1: Basic cleaning
        if options.get('remove_empty_rows', True):
            processed_df = processed_df.dropna(how='all')
        
        if options.get('remove_empty_columns', True):
            processed_df = processed_df.dropna(axis=1, how='all')
        
        # Step 2: Handle duplicates
        if options.get('remove_duplicates', False):
            processed_df = processed_df.drop_duplicates()
        
        # Step 3: Data type optimization
        if options.get('optimize_dtypes', True):
            processed_df = self._optimize_data_types(processed_df)
        
        # Step 4: Handle missing values
        if options.get('handle_missing', True):
            processed_df = self._handle_missing_values(processed_df, options)
        
        # Step 5: Normalize text data
        if options.get('normalize_text', True):
            processed_df = self._normalize_text_columns(processed_df)
        
        # Step 6: Handle outliers
        if options.get('handle_outliers', False):
            processed_df = self._handle_outliers(processed_df, options)
        
        return processed_df
    
    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency."""
        
        optimized_df = df.copy()
        
        for col in optimized_df.columns:
            col_data = optimized_df[col]
            
            # Skip if all null
            if col_data.isna().all():
                continue
            
            # Try to convert to numeric if possible
            if col_data.dtype == 'object':
                # Try numeric conversion
                try:
                    numeric_data = pd.to_numeric(col_data, errors='coerce')
                    if not numeric_data.isna().all():
                        # Check if it can be integer
                        if numeric_data.dropna().apply(lambda x: x.is_integer()).all():
                            optimized_df[col] = numeric_data.astype('Int64')  # Nullable integer
                        else:
                            optimized_df[col] = numeric_data
                        continue
                except Exception:
                    pass
                
                # Try datetime conversion
                try:
                    datetime_data = pd.to_datetime(col_data, errors='coerce', infer_datetime_format=True)
                    if not datetime_data.isna().all():
                        optimized_df[col] = datetime_data
                        continue
                except Exception:
                    pass
                
                # Try boolean conversion
                if col_data.dropna().str.lower().isin(['true', 'false', 'yes', 'no', '1', '0']).all():
                    bool_map = {'true': True, 'false': False, 'yes': True, 'no': False, '1': True, '0': False}
                    optimized_df[col] = col_data.str.lower().map(bool_map)
        
        return optimized_df
    
    def _handle_missing_values(self, df: pd.DataFrame, options: Dict[str, Any]) -> pd.DataFrame:
        """Handle missing values based on column types."""
        
        processed_df = df.copy()
        strategy = options.get('missing_strategy', 'auto')
        
        for col in processed_df.columns:
            col_data = processed_df[col]
            missing_count = col_data.isna().sum()
            
            if missing_count == 0:
                continue
            
            # Determine strategy based on data type and missing percentage
            missing_pct = missing_count / len(col_data)
            
            if missing_pct > 0.5:  # More than 50% missing
                if strategy == 'auto':
                    continue  # Keep as is for high missing percentage
            
            if pd.api.types.is_numeric_dtype(col_data):
                # Numeric columns: use median for auto strategy
                if strategy in ['auto', 'median']:
                    processed_df[col] = col_data.fillna(col_data.median())
                elif strategy == 'mean':
                    processed_df[col] = col_data.fillna(col_data.mean())
                elif strategy == 'zero':
                    processed_df[col] = col_data.fillna(0)
                    
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                # Datetime columns: forward fill or use mode
                if strategy in ['auto', 'forward']:
                    processed_df[col] = col_data.fillna(method='ffill')
                elif strategy == 'mode':
                    mode_val = col_data.mode()
                    if not mode_val.empty:
                        processed_df[col] = col_data.fillna(mode_val.iloc[0])
                        
            else:
                # Categorical/text columns: use mode or 'Unknown'
                if strategy in ['auto', 'mode']:
                    mode_val = col_data.mode()
                    if not mode_val.empty:
                        processed_df[col] = col_data.fillna(mode_val.iloc[0])
                    else:
                        processed_df[col] = col_data.fillna('Unknown')
                elif strategy == 'unknown':
                    processed_df[col] = col_data.fillna('Unknown')
        
        return processed_df
    
    def _normalize_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize text columns for consistency."""
        
        processed_df = df.copy()
        
        for col in processed_df.columns:
            if processed_df[col].dtype == 'object':
                # Basic text normalization
                text_data = processed_df[col].astype(str)
                
                # Strip whitespace
                text_data = text_data.str.strip()
                
                # Handle common inconsistencies
                text_data = text_data.replace(['N/A', 'n/a', 'NULL', 'null', 'None', 'none', ''], pd.NA)
                
                processed_df[col] = text_data
        
        return processed_df
    
    def _handle_outliers(self, df: pd.DataFrame, options: Dict[str, Any]) -> pd.DataFrame:
        """Handle outliers in numeric columns."""
        
        processed_df = df.copy()
        method = options.get('outlier_method', 'iqr')
        threshold = options.get('outlier_threshold', self.outlier_detection_threshold)
        
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = processed_df[col].dropna()
            
            if len(col_data) < 10:  # Skip if too few data points
                continue
            
            if method == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                processed_df[col] = processed_df[col].clip(lower=lower_bound, upper=upper_bound)
                
            elif method == 'zscore':
                mean_val = col_data.mean()
                std_val = col_data.std()
                
                if std_val > 0:
                    lower_bound = mean_val - threshold * std_val
                    upper_bound = mean_val + threshold * std_val
                    processed_df[col] = processed_df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return processed_df
    
    def _infer_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Infer schema from DataFrame."""
        
        schema_info = {
            "columns": {},
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "inferred_at": datetime.utcnow().isoformat()
        }
        
        for column in df.columns:
            col_info = {
                "name": column,
                "dtype": str(df[column].dtype),
                "non_null_count": int(df[column].count()),
                "null_count": int(df[column].isnull().sum()),
                "unique_count": int(df[column].nunique()),
                "inferred_type": self._infer_column_type(df[column])
            }
            
            # Add sample values
            sample_values = df[column].dropna().head(5).tolist()
            col_info["sample_values"] = [str(v) for v in sample_values]
            
            # Add statistics for numeric columns
            if pd.api.types.is_numeric_dtype(df[column]):
                col_info["statistics"] = {
                    "mean": float(df[column].mean()) if not df[column].empty else None,
                    "median": float(df[column].median()) if not df[column].empty else None,
                    "std": float(df[column].std()) if not df[column].empty else None,
                    "min": float(df[column].min()) if not df[column].empty else None,
                    "max": float(df[column].max()) if not df[column].empty else None
                }
            
            schema_info["columns"][column] = col_info
        
        return schema_info
    
    def _infer_column_type(self, series: pd.Series) -> str:
        """Infer the semantic type of a column."""
        
        # Skip if empty
        if series.empty or series.isnull().all():
            return "unknown"
        
        # Check for common patterns
        sample_values = series.dropna().astype(str).head(100)
        
        # Email pattern
        if sample_values.str.contains(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$').any():
            return "email"
        
        # URL pattern
        if sample_values.str.contains(r'^https?://').any():
            return "url"
        
        # Phone pattern
        if sample_values.str.contains(r'^\+?[\d\s\-\(\)]+$').any():
            return "phone"
        
        # Date patterns
        try:
            pd.to_datetime(sample_values.head(10))
            return "datetime"
        except:
            pass
        
        # Numeric types
        if pd.api.types.is_numeric_dtype(series):
            if pd.api.types.is_integer_dtype(series):
                return "integer"
            else:
                return "float"
        
        # Boolean
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        
        # Categorical (if low cardinality)
        if series.nunique() / len(series) < 0.1 and series.nunique() < 50:
            return "categorical"
        
        # Default to text
        return "text"
    
    def _generate_preview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate data preview."""
        
        # Limit rows for preview
        preview_df = df.head(self.max_preview_rows)
        
        # Convert to JSON-serializable format
        preview_data = {
            "columns": [
                {
                    "name": col,
                    "type": str(df[col].dtype),
                    "inferred_type": self._infer_column_type(df[col])
                }
                for col in df.columns
            ],
            "sample_data": preview_df.fillna("").to_dict('records'),
            "total_rows": len(df),
            "preview_rows": len(preview_df),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        # Add basic statistics
        if len(df) > 0:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                preview_data["statistics"] = df[numeric_cols].describe().to_dict()
        
        return preview_data
    
    def _generate_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate data quality report."""
        
        if df.empty:
            return {
                "total_rows": 0,
                "total_columns": 0,
                "missing_values": {},
                "duplicate_rows": 0,
                "data_type_issues": [],
                "outliers": {},
                "quality_score": 0.0,
                "recommendations": ["File appears to be empty or could not be processed"]
            }
        
        # Basic metrics
        total_rows = len(df)
        total_columns = len(df.columns)
        
        # Missing values
        missing_values = df.isnull().sum().to_dict()
        missing_values = {k: int(v) for k, v in missing_values.items()}
        
        # Duplicate rows
        duplicate_rows = int(df.duplicated().sum())
        
        # Data type issues
        data_type_issues = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for mixed types
                sample_values = df[col].dropna().head(100)
                if len(sample_values) > 0:
                    types_found = set()
                    for val in sample_values:
                        if isinstance(val, str):
                            if val.isdigit():
                                types_found.add('numeric_string')
                            elif val.lower() in ['true', 'false', 'yes', 'no']:
                                types_found.add('boolean_string')
                            else:
                                types_found.add('text')
                        else:
                            types_found.add(type(val).__name__)
                    
                    if len(types_found) > 1:
                        data_type_issues.append({
                            "column": col,
                            "issue": "mixed_types",
                            "types_found": list(types_found),
                            "recommendation": f"Consider standardizing data types in column '{col}'"
                        })
        
        # Outliers for numeric columns
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if len(df[col].dropna()) > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_values = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                if len(outlier_values) > 0:
                    outliers[col] = outlier_values.head(10).tolist()
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            total_rows, total_columns, missing_values, 
            duplicate_rows, data_type_issues, outliers
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            missing_values, duplicate_rows, data_type_issues, outliers
        )
        
        return {
            "total_rows": total_rows,
            "total_columns": total_columns,
            "missing_values": missing_values,
            "duplicate_rows": duplicate_rows,
            "data_type_issues": data_type_issues,
            "outliers": outliers,
            "quality_score": quality_score,
            "recommendations": recommendations,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def _calculate_quality_score(
        self, 
        total_rows: int, 
        total_columns: int, 
        missing_values: Dict[str, int],
        duplicate_rows: int,
        data_type_issues: List[Dict],
        outliers: Dict[str, List]
    ) -> float:
        """Calculate overall data quality score (0-100)."""
        
        if total_rows == 0 or total_columns == 0:
            return 0.0
        
        score = 100.0
        
        # Penalize missing values
        total_cells = total_rows * total_columns
        total_missing = sum(missing_values.values())
        missing_penalty = (total_missing / total_cells) * 30
        score -= missing_penalty
        
        # Penalize duplicates
        duplicate_penalty = (duplicate_rows / total_rows) * 20
        score -= duplicate_penalty
        
        # Penalize data type issues
        type_issue_penalty = len(data_type_issues) * 5
        score -= type_issue_penalty
        
        # Penalize outliers (less severe)
        outlier_penalty = len(outliers) * 2
        score -= outlier_penalty
        
        return max(0.0, min(100.0, score))
    
    def _generate_recommendations(
        self,
        missing_values: Dict[str, int],
        duplicate_rows: int,
        data_type_issues: List[Dict],
        outliers: Dict[str, List]
    ) -> List[str]:
        """Generate data quality recommendations."""
        
        recommendations = []
        
        # Missing values recommendations
        high_missing_cols = [col for col, count in missing_values.items() if count > 0]
        if high_missing_cols:
            recommendations.append(
                f"Consider handling missing values in columns: {', '.join(high_missing_cols[:5])}"
            )
        
        # Duplicate recommendations
        if duplicate_rows > 0:
            recommendations.append(
                f"Found {duplicate_rows} duplicate rows - consider removing duplicates"
            )
        
        # Data type recommendations
        if data_type_issues:
            recommendations.append(
                "Some columns have mixed data types - consider data type standardization"
            )
        
        # Outlier recommendations
        if outliers:
            recommendations.append(
                f"Outliers detected in {len(outliers)} columns - review for data entry errors"
            )
        
        # General recommendations
        if not recommendations:
            recommendations.append("Data quality looks good - no major issues detected")
        
        return recommendations
    
    async def get_processing_status(self, upload_id: str, db_session) -> Dict[str, Any]:
        """Get current processing status for an upload."""
        
        from ..models.database import FileUpload
        
        file_upload = db_session.query(FileUpload).filter_by(upload_id=upload_id).first()
        
        if not file_upload:
            raise EngineError(f"Upload with ID {upload_id} not found")
        
        return {
            "upload_id": upload_id,
            "status": file_upload.processing_status,
            "progress": file_upload.processing_progress,
            "message": file_upload.processing_message,
            "error_details": file_upload.error_details,
            "created_at": file_upload.created_at,
            "updated_at": file_upload.updated_at
        }
    
    async def validate_file_format(self, file: UploadFile) -> Dict[str, Any]:
        """Validate file format without uploading."""
        
        try:
            await self._validate_file_optimized(file)
            
            # Detect file type
            detected_type = await self._detect_file_type_optimized(
                file.content_type, file.filename, file
            )
            
            return {
                "valid": True,
                "detected_type": detected_type,
                "file_size": file.size,
                "content_type": file.content_type,
                "message": "File format is valid and supported"
            }
            
        except EngineError as e:
            return {
                "valid": False,
                "error": str(e),
                "file_size": file.size if file.size else 0,
                "content_type": file.content_type,
                "message": f"File validation failed: {str(e)}"
            }
    
    async def get_supported_formats(self) -> Dict[str, Any]:
        """Get detailed information about supported file formats."""
        
        formats = []
        
        for content_type, file_type in self.supported_types.items():
            format_info = {
                "content_type": content_type,
                "file_type": file_type,
                "description": self._get_format_description(file_type),
                "typical_extensions": self._get_typical_extensions(file_type),
                "max_size_mb": self.max_file_size / (1024 * 1024)
            }
            formats.append(format_info)
        
        return {
            "supported_formats": formats,
            "max_file_size": self.max_file_size,
            "max_file_size_mb": self.max_file_size / (1024 * 1024),
            "allowed_extensions": self.allowed_extensions,
            "features": {
                "automatic_schema_detection": True,
                "data_preview": True,
                "quality_analysis": True,
                "data_preprocessing": True,
                "progress_tracking": True,
                "security_validation": True
            }
        }
    
    def _get_format_description(self, file_type: str) -> str:
        """Get description for file format."""
        
        descriptions = {
            "csv": "Comma-separated values - tabular data in plain text format",
            "xlsx": "Microsoft Excel spreadsheet - binary format with multiple sheets support",
            "json": "JavaScript Object Notation - structured data format"
        }
        
        return descriptions.get(file_type, f"Supported format: {file_type}")
    
    def _get_typical_extensions(self, file_type: str) -> List[str]:
        """Get typical file extensions for format."""
        
        extensions = {
            "csv": [".csv", ".txt"],
            "xlsx": [".xlsx", ".xls"],
            "json": [".json"]
        }
        
        return extensions.get(file_type, [])
    
    async def list_user_uploads(
        self, 
        user_id: str, 
        db_session,
        skip: int = 0,
        limit: int = 100,
        status_filter: str = None,
        search_term: str = None
    ) -> List[Dict[str, Any]]:
        """List uploads for a user with filtering and search."""
        
        from ..models.database import FileUpload
        
        query = db_session.query(FileUpload).filter_by(user_id=user_id)
        
        if status_filter:
            query = query.filter_by(processing_status=status_filter)
        
        if search_term:
            search_pattern = f"%{search_term}%"
            query = query.filter(
                FileUpload.original_filename.ilike(search_pattern)
            )
        
        uploads = query.order_by(FileUpload.created_at.desc()).offset(skip).limit(limit).all()
        
        return [
            {
                "upload_id": upload.upload_id,
                "filename": upload.filename,
                "original_filename": upload.original_filename,
                "file_size": upload.file_size,
                "detected_type": upload.detected_type,
                "processing_status": upload.processing_status,
                "processing_progress": upload.processing_progress,
                "created_at": upload.created_at,
                "updated_at": upload.updated_at
            }
            for upload in uploads
        ]
    
    async def create_dataset_from_upload(
        self,
        upload_id: str,
        dataset_name: str,
        dataset_description: str = None,
        db_session = None
    ) -> str:
        """Create a dataset from a processed upload."""
        
        from ..models.database import FileUpload, Dataset
        import uuid
        
        # Get the upload record
        file_upload = db_session.query(FileUpload).filter_by(upload_id=upload_id).first()
        
        if not file_upload:
            raise EngineError(f"Upload with ID {upload_id} not found")
        
        if file_upload.processing_status != "completed":
            raise EngineError(f"Upload processing not completed. Status: {file_upload.processing_status}")
        
        # Create dataset record
        dataset_id = uuid.uuid4()
        dataset = Dataset(
            id=dataset_id,
            name=dataset_name,
            description=dataset_description,
            source_type="file_upload",
            data_schema=file_upload.schema_info,
            dataset_metadata={
                "upload_id": upload_id,
                "original_filename": file_upload.original_filename,
                "detected_type": file_upload.detected_type,
                "quality_report": file_upload.quality_report
            },
            row_count=file_upload.schema_info.get("total_rows", 0) if file_upload.schema_info else 0,
            file_path=file_upload.file_path,
            is_active=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        db_session.add(dataset)
        
        # Update file upload to reference the dataset
        file_upload.dataset_id = dataset_id
        
        db_session.commit()
        
        return str(dataset_id)
    
    # Base engine interface methods
    async def initialize(self) -> None:
        """Initialize the file processor engine."""
        
        # Create temp directory if it doesn't exist
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Verify required libraries are available
        try:
            import pandas as pd
            import numpy as np
            import json
        except ImportError as e:
            raise EngineError(f"Required library not available: {e}")
        
        self.status = "ready"
        logger.info("File Processor Engine initialized successfully")
    
    async def process(self, input_data: Any, parameters: Dict[str, Any] = None) -> Any:
        """Process input data (file upload)."""
        
        if not isinstance(input_data, dict) or "file" not in input_data:
            raise EngineError("Input data must contain 'file' key with UploadFile object")
        
        file = input_data["file"]
        user_id = input_data.get("user_id")
        storage_path = input_data.get("storage_path")
        db_session = input_data.get("db_session")
        
        if not all([user_id, storage_path, db_session]):
            raise EngineError("Missing required parameters: user_id, storage_path, db_session")
        
        # Process the file upload
        return await self.process_upload(
            file=file,
            user_id=user_id,
            storage_path=storage_path,
            db_session=db_session,
            auto_detect_schema=parameters.get("auto_detect_schema", True),
            generate_preview=parameters.get("generate_preview", True),
            use_background_processing=parameters.get("use_background_processing", True)
        )
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        
        # Clear progress callbacks
        self.progress_callbacks.clear()
        
        logger.info("File Processor Engine cleaned up successfully")
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        
        return {
            "engine_id": self.engine_id,
            "name": self.name,
            "status": self.status,
            "healthy": self.status == "ready",
            "supported_formats": len(self.supported_types),
            "max_file_size_mb": self.max_file_size / (1024 * 1024),
            "features": {
                "schema_detection": True,
                "data_preview": True,
                "quality_analysis": True,
                "preprocessing": True,
                "progress_tracking": True,
                "security_validation": True
            }
        }
    
    async def _load_csv_chunked(self, file_path: str, progress_tracker: ProgressTracker) -> pd.DataFrame:
        """Load CSV file in chunks with progress tracking."""
        
        chunks = []
        chunk_size = 10000  # Process 10k rows at a time
        
        try:
            # Get total number of lines for progress calculation
            with open(file_path, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f) - 1  # Subtract header
            
            await progress_tracker.update(10.0, "Starting chunked CSV processing...")
            
            # Read CSV in chunks
            chunk_reader = pd.read_csv(file_path, chunksize=chunk_size)
            processed_lines = 0
            
            for i, chunk in enumerate(chunk_reader):
                chunks.append(chunk)
                processed_lines += len(chunk)
                
                # Update progress
                progress = 10.0 + (processed_lines / total_lines) * 60.0  # 10-70% for loading
                await progress_tracker.update(
                    progress, 
                    f"Processed {processed_lines:,} of {total_lines:,} rows..."
                )
                
                # Prevent memory issues
                if len(chunks) > 100:  # Combine chunks periodically
                    combined_chunk = pd.concat(chunks, ignore_index=True)
                    chunks = [combined_chunk]
            
            # Combine all chunks
            await progress_tracker.update(70.0, "Combining data chunks...")
            df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            
            await progress_tracker.update(80.0, f"CSV loading completed: {len(df):,} rows")
            return df
            
        except Exception as e:
            await progress_tracker.update(0.0, f"CSV loading failed: {str(e)}")
            raise EngineError(f"Failed to load CSV file in chunks: {str(e)}")
    
    async def _infer_schema_async(self, df: pd.DataFrame, progress_tracker: ProgressTracker) -> Dict[str, Any]:
        """Infer schema asynchronously with progress tracking."""
        
        await progress_tracker.update(0.0, "Starting schema inference...")
        
        schema_info = {
            "columns": {},
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "inferred_at": datetime.utcnow().isoformat()
        }
        
        total_columns = len(df.columns)
        
        for i, column in enumerate(df.columns):
            # Update progress
            progress = (i / total_columns) * 100
            await progress_tracker.update(progress, f"Analyzing column: {column}")
            
            col_info = {
                "name": column,
                "dtype": str(df[column].dtype),
                "non_null_count": int(df[column].count()),
                "null_count": int(df[column].isnull().sum()),
                "unique_count": int(df[column].nunique()),
                "inferred_type": self._infer_column_type(df[column])
            }
            
            # Add sample values
            sample_values = df[column].dropna().head(5).tolist()
            col_info["sample_values"] = [str(v) for v in sample_values]
            
            # Add statistics for numeric columns
            if pd.api.types.is_numeric_dtype(df[column]):
                col_info["statistics"] = {
                    "mean": float(df[column].mean()) if not df[column].empty else None,
                    "median": float(df[column].median()) if not df[column].empty else None,
                    "std": float(df[column].std()) if not df[column].empty else None,
                    "min": float(df[column].min()) if not df[column].empty else None,
                    "max": float(df[column].max()) if not df[column].empty else None
                }
            
            schema_info["columns"][column] = col_info
            
            # Allow other tasks to run
            if i % 10 == 0:
                await asyncio.sleep(0.01)
        
        await progress_tracker.update(100.0, "Schema inference completed")
        return schema_info
    
    async def _generate_quality_report_async(self, df: pd.DataFrame, progress_tracker: ProgressTracker) -> Dict[str, Any]:
        """Generate quality report asynchronously with progress tracking."""
        
        await progress_tracker.update(0.0, "Starting quality analysis...")
        
        if df.empty:
            await progress_tracker.update(100.0, "Quality analysis completed (empty dataset)")
            return {
                "total_rows": 0,
                "total_columns": 0,
                "missing_values": {},
                "duplicate_rows": 0,
                "data_type_issues": [],
                "outliers": {},
                "quality_score": 0.0,
                "recommendations": ["File appears to be empty or could not be processed"]
            }
        
        # Basic metrics
        total_rows = len(df)
        total_columns = len(df.columns)
        
        await progress_tracker.update(20.0, "Analyzing missing values...")
        
        # Missing values
        missing_values = df.isnull().sum().to_dict()
        missing_values = {k: int(v) for k, v in missing_values.items()}
        
        await progress_tracker.update(40.0, "Checking for duplicates...")
        
        # Duplicate rows
        duplicate_rows = int(df.duplicated().sum())
        
        await progress_tracker.update(60.0, "Analyzing data types...")
        
        # Data type issues
        data_type_issues = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for mixed types
                sample_values = df[col].dropna().head(100)
                if len(sample_values) > 0:
                    types_found = set()
                    for val in sample_values:
                        if isinstance(val, str):
                            if val.isdigit():
                                types_found.add('numeric_string')
                            elif val.lower() in ['true', 'false', 'yes', 'no']:
                                types_found.add('boolean_string')
                            else:
                                types_found.add('text')
                        else:
                            types_found.add(type(val).__name__)
                    
                    if len(types_found) > 1:
                        data_type_issues.append({
                            "column": col,
                            "issue": "mixed_types",
                            "types_found": list(types_found),
                            "recommendation": f"Consider standardizing data types in column '{col}'"
                        })
        
        await progress_tracker.update(80.0, "Detecting outliers...")
        
        # Outliers for numeric columns
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if len(df[col].dropna()) > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_values = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                if len(outlier_values) > 0:
                    outliers[col] = outlier_values.head(10).tolist()
        
        await progress_tracker.update(90.0, "Calculating quality score...")
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            total_rows, total_columns, missing_values, 
            duplicate_rows, data_type_issues, outliers
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            missing_values, duplicate_rows, data_type_issues, outliers
        )
        
        await progress_tracker.update(100.0, "Quality analysis completed")
        
        return {
            "total_rows": total_rows,
            "total_columns": total_columns,
            "missing_values": missing_values,
            "duplicate_rows": duplicate_rows,
            "data_type_issues": data_type_issues,
            "outliers": outliers,
            "quality_score": quality_score,
            "recommendations": recommendations,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    async def get_processing_status(self, upload_id: str, db_session=None) -> Dict[str, Any]:
        """Get processing status for an upload."""
        
        if db_session:
            # Sync session provided
            file_upload = db_session.query(FileUpload).filter_by(upload_id=upload_id).first()
        else:
            # Use async session
            db_pool = await get_optimized_db_pool()
            async with db_pool.get_async_session() as session:
                file_upload = await session.get(FileUpload, upload_id)
        
        if not file_upload:
            raise EngineError(f"Upload {upload_id} not found")
        
        return {
            "upload_id": upload_id,
            "status": file_upload.processing_status,
            "progress": getattr(file_upload, 'processing_progress', 0.0),
            "message": getattr(file_upload, 'processing_message', ''),
            "filename": file_upload.original_filename,
            "file_size": file_upload.file_size,
            "created_at": file_upload.created_at.isoformat() if file_upload.created_at else None,
            "error_details": getattr(file_upload, 'error_details', None)
        }
    
    async def cancel_processing(self, upload_id: str, user_id: str) -> Dict[str, Any]:
        """Cancel file processing."""
        
        db_pool = await get_optimized_db_pool()
        async with db_pool.get_async_session() as session:
            file_upload = await session.get(FileUpload, upload_id)
            
            if not file_upload:
                raise EngineError(f"Upload {upload_id} not found")
            
            if file_upload.user_id != user_id:
                raise EngineError("Unauthorized to cancel this upload")
            
            if file_upload.processing_status in ['completed', 'failed', 'cancelled']:
                raise EngineError(f"Cannot cancel upload with status: {file_upload.processing_status}")
            
            # Update status
            file_upload.processing_status = "cancelled"
            file_upload.processing_message = "Processing cancelled by user"
            await session.commit()
            
            return {
                "upload_id": upload_id,
                "status": "cancelled",
                "message": "Processing cancelled successfully"
            }
    
    async def create_dataset_from_upload(
        self, 
        upload_id: str, 
        dataset_name: str, 
        dataset_description: str = None,
        db_session=None
    ) -> str:
        """Create a dataset from a completed file upload."""
        
        if db_session:
            # Sync session
            file_upload = db_session.query(FileUpload).filter_by(upload_id=upload_id).first()
        else:
            # Async session
            db_pool = await get_optimized_db_pool()
            async with db_pool.get_async_session() as session:
                file_upload = await session.get(FileUpload, upload_id)
        
        if not file_upload:
            raise EngineError(f"Upload {upload_id} not found")
        
        if file_upload.processing_status != "completed":
            raise EngineError(f"Upload processing not completed. Status: {file_upload.processing_status}")
        
        # Create dataset record
        dataset = Dataset(
            name=dataset_name,
            description=dataset_description,
            source_type="file_upload",
            data_schema=file_upload.schema_info,
            dataset_metadata={
                "upload_id": upload_id,
                "original_filename": file_upload.original_filename,
                "file_size": file_upload.file_size,
                "detected_type": file_upload.detected_type,
                "quality_report": file_upload.quality_report
            },
            row_count=file_upload.schema_info.get('total_rows', 0) if file_upload.schema_info else 0,
            file_path=file_upload.file_path,
            is_active=True
        )
        
        if db_session:
            db_session.add(dataset)
            db_session.commit()
            db_session.refresh(dataset)
        else:
            async with db_pool.get_async_session() as session:
                session.add(dataset)
                await session.commit()
                await session.refresh(dataset)
        
        # Update file upload with dataset reference
        file_upload.dataset_id = dataset.id
        
        if db_session:
            db_session.commit()
        else:
            async with db_pool.get_async_session() as session:
                await session.merge(file_upload)
                await session.commit()
        
        return str(dataset.id)

    @background_job("file_processing")
    async def process_file_background(
        self,
        upload_id: str,
        file_path: str,
        detected_type: str,
        auto_detect_schema: bool = True,
        generate_preview: bool = True,
        user_id: str = None,
        file_hash: str = None,
        progress_tracker: ProgressTracker = None
    ) -> Dict[str, Any]:
        """Background job for processing uploaded files with progress tracking.""""Background job for processing large files."""
        
        try:
            if progress_tracker:
                await progress_tracker.update(5.0, "Initializing file processing...")
            
            # Get database session
            db_pool = await get_optimized_db_pool()
            async with db_pool.get_async_session() as db_session:
                # Update status to processing
                file_upload = await db_session.get(FileUpload, upload_id)
                if not file_upload:
                    raise EngineError(f"File upload {upload_id} not found")
                
                file_upload.processing_status = "processing"
                file_upload.processing_progress = 0.1
                file_upload.processing_message = "Starting background processing..."
                await db_session.commit()
                
                if progress_tracker:
                    await progress_tracker.update(10.0, "Loading data from file...")
                
                # Load data based on file type with chunked processing for large files
                df = await self._load_data_chunked(file_path, detected_type, progress_tracker)
                
                file_upload.processing_progress = 0.3
                file_upload.processing_message = "Data loaded, analyzing schema..."
                await db_session.commit()
                
                if progress_tracker:
                    await progress_tracker.update(30.0, "Analyzing data schema...")
                
                # Auto-detect schema if requested
                schema_info = {}
                if auto_detect_schema:
                    schema_info = await self._infer_schema_async(df, progress_tracker)
                
                file_upload.processing_progress = 0.5
                file_upload.processing_message = "Schema analysis complete, generating preview..."
                await db_session.commit()
                
                if progress_tracker:
                    await progress_tracker.update(50.0, "Generating data preview...")
                
                # Generate preview if requested
                preview_data = None
                if generate_preview:
                    preview_data = self._generate_preview(df)
                
                file_upload.processing_progress = 0.7
                file_upload.processing_message = "Preview generated, running quality checks..."
                await db_session.commit()
                
                if progress_tracker:
                    await progress_tracker.update(70.0, "Running data quality analysis...")
                
                # Generate quality report
                quality_report = await self._generate_quality_report_async(df, progress_tracker)
                
                file_upload.processing_progress = 0.9
                file_upload.processing_message = "Quality analysis complete, finalizing..."
                await db_session.commit()
                
                if progress_tracker:
                    await progress_tracker.update(90.0, "Finalizing processing...")
                
                # Update file upload record
                file_upload.schema_info = schema_info
                file_upload.preview_data = preview_data
                file_upload.quality_report = quality_report
                file_upload.processing_status = "completed"
                file_upload.processing_progress = 1.0
                file_upload.processing_message = "Processing completed successfully"
                file_upload.file_hash = file_hash
                await db_session.commit()
                
                if progress_tracker:
                    await progress_tracker.update(100.0, "File processing completed successfully!")
                
                return {
                    "upload_id": upload_id,
                    "status": "completed",
                    "schema_info": schema_info,
                    "quality_report": quality_report,
                    "file_hash": file_hash
                }
                
        except Exception as e:
            # Update status to failed
            try:
                db_pool = await get_optimized_db_pool()
                async with db_pool.get_async_session() as db_session:
                    file_upload = await db_session.get(FileUpload, upload_id)
                    if file_upload:
                        file_upload.processing_status = "failed"
                        file_upload.error_details = {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
                        file_upload.processing_message = f"Processing failed: {str(e)}"
                        await db_session.commit()
            except:
                pass
            
            if progress_tracker:
                await progress_tracker.update(0.0, f"Processing failed: {str(e)}")
            
            raise EngineError(f"Background file processing failed: {str(e)}")
    
    async def _load_data_chunked(
        self, 
        file_path: str, 
        detected_type: str, 
        progress_tracker: Optional[ProgressTracker] = None
    ) -> pd.DataFrame:
        """Load data with chunked processing for large files."""
        
        if detected_type == 'csv':
            # For large CSV files, use chunked reading
            file_size = os.path.getsize(file_path)
            if file_size > 50 * 1024 * 1024:  # > 50MB
                return await self._load_csv_chunked(file_path, progress_tracker)
        
        # For smaller files or other types, use regular loading
        return await self._load_data(file_path, detected_type)
    
    async def _load_csv_chunked(
        self, 
        file_path: str, 
        progress_tracker: Optional[ProgressTracker] = None
    ) -> pd.DataFrame:
        """Load large CSV files in chunks."""
        
        chunk_size = 10000  # Process 10k rows at a time
        chunks = []
        total_rows = 0
        
        # First pass: count total rows for progress tracking
        if progress_tracker:
            with open(file_path, 'r', encoding='utf-8') as f:
                total_rows = sum(1 for _ in f) - 1  # Subtract header
            await progress_tracker.update(15.0, f"Processing {total_rows:,} rows in chunks...")
        
        # Second pass: load data in chunks
        processed_rows = 0
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunks.append(chunk)
            processed_rows += len(chunk)
            
            if progress_tracker and total_rows > 0:
                progress = 15.0 + (processed_rows / total_rows) * 15.0  # 15-30% range
                await progress_tracker.update(
                    progress, 
                    f"Processed {processed_rows:,} of {total_rows:,} rows..."
                )
            
            # Memory management: limit number of chunks in memory
            if len(chunks) > 10:
                # Combine chunks and keep only the combined result
                combined = pd.concat(chunks, ignore_index=True)
                chunks = [combined]
        
        # Combine all chunks
        if chunks:
            return pd.concat(chunks, ignore_index=True)
        else:
            return pd.DataFrame()
    
    async def _infer_schema_async(
        self, 
        df: pd.DataFrame, 
        progress_tracker: Optional[ProgressTracker] = None
    ) -> Dict[str, Any]:
        """Async version of schema inference with progress tracking."""
        
        if progress_tracker:
            await progress_tracker.update(35.0, "Analyzing column types and patterns...")
        
        # Use regular schema inference but with progress updates
        schema_info = self._infer_schema(df)
        
        if progress_tracker:
            await progress_tracker.update(45.0, f"Schema analysis complete for {len(df.columns)} columns")
        
        return schema_info
    
    async def _generate_quality_report_async(
        self, 
        df: pd.DataFrame, 
        progress_tracker: Optional[ProgressTracker] = None
    ) -> Dict[str, Any]:
        """Async version of quality report generation with progress tracking."""
        
        if progress_tracker:
            await progress_tracker.update(75.0, "Analyzing data quality metrics...")
        
        # Use regular quality report generation but with progress updates
        quality_report = self._generate_quality_report(df)
        
        if progress_tracker:
            await progress_tracker.update(85.0, "Data quality analysis complete")
        
        return quality_report
    
    async def get_processing_status(self, upload_id: str) -> Dict[str, Any]:
        """Get current processing status for a file upload."""
        
        try:
            db_pool = await get_optimized_db_pool()
            async with db_pool.get_async_session() as db_session:
                file_upload = await db_session.get(FileUpload, upload_id)
                
                if not file_upload:
                    return {"error": "Upload not found"}
                
                status = {
                    "upload_id": upload_id,
                    "status": file_upload.processing_status,
                    "progress": file_upload.processing_progress or 0.0,
                    "message": file_upload.processing_message or "",
                    "filename": file_upload.original_filename,
                    "file_size": file_upload.file_size,
                    "created_at": file_upload.created_at.isoformat() if file_upload.created_at else None
                }
                
                # Add error details if failed
                if file_upload.processing_status == "failed" and file_upload.error_details:
                    status["error_details"] = file_upload.error_details
                
                # Add results if completed
                if file_upload.processing_status == "completed":
                    status["schema_info"] = file_upload.schema_info
                    status["preview_data"] = file_upload.preview_data
                    status["quality_report"] = file_upload.quality_report
                
                return status
                
        except Exception as e:
            return {"error": f"Failed to get processing status: {str(e)}"}
    
    async def cancel_processing(self, upload_id: str, user_id: str) -> Dict[str, Any]:
        """Cancel file processing job."""
        
        try:
            # Get job processor to cancel background job
            job_processor = await get_job_processor()
            
            # Update database status
            db_pool = await get_optimized_db_pool()
            async with db_pool.get_async_session() as db_session:
                file_upload = await db_session.get(FileUpload, upload_id)
                
                if not file_upload:
                    return {"error": "Upload not found"}
                
                if file_upload.user_id != user_id:
                    return {"error": "Unauthorized"}
                
                if file_upload.processing_status in ["completed", "failed", "cancelled"]:
                    return {"error": f"Cannot cancel job with status: {file_upload.processing_status}"}
                
                file_upload.processing_status = "cancelled"
                file_upload.processing_message = "Processing cancelled by user"
                await db_session.commit()
                
                return {
                    "upload_id": upload_id,
                    "status": "cancelled",
                    "message": "File processing cancelled successfully"
                }
                
        except Exception as e:
            return {"error": f"Failed to cancel processing: {str(e)}"}
    
    def _parse_sql_schema(self, sql_content: str) -> Dict[str, Any]:
        """Parse SQL content to extract schema information."""
        
        schema_info = {
            "tables": {},
            "columns": {},
            "parsed_at": datetime.utcnow().isoformat()
        }
        
        # Simple SQL parsing (this could be enhanced with a proper SQL parser)
        lines = sql_content.upper().split('\n')
        current_table = None
        
        for line in lines:
            line = line.strip()
            
            # Look for CREATE TABLE statements
            if line.startswith('CREATE TABLE'):
                table_name = line.split()[2].replace('(', '').strip()
                current_table = table_name
                schema_info["tables"][table_name] = {"columns": []}
            
            # Look for column definitions
            elif current_table and line and not line.startswith('--'):
                # Simple column parsing
                if '(' in line and ')' in line:
                    continue  # Skip constraint lines
                
                parts = line.split()
                if len(parts) >= 2:
                    col_name = parts[0].replace(',', '').strip()
                    col_type = parts[1].replace(',', '').strip()
                    
                    if col_name and col_type:
                        schema_info["tables"][current_table]["columns"].append({
                            "name": col_name,
                            "type": col_type
                        })
                        schema_info["columns"][col_name] = col_type
        
        return schema_info
    
    async def get_processing_status(self, upload_id: str, db_session: Session) -> FileProcessingStatus:
        """Get processing status for an upload."""
        
        file_upload = db_session.query(FileUpload).filter_by(upload_id=upload_id).first()
        
        if not file_upload:
            raise EngineError(f"Upload with ID {upload_id} not found")
        
        return FileProcessingStatus(
            upload_id=upload_id,
            status=file_upload.processing_status,
            progress=file_upload.processing_progress,
            message=file_upload.processing_message,
            error_details=file_upload.error_details
        )
    
    async def create_dataset_from_upload(
        self, 
        upload_id: str, 
        dataset_name: str,
        dataset_description: Optional[str],
        db_session: Session
    ) -> str:
        """Create a dataset from a processed file upload."""
        
        file_upload = db_session.query(FileUpload).filter_by(upload_id=upload_id).first()
        
        if not file_upload:
            raise EngineError(f"Upload with ID {upload_id} not found")
        
        if file_upload.processing_status != "completed":
            raise EngineError(f"Upload processing not completed. Status: {file_upload.processing_status}")
        
        # Create dataset
        dataset = Dataset(
            name=dataset_name,
            description=dataset_description,
            source_type=file_upload.detected_type,
            data_schema=file_upload.schema_info,
            dataset_metadata={
                "upload_id": upload_id,
                "original_filename": file_upload.original_filename,
                "file_size": file_upload.file_size,
                "quality_report": file_upload.quality_report,
                "created_from_upload": True
            },
            file_path=file_upload.file_path,
            row_count=file_upload.schema_info.get("total_rows", 0) if file_upload.schema_info else 0
        )
        
        db_session.add(dataset)
        db_session.commit()
        
        # Update file upload with dataset reference
        file_upload.dataset_id = dataset.id
        db_session.commit()
        
        return str(dataset.id)
    
    # Abstract method implementations
    async def initialize(self) -> None:
        """Initialize the file processor engine."""
        self.status = "ready"
        
    async def process(self, input_data: Any, parameters: Dict[str, Any] = None) -> Any:
        """Process input data (not used for file processor)."""
        raise NotImplementedError("Use process_upload method instead")
        
    async def cleanup(self) -> None:
        """Clean up resources."""
        pass
        
    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "engine_id": self.engine_id,
            "name": self.name,
            "status": self.status,
            "healthy": True,
            "supported_types": list(self.supported_types.keys()),
            "max_file_size": self.max_file_size,
            "max_preview_rows": self.max_preview_rows
        }
    
    async def get_processing_status(self, upload_id: str, db_session: Session) -> Dict[str, Any]:
        """Get processing status for an uploaded file."""
        
        try:
            from ..models.database import FileUpload
            
            file_upload = db_session.query(FileUpload).filter_by(upload_id=upload_id).first()
            
            if not file_upload:
                raise EngineError(f"Upload with ID {upload_id} not found")
            
            return {
                "upload_id": upload_id,
                "status": file_upload.processing_status,
                "progress": file_upload.processing_progress or 0.0,
                "message": file_upload.processing_message or "",
                "error_details": file_upload.error_details,
                "created_at": file_upload.created_at,
                "updated_at": file_upload.updated_at
            }
            
        except Exception as e:
            raise EngineError(f"Failed to get processing status: {str(e)}")
    
    async def get_file_preview(self, upload_id: str, db_session: Session) -> Optional[Dict[str, Any]]:
        """Get file preview data."""
        
        try:
            from ..models.database import FileUpload
            
            file_upload = db_session.query(FileUpload).filter_by(upload_id=upload_id).first()
            
            if not file_upload:
                raise EngineError(f"Upload with ID {upload_id} not found")
            
            if file_upload.processing_status != "completed":
                raise EngineError(f"File processing not completed. Status: {file_upload.processing_status}")
            
            return file_upload.preview_data
            
        except Exception as e:
            raise EngineError(f"Failed to get file preview: {str(e)}")
    
    async def get_quality_report(self, upload_id: str, db_session: Session) -> Optional[Dict[str, Any]]:
        """Get file quality report."""
        
        try:
            from ..models.database import FileUpload
            
            file_upload = db_session.query(FileUpload).filter_by(upload_id=upload_id).first()
            
            if not file_upload:
                raise EngineError(f"Upload with ID {upload_id} not found")
            
            if file_upload.processing_status != "completed":
                raise EngineError(f"File processing not completed. Status: {file_upload.processing_status}")
            
            return file_upload.quality_report
            
        except Exception as e:
            raise EngineError(f"Failed to get quality report: {str(e)}")
    
    async def list_user_uploads(
        self, 
        user_id: str, 
        db_session: Session,
        skip: int = 0,
        limit: int = 100,
        status_filter: Optional[str] = None,
        search_term: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List uploads for a user with filtering and pagination."""
        
        try:
            from ..models.database import FileUpload
            
            query = db_session.query(FileUpload).filter_by(user_id=user_id)
            
            if status_filter:
                query = query.filter_by(processing_status=status_filter)
            
            if search_term:
                query = query.filter(FileUpload.original_filename.ilike(f"%{search_term}%"))
            
            uploads = query.order_by(FileUpload.created_at.desc()).offset(skip).limit(limit).all()
            
            return [
                {
                    "id": upload.upload_id,
                    "filename": upload.original_filename,
                    "size": upload.file_size,
                    "type": upload.content_type,
                    "status": upload.processing_status,
                    "progress": upload.processing_progress or 0,
                    "upload_time": upload.created_at,
                    "processing_time": getattr(upload, 'processing_time', None),
                    "quality_score": self._extract_quality_score(upload.quality_report),
                    "preview_available": upload.preview_data is not None,
                    "error_message": upload.error_details.get("error") if upload.error_details else None
                }
                for upload in uploads
            ]
            
        except Exception as e:
            raise EngineError(f"Failed to list uploads: {str(e)}")
    
    def _extract_quality_score(self, quality_report: Optional[Dict[str, Any]]) -> Optional[float]:
        """Extract quality score from quality report."""
        if not quality_report:
            return None
        return quality_report.get("quality_score")
    
    async def validate_file_format(self, file: UploadFile) -> Dict[str, Any]:
        """Validate file format and return detailed validation results."""
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "file_info": {
                "name": file.filename,
                "size": file.size,
                "type": file.content_type,
                "extension": Path(file.filename).suffix.lower() if file.filename else ""
            }
        }
        
        try:
            # Run all validation checks
            await self._validate_file_optimized(file)
            
        except EngineError as e:
            validation_result["valid"] = False
            validation_result["errors"].append(str(e))
        
        # Additional warnings
        if file.size < 1024:  # Less than 1KB
            validation_result["warnings"].append("File is very small and may be empty")
        
        if file.size > 50 * 1024 * 1024:  # Larger than 50MB
            validation_result["warnings"].append("Large file may take longer to process")
        
        return validation_result
    
    async def get_supported_formats(self) -> Dict[str, Any]:
        """Get list of supported file formats with details."""
        
        return {
            "formats": [
                {
                    "extension": ".csv",
                    "mime_types": ["text/csv"],
                    "description": "Comma-separated values file",
                    "features": ["Auto-delimiter detection", "Encoding detection", "Schema inference"]
                },
                {
                    "extension": ".xlsx",
                    "mime_types": ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"],
                    "description": "Microsoft Excel file (modern)",
                    "features": ["Multiple sheets support", "Schema inference", "Data validation"]
                },
                {
                    "extension": ".xls",
                    "mime_types": ["application/vnd.ms-excel"],
                    "description": "Microsoft Excel file (legacy)",
                    "features": ["Schema inference", "Data validation"]
                },
                {
                    "extension": ".json",
                    "mime_types": ["application/json"],
                    "description": "JSON data file",
                    "features": ["Nested structure support", "JSON Lines support", "Schema inference"]
                },
                {
                    "extension": ".sql",
                    "mime_types": ["text/plain", "application/sql"],
                    "description": "SQL script file",
                    "features": ["Schema extraction", "DDL parsing", "Table structure analysis"]
                }
            ],
            "max_file_size": self.max_file_size,
            "max_file_size_mb": self.max_file_size // (1024 * 1024),
            "chunk_size": self.chunk_size,
            "supported_encodings": ["utf-8", "latin-1", "cp1252"],
            "features": {
                "auto_detection": True,
                "schema_inference": True,
                "quality_analysis": True,
                "preview_generation": True,
                "background_processing": True,
                "progress_tracking": True
            }
        }

class FileProcessor:
    """Main file processing engine."""
    
    def __init__(self):
        self.supported_formats = {
            '.csv': self._process_csv,
            '.xlsx': self._process_excel,
            '.xls': self._process_excel,
            '.json': self._process_json,
            '.txt': self._process_text,
            '.parquet': self._process_parquet
        }
    
    async def process_file(self, file_path: str, file_type: str = None) -> Dict[str, Any]:
        """Process a file and return analysis results."""
        try:
            # Determine file type
            if not file_type:
                file_type = Path(file_path).suffix.lower()
            
            if file_type not in self.supported_formats:
                raise EngineError(f"Unsupported file type: {file_type}")
            
            # Process the file
            processor = self.supported_formats[file_type]
            result = await processor(file_path)
            
            return {
                "status": "success",
                "file_path": file_path,
                "file_type": file_type,
                "processed_at": datetime.utcnow().isoformat(),
                "result": result
            }
            
        except Exception as e:
            raise EngineError(f"File processing failed: {str(e)}")
    
    async def _process_csv(self, file_path: str) -> Dict[str, Any]:
        """Process CSV file."""
        df = pd.read_csv(file_path)
        return self._analyze_dataframe(df)
    
    async def _process_excel(self, file_path: str) -> Dict[str, Any]:
        """Process Excel file."""
        df = pd.read_excel(file_path)
        return self._analyze_dataframe(df)
    
    async def _process_json(self, file_path: str) -> Dict[str, Any]:
        """Process JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            df = pd.DataFrame(data)
            return self._analyze_dataframe(df)
        else:
            return {"data_type": "json", "structure": str(type(data)), "size": len(str(data))}
    
    async def _process_text(self, file_path: str) -> Dict[str, Any]:
        """Process text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "data_type": "text",
            "character_count": len(content),
            "line_count": content.count('\n') + 1,
            "word_count": len(content.split())
        }
    
    async def _process_parquet(self, file_path: str) -> Dict[str, Any]:
        """Process Parquet file."""
        df = pd.read_parquet(file_path)
        return self._analyze_dataframe(df)
    
    def _analyze_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze a pandas DataFrame."""
        analysis = {
            "data_type": "tabular",
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "sample_data": df.head(5).to_dict('records') if len(df) > 0 else []
        }
        
        # Add basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis["numeric_summary"] = df[numeric_cols].describe().to_dict()
        
        return analysis