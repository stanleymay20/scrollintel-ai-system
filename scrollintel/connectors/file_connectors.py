"""
File system connectors for CSV, JSON, Parquet, and Excel files.
"""
import asyncio
import logging
import os
import json
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

# Optional file processing dependencies
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import aiofiles
except ImportError:
    aiofiles = None

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

try:
    import openpyxl
except ImportError:
    openpyxl = None

from .base_connector import BaseConnector

logger = logging.getLogger(__name__)

class FileSystemConnector(BaseConnector):
    """Connector for various file formats."""
    
    def __init__(self):
        self.supported_formats = {
            "csv": self._handle_csv,
            "json": self._handle_json,
            "parquet": self._handle_parquet,
            "excel": self._handle_excel
        }
    
    async def test_connection(self, connection_config: Dict[str, Any], 
                            auth_config: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Test file system connection."""
        try:
            file_path = connection_config.get("file_path")
            file_format = connection_config.get("file_format")
            
            if not file_path:
                return False, "file_path is required", {}
            
            if not file_format or file_format not in self.supported_formats:
                return False, f"Unsupported file format: {file_format}", {}
            
            # Check if file exists and is readable
            if not os.path.exists(file_path):
                return False, f"File not found: {file_path}", {}
            
            if not os.access(file_path, os.R_OK):
                return False, f"File not readable: {file_path}", {}
            
            # Get file info
            file_stat = os.stat(file_path)
            file_info = {
                "file_size": file_stat.st_size,
                "modified_time": file_stat.st_mtime,
                "file_format": file_format,
                "exists": True
            }
            
            # Try to read a small sample to validate format
            handler = self.supported_formats[file_format]
            sample_success, sample_error, sample_info = await handler("test", connection_config, auth_config)
            
            if sample_success:
                file_info.update(sample_info)
                return True, None, file_info
            else:
                return False, sample_error, file_info
                
        except Exception as e:
            logger.error(f"File system connection test failed: {str(e)}")
            return False, str(e), {}
    
    async def create_connection(self, connection_config: Dict[str, Any], 
                              auth_config: Dict[str, Any]) -> Any:
        """Create file system connection."""
        try:
            file_path = connection_config.get("file_path")
            file_format = connection_config.get("file_format")
            
            return {
                "file_path": file_path,
                "file_format": file_format,
                "config": connection_config
            }
            
        except Exception as e:
            logger.error(f"Failed to create file system connection: {str(e)}")
            raise
    
    async def discover_schema(self, connection: Any) -> List[Dict[str, Any]]:
        """Discover file schema."""
        try:
            file_path = connection["file_path"]
            file_format = connection["file_format"]
            config = connection["config"]
            
            handler = self.supported_formats[file_format]
            success, schema_info, details = await handler("schema", config, {})
            
            if success:
                return [{
                    "schema_name": "file_system",
                    "table_name": Path(file_path).stem,
                    "columns": schema_info
                }]
            else:
                raise Exception(f"Schema discovery failed: {schema_info}")
                
        except Exception as e:
            logger.error(f"File schema discovery failed: {str(e)}")
            raise
    
    async def read_data(self, connection: Any, query_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Read data from file."""
        try:
            file_format = connection["file_format"]
            config = connection["config"]
            
            # Merge query config with connection config
            read_config = {**config, **query_config}
            
            handler = self.supported_formats[file_format]
            success, data, details = await handler("read", read_config, {})
            
            if success:
                return data
            else:
                raise Exception(f"Data reading failed: {data}")
                
        except Exception as e:
            logger.error(f"File data reading failed: {str(e)}")
            raise
    
    async def _handle_csv(self, operation: str, connection_config: Dict[str, Any], 
                         auth_config: Dict[str, Any]) -> Tuple[bool, Any, Dict[str, Any]]:
        """Handle CSV file operations."""
        try:
            if pd is None:
                return False, "pandas library not installed", {}
                
            file_path = connection_config.get("file_path")
            delimiter = connection_config.get("delimiter", ",")
            encoding = connection_config.get("encoding", "utf-8")
            has_header = connection_config.get("has_header", True)
            
            if operation == "test":
                # Read first few rows to validate format
                try:
                    df = pd.read_csv(
                        file_path, 
                        delimiter=delimiter, 
                        encoding=encoding,
                        nrows=5,
                        header=0 if has_header else None
                    )
                    
                    return True, None, {
                        "row_count_sample": len(df),
                        "column_count": len(df.columns),
                        "columns": list(df.columns)
                    }
                except Exception as e:
                    return False, f"CSV validation failed: {str(e)}", {}
            
            elif operation == "schema":
                df = pd.read_csv(
                    file_path, 
                    delimiter=delimiter, 
                    encoding=encoding,
                    nrows=100,  # Sample for schema detection
                    header=0 if has_header else None
                )
                
                columns = []
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    columns.append({
                        "name": str(col),
                        "type": self._pandas_to_standard_type(dtype),
                        "nullable": bool(df[col].isnull().any()),  # Convert numpy bool to Python bool
                        "primary_key": False
                    })
                
                return True, columns, {"total_columns": len(columns)}
            
            elif operation == "read":
                limit = connection_config.get("limit", 10000)
                skip_rows = connection_config.get("skip_rows", 0)
                
                df = pd.read_csv(
                    file_path,
                    delimiter=delimiter,
                    encoding=encoding,
                    header=0 if has_header else None,
                    skiprows=skip_rows,
                    nrows=limit
                )
                
                # Convert to list of dictionaries
                data = df.to_dict('records')
                return True, data, {"rows_read": len(data)}
            
        except Exception as e:
            return False, str(e), {}
    
    async def _handle_json(self, operation: str, connection_config: Dict[str, Any], 
                          auth_config: Dict[str, Any]) -> Tuple[bool, Any, Dict[str, Any]]:
        """Handle JSON file operations."""
        try:
            if aiofiles is None:
                return False, "aiofiles library not installed", {}
                
            file_path = connection_config.get("file_path")
            encoding = connection_config.get("encoding", "utf-8")
            json_path = connection_config.get("json_path")  # JSONPath for nested data
            
            if operation == "test":
                try:
                    async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                        content = await f.read()
                        data = json.loads(content)
                    
                    data_type = "object" if isinstance(data, dict) else "array" if isinstance(data, list) else "primitive"
                    
                    return True, None, {
                        "data_type": data_type,
                        "size_bytes": len(content),
                        "valid_json": True
                    }
                except Exception as e:
                    return False, f"JSON validation failed: {str(e)}", {}
            
            elif operation == "schema":
                async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                    content = await f.read()
                    data = json.loads(content)
                
                # Extract schema from JSON structure
                columns = self._extract_json_schema(data, json_path)
                return True, columns, {"schema_extracted": True}
            
            elif operation == "read":
                limit = connection_config.get("limit", 10000)
                
                async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                    content = await f.read()
                    data = json.loads(content)
                
                # Apply JSONPath if specified
                if json_path:
                    data = self._apply_json_path(data, json_path)
                
                # Ensure data is a list
                if not isinstance(data, list):
                    data = [data]
                
                # Apply limit
                if limit:
                    data = data[:limit]
                
                return True, data, {"rows_read": len(data)}
            
        except Exception as e:
            return False, str(e), {}
    
    async def _handle_parquet(self, operation: str, connection_config: Dict[str, Any], 
                            auth_config: Dict[str, Any]) -> Tuple[bool, Any, Dict[str, Any]]:
        """Handle Parquet file operations."""
        try:
            if pq is None or pd is None:
                return False, "pyarrow and pandas libraries not installed", {}
                
            file_path = connection_config.get("file_path")
            
            if operation == "test":
                try:
                    # Read metadata only
                    parquet_file = pq.ParquetFile(file_path)
                    metadata = parquet_file.metadata
                    
                    return True, None, {
                        "num_rows": metadata.num_rows,
                        "num_columns": metadata.num_columns,
                        "file_size": metadata.serialized_size
                    }
                except Exception as e:
                    return False, f"Parquet validation failed: {str(e)}", {}
            
            elif operation == "schema":
                parquet_file = pq.ParquetFile(file_path)
                schema = parquet_file.schema
                
                columns = []
                for i in range(len(schema)):
                    field = schema.field(i)
                    columns.append({
                        "name": field.name,
                        "type": str(field.type),
                        "nullable": bool(field.nullable),  # Ensure Python bool
                        "primary_key": False
                    })
                
                return True, columns, {"schema_extracted": True}
            
            elif operation == "read":
                limit = connection_config.get("limit", 10000)
                
                # Read parquet file
                df = pd.read_parquet(file_path)
                
                # Apply limit
                if limit:
                    df = df.head(limit)
                
                data = df.to_dict('records')
                return True, data, {"rows_read": len(data)}
            
        except Exception as e:
            return False, str(e), {}
    
    async def _handle_excel(self, operation: str, connection_config: Dict[str, Any], 
                          auth_config: Dict[str, Any]) -> Tuple[bool, Any, Dict[str, Any]]:
        """Handle Excel file operations."""
        try:
            if openpyxl is None or pd is None:
                return False, "openpyxl and pandas libraries not installed", {}
                
            file_path = connection_config.get("file_path")
            sheet_name = connection_config.get("sheet_name", 0)  # Default to first sheet
            has_header = connection_config.get("has_header", True)
            
            if operation == "test":
                try:
                    # Load workbook to check sheets
                    workbook = openpyxl.load_workbook(file_path, read_only=True)
                    sheet_names = workbook.sheetnames
                    workbook.close()
                    
                    return True, None, {
                        "sheet_names": sheet_names,
                        "total_sheets": len(sheet_names)
                    }
                except Exception as e:
                    return False, f"Excel validation failed: {str(e)}", {}
            
            elif operation == "schema":
                df = pd.read_excel(
                    file_path,
                    sheet_name=sheet_name,
                    header=0 if has_header else None,
                    nrows=100  # Sample for schema
                )
                
                columns = []
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    columns.append({
                        "name": str(col),
                        "type": self._pandas_to_standard_type(dtype),
                        "nullable": bool(df[col].isnull().any()),  # Convert numpy bool to Python bool
                        "primary_key": False
                    })
                
                return True, columns, {"schema_extracted": True}
            
            elif operation == "read":
                limit = connection_config.get("limit", 10000)
                skip_rows = connection_config.get("skip_rows", 0)
                
                df = pd.read_excel(
                    file_path,
                    sheet_name=sheet_name,
                    header=0 if has_header else None,
                    skiprows=skip_rows,
                    nrows=limit
                )
                
                data = df.to_dict('records')
                return True, data, {"rows_read": len(data)}
            
        except Exception as e:
            return False, str(e), {}
    
    def _pandas_to_standard_type(self, pandas_type: str) -> str:
        """Convert pandas dtype to standard type."""
        type_mapping = {
            "int64": "integer",
            "float64": "float",
            "object": "string",
            "bool": "boolean",
            "datetime64[ns]": "datetime",
            "category": "string"
        }
        
        return type_mapping.get(pandas_type, "string")
    
    def _extract_json_schema(self, data: Any, json_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract schema from JSON data structure."""
        if json_path:
            data = self._apply_json_path(data, json_path)
        
        columns = []
        
        if isinstance(data, list) and data:
            # Use first item as schema template
            sample_item = data[0]
            if isinstance(sample_item, dict):
                for key, value in sample_item.items():
                    columns.append({
                        "name": key,
                        "type": self._infer_json_type(value),
                        "nullable": True,
                        "primary_key": key == "id"
                    })
        elif isinstance(data, dict):
            for key, value in data.items():
                columns.append({
                    "name": key,
                    "type": self._infer_json_type(value),
                    "nullable": True,
                    "primary_key": key == "id"
                })
        
        return columns
    
    def _infer_json_type(self, value: Any) -> str:
        """Infer type from JSON value."""
        if isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "object"
        else:
            return "string"
    
    def _apply_json_path(self, data: Any, json_path: str) -> Any:
        """Apply simple JSONPath to extract nested data."""
        # Simple implementation - in production, use a proper JSONPath library
        path_parts = json_path.strip("$.").split(".")
        
        current = data
        for part in path_parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, list) and part.isdigit():
                index = int(part)
                if 0 <= index < len(current):
                    current = current[index]
                else:
                    return None
            else:
                return None
        
        return current
    
    def validate_config(self, connection_config: Dict[str, Any]) -> List[str]:
        """Validate file system connection configuration."""
        errors = []
        
        file_path = connection_config.get("file_path")
        if not file_path:
            errors.append("Missing required field: file_path")
        
        file_format = connection_config.get("file_format")
        if not file_format:
            errors.append("Missing required field: file_format")
        elif file_format not in self.supported_formats:
            errors.append(f"Unsupported file format: {file_format}")
        
        return errors