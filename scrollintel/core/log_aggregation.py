"""
ScrollIntel Log Aggregation System
Centralized logging with structured format and analysis capabilities
"""

import json
import logging
import logging.handlers
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager

# Optional aiofiles import
try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    aiofiles = None
    AIOFILES_AVAILABLE = False

# Optional elasticsearch import
try:
    from elasticsearch import AsyncElasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    AsyncElasticsearch = None
    ELASTICSEARCH_AVAILABLE = False

from ..core.config import get_settings

settings = get_settings()

@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: str
    component: str
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    agent_type: Optional[str] = None
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'component': record.name,
            'message': record.getMessage(),
            'filename': record.filename,
            'line_number': record.lineno,
            'function': record.funcName
        }
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'session_id'):
            log_entry['session_id'] = record.session_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'agent_type'):
            log_entry['agent_type'] = record.agent_type
        if hasattr(record, 'error_type'):
            log_entry['error_type'] = record.error_type
        if hasattr(record, 'metadata'):
            log_entry['metadata'] = record.metadata
            
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry)

class LogAggregator:
    """Centralized log aggregation and analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.log_buffer: List[LogEntry] = []
        self.buffer_size = 1000
        self.elasticsearch_client = None
        self.log_file_path = Path("logs/scrollintel.json")
        self.error_log_path = Path("logs/errors.json")
        self.audit_log_path = Path("logs/audit.json")
        
        # Ensure log directories exist
        self.log_file_path.parent.mkdir(exist_ok=True)
        
        # Initialize Elasticsearch if configured and available
        if (ELASTICSEARCH_AVAILABLE and 
            hasattr(settings, 'ELASTICSEARCH_URL') and 
            settings.ELASTICSEARCH_URL):
            self.elasticsearch_client = AsyncElasticsearch([settings.ELASTICSEARCH_URL])
        else:
            self.elasticsearch_client = None
    
    def setup_logging(self):
        """Configure application-wide structured logging"""
        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # JSON file handler for all logs
        json_handler = logging.handlers.RotatingFileHandler(
            self.log_file_path,
            maxBytes=100*1024*1024,  # 100MB
            backupCount=10
        )
        json_handler.setFormatter(StructuredFormatter())
        json_handler.setLevel(logging.INFO)
        root_logger.addHandler(json_handler)
        
        # Error-specific handler
        error_handler = logging.handlers.RotatingFileHandler(
            self.error_log_path,
            maxBytes=50*1024*1024,  # 50MB
            backupCount=5
        )
        error_handler.setFormatter(StructuredFormatter())
        error_handler.setLevel(logging.ERROR)
        root_logger.addHandler(error_handler)
        
        # Console handler for development
        if settings.DEBUG:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            console_handler.setLevel(logging.DEBUG)
            root_logger.addHandler(console_handler)
    
    async def log_structured(self, entry: LogEntry):
        """Log a structured entry"""
        self.log_buffer.append(entry)
        
        # Flush buffer if full
        if len(self.log_buffer) >= self.buffer_size:
            await self.flush_buffer()
    
    async def flush_buffer(self):
        """Flush log buffer to storage"""
        if not self.log_buffer:
            return
            
        try:
            # Write to file
            def datetime_converter(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj
            
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(self.log_file_path, 'a') as f:
                    for entry in self.log_buffer:
                        await f.write(json.dumps(asdict(entry), default=datetime_converter) + '\n')
            else:
                # Fallback to synchronous file writing
                with open(self.log_file_path, 'a') as f:
                    for entry in self.log_buffer:
                        f.write(json.dumps(asdict(entry), default=datetime_converter) + '\n')
            
            # Send to Elasticsearch if configured
            if self.elasticsearch_client:
                await self._send_to_elasticsearch(self.log_buffer)
            
            # Clear buffer
            self.log_buffer.clear()
            
        except Exception as e:
            self.logger.error(f"Error flushing log buffer: {e}")
    
    async def _send_to_elasticsearch(self, entries: List[LogEntry]):
        """Send log entries to Elasticsearch"""
        try:
            actions = []
            for entry in entries:
                action = {
                    "_index": f"scrollintel-logs-{datetime.utcnow().strftime('%Y-%m')}",
                    "_source": asdict(entry)
                }
                actions.append(action)
            
            if actions and ELASTICSEARCH_AVAILABLE:
                from elasticsearch.helpers import async_bulk
                await async_bulk(self.elasticsearch_client, actions)
                
        except Exception as e:
            self.logger.error(f"Error sending logs to Elasticsearch: {e}")
    
    async def search_logs(self, 
                         query: str = "*",
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         level: Optional[str] = None,
                         component: Optional[str] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """Search logs with filters"""
        if not self.elasticsearch_client:
            return await self._search_file_logs(query, start_time, end_time, level, component, limit)
        
        try:
            # Build Elasticsearch query
            es_query = {
                "query": {
                    "bool": {
                        "must": []
                    }
                },
                "sort": [{"timestamp": {"order": "desc"}}],
                "size": limit
            }
            
            # Add query string
            if query != "*":
                es_query["query"]["bool"]["must"].append({
                    "query_string": {"query": query}
                })
            
            # Add time range
            if start_time or end_time:
                time_range = {}
                if start_time:
                    time_range["gte"] = start_time.isoformat()
                if end_time:
                    time_range["lte"] = end_time.isoformat()
                
                es_query["query"]["bool"]["must"].append({
                    "range": {"timestamp": time_range}
                })
            
            # Add level filter
            if level:
                es_query["query"]["bool"]["must"].append({
                    "term": {"level": level}
                })
            
            # Add component filter
            if component:
                es_query["query"]["bool"]["must"].append({
                    "term": {"component": component}
                })
            
            # Execute search
            response = await self.elasticsearch_client.search(
                index="scrollintel-logs-*",
                body=es_query
            )
            
            return [hit["_source"] for hit in response["hits"]["hits"]]
            
        except Exception as e:
            self.logger.error(f"Error searching Elasticsearch logs: {e}")
            return []
    
    async def _search_file_logs(self, 
                               query: str,
                               start_time: Optional[datetime],
                               end_time: Optional[datetime],
                               level: Optional[str],
                               component: Optional[str],
                               limit: int) -> List[Dict[str, Any]]:
        """Search logs in files (fallback when Elasticsearch not available)"""
        results = []
        
        try:
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(self.log_file_path, 'r') as f:
                    async for line in f:
                        try:
                            log_entry = json.loads(line.strip())
                            
                            # Apply filters
                            if level and log_entry.get('level') != level:
                                continue
                            if component and log_entry.get('component') != component:
                                continue
                            
                            # Time range filter
                            if start_time or end_time:
                                entry_time = datetime.fromisoformat(log_entry['timestamp'].replace('Z', '+00:00'))
                                if start_time and entry_time < start_time:
                                    continue
                                if end_time and entry_time > end_time:
                                    continue
                            
                            # Simple text search
                            if query != "*" and query.lower() not in log_entry.get('message', '').lower():
                                continue
                            
                            results.append(log_entry)
                            
                            if len(results) >= limit:
                                break
                                
                        except json.JSONDecodeError:
                            continue
            else:
                # Fallback to synchronous file reading
                with open(self.log_file_path, 'r') as f:
                    for line in f:
                        try:
                            log_entry = json.loads(line.strip())
                            
                            # Apply filters
                            if level and log_entry.get('level') != level:
                                continue
                            if component and log_entry.get('component') != component:
                                continue
                            
                            # Time range filter
                            if start_time or end_time:
                                entry_time = datetime.fromisoformat(log_entry['timestamp'].replace('Z', '+00:00'))
                                if start_time and entry_time < start_time:
                                    continue
                                if end_time and entry_time > end_time:
                                    continue
                            
                            # Simple text search
                            if query != "*" and query.lower() not in log_entry.get('message', '').lower():
                                continue
                            
                            results.append(log_entry)
                            
                            if len(results) >= limit:
                                break
                                
                        except json.JSONDecodeError:
                            continue
                        
        except FileNotFoundError:
            self.logger.warning(f"Log file not found: {self.log_file_path}")
        except Exception as e:
            self.logger.error(f"Error searching file logs: {e}")
        
        return results
    
    async def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for the last N hours"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        error_logs = await self.search_logs(
            level="ERROR",
            start_time=start_time,
            end_time=end_time,
            limit=1000
        )
        
        # Analyze errors
        error_types = {}
        components = {}
        
        for log in error_logs:
            error_type = log.get('error_type', 'Unknown')
            component = log.get('component', 'Unknown')
            
            error_types[error_type] = error_types.get(error_type, 0) + 1
            components[component] = components.get(component, 0) + 1
        
        return {
            "total_errors": len(error_logs),
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "error_types": dict(sorted(error_types.items(), key=lambda x: x[1], reverse=True)),
            "components": dict(sorted(components.items(), key=lambda x: x[1], reverse=True)),
            "recent_errors": error_logs[:10]  # Last 10 errors
        }
    
    async def log_audit_event(self, 
                             user_id: str,
                             action: str,
                             resource: str,
                             details: Optional[Dict[str, Any]] = None):
        """Log audit event"""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "details": details or {},
            "ip_address": None,  # Would be filled by middleware
            "user_agent": None   # Would be filled by middleware
        }
        
        try:
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(self.audit_log_path, 'a') as f:
                    await f.write(json.dumps(audit_entry) + '\n')
            else:
                # Fallback to synchronous file writing
                with open(self.audit_log_path, 'a') as f:
                    f.write(json.dumps(audit_entry) + '\n')
        except Exception as e:
            self.logger.error(f"Error writing audit log: {e}")
    
    async def cleanup_old_logs(self, days: int = 30):
        """Clean up logs older than specified days"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Clean up Elasticsearch indices
        if self.elasticsearch_client:
            try:
                # Delete old indices
                index_pattern = f"scrollintel-logs-{cutoff_date.strftime('%Y-%m')}"
                await self.elasticsearch_client.indices.delete(
                    index=index_pattern,
                    ignore=[404]
                )
            except Exception as e:
                self.logger.error(f"Error cleaning up Elasticsearch indices: {e}")
        
        self.logger.info(f"Log cleanup completed for logs older than {days} days")

# Global log aggregator instance
log_aggregator = LogAggregator()

# Context manager for request logging
@asynccontextmanager
async def log_request_context(request_id: str, user_id: Optional[str] = None):
    """Context manager for request-scoped logging"""
    # Add request context to all logs within this context
    old_factory = logging.getLogRecordFactory()
    
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.request_id = request_id
        if user_id:
            record.user_id = user_id
        return record
    
    logging.setLogRecordFactory(record_factory)
    
    try:
        yield
    finally:
        logging.setLogRecordFactory(old_factory)

# Utility functions for structured logging
def log_agent_activity(agent_type: str, action: str, duration: float, success: bool, metadata: Optional[Dict] = None):
    """Log agent activity with structured format"""
    logger = logging.getLogger(f"scrollintel.agents.{agent_type}")
    
    extra = {
        'agent_type': agent_type,
        'action': action,
        'duration': duration,
        'success': success,
        'metadata': metadata or {}
    }
    
    if success:
        logger.info(f"Agent {agent_type} completed {action} in {duration:.2f}s", extra=extra)
    else:
        logger.error(f"Agent {agent_type} failed {action} after {duration:.2f}s", extra=extra)

def log_user_action(user_id: str, action: str, resource: str, metadata: Optional[Dict] = None):
    """Log user action for audit trail"""
    logger = logging.getLogger("scrollintel.audit")
    
    extra = {
        'user_id': user_id,
        'action': action,
        'resource': resource,
        'metadata': metadata or {}
    }
    
    logger.info(f"User {user_id} performed {action} on {resource}", extra=extra)

def log_error_with_context(error: Exception, component: str, context: Optional[Dict] = None):
    """Log error with full context"""
    logger = logging.getLogger(f"scrollintel.{component}")
    
    extra = {
        'error_type': type(error).__name__,
        'component': component,
        'metadata': context or {}
    }
    
    logger.error(f"Error in {component}: {str(error)}", extra=extra, exc_info=True)