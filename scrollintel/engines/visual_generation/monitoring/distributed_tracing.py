"""
Distributed tracing system for visual generation workflows.
Provides end-to-end tracing across multiple services and workers.
"""

import asyncio
import logging
import uuid
import time
import json
from typing import Dict, List, Optional, Any, AsyncContextManager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from contextlib import asynccontextmanager
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)


class SpanKind(Enum):
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"
    INTERNAL = "internal"


class SpanStatus(Enum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class SpanContext:
    """Context information for a trace span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "baggage": self.baggage
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpanContext':
        """Create from dictionary."""
        return cls(
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            parent_span_id=data.get("parent_span_id"),
            baggage=data.get("baggage", {})
        )


@dataclass
class Span:
    """Represents a single span in a distributed trace."""
    context: SpanContext
    operation_name: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    status: SpanStatus = SpanStatus.OK
    kind: SpanKind = SpanKind.INTERNAL
    service_name: str = "visual_generation"
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    
    def set_tag(self, key: str, value: Any):
        """Set a tag on the span."""
        self.tags[key] = value
    
    def set_baggage(self, key: str, value: str):
        """Set baggage item that propagates to child spans."""
        self.context.baggage[key] = value
    
    def log(self, message: str, level: str = "info", **kwargs):
        """Add a log entry to the span."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            **kwargs
        }
        self.logs.append(log_entry)
    
    def set_error(self, error: Exception):
        """Mark span as error and record error details."""
        self.status = SpanStatus.ERROR
        self.error = str(error)
        self.set_tag("error", True)
        self.set_tag("error.type", type(error).__name__)
        self.set_tag("error.message", str(error))
    
    def finish(self):
        """Finish the span and calculate duration."""
        if self.end_time is None:
            self.end_time = datetime.now()
            self.duration = (self.end_time - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for export."""
        return {
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "status": self.status.value,
            "kind": self.kind.value,
            "service_name": self.service_name,
            "tags": self.tags,
            "logs": self.logs,
            "error": self.error
        }


class TraceCollector:
    """Collects and manages distributed traces."""
    
    def __init__(self, max_traces: int = 10000):
        self.max_traces = max_traces
        self.traces: Dict[str, List[Span]] = defaultdict(list)
        self.active_spans: Dict[str, Span] = {}
        self.trace_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Export configuration
        self.export_endpoints: List[str] = []
        self.export_interval = 30  # seconds
        self.batch_size = 100
        
        # Background tasks
        self._export_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start the trace collector."""
        self._running = True
        self._export_task = asyncio.create_task(self._export_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Distributed trace collector started")
    
    async def stop(self):
        """Stop the trace collector."""
        self._running = False
        
        if self._export_task:
            self._export_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Wait for tasks to complete
        for task in [self._export_task, self._cleanup_task]:
            if task:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Distributed trace collector stopped")
    
    def add_export_endpoint(self, endpoint: str):
        """Add an endpoint for trace export."""
        self.export_endpoints.append(endpoint)
        logger.info(f"Added trace export endpoint: {endpoint}")
    
    def record_span(self, span: Span):
        """Record a completed span."""
        trace_id = span.context.trace_id
        self.traces[trace_id].append(span)
        
        # Update trace metadata
        if trace_id not in self.trace_metadata:
            self.trace_metadata[trace_id] = {
                "start_time": span.start_time,
                "service_names": set(),
                "operation_names": set(),
                "span_count": 0,
                "error_count": 0
            }
        
        metadata = self.trace_metadata[trace_id]
        metadata["service_names"].add(span.service_name)
        metadata["operation_names"].add(span.operation_name)
        metadata["span_count"] += 1
        
        if span.status == SpanStatus.ERROR:
            metadata["error_count"] += 1
        
        # Update end time
        if span.end_time:
            if "end_time" not in metadata or span.end_time > metadata["end_time"]:
                metadata["end_time"] = span.end_time
        
        # Clean up if we have too many traces
        if len(self.traces) > self.max_traces:
            self._cleanup_old_traces()
    
    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace."""
        return self.traces.get(trace_id, [])
    
    def get_trace_summary(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get summary information for a trace."""
        if trace_id not in self.trace_metadata:
            return None
        
        metadata = self.trace_metadata[trace_id]
        spans = self.traces[trace_id]
        
        # Calculate total duration
        if spans and "end_time" in metadata:
            total_duration = (metadata["end_time"] - metadata["start_time"]).total_seconds()
        else:
            total_duration = None
        
        return {
            "trace_id": trace_id,
            "start_time": metadata["start_time"].isoformat(),
            "end_time": metadata.get("end_time").isoformat() if metadata.get("end_time") else None,
            "total_duration": total_duration,
            "span_count": metadata["span_count"],
            "error_count": metadata["error_count"],
            "service_names": list(metadata["service_names"]),
            "operation_names": list(metadata["operation_names"]),
            "has_errors": metadata["error_count"] > 0
        }
    
    def search_traces(self, 
                     service_name: Optional[str] = None,
                     operation_name: Optional[str] = None,
                     has_errors: Optional[bool] = None,
                     min_duration: Optional[float] = None,
                     max_duration: Optional[float] = None,
                     limit: int = 100) -> List[Dict[str, Any]]:
        """Search traces based on criteria."""
        results = []
        
        for trace_id, metadata in self.trace_metadata.items():
            # Apply filters
            if service_name and service_name not in metadata["service_names"]:
                continue
            
            if operation_name and operation_name not in metadata["operation_names"]:
                continue
            
            if has_errors is not None:
                trace_has_errors = metadata["error_count"] > 0
                if has_errors != trace_has_errors:
                    continue
            
            # Duration filtering
            if min_duration is not None or max_duration is not None:
                if "end_time" not in metadata:
                    continue
                
                duration = (metadata["end_time"] - metadata["start_time"]).total_seconds()
                
                if min_duration is not None and duration < min_duration:
                    continue
                
                if max_duration is not None and duration > max_duration:
                    continue
            
            summary = self.get_trace_summary(trace_id)
            if summary:
                results.append(summary)
            
            if len(results) >= limit:
                break
        
        # Sort by start time (newest first)
        results.sort(key=lambda x: x["start_time"], reverse=True)
        return results
    
    def _cleanup_old_traces(self):
        """Clean up old traces to maintain memory limits."""
        # Sort traces by start time and keep only the most recent
        trace_items = list(self.trace_metadata.items())
        trace_items.sort(key=lambda x: x[1]["start_time"], reverse=True)
        
        # Keep only the most recent traces
        keep_count = int(self.max_traces * 0.8)  # Keep 80% of max
        traces_to_remove = trace_items[keep_count:]
        
        for trace_id, _ in traces_to_remove:
            del self.traces[trace_id]
            del self.trace_metadata[trace_id]
        
        logger.info(f"Cleaned up {len(traces_to_remove)} old traces")
    
    async def _export_loop(self):
        """Background loop for exporting traces."""
        while self._running:
            try:
                await self._export_traces()
                await asyncio.sleep(self.export_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in trace export loop: {e}")
                await asyncio.sleep(10)
    
    async def _cleanup_loop(self):
        """Background loop for cleaning up old traces."""
        while self._running:
            try:
                # Clean up traces older than 24 hours
                cutoff_time = datetime.now() - timedelta(hours=24)
                traces_to_remove = []
                
                for trace_id, metadata in self.trace_metadata.items():
                    if metadata["start_time"] < cutoff_time:
                        traces_to_remove.append(trace_id)
                
                for trace_id in traces_to_remove:
                    del self.traces[trace_id]
                    del self.trace_metadata[trace_id]
                
                if traces_to_remove:
                    logger.info(f"Cleaned up {len(traces_to_remove)} expired traces")
                
                await asyncio.sleep(3600)  # Run every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in trace cleanup loop: {e}")
                await asyncio.sleep(300)
    
    async def _export_traces(self):
        """Export traces to configured endpoints."""
        if not self.export_endpoints or not self.traces:
            return
        
        # Collect completed traces for export
        completed_traces = []
        
        for trace_id, spans in list(self.traces.items()):
            # Check if trace is complete (no active spans)
            has_active_spans = any(
                span.end_time is None for span in spans
            )
            
            if not has_active_spans and len(spans) > 0:
                # Export this trace
                trace_data = {
                    "trace_id": trace_id,
                    "spans": [span.to_dict() for span in spans],
                    "metadata": self.get_trace_summary(trace_id)
                }
                completed_traces.append(trace_data)
        
        if not completed_traces:
            return
        
        # Export in batches
        for i in range(0, len(completed_traces), self.batch_size):
            batch = completed_traces[i:i + self.batch_size]
            
            for endpoint in self.export_endpoints:
                try:
                    await self._export_batch(endpoint, batch)
                except Exception as e:
                    logger.error(f"Error exporting traces to {endpoint}: {e}")
    
    async def _export_batch(self, endpoint: str, traces: List[Dict[str, Any]]):
        """Export a batch of traces to an endpoint."""
        # Placeholder for actual HTTP export
        logger.debug(f"Exporting {len(traces)} traces to {endpoint}")
        
        # In a real implementation, this would be an HTTP POST
        # import aiohttp
        # async with aiohttp.ClientSession() as session:
        #     await session.post(endpoint, json={"traces": traces})


class DistributedTracer:
    """Main distributed tracer for visual generation system."""
    
    def __init__(self, service_name: str = "visual_generation"):
        self.service_name = service_name
        self.collector = TraceCollector()
        
        # Thread-local storage for current span context
        self._local = threading.local()
    
    async def start(self):
        """Start the tracer."""
        await self.collector.start()
        logger.info(f"Distributed tracer started for service: {self.service_name}")
    
    async def stop(self):
        """Stop the tracer."""
        await self.collector.stop()
        logger.info("Distributed tracer stopped")
    
    def start_trace(self, operation_name: str, **tags) -> SpanContext:
        """Start a new trace with a root span."""
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        
        context = SpanContext(trace_id=trace_id, span_id=span_id)
        span = Span(
            context=context,
            operation_name=operation_name,
            service_name=self.service_name,
            kind=SpanKind.SERVER
        )
        
        # Set initial tags
        for key, value in tags.items():
            span.set_tag(key, value)
        
        self._set_current_span(span)
        return context
    
    def start_span(self, operation_name: str, 
                   parent_context: Optional[SpanContext] = None,
                   kind: SpanKind = SpanKind.INTERNAL,
                   **tags) -> SpanContext:
        """Start a new span."""
        # Use provided parent context or current span context
        if parent_context is None:
            current_span = self._get_current_span()
            if current_span:
                parent_context = current_span.context
        
        # Generate new span ID
        span_id = str(uuid.uuid4())
        
        if parent_context:
            # Child span
            context = SpanContext(
                trace_id=parent_context.trace_id,
                span_id=span_id,
                parent_span_id=parent_context.span_id,
                baggage=parent_context.baggage.copy()
            )
        else:
            # Root span (new trace)
            trace_id = str(uuid.uuid4())
            context = SpanContext(trace_id=trace_id, span_id=span_id)
        
        span = Span(
            context=context,
            operation_name=operation_name,
            service_name=self.service_name,
            kind=kind
        )
        
        # Set initial tags
        for key, value in tags.items():
            span.set_tag(key, value)
        
        self._set_current_span(span)
        return context
    
    def finish_span(self, context: Optional[SpanContext] = None, 
                   status: SpanStatus = SpanStatus.OK,
                   error: Optional[Exception] = None):
        """Finish a span."""
        span = self._get_current_span()
        if not span:
            return
        
        # Set status and error if provided
        span.status = status
        if error:
            span.set_error(error)
        
        # Finish the span
        span.finish()
        
        # Record in collector
        self.collector.record_span(span)
        
        # Clear current span
        self._clear_current_span()
    
    @asynccontextmanager
    async def trace(self, operation_name: str, 
                   parent_context: Optional[SpanContext] = None,
                   kind: SpanKind = SpanKind.INTERNAL,
                   **tags) -> AsyncContextManager[Span]:
        """Context manager for tracing an operation."""
        context = self.start_span(operation_name, parent_context, kind, **tags)
        span = self._get_current_span()
        
        try:
            yield span
            self.finish_span(context, SpanStatus.OK)
        except Exception as e:
            self.finish_span(context, SpanStatus.ERROR, e)
            raise
    
    def inject_context(self, context: SpanContext) -> Dict[str, str]:
        """Inject span context into headers for propagation."""
        return {
            "X-Trace-Id": context.trace_id,
            "X-Span-Id": context.span_id,
            "X-Parent-Span-Id": context.parent_span_id or "",
            "X-Baggage": json.dumps(context.baggage)
        }
    
    def extract_context(self, headers: Dict[str, str]) -> Optional[SpanContext]:
        """Extract span context from headers."""
        trace_id = headers.get("X-Trace-Id")
        span_id = headers.get("X-Span-Id")
        
        if not trace_id or not span_id:
            return None
        
        parent_span_id = headers.get("X-Parent-Span-Id") or None
        baggage_str = headers.get("X-Baggage", "{}")
        
        try:
            baggage = json.loads(baggage_str)
        except (json.JSONDecodeError, TypeError):
            baggage = {}
        
        return SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            baggage=baggage
        )
    
    def get_current_context(self) -> Optional[SpanContext]:
        """Get the current span context."""
        span = self._get_current_span()
        return span.context if span else None
    
    def _get_current_span(self) -> Optional[Span]:
        """Get the current span from thread-local storage."""
        return getattr(self._local, 'current_span', None)
    
    def _set_current_span(self, span: Span):
        """Set the current span in thread-local storage."""
        self._local.current_span = span
    
    def _clear_current_span(self):
        """Clear the current span from thread-local storage."""
        if hasattr(self._local, 'current_span'):
            delattr(self._local, 'current_span')


# Global distributed tracer instance
distributed_tracer = DistributedTracer()