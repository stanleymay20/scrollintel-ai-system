"""
Progressive Content Loading System for ScrollIntel.
Loads content in stages with partial results and user feedback,
ensuring users always see progress and can interact with available data.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json
import uuid

from .intelligent_fallback_manager import (
    intelligent_fallback_manager, ContentContext, ContentType, FallbackContent
)

logger = logging.getLogger(__name__)


class LoadingStage(Enum):
    """Stages of progressive content loading."""
    INITIALIZING = "initializing"
    LOADING_METADATA = "loading_metadata"
    LOADING_PARTIAL = "loading_partial"
    LOADING_FULL = "loading_full"
    ENHANCING = "enhancing"
    COMPLETE = "complete"
    FAILED = "failed"


class ContentPriority(Enum):
    """Priority levels for content loading."""
    CRITICAL = "critical"    # Must load first (core functionality)
    HIGH = "high"           # Important for user experience
    MEDIUM = "medium"       # Nice to have
    LOW = "low"            # Can be deferred
    BACKGROUND = "background"  # Load when resources available


@dataclass
class LoadingProgress:
    """Progress information for content loading."""
    stage: LoadingStage
    progress_percentage: float  # 0.0 to 100.0
    estimated_time_remaining: Optional[float] = None  # seconds
    current_operation: str = ""
    partial_results: Optional[Any] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContentChunk:
    """A chunk of content that can be loaded progressively."""
    chunk_id: str
    priority: ContentPriority
    content_type: ContentType
    loader_function: Callable
    dependencies: List[str] = field(default_factory=list)
    estimated_load_time: float = 1.0  # seconds
    cache_key: Optional[str] = None
    fallback_content: Optional[Any] = None
    loaded: bool = False
    loading: bool = False
    error: Optional[Exception] = None
    result: Optional[Any] = None
    load_start_time: Optional[float] = None
    load_end_time: Optional[float] = None


@dataclass
class ProgressiveLoadRequest:
    """Request for progressive content loading."""
    request_id: str
    user_id: Optional[str]
    content_chunks: List[ContentChunk]
    context: ContentContext
    progress_callback: Optional[Callable[[LoadingProgress], None]] = None
    timeout_seconds: float = 60.0
    allow_partial_results: bool = True
    fallback_on_timeout: bool = True


class ProgressiveContentLoader:
    """Main progressive content loading system."""
    
    def __init__(self):
        self.active_requests: Dict[str, ProgressiveLoadRequest] = {}
        self.loading_strategies: Dict[ContentType, Callable] = {}
        self.chunk_cache: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.max_concurrent_chunks = 5
        self.chunk_timeout_seconds = 30.0
        self.progress_update_interval = 0.5  # seconds
        
        # Initialize loading strategies
        self._initialize_loading_strategies()
    
    def _initialize_loading_strategies(self):
        """Initialize content-specific loading strategies."""
        self.loading_strategies = {
            ContentType.CHART: self._load_chart_progressively,
            ContentType.TABLE: self._load_table_progressively,
            ContentType.DASHBOARD: self._load_dashboard_progressively,
            ContentType.REPORT: self._load_report_progressively,
            ContentType.ANALYSIS: self._load_analysis_progressively,
        }
    
    async def load_content_progressively(self, request: ProgressiveLoadRequest) -> AsyncGenerator[LoadingProgress, None]:
        """Load content progressively with real-time progress updates."""
        self.active_requests[request.request_id] = request
        
        try:
            # Initialize progress
            progress = LoadingProgress(
                stage=LoadingStage.INITIALIZING,
                progress_percentage=0.0,
                current_operation="Preparing to load content..."
            )
            yield progress
            
            # Sort chunks by priority and dependencies
            sorted_chunks = self._sort_chunks_by_priority(request.content_chunks)
            
            # Load metadata first
            progress.stage = LoadingStage.LOADING_METADATA
            progress.current_operation = "Loading content metadata..."
            progress.progress_percentage = 5.0
            yield progress
            
            await self._load_metadata(request, sorted_chunks)
            
            # Load chunks progressively
            total_chunks = len(sorted_chunks)
            loaded_chunks = 0
            
            # Start with critical and high priority chunks
            critical_chunks = [c for c in sorted_chunks if c.priority in [ContentPriority.CRITICAL, ContentPriority.HIGH]]
            other_chunks = [c for c in sorted_chunks if c.priority not in [ContentPriority.CRITICAL, ContentPriority.HIGH]]
            
            # Load critical chunks first
            if critical_chunks:
                progress.stage = LoadingStage.LOADING_PARTIAL
                progress.current_operation = "Loading critical content..."
                
                async for chunk_progress in self._load_chunks_batch(request, critical_chunks):
                    loaded_chunks += chunk_progress
                    progress.progress_percentage = 10.0 + (loaded_chunks / total_chunks) * 60.0
                    progress.partial_results = self._get_partial_results(request)
                    yield progress
            
            # Load remaining chunks
            if other_chunks:
                progress.stage = LoadingStage.LOADING_FULL
                progress.current_operation = "Loading additional content..."
                
                async for chunk_progress in self._load_chunks_batch(request, other_chunks):
                    loaded_chunks += chunk_progress
                    progress.progress_percentage = 10.0 + (loaded_chunks / total_chunks) * 60.0
                    progress.partial_results = self._get_partial_results(request)
                    yield progress
            
            # Enhancement phase
            progress.stage = LoadingStage.ENHANCING
            progress.current_operation = "Enhancing content..."
            progress.progress_percentage = 80.0
            yield progress
            
            await self._enhance_loaded_content(request)
            
            # Complete
            progress.stage = LoadingStage.COMPLETE
            progress.current_operation = "Content loading complete"
            progress.progress_percentage = 100.0
            progress.partial_results = self._get_final_results(request)
            yield progress
            
        except asyncio.TimeoutError:
            if request.fallback_on_timeout:
                progress = await self._handle_timeout_with_fallback(request)
                yield progress
            else:
                progress.stage = LoadingStage.FAILED
                progress.error_message = "Content loading timed out"
                yield progress
                
        except Exception as e:
            logger.error(f"Progressive loading failed for request {request.request_id}: {e}")
            progress.stage = LoadingStage.FAILED
            progress.error_message = str(e)
            progress.partial_results = await self._get_fallback_results(request)
            yield progress
            
        finally:
            # Cleanup
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]
    
    def _sort_chunks_by_priority(self, chunks: List[ContentChunk]) -> List[ContentChunk]:
        """Sort chunks by priority and resolve dependencies."""
        # Create dependency graph
        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        
        # Topological sort considering priorities
        sorted_chunks = []
        remaining_chunks = chunks.copy()
        
        while remaining_chunks:
            # Find chunks with no unresolved dependencies
            ready_chunks = []
            for chunk in remaining_chunks:
                dependencies_met = all(
                    dep_id in [c.chunk_id for c in sorted_chunks]
                    for dep_id in chunk.dependencies
                )
                if dependencies_met:
                    ready_chunks.append(chunk)
            
            if not ready_chunks:
                # Circular dependency or missing dependency - add remaining chunks
                ready_chunks = remaining_chunks
            
            # Sort ready chunks by priority
            priority_order = [ContentPriority.CRITICAL, ContentPriority.HIGH, 
                            ContentPriority.MEDIUM, ContentPriority.LOW, ContentPriority.BACKGROUND]
            
            ready_chunks.sort(key=lambda c: priority_order.index(c.priority))
            
            # Add highest priority chunk
            next_chunk = ready_chunks[0]
            sorted_chunks.append(next_chunk)
            remaining_chunks.remove(next_chunk)
        
        return sorted_chunks
    
    async def _load_metadata(self, request: ProgressiveLoadRequest, chunks: List[ContentChunk]):
        """Load metadata for content chunks."""
        for chunk in chunks:
            try:
                # Try to get cached metadata
                cache_key = f"metadata_{chunk.chunk_id}"
                if cache_key in self.chunk_cache:
                    chunk.metadata = self.chunk_cache[cache_key]
                    continue
                
                # Load metadata if loader supports it
                if hasattr(chunk.loader_function, 'load_metadata'):
                    metadata = await chunk.loader_function.load_metadata()
                    chunk.metadata = metadata
                    self.chunk_cache[cache_key] = metadata
                
            except Exception as e:
                logger.warning(f"Failed to load metadata for chunk {chunk.chunk_id}: {e}")
    
    async def _load_chunks_batch(self, request: ProgressiveLoadRequest, 
                                chunks: List[ContentChunk]) -> AsyncGenerator[int, None]:
        """Load chunks in batches with concurrency control."""
        semaphore = asyncio.Semaphore(self.max_concurrent_chunks)
        
        async def load_single_chunk(chunk: ContentChunk) -> bool:
            async with semaphore:
                return await self._load_single_chunk(request, chunk)
        
        # Process chunks in batches
        batch_size = self.max_concurrent_chunks
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Start loading batch
            tasks = [load_single_chunk(chunk) for chunk in batch]
            
            # Wait for batch completion with progress updates
            completed = 0
            for task in asyncio.as_completed(tasks):
                try:
                    success = await task
                    completed += 1
                    yield 1  # One chunk completed
                    
                    # Update performance metrics
                    if success:
                        self._record_performance_metric(batch[completed-1])
                        
                except Exception as e:
                    logger.error(f"Chunk loading failed: {e}")
                    completed += 1
                    yield 1
    
    async def _load_single_chunk(self, request: ProgressiveLoadRequest, chunk: ContentChunk) -> bool:
        """Load a single content chunk."""
        if chunk.loaded or chunk.loading:
            return chunk.loaded
        
        chunk.loading = True
        chunk.load_start_time = time.time()
        
        try:
            # Check cache first
            if chunk.cache_key and chunk.cache_key in self.chunk_cache:
                chunk.result = self.chunk_cache[chunk.cache_key]
                chunk.loaded = True
                chunk.load_end_time = time.time()
                return True
            
            # Use content-specific loading strategy
            strategy = self.loading_strategies.get(chunk.content_type)
            if strategy:
                result = await strategy(request, chunk)
            else:
                # Generic loading
                result = await chunk.loader_function()
            
            chunk.result = result
            chunk.loaded = True
            chunk.load_end_time = time.time()
            
            # Cache result if cache key provided
            if chunk.cache_key:
                self.chunk_cache[chunk.cache_key] = result
            
            return True
            
        except asyncio.TimeoutError:
            logger.warning(f"Chunk {chunk.chunk_id} timed out")
            chunk.error = asyncio.TimeoutError("Chunk loading timed out")
            
            # Try to provide fallback
            if chunk.fallback_content:
                chunk.result = chunk.fallback_content
                chunk.loaded = True
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to load chunk {chunk.chunk_id}: {e}")
            chunk.error = e
            
            # Try to generate fallback
            try:
                fallback = await self._generate_chunk_fallback(request, chunk)
                chunk.result = fallback
                chunk.loaded = True
                return True
            except Exception as fallback_error:
                logger.error(f"Fallback generation failed for chunk {chunk.chunk_id}: {fallback_error}")
                return False
                
        finally:
            chunk.loading = False
    
    async def _load_chart_progressively(self, request: ProgressiveLoadRequest, 
                                      chunk: ContentChunk) -> Any:
        """Load chart content progressively."""
        # Start with basic chart structure
        partial_chart = {
            "type": "loading",
            "title": "Loading Chart...",
            "data": [],
            "loading": True
        }
        
        # Update chunk with partial result
        chunk.result = partial_chart
        
        # Load actual chart data
        try:
            # This would call the actual chart generation function
            chart_data = await asyncio.wait_for(
                chunk.loader_function(),
                timeout=self.chunk_timeout_seconds
            )
            
            return chart_data
            
        except asyncio.TimeoutError:
            # Return partial chart with timeout message
            partial_chart.update({
                "title": "Chart Loading...",
                "message": "Chart is taking longer than expected to load",
                "data": [{"x": "Loading", "y": 0}]
            })
            return partial_chart
    
    async def _load_table_progressively(self, request: ProgressiveLoadRequest, 
                                      chunk: ContentChunk) -> Any:
        """Load table content progressively."""
        # Start with table structure
        partial_table = {
            "columns": [],
            "rows": [],
            "loading": True,
            "message": "Loading table data..."
        }
        
        chunk.result = partial_table
        
        try:
            # Load table data in stages
            table_data = await asyncio.wait_for(
                chunk.loader_function(),
                timeout=self.chunk_timeout_seconds
            )
            
            # If table is large, load in batches
            if isinstance(table_data, dict) and 'rows' in table_data:
                if len(table_data['rows']) > 100:
                    # Load first 50 rows immediately
                    partial_table.update({
                        "columns": table_data.get('columns', []),
                        "rows": table_data['rows'][:50],
                        "total_rows": len(table_data['rows']),
                        "partial": True,
                        "message": f"Showing first 50 of {len(table_data['rows'])} rows"
                    })
                    
                    # Schedule loading of remaining rows
                    asyncio.create_task(self._load_remaining_table_rows(chunk, table_data))
                    
                    return partial_table
            
            return table_data
            
        except asyncio.TimeoutError:
            partial_table.update({
                "message": "Table is taking longer than expected to load",
                "columns": ["Column 1", "Column 2", "Column 3"],
                "rows": [["Loading...", "Loading...", "Loading..."]]
            })
            return partial_table
    
    async def _load_dashboard_progressively(self, request: ProgressiveLoadRequest, 
                                          chunk: ContentChunk) -> Any:
        """Load dashboard content progressively."""
        # Create dashboard skeleton
        dashboard_skeleton = {
            "widgets": [],
            "layout": "grid",
            "loading": True,
            "message": "Loading dashboard components..."
        }
        
        chunk.result = dashboard_skeleton
        
        try:
            # Load dashboard data
            dashboard_data = await asyncio.wait_for(
                chunk.loader_function(),
                timeout=self.chunk_timeout_seconds
            )
            
            # Load widgets progressively
            if isinstance(dashboard_data, dict) and 'widgets' in dashboard_data:
                widgets = dashboard_data['widgets']
                loaded_widgets = []
                
                for i, widget in enumerate(widgets):
                    try:
                        # Load widget with shorter timeout
                        widget_data = await asyncio.wait_for(
                            self._load_widget(widget),
                            timeout=5.0
                        )
                        loaded_widgets.append(widget_data)
                        
                        # Update dashboard with loaded widgets
                        dashboard_skeleton['widgets'] = loaded_widgets
                        dashboard_skeleton['message'] = f"Loaded {len(loaded_widgets)} of {len(widgets)} widgets"
                        
                    except asyncio.TimeoutError:
                        # Add placeholder widget
                        placeholder_widget = {
                            "id": widget.get('id', f'widget_{i}'),
                            "type": "placeholder",
                            "title": widget.get('title', 'Loading...'),
                            "loading": True
                        }
                        loaded_widgets.append(placeholder_widget)
                
                dashboard_skeleton['widgets'] = loaded_widgets
                dashboard_skeleton['loading'] = False
                dashboard_skeleton['message'] = "Dashboard loaded"
            
            return dashboard_skeleton
            
        except asyncio.TimeoutError:
            # Return skeleton with placeholder widgets
            placeholder_widgets = [
                {"id": "widget_1", "type": "placeholder", "title": "Loading Widget 1"},
                {"id": "widget_2", "type": "placeholder", "title": "Loading Widget 2"},
                {"id": "widget_3", "type": "placeholder", "title": "Loading Widget 3"}
            ]
            dashboard_skeleton.update({
                "widgets": placeholder_widgets,
                "message": "Dashboard is taking longer than expected to load"
            })
            return dashboard_skeleton
    
    async def _load_report_progressively(self, request: ProgressiveLoadRequest, 
                                       chunk: ContentChunk) -> Any:
        """Load report content progressively."""
        # Create report outline
        report_outline = {
            "title": "Loading Report...",
            "sections": [],
            "loading": True,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        chunk.result = report_outline
        
        try:
            # Load report data
            report_data = await asyncio.wait_for(
                chunk.loader_function(),
                timeout=self.chunk_timeout_seconds
            )
            
            return report_data
            
        except asyncio.TimeoutError:
            # Return outline with placeholder sections
            placeholder_sections = [
                {
                    "title": "Executive Summary",
                    "content": "Report section is being generated...",
                    "loading": True
                },
                {
                    "title": "Key Findings",
                    "content": "Analysis in progress...",
                    "loading": True
                }
            ]
            report_outline.update({
                "sections": placeholder_sections,
                "message": "Report is taking longer than expected to generate"
            })
            return report_outline
    
    async def _load_analysis_progressively(self, request: ProgressiveLoadRequest, 
                                         chunk: ContentChunk) -> Any:
        """Load analysis content progressively."""
        # Start with analysis placeholder
        analysis_placeholder = {
            "summary": "Analysis in progress...",
            "insights": [],
            "confidence": 0.0,
            "loading": True,
            "progress": 0
        }
        
        chunk.result = analysis_placeholder
        
        try:
            # Load analysis with progress updates
            analysis_data = await asyncio.wait_for(
                chunk.loader_function(),
                timeout=self.chunk_timeout_seconds
            )
            
            return analysis_data
            
        except asyncio.TimeoutError:
            # Return partial analysis
            analysis_placeholder.update({
                "summary": "Analysis is taking longer than expected",
                "insights": [
                    "Analysis is processing your data",
                    "Results will be available shortly",
                    "You can continue with other tasks"
                ],
                "message": "Analysis timeout - partial results shown"
            })
            return analysis_placeholder
    
    async def _load_widget(self, widget_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load individual dashboard widget."""
        widget_type = widget_config.get('type', 'unknown')
        
        if widget_type == 'chart':
            return {
                "id": widget_config.get('id'),
                "type": "chart",
                "title": widget_config.get('title', 'Chart'),
                "data": [{"x": "Sample", "y": 100}],  # Placeholder data
                "loading": False
            }
        elif widget_type == 'metric':
            return {
                "id": widget_config.get('id'),
                "type": "metric",
                "title": widget_config.get('title', 'Metric'),
                "value": "---",
                "loading": True
            }
        else:
            return {
                "id": widget_config.get('id'),
                "type": widget_type,
                "title": widget_config.get('title', 'Widget'),
                "loading": True
            }
    
    async def _load_remaining_table_rows(self, chunk: ContentChunk, full_table_data: Dict[str, Any]):
        """Load remaining table rows in background."""
        try:
            await asyncio.sleep(1)  # Small delay to let initial render complete
            
            # Update chunk result with full data
            if chunk.result and isinstance(chunk.result, dict):
                chunk.result.update({
                    "rows": full_table_data['rows'],
                    "partial": False,
                    "message": f"All {len(full_table_data['rows'])} rows loaded"
                })
                
        except Exception as e:
            logger.error(f"Failed to load remaining table rows: {e}")
    
    async def _generate_chunk_fallback(self, request: ProgressiveLoadRequest, 
                                     chunk: ContentChunk) -> Any:
        """Generate fallback content for a failed chunk."""
        context = ContentContext(
            user_id=request.user_id,
            content_type=chunk.content_type,
            original_request={"chunk_id": chunk.chunk_id},
            error_context=chunk.error
        )
        
        fallback = await intelligent_fallback_manager.generate_fallback_content(context)
        return fallback.content
    
    async def _enhance_loaded_content(self, request: ProgressiveLoadRequest):
        """Enhance loaded content with additional processing."""
        for chunk in request.content_chunks:
            if chunk.loaded and chunk.result:
                try:
                    # Apply content-specific enhancements
                    if chunk.content_type == ContentType.CHART:
                        await self._enhance_chart(chunk)
                    elif chunk.content_type == ContentType.TABLE:
                        await self._enhance_table(chunk)
                    elif chunk.content_type == ContentType.DASHBOARD:
                        await self._enhance_dashboard(chunk)
                        
                except Exception as e:
                    logger.warning(f"Failed to enhance chunk {chunk.chunk_id}: {e}")
    
    async def _enhance_chart(self, chunk: ContentChunk):
        """Enhance chart with additional features."""
        if isinstance(chunk.result, dict):
            chunk.result.update({
                "enhanced": True,
                "interactive": True,
                "export_options": ["PNG", "SVG", "PDF"]
            })
    
    async def _enhance_table(self, chunk: ContentChunk):
        """Enhance table with additional features."""
        if isinstance(chunk.result, dict):
            chunk.result.update({
                "enhanced": True,
                "sortable": True,
                "filterable": True,
                "export_options": ["CSV", "Excel", "PDF"]
            })
    
    async def _enhance_dashboard(self, chunk: ContentChunk):
        """Enhance dashboard with additional features."""
        if isinstance(chunk.result, dict):
            chunk.result.update({
                "enhanced": True,
                "customizable": True,
                "real_time_updates": True
            })
    
    def _get_partial_results(self, request: ProgressiveLoadRequest) -> Dict[str, Any]:
        """Get partial results from loaded chunks."""
        results = {}
        
        for chunk in request.content_chunks:
            if chunk.loaded and chunk.result:
                results[chunk.chunk_id] = {
                    "content": chunk.result,
                    "content_type": chunk.content_type.value,
                    "load_time": chunk.load_end_time - chunk.load_start_time if chunk.load_end_time and chunk.load_start_time else None,
                    "enhanced": getattr(chunk.result, 'enhanced', False) if isinstance(chunk.result, dict) else False
                }
            elif chunk.loading:
                results[chunk.chunk_id] = {
                    "content": None,
                    "content_type": chunk.content_type.value,
                    "status": "loading"
                }
            elif chunk.error:
                results[chunk.chunk_id] = {
                    "content": None,
                    "content_type": chunk.content_type.value,
                    "status": "error",
                    "error": str(chunk.error)
                }
        
        return results
    
    def _get_final_results(self, request: ProgressiveLoadRequest) -> Dict[str, Any]:
        """Get final results from all chunks."""
        results = self._get_partial_results(request)
        
        # Add summary information
        total_chunks = len(request.content_chunks)
        loaded_chunks = sum(1 for chunk in request.content_chunks if chunk.loaded)
        failed_chunks = sum(1 for chunk in request.content_chunks if chunk.error)
        
        results["_summary"] = {
            "total_chunks": total_chunks,
            "loaded_chunks": loaded_chunks,
            "failed_chunks": failed_chunks,
            "success_rate": loaded_chunks / total_chunks if total_chunks > 0 else 0,
            "load_complete": loaded_chunks == total_chunks
        }
        
        return results
    
    async def _get_fallback_results(self, request: ProgressiveLoadRequest) -> Dict[str, Any]:
        """Get fallback results when loading fails."""
        results = {}
        
        for chunk in request.content_chunks:
            try:
                fallback = await self._generate_chunk_fallback(request, chunk)
                results[chunk.chunk_id] = {
                    "content": fallback,
                    "content_type": chunk.content_type.value,
                    "status": "fallback"
                }
            except Exception as e:
                logger.error(f"Failed to generate fallback for chunk {chunk.chunk_id}: {e}")
                results[chunk.chunk_id] = {
                    "content": None,
                    "content_type": chunk.content_type.value,
                    "status": "failed"
                }
        
        return results
    
    async def _handle_timeout_with_fallback(self, request: ProgressiveLoadRequest) -> LoadingProgress:
        """Handle timeout by providing fallback content."""
        # Get partial results
        partial_results = self._get_partial_results(request)
        
        # Generate fallbacks for failed chunks
        for chunk in request.content_chunks:
            if not chunk.loaded and chunk.chunk_id not in partial_results:
                try:
                    fallback = await self._generate_chunk_fallback(request, chunk)
                    partial_results[chunk.chunk_id] = {
                        "content": fallback,
                        "content_type": chunk.content_type.value,
                        "status": "fallback"
                    }
                except Exception as e:
                    logger.error(f"Fallback generation failed for chunk {chunk.chunk_id}: {e}")
        
        return LoadingProgress(
            stage=LoadingStage.COMPLETE,
            progress_percentage=100.0,
            current_operation="Content loading completed with fallbacks",
            partial_results=partial_results,
            metadata={"timeout_occurred": True, "fallbacks_used": True}
        )
    
    def _record_performance_metric(self, chunk: ContentChunk):
        """Record performance metrics for chunk loading."""
        if chunk.load_start_time and chunk.load_end_time:
            load_time = chunk.load_end_time - chunk.load_start_time
            metric_key = f"{chunk.content_type.value}_{chunk.priority.value}"
            self.performance_metrics[metric_key].append(load_time)
            
            # Keep only recent metrics
            if len(self.performance_metrics[metric_key]) > 100:
                self.performance_metrics[metric_key] = self.performance_metrics[metric_key][-100:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {}
        
        for metric_key, times in self.performance_metrics.items():
            if times:
                stats[metric_key] = {
                    "avg_load_time": sum(times) / len(times),
                    "min_load_time": min(times),
                    "max_load_time": max(times),
                    "sample_count": len(times)
                }
        
        return stats
    
    def create_loading_request(self, user_id: str = None, 
                             content_chunks: List[ContentChunk] = None,
                             context: ContentContext = None,
                             timeout_seconds: float = 60.0) -> ProgressiveLoadRequest:
        """Create a new progressive loading request."""
        request_id = str(uuid.uuid4())
        
        return ProgressiveLoadRequest(
            request_id=request_id,
            user_id=user_id,
            content_chunks=content_chunks or [],
            context=context or ContentContext(),
            timeout_seconds=timeout_seconds
        )


# Global instance
progressive_content_loader = ProgressiveContentLoader()


# Convenience functions
def create_content_chunk(chunk_id: str, content_type: ContentType, 
                        loader_function: Callable, priority: ContentPriority = ContentPriority.MEDIUM,
                        dependencies: List[str] = None) -> ContentChunk:
    """Create a content chunk for progressive loading."""
    return ContentChunk(
        chunk_id=chunk_id,
        priority=priority,
        content_type=content_type,
        loader_function=loader_function,
        dependencies=dependencies or []
    )


async def load_content_with_progress(chunks: List[ContentChunk], user_id: str = None,
                                   timeout_seconds: float = 60.0) -> AsyncGenerator[LoadingProgress, None]:
    """Convenience function to load content with progress updates."""
    request = progressive_content_loader.create_loading_request(
        user_id=user_id,
        content_chunks=chunks,
        timeout_seconds=timeout_seconds
    )
    
    async for progress in progressive_content_loader.load_content_progressively(request):
        yield progress