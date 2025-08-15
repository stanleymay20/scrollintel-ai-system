"""
Performance monitoring and optimization routes for ScrollIntel API.
Provides endpoints for database optimization, file processing performance, and system monitoring.
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, status, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from ...core.database_pool import get_optimized_db_pool
from ...core.database_optimizer import get_database_optimizer
from ...core.background_jobs import get_job_processor
from ...core.performance_monitor import (
    performance_dashboard, performance_optimizer, 
    response_tracker, db_monitor, cache_manager
)
from ...api.middleware.performance_middleware import get_performance_metrics
from ...security.middleware import require_permission
from ...security.permissions import Permission
from ...security.audit import audit_logger, AuditAction
from ...core.interfaces import SecurityContext


def create_performance_router() -> APIRouter:
    """Create performance monitoring router."""
    
    router = APIRouter()
    
    @router.get("/dashboard")
    async def get_performance_dashboard(
        context: SecurityContext = Depends(require_permission(Permission.SYSTEM_HEALTH))
    ):
        """Get real-time performance dashboard data."""
        
        try:
            dashboard_data = await performance_dashboard.get_dashboard_data()
            
            # Log dashboard access
            await audit_logger.log(
                action=AuditAction.SYSTEM_ACCESS,
                resource_type="performance_dashboard",
                resource_id="dashboard",
                user_id=context.user_id,
                session_id=context.session_id,
                details={"dashboard_accessed": True},
                success=True
            )
            
            return dashboard_data
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get performance dashboard: {str(e)}"
            )
    
    @router.get("/metrics")
    async def get_performance_metrics_endpoint(
        context: SecurityContext = Depends(require_permission(Permission.SYSTEM_HEALTH))
    ):
        """Get detailed performance metrics."""
        
        try:
            metrics = get_performance_metrics()
            summary = metrics.get_summary()
            
            return {
                "summary": summary,
                "slow_requests": metrics.get_slow_requests(limit=10),
                "error_requests": metrics.get_error_requests(limit=10),
                "endpoint_metrics": metrics.get_endpoint_metrics(limit=20)
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get performance metrics: {str(e)}"
            )
    
    @router.get("/database/status")
    async def get_database_status(
        context: SecurityContext = Depends(require_permission(Permission.SYSTEM_HEALTH))
    ):
        """Get database connection pool status."""
        
        try:
            db_pool = await get_optimized_db_pool()
            pool_status = db_pool.get_pool_status()
            performance_stats = db_pool.get_performance_stats()
            
            return {
                "pool_status": pool_status,
                "performance_stats": performance_stats
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get database status: {str(e)}"
            )
    
    @router.get("/database/health")
    async def get_database_health(
        context: SecurityContext = Depends(require_permission(Permission.SYSTEM_HEALTH))
    ):
        """Get comprehensive database health report."""
        
        try:
            optimizer = await get_database_optimizer()
            health_report = await optimizer.get_database_health_report()
            
            return health_report
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get database health: {str(e)}"
            )
    
    @router.post("/database/optimize")
    async def optimize_database(
        run_maintenance: bool = Query(True, description="Run maintenance tasks"),
        generate_recommendations: bool = Query(True, description="Generate index recommendations"),
        context: SecurityContext = Depends(require_permission(Permission.SYSTEM_CONFIG))
    ):
        """Run comprehensive database optimization."""
        
        try:
            # Log optimization start
            await audit_logger.log(
                action=AuditAction.SYSTEM_OPTIMIZE,
                resource_type="database",
                resource_id="optimization",
                user_id=context.user_id,
                session_id=context.session_id,
                details={
                    "run_maintenance": run_maintenance,
                    "generate_recommendations": generate_recommendations
                },
                success=True
            )
            
            optimizer = await get_database_optimizer()
            
            if run_maintenance and generate_recommendations:
                # Run comprehensive optimization
                result = await optimizer.run_comprehensive_optimization()
            else:
                # Run specific optimizations
                result = {"started_at": "now", "status": "partial"}
                
                if run_maintenance:
                    maintenance_results = await optimizer.run_maintenance_tasks()
                    result["maintenance"] = maintenance_results
                
                if generate_recommendations:
                    table_stats = await optimizer._analyze_table_statistics()
                    recommendations = await optimizer._generate_index_recommendations(table_stats)
                    result["index_recommendations"] = recommendations
            
            # Log optimization completion
            await audit_logger.log(
                action=AuditAction.SYSTEM_OPTIMIZE_COMPLETE,
                resource_type="database",
                resource_id="optimization",
                user_id=context.user_id,
                session_id=context.session_id,
                details={
                    "status": result.get("status", "unknown"),
                    "execution_time": result.get("execution_time", 0)
                },
                success=True
            )
            
            return result
            
        except Exception as e:
            # Log optimization error
            await audit_logger.log(
                action=AuditAction.SYSTEM_OPTIMIZE,
                resource_type="database",
                resource_id="optimization",
                user_id=context.user_id,
                session_id=context.session_id,
                details={"error": str(e)},
                success=False,
                error_message=str(e)
            )
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database optimization failed: {str(e)}"
            )
    
    @router.post("/database/analyze-query")
    async def analyze_query_performance(
        query: str,
        context: SecurityContext = Depends(require_permission(Permission.SYSTEM_HEALTH))
    ):
        """Analyze performance of a specific query."""
        
        try:
            if len(query.strip()) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Query cannot be empty"
                )
            
            optimizer = await get_database_optimizer()
            analysis = await optimizer.optimize_query_performance(query)
            
            return analysis
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Query analysis failed: {str(e)}"
            )
    
    @router.get("/background-jobs/status")
    async def get_background_jobs_status(
        context: SecurityContext = Depends(require_permission(Permission.SYSTEM_HEALTH))
    ):
        """Get background job processor status."""
        
        try:
            job_processor = await get_job_processor()
            stats = await job_processor.get_stats()
            
            return stats
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get background jobs status: {str(e)}"
            )
    
    @router.get("/background-jobs/{job_id}/progress")
    async def get_job_progress(
        job_id: str,
        context: SecurityContext = Depends(require_permission(Permission.DATA_READ))
    ):
        """Get progress for a specific background job."""
        
        try:
            job_processor = await get_job_processor()
            progress = await job_processor.get_job_progress(job_id)
            
            return progress
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job progress not found: {str(e)}"
            )
    
    @router.get("/background-jobs/{job_id}/result")
    async def get_job_result(
        job_id: str,
        context: SecurityContext = Depends(require_permission(Permission.DATA_READ))
    ):
        """Get result for a completed background job."""
        
        try:
            job_processor = await get_job_processor()
            result = await job_processor.get_job_result(job_id)
            
            if not result:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Job result not found for job {job_id}"
                )
            
            return {
                "job_id": result.job_id,
                "status": result.status.value,
                "result": result.result,
                "error": result.error,
                "execution_time": result.execution_time,
                "started_at": result.started_at.isoformat() if result.started_at else None,
                "completed_at": result.completed_at.isoformat() if result.completed_at else None,
                "progress": result.progress,
                "metadata": result.metadata
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get job result: {str(e)}"
            )
    
    @router.get("/cache/stats")
    async def get_cache_stats(
        context: SecurityContext = Depends(require_permission(Permission.SYSTEM_HEALTH))
    ):
        """Get cache performance statistics."""
        
        try:
            cache_stats = cache_manager.get_cache_stats()
            return cache_stats
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get cache stats: {str(e)}"
            )
    
    @router.delete("/cache/clear")
    async def clear_cache(
        cache_key: Optional[str] = Query(None, description="Specific cache key to clear"),
        context: SecurityContext = Depends(require_permission(Permission.SYSTEM_CONFIG))
    ):
        """Clear cache entries."""
        
        try:
            if cache_key:
                await cache_manager.delete(cache_key)
                message = f"Cache key '{cache_key}' cleared"
            else:
                # Clear local cache
                cache_manager.local_cache.clear()
                cache_manager.hit_count = 0
                cache_manager.miss_count = 0
                message = "All cache entries cleared"
            
            # Log cache clear
            await audit_logger.log(
                action=AuditAction.SYSTEM_ADMIN,
                resource_type="cache",
                resource_id=cache_key or "all",
                user_id=context.user_id,
                session_id=context.session_id,
                details={"cache_cleared": True, "cache_key": cache_key},
                success=True
            )
            
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"message": message}
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to clear cache: {str(e)}"
            )
    
    @router.get("/recommendations")
    async def get_performance_recommendations(
        context: SecurityContext = Depends(require_permission(Permission.SYSTEM_HEALTH))
    ):
        """Get performance optimization recommendations."""
        
        try:
            recommendations = await performance_optimizer.get_performance_recommendations()
            return recommendations
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get performance recommendations: {str(e)}"
            )
    
    @router.get("/slow-queries")
    async def get_slow_queries(
        limit: int = Query(10, ge=1, le=50, description="Number of slow queries to return"),
        threshold: float = Query(1.0, ge=0.1, description="Minimum execution time in seconds"),
        context: SecurityContext = Depends(require_permission(Permission.SYSTEM_HEALTH))
    ):
        """Get slow database queries."""
        
        try:
            slow_queries = db_monitor.get_slow_queries(threshold=threshold)
            return slow_queries[:limit]
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get slow queries: {str(e)}"
            )
    
    @router.get("/slow-endpoints")
    async def get_slow_endpoints(
        limit: int = Query(10, ge=1, le=50, description="Number of slow endpoints to return"),
        threshold: float = Query(1.0, ge=0.1, description="Minimum response time in seconds"),
        context: SecurityContext = Depends(require_permission(Permission.SYSTEM_HEALTH))
    ):
        """Get slow API endpoints."""
        
        try:
            slow_endpoints = response_tracker.get_slow_endpoints(threshold=threshold)
            return slow_endpoints[:limit]
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get slow endpoints: {str(e)}"
            )
    
    return router

# Create the router instance
router = create_performance_router()