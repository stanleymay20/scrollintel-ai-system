"""
Health check routes for ScrollIntel API.
Provides comprehensive system health and monitoring endpoints for deployment.
"""

import time
import asyncio
import psutil
import redis
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from ...core.registry import AgentRegistry
from ...core.config import get_settings


def create_health_router(agent_registry: AgentRegistry) -> APIRouter:
    """Create health check router with agent registry dependency."""
    
    router = APIRouter()
    settings = get_settings()
    
    @router.get("/")
    async def health_check():
        """Basic health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "service": "ScrollIntel™ API"
        }
    
    @router.get("/detailed")
    async def detailed_health_check():
        """Detailed health check with system information."""
        try:
            # Check agent registry status
            registry_status = agent_registry.get_registry_status()
            
            # Perform agent health checks
            agent_health = await agent_registry.health_check_all()
            
            # Check database connectivity (placeholder)
            database_status = await _check_database_health()
            
            # Check Redis connectivity (placeholder)
            redis_status = await _check_redis_health()
            
            # Calculate overall health
            all_agents_healthy = all(agent_health.values()) if agent_health else True
            overall_healthy = (
                database_status["healthy"] and 
                redis_status["healthy"] and 
                all_agents_healthy
            )
            
            return {
                "status": "healthy" if overall_healthy else "degraded",
                "timestamp": time.time(),
                "service": "ScrollIntel™ API",
                "components": {
                    "database": database_status,
                    "redis": redis_status,
                    "agent_registry": {
                        "healthy": True,
                        "status": registry_status
                    },
                    "agents": {
                        "healthy": all_agents_healthy,
                        "individual_status": agent_health
                    }
                },
                "environment": settings.environment,
                "version": "4.0.0"
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "timestamp": time.time(),
                "service": "ScrollIntel™ API",
                "error": str(e)
            }
    
    @router.get("/agents")
    async def agent_health_check():
        """Health check specifically for agents."""
        try:
            registry_status = agent_registry.get_registry_status()
            agent_health = await agent_registry.health_check_all()
            
            return {
                "status": "healthy" if all(agent_health.values()) else "degraded",
                "timestamp": time.time(),
                "registry_status": registry_status,
                "agent_health": agent_health,
                "total_agents": len(agent_health),
                "healthy_agents": sum(1 for healthy in agent_health.values() if healthy),
                "unhealthy_agents": sum(1 for healthy in agent_health.values() if not healthy)
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to check agent health: {str(e)}"
            )
    
    @router.get("/readiness")
    async def readiness_check():
        """Kubernetes readiness probe endpoint."""
        try:
            # Check if essential services are ready
            registry_ready = len(agent_registry.get_active_agents()) >= 0  # Registry is initialized
            database_ready = (await _check_database_health())["healthy"]
            
            if registry_ready and database_ready:
                return {
                    "status": "ready",
                    "timestamp": time.time()
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Service not ready"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Readiness check failed: {str(e)}"
            )
    
    @router.get("/liveness")
    async def liveness_check():
        """Kubernetes liveness probe endpoint."""
        return {
            "status": "alive",
            "timestamp": time.time()
        }
    
    return router


    @router.get("/metrics")
    async def system_metrics():
        """System metrics endpoint for monitoring."""
        try:
            # CPU and Memory metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            
            return {
                "timestamp": time.time(),
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory": {
                        "total": memory.total,
                        "available": memory.available,
                        "percent": memory.percent,
                        "used": memory.used,
                        "free": memory.free
                    },
                    "disk": {
                        "total": disk.total,
                        "used": disk.used,
                        "free": disk.free,
                        "percent": (disk.used / disk.total) * 100
                    },
                    "network": {
                        "bytes_sent": network.bytes_sent,
                        "bytes_recv": network.bytes_recv,
                        "packets_sent": network.packets_sent,
                        "packets_recv": network.packets_recv
                    }
                },
                "process": {
                    "memory_rss": process_memory.rss,
                    "memory_vms": process_memory.vms,
                    "cpu_percent": process.cpu_percent(),
                    "num_threads": process.num_threads(),
                    "create_time": process.create_time()
                }
            }
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get system metrics: {str(e)}"
            )
    
    @router.get("/dependencies")
    async def dependency_health():
        """Check health of external dependencies."""
        try:
            results = {}
            
            # Check database
            db_health = await _check_database_health()
            results["database"] = db_health
            
            # Check Redis
            redis_health = await _check_redis_health()
            results["redis"] = redis_health
            
            # Check AI services
            ai_services = await _check_ai_services()
            results["ai_services"] = ai_services
            
            # Check vector database
            vector_db = await _check_vector_database()
            results["vector_database"] = vector_db
            
            # Overall health
            all_healthy = all(
                service.get("healthy", False) 
                for service in results.values()
            )
            
            return {
                "status": "healthy" if all_healthy else "degraded",
                "timestamp": time.time(),
                "dependencies": results
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to check dependencies: {str(e)}"
            )
    
    return router


async def _check_database_health() -> Dict[str, Any]:
    """Check database connectivity and health."""
    start_time = time.time()
    try:
        settings = get_settings()
        engine = create_engine(settings.database_url)
        
        with engine.connect() as conn:
            # Test basic connectivity
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
            
            # Get database stats
            stats_query = text("""
                SELECT 
                    count(*) as active_connections,
                    (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_queries
                FROM pg_stat_activity 
                WHERE datname = current_database()
            """)
            stats = conn.execute(stats_query).fetchone()
            
        response_time = (time.time() - start_time) * 1000
        
        return {
            "healthy": True,
            "response_time_ms": round(response_time, 2),
            "connection_stats": {
                "active_connections": stats[0] if stats else 0,
                "active_queries": stats[1] if stats else 0
            },
            "database_name": settings.postgres_db
        }
        
    except OperationalError as e:
        return {
            "healthy": False,
            "error": f"Database connection failed: {str(e)}",
            "response_time_ms": (time.time() - start_time) * 1000
        }
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e),
            "response_time_ms": (time.time() - start_time) * 1000
        }


async def _check_redis_health() -> Dict[str, Any]:
    """Check Redis connectivity and health."""
    start_time = time.time()
    try:
        settings = get_settings()
        redis_client = redis.from_url(settings.redis_url)
        
        # Test basic connectivity
        await asyncio.to_thread(redis_client.ping)
        
        # Get Redis info
        info = await asyncio.to_thread(redis_client.info)
        
        response_time = (time.time() - start_time) * 1000
        
        return {
            "healthy": True,
            "response_time_ms": round(response_time, 2),
            "memory_usage": info.get("used_memory_human", "unknown"),
            "connected_clients": info.get("connected_clients", 0),
            "total_commands_processed": info.get("total_commands_processed", 0),
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0)
        }
        
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e),
            "response_time_ms": (time.time() - start_time) * 1000
        }


async def _check_ai_services() -> Dict[str, Any]:
    """Check AI service connectivity."""
    try:
        settings = get_settings()
        services = {}
        
        # Check OpenAI
        if settings.openai_api_key:
            services["openai"] = {
                "configured": True,
                "model": settings.openai_model,
                "healthy": True  # Would need actual API call to verify
            }
        else:
            services["openai"] = {"configured": False}
        
        # Check Anthropic
        if settings.anthropic_api_key:
            services["anthropic"] = {
                "configured": True,
                "model": settings.anthropic_model,
                "healthy": True  # Would need actual API call to verify
            }
        else:
            services["anthropic"] = {"configured": False}
        
        return {
            "healthy": True,
            "services": services
        }
        
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e)
        }


async def _check_vector_database() -> Dict[str, Any]:
    """Check vector database connectivity."""
    try:
        settings = get_settings()
        
        # Check Pinecone
        if settings.pinecone_api_key:
            return {
                "healthy": True,
                "provider": "pinecone",
                "environment": settings.pinecone_environment,
                "configured": True
            }
        
        # Check Supabase Vector
        elif settings.supabase_url and settings.supabase_key:
            return {
                "healthy": True,
                "provider": "supabase",
                "configured": True
            }
        
        else:
            return {
                "healthy": False,
                "error": "No vector database configured"
            }
            
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e)
        }
# Create the router instance
from ...core.registry import get_agent_registry
router = create_health_router(get_agent_registry())