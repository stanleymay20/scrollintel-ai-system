"""
Distributed Storage Layer with Redis Clustering for High-Performance Caching
"""
import redis
import redis.sentinel
from typing import Dict, Any, Optional, List
import json
import logging
from dataclasses import dataclass
from scrollintel.core.config import get_config

logger = logging.getLogger(__name__)

@dataclass
class RedisClusterConfig:
    """Configuration for Redis cluster setup"""
    nodes: List[Dict[str, Any]]
    password: Optional[str] = None
    max_connections: int = 100
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    health_check_interval: int = 30

class DistributedStorageManager:
    """Manages distributed storage with Redis clustering for high-performance caching"""
    
    def __init__(self, config: RedisClusterConfig):
        self.config = config
        self.cluster = None
        self.sentinel = None
        self._initialize_cluster()
    
    def _initialize_cluster(self):
        """Initialize Redis cluster connection"""
        try:
            # Try Redis Cluster first
            startup_nodes = [
                {"host": node["host"], "port": node["port"]} 
                for node in self.config.nodes
            ]
            
            self.cluster = redis.RedisCluster(
                startup_nodes=startup_nodes,
                password=self.config.password,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                decode_responses=True,
                skip_full_coverage_check=True
            )
            
            # Test connection
            self.cluster.ping()
            logger.info("Redis cluster initialized successfully")
            
        except Exception as e:
            logger.warning(f"Redis cluster initialization failed: {e}")
            # Fallback to Sentinel
            self._initialize_sentinel()
    
    def _initialize_sentinel(self):
        """Initialize Redis Sentinel as fallback"""
        try:
            sentinel_nodes = [
                (node["host"], node.get("sentinel_port", 26379)) 
                for node in self.config.nodes
            ]
            
            self.sentinel = redis.sentinel.Sentinel(
                sentinel_nodes,
                socket_timeout=self.config.socket_timeout
            )
            
            logger.info("Redis Sentinel initialized successfully")
            
        except Exception as e:
            logger.error(f"Redis Sentinel initialization failed: {e}")
            raise
    
    def get_connection(self) -> redis.Redis:
        """Get Redis connection (cluster or sentinel)"""
        if self.cluster:
            return self.cluster
        elif self.sentinel:
            return self.sentinel.master_for('mymaster', password=self.config.password)
        else:
            raise RuntimeError("No Redis connection available")
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in distributed cache"""
        try:
            conn = self.get_connection()
            serialized_value = json.dumps(value) if not isinstance(value, str) else value
            
            if ttl:
                return conn.setex(key, ttl, serialized_value)
            else:
                return conn.set(key, serialized_value)
                
        except Exception as e:
            logger.error(f"Failed to set key {key}: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from distributed cache"""
        try:
            conn = self.get_connection()
            value = conn.get(key)
            
            if value is None:
                return None
                
            # Try to deserialize JSON, fallback to string
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
                
        except Exception as e:
            logger.error(f"Failed to get key {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete key from distributed cache"""
        try:
            conn = self.get_connection()
            return bool(conn.delete(key))
        except Exception as e:
            logger.error(f"Failed to delete key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            conn = self.get_connection()
            return bool(conn.exists(key))
        except Exception as e:
            logger.error(f"Failed to check key existence {key}: {e}")
            return False
    
    def flush_all(self) -> bool:
        """Flush all keys from cache (use with caution)"""
        try:
            conn = self.get_connection()
            return conn.flushall()
        except Exception as e:
            logger.error(f"Failed to flush cache: {e}")
            return False
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get cluster information and health status"""
        try:
            if self.cluster:
                info = self.cluster.cluster_info()
                nodes = self.cluster.cluster_nodes()
                return {
                    "type": "cluster",
                    "info": info,
                    "nodes": nodes,
                    "healthy": info.get("cluster_state") == "ok"
                }
            elif self.sentinel:
                masters = self.sentinel.discover_master('mymaster')
                slaves = self.sentinel.discover_slaves('mymaster')
                return {
                    "type": "sentinel",
                    "masters": masters,
                    "slaves": slaves,
                    "healthy": len(masters) > 0
                }
        except Exception as e:
            logger.error(f"Failed to get cluster info: {e}")
            return {"type": "unknown", "healthy": False, "error": str(e)}

# Global storage manager instance
_storage_manager: Optional[DistributedStorageManager] = None

def get_storage_manager() -> DistributedStorageManager:
    """Get global storage manager instance"""
    global _storage_manager
    
    if _storage_manager is None:
        config = get_config()
        redis_config = RedisClusterConfig(
            nodes=config.get("redis_nodes", [{"host": "localhost", "port": 6379}]),
            password=config.get("redis_password"),
            max_connections=config.get("redis_max_connections", 100)
        )
        _storage_manager = DistributedStorageManager(redis_config)
    
    return _storage_manager

def cache_key(prefix: str, *args) -> str:
    """Generate cache key with prefix"""
    key_parts = [prefix] + [str(arg) for arg in args]
    return ":".join(key_parts)