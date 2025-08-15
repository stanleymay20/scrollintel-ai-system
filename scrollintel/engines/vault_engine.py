"""
ScrollVaultEngine - Secure Insight Storage & Versioning
End-to-end encrypted storage with semantic search and audit trails.
"""

import asyncio
import json
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from uuid import uuid4
from dataclasses import dataclass, asdict
from enum import Enum
import base64
import logging

# Encryption
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# Vector search
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

from .base_engine import BaseEngine, EngineStatus, EngineCapability

logger = logging.getLogger(__name__)


class InsightType(str, Enum):
    """Types of insights that can be stored."""
    ANALYSIS_RESULT = "analysis_result"
    MODEL_EXPLANATION = "model_explanation"
    PREDICTION = "prediction"
    REPORT = "report"
    VISUALIZATION = "visualization"
    RECOMMENDATION = "recommendation"
    AUDIT_LOG = "audit_log"
    RESEARCH_FINDING = "research_finding"


class AccessLevel(str, Enum):
    """Access levels for insights."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class RetentionPolicy(str, Enum):
    """Data retention policies."""
    PERMANENT = "permanent"
    LONG_TERM = "long_term"  # 7 years
    MEDIUM_TERM = "medium_term"  # 3 years
    SHORT_TERM = "short_term"  # 1 year
    TEMPORARY = "temporary"  # 90 days


@dataclass
class VaultInsight:
    """Encrypted insight stored in the vault."""
    id: str
    title: str
    content: str  # Encrypted
    insight_type: InsightType
    access_level: AccessLevel
    retention_policy: RetentionPolicy
    creator_id: str
    organization_id: str
    tags: List[str]
    metadata: Dict[str, Any]
    version: int
    parent_id: Optional[str] = None
    encryption_key_id: str = None
    content_hash: str = None
    created_at: datetime = None
    updated_at: datetime = None
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = self.created_at
        if self.expires_at is None and self.retention_policy != RetentionPolicy.PERMANENT:
            self.expires_at = self._calculate_expiry()
    
    def _calculate_expiry(self) -> datetime:
        """Calculate expiry date based on retention policy."""
        now = datetime.utcnow()
        if self.retention_policy == RetentionPolicy.TEMPORARY:
            return now + timedelta(days=90)
        elif self.retention_policy == RetentionPolicy.SHORT_TERM:
            return now + timedelta(days=365)
        elif self.retention_policy == RetentionPolicy.MEDIUM_TERM:
            return now + timedelta(days=365 * 3)
        elif self.retention_policy == RetentionPolicy.LONG_TERM:
            return now + timedelta(days=365 * 7)
        else:
            return None


@dataclass
class AccessAuditLog:
    """Audit log for insight access."""
    id: str
    insight_id: str
    user_id: str
    action: str  # "read", "write", "delete", "share"
    ip_address: str
    user_agent: str
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class SearchQuery:
    """Search query for vault insights."""
    query: str
    filters: Dict[str, Any] = None
    user_id: str = None
    access_levels: List[AccessLevel] = None
    insight_types: List[InsightType] = None
    date_range: Tuple[datetime, datetime] = None
    limit: int = 50
    offset: int = 0
    
    def __post_init__(self):
        if self.filters is None:
            self.filters = {}
        if self.access_levels is None:
            self.access_levels = [AccessLevel.PUBLIC, AccessLevel.INTERNAL]


class ScrollVaultEngine(BaseEngine):
    """Secure insight storage engine with encryption and semantic search."""
    
    def __init__(self):
        super().__init__(
            engine_id="scroll-vault-engine",
            name="ScrollVault Engine",
            capabilities=[
                EngineCapability.SECURE_STORAGE,
                EngineCapability.DATA_ANALYSIS
            ]
        )
        
        # Encryption components
        self.master_key = None
        self.encryption_keys = {}
        self.cipher_suite = None
        
        # Storage
        self.insights = {}  # In production, this would be a database
        self.audit_logs = []
        self.search_index = {}
        
        # Semantic search
        self.embedding_model = None
        self.embeddings_cache = {}
        
        # Access control
        self.access_policies = {}
        self.user_permissions = {}
    
    async def initialize(self) -> None:
        """Initialize the vault engine."""
        try:
            # Initialize encryption
            await self._initialize_encryption()
            
            # Initialize semantic search
            if EMBEDDINGS_AVAILABLE:
                try:
                    self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    logger.info("Semantic search model loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load embedding model: {e}")
            
            # Initialize access policies
            await self._initialize_access_policies()
            
            self.status = EngineStatus.READY
            logger.info("ScrollVaultEngine initialized successfully")
            
        except Exception as e:
            self.status = EngineStatus.ERROR
            logger.error(f"Failed to initialize ScrollVaultEngine: {e}")
            raise
    
    async def process(self, input_data: Any, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process vault operations."""
        params = parameters or {}
        operation = params.get("operation")
        user_id = params.get("user_id")
        
        if operation == "store_insight":
            return await self._store_insight(input_data, user_id, params)
        elif operation == "retrieve_insight":
            return await self._retrieve_insight(params.get("insight_id"), user_id, params)
        elif operation == "search_insights":
            return await self._search_insights(input_data, user_id, params)
        elif operation == "update_insight":
            return await self._update_insight(params.get("insight_id"), input_data, user_id, params)
        elif operation == "delete_insight":
            return await self._delete_insight(params.get("insight_id"), user_id, params)
        elif operation == "get_insight_history":
            return await self._get_insight_history(params.get("insight_id"), user_id, params)
        elif operation == "audit_access":
            return await self._get_access_audit(user_id, params)
        elif operation == "cleanup_expired":
            return await self._cleanup_expired_insights()
        else:
            raise ValueError(f"Unknown vault operation: {operation}")
    
    async def _store_insight(self, insight_data: Dict[str, Any], user_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Store an encrypted insight in the vault."""
        try:
            # Create insight object
            insight = VaultInsight(
                id=f"insight-{uuid4()}",
                title=insight_data.get("title", "Untitled Insight"),
                content="",  # Will be encrypted
                insight_type=InsightType(insight_data.get("type", InsightType.ANALYSIS_RESULT)),
                access_level=AccessLevel(insight_data.get("access_level", AccessLevel.INTERNAL)),
                retention_policy=RetentionPolicy(insight_data.get("retention_policy", RetentionPolicy.MEDIUM_TERM)),
                creator_id=user_id,
                organization_id=params.get("organization_id", "default"),
                tags=insight_data.get("tags", []),
                metadata=insight_data.get("metadata", {}),
                version=1
            )
            
            # Encrypt content
            content = json.dumps(insight_data.get("content", {}))
            encrypted_content, key_id = await self._encrypt_content(content)
            insight.content = encrypted_content
            insight.encryption_key_id = key_id
            insight.content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            # Check access permissions
            if not await self._check_write_permission(user_id, insight.access_level):
                raise PermissionError(f"User {user_id} does not have write permission for {insight.access_level}")
            
            # Store insight
            self.insights[insight.id] = insight
            
            # Create search embeddings
            if self.embedding_model:
                search_text = f"{insight.title} {' '.join(insight.tags)} {json.dumps(insight.metadata)}"
                embedding = self.embedding_model.encode(search_text)
                self.embeddings_cache[insight.id] = embedding
            
            # Audit log
            await self._log_access(insight.id, user_id, "write", params.get("ip_address", "unknown"), True)
            
            return {
                "success": True,
                "insight_id": insight.id,
                "version": insight.version,
                "encrypted": True,
                "expires_at": insight.expires_at.isoformat() if insight.expires_at else None,
                "created_at": insight.created_at.isoformat()
            }
            
        except Exception as e:
            await self._log_access("unknown", user_id, "write", params.get("ip_address", "unknown"), False, str(e))
            raise
    
    async def _retrieve_insight(self, insight_id: str, user_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve and decrypt an insight."""
        try:
            if insight_id not in self.insights:
                raise ValueError(f"Insight {insight_id} not found")
            
            insight = self.insights[insight_id]
            
            # Check access permissions
            if not await self._check_read_permission(user_id, insight.access_level, insight.creator_id):
                raise PermissionError(f"User {user_id} does not have read permission for insight {insight_id}")
            
            # Check if expired
            if insight.expires_at and datetime.utcnow() > insight.expires_at:
                raise ValueError(f"Insight {insight_id} has expired")
            
            # Decrypt content
            decrypted_content = await self._decrypt_content(insight.content, insight.encryption_key_id)
            
            # Update access tracking
            insight.access_count += 1
            insight.last_accessed = datetime.utcnow()
            
            # Audit log
            await self._log_access(insight_id, user_id, "read", params.get("ip_address", "unknown"), True)
            
            return {
                "success": True,
                "insight": {
                    "id": insight.id,
                    "title": insight.title,
                    "content": json.loads(decrypted_content),
                    "type": insight.insight_type.value,
                    "access_level": insight.access_level.value,
                    "creator_id": insight.creator_id,
                    "tags": insight.tags,
                    "metadata": insight.metadata,
                    "version": insight.version,
                    "created_at": insight.created_at.isoformat(),
                    "updated_at": insight.updated_at.isoformat(),
                    "access_count": insight.access_count,
                    "last_accessed": insight.last_accessed.isoformat() if insight.last_accessed else None
                }
            }
            
        except Exception as e:
            await self._log_access(insight_id, user_id, "read", params.get("ip_address", "unknown"), False, str(e))
            raise
    
    async def _search_insights(self, query_data: Dict[str, Any], user_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Search insights using semantic search and filters."""
        try:
            search_query = SearchQuery(
                query=query_data.get("query", ""),
                filters=query_data.get("filters", {}),
                user_id=user_id,
                access_levels=query_data.get("access_levels"),
                insight_types=query_data.get("insight_types"),
                limit=query_data.get("limit", 50),
                offset=query_data.get("offset", 0)
            )
            
            # Get accessible insights
            accessible_insights = []
            for insight in self.insights.values():
                if await self._check_read_permission(user_id, insight.access_level, insight.creator_id):
                    # Check if expired
                    if not insight.expires_at or datetime.utcnow() <= insight.expires_at:
                        accessible_insights.append(insight)
            
            # Apply filters
            filtered_insights = await self._apply_search_filters(accessible_insights, search_query)
            
            # Semantic search if query provided
            if search_query.query and self.embedding_model:
                ranked_insights = await self._semantic_search(filtered_insights, search_query.query)
            else:
                ranked_insights = filtered_insights
            
            # Pagination
            start_idx = search_query.offset
            end_idx = start_idx + search_query.limit
            paginated_insights = ranked_insights[start_idx:end_idx]
            
            # Format results
            results = []
            for insight in paginated_insights:
                results.append({
                    "id": insight.id,
                    "title": insight.title,
                    "type": insight.insight_type.value,
                    "access_level": insight.access_level.value,
                    "creator_id": insight.creator_id,
                    "tags": insight.tags,
                    "created_at": insight.created_at.isoformat(),
                    "updated_at": insight.updated_at.isoformat(),
                    "version": insight.version,
                    "access_count": insight.access_count
                })
            
            return {
                "success": True,
                "results": results,
                "total_count": len(ranked_insights),
                "query": search_query.query,
                "filters_applied": search_query.filters,
                "offset": search_query.offset,
                "limit": search_query.limit
            }
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    async def _update_insight(self, insight_id: str, update_data: Dict[str, Any], user_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing insight with version control."""
        try:
            if insight_id not in self.insights:
                raise ValueError(f"Insight {insight_id} not found")
            
            insight = self.insights[insight_id]
            
            # Check write permissions
            if not await self._check_write_permission(user_id, insight.access_level):
                raise PermissionError(f"User {user_id} does not have write permission for insight {insight_id}")
            
            # Check if expired
            if insight.expires_at and datetime.utcnow() > insight.expires_at:
                raise ValueError(f"Insight {insight_id} has expired and cannot be updated")
            
            # Create new version
            new_insight = VaultInsight(
                id=f"insight-{uuid4()}",
                title=update_data.get("title", insight.title),
                content="",  # Will be encrypted
                insight_type=insight.insight_type,
                access_level=AccessLevel(update_data.get("access_level", insight.access_level.value)),
                retention_policy=RetentionPolicy(update_data.get("retention_policy", insight.retention_policy.value)),
                creator_id=insight.creator_id,
                organization_id=insight.organization_id,
                tags=update_data.get("tags", insight.tags),
                metadata=update_data.get("metadata", insight.metadata),
                version=insight.version + 1,
                parent_id=insight_id
            )
            
            # Encrypt new content if provided
            if "content" in update_data:
                content = json.dumps(update_data["content"])
                encrypted_content, key_id = await self._encrypt_content(content)
                new_insight.content = encrypted_content
                new_insight.encryption_key_id = key_id
                new_insight.content_hash = hashlib.sha256(content.encode()).hexdigest()
            else:
                # Keep existing content
                new_insight.content = insight.content
                new_insight.encryption_key_id = insight.encryption_key_id
                new_insight.content_hash = insight.content_hash
            
            # Store new version
            self.insights[new_insight.id] = new_insight
            
            # Update search embeddings if content changed
            if "content" in update_data and self.embedding_model:
                search_text = f"{new_insight.title} {' '.join(new_insight.tags)} {json.dumps(new_insight.metadata)}"
                embedding = self.embedding_model.encode(search_text)
                self.embeddings_cache[new_insight.id] = embedding
            
            # Audit log
            await self._log_access(new_insight.id, user_id, "update", params.get("ip_address", "unknown"), True)
            
            return {
                "success": True,
                "insight_id": new_insight.id,
                "version": new_insight.version,
                "parent_id": new_insight.parent_id,
                "updated_at": new_insight.updated_at.isoformat()
            }
            
        except Exception as e:
            await self._log_access(insight_id, user_id, "update", params.get("ip_address", "unknown"), False, str(e))
            raise
    
    async def _delete_insight(self, insight_id: str, user_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete an insight (soft delete with audit trail)."""
        try:
            if insight_id not in self.insights:
                raise ValueError(f"Insight {insight_id} not found")
            
            insight = self.insights[insight_id]
            
            # Check delete permissions (only creator or admin)
            if insight.creator_id != user_id and not await self._check_admin_permission(user_id):
                raise PermissionError(f"User {user_id} does not have delete permission for insight {insight_id}")
            
            # Remove from storage
            del self.insights[insight_id]
            
            # Remove from search cache
            if insight_id in self.embeddings_cache:
                del self.embeddings_cache[insight_id]
            
            # Audit log
            await self._log_access(insight_id, user_id, "delete", params.get("ip_address", "unknown"), True)
            
            return {
                "success": True,
                "insight_id": insight_id,
                "deleted_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            await self._log_access(insight_id, user_id, "delete", params.get("ip_address", "unknown"), False, str(e))
            raise
    
    async def _get_insight_history(self, insight_id: str, user_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get version history for an insight."""
        try:
            if insight_id not in self.insights:
                raise ValueError(f"Insight {insight_id} not found")
            
            base_insight = self.insights[insight_id]
            
            # Check read permissions
            if not await self._check_read_permission(user_id, base_insight.access_level, base_insight.creator_id):
                raise PermissionError(f"User {user_id} does not have read permission for insight {insight_id}")
            
            # Find all versions
            versions = []
            for insight in self.insights.values():
                if insight.parent_id == insight_id or insight.id == insight_id:
                    versions.append({
                        "id": insight.id,
                        "version": insight.version,
                        "title": insight.title,
                        "created_at": insight.created_at.isoformat(),
                        "updated_at": insight.updated_at.isoformat(),
                        "creator_id": insight.creator_id,
                        "is_current": insight.id == insight_id
                    })
            
            # Sort by version
            versions.sort(key=lambda x: x["version"])
            
            return {
                "success": True,
                "insight_id": insight_id,
                "versions": versions,
                "total_versions": len(versions)
            }
            
        except Exception as e:
            logger.error(f"Get insight history failed: {e}")
            raise
    
    async def _get_access_audit(self, user_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get access audit logs for insights."""
        try:
            # Filter audit logs by user if not admin
            if not await self._check_admin_permission(user_id):
                filtered_logs = [log for log in self.audit_logs if log.user_id == user_id]
            else:
                filtered_logs = self.audit_logs
            
            # Apply filters
            insight_id = params.get("insight_id")
            if insight_id:
                filtered_logs = [log for log in filtered_logs if log.insight_id == insight_id]
            
            action = params.get("action")
            if action:
                filtered_logs = [log for log in filtered_logs if log.action == action]
            
            # Pagination
            limit = params.get("limit", 50)
            offset = params.get("offset", 0)
            paginated_logs = filtered_logs[offset:offset + limit]
            
            # Format results
            results = []
            for log in paginated_logs:
                results.append({
                    "id": log.id,
                    "insight_id": log.insight_id,
                    "user_id": log.user_id,
                    "action": log.action,
                    "ip_address": log.ip_address,
                    "success": log.success,
                    "error_message": log.error_message,
                    "timestamp": log.timestamp.isoformat()
                })
            
            return {
                "success": True,
                "audit_logs": results,
                "total_count": len(filtered_logs),
                "offset": offset,
                "limit": limit
            }
            
        except Exception as e:
            logger.error(f"Get access audit failed: {e}")
            raise
    
    async def _cleanup_expired_insights(self) -> Dict[str, Any]:
        """Clean up expired insights based on retention policies."""
        try:
            now = datetime.utcnow()
            expired_insights = []
            
            for insight_id, insight in list(self.insights.items()):
                if insight.expires_at and now > insight.expires_at:
                    expired_insights.append(insight_id)
                    
                    # Remove from storage
                    del self.insights[insight_id]
                    
                    # Remove from search cache
                    if insight_id in self.embeddings_cache:
                        del self.embeddings_cache[insight_id]
                    
                    # Log cleanup
                    await self._log_access(insight_id, "system", "cleanup", "system", True)
            
            return {
                "success": True,
                "cleaned_up_count": len(expired_insights),
                "expired_insights": expired_insights,
                "cleanup_timestamp": now.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Cleanup expired insights failed: {e}")
            raise
    
    async def _initialize_encryption(self):
        """Initialize encryption components."""
        if not CRYPTO_AVAILABLE:
            logger.warning("Cryptography library not available, using mock encryption")
            return
        
        # Generate or load master key
        master_key_data = b"scroll-vault-master-key-32-bytes!"  # In production, load from secure storage
        # Ensure key is exactly 32 bytes
        if len(master_key_data) != 32:
            master_key_data = master_key_data[:32].ljust(32, b'!')
        self.master_key = base64.urlsafe_b64encode(master_key_data)
        self.cipher_suite = Fernet(self.master_key)
        
        # Initialize key rotation
        await self._initialize_key_rotation()
    
    async def _encrypt_content(self, content: str) -> Tuple[str, str]:
        """Encrypt content and return encrypted data with key ID."""
        if not CRYPTO_AVAILABLE or not self.cipher_suite:
            # Mock encryption
            return base64.b64encode(content.encode()).decode(), "mock-key-id"
        
        try:
            encrypted_data = self.cipher_suite.encrypt(content.encode())
            key_id = f"key-{uuid4()}"
            
            # Store key reference (in production, use proper key management)
            self.encryption_keys[key_id] = self.master_key
            
            return base64.b64encode(encrypted_data).decode(), key_id
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    async def _decrypt_content(self, encrypted_content: str, key_id: str) -> str:
        """Decrypt content using the specified key."""
        if not CRYPTO_AVAILABLE or not self.cipher_suite:
            # Mock decryption
            return base64.b64decode(encrypted_content.encode()).decode()
        
        try:
            encrypted_data = base64.b64decode(encrypted_content.encode())
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            return decrypted_data.decode()
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    async def _check_read_permission(self, user_id: str, access_level: AccessLevel, creator_id: str) -> bool:
        """Check if user has read permission for the access level."""
        # Mock permission check - in production, integrate with EXOUSIA
        if user_id == creator_id:
            return True
        
        user_clearance = self.user_permissions.get(user_id, AccessLevel.PUBLIC)
        
        access_hierarchy = {
            AccessLevel.PUBLIC: 0,
            AccessLevel.INTERNAL: 1,
            AccessLevel.CONFIDENTIAL: 2,
            AccessLevel.RESTRICTED: 3,
            AccessLevel.TOP_SECRET: 4
        }
        
        return access_hierarchy.get(user_clearance, 0) >= access_hierarchy.get(access_level, 0)
    
    async def _check_write_permission(self, user_id: str, access_level: AccessLevel) -> bool:
        """Check if user has write permission for the access level."""
        # Mock permission check
        return await self._check_read_permission(user_id, access_level, user_id)
    
    async def _check_admin_permission(self, user_id: str) -> bool:
        """Check if user has admin permissions."""
        # Mock admin check - in production, integrate with EXOUSIA
        user_clearance = self.user_permissions.get(user_id, AccessLevel.PUBLIC)
        return user_clearance == AccessLevel.TOP_SECRET or user_id == "admin"
    
    async def _log_access(self, insight_id: str, user_id: str, action: str, ip_address: str, 
                         success: bool, error_message: str = None):
        """Log access attempt for audit purposes."""
        audit_log = AccessAuditLog(
            id=f"audit-{uuid4()}",
            insight_id=insight_id,
            user_id=user_id,
            action=action,
            ip_address=ip_address,
            user_agent="ScrollIntel-Vault",
            success=success,
            error_message=error_message
        )
        
        self.audit_logs.append(audit_log)
        
        # In production, store in secure audit database
        logger.info(f"Audit: {action} on {insight_id} by {user_id} - {'SUCCESS' if success else 'FAILED'}")
    
    async def cleanup(self) -> None:
        """Clean up vault engine resources."""
        self.insights.clear()
        self.audit_logs.clear()
        self.search_index.clear()
        self.embeddings_cache.clear()
        self.encryption_keys.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """Get vault engine status."""
        return {
            "engine_id": self.engine_id,
            "status": self.status.value,
            "crypto_available": CRYPTO_AVAILABLE,
            "embeddings_available": EMBEDDINGS_AVAILABLE,
            "stored_insights": len(self.insights),
            "audit_logs": len(self.audit_logs),
            "cached_embeddings": len(self.embeddings_cache),
            "encryption_keys": len(self.encryption_keys),
            "healthy": self.status == EngineStatus.READY
        }
    
    # Helper methods
    async def _initialize_access_policies(self):
        """Initialize default access policies."""
        # Mock user permissions
        self.user_permissions = {
            "admin": AccessLevel.TOP_SECRET,
            "analyst": AccessLevel.CONFIDENTIAL,
            "viewer": AccessLevel.INTERNAL
        }
    
    async def _initialize_key_rotation(self):
        """Initialize key rotation schedule."""
        # In production, implement automatic key rotation
        pass
    
    async def _apply_search_filters(self, insights: List[VaultInsight], query: SearchQuery) -> List[VaultInsight]:
        """Apply search filters to insights."""
        filtered = insights
        
        # Filter by insight types
        if query.insight_types:
            filtered = [i for i in filtered if i.insight_type in query.insight_types]
        
        # Filter by access levels
        if query.access_levels:
            filtered = [i for i in filtered if i.access_level in query.access_levels]
        
        # Filter by date range
        if query.date_range:
            start_date, end_date = query.date_range
            filtered = [i for i in filtered if start_date <= i.created_at <= end_date]
        
        # Apply custom filters
        for key, value in query.filters.items():
            if key == "tags":
                filtered = [i for i in filtered if any(tag in i.tags for tag in value)]
            elif key == "creator_id":
                filtered = [i for i in filtered if i.creator_id == value]
        
        return filtered
    
    async def _semantic_search(self, insights: List[VaultInsight], query: str) -> List[VaultInsight]:
        """Perform semantic search on insights."""
        if not self.embedding_model:
            return insights
        
        try:
            # Get query embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Calculate similarities
            similarities = []
            for insight in insights:
                if insight.id in self.embeddings_cache:
                    insight_embedding = self.embeddings_cache[insight.id]
                    similarity = np.dot(query_embedding, insight_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(insight_embedding)
                    )
                    similarities.append((insight, similarity))
                else:
                    similarities.append((insight, 0.0))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return [insight for insight, _ in similarities]
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return insights