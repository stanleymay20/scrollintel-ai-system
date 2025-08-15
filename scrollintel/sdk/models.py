"""
Data models for ScrollIntel Python SDK.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum


class PromptVariableType(Enum):
    """Types of prompt variables."""
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    LIST = "list"
    OBJECT = "object"


@dataclass
class PromptVariable:
    """Represents a variable in a prompt template."""
    name: str
    type: PromptVariableType = PromptVariableType.STRING
    default: Optional[Union[str, int, float, bool, list, dict]] = None
    description: Optional[str] = None
    required: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "type": self.type.value,
            "default": self.default,
            "description": self.description,
            "required": self.required
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptVariable":
        """Create from dictionary representation."""
        return cls(
            name=data["name"],
            type=PromptVariableType(data.get("type", "string")),
            default=data.get("default"),
            description=data.get("description"),
            required=data.get("required", True)
        )


@dataclass
class PromptTemplate:
    """Represents a prompt template."""
    id: str
    name: str
    content: str
    category: str
    tags: List[str] = field(default_factory=list)
    variables: List[PromptVariable] = field(default_factory=list)
    description: Optional[str] = None
    is_active: bool = True
    created_by: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "content": self.content,
            "category": self.category,
            "tags": self.tags,
            "variables": [var.to_dict() for var in self.variables],
            "description": self.description,
            "is_active": self.is_active,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """Create from dictionary representation."""
        variables = []
        if data.get("variables"):
            variables = [PromptVariable.from_dict(var) for var in data["variables"]]
        
        return cls(
            id=data["id"],
            name=data["name"],
            content=data["content"],
            category=data["category"],
            tags=data.get("tags", []),
            variables=variables,
            description=data.get("description"),
            is_active=data.get("is_active", True),
            created_by=data.get("created_by", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None
        )
    
    def substitute_variables(self, variables: Dict[str, Any]) -> str:
        """Substitute variables in the prompt content."""
        import re
        result = self.content
        for var_name, var_value in variables.items():
            pattern = r'\{\{\s*' + re.escape(var_name) + r'\s*\}\}'
            result = re.sub(pattern, str(var_value), result)
        return result


@dataclass
class PromptVersion:
    """Represents a version of a prompt template."""
    id: str
    prompt_id: str
    version: str
    content: str
    changes: Optional[str] = None
    variables: List[PromptVariable] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_by: str = ""
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "prompt_id": self.prompt_id,
            "version": self.version,
            "content": self.content,
            "changes": self.changes,
            "variables": [var.to_dict() for var in self.variables],
            "tags": self.tags,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptVersion":
        """Create from dictionary representation."""
        variables = []
        if data.get("variables"):
            variables = [PromptVariable.from_dict(var) for var in data["variables"]]
        
        return cls(
            id=data["id"],
            prompt_id=data["prompt_id"],
            version=data["version"],
            content=data["content"],
            changes=data.get("changes"),
            variables=variables,
            tags=data.get("tags", []),
            created_by=data.get("created_by", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
        )


@dataclass
class SearchQuery:
    """Search query parameters for prompt templates."""
    text: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    created_by: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = 50
    offset: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "category": self.category,
            "tags": self.tags,
            "created_by": self.created_by,
            "date_from": self.date_from.isoformat() if self.date_from else None,
            "date_to": self.date_to.isoformat() if self.date_to else None,
            "limit": self.limit,
            "offset": self.offset
        }


@dataclass
class PaginatedResponse:
    """Paginated response wrapper."""
    items: List[Any]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_previous: bool
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaginatedResponse":
        """Create from dictionary representation."""
        return cls(
            items=data["items"],
            total=data["total"],
            page=data["page"],
            page_size=data["page_size"],
            has_next=data["has_next"],
            has_previous=data["has_previous"]
        )


@dataclass
class APIResponse:
    """Standard API response wrapper."""
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    timestamp: Optional[datetime] = None
    version: str = "1.0"
    request_id: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "APIResponse":
        """Create from dictionary representation."""
        return cls(
            success=data["success"],
            data=data.get("data"),
            message=data.get("message"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None,
            version=data.get("version", "1.0"),
            request_id=data.get("request_id")
        )


@dataclass
class PromptUsageMetrics:
    """Prompt usage metrics."""
    prompt_id: str
    total_uses: int
    unique_users: int
    avg_response_time: float
    success_rate: float
    last_used: Optional[datetime] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptUsageMetrics":
        """Create from dictionary representation."""
        return cls(
            prompt_id=data["prompt_id"],
            total_uses=data["total_uses"],
            unique_users=data["unique_users"],
            avg_response_time=data["avg_response_time"],
            success_rate=data["success_rate"],
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None
        )


@dataclass
class BatchOperation:
    """Represents a batch operation."""
    type: str  # "create", "update", "delete"
    prompt_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": self.type,
            "prompt_id": self.prompt_id,
            "data": self.data
        }


@dataclass
class WebhookConfig:
    """Webhook configuration."""
    url: str
    events: List[str]
    secret: Optional[str] = None
    active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "url": self.url,
            "events": self.events,
            "secret": self.secret,
            "active": self.active
        }