"""
Permission system for ScrollIntel API
"""

from typing import List, Dict, Any
from fastapi import HTTPException, status, Depends
from ..security.auth import get_current_user


async def require_permissions(user: Dict[str, Any], required_permissions: List[str]) -> Dict[str, Any]:
    """
    Check if user has required permissions
    
    Args:
        user: Current user data
        required_permissions: List of required permissions
        
    Returns:
        User data if permissions are valid
        
    Raises:
        HTTPException: If user lacks required permissions
    """
    user_permissions = user.get("permissions", [])
    user_role = user.get("role", "user")
    
    # Admin users have all permissions
    if user_role == "admin":
        return user
    
    # Check if user has any of the required permissions
    if not any(perm in user_permissions for perm in required_permissions):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Insufficient permissions. Required: {required_permissions}"
        )
    
    return user


def create_permission_dependency(required_permissions: List[str]):
    """
    Create a FastAPI dependency that checks for specific permissions
    
    Args:
        required_permissions: List of required permissions
        
    Returns:
        FastAPI dependency function
    """
    async def permission_dependency(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        return await require_permissions(user, required_permissions)
    
    return permission_dependency


# Common permission dependencies
require_visual_generation = create_permission_dependency(["visual_generation"])
require_admin = create_permission_dependency(["admin"])
require_api_access = create_permission_dependency(["api_access"])