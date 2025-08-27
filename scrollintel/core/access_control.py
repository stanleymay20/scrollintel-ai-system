"""
Access control system for prompt management.
"""
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Set
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
import json

from scrollintel.models.audit_models import (
    AccessControl, AccessControlCreate
)


class AccessControlManager:
    """Service for managing access control and permissions."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.permission_hierarchy = {
            "read": [],
            "write": ["read"],
            "delete": ["read", "write"],
            "approve": ["read", "write"],
            "admin": ["read", "write", "delete", "approve"]
        }
    
    def grant_access(
        self,
        resource_type: str,
        resource_id: str,
        granted_by: str,
        user_id: Optional[str] = None,
        role: Optional[str] = None,
        team_id: Optional[str] = None,
        permissions: List[str] = None,
        conditions: Optional[Dict[str, Any]] = None,
        expires_at: Optional[datetime] = None
    ) -> str:
        """Grant access to a resource."""
        
        if not any([user_id, role, team_id]):
            raise ValueError("Must specify user_id, role, or team_id")
        
        if not permissions:
            permissions = ["read"]
        
        # Validate permissions
        valid_permissions = set(self.permission_hierarchy.keys())
        invalid_perms = set(permissions) - valid_permissions
        if invalid_perms:
            raise ValueError(f"Invalid permissions: {invalid_perms}")
        
        access_control = AccessControl(
            id=str(uuid.uuid4()),
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=user_id,
            role=role,
            team_id=team_id,
            permissions=permissions,
            conditions=conditions or {},
            granted_by=granted_by,
            granted_at=datetime.utcnow(),
            expires_at=expires_at
        )
        
        self.db.add(access_control)
        self.db.commit()
        
        return access_control.id
    
    def revoke_access(self, access_id: str, revoked_by: str) -> bool:
        """Revoke access control."""
        
        access_control = self.db.query(AccessControl).filter(
            AccessControl.id == access_id
        ).first()
        
        if not access_control:
            return False
        
        access_control.revoked_at = datetime.utcnow()
        access_control.revoked_by = revoked_by
        self.db.commit()
        
        return True
    
    def check_permission(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        permission: str,
        user_roles: Optional[List[str]] = None,
        user_teams: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Check if user has permission for a resource."""
        
        # Get all applicable access controls
        access_controls = self._get_applicable_access_controls(
            user_id, resource_type, resource_id, user_roles, user_teams
        )
        
        # Check each access control
        for access_control in access_controls:
            if self._has_permission(access_control, permission):
                if self._check_conditions(access_control, context):
                    return {
                        "allowed": True,
                        "access_control_id": access_control.id,
                        "granted_by": access_control.granted_by,
                        "permissions": access_control.permissions
                    }
        
        return {
            "allowed": False,
            "reason": "No matching access control found",
            "required_permission": permission
        }
    
    def _get_applicable_access_controls(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        user_roles: Optional[List[str]] = None,
        user_teams: Optional[List[str]] = None
    ) -> List[AccessControl]:
        """Get applicable access controls for user and resource."""
        
        query = self.db.query(AccessControl).filter(
            and_(
                AccessControl.resource_type == resource_type,
                AccessControl.resource_id == resource_id,
                AccessControl.revoked_at.is_(None),
                or_(
                    AccessControl.expires_at.is_(None),
                    AccessControl.expires_at > datetime.utcnow()
                )
            )
        )
        
        # Build conditions for user, role, or team access
        conditions = [AccessControl.user_id == user_id]
        
        if user_roles:
            conditions.append(AccessControl.role.in_(user_roles))
        
        if user_teams:
            conditions.append(AccessControl.team_id.in_(user_teams))
        
        query = query.filter(or_(*conditions))
        
        return query.all()
    
    def _has_permission(self, access_control: AccessControl, required_permission: str) -> bool:
        """Check if access control grants the required permission."""
        
        granted_permissions = set(access_control.permissions)
        
        # Check direct permission
        if required_permission in granted_permissions:
            return True
        
        # Check hierarchical permissions
        for granted_perm in granted_permissions:
            if granted_perm in self.permission_hierarchy:
                implied_perms = self.permission_hierarchy[granted_perm]
                if required_permission in implied_perms:
                    return True
        
        return False
    
    def _check_conditions(self, access_control: AccessControl, context: Optional[Dict[str, Any]]) -> bool:
        """Check if access control conditions are met."""
        
        if not access_control.conditions:
            return True
        
        conditions = access_control.conditions
        context = context or {}
        
        # Time-based conditions
        if "time_restrictions" in conditions:
            time_restrictions = conditions["time_restrictions"]
            current_time = datetime.utcnow()
            
            # Check allowed hours
            if "allowed_hours" in time_restrictions:
                allowed_hours = time_restrictions["allowed_hours"]
                current_hour = current_time.hour
                if current_hour not in allowed_hours:
                    return False
            
            # Check allowed days
            if "allowed_days" in time_restrictions:
                allowed_days = time_restrictions["allowed_days"]
                current_day = current_time.weekday()  # 0 = Monday
                if current_day not in allowed_days:
                    return False
        
        # IP-based conditions
        if "ip_restrictions" in conditions:
            ip_restrictions = conditions["ip_restrictions"]
            user_ip = context.get("ip_address")
            
            if "allowed_ips" in ip_restrictions:
                allowed_ips = ip_restrictions["allowed_ips"]
                if user_ip not in allowed_ips:
                    return False
            
            if "blocked_ips" in ip_restrictions:
                blocked_ips = ip_restrictions["blocked_ips"]
                if user_ip in blocked_ips:
                    return False
        
        # Location-based conditions
        if "location_restrictions" in conditions:
            location_restrictions = conditions["location_restrictions"]
            user_location = context.get("location")
            
            if "allowed_countries" in location_restrictions:
                allowed_countries = location_restrictions["allowed_countries"]
                user_country = user_location.get("country") if user_location else None
                if user_country not in allowed_countries:
                    return False
        
        return True
    
    def get_user_permissions(
        self,
        user_id: str,
        resource_type: Optional[str] = None,
        user_roles: Optional[List[str]] = None,
        user_teams: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get all permissions for a user."""
        
        query = self.db.query(AccessControl).filter(
            and_(
                AccessControl.revoked_at.is_(None),
                or_(
                    AccessControl.expires_at.is_(None),
                    AccessControl.expires_at > datetime.utcnow()
                )
            )
        )
        
        if resource_type:
            query = query.filter(AccessControl.resource_type == resource_type)
        
        # Build conditions for user, role, or team access
        conditions = [AccessControl.user_id == user_id]
        
        if user_roles:
            conditions.append(AccessControl.role.in_(user_roles))
        
        if user_teams:
            conditions.append(AccessControl.team_id.in_(user_teams))
        
        query = query.filter(or_(*conditions))
        
        access_controls = query.all()
        
        permissions = []
        for ac in access_controls:
            permissions.append({
                "id": ac.id,
                "resource_type": ac.resource_type,
                "resource_id": ac.resource_id,
                "permissions": ac.permissions,
                "conditions": ac.conditions,
                "granted_by": ac.granted_by,
                "granted_at": ac.granted_at,
                "expires_at": ac.expires_at
            })
        
        return permissions
    
    def get_resource_permissions(self, resource_type: str, resource_id: str) -> List[Dict[str, Any]]:
        """Get all permissions for a resource."""
        
        access_controls = self.db.query(AccessControl).filter(
            and_(
                AccessControl.resource_type == resource_type,
                AccessControl.resource_id == resource_id,
                AccessControl.revoked_at.is_(None)
            )
        ).all()
        
        permissions = []
        for ac in access_controls:
            permissions.append({
                "id": ac.id,
                "user_id": ac.user_id,
                "role": ac.role,
                "team_id": ac.team_id,
                "permissions": ac.permissions,
                "conditions": ac.conditions,
                "granted_by": ac.granted_by,
                "granted_at": ac.granted_at,
                "expires_at": ac.expires_at
            })
        
        return permissions
    
    def bulk_grant_access(
        self,
        resource_type: str,
        resource_ids: List[str],
        granted_by: str,
        access_config: AccessControlCreate
    ) -> List[str]:
        """Grant access to multiple resources."""
        
        access_ids = []
        
        for resource_id in resource_ids:
            access_id = self.grant_access(
                resource_type=resource_type,
                resource_id=resource_id,
                granted_by=granted_by,
                user_id=access_config.user_id,
                role=access_config.role,
                team_id=access_config.team_id,
                permissions=access_config.permissions,
                conditions=access_config.conditions,
                expires_at=access_config.expires_at
            )
            access_ids.append(access_id)
        
        return access_ids
    
    def cleanup_expired_access(self) -> int:
        """Clean up expired access controls."""
        
        expired_count = self.db.query(AccessControl).filter(
            and_(
                AccessControl.expires_at < datetime.utcnow(),
                AccessControl.revoked_at.is_(None)
            )
        ).update({
            "revoked_at": datetime.utcnow(),
            "revoked_by": "system"
        })
        
        self.db.commit()
        
        return expired_count
    
    def get_access_statistics(self) -> Dict[str, Any]:
        """Get access control statistics."""
        
        total_access_controls = self.db.query(AccessControl).count()
        
        active_access_controls = self.db.query(AccessControl).filter(
            and_(
                AccessControl.revoked_at.is_(None),
                or_(
                    AccessControl.expires_at.is_(None),
                    AccessControl.expires_at > datetime.utcnow()
                )
            )
        ).count()
        
        expired_access_controls = self.db.query(AccessControl).filter(
            and_(
                AccessControl.expires_at < datetime.utcnow(),
                AccessControl.revoked_at.is_(None)
            )
        ).count()
        
        revoked_access_controls = self.db.query(AccessControl).filter(
            AccessControl.revoked_at.is_not(None)
        ).count()
        
        # Permission distribution
        permission_counts = {}
        access_controls = self.db.query(AccessControl).filter(
            AccessControl.revoked_at.is_(None)
        ).all()
        
        for ac in access_controls:
            for perm in ac.permissions:
                permission_counts[perm] = permission_counts.get(perm, 0) + 1
        
        return {
            "total_access_controls": total_access_controls,
            "active_access_controls": active_access_controls,
            "expired_access_controls": expired_access_controls,
            "revoked_access_controls": revoked_access_controls,
            "permission_distribution": permission_counts
        }
    
    def validate_permission_request(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        permission: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate and log permission request."""
        
        # Check permission
        result = self.check_permission(user_id, resource_type, resource_id, permission, context=context)
        
        # Log the permission check
        from scrollintel.core.audit_logger import AuditLogger
        audit_logger = AuditLogger(self.db)
        
        audit_logger.log_action(
            user_id=user_id,
            user_email=context.get("user_email", "unknown") if context else "unknown",
            action="permission_check",
            resource_type=resource_type,
            resource_id=resource_id,
            metadata={
                "requested_permission": permission,
                "result": result,
                "context": context
            }
        )
        
        return result