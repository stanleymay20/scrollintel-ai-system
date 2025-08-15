"""
Sharing Manager for project collaboration and permissions.
"""

import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from scrollintel.models.collaboration_models import (
    Project, ProjectShare, SharePermission, ShareRequest,
    ApprovalWorkflow, ApprovalRequest, ApprovalResponse, ApprovalStatus
)
from scrollintel.models.database_utils import get_sync_db

def get_db_session():
    """Wrapper for database session."""
    return get_sync_db()
from scrollintel.core.config import get_config


class SharingManager:
    """Manages project sharing and collaboration permissions."""
    
    def __init__(self):
        self.config = get_config()
        # Use a default value since it's not in the config
        self.default_share_expiry_days = 30
    
    async def share_project(self, project_id: int, share_request: ShareRequest, 
                          shared_by: str) -> bool:
        """Share a project with another user."""
        with get_db_session() as db:
            # Check if user owns the project or has admin access
            if not await self._has_admin_access(project_id, shared_by, db):
                return False
            
            # Check if share already exists
            existing_share = db.query(ProjectShare).filter(
                and_(
                    ProjectShare.project_id == project_id,
                    ProjectShare.shared_with_user_id == share_request.user_id
                )
            ).first()
            
            if existing_share:
                # Update existing share
                existing_share.permission_level = share_request.permission_level.value
                existing_share.expires_at = share_request.expires_at
                existing_share.shared_by = shared_by
            else:
                # Create new share
                new_share = ProjectShare(
                    project_id=project_id,
                    shared_with_user_id=share_request.user_id,
                    permission_level=share_request.permission_level.value,
                    shared_by=shared_by,
                    expires_at=share_request.expires_at or (
                        datetime.utcnow() + timedelta(days=self.default_share_expiry_days)
                    )
                )
                db.add(new_share)
            
            db.commit()
            
            # Send notification (placeholder for notification system)
            await self._send_share_notification(project_id, share_request.user_id, shared_by)
            
            return True
    
    async def revoke_project_access(self, project_id: int, user_id: str, revoked_by: str) -> bool:
        """Revoke project access for a user."""
        with get_db_session() as db:
            # Check if user has admin access
            if not await self._has_admin_access(project_id, revoked_by, db):
                return False
            
            # Find and remove the share
            share = db.query(ProjectShare).filter(
                and_(
                    ProjectShare.project_id == project_id,
                    ProjectShare.shared_with_user_id == user_id
                )
            ).first()
            
            if share:
                db.delete(share)
                db.commit()
                
                # Send notification
                await self._send_revoke_notification(project_id, user_id, revoked_by)
                
                return True
            
            return False
    
    async def update_permission(self, project_id: int, user_id: str, 
                              new_permission: SharePermission, updated_by: str) -> bool:
        """Update permission level for a shared user."""
        with get_db_session() as db:
            # Check if user has admin access
            if not await self._has_admin_access(project_id, updated_by, db):
                return False
            
            # Find and update the share
            share = db.query(ProjectShare).filter(
                and_(
                    ProjectShare.project_id == project_id,
                    ProjectShare.shared_with_user_id == user_id
                )
            ).first()
            
            if share:
                share.permission_level = new_permission.value
                db.commit()
                
                # Send notification
                await self._send_permission_update_notification(project_id, user_id, new_permission, updated_by)
                
                return True
            
            return False
    
    async def get_project_collaborators(self, project_id: int, user_id: str) -> List[Dict[str, Any]]:
        """Get list of project collaborators."""
        with get_db_session() as db:
            # Check if user has access to the project
            if not await self._has_project_access(project_id, user_id, db):
                return []
            
            # Get project owner
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project:
                return []
            
            collaborators = [{
                "user_id": project.owner_id,
                "permission_level": "owner",
                "shared_by": None,
                "shared_at": project.created_at,
                "expires_at": None,
                "is_owner": True
            }]
            
            # Get shared users
            shares = db.query(ProjectShare).filter(
                ProjectShare.project_id == project_id
            ).order_by(desc(ProjectShare.created_at)).all()
            
            for share in shares:
                # Check if share is still valid
                if share.expires_at and share.expires_at < datetime.utcnow():
                    continue
                
                collaborators.append({
                    "user_id": share.shared_with_user_id,
                    "permission_level": share.permission_level,
                    "shared_by": share.shared_by,
                    "shared_at": share.created_at,
                    "expires_at": share.expires_at,
                    "is_owner": False
                })
            
            return collaborators
    
    async def create_share_link(self, project_id: int, user_id: str, 
                              permission_level: SharePermission = SharePermission.VIEW,
                              expires_in_hours: int = 24) -> Optional[str]:
        """Create a shareable link for the project."""
        with get_db_session() as db:
            # Check if user has admin access
            if not await self._has_admin_access(project_id, user_id, db):
                return None
            
            # Generate unique share token
            share_token = str(uuid.uuid4())
            expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
            
            # Store share link info in project metadata
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project:
                return None
            
            if not project.project_metadata:
                project.project_metadata = {}
            
            if "share_links" not in project.project_metadata:
                project.project_metadata["share_links"] = {}
            
            project.project_metadata["share_links"][share_token] = {
                "permission_level": permission_level.value,
                "created_by": user_id,
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": expires_at.isoformat()
            }
            
            db.commit()
            
            # Return shareable URL (placeholder - would be actual URL in production)
            return f"https://scrollintel.com/shared/{share_token}"
    
    async def access_via_share_link(self, share_token: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Access project via share link."""
        with get_db_session() as db:
            # Find project with this share token
            projects = db.query(Project).filter(
                Project.project_metadata.contains({"share_links": {share_token: {}}})
            ).all()
            
            for project in projects:
                share_links = project.project_metadata.get("share_links", {})
                if share_token in share_links:
                    link_info = share_links[share_token]
                    
                    # Check if link is still valid
                    expires_at = datetime.fromisoformat(link_info["expires_at"])
                    if expires_at < datetime.utcnow():
                        continue
                    
                    # Grant temporary access
                    permission_level = SharePermission(link_info["permission_level"])
                    
                    # Create or update share record
                    existing_share = db.query(ProjectShare).filter(
                        and_(
                            ProjectShare.project_id == project.id,
                            ProjectShare.shared_with_user_id == user_id
                        )
                    ).first()
                    
                    if existing_share:
                        # Update existing share if new permission is higher
                        if self._is_higher_permission(permission_level, SharePermission(existing_share.permission_level)):
                            existing_share.permission_level = permission_level.value
                            existing_share.expires_at = expires_at
                    else:
                        # Create new share
                        new_share = ProjectShare(
                            project_id=project.id,
                            shared_with_user_id=user_id,
                            permission_level=permission_level.value,
                            shared_by=link_info["created_by"],
                            expires_at=expires_at
                        )
                        db.add(new_share)
                    
                    db.commit()
                    
                    return {
                        "project_id": project.id,
                        "project_name": project.name,
                        "permission_level": permission_level.value,
                        "expires_at": expires_at
                    }
            
            return None
    
    async def create_approval_workflow(self, project_id: int, approval_request: ApprovalRequest,
                                     requested_by: str) -> Optional[int]:
        """Create an approval workflow for a project."""
        with get_db_session() as db:
            # Check if user has access to the project
            if not await self._has_project_access(project_id, requested_by, db):
                return None
            
            # Create approval workflow
            approval = ApprovalWorkflow(
                project_id=project_id,
                workflow_name=approval_request.workflow_name,
                requested_by=requested_by,
                assigned_to=approval_request.assigned_to,
                due_date=approval_request.due_date
            )
            
            db.add(approval)
            db.commit()
            db.refresh(approval)
            
            # Send notification to assigned user
            await self._send_approval_request_notification(approval.id, approval_request.assigned_to)
            
            return approval.id
    
    async def respond_to_approval(self, approval_id: int, response: ApprovalResponse,
                                user_id: str) -> bool:
        """Respond to an approval request."""
        with get_db_session() as db:
            approval = db.query(ApprovalWorkflow).filter(
                and_(
                    ApprovalWorkflow.id == approval_id,
                    ApprovalWorkflow.assigned_to == user_id,
                    ApprovalWorkflow.status == ApprovalStatus.PENDING.value
                )
            ).first()
            
            if not approval:
                return False
            
            approval.status = response.status.value
            approval.feedback = response.feedback
            approval.updated_at = datetime.utcnow()
            
            db.commit()
            
            # Send notification to requester
            await self._send_approval_response_notification(approval_id, approval.requested_by, response.status)
            
            return True
    
    async def get_pending_approvals(self, user_id: str) -> List[Dict[str, Any]]:
        """Get pending approvals assigned to the user."""
        with get_db_session() as db:
            approvals = db.query(ApprovalWorkflow).filter(
                and_(
                    ApprovalWorkflow.assigned_to == user_id,
                    ApprovalWorkflow.status == ApprovalStatus.PENDING.value
                )
            ).order_by(desc(ApprovalWorkflow.created_at)).all()
            
            approval_list = []
            for approval in approvals:
                project = db.query(Project).filter(Project.id == approval.project_id).first()
                
                approval_list.append({
                    "id": approval.id,
                    "workflow_name": approval.workflow_name,
                    "project_id": approval.project_id,
                    "project_name": project.name if project else "Unknown",
                    "requested_by": approval.requested_by,
                    "due_date": approval.due_date,
                    "created_at": approval.created_at
                })
            
            return approval_list
    
    async def _has_project_access(self, project_id: int, user_id: str, db: Session) -> bool:
        """Check if user has access to project."""
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            return False
        
        # Owner has access
        if project.owner_id == user_id:
            return True
        
        # Check shared access
        share = db.query(ProjectShare).filter(
            and_(
                ProjectShare.project_id == project_id,
                ProjectShare.shared_with_user_id == user_id,
                or_(
                    ProjectShare.expires_at.is_(None),
                    ProjectShare.expires_at > datetime.utcnow()
                )
            )
        ).first()
        
        return share is not None
    
    async def _has_admin_access(self, project_id: int, user_id: str, db: Session) -> bool:
        """Check if user has admin access to project."""
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            return False
        
        # Owner has admin access
        if project.owner_id == user_id:
            return True
        
        # Check shared admin access
        share = db.query(ProjectShare).filter(
            and_(
                ProjectShare.project_id == project_id,
                ProjectShare.shared_with_user_id == user_id,
                ProjectShare.permission_level == SharePermission.ADMIN.value,
                or_(
                    ProjectShare.expires_at.is_(None),
                    ProjectShare.expires_at > datetime.utcnow()
                )
            )
        ).first()
        
        return share is not None
    
    def _is_higher_permission(self, perm1: SharePermission, perm2: SharePermission) -> bool:
        """Check if perm1 is higher than perm2."""
        permission_hierarchy = {
            SharePermission.VIEW: 1,
            SharePermission.COMMENT: 2,
            SharePermission.EDIT: 3,
            SharePermission.ADMIN: 4
        }
        
        return permission_hierarchy[perm1] > permission_hierarchy[perm2]
    
    async def _send_share_notification(self, project_id: int, user_id: str, shared_by: str):
        """Send notification when project is shared (placeholder)."""
        # In a real implementation, this would send email/push notifications
        print(f"Notification: Project {project_id} shared with {user_id} by {shared_by}")
    
    async def _send_revoke_notification(self, project_id: int, user_id: str, revoked_by: str):
        """Send notification when access is revoked (placeholder)."""
        print(f"Notification: Access to project {project_id} revoked for {user_id} by {revoked_by}")
    
    async def _send_permission_update_notification(self, project_id: int, user_id: str, 
                                                 new_permission: SharePermission, updated_by: str):
        """Send notification when permission is updated (placeholder)."""
        print(f"Notification: Permission for {user_id} on project {project_id} updated to {new_permission.value} by {updated_by}")
    
    async def _send_approval_request_notification(self, approval_id: int, assigned_to: str):
        """Send notification for approval request (placeholder)."""
        print(f"Notification: Approval request {approval_id} assigned to {assigned_to}")
    
    async def _send_approval_response_notification(self, approval_id: int, requested_by: str, status: ApprovalStatus):
        """Send notification for approval response (placeholder)."""
        print(f"Notification: Approval request {approval_id} {status.value} - notifying {requested_by}")