"""
API routes for collaboration and sharing functionality.
"""

from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from fastapi.security import HTTPBearer
from pydantic import BaseModel

from scrollintel.engines.visual_generation.collaboration.project_manager import ProjectManager
from scrollintel.engines.visual_generation.collaboration.sharing_manager import SharingManager
from scrollintel.engines.visual_generation.collaboration.comment_system import CommentSystem
from scrollintel.engines.visual_generation.collaboration.version_control import VersionControl
from scrollintel.models.collaboration_models import (
    ProjectCreateRequest, ProjectUpdateRequest, ShareRequest, CommentRequest,
    ApprovalRequest, ApprovalResponse, SharePermission, ApprovalStatus
)
from scrollintel.security.auth import get_current_user


router = APIRouter(prefix="/api/v1/collaboration", tags=["collaboration"])
security = HTTPBearer()

# Initialize managers
project_manager = ProjectManager()
sharing_manager = SharingManager()
comment_system = CommentSystem()
version_control = VersionControl()


# Pydantic models for request/response
class ProjectListResponse(BaseModel):
    projects: List[dict]
    total_count: int
    page: int
    page_size: int


class CollaboratorResponse(BaseModel):
    collaborators: List[dict]


class CommentListResponse(BaseModel):
    comments: List[dict]


class VersionHistoryResponse(BaseModel):
    versions: List[dict]


class VersionComparisonResponse(BaseModel):
    comparison: dict


# Project Management Routes
@router.post("/projects")
async def create_project(
    request: ProjectCreateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create a new project."""
    try:
        project = await project_manager.create_project(request, current_user["user_id"])
        return {"success": True, "project": project}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}")
async def get_project(
    project_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Get a project by ID."""
    project = await project_manager.get_project(project_id, current_user["user_id"])
    if not project:
        raise HTTPException(status_code=404, detail="Project not found or access denied")
    
    return {"success": True, "project": project}


@router.put("/projects/{project_id}")
async def update_project(
    project_id: int,
    request: ProjectUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Update a project."""
    project = await project_manager.update_project(project_id, request, current_user["user_id"])
    if not project:
        raise HTTPException(status_code=404, detail="Project not found or access denied")
    
    return {"success": True, "project": project}


@router.delete("/projects/{project_id}")
async def delete_project(
    project_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Delete a project."""
    success = await project_manager.delete_project(project_id, current_user["user_id"])
    if not success:
        raise HTTPException(status_code=404, detail="Project not found or access denied")
    
    return {"success": True, "message": "Project deleted successfully"}


@router.get("/projects", response_model=ProjectListResponse)
async def list_projects(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None),
    tags: Optional[List[str]] = Query(None),
    current_user: dict = Depends(get_current_user)
):
    """List projects accessible to the user."""
    projects, total_count = await project_manager.list_projects(
        current_user["user_id"], page, page_size, search, tags
    )
    
    return ProjectListResponse(
        projects=projects,
        total_count=total_count,
        page=page,
        page_size=page_size
    )


@router.get("/projects/{project_id}/content")
async def get_project_content(
    project_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Get content items in a project."""
    content_items = await project_manager.get_content_items(project_id, current_user["user_id"])
    return {"success": True, "content_items": content_items}


@router.post("/projects/{project_id}/content")
async def add_content_item(
    project_id: int,
    name: str,
    content_type: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Add a content item to a project."""
    # Save uploaded file (simplified - in production, use proper file handling)
    file_path = f"temp/{file.filename}"
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    content_item = await project_manager.add_content_item(
        project_id, name, content_type, file_path, current_user["user_id"]
    )
    
    if not content_item:
        raise HTTPException(status_code=404, detail="Project not found or access denied")
    
    return {"success": True, "content_item": content_item}


@router.post("/projects/{project_id}/organize")
async def organize_project_content(
    project_id: int,
    organization_rules: dict,
    current_user: dict = Depends(get_current_user)
):
    """Organize content items in a project."""
    success = await project_manager.organize_content(
        project_id, organization_rules, current_user["user_id"]
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Project not found or access denied")
    
    return {"success": True, "message": "Content organized successfully"}


@router.get("/projects/{project_id}/search")
async def search_project_content(
    project_id: int,
    query: str = Query(...),
    filters: Optional[dict] = None,
    current_user: dict = Depends(get_current_user)
):
    """Search content items within a project."""
    content_items = await project_manager.search_content(
        project_id, query, current_user["user_id"], filters
    )
    
    return {"success": True, "content_items": content_items}


@router.get("/metrics")
async def get_collaboration_metrics(
    current_user: dict = Depends(get_current_user)
):
    """Get collaboration metrics for the user."""
    metrics = await project_manager.get_collaboration_metrics(current_user["user_id"])
    return {"success": True, "metrics": metrics}


# Sharing and Permissions Routes
@router.post("/projects/{project_id}/share")
async def share_project(
    project_id: int,
    request: ShareRequest,
    current_user: dict = Depends(get_current_user)
):
    """Share a project with another user."""
    success = await sharing_manager.share_project(
        project_id, request, current_user["user_id"]
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Project not found or access denied")
    
    return {"success": True, "message": "Project shared successfully"}


@router.delete("/projects/{project_id}/share/{user_id}")
async def revoke_project_access(
    project_id: int,
    user_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Revoke project access for a user."""
    success = await sharing_manager.revoke_project_access(
        project_id, user_id, current_user["user_id"]
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Project not found or access denied")
    
    return {"success": True, "message": "Access revoked successfully"}


@router.put("/projects/{project_id}/share/{user_id}")
async def update_user_permission(
    project_id: int,
    user_id: str,
    permission: SharePermission,
    current_user: dict = Depends(get_current_user)
):
    """Update permission level for a shared user."""
    success = await sharing_manager.update_permission(
        project_id, user_id, permission, current_user["user_id"]
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Project not found or access denied")
    
    return {"success": True, "message": "Permission updated successfully"}


@router.get("/projects/{project_id}/collaborators", response_model=CollaboratorResponse)
async def get_project_collaborators(
    project_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Get list of project collaborators."""
    collaborators = await sharing_manager.get_project_collaborators(
        project_id, current_user["user_id"]
    )
    
    return CollaboratorResponse(collaborators=collaborators)


@router.post("/projects/{project_id}/share-link")
async def create_share_link(
    project_id: int,
    permission_level: SharePermission = SharePermission.VIEW,
    expires_in_hours: int = 24,
    current_user: dict = Depends(get_current_user)
):
    """Create a shareable link for the project."""
    share_link = await sharing_manager.create_share_link(
        project_id, current_user["user_id"], permission_level, expires_in_hours
    )
    
    if not share_link:
        raise HTTPException(status_code=404, detail="Project not found or access denied")
    
    return {"success": True, "share_link": share_link}


@router.post("/share-link/{share_token}")
async def access_via_share_link(
    share_token: str,
    current_user: dict = Depends(get_current_user)
):
    """Access project via share link."""
    access_info = await sharing_manager.access_via_share_link(
        share_token, current_user["user_id"]
    )
    
    if not access_info:
        raise HTTPException(status_code=404, detail="Invalid or expired share link")
    
    return {"success": True, "access_info": access_info}


# Approval Workflow Routes
@router.post("/projects/{project_id}/approvals")
async def create_approval_workflow(
    project_id: int,
    request: ApprovalRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create an approval workflow for a project."""
    approval_id = await sharing_manager.create_approval_workflow(
        project_id, request, current_user["user_id"]
    )
    
    if not approval_id:
        raise HTTPException(status_code=404, detail="Project not found or access denied")
    
    return {"success": True, "approval_id": approval_id}


@router.post("/approvals/{approval_id}/respond")
async def respond_to_approval(
    approval_id: int,
    response: ApprovalResponse,
    current_user: dict = Depends(get_current_user)
):
    """Respond to an approval request."""
    success = await sharing_manager.respond_to_approval(
        approval_id, response, current_user["user_id"]
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Approval not found or access denied")
    
    return {"success": True, "message": "Approval response recorded"}


@router.get("/approvals/pending")
async def get_pending_approvals(
    current_user: dict = Depends(get_current_user)
):
    """Get pending approvals assigned to the user."""
    approvals = await sharing_manager.get_pending_approvals(current_user["user_id"])
    return {"success": True, "approvals": approvals}


# Comment System Routes
@router.post("/projects/{project_id}/comments")
async def add_project_comment(
    project_id: int,
    request: CommentRequest,
    current_user: dict = Depends(get_current_user)
):
    """Add a comment to a project."""
    comment = await comment_system.add_project_comment(
        project_id, request, current_user["user_id"]
    )
    
    if not comment:
        raise HTTPException(status_code=404, detail="Project not found or access denied")
    
    return {"success": True, "comment": comment}


@router.post("/content/{content_item_id}/comments")
async def add_content_comment(
    content_item_id: int,
    request: CommentRequest,
    current_user: dict = Depends(get_current_user)
):
    """Add a comment to a content item."""
    comment = await comment_system.add_content_comment(
        content_item_id, request, current_user["user_id"]
    )
    
    if not comment:
        raise HTTPException(status_code=404, detail="Content item not found or access denied")
    
    return {"success": True, "comment": comment}


@router.get("/projects/{project_id}/comments", response_model=CommentListResponse)
async def get_project_comments(
    project_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    current_user: dict = Depends(get_current_user)
):
    """Get comments for a project."""
    comments = await comment_system.get_project_comments(
        project_id, current_user["user_id"], page, page_size
    )
    
    return CommentListResponse(comments=comments)


@router.get("/content/{content_item_id}/comments", response_model=CommentListResponse)
async def get_content_comments(
    content_item_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Get comments for a content item."""
    comments = await comment_system.get_content_comments(
        content_item_id, current_user["user_id"]
    )
    
    return CommentListResponse(comments=comments)


@router.put("/comments/{comment_id}")
async def update_comment(
    comment_id: int,
    new_content: str,
    comment_type: str = Query("project", regex="^(project|content)$"),
    current_user: dict = Depends(get_current_user)
):
    """Update a comment."""
    success = await comment_system.update_comment(
        comment_id, new_content, current_user["user_id"], comment_type
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Comment not found or access denied")
    
    return {"success": True, "message": "Comment updated successfully"}


@router.delete("/comments/{comment_id}")
async def delete_comment(
    comment_id: int,
    comment_type: str = Query("project", regex="^(project|content)$"),
    current_user: dict = Depends(get_current_user)
):
    """Delete a comment."""
    success = await comment_system.delete_comment(
        comment_id, current_user["user_id"], comment_type
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Comment not found or access denied")
    
    return {"success": True, "message": "Comment deleted successfully"}


@router.get("/projects/{project_id}/comments/statistics")
async def get_comment_statistics(
    project_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Get comment statistics for a project."""
    statistics = await comment_system.get_comment_statistics(
        project_id, current_user["user_id"]
    )
    
    return {"success": True, "statistics": statistics}


@router.get("/comments/search")
async def search_comments(
    project_id: int = Query(...),
    query: str = Query(...),
    content_item_id: Optional[int] = Query(None),
    current_user: dict = Depends(get_current_user)
):
    """Search comments within a project."""
    comments = await comment_system.search_comments(
        project_id, query, current_user["user_id"], content_item_id
    )
    
    return {"success": True, "comments": comments}


@router.get("/comments/mentions")
async def get_user_mentions(
    current_user: dict = Depends(get_current_user)
):
    """Get comments that mention the user."""
    mentions = await comment_system.get_user_mentions(current_user["user_id"])
    return {"success": True, "mentions": mentions}


# Version Control Routes
@router.post("/content/{content_item_id}/versions")
async def create_version(
    content_item_id: int,
    change_description: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Create a new version of a content item."""
    # Save uploaded file (simplified - in production, use proper file handling)
    file_path = f"temp/{file.filename}"
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    version = await version_control.create_version(
        content_item_id, file_path, current_user["user_id"], change_description
    )
    
    if not version:
        raise HTTPException(status_code=404, detail="Content item not found or access denied")
    
    return {"success": True, "version": version}


@router.get("/content/{content_item_id}/versions", response_model=VersionHistoryResponse)
async def get_version_history(
    content_item_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Get version history for a content item."""
    versions = await version_control.get_version_history(
        content_item_id, current_user["user_id"]
    )
    
    return VersionHistoryResponse(versions=versions)


@router.get("/content/{content_item_id}/versions/{version_number}")
async def get_version(
    content_item_id: int,
    version_number: int,
    current_user: dict = Depends(get_current_user)
):
    """Get a specific version of a content item."""
    version = await version_control.get_version(
        content_item_id, version_number, current_user["user_id"]
    )
    
    if not version:
        raise HTTPException(status_code=404, detail="Version not found or access denied")
    
    return {"success": True, "version": version}


@router.post("/content/{content_item_id}/versions/{version_number}/revert")
async def revert_to_version(
    content_item_id: int,
    version_number: int,
    revert_description: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Revert content item to a specific version."""
    success = await version_control.revert_to_version(
        content_item_id, version_number, current_user["user_id"], revert_description
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Version not found or access denied")
    
    return {"success": True, "message": "Reverted to version successfully"}


@router.get("/content/{content_item_id}/versions/compare", response_model=VersionComparisonResponse)
async def compare_versions(
    content_item_id: int,
    version1: int = Query(...),
    version2: int = Query(...),
    current_user: dict = Depends(get_current_user)
):
    """Compare two versions of a content item."""
    comparison = await version_control.compare_versions(
        content_item_id, version1, version2, current_user["user_id"]
    )
    
    if not comparison:
        raise HTTPException(status_code=404, detail="Versions not found or access denied")
    
    return VersionComparisonResponse(comparison=comparison)


@router.get("/content/{content_item_id}/versions/statistics")
async def get_version_statistics(
    content_item_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Get statistics for version history."""
    statistics = await version_control.get_version_statistics(
        content_item_id, current_user["user_id"]
    )
    
    if not statistics:
        raise HTTPException(status_code=404, detail="Content item not found or access denied")
    
    return {"success": True, "statistics": statistics}


@router.delete("/content/{content_item_id}/versions/{version_number}")
async def delete_version(
    content_item_id: int,
    version_number: int,
    current_user: dict = Depends(get_current_user)
):
    """Delete a specific version."""
    success = await version_control.delete_version(
        content_item_id, version_number, current_user["user_id"]
    )
    
    if not success:
        raise HTTPException(status_code=400, detail="Cannot delete version or access denied")
    
    return {"success": True, "message": "Version deleted successfully"}


@router.post("/content/{content_item_id}/branches")
async def create_branch(
    content_item_id: int,
    branch_name: str,
    from_version: int,
    current_user: dict = Depends(get_current_user)
):
    """Create a branch from a specific version."""
    branch = await version_control.create_branch(
        content_item_id, branch_name, from_version, current_user["user_id"]
    )
    
    if not branch:
        raise HTTPException(status_code=404, detail="Content item not found or access denied")
    
    return {"success": True, "branch": branch}