"""
Collaborative editing system for prompt management with real-time conflict resolution.
"""
import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
from sqlalchemy.orm import Session

from .prompt_version_control import PromptVersionControl
from .prompt_diff_merge import PromptDiffMerge
from ..models.prompt_models import ConflictResolutionStrategy
from ..models.prompt_models import AdvancedPromptTemplate, AdvancedPromptVersion
from ..models.database import Base


class EditOperation(Enum):
    """Types of edit operations."""
    INSERT = "insert"
    DELETE = "delete"
    REPLACE = "replace"
    MOVE = "move"


class LockType(Enum):
    """Types of editing locks."""
    READ = "read"
    WRITE = "write"
    EXCLUSIVE = "exclusive"


@dataclass
class EditEvent:
    """Represents an edit event in collaborative editing."""
    id: str
    prompt_id: str
    user_id: str
    operation: EditOperation
    position: int
    length: int
    content: str
    timestamp: datetime
    version: int


@dataclass
class EditLock:
    """Represents a lock on a prompt for editing."""
    prompt_id: str
    user_id: str
    lock_type: LockType
    acquired_at: datetime
    expires_at: datetime
    section_start: Optional[int] = None
    section_end: Optional[int] = None


@dataclass
class CollaborationSession:
    """Represents an active collaboration session."""
    session_id: str
    prompt_id: str
    participants: Set[str]
    created_at: datetime
    last_activity: datetime
    is_active: bool = True


class PromptCollaboration:
    """Manages collaborative editing of prompts with conflict resolution."""
    
    def __init__(self, db: Session):
        self.db = db
        self.version_control = PromptVersionControl(db)
        self.diff_merge = PromptDiffMerge()
        
        # In-memory stores for real-time collaboration
        self.active_sessions: Dict[str, CollaborationSession] = {}
        self.edit_locks: Dict[str, List[EditLock]] = {}
        self.pending_edits: Dict[str, List[EditEvent]] = {}
        self.user_cursors: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.lock_timeout = timedelta(minutes=5)
        self.session_timeout = timedelta(hours=2)
        self.conflict_resolution_timeout = timedelta(minutes=1)
    
    def start_collaboration_session(self, prompt_id: str, user_id: str) -> CollaborationSession:
        """Start or join a collaboration session."""
        session_id = f"{prompt_id}_{int(datetime.utcnow().timestamp())}"
        
        # Check if there's an existing active session
        existing_session = None
        for session in self.active_sessions.values():
            if session.prompt_id == prompt_id and session.is_active:
                existing_session = session
                break
        
        if existing_session:
            # Join existing session
            existing_session.participants.add(user_id)
            existing_session.last_activity = datetime.utcnow()
            return existing_session
        else:
            # Create new session
            session = CollaborationSession(
                session_id=session_id,
                prompt_id=prompt_id,
                participants={user_id},
                created_at=datetime.utcnow(),
                last_activity=datetime.utcnow()
            )
            
            self.active_sessions[session_id] = session
            return session
    
    def end_collaboration_session(self, session_id: str, user_id: str) -> bool:
        """End collaboration session for a user."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        session.participants.discard(user_id)
        
        # Release all locks held by this user
        self.release_user_locks(session.prompt_id, user_id)
        
        # End session if no participants left
        if not session.participants:
            session.is_active = False
            del self.active_sessions[session_id]
        
        return True
    
    def acquire_edit_lock(self, prompt_id: str, user_id: str, 
                         lock_type: LockType = LockType.WRITE,
                         section_start: Optional[int] = None,
                         section_end: Optional[int] = None) -> bool:
        """Acquire an edit lock on a prompt or section."""
        now = datetime.utcnow()
        
        # Clean up expired locks
        self._cleanup_expired_locks(prompt_id)
        
        # Check for conflicting locks
        if self._has_conflicting_locks(prompt_id, user_id, lock_type, section_start, section_end):
            return False
        
        # Create new lock
        lock = EditLock(
            prompt_id=prompt_id,
            user_id=user_id,
            lock_type=lock_type,
            acquired_at=now,
            expires_at=now + self.lock_timeout,
            section_start=section_start,
            section_end=section_end
        )
        
        if prompt_id not in self.edit_locks:
            self.edit_locks[prompt_id] = []
        
        self.edit_locks[prompt_id].append(lock)
        return True
    
    def release_edit_lock(self, prompt_id: str, user_id: str,
                         section_start: Optional[int] = None,
                         section_end: Optional[int] = None) -> bool:
        """Release a specific edit lock."""
        if prompt_id not in self.edit_locks:
            return False
        
        locks = self.edit_locks[prompt_id]
        for i, lock in enumerate(locks):
            if (lock.user_id == user_id and 
                lock.section_start == section_start and 
                lock.section_end == section_end):
                del locks[i]
                return True
        
        return False
    
    def release_user_locks(self, prompt_id: str, user_id: str) -> int:
        """Release all locks held by a user on a prompt."""
        if prompt_id not in self.edit_locks:
            return 0
        
        locks = self.edit_locks[prompt_id]
        initial_count = len(locks)
        
        self.edit_locks[prompt_id] = [
            lock for lock in locks if lock.user_id != user_id
        ]
        
        return initial_count - len(self.edit_locks[prompt_id])
    
    def apply_edit(self, prompt_id: str, user_id: str, edit_event: EditEvent) -> Dict[str, Any]:
        """Apply an edit operation with conflict detection."""
        # Check if user has appropriate lock
        if not self._user_has_write_access(prompt_id, user_id, edit_event.position):
            return {
                "success": False,
                "error": "No write access to this section",
                "requires_lock": True
            }
        
        # Get current prompt content
        template = self.db.query(AdvancedPromptTemplate).filter(
            AdvancedPromptTemplate.id == prompt_id
        ).first()
        
        if not template:
            return {
                "success": False,
                "error": "Prompt not found"
            }
        
        # Check for conflicts with pending edits
        conflicts = self._detect_edit_conflicts(prompt_id, edit_event)
        
        if conflicts:
            return {
                "success": False,
                "error": "Edit conflicts detected",
                "conflicts": conflicts,
                "requires_resolution": True
            }
        
        # Apply the edit
        try:
            new_content = self._apply_edit_operation(template.content, edit_event)
            
            # Create new version
            new_version = AdvancedPromptVersion(
                prompt_id=prompt_id,
                version=self.version_control._generate_next_version(prompt_id),
                content=new_content,
                changes=f"Collaborative edit by {user_id}: {edit_event.operation.value}",
                variables=template.variables,
                tags=template.tags,
                created_by=user_id
            )
            
            self.db.add(new_version)
            
            # Update template
            template.content = new_content
            template.updated_at = datetime.utcnow()
            
            self.db.commit()
            self.db.refresh(new_version)
            
            # Store edit event for conflict detection
            if prompt_id not in self.pending_edits:
                self.pending_edits[prompt_id] = []
            
            self.pending_edits[prompt_id].append(edit_event)
            
            # Broadcast edit to other collaborators
            self._broadcast_edit(prompt_id, user_id, edit_event, new_version)
            
            return {
                "success": True,
                "version_id": new_version.id,
                "new_content": new_content
            }
            
        except Exception as e:
            self.db.rollback()
            return {
                "success": False,
                "error": f"Failed to apply edit: {str(e)}"
            }
    
    def resolve_conflict(self, prompt_id: str, conflict_id: str, 
                        resolution: Dict[str, Any], resolved_by: str) -> Dict[str, Any]:
        """Resolve a collaborative editing conflict."""
        # In a real implementation, this would handle specific conflict resolution
        # For now, we'll implement a basic resolution mechanism
        
        template = self.db.query(AdvancedPromptTemplate).filter(
            AdvancedPromptTemplate.id == prompt_id
        ).first()
        
        if not template:
            return {"success": False, "error": "Prompt not found"}
        
        try:
            # Apply resolution
            if resolution.get("strategy") == "accept_all":
                # Accept all pending changes
                resolved_content = resolution.get("content", template.content)
            elif resolution.get("strategy") == "manual":
                # Use manually resolved content
                resolved_content = resolution.get("resolved_content", template.content)
            else:
                return {"success": False, "error": "Invalid resolution strategy"}
            
            # Create resolution version
            resolution_version = AdvancedPromptVersion(
                prompt_id=prompt_id,
                version=self.version_control._generate_next_version(prompt_id),
                content=resolved_content,
                changes=f"Conflict resolution by {resolved_by}",
                variables=template.variables,
                tags=template.tags,
                created_by=resolved_by
            )
            
            self.db.add(resolution_version)
            
            # Update template
            template.content = resolved_content
            template.updated_at = datetime.utcnow()
            
            self.db.commit()
            self.db.refresh(resolution_version)
            
            # Clear pending edits for this prompt
            if prompt_id in self.pending_edits:
                del self.pending_edits[prompt_id]
            
            # Broadcast resolution to collaborators
            self._broadcast_conflict_resolution(prompt_id, resolved_by, resolution_version)
            
            return {
                "success": True,
                "version_id": resolution_version.id,
                "resolved_content": resolved_content
            }
            
        except Exception as e:
            self.db.rollback()
            return {"success": False, "error": f"Failed to resolve conflict: {str(e)}"}
    
    def get_collaboration_status(self, prompt_id: str) -> Dict[str, Any]:
        """Get current collaboration status for a prompt."""
        # Find active session
        active_session = None
        for session in self.active_sessions.values():
            if session.prompt_id == prompt_id and session.is_active:
                active_session = session
                break
        
        # Get current locks
        current_locks = self.edit_locks.get(prompt_id, [])
        self._cleanup_expired_locks(prompt_id)
        
        # Get pending edits
        pending_edits = self.pending_edits.get(prompt_id, [])
        
        return {
            "has_active_session": active_session is not None,
            "participants": list(active_session.participants) if active_session else [],
            "active_locks": len(current_locks),
            "pending_edits": len(pending_edits),
            "last_activity": active_session.last_activity.isoformat() if active_session else None,
            "conflicts_detected": len(self._detect_all_conflicts(prompt_id)) > 0
        }
    
    def get_user_cursor_position(self, prompt_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user's cursor position in collaborative editing."""
        user_key = f"{prompt_id}_{user_id}"
        return self.user_cursors.get(user_key)
    
    def update_user_cursor(self, prompt_id: str, user_id: str, 
                          position: int, selection_start: Optional[int] = None,
                          selection_end: Optional[int] = None) -> None:
        """Update user's cursor position."""
        user_key = f"{prompt_id}_{user_id}"
        self.user_cursors[user_key] = {
            "position": position,
            "selection_start": selection_start,
            "selection_end": selection_end,
            "updated_at": datetime.utcnow().isoformat()
        }
    
    def _cleanup_expired_locks(self, prompt_id: str) -> None:
        """Remove expired locks."""
        if prompt_id not in self.edit_locks:
            return
        
        now = datetime.utcnow()
        self.edit_locks[prompt_id] = [
            lock for lock in self.edit_locks[prompt_id]
            if lock.expires_at > now
        ]
    
    def _has_conflicting_locks(self, prompt_id: str, user_id: str, 
                              lock_type: LockType,
                              section_start: Optional[int] = None,
                              section_end: Optional[int] = None) -> bool:
        """Check if there are conflicting locks."""
        if prompt_id not in self.edit_locks:
            return False
        
        for lock in self.edit_locks[prompt_id]:
            if lock.user_id == user_id:
                continue  # User's own locks don't conflict
            
            # Check for lock type conflicts
            if lock_type == LockType.EXCLUSIVE or lock.lock_type == LockType.EXCLUSIVE:
                return True
            
            if lock_type == LockType.WRITE and lock.lock_type == LockType.WRITE:
                # Check for section overlap
                if self._sections_overlap(
                    section_start, section_end,
                    lock.section_start, lock.section_end
                ):
                    return True
        
        return False
    
    def _sections_overlap(self, start1: Optional[int], end1: Optional[int],
                         start2: Optional[int], end2: Optional[int]) -> bool:
        """Check if two sections overlap."""
        # If either section is None (whole document), they overlap
        if start1 is None or start2 is None:
            return True
        
        # Check for overlap
        return not (end1 < start2 or end2 < start1)
    
    def _user_has_write_access(self, prompt_id: str, user_id: str, position: int) -> bool:
        """Check if user has write access to a specific position."""
        if prompt_id not in self.edit_locks:
            return True  # No locks, anyone can edit
        
        for lock in self.edit_locks[prompt_id]:
            if lock.user_id == user_id and lock.lock_type in [LockType.WRITE, LockType.EXCLUSIVE]:
                # Check if position is within locked section
                if lock.section_start is None:  # Whole document lock
                    return True
                if lock.section_start <= position <= (lock.section_end or float('inf')):
                    return True
        
        return False
    
    def _detect_edit_conflicts(self, prompt_id: str, edit_event: EditEvent) -> List[Dict[str, Any]]:
        """Detect conflicts with pending edits."""
        if prompt_id not in self.pending_edits:
            return []
        
        conflicts = []
        for pending_edit in self.pending_edits[prompt_id]:
            if pending_edit.user_id == edit_event.user_id:
                continue  # User's own edits don't conflict
            
            # Check for position conflicts
            if self._edits_conflict(pending_edit, edit_event):
                conflicts.append({
                    "edit_id": pending_edit.id,
                    "user_id": pending_edit.user_id,
                    "operation": pending_edit.operation.value,
                    "position": pending_edit.position,
                    "timestamp": pending_edit.timestamp.isoformat()
                })
        
        return conflicts
    
    def _edits_conflict(self, edit1: EditEvent, edit2: EditEvent) -> bool:
        """Check if two edits conflict."""
        # Simple conflict detection based on position overlap
        edit1_end = edit1.position + edit1.length
        edit2_end = edit2.position + edit2.length
        
        return not (edit1_end < edit2.position or edit2_end < edit1.position)
    
    def _detect_all_conflicts(self, prompt_id: str) -> List[Dict[str, Any]]:
        """Detect all conflicts for a prompt."""
        if prompt_id not in self.pending_edits:
            return []
        
        conflicts = []
        edits = self.pending_edits[prompt_id]
        
        for i, edit1 in enumerate(edits):
            for edit2 in edits[i+1:]:
                if edit1.user_id != edit2.user_id and self._edits_conflict(edit1, edit2):
                    conflicts.append({
                        "edit1": asdict(edit1),
                        "edit2": asdict(edit2),
                        "conflict_type": "position_overlap"
                    })
        
        return conflicts
    
    def _apply_edit_operation(self, content: str, edit_event: EditEvent) -> str:
        """Apply an edit operation to content."""
        if edit_event.operation == EditOperation.INSERT:
            return (content[:edit_event.position] + 
                   edit_event.content + 
                   content[edit_event.position:])
        
        elif edit_event.operation == EditOperation.DELETE:
            return (content[:edit_event.position] + 
                   content[edit_event.position + edit_event.length:])
        
        elif edit_event.operation == EditOperation.REPLACE:
            return (content[:edit_event.position] + 
                   edit_event.content + 
                   content[edit_event.position + edit_event.length:])
        
        else:  # MOVE operation would be more complex
            return content
    
    def _broadcast_edit(self, prompt_id: str, user_id: str, 
                       edit_event: EditEvent, version: AdvancedPromptVersion) -> None:
        """Broadcast edit to other collaborators."""
        # In a real implementation, this would use WebSockets or similar
        # to notify other users of the edit
        pass
    
    def _broadcast_conflict_resolution(self, prompt_id: str, resolved_by: str,
                                     version: AdvancedPromptVersion) -> None:
        """Broadcast conflict resolution to collaborators."""
        # In a real implementation, this would notify all collaborators
        # that conflicts have been resolved
        pass