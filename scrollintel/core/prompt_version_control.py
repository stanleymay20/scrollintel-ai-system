"""
Git-like version control system for prompt management.
"""
import hashlib
import difflib
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from ..models.prompt_models import (
    AdvancedPromptTemplate, AdvancedPromptVersion, PromptBranch, 
    PromptCommit, PromptMergeRequest, PromptVersionTag,
    VersionControlAction, ConflictResolutionStrategy
)
from ..models.database import Base


class PromptVersionControl:
    """Git-like version control system for prompts."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_branch(self, prompt_id: str, branch_name: str, 
                     parent_branch_name: str = "main", 
                     description: str = "", created_by: str = "system") -> PromptBranch:
        """Create a new branch from an existing branch."""
        # Check if branch already exists
        existing_branch = self.db.query(PromptBranch).filter(
            and_(
                PromptBranch.prompt_id == prompt_id,
                PromptBranch.name == branch_name,
                PromptBranch.is_active == True
            )
        ).first()
        
        if existing_branch:
            if branch_name == "main" and parent_branch_name == "main":
                # Return existing main branch
                return existing_branch
            else:
                raise ValueError(f"Branch '{branch_name}' already exists")
        
        # Get parent branch
        parent_branch = None
        if branch_name != "main":  # Don't look for parent when creating main branch
            parent_branch = self.db.query(PromptBranch).filter(
                and_(
                    PromptBranch.prompt_id == prompt_id,
                    PromptBranch.name == parent_branch_name,
                    PromptBranch.is_active == True
                )
            ).first()
            
            if not parent_branch:
                # Create main branch if it doesn't exist
                if parent_branch_name == "main":
                    parent_branch = self._create_main_branch(prompt_id, created_by)
                else:
                    raise ValueError(f"Parent branch '{parent_branch_name}' not found")
        
        # Create new branch
        new_branch = PromptBranch(
            prompt_id=prompt_id,
            name=branch_name,
            description=description,
            parent_branch_id=parent_branch.id if parent_branch else None,
            head_version_id=parent_branch.head_version_id if parent_branch else None,
            is_protected=(branch_name == "main"),
            created_by=created_by
        )
        
        self.db.add(new_branch)
        self.db.commit()
        self.db.refresh(new_branch)
        
        return new_branch
    
    def commit_changes(self, branch_id: str, version_id: str, 
                      message: str, author: str) -> PromptCommit:
        """Create a commit for the changes."""
        branch = self.db.query(PromptBranch).filter(PromptBranch.id == branch_id).first()
        if not branch:
            raise ValueError(f"Branch with id '{branch_id}' not found")
        
        version = self.db.query(AdvancedPromptVersion).filter(
            AdvancedPromptVersion.id == version_id
        ).first()
        if not version:
            raise ValueError(f"Version with id '{version_id}' not found")
        
        # Get parent commit (latest commit in branch)
        parent_commit = self.db.query(PromptCommit).filter(
            PromptCommit.branch_id == branch_id
        ).order_by(PromptCommit.committed_at.desc()).first()
        
        # Generate commit hash
        commit_hash = self._generate_commit_hash(version_id, message, author)
        
        # Create commit
        commit = PromptCommit(
            branch_id=branch_id,
            version_id=version_id,
            parent_commit_id=parent_commit.id if parent_commit else None,
            commit_hash=commit_hash,
            message=message,
            author=author,
            committer=author
        )
        
        self.db.add(commit)
        
        # Update branch head
        branch.head_version_id = version_id
        
        self.db.commit()
        self.db.refresh(commit)
        
        return commit
    
    def create_merge_request(self, source_branch_id: str, target_branch_id: str,
                           title: str, description: str = "", 
                           author: str = "system") -> PromptMergeRequest:
        """Create a merge request between branches."""
        # Validate branches exist
        source_branch = self.db.query(PromptBranch).filter(
            PromptBranch.id == source_branch_id
        ).first()
        target_branch = self.db.query(PromptBranch).filter(
            PromptBranch.id == target_branch_id
        ).first()
        
        if not source_branch or not target_branch:
            raise ValueError("Source or target branch not found")
        
        if source_branch.prompt_id != target_branch.prompt_id:
            raise ValueError("Branches must belong to the same prompt")
        
        # Check for conflicts
        conflicts = self._detect_conflicts(source_branch_id, target_branch_id)
        
        merge_request = PromptMergeRequest(
            source_branch_id=source_branch_id,
            target_branch_id=target_branch_id,
            title=title,
            description=description,
            author=author,
            conflicts=conflicts
        )
        
        self.db.add(merge_request)
        self.db.commit()
        self.db.refresh(merge_request)
        
        return merge_request
    
    def merge_branches(self, merge_request_id: str, merged_by: str,
                      strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.AUTO_MERGE) -> bool:
        """Merge source branch into target branch."""
        merge_request = self.db.query(PromptMergeRequest).filter(
            PromptMergeRequest.id == merge_request_id
        ).first()
        
        if not merge_request:
            raise ValueError(f"Merge request '{merge_request_id}' not found")
        
        if merge_request.status != "open":
            raise ValueError(f"Merge request is not open (status: {merge_request.status})")
        
        source_branch = merge_request.source_branch
        target_branch = merge_request.target_branch
        
        # Handle conflicts based on strategy
        if merge_request.conflicts and strategy == ConflictResolutionStrategy.MANUAL:
            raise ValueError("Manual conflict resolution required")
        
        # Perform merge
        merged_version = self._perform_merge(source_branch, target_branch, strategy, merged_by)
        
        # Create merge commit
        merge_commit = self.commit_changes(
            target_branch.id,
            merged_version.id,
            f"Merge branch '{source_branch.name}' into '{target_branch.name}'",
            merged_by
        )
        
        # Update merge request
        merge_request.status = "merged"
        merge_request.merged_at = datetime.utcnow()
        merge_request.merged_by = merged_by
        
        self.db.commit()
        
        return True
    
    def create_tag(self, version_id: str, tag_name: str, 
                  description: str = "", tag_type: str = "release",
                  created_by: str = "system") -> PromptVersionTag:
        """Create a tag for a specific version."""
        version = self.db.query(AdvancedPromptVersion).filter(
            AdvancedPromptVersion.id == version_id
        ).first()
        
        if not version:
            raise ValueError(f"Version '{version_id}' not found")
        
        # Check if tag already exists
        existing_tag = self.db.query(PromptVersionTag).filter(
            PromptVersionTag.name == tag_name
        ).first()
        
        if existing_tag:
            raise ValueError(f"Tag '{tag_name}' already exists")
        
        tag = PromptVersionTag(
            version_id=version_id,
            name=tag_name,
            description=description,
            tag_type=tag_type,
            created_by=created_by
        )
        
        self.db.add(tag)
        self.db.commit()
        self.db.refresh(tag)
        
        return tag
    
    def rollback_to_version(self, prompt_id: str, target_version_id: str,
                           branch_name: str = "main", rolled_back_by: str = "system") -> AdvancedPromptVersion:
        """Rollback prompt to a specific version."""
        target_version = self.db.query(AdvancedPromptVersion).filter(
            and_(
                AdvancedPromptVersion.id == target_version_id,
                AdvancedPromptVersion.prompt_id == prompt_id
            )
        ).first()
        
        if not target_version:
            raise ValueError(f"Target version '{target_version_id}' not found")
        
        # Get current template
        template = self.db.query(AdvancedPromptTemplate).filter(
            AdvancedPromptTemplate.id == prompt_id
        ).first()
        
        if not template:
            raise ValueError(f"Prompt template '{prompt_id}' not found")
        
        # Create new version with rollback content
        rollback_version = AdvancedPromptVersion(
            prompt_id=prompt_id,
            version=self._generate_next_version(prompt_id),
            content=target_version.content,
            changes=f"Rollback to version {target_version.version}",
            variables=target_version.variables,
            tags=target_version.tags,
            created_by=rolled_back_by
        )
        
        self.db.add(rollback_version)
        
        # Update template content
        template.content = target_version.content
        template.variables = target_version.variables
        template.tags = target_version.tags
        template.updated_at = datetime.utcnow()
        
        # Create commit for rollback
        branch = self.db.query(PromptBranch).filter(
            and_(
                PromptBranch.prompt_id == prompt_id,
                PromptBranch.name == branch_name
            )
        ).first()
        
        if branch:
            self.commit_changes(
                branch.id,
                rollback_version.id,
                f"Rollback to version {target_version.version}",
                rolled_back_by
            )
        
        self.db.commit()
        self.db.refresh(rollback_version)
        
        return rollback_version
    
    def get_diff(self, version1_id: str, version2_id: str) -> Dict[str, Any]:
        """Get diff between two versions."""
        version1 = self.db.query(AdvancedPromptVersion).filter(
            AdvancedPromptVersion.id == version1_id
        ).first()
        version2 = self.db.query(AdvancedPromptVersion).filter(
            AdvancedPromptVersion.id == version2_id
        ).first()
        
        if not version1 or not version2:
            raise ValueError("One or both versions not found")
        
        # Generate content diff
        content_diff = list(difflib.unified_diff(
            version1.content.splitlines(keepends=True),
            version2.content.splitlines(keepends=True),
            fromfile=f"version_{version1.version}",
            tofile=f"version_{version2.version}",
            lineterm=""
        ))
        
        # Compare variables
        vars1 = set(str(var) for var in (version1.variables or []))
        vars2 = set(str(var) for var in (version2.variables or []))
        
        variables_diff = {
            "added": list(vars2 - vars1),
            "removed": list(vars1 - vars2),
            "common": list(vars1 & vars2)
        }
        
        # Compare tags
        tags1 = set(version1.tags or [])
        tags2 = set(version2.tags or [])
        
        tags_diff = {
            "added": list(tags2 - tags1),
            "removed": list(tags1 - tags2),
            "common": list(tags1 & tags2)
        }
        
        return {
            "content_diff": content_diff,
            "variables_diff": variables_diff,
            "tags_diff": tags_diff,
            "version1": version1.to_dict(),
            "version2": version2.to_dict()
        }
    
    def get_branch_history(self, branch_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get commit history for a branch."""
        commits = self.db.query(PromptCommit).filter(
            PromptCommit.branch_id == branch_id
        ).order_by(PromptCommit.committed_at.desc()).limit(limit).all()
        
        history = []
        for commit in commits:
            commit_data = commit.to_dict()
            commit_data["version"] = commit.version.to_dict() if commit.version else None
            history.append(commit_data)
        
        return history
    
    def _create_main_branch(self, prompt_id: str, created_by: str) -> PromptBranch:
        """Create the main branch for a prompt."""
        # Get latest version
        latest_version = self.db.query(AdvancedPromptVersion).filter(
            AdvancedPromptVersion.prompt_id == prompt_id
        ).order_by(AdvancedPromptVersion.created_at.desc()).first()
        
        main_branch = PromptBranch(
            prompt_id=prompt_id,
            name="main",
            description="Main development branch",
            head_version_id=latest_version.id if latest_version else None,
            is_protected=True,
            created_by=created_by
        )
        
        self.db.add(main_branch)
        self.db.commit()
        self.db.refresh(main_branch)
        
        return main_branch
    
    def _generate_commit_hash(self, version_id: str, message: str, author: str) -> str:
        """Generate a unique hash for the commit."""
        content = f"{version_id}{message}{author}{datetime.utcnow().isoformat()}"
        return hashlib.sha1(content.encode()).hexdigest()
    
    def _generate_next_version(self, prompt_id: str) -> str:
        """Generate the next version number."""
        latest_version = self.db.query(AdvancedPromptVersion).filter(
            AdvancedPromptVersion.prompt_id == prompt_id
        ).order_by(AdvancedPromptVersion.created_at.desc()).first()
        
        if not latest_version:
            return "1.0.0"
        
        # Simple version increment (major.minor.patch)
        try:
            parts = latest_version.version.split(".")
            if len(parts) == 3:
                major, minor, patch = map(int, parts)
                return f"{major}.{minor}.{patch + 1}"
        except ValueError:
            pass
        
        # Fallback to timestamp-based version
        return f"1.0.{int(datetime.utcnow().timestamp())}"
    
    def _detect_conflicts(self, source_branch_id: str, target_branch_id: str) -> List[str]:
        """Detect potential merge conflicts between branches."""
        source_branch = self.db.query(PromptBranch).filter(
            PromptBranch.id == source_branch_id
        ).first()
        target_branch = self.db.query(PromptBranch).filter(
            PromptBranch.id == target_branch_id
        ).first()
        
        if not source_branch or not target_branch:
            return ["Branch not found"]
        
        conflicts = []
        
        # Get head versions
        source_version = source_branch.head_version
        target_version = target_branch.head_version
        
        if source_version and target_version:
            # Check for content conflicts
            if source_version.content != target_version.content:
                # Simple conflict detection - in practice, this would be more sophisticated
                source_lines = set(source_version.content.splitlines())
                target_lines = set(target_version.content.splitlines())
                
                if source_lines & target_lines != source_lines and source_lines & target_lines != target_lines:
                    conflicts.append("Content conflicts detected")
            
            # Check for variable conflicts
            source_vars = set(str(var) for var in (source_version.variables or []))
            target_vars = set(str(var) for var in (target_version.variables or []))
            
            if source_vars != target_vars:
                conflicts.append("Variable definition conflicts")
        
        return conflicts
    
    def _perform_merge(self, source_branch: PromptBranch, target_branch: PromptBranch,
                      strategy: ConflictResolutionStrategy, merged_by: str) -> AdvancedPromptVersion:
        """Perform the actual merge operation."""
        source_version = source_branch.head_version
        target_version = target_branch.head_version
        
        if not source_version:
            raise ValueError("Source branch has no head version")
        
        # Determine merged content based on strategy
        if strategy == ConflictResolutionStrategy.ACCEPT_INCOMING:
            merged_content = source_version.content
            merged_variables = source_version.variables
            merged_tags = source_version.tags
        elif strategy == ConflictResolutionStrategy.ACCEPT_CURRENT:
            merged_content = target_version.content if target_version else ""
            merged_variables = target_version.variables if target_version else []
            merged_tags = target_version.tags if target_version else []
        else:  # AUTO_MERGE
            # Simple auto-merge - combine content and merge variables/tags
            merged_content = source_version.content
            
            # Merge variables (source takes precedence)
            target_vars = target_version.variables if target_version else []
            source_vars = source_version.variables or []
            merged_variables = source_vars + [var for var in target_vars if var not in source_vars]
            
            # Merge tags
            target_tags = set(target_version.tags if target_version else [])
            source_tags = set(source_version.tags or [])
            merged_tags = list(source_tags | target_tags)
        
        # Create merged version
        merged_version = AdvancedPromptVersion(
            prompt_id=target_branch.prompt_id,
            version=self._generate_next_version(target_branch.prompt_id),
            content=merged_content,
            changes=f"Merge from {source_branch.name} into {target_branch.name}",
            variables=merged_variables,
            tags=merged_tags,
            created_by=merged_by
        )
        
        self.db.add(merged_version)
        self.db.commit()
        self.db.refresh(merged_version)
        
        return merged_version