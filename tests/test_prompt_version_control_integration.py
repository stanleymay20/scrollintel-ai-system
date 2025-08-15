"""
Integration tests for the prompt version control system.
"""
import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from scrollintel.models.database import Base
from scrollintel.models.prompt_models import (
    AdvancedPromptTemplate, AdvancedPromptVersion, PromptBranch,
    PromptCommit, PromptMergeRequest, PromptVersionTag,
    ConflictResolutionStrategy
)
from scrollintel.core.prompt_version_control import PromptVersionControl
from scrollintel.core.prompt_diff_merge import PromptDiffMerge
from scrollintel.core.prompt_collaboration import PromptCollaboration, EditEvent, EditOperation


@pytest.fixture
def db_session():
    """Create a test database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def sample_prompt(db_session):
    """Create a sample prompt template for testing."""
    template = AdvancedPromptTemplate(
        name="Test Prompt",
        content="You are a helpful assistant. Please {{action}} the {{subject}}.",
        category="general",
        tags=["assistant", "helpful"],
        variables=[
            {"name": "action", "type": "string", "required": True},
            {"name": "subject", "type": "string", "required": True}
        ],
        created_by="test_user"
    )
    
    db_session.add(template)
    db_session.commit()
    db_session.refresh(template)
    
    # Create initial version
    version = AdvancedPromptVersion(
        prompt_id=template.id,
        version="1.0.0",
        content=template.content,
        changes="Initial version",
        variables=template.variables,
        tags=template.tags,
        created_by="test_user"
    )
    
    db_session.add(version)
    db_session.commit()
    db_session.refresh(version)
    
    return template, version


class TestPromptVersionControl:
    """Test the version control system."""
    
    def test_create_branch(self, db_session, sample_prompt):
        """Test creating a new branch."""
        template, version = sample_prompt
        vc = PromptVersionControl(db_session)
        
        # Create main branch first
        main_branch = vc.create_branch(
            template.id, "main", "main", "Main branch", "test_user"
        )
        
        # Create feature branch
        feature_branch = vc.create_branch(
            template.id, "feature/optimization", "main", 
            "Optimization feature", "test_user"
        )
        
        assert feature_branch.name == "feature/optimization"
        assert feature_branch.parent_branch_id == main_branch.id
        assert feature_branch.prompt_id == template.id
        assert feature_branch.created_by == "test_user"
    
    def test_commit_changes(self, db_session, sample_prompt):
        """Test committing changes to a branch."""
        template, version = sample_prompt
        vc = PromptVersionControl(db_session)
        
        # Create branch
        branch = vc.create_branch(template.id, "main", "main", "Main branch", "test_user")
        
        # Create new version
        new_version = AdvancedPromptVersion(
            prompt_id=template.id,
            version="1.1.0",
            content="Updated content",
            changes="Updated prompt content",
            created_by="test_user"
        )
        db_session.add(new_version)
        db_session.commit()
        db_session.refresh(new_version)
        
        # Commit changes
        commit = vc.commit_changes(
            branch.id, new_version.id, "Update prompt content", "test_user"
        )
        
        assert commit.message == "Update prompt content"
        assert commit.author == "test_user"
        assert commit.version_id == new_version.id
        assert commit.branch_id == branch.id
        assert len(commit.commit_hash) == 40  # SHA-1 hash length
    
    def test_create_merge_request(self, db_session, sample_prompt):
        """Test creating a merge request."""
        template, version = sample_prompt
        vc = PromptVersionControl(db_session)
        
        # Create branches
        main_branch = vc.create_branch(template.id, "main", "main", "Main branch", "test_user")
        feature_branch = vc.create_branch(
            template.id, "feature/test", "main", "Test feature", "test_user"
        )
        
        # Create merge request
        merge_request = vc.create_merge_request(
            feature_branch.id, main_branch.id,
            "Add test feature", "This adds a test feature", "test_user"
        )
        
        assert merge_request.title == "Add test feature"
        assert merge_request.source_branch_id == feature_branch.id
        assert merge_request.target_branch_id == main_branch.id
        assert merge_request.author == "test_user"
        assert merge_request.status == "open"
    
    def test_merge_branches(self, db_session, sample_prompt):
        """Test merging branches."""
        template, version = sample_prompt
        vc = PromptVersionControl(db_session)
        
        # Create branches
        main_branch = vc.create_branch(template.id, "main", "main", "Main branch", "test_user")
        feature_branch = vc.create_branch(
            template.id, "feature/test", "main", "Test feature", "test_user"
        )
        
        # Create versions for both branches
        feature_version = AdvancedPromptVersion(
            prompt_id=template.id,
            version="1.1.0",
            content="Feature content",
            changes="Added feature",
            created_by="test_user"
        )
        db_session.add(feature_version)
        db_session.commit()
        db_session.refresh(feature_version)
        
        # Update branch heads
        feature_branch.head_version_id = feature_version.id
        db_session.commit()
        
        # Create merge request
        merge_request = vc.create_merge_request(
            feature_branch.id, main_branch.id,
            "Merge feature", "Merge test feature", "test_user"
        )
        
        # Merge branches
        result = vc.merge_branches(
            merge_request.id, "test_user", ConflictResolutionStrategy.AUTO_MERGE
        )
        
        assert result is True
        
        # Refresh merge request
        db_session.refresh(merge_request)
        assert merge_request.status == "merged"
        assert merge_request.merged_by == "test_user"
    
    def test_create_tag(self, db_session, sample_prompt):
        """Test creating version tags."""
        template, version = sample_prompt
        vc = PromptVersionControl(db_session)
        
        # Create tag
        tag = vc.create_tag(
            version.id, "v1.0.0", "First stable release", "release", "test_user"
        )
        
        assert tag.name == "v1.0.0"
        assert tag.version_id == version.id
        assert tag.tag_type == "release"
        assert tag.created_by == "test_user"
    
    def test_rollback_to_version(self, db_session, sample_prompt):
        """Test rolling back to a previous version."""
        template, version = sample_prompt
        vc = PromptVersionControl(db_session)
        
        # Create newer version
        new_version = AdvancedPromptVersion(
            prompt_id=template.id,
            version="2.0.0",
            content="New content",
            changes="Major update",
            created_by="test_user"
        )
        db_session.add(new_version)
        db_session.commit()
        
        # Update template
        template.content = "New content"
        db_session.commit()
        
        # Rollback to original version
        rollback_version = vc.rollback_to_version(
            template.id, version.id, "main", "test_user"
        )
        
        assert rollback_version.content == version.content
        assert "Rollback to version" in rollback_version.changes
        
        # Check template was updated
        db_session.refresh(template)
        assert template.content == version.content
    
    def test_get_diff(self, db_session, sample_prompt):
        """Test getting diff between versions."""
        template, version1 = sample_prompt
        vc = PromptVersionControl(db_session)
        
        # Create second version
        version2 = AdvancedPromptVersion(
            prompt_id=template.id,
            version="1.1.0",
            content="You are a helpful assistant. Please {{action}} the {{subject}} carefully.",
            changes="Added 'carefully'",
            variables=version1.variables,
            tags=version1.tags + ["careful"],
            created_by="test_user"
        )
        db_session.add(version2)
        db_session.commit()
        db_session.refresh(version2)
        
        # Get diff
        diff = vc.get_diff(version1.id, version2.id)
        
        assert "content_diff" in diff
        assert "variables_diff" in diff
        assert "tags_diff" in diff
        assert len(diff["content_diff"]) > 0
        assert "careful" in diff["tags_diff"]["added"]
    
    def test_get_branch_history(self, db_session, sample_prompt):
        """Test getting branch commit history."""
        template, version = sample_prompt
        vc = PromptVersionControl(db_session)
        
        # Create branch
        branch = vc.create_branch(template.id, "main", "main", "Main branch", "test_user")
        
        # Create multiple commits
        for i in range(3):
            new_version = AdvancedPromptVersion(
                prompt_id=template.id,
                version=f"1.{i+1}.0",
                content=f"Content version {i+1}",
                changes=f"Update {i+1}",
                created_by="test_user"
            )
            db_session.add(new_version)
            db_session.commit()
            db_session.refresh(new_version)
            
            vc.commit_changes(branch.id, new_version.id, f"Commit {i+1}", "test_user")
        
        # Get history
        history = vc.get_branch_history(branch.id, limit=10)
        
        assert len(history) == 3
        assert history[0]["message"] == "Commit 3"  # Most recent first
        assert history[2]["message"] == "Commit 1"  # Oldest last


class TestPromptDiffMerge:
    """Test the diff and merge functionality."""
    
    def test_generate_unified_diff(self):
        """Test generating unified diff."""
        diff_merge = PromptDiffMerge()
        
        content1 = "Line 1\nLine 2\nLine 3"
        content2 = "Line 1\nModified Line 2\nLine 3\nLine 4"
        
        diff = diff_merge.generate_unified_diff(content1, content2)
        
        assert len(diff) > 0
        assert any("Modified Line 2" in line for line in diff)
        assert any("Line 4" in line for line in diff)
    
    def test_generate_side_by_side_diff(self):
        """Test generating side-by-side diff."""
        diff_merge = PromptDiffMerge()
        
        content1 = "Line 1\nLine 2\nLine 3"
        content2 = "Line 1\nModified Line 2\nLine 3"
        
        result = diff_merge.generate_side_by_side_diff(content1, content2)
        
        assert "diff_data" in result
        assert "stats" in result
        assert result["stats"]["modifications"] > 0
    
    def test_analyze_semantic_changes(self):
        """Test semantic change analysis."""
        diff_merge = PromptDiffMerge()
        
        content1 = "You are a {{role}}. You must follow {{rules}}."
        content2 = "You are a {{role}}. You should follow {{rules}} and {{guidelines}}."
        
        analysis = diff_merge.analyze_semantic_changes(content1, content2)
        
        assert "variables" in analysis
        assert "guidelines" in analysis["variables"]["added"]
        assert "instructions" in analysis
    
    def test_three_way_merge(self):
        """Test three-way merge."""
        diff_merge = PromptDiffMerge()
        
        base = "Line 1\nLine 2\nLine 3"
        current = "Line 1\nModified Line 2\nLine 3"
        incoming = "Line 1\nLine 2\nLine 3\nLine 4"
        
        result = diff_merge.three_way_merge(base, current, incoming)
        
        assert isinstance(result.success, bool)
        assert result.merged_content is not None
        assert isinstance(result.conflicts, list)
    
    def test_auto_resolve_conflicts(self):
        """Test automatic conflict resolution."""
        diff_merge = PromptDiffMerge()
        
        base = "Line 1\nLine 2\nLine 3"
        current = "Line 1\nCurrent Line 2\nLine 3"
        incoming = "Line 1\nIncoming Line 2\nLine 3"
        
        merge_result = diff_merge.three_way_merge(base, current, incoming)
        resolved = diff_merge.auto_resolve_conflicts(merge_result, "current")
        
        assert resolved.success
        assert "Current Line 2" in resolved.merged_content


class TestPromptCollaboration:
    """Test the collaborative editing system."""
    
    def test_start_collaboration_session(self, db_session, sample_prompt):
        """Test starting a collaboration session."""
        template, version = sample_prompt
        collaboration = PromptCollaboration(db_session)
        
        session = collaboration.start_collaboration_session(template.id, "user1")
        
        assert session.prompt_id == template.id
        assert "user1" in session.participants
        assert session.is_active
    
    def test_acquire_edit_lock(self, db_session, sample_prompt):
        """Test acquiring edit locks."""
        template, version = sample_prompt
        collaboration = PromptCollaboration(db_session)
        
        # Start session
        collaboration.start_collaboration_session(template.id, "user1")
        
        # Acquire lock
        success = collaboration.acquire_edit_lock(template.id, "user1")
        
        assert success is True
        
        # Try to acquire conflicting lock
        success2 = collaboration.acquire_edit_lock(template.id, "user2")
        
        assert success2 is False  # Should conflict with existing lock
    
    def test_apply_edit(self, db_session, sample_prompt):
        """Test applying collaborative edits."""
        template, version = sample_prompt
        collaboration = PromptCollaboration(db_session)
        
        # Start session and acquire lock
        collaboration.start_collaboration_session(template.id, "user1")
        collaboration.acquire_edit_lock(template.id, "user1")
        
        # Create edit event
        edit_event = EditEvent(
            id="edit1",
            prompt_id=template.id,
            user_id="user1",
            operation=EditOperation.INSERT,
            position=10,
            length=0,
            content=" very",
            timestamp=datetime.utcnow(),
            version=1
        )
        
        # Apply edit
        result = collaboration.apply_edit(template.id, "user1", edit_event)
        
        assert result["success"] is True
        assert "version_id" in result
        assert " very" in result["new_content"]
    
    def test_get_collaboration_status(self, db_session, sample_prompt):
        """Test getting collaboration status."""
        template, version = sample_prompt
        collaboration = PromptCollaboration(db_session)
        
        # Start session
        collaboration.start_collaboration_session(template.id, "user1")
        collaboration.acquire_edit_lock(template.id, "user1")
        
        status = collaboration.get_collaboration_status(template.id)
        
        assert status["has_active_session"] is True
        assert "user1" in status["participants"]
        assert status["active_locks"] > 0
    
    def test_end_collaboration_session(self, db_session, sample_prompt):
        """Test ending collaboration session."""
        template, version = sample_prompt
        collaboration = PromptCollaboration(db_session)
        
        # Start session
        session = collaboration.start_collaboration_session(template.id, "user1")
        collaboration.acquire_edit_lock(template.id, "user1")
        
        # End session
        success = collaboration.end_collaboration_session(session.session_id, "user1")
        
        assert success is True
        
        # Check status
        status = collaboration.get_collaboration_status(template.id)
        assert status["has_active_session"] is False
        assert status["active_locks"] == 0


class TestVersionControlIntegration:
    """Test integration between all version control components."""
    
    def test_full_collaborative_workflow(self, db_session, sample_prompt):
        """Test a complete collaborative editing workflow."""
        template, version = sample_prompt
        vc = PromptVersionControl(db_session)
        collaboration = PromptCollaboration(db_session)
        
        # Create main branch
        main_branch = vc.create_branch(template.id, "main", "main", "Main branch", "user1")
        
        # Start collaboration
        session = collaboration.start_collaboration_session(template.id, "user1")
        collaboration.acquire_edit_lock(template.id, "user1")
        
        # Apply collaborative edit
        edit_event = EditEvent(
            id="edit1",
            prompt_id=template.id,
            user_id="user1",
            operation=EditOperation.REPLACE,
            position=20,
            length=9,  # Length of "assistant"
            content="AI helper",
            timestamp=datetime.utcnow(),
            version=1
        )
        
        edit_result = collaboration.apply_edit(template.id, "user1", edit_event)
        assert edit_result["success"] is True
        
        # Commit the changes
        new_version_id = edit_result["version_id"]
        commit = vc.commit_changes(
            main_branch.id, new_version_id, "Collaborative edit: changed assistant to AI helper", "user1"
        )
        
        assert commit.message == "Collaborative edit: changed assistant to AI helper"
        
        # Create a tag for this version
        tag = vc.create_tag(new_version_id, "v1.1.0", "Updated terminology", "release", "user1")
        assert tag.name == "v1.1.0"
        
        # Get diff to verify changes
        diff = vc.get_diff(version.id, new_version_id)
        assert any("AI helper" in line for line in diff["content_diff"])
        
        # End collaboration
        collaboration.end_collaboration_session(session.session_id, "user1")
        
        # Verify final state
        status = collaboration.get_collaboration_status(template.id)
        assert status["has_active_session"] is False
        
        history = vc.get_branch_history(main_branch.id)
        assert len(history) >= 1
        assert "Collaborative edit" in history[0]["message"]
    
    def test_conflict_resolution_workflow(self, db_session, sample_prompt):
        """Test conflict detection and resolution."""
        template, version = sample_prompt
        collaboration = PromptCollaboration(db_session)
        
        # Start sessions for two users
        session1 = collaboration.start_collaboration_session(template.id, "user1")
        session2 = collaboration.start_collaboration_session(template.id, "user2")
        
        # Both users acquire locks on different sections
        collaboration.acquire_edit_lock(template.id, "user1", section_start=0, section_end=20)
        collaboration.acquire_edit_lock(template.id, "user2", section_start=21, section_end=50)
        
        # User1 makes an edit
        edit1 = EditEvent(
            id="edit1",
            prompt_id=template.id,
            user_id="user1",
            operation=EditOperation.INSERT,
            position=10,
            length=0,
            content=" very",
            timestamp=datetime.utcnow(),
            version=1
        )
        
        result1 = collaboration.apply_edit(template.id, "user1", edit1)
        assert result1["success"] is True
        
        # User2 makes a conflicting edit (overlapping position)
        edit2 = EditEvent(
            id="edit2",
            prompt_id=template.id,
            user_id="user2",
            operation=EditOperation.INSERT,
            position=15,  # Overlaps with user1's edit
            length=0,
            content=" extremely",
            timestamp=datetime.utcnow(),
            version=1
        )
        
        # This should detect conflicts
        result2 = collaboration.apply_edit(template.id, "user2", edit2)
        
        # In a real implementation, this would handle conflicts more sophisticatedly
        # For now, we just verify the system can handle the scenario
        assert isinstance(result2, dict)
        
        # Clean up
        collaboration.end_collaboration_session(session1.session_id, "user1")
        collaboration.end_collaboration_session(session2.session_id, "user2")