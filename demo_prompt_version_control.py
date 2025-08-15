"""
Demo script for the Advanced Prompt Management Version Control System.
"""
import os
import sys
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scrollintel.models.database import Base
from scrollintel.models.prompt_models import (
    AdvancedPromptTemplate, AdvancedPromptVersion, PromptBranch,
    PromptCommit, PromptMergeRequest, PromptVersionTag,
    ConflictResolutionStrategy
)
from scrollintel.core.prompt_version_control import PromptVersionControl
from scrollintel.core.prompt_diff_merge import PromptDiffMerge
from scrollintel.core.prompt_collaboration import PromptCollaboration, EditEvent, EditOperation


def create_demo_database():
    """Create an in-memory database for the demo."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def create_sample_prompt(db_session):
    """Create a sample prompt template."""
    template = AdvancedPromptTemplate(
        name="Customer Service Assistant",
        content="""You are a helpful customer service assistant for {{company_name}}.

Your role is to:
- Answer customer questions about {{product_type}}
- Provide {{support_level}} support
- Maintain a {{tone}} tone throughout the conversation

Guidelines:
- Always be polite and professional
- If you don't know something, admit it and offer to find out
- Escalate complex issues to human agents when necessary

Customer Query: {{customer_query}}

Response:""",
        category="customer_service",
        tags=["assistant", "customer_service", "support"],
        variables=[
            {"name": "company_name", "type": "string", "required": True},
            {"name": "product_type", "type": "string", "required": True},
            {"name": "support_level", "type": "string", "required": True},
            {"name": "tone", "type": "string", "required": True},
            {"name": "customer_query", "type": "string", "required": True}
        ],
        created_by="demo_user"
    )
    
    db_session.add(template)
    db_session.commit()
    db_session.refresh(template)
    
    # Create initial version
    version = AdvancedPromptVersion(
        prompt_id=template.id,
        version="1.0.0",
        content=template.content,
        changes="Initial version of customer service assistant prompt",
        variables=template.variables,
        tags=template.tags,
        created_by="demo_user"
    )
    
    db_session.add(version)
    db_session.commit()
    db_session.refresh(version)
    
    return template, version


def demo_version_control_workflow(db_session):
    """Demonstrate the complete version control workflow."""
    print("🚀 Advanced Prompt Management Version Control Demo")
    print("=" * 60)
    
    # Create sample prompt
    print("\n1. Creating sample prompt template...")
    template, initial_version = create_sample_prompt(db_session)
    print(f"   ✅ Created prompt: {template.name}")
    print(f"   ✅ Initial version: {initial_version.version}")
    
    # Initialize version control
    vc = PromptVersionControl(db_session)
    diff_merge = PromptDiffMerge()
    
    # Create main branch
    print("\n2. Creating main branch...")
    main_branch = vc.create_branch(
        template.id, "main", "main", "Main development branch", "demo_user"
    )
    print(f"   ✅ Created main branch: {main_branch.name}")
    
    # Create feature branch
    print("\n3. Creating feature branch...")
    feature_branch = vc.create_branch(
        template.id, "feature/enhanced-guidelines", "main", 
        "Add enhanced guidelines for better customer service", "demo_user"
    )
    print(f"   ✅ Created feature branch: {feature_branch.name}")
    
    # Create enhanced version
    print("\n4. Creating enhanced version...")
    enhanced_content = """You are a helpful customer service assistant for {{company_name}}.

Your role is to:
- Answer customer questions about {{product_type}}
- Provide {{support_level}} support
- Maintain a {{tone}} tone throughout the conversation

Enhanced Guidelines:
- Always be polite and professional
- Use active listening techniques
- Personalize responses when possible
- If you don't know something, admit it and offer to find out
- Escalate complex issues to human agents when necessary
- Follow up to ensure customer satisfaction

Quality Standards:
- Response time: Under 2 minutes
- Accuracy: 95% or higher
- Customer satisfaction: 4.5/5 or higher

Customer Query: {{customer_query}}

Response:"""
    
    enhanced_version = AdvancedPromptVersion(
        prompt_id=template.id,
        version="1.1.0",
        content=enhanced_content,
        changes="Added enhanced guidelines and quality standards",
        variables=template.variables,
        tags=template.tags + ["enhanced", "quality"],
        created_by="demo_user"
    )
    
    db_session.add(enhanced_version)
    db_session.commit()
    db_session.refresh(enhanced_version)
    
    # Update feature branch head
    feature_branch.head_version_id = enhanced_version.id
    db_session.commit()
    
    print(f"   ✅ Created enhanced version: {enhanced_version.version}")
    
    # Commit changes to feature branch
    print("\n5. Committing changes...")
    commit = vc.commit_changes(
        feature_branch.id, enhanced_version.id,
        "Add enhanced guidelines and quality standards", "demo_user"
    )
    print(f"   ✅ Created commit: {commit.commit_hash[:8]}")
    print(f"   ✅ Commit message: {commit.message}")
    
    # Generate diff
    print("\n6. Generating diff between versions...")
    diff = vc.get_diff(initial_version.id, enhanced_version.id)
    print(f"   ✅ Content changes: {len(diff['content_diff'])} lines")
    print(f"   ✅ Tags added: {diff['tags_diff']['added']}")
    
    # Show side-by-side diff
    side_diff = diff_merge.generate_side_by_side_diff(
        initial_version.content, enhanced_version.content
    )
    print(f"   ✅ Diff stats: +{side_diff['stats']['additions']} -{side_diff['stats']['deletions']}")
    
    # Create merge request
    print("\n7. Creating merge request...")
    merge_request = vc.create_merge_request(
        feature_branch.id, main_branch.id,
        "Enhanced Customer Service Guidelines",
        "This PR adds enhanced guidelines and quality standards to improve customer service interactions.",
        "demo_user"
    )
    print(f"   ✅ Created merge request: {merge_request.title}")
    print(f"   ✅ Status: {merge_request.status}")
    print(f"   ✅ Conflicts detected: {len(merge_request.conflicts)}")
    
    # Merge branches
    print("\n8. Merging feature branch...")
    merge_success = vc.merge_branches(
        merge_request.id, "demo_user", ConflictResolutionStrategy.AUTO_MERGE
    )
    print(f"   ✅ Merge successful: {merge_success}")
    
    # Refresh merge request
    db_session.refresh(merge_request)
    print(f"   ✅ Merge request status: {merge_request.status}")
    print(f"   ✅ Merged by: {merge_request.merged_by}")
    
    # Create version tag
    print("\n9. Creating version tag...")
    tag = vc.create_tag(
        enhanced_version.id, "v1.1.0-stable", 
        "Stable release with enhanced guidelines", "release", "demo_user"
    )
    print(f"   ✅ Created tag: {tag.name}")
    print(f"   ✅ Tag type: {tag.tag_type}")
    
    # Get branch history
    print("\n10. Retrieving branch history...")
    history = vc.get_branch_history(main_branch.id, limit=5)
    print(f"    ✅ Branch history ({len(history)} commits):")
    for i, commit_data in enumerate(history):
        print(f"       {i+1}. {commit_data['commit_hash'][:8]} - {commit_data['message']}")
        print(f"          Author: {commit_data['author']}")
        print(f"          Date: {commit_data['committed_at']}")
    
    return template, main_branch, feature_branch


def demo_collaborative_editing(db_session, template):
    """Demonstrate collaborative editing features."""
    print("\n" + "=" * 60)
    print("🤝 Collaborative Editing Demo")
    print("=" * 60)
    
    collaboration = PromptCollaboration(db_session)
    
    # Start collaboration sessions
    print("\n1. Starting collaboration sessions...")
    session1 = collaboration.start_collaboration_session(template.id, "user1")
    session2 = collaboration.start_collaboration_session(template.id, "user2")
    print(f"   ✅ User1 session: {session1.session_id}")
    print(f"   ✅ User2 joined session: {len(session2.participants)} participants")
    
    # Acquire edit locks
    print("\n2. Acquiring edit locks...")
    lock1_success = collaboration.acquire_edit_lock(template.id, "user1", section_start=0, section_end=100)
    lock2_success = collaboration.acquire_edit_lock(template.id, "user2", section_start=101, section_end=200)
    print(f"   ✅ User1 lock acquired: {lock1_success}")
    print(f"   ✅ User2 lock acquired: {lock2_success}")
    
    # Try conflicting lock
    conflict_lock = collaboration.acquire_edit_lock(template.id, "user2", section_start=50, section_end=150)
    print(f"   ❌ Conflicting lock rejected: {not conflict_lock}")
    
    # Apply collaborative edits
    print("\n3. Applying collaborative edits...")
    edit1 = EditEvent(
        id="edit1",
        prompt_id=template.id,
        user_id="user1",
        operation=EditOperation.INSERT,
        position=20,
        length=0,
        content=" highly",
        timestamp=datetime.utcnow(),
        version=1
    )
    
    result1 = collaboration.apply_edit(template.id, "user1", edit1)
    print(f"   ✅ User1 edit applied: {result1.get('success', False)}")
    
    # Get collaboration status
    print("\n4. Checking collaboration status...")
    status = collaboration.get_collaboration_status(template.id)
    print(f"   ✅ Active session: {status['has_active_session']}")
    print(f"   ✅ Participants: {status['participants']}")
    print(f"   ✅ Active locks: {status['active_locks']}")
    print(f"   ✅ Pending edits: {status['pending_edits']}")
    
    # End collaboration sessions
    print("\n5. Ending collaboration sessions...")
    collaboration.end_collaboration_session(session1.session_id, "user1")
    collaboration.end_collaboration_session(session2.session_id, "user2")
    
    final_status = collaboration.get_collaboration_status(template.id)
    print(f"   ✅ Session ended: {not final_status['has_active_session']}")
    print(f"   ✅ Locks released: {final_status['active_locks'] == 0}")


def demo_advanced_diff_merge():
    """Demonstrate advanced diff and merge capabilities."""
    print("\n" + "=" * 60)
    print("🔍 Advanced Diff & Merge Demo")
    print("=" * 60)
    
    diff_merge = PromptDiffMerge()
    
    # Sample content for diff analysis
    content1 = """You are a {{role}} assistant.
Your task is to {{task}}.
Always be helpful and accurate."""
    
    content2 = """You are a {{role}} AI assistant.
Your primary task is to {{task}} efficiently.
Always be helpful, accurate, and {{style}}.
Provide {{detail_level}} responses."""
    
    print("\n1. Generating unified diff...")
    unified_diff = diff_merge.generate_unified_diff(content1, content2)
    print(f"   ✅ Unified diff lines: {len(unified_diff)}")
    
    print("\n2. Analyzing semantic changes...")
    analysis = diff_merge.analyze_semantic_changes(content1, content2)
    print(f"   ✅ Variables added: {analysis['variables']['added']}")
    print(f"   ✅ Instructions modified: {len(analysis['instructions']['modified'])}")
    print(f"   ✅ Complexity change: {analysis['complexity_change']['word_count_change']} words")
    
    print("\n3. Performing three-way merge...")
    base = "You are a helpful assistant."
    current = "You are a very helpful assistant."
    incoming = "You are a helpful AI assistant."
    
    merge_result = diff_merge.three_way_merge(base, current, incoming)
    print(f"   ✅ Merge successful: {merge_result.success}")
    print(f"   ✅ Conflicts detected: {len(merge_result.conflicts)}")
    
    if merge_result.conflicts:
        print("   ⚠️  Conflict details:")
        for conflict in merge_result.conflicts:
            print(f"      - Type: {conflict.type.value}")
            print(f"      - Description: {conflict.description}")
    
    print("\n4. Auto-resolving conflicts...")
    if merge_result.conflicts:
        resolved = diff_merge.auto_resolve_conflicts(merge_result, "smart")
        print(f"   ✅ Auto-resolution successful: {resolved.success}")
        print(f"   ✅ Remaining conflicts: {len(resolved.conflicts)}")


def main():
    """Run the complete demo."""
    print("🎯 Starting Advanced Prompt Management Version Control Demo")
    print("This demo showcases Git-like version control for AI prompts")
    
    # Create demo database
    db_session = create_demo_database()
    
    try:
        # Run version control workflow demo
        template, main_branch, feature_branch = demo_version_control_workflow(db_session)
        
        # Run collaborative editing demo
        demo_collaborative_editing(db_session, template)
        
        # Run advanced diff and merge demo
        demo_advanced_diff_merge()
        
        print("\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("✅ Git-like branching and merging")
        print("✅ Commit history and version tracking")
        print("✅ Merge requests and conflict detection")
        print("✅ Version tagging and releases")
        print("✅ Collaborative real-time editing")
        print("✅ Advanced diff and merge algorithms")
        print("✅ Semantic change analysis")
        print("✅ Conflict resolution strategies")
        
        print("\nThe Advanced Prompt Management Version Control System is ready for production use!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        db_session.close()


if __name__ == "__main__":
    main()