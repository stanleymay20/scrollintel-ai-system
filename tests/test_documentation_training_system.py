"""
Tests for Documentation and Training System
Comprehensive tests for all documentation, training, awareness, and knowledge management functionality
"""

import pytest
import json
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

from security.documentation.documentation_training_system import IntegratedDocumentationTrainingSystem
from security.training.training_system import TrainingType, DifficultyLevel
from security.awareness.phishing_simulator import ContentType as AwarenessContentType
from security.policy.policy_management_system import PolicyType, PolicyStatus
from security.knowledge_base.knowledge_base_system import ContentType as KBContentType, AccessLevel

class TestIntegratedDocumentationTrainingSystem:
    """Test the integrated documentation and training system"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def doc_training_system(self, temp_dir):
        """Create test documentation and training system"""
        return IntegratedDocumentationTrainingSystem(base_path=temp_dir)
    
    def test_system_initialization(self, doc_training_system):
        """Test that all subsystems are properly initialized"""
        assert doc_training_system.documentation_manager is not None
        assert doc_training_system.training_system is not None
        assert doc_training_system.awareness_system is not None
        assert doc_training_system.playbook_system is not None
        assert doc_training_system.policy_system is not None
        assert doc_training_system.knowledge_base is not None
    
    def test_create_comprehensive_security_program(self, doc_training_system):
        """Test creating a comprehensive security program"""
        program_name = "Data Protection Program"
        requirements = [
            "Encrypt all sensitive data",
            "Implement access controls",
            "Monitor data access",
            "Train staff on data protection"
        ]
        
        program_ids = doc_training_system.create_comprehensive_security_program(
            program_name, requirements
        )
        
        # Verify all components were created
        assert 'policy' in program_ids
        assert 'training_modules' in program_ids
        assert 'awareness_campaign' in program_ids
        assert 'incident_playbook' in program_ids
        assert 'knowledge_articles' in program_ids
        assert 'documentation' in program_ids
        
        # Verify policy was created
        policy_id = program_ids['policy']
        assert policy_id in doc_training_system.policy_system.policies
        policy = doc_training_system.policy_system.policies[policy_id]
        assert program_name in policy.title
        
        # Verify training modules were created
        training_modules = program_ids['training_modules']
        assert len(training_modules) > 0
        for module_id in training_modules:
            assert module_id in doc_training_system.training_system.modules
        
        # Verify knowledge base articles were created
        kb_articles = program_ids['knowledge_articles']
        assert len(kb_articles) > 0
        for article_id in kb_articles:
            assert article_id in doc_training_system.knowledge_base.articles

class TestDocumentationManager:
    """Test the documentation management system"""
    
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def doc_system(self, temp_dir):
        return IntegratedDocumentationTrainingSystem(base_path=temp_dir)
    
    def test_create_document(self, doc_system):
        """Test creating a new document"""
        doc_id = doc_system.documentation_manager.create_document(
            doc_id="test-doc-001",
            title="Test Security Document",
            content="This is a test security document.",
            classification="internal",
            tags=["test", "security"]
        )
        
        assert doc_id == "test-doc-001"
        assert doc_id in doc_system.documentation_manager.documents
        
        doc = doc_system.documentation_manager.documents[doc_id]
        assert doc.title == "Test Security Document"
        assert doc.classification == "internal"
        assert "test" in doc.tags
    
    def test_update_document(self, doc_system):
        """Test updating an existing document"""
        # Create document first
        doc_id = doc_system.documentation_manager.create_document(
            doc_id="test-doc-002",
            title="Test Document",
            content="Original content"
        )
        
        # Update document
        success = doc_system.documentation_manager.update_document(
            doc_id=doc_id,
            content="Updated content",
            version_increment="minor"
        )
        
        assert success
        doc = doc_system.documentation_manager.documents[doc_id]
        assert "Updated content" in doc.content
        assert doc.version == "1.1"
    
    def test_approve_document(self, doc_system):
        """Test approving a document"""
        doc_id = doc_system.documentation_manager.create_document(
            doc_id="test-doc-003",
            title="Test Document",
            content="Test content"
        )
        
        success = doc_system.documentation_manager.approve_document(doc_id, "test-approver")
        assert success
        
        doc = doc_system.documentation_manager.documents[doc_id]
        assert doc.status == "approved"

class TestTrainingSystem:
    """Test the training management system"""
    
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def training_system(self, temp_dir):
        doc_system = IntegratedDocumentationTrainingSystem(base_path=temp_dir)
        return doc_system.training_system
    
    def test_create_training_module(self, training_system):
        """Test creating a training module"""
        module_data = {
            "title": "Test Security Training",
            "description": "Test training module",
            "type": TrainingType.SECURITY,
            "difficulty": DifficultyLevel.BEGINNER,
            "duration_minutes": 30,
            "prerequisites": [],
            "learning_objectives": ["Learn security basics"],
            "content_sections": [],
            "assessment_questions": [],
            "passing_score": 80,
            "certification_points": 10,
            "mandatory": True
        }
        
        module_id = training_system.create_training_module(module_data)
        assert module_id in training_system.modules
        
        module = training_system.modules[module_id]
        assert module.title == "Test Security Training"
        assert module.type == TrainingType.SECURITY
    
    def test_start_training(self, training_system):
        """Test starting training for a user"""
        # Use existing default module
        module_id = "sec-basics-001"  # Default module created during initialization
        user_id = "test-user-001"
        
        success = training_system.start_training(user_id, module_id)
        assert success
        
        # Check user progress
        user_progress = training_system.user_progress.get(user_id, [])
        assert len(user_progress) > 0
        
        progress = user_progress[0]
        assert progress.module_id == module_id
        assert progress.status == "in_progress"
    
    def test_complete_assessment(self, training_system):
        """Test completing a training assessment"""
        module_id = "sec-basics-001"
        user_id = "test-user-002"
        
        # Start training first
        training_system.start_training(user_id, module_id)
        
        # Complete assessment with correct answers
        answers = [1]  # Correct answer for the default question
        result = training_system.complete_assessment(user_id, module_id, answers)
        
        assert result['passed'] == True
        assert result['score'] >= 80
    
    def test_get_training_status(self, training_system):
        """Test getting user training status"""
        user_id = "test-user-003"
        module_id = "sec-basics-001"
        
        # Start and complete training
        training_system.start_training(user_id, module_id)
        training_system.complete_assessment(user_id, module_id, [1])
        
        status = training_system.get_user_training_status(user_id)
        
        assert status['user_id'] == user_id
        assert len(status['completed_modules']) > 0
        assert status['mandatory_compliance'] == True

class TestAwarenessSystem:
    """Test the security awareness system"""
    
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def awareness_system(self, temp_dir):
        doc_system = IntegratedDocumentationTrainingSystem(base_path=temp_dir)
        return doc_system.awareness_system
    
    def test_create_phishing_campaign(self, awareness_system):
        """Test creating a phishing campaign"""
        campaign_data = {
            "name": "Test Phishing Campaign",
            "description": "Test campaign for security awareness",
            "template_ids": ["phish-001"],
            "target_groups": ["test_group"],
            "start_date": datetime.now(),
            "end_date": datetime.now() + timedelta(days=30),
            "frequency": "one_time",
            "created_by": "test-admin"
        }
        
        campaign_id = awareness_system.create_phishing_campaign(campaign_data)
        assert campaign_id in awareness_system.campaigns
        
        campaign = awareness_system.campaigns[campaign_id]
        assert campaign.name == "Test Phishing Campaign"
    
    def test_launch_phishing_campaign(self, awareness_system):
        """Test launching a phishing campaign"""
        # Create campaign first
        campaign_data = {
            "name": "Test Launch Campaign",
            "description": "Test campaign launch",
            "template_ids": ["phish-001"],
            "target_groups": ["test_group"],
            "start_date": datetime.now(),
            "end_date": datetime.now() + timedelta(days=30),
            "frequency": "one_time",
            "created_by": "test-admin"
        }
        
        campaign_id = awareness_system.create_phishing_campaign(campaign_data)
        
        # Launch campaign
        target_users = [
            {"user_id": "user1", "email": "user1@test.com"},
            {"user_id": "user2", "email": "user2@test.com"}
        ]
        
        success = awareness_system.launch_phishing_campaign(campaign_id, target_users)
        assert success
        
        # Check that results were created
        campaign_results = [r for r in awareness_system.results.values() if r.campaign_id == campaign_id]
        assert len(campaign_results) > 0
    
    def test_get_user_awareness_score(self, awareness_system):
        """Test getting user awareness score"""
        user_id = "test-user-awareness"
        
        # Get score for user with no data
        score = awareness_system.get_user_awareness_score(user_id)
        assert score['user_id'] == user_id
        assert score['awareness_score'] == 100  # No data, assume good
        assert score['risk_level'] == "low"

class TestPolicySystem:
    """Test the policy management system"""
    
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def policy_system(self, temp_dir):
        doc_system = IntegratedDocumentationTrainingSystem(base_path=temp_dir)
        return doc_system.policy_system
    
    def test_create_policy(self, policy_system):
        """Test creating a new policy"""
        policy_data = {
            "title": "Test Security Policy",
            "description": "Test policy for security",
            "policy_type": PolicyType.SECURITY,
            "content": "This is a test security policy content.",
            "author": "test-author",
            "owner": "test-owner",
            "tags": ["test", "security"],
            "compliance_frameworks": ["ISO27001"],
            "approval_required": True
        }
        
        policy_id = policy_system.create_policy(policy_data)
        assert policy_id in policy_system.policies
        
        policy = policy_system.policies[policy_id]
        assert policy.title == "Test Security Policy"
        assert policy.status == PolicyStatus.DRAFT
    
    def test_update_policy(self, policy_system):
        """Test updating a policy"""
        # Create policy first
        policy_data = {
            "title": "Test Policy Update",
            "description": "Test policy",
            "policy_type": PolicyType.SECURITY,
            "content": "Original content",
            "author": "test-author",
            "owner": "test-owner"
        }
        
        policy_id = policy_system.create_policy(policy_data)
        
        # Update policy
        version_id = policy_system.update_policy(
            policy_id=policy_id,
            content="Updated content",
            changes_summary="Updated for testing",
            author="test-updater",
            change_type="minor"
        )
        
        assert version_id in policy_system.versions
        
        policy = policy_system.policies[policy_id]
        assert "Updated content" in policy.content
        assert policy.version == "1.1"
    
    def test_policy_approval_workflow(self, policy_system):
        """Test policy approval workflow"""
        # Create policy
        policy_data = {
            "title": "Test Approval Policy",
            "description": "Test policy approval",
            "policy_type": PolicyType.SECURITY,
            "content": "Test content",
            "author": "test-author",
            "owner": "test-owner"
        }
        
        policy_id = policy_system.create_policy(policy_data)
        
        # Submit for approval
        approval_steps = [
            {"approver": "manager1", "role": "manager"},
            {"approver": "ciso", "role": "ciso"}
        ]
        
        workflow_id = policy_system.submit_for_approval(
            policy_id, "test-requester", approval_steps
        )
        
        assert workflow_id in policy_system.workflows
        
        # Approve first step
        success = policy_system.approve_policy_step(
            workflow_id, "manager1", True, "Approved by manager"
        )
        assert success
        
        # Approve second step
        success = policy_system.approve_policy_step(
            workflow_id, "ciso", True, "Final approval"
        )
        assert success
        
        # Check policy status
        policy = policy_system.policies[policy_id]
        assert policy.status == PolicyStatus.APPROVED

class TestKnowledgeBase:
    """Test the knowledge base system"""
    
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def kb_system(self, temp_dir):
        doc_system = IntegratedDocumentationTrainingSystem(base_path=temp_dir)
        return doc_system.knowledge_base
    
    def test_create_article(self, kb_system):
        """Test creating a knowledge base article"""
        article_data = {
            "title": "Test KB Article",
            "summary": "Test article summary",
            "content": "This is test content for the knowledge base article.",
            "content_type": KBContentType.TUTORIAL,
            "access_level": AccessLevel.INTERNAL,
            "author": "test-author",
            "tags": ["test", "tutorial"],
            "categories": ["testing"],
            "search_keywords": ["test", "knowledge", "article"]
        }
        
        article_id = kb_system.create_article(article_data)
        assert article_id in kb_system.articles
        
        article = kb_system.articles[article_id]
        assert article.title == "Test KB Article"
        assert article.content_type == KBContentType.TUTORIAL
    
    def test_search_articles(self, kb_system):
        """Test searching knowledge base articles"""
        # Create test article
        article_data = {
            "title": "Security Best Practices",
            "summary": "Best practices for security",
            "content": "This article covers security best practices including password management.",
            "content_type": KBContentType.BEST_PRACTICE,
            "status": "published",
            "access_level": AccessLevel.INTERNAL,
            "author": "security-team",
            "tags": ["security", "best-practices"],
            "categories": ["security"],
            "search_keywords": ["security", "password", "best", "practices"]
        }
        
        kb_system.create_article(article_data)
        
        # Search for articles
        results = kb_system.search_articles("security password", "test-user", AccessLevel.INTERNAL)
        
        assert len(results) > 0
        assert any("Security Best Practices" in result['title'] for result in results)
    
    def test_submit_feedback(self, kb_system):
        """Test submitting feedback for an article"""
        # Use existing default article
        article_id = "kb-001"  # Default article created during initialization
        
        feedback_id = kb_system.submit_feedback(
            article_id=article_id,
            user_id="test-user",
            rating=5,
            feedback_text="Very helpful article!",
            helpful=True
        )
        
        assert feedback_id in kb_system.feedback
        
        feedback = kb_system.feedback[feedback_id]
        assert feedback.rating == 5
        assert feedback.helpful == True
        
        # Check that article rating was updated
        article = kb_system.articles[article_id]
        assert article.rating > 0
        assert article.rating_count > 0
    
    def test_get_popular_articles(self, kb_system):
        """Test getting popular articles"""
        # Increment view count for an article
        article_id = "kb-001"
        article = kb_system.articles[article_id]
        article.view_count = 100
        
        popular_articles = kb_system.get_popular_articles(limit=5, access_level=AccessLevel.INTERNAL)
        
        assert len(popular_articles) > 0
        assert any(article['id'] == article_id for article in popular_articles)

class TestIntegrationFeatures:
    """Test integration features between systems"""
    
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def integrated_system(self, temp_dir):
        return IntegratedDocumentationTrainingSystem(base_path=temp_dir)
    
    def test_user_security_dashboard(self, integrated_system):
        """Test getting comprehensive user security dashboard"""
        user_id = "test-dashboard-user"
        
        # Complete some training
        integrated_system.training_system.start_training(user_id, "sec-basics-001")
        integrated_system.training_system.complete_assessment(user_id, "sec-basics-001", [1])
        
        # Get dashboard
        dashboard = integrated_system.get_user_security_dashboard(user_id)
        
        assert dashboard['user_id'] == user_id
        assert 'overall_security_score' in dashboard
        assert 'training_status' in dashboard
        assert 'awareness_score' in dashboard
        assert 'knowledge_base_activity' in dashboard
        assert 'policy_compliance' in dashboard
        assert 'recommendations' in dashboard
    
    def test_comprehensive_report(self, integrated_system):
        """Test generating comprehensive report"""
        report = integrated_system.generate_comprehensive_report()
        
        assert 'report_type' in report
        assert 'generated_date' in report
        assert 'integration_score' in report
        assert 'documentation' in report
        assert 'training' in report
        assert 'awareness' in report
        assert 'incident_response' in report
        assert 'policies' in report
        assert 'knowledge_base' in report
        assert 'summary' in report
    
    def test_automated_maintenance(self, integrated_system):
        """Test automated maintenance functionality"""
        results = integrated_system.perform_automated_maintenance()
        
        assert 'documentation_updates' in results
        assert 'policies_needing_review' in results
        assert 'kb_index_updated' in results
        assert 'training_reminders_sent' in results

if __name__ == "__main__":
    pytest.main([__file__])