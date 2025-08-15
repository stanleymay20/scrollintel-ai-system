"""
Comprehensive tests for the ScrollIntel support system.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from sqlalchemy.orm import Session

from scrollintel.core.support_system import (
    SupportTicketManager, KnowledgeBaseManager, 
    FeedbackManager, ContactFormManager
)
from scrollintel.models.support_models import (
    SupportTicket, SupportMessage, KnowledgeBaseArticle, FAQ, UserFeedback,
    TicketStatus, TicketPriority, SupportTicketCreate, SupportMessageCreate,
    FeedbackCreate, ContactFormSubmission
)

class TestSupportTicketManager:
    """Test support ticket management functionality."""
    
    @pytest.fixture
    def db_session(self):
        """Mock database session."""
        return Mock(spec=Session)
    
    @pytest.fixture
    def ticket_manager(self, db_session):
        """Create ticket manager instance."""
        return SupportTicketManager(db_session)
    
    @pytest.fixture
    def sample_ticket_data(self):
        """Sample ticket creation data."""
        return SupportTicketCreate(
            email="user@example.com",
            subject="Test Issue",
            description="This is a test issue description",
            category="technical_issue",
            priority=TicketPriority.MEDIUM,
            user_id="user123"
        )
    
    def test_create_ticket_success(self, ticket_manager, sample_ticket_data, db_session):
        """Test successful ticket creation."""
        # Mock database operations
        db_session.add = Mock()
        db_session.commit = Mock()
        db_session.refresh = Mock()
        
        # Create mock ticket
        mock_ticket = Mock(spec=SupportTicket)
        mock_ticket.ticket_number = "ST-20240115-ABC12345"
        db_session.refresh.side_effect = lambda obj: setattr(obj, 'ticket_number', mock_ticket.ticket_number)
        
        with patch.object(ticket_manager, '_generate_ticket_number', return_value="ST-20240115-ABC12345"):
            with patch.object(ticket_manager, '_send_ticket_confirmation'):
                ticket = ticket_manager.create_ticket(sample_ticket_data)
                
                # Verify database operations
                db_session.add.assert_called_once()
                db_session.commit.assert_called_once()
                db_session.refresh.assert_called_once()
    
    def test_create_ticket_database_error(self, ticket_manager, sample_ticket_data, db_session):
        """Test ticket creation with database error."""
        # Mock database error
        db_session.add = Mock()
        db_session.commit = Mock(side_effect=Exception("Database error"))
        db_session.rollback = Mock()
        
        with patch.object(ticket_manager, '_generate_ticket_number', return_value="ST-20240115-ABC12345"):
            with pytest.raises(Exception):
                ticket_manager.create_ticket(sample_ticket_data)
                
            # Verify rollback was called
            db_session.rollback.assert_called_once()
    
    def test_add_message_success(self, ticket_manager, db_session):
        """Test adding message to ticket."""
        message_data = SupportMessageCreate(
            message="This is a test message",
            sender_name="John Doe",
            sender_email="john@example.com"
        )
        
        # Mock database operations
        db_session.add = Mock()
        db_session.commit = Mock()
        db_session.refresh = Mock()
        
        # Mock ticket query
        mock_ticket = Mock(spec=SupportTicket)
        db_session.query.return_value.filter.return_value.first.return_value = mock_ticket
        
        message = ticket_manager.add_message(1, message_data)
        
        # Verify database operations
        db_session.add.assert_called_once()
        db_session.commit.assert_called_once()
        db_session.refresh.assert_called_once()
    
    def test_update_ticket_status_success(self, ticket_manager, db_session):
        """Test updating ticket status."""
        # Mock ticket query
        mock_ticket = Mock(spec=SupportTicket)
        mock_ticket.ticket_number = "ST-20240115-ABC12345"
        db_session.query.return_value.filter.return_value.first.return_value = mock_ticket
        db_session.commit = Mock()
        db_session.refresh = Mock()
        
        updated_ticket = ticket_manager.update_ticket_status(
            1, TicketStatus.RESOLVED, "agent@example.com"
        )
        
        # Verify status update
        assert mock_ticket.status == TicketStatus.RESOLVED
        assert mock_ticket.assigned_to == "agent@example.com"
        assert mock_ticket.resolved_at is not None
        
        db_session.commit.assert_called_once()
        db_session.refresh.assert_called_once()
    
    def test_update_ticket_status_not_found(self, ticket_manager, db_session):
        """Test updating non-existent ticket."""
        # Mock empty query result
        db_session.query.return_value.filter.return_value.first.return_value = None
        
        with pytest.raises(ValueError, match="Ticket 1 not found"):
            ticket_manager.update_ticket_status(1, TicketStatus.RESOLVED)
    
    def test_get_ticket_metrics(self, ticket_manager, db_session):
        """Test getting ticket metrics."""
        # Mock query results
        db_session.query.return_value.count.return_value = 100
        db_session.query.return_value.filter.return_value.count.side_effect = [20, 60]
        
        # Mock resolved tickets with timing
        mock_resolved_tickets = [
            Mock(
                resolved_at=datetime.utcnow(),
                created_at=datetime.utcnow() - timedelta(hours=2)
            ),
            Mock(
                resolved_at=datetime.utcnow(),
                created_at=datetime.utcnow() - timedelta(hours=4)
            )
        ]
        db_session.query.return_value.filter.return_value.all.return_value = mock_resolved_tickets
        
        metrics = ticket_manager.get_ticket_metrics()
        
        assert metrics["total_tickets"] == 100
        assert metrics["open_tickets"] == 20
        assert metrics["resolved_tickets"] == 60
        assert metrics["resolution_rate"] == 0.6
        assert "avg_resolution_hours" in metrics
    
    def test_generate_ticket_number(self, ticket_manager):
        """Test ticket number generation."""
        ticket_number = ticket_manager._generate_ticket_number()
        
        assert ticket_number.startswith("ST-")
        assert len(ticket_number) == 20  # ST-YYYYMMDD-XXXXXXXX
    
    def test_search_tickets(self, ticket_manager, db_session):
        """Test ticket search functionality."""
        # Mock query chain
        mock_query = Mock()
        db_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value.all.return_value = []
        
        results = ticket_manager.search_tickets(
            query="test",
            status=TicketStatus.OPEN,
            priority=TicketPriority.HIGH,
            limit=25
        )
        
        # Verify query was built correctly
        mock_query.filter.assert_called()
        mock_query.order_by.assert_called()
        mock_query.limit.assert_called_with(25)

class TestKnowledgeBaseManager:
    """Test knowledge base management functionality."""
    
    @pytest.fixture
    def db_session(self):
        """Mock database session."""
        return Mock(spec=Session)
    
    @pytest.fixture
    def kb_manager(self, db_session):
        """Create knowledge base manager instance."""
        return KnowledgeBaseManager(db_session)
    
    def test_get_articles_success(self, kb_manager, db_session):
        """Test getting knowledge base articles."""
        # Mock query chain
        mock_query = Mock()
        db_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value.all.return_value = []
        
        articles = kb_manager.get_articles(category="getting_started", limit=10)
        
        # Verify query was built correctly
        mock_query.filter.assert_called()
        mock_query.order_by.assert_called()
        mock_query.limit.assert_called_with(10)
    
    def test_get_article_by_slug_increments_views(self, kb_manager, db_session):
        """Test that getting article by slug increments view count."""
        # Mock article
        mock_article = Mock(spec=KnowledgeBaseArticle)
        mock_article.view_count = 5
        
        # Mock query chain
        mock_query = Mock()
        db_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = mock_article
        db_session.commit = Mock()
        
        article = kb_manager.get_article_by_slug("test-article")
        
        # Verify view count was incremented
        assert mock_article.view_count == 6
        db_session.commit.assert_called_once()
    
    def test_search_help_content(self, kb_manager, db_session):
        """Test searching help content."""
        # Mock query results
        mock_articles = [Mock(spec=KnowledgeBaseArticle)]
        mock_faqs = [Mock(spec=FAQ)]
        
        # Mock query chains
        mock_query = Mock()
        db_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.limit.return_value.all.side_effect = [mock_articles, mock_faqs]
        
        results = kb_manager.search_help_content("test query", limit=20)
        
        assert "articles" in results
        assert "faqs" in results
        assert results["articles"] == mock_articles
        assert results["faqs"] == mock_faqs
    
    def test_vote_helpful_article(self, kb_manager, db_session):
        """Test voting on article helpfulness."""
        # Mock article
        mock_article = Mock(spec=KnowledgeBaseArticle)
        mock_article.helpful_votes = 5
        mock_article.unhelpful_votes = 2
        
        # Mock query
        db_session.query.return_value.filter.return_value.first.return_value = mock_article
        db_session.commit = Mock()
        
        # Test helpful vote
        kb_manager.vote_helpful("article", 1, True)
        assert mock_article.helpful_votes == 6
        
        # Test unhelpful vote
        kb_manager.vote_helpful("article", 1, False)
        assert mock_article.unhelpful_votes == 3
        
        db_session.commit.assert_called()
    
    def test_vote_helpful_faq(self, kb_manager, db_session):
        """Test voting on FAQ helpfulness."""
        # Mock FAQ
        mock_faq = Mock(spec=FAQ)
        mock_faq.helpful_votes = 3
        mock_faq.unhelpful_votes = 1
        
        # Mock query
        db_session.query.return_value.filter.return_value.first.return_value = mock_faq
        db_session.commit = Mock()
        
        # Test helpful vote
        kb_manager.vote_helpful("faq", 1, True)
        assert mock_faq.helpful_votes == 4
        
        db_session.commit.assert_called()

class TestFeedbackManager:
    """Test feedback management functionality."""
    
    @pytest.fixture
    def db_session(self):
        """Mock database session."""
        return Mock(spec=Session)
    
    @pytest.fixture
    def feedback_manager(self, db_session):
        """Create feedback manager instance."""
        return FeedbackManager(db_session)
    
    @pytest.fixture
    def sample_feedback_data(self):
        """Sample feedback data."""
        return FeedbackCreate(
            feedback_type="feature_request",
            title="New Feature Request",
            description="Please add this new feature",
            rating=4,
            email="user@example.com",
            user_id="user123"
        )
    
    def test_submit_feedback_success(self, feedback_manager, sample_feedback_data, db_session):
        """Test successful feedback submission."""
        # Mock database operations
        db_session.add = Mock()
        db_session.commit = Mock()
        db_session.refresh = Mock()
        
        feedback = feedback_manager.submit_feedback(sample_feedback_data)
        
        # Verify database operations
        db_session.add.assert_called_once()
        db_session.commit.assert_called_once()
        db_session.refresh.assert_called_once()
    
    def test_submit_feedback_database_error(self, feedback_manager, sample_feedback_data, db_session):
        """Test feedback submission with database error."""
        # Mock database error
        db_session.add = Mock()
        db_session.commit = Mock(side_effect=Exception("Database error"))
        db_session.rollback = Mock()
        
        with pytest.raises(Exception):
            feedback_manager.submit_feedback(sample_feedback_data)
            
        # Verify rollback was called
        db_session.rollback.assert_called_once()
    
    def test_get_feedback_summary(self, feedback_manager, db_session):
        """Test getting feedback summary."""
        # Mock query results
        db_session.query.return_value.count.return_value = 50
        db_session.query.return_value.group_by.return_value.all.return_value = [
            ("bug_report", 20),
            ("feature_request", 30)
        ]
        db_session.query.return_value.filter.return_value.scalar.return_value = 4.2
        
        summary = feedback_manager.get_feedback_summary()
        
        assert summary["total_feedback"] == 50
        assert summary["feedback_by_type"]["bug_report"] == 20
        assert summary["feedback_by_type"]["feature_request"] == 30
        assert summary["average_rating"] == 4.2

class TestContactFormManager:
    """Test contact form management functionality."""
    
    @pytest.fixture
    def db_session(self):
        """Mock database session."""
        return Mock(spec=Session)
    
    @pytest.fixture
    def contact_manager(self, db_session):
        """Create contact form manager instance."""
        return ContactFormManager(db_session)
    
    @pytest.fixture
    def sample_contact_data(self):
        """Sample contact form data."""
        return ContactFormSubmission(
            name="John Doe",
            email="john@example.com",
            company="Test Company",
            subject="Sales Inquiry",
            message="I'm interested in your services",
            phone="555-1234",
            inquiry_type="sales"
        )
    
    def test_submit_contact_form_success(self, contact_manager, sample_contact_data, db_session):
        """Test successful contact form submission."""
        # Mock ticket manager
        with patch('scrollintel.core.support_system.SupportTicketManager') as mock_ticket_manager:
            mock_ticket_instance = Mock()
            mock_ticket_manager.return_value = mock_ticket_instance
            
            mock_ticket = Mock(spec=SupportTicket)
            mock_ticket.ticket_number = "ST-20240115-ABC12345"
            mock_ticket_instance.create_ticket.return_value = mock_ticket
            
            ticket = contact_manager.submit_contact_form(sample_contact_data)
            
            # Verify ticket was created
            mock_ticket_instance.create_ticket.assert_called_once()
            assert ticket.ticket_number == "ST-20240115-ABC12345"
    
    def test_submit_contact_form_error(self, contact_manager, sample_contact_data, db_session):
        """Test contact form submission with error."""
        # Mock ticket manager error
        with patch('scrollintel.core.support_system.SupportTicketManager') as mock_ticket_manager:
            mock_ticket_instance = Mock()
            mock_ticket_manager.return_value = mock_ticket_instance
            mock_ticket_instance.create_ticket.side_effect = Exception("Ticket creation failed")
            
            with pytest.raises(Exception):
                contact_manager.submit_contact_form(sample_contact_data)

class TestSupportSystemIntegration:
    """Integration tests for the support system."""
    
    def test_ticket_to_knowledge_base_workflow(self):
        """Test workflow from ticket creation to knowledge base update."""
        # This would test the complete workflow of:
        # 1. User creates ticket
        # 2. Agent resolves ticket
        # 3. Solution is added to knowledge base
        # 4. Future users find solution in KB
        pass
    
    def test_feedback_to_feature_development_workflow(self):
        """Test workflow from feedback to feature development."""
        # This would test the complete workflow of:
        # 1. User submits feature request
        # 2. Feedback is reviewed and prioritized
        # 3. Feature is developed
        # 4. User is notified of feature completion
        pass
    
    def test_contact_form_to_sales_workflow(self):
        """Test workflow from contact form to sales process."""
        # This would test the complete workflow of:
        # 1. User submits contact form
        # 2. Sales team is notified
        # 3. Follow-up communication occurs
        # 4. Lead is tracked through sales pipeline
        pass

if __name__ == "__main__":
    pytest.main([__file__])