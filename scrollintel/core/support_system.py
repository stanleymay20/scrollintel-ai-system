"""
Core support system for ScrollIntel Launch MVP.
Handles ticket management, knowledge base, and customer support operations.
"""

import uuid
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func

from scrollintel.models.support_models import (
    SupportTicket, SupportMessage, KnowledgeBaseArticle, FAQ, UserFeedback,
    TicketStatus, TicketPriority, SupportTicketCreate, SupportMessageCreate,
    FeedbackCreate, ContactFormSubmission
)

logger = logging.getLogger(__name__)

class SupportTicketManager:
    """Manages support tickets and customer communications."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def create_ticket(self, ticket_data: SupportTicketCreate) -> SupportTicket:
        """Create a new support ticket."""
        try:
            # Generate unique ticket number
            ticket_number = self._generate_ticket_number()
            
            # Create ticket
            ticket = SupportTicket(
                ticket_number=ticket_number,
                user_id=ticket_data.user_id,
                email=ticket_data.email,
                subject=ticket_data.subject,
                description=ticket_data.description,
                category=ticket_data.category,
                priority=ticket_data.priority,
                status=TicketStatus.OPEN
            )
            
            self.db.add(ticket)
            self.db.commit()
            self.db.refresh(ticket)
            
            # Send confirmation email (would integrate with email service)
            self._send_ticket_confirmation(ticket)
            
            logger.info(f"Created support ticket {ticket_number}")
            return ticket
            
        except Exception as e:
            logger.error(f"Error creating support ticket: {str(e)}")
            self.db.rollback()
            raise
    
    def add_message(self, ticket_id: int, message_data: SupportMessageCreate, 
                   sender_type: str = "customer") -> SupportMessage:
        """Add a message to an existing ticket."""
        try:
            message = SupportMessage(
                ticket_id=ticket_id,
                sender_type=sender_type,
                sender_name=message_data.sender_name,
                sender_email=message_data.sender_email,
                message=message_data.message,
                attachments=message_data.attachments
            )
            
            self.db.add(message)
            
            # Update ticket timestamp
            ticket = self.db.query(SupportTicket).filter(
                SupportTicket.id == ticket_id
            ).first()
            if ticket:
                ticket.updated_at = datetime.utcnow()
            
            self.db.commit()
            self.db.refresh(message)
            
            logger.info(f"Added message to ticket {ticket_id}")
            return message
            
        except Exception as e:
            logger.error(f"Error adding message to ticket: {str(e)}")
            self.db.rollback()
            raise
    
    def update_ticket_status(self, ticket_id: int, status: TicketStatus, 
                           assigned_to: Optional[str] = None) -> SupportTicket:
        """Update ticket status and assignment."""
        try:
            ticket = self.db.query(SupportTicket).filter(
                SupportTicket.id == ticket_id
            ).first()
            
            if not ticket:
                raise ValueError(f"Ticket {ticket_id} not found")
            
            ticket.status = status
            ticket.updated_at = datetime.utcnow()
            
            if assigned_to:
                ticket.assigned_to = assigned_to
            
            if status == TicketStatus.RESOLVED:
                ticket.resolved_at = datetime.utcnow()
            
            self.db.commit()
            self.db.refresh(ticket)
            
            logger.info(f"Updated ticket {ticket.ticket_number} status to {status}")
            return ticket
            
        except Exception as e:
            logger.error(f"Error updating ticket status: {str(e)}")
            self.db.rollback()
            raise
    
    def get_ticket_by_number(self, ticket_number: str) -> Optional[SupportTicket]:
        """Get ticket by ticket number."""
        return self.db.query(SupportTicket).filter(
            SupportTicket.ticket_number == ticket_number
        ).first()
    
    def get_user_tickets(self, user_id: str, limit: int = 50) -> List[SupportTicket]:
        """Get tickets for a specific user."""
        return self.db.query(SupportTicket).filter(
            SupportTicket.user_id == user_id
        ).order_by(desc(SupportTicket.created_at)).limit(limit).all()
    
    def search_tickets(self, query: str, status: Optional[TicketStatus] = None,
                      priority: Optional[TicketPriority] = None,
                      limit: int = 50) -> List[SupportTicket]:
        """Search tickets by various criteria."""
        filters = []
        
        if query:
            filters.append(or_(
                SupportTicket.subject.ilike(f"%{query}%"),
                SupportTicket.description.ilike(f"%{query}%"),
                SupportTicket.ticket_number.ilike(f"%{query}%")
            ))
        
        if status:
            filters.append(SupportTicket.status == status)
        
        if priority:
            filters.append(SupportTicket.priority == priority)
        
        query_obj = self.db.query(SupportTicket)
        if filters:
            query_obj = query_obj.filter(and_(*filters))
        
        return query_obj.order_by(desc(SupportTicket.created_at)).limit(limit).all()
    
    def get_ticket_metrics(self) -> Dict[str, Any]:
        """Get support ticket metrics."""
        try:
            total_tickets = self.db.query(SupportTicket).count()
            open_tickets = self.db.query(SupportTicket).filter(
                SupportTicket.status == TicketStatus.OPEN
            ).count()
            resolved_tickets = self.db.query(SupportTicket).filter(
                SupportTicket.status == TicketStatus.RESOLVED
            ).count()
            
            # Average resolution time
            resolved_with_time = self.db.query(SupportTicket).filter(
                and_(
                    SupportTicket.status == TicketStatus.RESOLVED,
                    SupportTicket.resolved_at.isnot(None)
                )
            ).all()
            
            avg_resolution_hours = 0
            if resolved_with_time:
                total_hours = sum([
                    (ticket.resolved_at - ticket.created_at).total_seconds() / 3600
                    for ticket in resolved_with_time
                ])
                avg_resolution_hours = total_hours / len(resolved_with_time)
            
            return {
                "total_tickets": total_tickets,
                "open_tickets": open_tickets,
                "resolved_tickets": resolved_tickets,
                "resolution_rate": resolved_tickets / total_tickets if total_tickets > 0 else 0,
                "avg_resolution_hours": round(avg_resolution_hours, 2)
            }
            
        except Exception as e:
            logger.error(f"Error getting ticket metrics: {str(e)}")
            return {}
    
    def _generate_ticket_number(self) -> str:
        """Generate unique ticket number."""
        timestamp = datetime.utcnow().strftime("%Y%m%d")
        random_suffix = str(uuid.uuid4())[:8].upper()
        return f"ST-{timestamp}-{random_suffix}"
    
    def _send_ticket_confirmation(self, ticket: SupportTicket):
        """Send ticket confirmation email (placeholder for email integration)."""
        # This would integrate with an email service like SendGrid, AWS SES, etc.
        logger.info(f"Sending confirmation email for ticket {ticket.ticket_number}")

class KnowledgeBaseManager:
    """Manages knowledge base articles and FAQ."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def get_articles(self, category: Optional[str] = None, 
                    search_query: Optional[str] = None,
                    limit: int = 50) -> List[KnowledgeBaseArticle]:
        """Get knowledge base articles."""
        query = self.db.query(KnowledgeBaseArticle).filter(
            KnowledgeBaseArticle.status == "published"
        )
        
        if category:
            query = query.filter(KnowledgeBaseArticle.category == category)
        
        if search_query:
            query = query.filter(or_(
                KnowledgeBaseArticle.title.ilike(f"%{search_query}%"),
                KnowledgeBaseArticle.content.ilike(f"%{search_query}%"),
                KnowledgeBaseArticle.summary.ilike(f"%{search_query}%")
            ))
        
        return query.order_by(desc(KnowledgeBaseArticle.view_count)).limit(limit).all()
    
    def get_article_by_slug(self, slug: str) -> Optional[KnowledgeBaseArticle]:
        """Get article by slug and increment view count."""
        article = self.db.query(KnowledgeBaseArticle).filter(
            and_(
                KnowledgeBaseArticle.slug == slug,
                KnowledgeBaseArticle.status == "published"
            )
        ).first()
        
        if article:
            article.view_count += 1
            self.db.commit()
        
        return article
    
    def get_faqs(self, category: Optional[str] = None, 
                featured_only: bool = False) -> List[FAQ]:
        """Get FAQ entries."""
        query = self.db.query(FAQ)
        
        if category:
            query = query.filter(FAQ.category == category)
        
        if featured_only:
            query = query.filter(FAQ.is_featured == True)
        
        return query.order_by(FAQ.order_index, FAQ.id).all()
    
    def search_help_content(self, query: str, limit: int = 20) -> Dict[str, List]:
        """Search across articles and FAQs."""
        # Search articles
        articles = self.db.query(KnowledgeBaseArticle).filter(
            and_(
                KnowledgeBaseArticle.status == "published",
                or_(
                    KnowledgeBaseArticle.title.ilike(f"%{query}%"),
                    KnowledgeBaseArticle.content.ilike(f"%{query}%"),
                    KnowledgeBaseArticle.summary.ilike(f"%{query}%")
                )
            )
        ).limit(limit // 2).all()
        
        # Search FAQs
        faqs = self.db.query(FAQ).filter(or_(
            FAQ.question.ilike(f"%{query}%"),
            FAQ.answer.ilike(f"%{query}%")
        )).limit(limit // 2).all()
        
        return {
            "articles": articles,
            "faqs": faqs
        }
    
    def vote_helpful(self, content_type: str, content_id: int, helpful: bool):
        """Record helpful/unhelpful vote."""
        try:
            if content_type == "article":
                article = self.db.query(KnowledgeBaseArticle).filter(
                    KnowledgeBaseArticle.id == content_id
                ).first()
                if article:
                    if helpful:
                        article.helpful_votes += 1
                    else:
                        article.unhelpful_votes += 1
            
            elif content_type == "faq":
                faq = self.db.query(FAQ).filter(FAQ.id == content_id).first()
                if faq:
                    if helpful:
                        faq.helpful_votes += 1
                    else:
                        faq.unhelpful_votes += 1
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error recording vote: {str(e)}")
            self.db.rollback()

class FeedbackManager:
    """Manages user feedback and feature requests."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def submit_feedback(self, feedback_data: FeedbackCreate) -> UserFeedback:
        """Submit user feedback."""
        try:
            feedback = UserFeedback(
                user_id=feedback_data.user_id,
                email=feedback_data.email,
                feedback_type=feedback_data.feedback_type,
                title=feedback_data.title,
                description=feedback_data.description,
                rating=feedback_data.rating,
                page_url=feedback_data.page_url
            )
            
            self.db.add(feedback)
            self.db.commit()
            self.db.refresh(feedback)
            
            logger.info(f"Submitted feedback: {feedback.feedback_type}")
            return feedback
            
        except Exception as e:
            logger.error(f"Error submitting feedback: {str(e)}")
            self.db.rollback()
            raise
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get feedback summary and metrics."""
        try:
            total_feedback = self.db.query(UserFeedback).count()
            
            # Feedback by type
            feedback_by_type = self.db.query(
                UserFeedback.feedback_type,
                func.count(UserFeedback.id).label('count')
            ).group_by(UserFeedback.feedback_type).all()
            
            # Average rating
            avg_rating = self.db.query(func.avg(UserFeedback.rating)).filter(
                UserFeedback.rating.isnot(None)
            ).scalar()
            
            return {
                "total_feedback": total_feedback,
                "feedback_by_type": dict(feedback_by_type),
                "average_rating": round(float(avg_rating or 0), 2)
            }
            
        except Exception as e:
            logger.error(f"Error getting feedback summary: {str(e)}")
            return {}

class ContactFormManager:
    """Manages contact form submissions."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def submit_contact_form(self, form_data: ContactFormSubmission) -> SupportTicket:
        """Submit contact form as support ticket."""
        try:
            # Convert contact form to support ticket
            ticket_data = SupportTicketCreate(
                email=form_data.email,
                subject=f"Contact Form: {form_data.subject}",
                description=f"Name: {form_data.name}\n"
                           f"Company: {form_data.company or 'Not provided'}\n"
                           f"Phone: {form_data.phone or 'Not provided'}\n"
                           f"Inquiry Type: {form_data.inquiry_type or 'General'}\n\n"
                           f"Message:\n{form_data.message}",
                category="contact_form",
                priority=TicketPriority.MEDIUM
            )
            
            ticket_manager = SupportTicketManager(self.db)
            ticket = ticket_manager.create_ticket(ticket_data)
            
            logger.info(f"Created ticket from contact form: {ticket.ticket_number}")
            return ticket
            
        except Exception as e:
            logger.error(f"Error submitting contact form: {str(e)}")
            raise