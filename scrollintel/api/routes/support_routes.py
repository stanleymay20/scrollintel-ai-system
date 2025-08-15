"""
Support system API routes for ScrollIntel Launch MVP.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session

from scrollintel.core.support_system import (
    SupportTicketManager, KnowledgeBaseManager, 
    FeedbackManager, ContactFormManager
)
from scrollintel.models.support_models import (
    SupportTicketCreate, SupportMessageCreate, SupportTicketResponse,
    KnowledgeBaseArticleResponse, FAQResponse, FeedbackCreate,
    ContactFormSubmission, TicketStatus, TicketPriority
)
from scrollintel.models.database import get_db

router = APIRouter(prefix="/api/support", tags=["support"])

# Support Tickets
@router.post("/tickets", response_model=SupportTicketResponse)
async def create_support_ticket(
    ticket_data: SupportTicketCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create a new support ticket."""
    try:
        ticket_manager = SupportTicketManager(db)
        ticket = ticket_manager.create_ticket(ticket_data)
        
        # Send notification email in background
        background_tasks.add_task(send_ticket_notification, ticket.ticket_number)
        
        return ticket
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating ticket: {str(e)}")

@router.get("/tickets/{ticket_number}", response_model=SupportTicketResponse)
async def get_support_ticket(
    ticket_number: str,
    db: Session = Depends(get_db)
):
    """Get support ticket by ticket number."""
    ticket_manager = SupportTicketManager(db)
    ticket = ticket_manager.get_ticket_by_number(ticket_number)
    
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    return ticket

@router.get("/tickets", response_model=List[SupportTicketResponse])
async def search_support_tickets(
    query: Optional[str] = Query(None, description="Search query"),
    status: Optional[TicketStatus] = Query(None, description="Filter by status"),
    priority: Optional[TicketPriority] = Query(None, description="Filter by priority"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    limit: int = Query(50, le=100, description="Maximum number of results"),
    db: Session = Depends(get_db)
):
    """Search support tickets."""
    ticket_manager = SupportTicketManager(db)
    
    if user_id:
        tickets = ticket_manager.get_user_tickets(user_id, limit)
    else:
        tickets = ticket_manager.search_tickets(query, status, priority, limit)
    
    return tickets

@router.post("/tickets/{ticket_id}/messages")
async def add_ticket_message(
    ticket_id: int,
    message_data: SupportMessageCreate,
    db: Session = Depends(get_db)
):
    """Add a message to an existing ticket."""
    try:
        ticket_manager = SupportTicketManager(db)
        message = ticket_manager.add_message(ticket_id, message_data)
        return {"message": "Message added successfully", "message_id": message.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding message: {str(e)}")

@router.put("/tickets/{ticket_id}/status")
async def update_ticket_status(
    ticket_id: int,
    status: TicketStatus,
    assigned_to: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Update ticket status."""
    try:
        ticket_manager = SupportTicketManager(db)
        ticket = ticket_manager.update_ticket_status(ticket_id, status, assigned_to)
        return {"message": "Ticket status updated", "ticket": ticket}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating ticket: {str(e)}")

@router.get("/tickets/metrics")
async def get_ticket_metrics(db: Session = Depends(get_db)):
    """Get support ticket metrics."""
    ticket_manager = SupportTicketManager(db)
    metrics = ticket_manager.get_ticket_metrics()
    return metrics

# Knowledge Base
@router.get("/kb/articles", response_model=List[KnowledgeBaseArticleResponse])
async def get_kb_articles(
    category: Optional[str] = Query(None, description="Filter by category"),
    search: Optional[str] = Query(None, description="Search query"),
    limit: int = Query(50, le=100, description="Maximum number of results"),
    db: Session = Depends(get_db)
):
    """Get knowledge base articles."""
    kb_manager = KnowledgeBaseManager(db)
    articles = kb_manager.get_articles(category, search, limit)
    return articles

@router.get("/kb/articles/{slug}", response_model=KnowledgeBaseArticleResponse)
async def get_kb_article(
    slug: str,
    db: Session = Depends(get_db)
):
    """Get knowledge base article by slug."""
    kb_manager = KnowledgeBaseManager(db)
    article = kb_manager.get_article_by_slug(slug)
    
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    
    return article

@router.get("/kb/faqs", response_model=List[FAQResponse])
async def get_faqs(
    category: Optional[str] = Query(None, description="Filter by category"),
    featured: bool = Query(False, description="Show only featured FAQs"),
    db: Session = Depends(get_db)
):
    """Get FAQ entries."""
    kb_manager = KnowledgeBaseManager(db)
    faqs = kb_manager.get_faqs(category, featured)
    return faqs

@router.get("/kb/search")
async def search_help_content(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, le=50, description="Maximum number of results"),
    db: Session = Depends(get_db)
):
    """Search help content across articles and FAQs."""
    kb_manager = KnowledgeBaseManager(db)
    results = kb_manager.search_help_content(q, limit)
    return results

@router.post("/kb/vote")
async def vote_helpful(
    content_type: str,
    content_id: int,
    helpful: bool,
    db: Session = Depends(get_db)
):
    """Vote on content helpfulness."""
    if content_type not in ["article", "faq"]:
        raise HTTPException(status_code=400, detail="Invalid content type")
    
    kb_manager = KnowledgeBaseManager(db)
    kb_manager.vote_helpful(content_type, content_id, helpful)
    return {"message": "Vote recorded successfully"}

# Feedback
@router.post("/feedback")
async def submit_feedback(
    feedback_data: FeedbackCreate,
    db: Session = Depends(get_db)
):
    """Submit user feedback."""
    try:
        feedback_manager = FeedbackManager(db)
        feedback = feedback_manager.submit_feedback(feedback_data)
        return {"message": "Feedback submitted successfully", "feedback_id": feedback.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting feedback: {str(e)}")

@router.get("/feedback/summary")
async def get_feedback_summary(db: Session = Depends(get_db)):
    """Get feedback summary and metrics."""
    feedback_manager = FeedbackManager(db)
    summary = feedback_manager.get_feedback_summary()
    return summary

# Contact Form
@router.post("/contact")
async def submit_contact_form(
    form_data: ContactFormSubmission,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Submit contact form."""
    try:
        contact_manager = ContactFormManager(db)
        ticket = contact_manager.submit_contact_form(form_data)
        
        # Send confirmation email in background
        background_tasks.add_task(send_contact_confirmation, form_data.email, ticket.ticket_number)
        
        return {
            "message": "Contact form submitted successfully",
            "ticket_number": ticket.ticket_number
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting contact form: {str(e)}")

# Chat Support (WebSocket endpoint would be implemented separately)
@router.get("/chat/status")
async def get_chat_status():
    """Get chat support availability status."""
    # This would check agent availability, business hours, etc.
    return {
        "available": True,
        "estimated_wait_time": "< 2 minutes",
        "business_hours": "9 AM - 6 PM EST, Monday - Friday"
    }

# Help Categories
@router.get("/categories")
async def get_help_categories(db: Session = Depends(get_db)):
    """Get available help categories."""
    # This would query the database for available categories
    return {
        "ticket_categories": [
            "technical_issue",
            "billing_question",
            "feature_request",
            "account_access",
            "data_processing",
            "api_integration",
            "general_inquiry"
        ],
        "kb_categories": [
            "getting_started",
            "data_upload",
            "ai_agents",
            "billing",
            "api_documentation",
            "troubleshooting",
            "best_practices"
        ],
        "feedback_types": [
            "bug_report",
            "feature_request",
            "general_feedback",
            "user_experience",
            "performance_issue"
        ]
    }

# Background tasks
async def send_ticket_notification(ticket_number: str):
    """Send ticket creation notification (placeholder)."""
    # This would integrate with email service
    print(f"Sending notification for ticket {ticket_number}")

async def send_contact_confirmation(email: str, ticket_number: str):
    """Send contact form confirmation (placeholder)."""
    # This would integrate with email service
    print(f"Sending confirmation to {email} for ticket {ticket_number}")

# Health check
@router.get("/health")
async def support_health_check():
    """Support system health check."""
    return {
        "status": "healthy",
        "services": {
            "ticket_system": "operational",
            "knowledge_base": "operational",
            "feedback_system": "operational",
            "chat_support": "operational"
        }
    }