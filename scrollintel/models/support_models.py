"""
Support system data models for ScrollIntel Launch MVP.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class TicketStatus(str, Enum):
    """Support ticket status enumeration."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    WAITING_FOR_CUSTOMER = "waiting_for_customer"
    RESOLVED = "resolved"
    CLOSED = "closed"

class TicketPriority(str, Enum):
    """Support ticket priority enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class SupportTicket(Base):
    """Support ticket database model."""
    __tablename__ = "support_tickets"
    
    id = Column(Integer, primary_key=True, index=True)
    ticket_number = Column(String(20), unique=True, index=True)
    user_id = Column(String(50), index=True)
    email = Column(String(255), nullable=False)
    subject = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    status = Column(String(50), default=TicketStatus.OPEN)
    priority = Column(String(20), default=TicketPriority.MEDIUM)
    category = Column(String(100))
    tags = Column(JSON)
    assigned_to = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    resolved_at = Column(DateTime)
    
    # Relationships
    messages = relationship("SupportMessage", back_populates="ticket")

class SupportMessage(Base):
    """Support ticket message database model."""
    __tablename__ = "support_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(Integer, ForeignKey("support_tickets.id"))
    sender_type = Column(String(20))  # 'customer' or 'agent'
    sender_name = Column(String(255))
    sender_email = Column(String(255))
    message = Column(Text, nullable=False)
    is_internal = Column(Boolean, default=False)
    attachments = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    ticket = relationship("SupportTicket", back_populates="messages")

class KnowledgeBaseArticle(Base):
    """Knowledge base article database model."""
    __tablename__ = "kb_articles"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False)
    slug = Column(String(200), unique=True, index=True)
    content = Column(Text, nullable=False)
    summary = Column(Text)
    category = Column(String(100))
    tags = Column(JSON)
    author = Column(String(255))
    status = Column(String(20), default="published")  # draft, published, archived
    view_count = Column(Integer, default=0)
    helpful_votes = Column(Integer, default=0)
    unhelpful_votes = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class FAQ(Base):
    """FAQ database model."""
    __tablename__ = "faqs"
    
    id = Column(Integer, primary_key=True, index=True)
    question = Column(String(1000), nullable=False)
    answer = Column(Text, nullable=False)
    category = Column(String(100))
    order_index = Column(Integer, default=0)
    is_featured = Column(Boolean, default=False)
    view_count = Column(Integer, default=0)
    helpful_votes = Column(Integer, default=0)
    unhelpful_votes = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class UserFeedback(Base):
    """User feedback database model."""
    __tablename__ = "user_feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(50), index=True)
    email = Column(String(255))
    feedback_type = Column(String(50))  # bug_report, feature_request, general
    title = Column(String(500))
    description = Column(Text, nullable=False)
    rating = Column(Integer)  # 1-5 stars
    page_url = Column(String(1000))
    user_agent = Column(String(500))
    status = Column(String(50), default="new")  # new, reviewed, in_progress, completed
    priority = Column(String(20), default="medium")
    tags = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Pydantic models for API
class SupportTicketCreate(BaseModel):
    """Support ticket creation model."""
    email: str = Field(..., description="Customer email address")
    subject: str = Field(..., max_length=500, description="Ticket subject")
    description: str = Field(..., description="Detailed description of the issue")
    category: Optional[str] = Field(None, description="Issue category")
    priority: TicketPriority = Field(TicketPriority.MEDIUM, description="Ticket priority")
    user_id: Optional[str] = Field(None, description="User ID if authenticated")

class SupportMessageCreate(BaseModel):
    """Support message creation model."""
    message: str = Field(..., description="Message content")
    sender_name: Optional[str] = Field(None, description="Sender name")
    sender_email: Optional[str] = Field(None, description="Sender email")
    attachments: Optional[List[Dict[str, Any]]] = Field(None, description="Message attachments")

class SupportTicketResponse(BaseModel):
    """Support ticket response model."""
    id: int
    ticket_number: str
    subject: str
    description: str
    status: TicketStatus
    priority: TicketPriority
    category: Optional[str]
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime]
    
    class Config:
        from_attributes = True

class KnowledgeBaseArticleResponse(BaseModel):
    """Knowledge base article response model."""
    id: int
    title: str
    slug: str
    content: str
    summary: Optional[str]
    category: Optional[str]
    tags: Optional[List[str]]
    view_count: int
    helpful_votes: int
    unhelpful_votes: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class FAQResponse(BaseModel):
    """FAQ response model."""
    id: int
    question: str
    answer: str
    category: Optional[str]
    is_featured: bool
    view_count: int
    helpful_votes: int
    unhelpful_votes: int
    
    class Config:
        from_attributes = True

class FeedbackCreate(BaseModel):
    """User feedback creation model."""
    feedback_type: str = Field(..., description="Type of feedback")
    title: Optional[str] = Field(None, max_length=500, description="Feedback title")
    description: str = Field(..., description="Detailed feedback description")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating from 1-5 stars")
    page_url: Optional[str] = Field(None, description="URL where feedback was submitted")
    email: Optional[str] = Field(None, description="User email for follow-up")
    user_id: Optional[str] = Field(None, description="User ID if authenticated")

class ContactFormSubmission(BaseModel):
    """Contact form submission model."""
    name: str = Field(..., max_length=255, description="Contact name")
    email: str = Field(..., description="Contact email")
    company: Optional[str] = Field(None, max_length=255, description="Company name")
    subject: str = Field(..., max_length=500, description="Message subject")
    message: str = Field(..., description="Message content")
    phone: Optional[str] = Field(None, max_length=50, description="Phone number")
    inquiry_type: Optional[str] = Field(None, description="Type of inquiry")