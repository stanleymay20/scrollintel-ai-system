"""
User Onboarding and Support System
Handles user registration, guided onboarding, and basic support
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import json
import uuid
from sqlalchemy.orm import Session
from passlib.context import CryptContext
import jwt
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

logger = logging.getLogger(__name__)

class OnboardingStep(Enum):
    REGISTRATION = "registration"
    EMAIL_VERIFICATION = "email_verification"
    PROFILE_SETUP = "profile_setup"
    TUTORIAL_INTRO = "tutorial_intro"
    FIRST_AGENT_INTERACTION = "first_agent_interaction"
    FEATURE_TOUR = "feature_tour"
    COMPLETED = "completed"

class SupportTicketStatus(Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"

class SupportTicketPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class User:
    id: str
    email: str
    username: str
    password_hash: str
    is_verified: bool = False
    onboarding_step: OnboardingStep = OnboardingStep.REGISTRATION
    created_at: datetime = None
    last_login: datetime = None
    profile_data: Dict[str, Any] = None

@dataclass
class OnboardingProgress:
    user_id: str
    current_step: OnboardingStep
    completed_steps: List[OnboardingStep]
    step_data: Dict[str, Any]
    started_at: datetime
    completed_at: Optional[datetime] = None

@dataclass
class SupportTicket:
    id: str
    user_id: str
    subject: str
    description: str
    status: SupportTicketStatus
    priority: SupportTicketPriority
    category: str
    created_at: datetime
    updated_at: datetime
    assigned_to: Optional[str] = None
    resolution: Optional[str] = None

class UserOnboardingSystem:
    """Comprehensive user onboarding and support system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.jwt_secret = config.get('jwt_secret', 'your-secret-key')
        self.jwt_algorithm = config.get('jwt_algorithm', 'HS256')
        self.email_service = EmailService(config.get('email', {}))
        self.tutorial_manager = TutorialManager()
        self.support_system = SupportSystem()
        
    async def register_user(self, email: str, username: str, password: str) -> Dict[str, Any]:
        """Register a new user"""
        try:
            # Validate input
            if not self._validate_email(email):
                return {'success': False, 'error': 'Invalid email format'}
                
            if not self._validate_password(password):
                return {'success': False, 'error': 'Password does not meet requirements'}
            
            # Check if user already exists
            if await self._user_exists(email, username):
                return {'success': False, 'error': 'User already exists'}
            
            # Create user
            user_id = str(uuid.uuid4())
            password_hash = self.pwd_context.hash(password)
            
            user = User(
                id=user_id,
                email=email,
                username=username,
                password_hash=password_hash,
                created_at=datetime.utcnow(),
                profile_data={}
            )
            
            # Save user to database
            await self._save_user(user)
            
            # Initialize onboarding
            onboarding = OnboardingProgress(
                user_id=user_id,
                current_step=OnboardingStep.EMAIL_VERIFICATION,
                completed_steps=[OnboardingStep.REGISTRATION],
                step_data={},
                started_at=datetime.utcnow()
            )
            
            await self._save_onboarding_progress(onboarding)
            
            # Send verification email
            verification_token = self._generate_verification_token(user_id)
            await self.email_service.send_verification_email(email, verification_token)
            
            logger.info(f"User registered successfully: {email}")
            
            return {
                'success': True,
                'user_id': user_id,
                'message': 'Registration successful. Please check your email for verification.'
            }
            
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return {'success': False, 'error': 'Registration failed'}
    
    async def verify_email(self, token: str) -> Dict[str, Any]:
        """Verify user email with token"""
        try:
            user_id = self._verify_token(token)
            if not user_id:
                return {'success': False, 'error': 'Invalid or expired token'}
            
            # Update user verification status
            await self._update_user_verification(user_id, True)
            
            # Update onboarding progress
            await self._advance_onboarding_step(user_id, OnboardingStep.PROFILE_SETUP)
            
            return {'success': True, 'message': 'Email verified successfully'}
            
        except Exception as e:
            logger.error(f"Email verification failed: {e}")
            return {'success': False, 'error': 'Verification failed'}
    
    async def authenticate_user(self, email: str, password: str) -> Dict[str, Any]:
        """Authenticate user login"""
        try:
            user = await self._get_user_by_email(email)
            if not user:
                return {'success': False, 'error': 'Invalid credentials'}
            
            if not self.pwd_context.verify(password, user.password_hash):
                return {'success': False, 'error': 'Invalid credentials'}
            
            if not user.is_verified:
                return {'success': False, 'error': 'Email not verified'}
            
            # Update last login
            await self._update_last_login(user.id)
            
            # Generate JWT token
            access_token = self._generate_access_token(user.id)
            
            return {
                'success': True,
                'access_token': access_token,
                'user': {
                    'id': user.id,
                    'email': user.email,
                    'username': user.username,
                    'onboarding_step': user.onboarding_step.value
                }
            }
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return {'success': False, 'error': 'Authentication failed'}
    
    async def get_onboarding_status(self, user_id: str) -> Dict[str, Any]:
        """Get user's onboarding status and next steps"""
        try:
            progress = await self._get_onboarding_progress(user_id)
            if not progress:
                return {'success': False, 'error': 'Onboarding not found'}
            
            next_step_info = self.tutorial_manager.get_step_info(progress.current_step)
            
            return {
                'success': True,
                'current_step': progress.current_step.value,
                'completed_steps': [step.value for step in progress.completed_steps],
                'progress_percentage': len(progress.completed_steps) / len(OnboardingStep) * 100,
                'next_step_info': next_step_info,
                'step_data': progress.step_data
            }
            
        except Exception as e:
            logger.error(f"Failed to get onboarding status: {e}")
            return {'success': False, 'error': 'Failed to get onboarding status'}
    
    async def complete_onboarding_step(self, user_id: str, step: OnboardingStep, 
                                     step_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Complete an onboarding step"""
        try:
            progress = await self._get_onboarding_progress(user_id)
            if not progress:
                return {'success': False, 'error': 'Onboarding not found'}
            
            if step != progress.current_step:
                return {'success': False, 'error': 'Invalid step sequence'}
            
            # Update progress
            progress.completed_steps.append(step)
            if step_data:
                progress.step_data.update(step_data)
            
            # Determine next step
            next_step = self._get_next_onboarding_step(step)
            progress.current_step = next_step
            
            if next_step == OnboardingStep.COMPLETED:
                progress.completed_at = datetime.utcnow()
            
            await self._save_onboarding_progress(progress)
            
            # Update user's onboarding step
            await self._update_user_onboarding_step(user_id, next_step)
            
            return {
                'success': True,
                'completed_step': step.value,
                'next_step': next_step.value,
                'is_completed': next_step == OnboardingStep.COMPLETED
            }
            
        except Exception as e:
            logger.error(f"Failed to complete onboarding step: {e}")
            return {'success': False, 'error': 'Failed to complete step'}
    
    async def create_support_ticket(self, user_id: str, subject: str, description: str,
                                  category: str, priority: SupportTicketPriority = SupportTicketPriority.MEDIUM) -> Dict[str, Any]:
        """Create a support ticket"""
        try:
            ticket_id = str(uuid.uuid4())
            
            ticket = SupportTicket(
                id=ticket_id,
                user_id=user_id,
                subject=subject,
                description=description,
                status=SupportTicketStatus.OPEN,
                priority=priority,
                category=category,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            await self.support_system.save_ticket(ticket)
            
            # Send notification to support team
            await self.support_system.notify_support_team(ticket)
            
            return {
                'success': True,
                'ticket_id': ticket_id,
                'message': 'Support ticket created successfully'
            }
            
        except Exception as e:
            logger.error(f"Failed to create support ticket: {e}")
            return {'success': False, 'error': 'Failed to create ticket'}
    
    def _validate_email(self, email: str) -> bool:
        """Validate email format"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def _validate_password(self, password: str) -> bool:
        """Validate password strength"""
        return (len(password) >= 8 and 
                any(c.isupper() for c in password) and
                any(c.islower() for c in password) and
                any(c.isdigit() for c in password))
    
    def _generate_verification_token(self, user_id: str) -> str:
        """Generate email verification token"""
        payload = {
            'user_id': user_id,
            'type': 'email_verification',
            'exp': datetime.utcnow() + timedelta(hours=24)
        }
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def _generate_access_token(self, user_id: str) -> str:
        """Generate JWT access token"""
        payload = {
            'user_id': user_id,
            'type': 'access_token',
            'exp': datetime.utcnow() + timedelta(hours=24)
        }
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def _verify_token(self, token: str) -> Optional[str]:
        """Verify JWT token and return user_id"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return payload.get('user_id')
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def _get_next_onboarding_step(self, current_step: OnboardingStep) -> OnboardingStep:
        """Get next onboarding step"""
        steps = list(OnboardingStep)
        current_index = steps.index(current_step)
        
        if current_index < len(steps) - 1:
            return steps[current_index + 1]
        else:
            return OnboardingStep.COMPLETED
    
    # Database operations (would be implemented with actual database)
    async def _user_exists(self, email: str, username: str) -> bool:
        """Check if user exists"""
        # Implementation would check database
        return False
    
    async def _save_user(self, user: User):
        """Save user to database"""
        # Implementation would save to database
        pass
    
    async def _get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        # Implementation would query database
        return None
    
    async def _update_user_verification(self, user_id: str, is_verified: bool):
        """Update user verification status"""
        # Implementation would update database
        pass
    
    async def _update_last_login(self, user_id: str):
        """Update user's last login time"""
        # Implementation would update database
        pass
    
    async def _update_user_onboarding_step(self, user_id: str, step: OnboardingStep):
        """Update user's onboarding step"""
        # Implementation would update database
        pass
    
    async def _save_onboarding_progress(self, progress: OnboardingProgress):
        """Save onboarding progress"""
        # Implementation would save to database
        pass
    
    async def _get_onboarding_progress(self, user_id: str) -> Optional[OnboardingProgress]:
        """Get onboarding progress"""
        # Implementation would query database
        return None
    
    async def _advance_onboarding_step(self, user_id: str, next_step: OnboardingStep):
        """Advance to next onboarding step"""
        # Implementation would update database
        pass

class EmailService:
    """Email service for notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.smtp_server = config.get('smtp_server', 'localhost')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username', '')
        self.password = config.get('password', '')
        self.from_email = config.get('from_email', 'noreply@scrollintel.com')
    
    async def send_verification_email(self, email: str, token: str):
        """Send email verification"""
        try:
            verification_url = f"{self.config.get('base_url', '')}/verify-email?token={token}"
            
            subject = "Verify Your ScrollIntel Account"
            body = f"""
            Welcome to ScrollIntel!
            
            Please click the link below to verify your email address:
            {verification_url}
            
            This link will expire in 24 hours.
            
            If you didn't create this account, please ignore this email.
            
            Best regards,
            The ScrollIntel Team
            """
            
            await self._send_email(email, subject, body)
            
        except Exception as e:
            logger.error(f"Failed to send verification email: {e}")
    
    async def _send_email(self, to_email: str, subject: str, body: str):
        """Send email using SMTP"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = to_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            text = msg.as_string()
            server.sendmail(self.from_email, to_email, text)
            server.quit()
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")

class TutorialManager:
    """Manages onboarding tutorials and guides"""
    
    def __init__(self):
        self.step_info = {
            OnboardingStep.REGISTRATION: {
                'title': 'Welcome to ScrollIntel',
                'description': 'Create your account to get started',
                'estimated_time': '2 minutes',
                'actions': ['Fill registration form', 'Submit']
            },
            OnboardingStep.EMAIL_VERIFICATION: {
                'title': 'Verify Your Email',
                'description': 'Check your email and click the verification link',
                'estimated_time': '1 minute',
                'actions': ['Check email', 'Click verification link']
            },
            OnboardingStep.PROFILE_SETUP: {
                'title': 'Set Up Your Profile',
                'description': 'Tell us about yourself and your goals',
                'estimated_time': '3 minutes',
                'actions': ['Add profile picture', 'Fill profile information', 'Set preferences']
            },
            OnboardingStep.TUTORIAL_INTRO: {
                'title': 'ScrollIntel Introduction',
                'description': 'Learn about ScrollIntel\'s key features',
                'estimated_time': '5 minutes',
                'actions': ['Watch intro video', 'Read feature overview']
            },
            OnboardingStep.FIRST_AGENT_INTERACTION: {
                'title': 'Meet Your First Agent',
                'description': 'Interact with an AI agent to see the magic',
                'estimated_time': '3 minutes',
                'actions': ['Start chat', 'Ask a question', 'Explore responses']
            },
            OnboardingStep.FEATURE_TOUR: {
                'title': 'Feature Tour',
                'description': 'Explore the main features of ScrollIntel',
                'estimated_time': '10 minutes',
                'actions': ['Visit dashboard', 'Try different agents', 'Explore settings']
            }
        }
    
    def get_step_info(self, step: OnboardingStep) -> Dict[str, Any]:
        """Get information about an onboarding step"""
        return self.step_info.get(step, {})
    
    def get_tutorial_content(self, step: OnboardingStep) -> Dict[str, Any]:
        """Get detailed tutorial content for a step"""
        # This would return rich content including videos, images, interactive elements
        return {
            'step': step.value,
            'content': self.step_info.get(step, {}),
            'interactive_elements': [],
            'help_links': []
        }

class SupportSystem:
    """Support ticket management system"""
    
    def __init__(self):
        self.tickets = {}  # In production, this would be a database
        self.support_team_emails = ['support@scrollintel.com']
    
    async def save_ticket(self, ticket: SupportTicket):
        """Save support ticket"""
        self.tickets[ticket.id] = ticket
    
    async def get_ticket(self, ticket_id: str) -> Optional[SupportTicket]:
        """Get support ticket by ID"""
        return self.tickets.get(ticket_id)
    
    async def update_ticket_status(self, ticket_id: str, status: SupportTicketStatus):
        """Update ticket status"""
        if ticket_id in self.tickets:
            self.tickets[ticket_id].status = status
            self.tickets[ticket_id].updated_at = datetime.utcnow()
    
    async def notify_support_team(self, ticket: SupportTicket):
        """Notify support team of new ticket"""
        # Implementation would send notification to support team
        logger.info(f"New support ticket created: {ticket.id} - {ticket.subject}")
    
    async def get_user_tickets(self, user_id: str) -> List[SupportTicket]:
        """Get all tickets for a user"""
        return [ticket for ticket in self.tickets.values() 
                if ticket.user_id == user_id]

# Global onboarding system instance
onboarding_system = None

def initialize_onboarding_system(config: Dict[str, Any]) -> UserOnboardingSystem:
    """Initialize global onboarding system"""
    global onboarding_system
    onboarding_system = UserOnboardingSystem(config)
    return onboarding_system

def get_onboarding_system() -> UserOnboardingSystem:
    """Get global onboarding system instance"""
    if onboarding_system is None:
        raise RuntimeError("Onboarding system not initialized")
    return onboarding_system