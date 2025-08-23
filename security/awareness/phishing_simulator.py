"""
Security Awareness and Phishing Simulation System
Provides comprehensive security awareness programs with phishing simulation and testing
"""

import json
import uuid
import random
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import logging

logger = logging.getLogger(__name__)

@dataclass
class PhishingTemplate:
    """Phishing email template"""
    id: str
    name: str
    category: str  # credential_harvesting, malware, social_engineering, etc.
    difficulty: str  # easy, medium, hard
    subject_line: str
    sender_name: str
    sender_email: str
    html_content: str
    text_content: str
    landing_page_url: str
    indicators: List[str]  # Red flags users should notice
    created_date: datetime
    success_rate: float  # Historical success rate

@dataclass
class PhishingCampaign:
    """Phishing simulation campaign"""
    id: str
    name: str
    description: str
    template_ids: List[str]
    target_groups: List[str]
    start_date: datetime
    end_date: datetime
    frequency: str  # one_time, weekly, monthly
    status: str  # planned, active, completed, paused
    created_by: str
    metrics: Dict[str, Any]

@dataclass
class PhishingResult:
    """Individual phishing test result"""
    id: str
    campaign_id: str
    template_id: str
    user_id: str
    user_email: str
    sent_date: datetime
    opened_date: Optional[datetime]
    clicked_date: Optional[datetime]
    reported_date: Optional[datetime]
    data_entered: bool
    ip_address: Optional[str]
    user_agent: Optional[str]
    status: str  # sent, opened, clicked, reported, failed

@dataclass
class AwarenessContent:
    """Security awareness content"""
    id: str
    title: str
    content_type: str  # article, video, infographic, quiz
    category: str  # phishing, malware, social_engineering, etc.
    difficulty: str
    content_url: str
    duration_minutes: int
    learning_objectives: List[str]
    created_date: datetime
    engagement_score: float

class SecurityAwarenessSystem:
    """Comprehensive security awareness and phishing simulation system"""
    
    def __init__(self, awareness_path: str = "security/awareness"):
        self.awareness_path = Path(awareness_path)
        self.awareness_path.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.phishing_templates: Dict[str, PhishingTemplate] = {}
        self.campaigns: Dict[str, PhishingCampaign] = {}
        self.results: Dict[str, PhishingResult] = {}
        self.awareness_content: Dict[str, AwarenessContent] = {}
        
        self._load_awareness_data()
        self._initialize_default_content()
    
    def _load_awareness_data(self):
        """Load awareness data from storage"""
        # Load phishing templates
        templates_file = self.awareness_path / "phishing_templates.json"
        if templates_file.exists():
            with open(templates_file, 'r') as f:
                data = json.load(f)
                for template_id, template_data in data.items():
                    if template_data.get('created_date'):
                        template_data['created_date'] = datetime.fromisoformat(template_data['created_date'])
                    self.phishing_templates[template_id] = PhishingTemplate(**template_data)
        
        # Load campaigns
        campaigns_file = self.awareness_path / "campaigns.json"
        if campaigns_file.exists():
            with open(campaigns_file, 'r') as f:
                data = json.load(f)
                for campaign_id, campaign_data in data.items():
                    for date_field in ['start_date', 'end_date']:
                        if campaign_data.get(date_field):
                            campaign_data[date_field] = datetime.fromisoformat(campaign_data[date_field])
                    self.campaigns[campaign_id] = PhishingCampaign(**campaign_data)
        
        # Load results
        results_file = self.awareness_path / "results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                data = json.load(f)
                for result_id, result_data in data.items():
                    for date_field in ['sent_date', 'opened_date', 'clicked_date', 'reported_date']:
                        if result_data.get(date_field):
                            result_data[date_field] = datetime.fromisoformat(result_data[date_field])
                    self.results[result_id] = PhishingResult(**result_data)
        
        # Load awareness content
        content_file = self.awareness_path / "awareness_content.json"
        if content_file.exists():
            with open(content_file, 'r') as f:
                data = json.load(f)
                for content_id, content_data in data.items():
                    if content_data.get('created_date'):
                        content_data['created_date'] = datetime.fromisoformat(content_data['created_date'])
                    self.awareness_content[content_id] = AwarenessContent(**content_data)
    
    def _save_awareness_data(self):
        """Save awareness data to storage"""
        # Save phishing templates
        templates_data = {}
        for template_id, template in self.phishing_templates.items():
            template_data = asdict(template)
            if template_data.get('created_date'):
                template_data['created_date'] = template_data['created_date'].isoformat()
            templates_data[template_id] = template_data
        
        with open(self.awareness_path / "phishing_templates.json", 'w') as f:
            json.dump(templates_data, f, indent=2)
        
        # Save campaigns
        campaigns_data = {}
        for campaign_id, campaign in self.campaigns.items():
            campaign_data = asdict(campaign)
            for date_field in ['start_date', 'end_date']:
                if campaign_data.get(date_field):
                    campaign_data[date_field] = campaign_data[date_field].isoformat()
            campaigns_data[campaign_id] = campaign_data
        
        with open(self.awareness_path / "campaigns.json", 'w') as f:
            json.dump(campaigns_data, f, indent=2)
        
        # Save results
        results_data = {}
        for result_id, result in self.results.items():
            result_data = asdict(result)
            for date_field in ['sent_date', 'opened_date', 'clicked_date', 'reported_date']:
                if result_data.get(date_field) and result_data[date_field] is not None:
                    result_data[date_field] = result_data[date_field].isoformat()
            results_data[result_id] = result_data
        
        with open(self.awareness_path / "results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save awareness content
        content_data = {}
        for content_id, content in self.awareness_content.items():
            content_dict = asdict(content)
            if content_dict.get('created_date'):
                content_dict['created_date'] = content_dict['created_date'].isoformat()
            content_data[content_id] = content_dict
        
        with open(self.awareness_path / "awareness_content.json", 'w') as f:
            json.dump(content_data, f, indent=2)
    
    def _initialize_default_content(self):
        """Initialize default phishing templates and awareness content"""
        if not self.phishing_templates:
            default_templates = [
                {
                    "id": "phish-001",
                    "name": "Fake IT Support",
                    "category": "credential_harvesting",
                    "difficulty": "easy",
                    "subject_line": "Urgent: Your account will be suspended",
                    "sender_name": "IT Support",
                    "sender_email": "it-support@company-security.com",
                    "html_content": """
                    <html>
                    <body>
                    <h2>Account Security Alert</h2>
                    <p>Dear Employee,</p>
                    <p>We have detected suspicious activity on your account. To prevent suspension, 
                    please verify your credentials immediately.</p>
                    <p><a href="{landing_page_url}">Click here to verify your account</a></p>
                    <p>This link will expire in 24 hours.</p>
                    <p>Best regards,<br>IT Security Team</p>
                    </body>
                    </html>
                    """,
                    "text_content": "Account Security Alert - Click link to verify credentials",
                    "landing_page_url": "https://phishing-sim.company.com/verify",
                    "indicators": [
                        "Urgent language",
                        "Suspicious sender domain",
                        "Generic greeting",
                        "Threat of account suspension"
                    ],
                    "success_rate": 0.35
                },
                {
                    "id": "phish-002", 
                    "name": "CEO Impersonation",
                    "category": "social_engineering",
                    "difficulty": "medium",
                    "subject_line": "Confidential: Urgent wire transfer needed",
                    "sender_name": "CEO John Smith",
                    "sender_email": "j.smith@company-exec.com",
                    "html_content": """
                    <html>
                    <body>
                    <p>Hi,</p>
                    <p>I need you to process an urgent wire transfer for a confidential acquisition. 
                    Please transfer $50,000 to the following account immediately.</p>
                    <p>Account: 123456789<br>
                    Routing: 987654321<br>
                    Bank: First National</p>
                    <p>This is time-sensitive and confidential. Please confirm once completed.</p>
                    <p>Thanks,<br>John</p>
                    </body>
                    </html>
                    """,
                    "text_content": "Urgent wire transfer request from CEO",
                    "landing_page_url": "https://phishing-sim.company.com/transfer",
                    "indicators": [
                        "Impersonation of executive",
                        "Urgent financial request",
                        "Unusual sender domain",
                        "Request for confidentiality"
                    ],
                    "success_rate": 0.22
                },
                {
                    "id": "phish-003",
                    "name": "Software Update",
                    "category": "malware",
                    "difficulty": "hard",
                    "subject_line": "Critical Security Update Available",
                    "sender_name": "Microsoft Security",
                    "sender_email": "security@microsoft-updates.com",
                    "html_content": """
                    <html>
                    <body>
                    <div style="background: #0078d4; color: white; padding: 10px;">
                    <h2>Microsoft Security Update</h2>
                    </div>
                    <p>A critical security vulnerability has been discovered in Windows. 
                    Please install the attached security patch immediately.</p>
                    <p>This update addresses CVE-2024-0001 which could allow remote code execution.</p>
                    <p><strong>Action Required:</strong> Download and install the security patch</p>
                    <p><a href="{landing_page_url}">Download Security Patch</a></p>
                    <p>Microsoft Security Response Center</p>
                    </body>
                    </html>
                    """,
                    "text_content": "Critical Windows security update - download patch",
                    "landing_page_url": "https://phishing-sim.company.com/update",
                    "indicators": [
                        "Fake Microsoft branding",
                        "Suspicious download link",
                        "Fake CVE reference",
                        "Urgent security language"
                    ],
                    "success_rate": 0.18
                }
            ]
            
            for template_data in default_templates:
                template_data['created_date'] = datetime.now()
                template = PhishingTemplate(**template_data)
                self.phishing_templates[template.id] = template
            
            self._save_awareness_data()
        
        if not self.awareness_content:
            default_content = [
                {
                    "id": "aware-001",
                    "title": "How to Spot Phishing Emails",
                    "content_type": "video",
                    "category": "phishing",
                    "difficulty": "beginner",
                    "content_url": "/awareness/videos/spot-phishing.mp4",
                    "duration_minutes": 8,
                    "learning_objectives": [
                        "Identify common phishing indicators",
                        "Verify sender authenticity",
                        "Report suspicious emails"
                    ],
                    "engagement_score": 4.2
                },
                {
                    "id": "aware-002",
                    "title": "Social Engineering Tactics",
                    "content_type": "article",
                    "category": "social_engineering",
                    "difficulty": "intermediate",
                    "content_url": "/awareness/articles/social-engineering.html",
                    "duration_minutes": 12,
                    "learning_objectives": [
                        "Understand social engineering techniques",
                        "Recognize manipulation tactics",
                        "Implement verification procedures"
                    ],
                    "engagement_score": 3.8
                },
                {
                    "id": "aware-003",
                    "title": "Password Security Best Practices",
                    "content_type": "infographic",
                    "category": "authentication",
                    "difficulty": "beginner",
                    "content_url": "/awareness/infographics/password-security.png",
                    "duration_minutes": 5,
                    "learning_objectives": [
                        "Create strong passwords",
                        "Use password managers",
                        "Enable multi-factor authentication"
                    ],
                    "engagement_score": 4.5
                }
            ]
            
            for content_data in default_content:
                content_data['created_date'] = datetime.now()
                content = AwarenessContent(**content_data)
                self.awareness_content[content.id] = content
            
            self._save_awareness_data()
    
    def create_phishing_campaign(self, campaign_data: Dict[str, Any]) -> str:
        """Create a new phishing simulation campaign"""
        try:
            campaign_id = campaign_data.get('id', str(uuid.uuid4()))
            campaign_data['id'] = campaign_id
            campaign_data['status'] = 'planned'
            campaign_data['metrics'] = {}
            
            # Convert date strings to datetime objects
            if isinstance(campaign_data.get('start_date'), str):
                campaign_data['start_date'] = datetime.fromisoformat(campaign_data['start_date'])
            if isinstance(campaign_data.get('end_date'), str):
                campaign_data['end_date'] = datetime.fromisoformat(campaign_data['end_date'])
            
            campaign = PhishingCampaign(**campaign_data)
            self.campaigns[campaign_id] = campaign
            self._save_awareness_data()
            
            logger.info(f"Created phishing campaign: {campaign_id}")
            return campaign_id
            
        except Exception as e:
            logger.error(f"Failed to create phishing campaign: {str(e)}")
            raise
    
    def launch_phishing_campaign(self, campaign_id: str, target_users: List[Dict[str, str]]) -> bool:
        """Launch a phishing simulation campaign"""
        try:
            if campaign_id not in self.campaigns:
                raise ValueError(f"Campaign {campaign_id} not found")
            
            campaign = self.campaigns[campaign_id]
            campaign.status = 'active'
            
            # Send phishing emails to target users
            for user in target_users:
                for template_id in campaign.template_ids:
                    if template_id not in self.phishing_templates:
                        continue
                    
                    template = self.phishing_templates[template_id]
                    
                    # Create result record
                    result_id = str(uuid.uuid4())
                    result = PhishingResult(
                        id=result_id,
                        campaign_id=campaign_id,
                        template_id=template_id,
                        user_id=user['user_id'],
                        user_email=user['email'],
                        sent_date=datetime.now(),
                        opened_date=None,
                        clicked_date=None,
                        reported_date=None,
                        data_entered=False,
                        ip_address=None,
                        user_agent=None,
                        status='sent'
                    )
                    
                    self.results[result_id] = result
                    
                    # Send email (simulated)
                    self._send_phishing_email(template, user, result_id)
            
            self._save_awareness_data()
            logger.info(f"Launched phishing campaign: {campaign_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to launch phishing campaign: {str(e)}")
            return False
    
    def _send_phishing_email(self, template: PhishingTemplate, user: Dict[str, str], result_id: str):
        """Send phishing email (simulated)"""
        # In a real implementation, this would send actual emails
        # For simulation purposes, we'll just log the action
        logger.info(f"Sent phishing email to {user['email']} using template {template.id}")
        
        # Simulate some users opening/clicking based on template success rate
        if random.random() < template.success_rate:
            # Simulate user interaction after random delay
            self._simulate_user_interaction(result_id, template.success_rate)
    
    def _simulate_user_interaction(self, result_id: str, success_rate: float):
        """Simulate user interaction with phishing email"""
        if result_id not in self.results:
            return
        
        result = self.results[result_id]
        
        # Simulate opening email
        if random.random() < 0.8:  # 80% open rate
            result.opened_date = datetime.now() + timedelta(minutes=random.randint(5, 120))
            result.status = 'opened'
            
            # Simulate clicking link
            if random.random() < success_rate:
                result.clicked_date = result.opened_date + timedelta(minutes=random.randint(1, 30))
                result.status = 'clicked'
                
                # Simulate entering data
                if random.random() < 0.6:  # 60% of clickers enter data
                    result.data_entered = True
            
            # Simulate reporting (security-aware users)
            elif random.random() < 0.1:  # 10% report suspicious emails
                result.reported_date = result.opened_date + timedelta(minutes=random.randint(1, 60))
                result.status = 'reported'
        
        self._save_awareness_data()
    
    def record_email_interaction(self, result_id: str, interaction_type: str, 
                                ip_address: str = None, user_agent: str = None) -> bool:
        """Record user interaction with phishing email"""
        try:
            if result_id not in self.results:
                return False
            
            result = self.results[result_id]
            now = datetime.now()
            
            if interaction_type == 'opened' and not result.opened_date:
                result.opened_date = now
                result.status = 'opened'
            elif interaction_type == 'clicked' and not result.clicked_date:
                result.clicked_date = now
                result.status = 'clicked'
            elif interaction_type == 'reported' and not result.reported_date:
                result.reported_date = now
                result.status = 'reported'
            elif interaction_type == 'data_entered':
                result.data_entered = True
            
            if ip_address:
                result.ip_address = ip_address
            if user_agent:
                result.user_agent = user_agent
            
            self._save_awareness_data()
            return True
            
        except Exception as e:
            logger.error(f"Failed to record interaction: {str(e)}")
            return False
    
    def get_campaign_results(self, campaign_id: str) -> Dict[str, Any]:
        """Get comprehensive results for a phishing campaign"""
        if campaign_id not in self.campaigns:
            raise ValueError(f"Campaign {campaign_id} not found")
        
        campaign = self.campaigns[campaign_id]
        campaign_results = [r for r in self.results.values() if r.campaign_id == campaign_id]
        
        total_sent = len(campaign_results)
        opened_count = sum(1 for r in campaign_results if r.opened_date)
        clicked_count = sum(1 for r in campaign_results if r.clicked_date)
        reported_count = sum(1 for r in campaign_results if r.reported_date)
        data_entered_count = sum(1 for r in campaign_results if r.data_entered)
        
        # Calculate rates
        open_rate = (opened_count / total_sent * 100) if total_sent > 0 else 0
        click_rate = (clicked_count / total_sent * 100) if total_sent > 0 else 0
        report_rate = (reported_count / total_sent * 100) if total_sent > 0 else 0
        data_entry_rate = (data_entered_count / total_sent * 100) if total_sent > 0 else 0
        
        # Template performance
        template_performance = {}
        for template_id in campaign.template_ids:
            template_results = [r for r in campaign_results if r.template_id == template_id]
            if template_results:
                template_clicked = sum(1 for r in template_results if r.clicked_date)
                template_performance[template_id] = {
                    "sent": len(template_results),
                    "clicked": template_clicked,
                    "click_rate": (template_clicked / len(template_results) * 100)
                }
        
        return {
            "campaign_id": campaign_id,
            "campaign_name": campaign.name,
            "total_sent": total_sent,
            "opened": opened_count,
            "clicked": clicked_count,
            "reported": reported_count,
            "data_entered": data_entered_count,
            "open_rate": round(open_rate, 2),
            "click_rate": round(click_rate, 2),
            "report_rate": round(report_rate, 2),
            "data_entry_rate": round(data_entry_rate, 2),
            "template_performance": template_performance,
            "status": campaign.status,
            "start_date": campaign.start_date.isoformat(),
            "end_date": campaign.end_date.isoformat()
        }
    
    def get_user_awareness_score(self, user_id: str) -> Dict[str, Any]:
        """Calculate security awareness score for a user"""
        user_results = [r for r in self.results.values() if r.user_id == user_id]
        
        if not user_results:
            return {
                "user_id": user_id,
                "awareness_score": 100,  # No data, assume good
                "risk_level": "low",
                "total_tests": 0,
                "failed_tests": 0,
                "improvement_areas": []
            }
        
        total_tests = len(user_results)
        clicked_count = sum(1 for r in user_results if r.clicked_date)
        reported_count = sum(1 for r in user_results if r.reported_date)
        data_entered_count = sum(1 for r in user_results if r.data_entered)
        
        # Calculate awareness score (0-100)
        base_score = 100
        click_penalty = clicked_count * 20  # -20 points per click
        data_penalty = data_entered_count * 30  # -30 points per data entry
        report_bonus = reported_count * 10  # +10 points per report
        
        awareness_score = max(0, base_score - click_penalty - data_penalty + report_bonus)
        
        # Determine risk level
        if awareness_score >= 80:
            risk_level = "low"
        elif awareness_score >= 60:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        # Identify improvement areas
        improvement_areas = []
        if clicked_count > 0:
            improvement_areas.append("Link verification")
        if data_entered_count > 0:
            improvement_areas.append("Credential protection")
        if reported_count == 0 and total_tests > 2:
            improvement_areas.append("Threat reporting")
        
        return {
            "user_id": user_id,
            "awareness_score": awareness_score,
            "risk_level": risk_level,
            "total_tests": total_tests,
            "failed_tests": clicked_count,
            "reported_tests": reported_count,
            "improvement_areas": improvement_areas,
            "last_test_date": max(r.sent_date for r in user_results).isoformat()
        }
    
    def generate_awareness_report(self) -> Dict[str, Any]:
        """Generate comprehensive security awareness report"""
        total_campaigns = len(self.campaigns)
        active_campaigns = sum(1 for c in self.campaigns.values() if c.status == 'active')
        total_results = len(self.results)
        
        if total_results == 0:
            return {
                "total_campaigns": total_campaigns,
                "active_campaigns": active_campaigns,
                "total_tests": 0,
                "overall_click_rate": 0,
                "overall_report_rate": 0,
                "high_risk_users": 0,
                "report_generated": datetime.now().isoformat()
            }
        
        # Overall statistics
        clicked_count = sum(1 for r in self.results.values() if r.clicked_date)
        reported_count = sum(1 for r in self.results.values() if r.reported_date)
        
        overall_click_rate = (clicked_count / total_results * 100)
        overall_report_rate = (reported_count / total_results * 100)
        
        # User risk analysis
        unique_users = set(r.user_id for r in self.results.values())
        high_risk_users = 0
        
        for user_id in unique_users:
            user_score = self.get_user_awareness_score(user_id)
            if user_score['risk_level'] == 'high':
                high_risk_users += 1
        
        # Template effectiveness
        template_stats = {}
        for template_id, template in self.phishing_templates.items():
            template_results = [r for r in self.results.values() if r.template_id == template_id]
            if template_results:
                template_clicked = sum(1 for r in template_results if r.clicked_date)
                template_stats[template_id] = {
                    "name": template.name,
                    "category": template.category,
                    "difficulty": template.difficulty,
                    "sent": len(template_results),
                    "clicked": template_clicked,
                    "click_rate": (template_clicked / len(template_results) * 100)
                }
        
        return {
            "total_campaigns": total_campaigns,
            "active_campaigns": active_campaigns,
            "total_tests": total_results,
            "overall_click_rate": round(overall_click_rate, 2),
            "overall_report_rate": round(overall_report_rate, 2),
            "total_users_tested": len(unique_users),
            "high_risk_users": high_risk_users,
            "template_effectiveness": template_stats,
            "report_generated": datetime.now().isoformat()
        }