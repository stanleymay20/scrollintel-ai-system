"""
Security Knowledge Base System
Provides searchable security procedures, best practices, and knowledge management
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)

class ContentType(Enum):
    PROCEDURE = "procedure"
    BEST_PRACTICE = "best_practice"
    TROUBLESHOOTING = "troubleshooting"
    FAQ = "faq"
    TUTORIAL = "tutorial"
    REFERENCE = "reference"
    CHECKLIST = "checklist"

class ContentStatus(Enum):
    DRAFT = "draft"
    REVIEW = "review"
    PUBLISHED = "published"
    ARCHIVED = "archived"

class AccessLevel(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"

@dataclass
class KnowledgeArticle:
    """Knowledge base article"""
    id: str
    title: str
    summary: str
    content: str
    content_type: ContentType
    status: ContentStatus
    access_level: AccessLevel
    author: str
    created_date: datetime
    last_updated: datetime
    last_reviewed: datetime
    review_frequency_days: int
    tags: List[str]
    categories: List[str]
    related_articles: List[str]
    attachments: List[str]
    view_count: int
    rating: float
    rating_count: int
    search_keywords: List[str]

@dataclass
class SearchIndex:
    """Search index entry"""
    article_id: str
    title: str
    content: str
    tags: List[str]
    categories: List[str]
    keywords: List[str]
    last_indexed: datetime

@dataclass
class UserFeedback:
    """User feedback on articles"""
    id: str
    article_id: str
    user_id: str
    rating: int  # 1-5 stars
    feedback_text: str
    helpful: bool
    created_date: datetime

@dataclass
class SearchQuery:
    """Search query tracking"""
    id: str
    query: str
    user_id: str
    timestamp: datetime
    results_count: int
    clicked_articles: List[str]

class SecurityKnowledgeBaseSystem:
    """Comprehensive security knowledge base system"""
    
    def __init__(self, kb_path: str = "security/knowledge_base"):
        self.kb_path = Path(kb_path)
        self.kb_path.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.articles: Dict[str, KnowledgeArticle] = {}
        self.search_index: Dict[str, SearchIndex] = {}
        self.feedback: Dict[str, UserFeedback] = {}
        self.search_queries: Dict[str, SearchQuery] = {}
        
        self._load_knowledge_base_data()
        self._initialize_default_content()
        self._build_search_index()
    
    def _load_knowledge_base_data(self):
        """Load knowledge base data from storage"""
        # Load articles
        articles_file = self.kb_path / "articles.json"
        if articles_file.exists():
            with open(articles_file, 'r') as f:
                data = json.load(f)
                for article_id, article_data in data.items():
                    # Convert datetime strings
                    for date_field in ['created_date', 'last_updated', 'last_reviewed']:
                        if article_data.get(date_field):
                            article_data[date_field] = datetime.fromisoformat(article_data[date_field])
                    
                    # Convert enums
                    article_data['content_type'] = ContentType(article_data['content_type'])
                    article_data['status'] = ContentStatus(article_data['status'])
                    article_data['access_level'] = AccessLevel(article_data['access_level'])
                    
                    self.articles[article_id] = KnowledgeArticle(**article_data)
        
        # Load search index
        index_file = self.kb_path / "search_index.json"
        if index_file.exists():
            with open(index_file, 'r') as f:
                data = json.load(f)
                for index_id, index_data in data.items():
                    if index_data.get('last_indexed'):
                        index_data['last_indexed'] = datetime.fromisoformat(index_data['last_indexed'])
                    self.search_index[index_id] = SearchIndex(**index_data)
        
        # Load feedback
        feedback_file = self.kb_path / "feedback.json"
        if feedback_file.exists():
            with open(feedback_file, 'r') as f:
                data = json.load(f)
                for feedback_id, feedback_data in data.items():
                    if feedback_data.get('created_date'):
                        feedback_data['created_date'] = datetime.fromisoformat(feedback_data['created_date'])
                    self.feedback[feedback_id] = UserFeedback(**feedback_data)
        
        # Load search queries
        queries_file = self.kb_path / "search_queries.json"
        if queries_file.exists():
            with open(queries_file, 'r') as f:
                data = json.load(f)
                for query_id, query_data in data.items():
                    if query_data.get('timestamp'):
                        query_data['timestamp'] = datetime.fromisoformat(query_data['timestamp'])
                    self.search_queries[query_id] = SearchQuery(**query_data)
    
    def _save_knowledge_base_data(self):
        """Save knowledge base data to storage"""
        # Save articles
        articles_data = {}
        for article_id, article in self.articles.items():
            article_data = asdict(article)
            
            # Convert datetime objects to strings
            for date_field in ['created_date', 'last_updated', 'last_reviewed']:
                if article_data.get(date_field):
                    article_data[date_field] = article_data[date_field].isoformat()
            
            # Convert enums
            article_data['content_type'] = article_data['content_type'].value
            article_data['status'] = article_data['status'].value
            article_data['access_level'] = article_data['access_level'].value
            
            articles_data[article_id] = article_data
        
        with open(self.kb_path / "articles.json", 'w') as f:
            json.dump(articles_data, f, indent=2)
        
        # Save search index
        index_data = {}
        for index_id, index in self.search_index.items():
            index_dict = asdict(index)
            if index_dict.get('last_indexed'):
                index_dict['last_indexed'] = index_dict['last_indexed'].isoformat()
            index_data[index_id] = index_dict
        
        with open(self.kb_path / "search_index.json", 'w') as f:
            json.dump(index_data, f, indent=2)
        
        # Save feedback
        feedback_data = {}
        for feedback_id, feedback in self.feedback.items():
            feedback_dict = asdict(feedback)
            if feedback_dict.get('created_date'):
                feedback_dict['created_date'] = feedback_dict['created_date'].isoformat()
            feedback_data[feedback_id] = feedback_dict
        
        with open(self.kb_path / "feedback.json", 'w') as f:
            json.dump(feedback_data, f, indent=2)
        
        # Save search queries
        queries_data = {}
        for query_id, query in self.search_queries.items():
            query_dict = asdict(query)
            if query_dict.get('timestamp'):
                query_dict['timestamp'] = query_dict['timestamp'].isoformat()
            queries_data[query_id] = query_dict
        
        with open(self.kb_path / "search_queries.json", 'w') as f:
            json.dump(queries_data, f, indent=2)
    
    def _initialize_default_content(self):
        """Initialize default knowledge base content"""
        if not self.articles:
            default_articles = [
                {
                    "id": "kb-001",
                    "title": "How to Report a Security Incident",
                    "summary": "Step-by-step procedure for reporting security incidents",
                    "content": """
# How to Report a Security Incident

## Overview
This procedure outlines the steps to report a security incident to ensure rapid response and containment.

## When to Report
Report immediately if you observe:
- Suspicious network activity
- Unauthorized access attempts
- Malware infections
- Data breaches or unauthorized data access
- Phishing attempts
- Physical security breaches

## Reporting Steps

### Step 1: Immediate Actions
1. **Do not panic** - Stay calm and follow procedures
2. **Do not touch anything** - Preserve evidence
3. **Disconnect if safe** - If malware is suspected, disconnect from network
4. **Document what you see** - Take screenshots if possible

### Step 2: Contact Security Team
- **Email**: security@company.com
- **Phone**: +1-555-SECURITY (24/7 hotline)
- **Internal Chat**: #security-incidents channel

### Step 3: Provide Information
Include the following in your report:
- Your name and contact information
- Date and time of incident
- Description of what happened
- Systems or data affected
- Any actions you've already taken

### Step 4: Follow Up
- Preserve any evidence
- Cooperate with security team investigation
- Do not discuss incident with unauthorized personnel

## Examples

### Phishing Email
"I received a suspicious email claiming to be from IT asking for my password. The sender was it-support@company-security.com which looks suspicious."

### Malware Detection
"My antivirus detected malware on my workstation. I immediately disconnected from the network and am reporting this incident."

## Important Notes
- **Time is critical** - Report immediately
- **Better safe than sorry** - Report anything suspicious
- **No blame culture** - You will not be penalized for reporting

## Related Articles
- [Incident Response Playbook](kb-002)
- [Phishing Identification Guide](kb-003)
- [Malware Response Procedures](kb-004)
                    """,
                    "content_type": ContentType.PROCEDURE,
                    "status": ContentStatus.PUBLISHED,
                    "access_level": AccessLevel.INTERNAL,
                    "author": "security-team",
                    "review_frequency_days": 180,
                    "tags": ["incident-reporting", "security", "procedure"],
                    "categories": ["incident-response", "procedures"],
                    "related_articles": ["kb-002", "kb-003", "kb-004"],
                    "attachments": [],
                    "view_count": 0,
                    "rating": 0.0,
                    "rating_count": 0,
                    "search_keywords": ["report", "incident", "security", "breach", "malware", "phishing"]
                },
                {
                    "id": "kb-002",
                    "title": "Password Security Best Practices",
                    "summary": "Guidelines for creating and managing secure passwords",
                    "content": """
# Password Security Best Practices

## Overview
Strong passwords are your first line of defense against unauthorized access. Follow these guidelines to create and manage secure passwords.

## Password Requirements

### Minimum Standards
- **Length**: At least 12 characters (longer is better)
- **Complexity**: Mix of uppercase, lowercase, numbers, and symbols
- **Uniqueness**: Different password for each account
- **No personal information**: Avoid names, birthdays, addresses

### Good Password Examples
- `MyDog$Loves2Run!` (passphrase with substitutions)
- `Tr@il#Running2024!` (hobby-based with complexity)
- `Coffee&Code@5AM` (routine-based with symbols)

### Avoid These Patterns
- `Password123!` (common patterns)
- `Company2024` (predictable information)
- `qwerty123` (keyboard patterns)
- `admin` or `password` (default passwords)

## Password Management

### Use a Password Manager
**Recommended Tools:**
- 1Password (Enterprise)
- Bitwarden (Open source)
- LastPass (Consumer)

**Benefits:**
- Generate strong, unique passwords
- Secure encrypted storage
- Auto-fill capabilities
- Cross-device synchronization

### Multi-Factor Authentication (MFA)
Always enable MFA when available:
- **Authenticator apps**: Google Authenticator, Authy
- **Hardware tokens**: YubiKey, RSA SecurID
- **SMS**: Less secure but better than nothing
- **Biometrics**: Fingerprint, face recognition

## Account Security

### Regular Maintenance
- **Change passwords** if compromised
- **Review account activity** regularly
- **Remove unused accounts** to reduce attack surface
- **Update recovery information** (email, phone)

### Warning Signs
Watch for these indicators of compromise:
- Unexpected password reset emails
- Unfamiliar login notifications
- Changes you didn't make
- Suspicious account activity

## Workplace Policies

### Do's
- Use company-approved password managers
- Report suspected compromises immediately
- Follow password complexity requirements
- Enable MFA on all business accounts

### Don'ts
- Share passwords with colleagues
- Write passwords on sticky notes
- Use personal passwords for business accounts
- Reuse passwords across systems

## Emergency Procedures

### If Your Password is Compromised
1. **Change immediately** on affected account
2. **Check other accounts** using same password
3. **Report to IT security** if business account
4. **Monitor accounts** for suspicious activity
5. **Update password manager** with new credentials

### If You Forget Your Password
1. **Use password manager** to retrieve
2. **Use official reset process** (not suspicious emails)
3. **Verify identity** through approved channels
4. **Contact IT support** for business accounts

## Related Articles
- [Multi-Factor Authentication Setup](kb-005)
- [Phishing Prevention Guide](kb-003)
- [Account Security Checklist](kb-006)
                    """,
                    "content_type": ContentType.BEST_PRACTICE,
                    "status": ContentStatus.PUBLISHED,
                    "access_level": AccessLevel.INTERNAL,
                    "author": "security-team",
                    "review_frequency_days": 365,
                    "tags": ["passwords", "authentication", "best-practices"],
                    "categories": ["authentication", "best-practices"],
                    "related_articles": ["kb-005", "kb-003", "kb-006"],
                    "attachments": [],
                    "view_count": 0,
                    "rating": 0.0,
                    "rating_count": 0,
                    "search_keywords": ["password", "authentication", "security", "mfa", "manager"]
                },
                {
                    "id": "kb-003",
                    "title": "Phishing Email Identification Guide",
                    "summary": "How to identify and handle phishing emails",
                    "content": """
# Phishing Email Identification Guide

## What is Phishing?
Phishing is a cyber attack where criminals send fraudulent emails to steal sensitive information like passwords, credit card numbers, or personal data.

## Common Phishing Indicators

### Sender Red Flags
- **Suspicious domains**: `amaz0n.com` instead of `amazon.com`
- **Generic greetings**: "Dear Customer" instead of your name
- **Urgent language**: "Act now!" or "Account will be closed!"
- **Mismatched sender**: Email from "bank" but sender is `noreply@suspicious-domain.com`

### Content Red Flags
- **Grammar/spelling errors**: Professional companies proofread emails
- **Generic signatures**: No specific contact information
- **Suspicious attachments**: Unexpected .exe, .zip, or .doc files
- **Shortened URLs**: bit.ly, tinyurl hiding real destination

### Request Red Flags
- **Password requests**: Legitimate companies never ask for passwords via email
- **Urgent financial actions**: "Wire money immediately"
- **Personal information**: SSN, credit card numbers, account details
- **Software downloads**: "Update your security software"

## Types of Phishing Attacks

### Email Phishing
Most common type targeting large groups with generic messages.

**Example:**
```
From: security@yourbankk.com
Subject: Account Suspended - Verify Now!

Your account has been suspended due to suspicious activity.
Click here to verify: http://bit.ly/verify-account
```

### Spear Phishing
Targeted attacks using personal information about the victim.

**Example:**
```
From: ceo@company.com
Subject: Urgent: Confidential Project

Hi [Your Name],
I need you to review this confidential document about Project Alpha.
Please download and review immediately.
[Malicious attachment]
```

### Whaling
High-value targets like executives or key personnel.

### Smishing (SMS Phishing)
Phishing via text messages.

**Example:**
```
ALERT: Your bank account has been compromised.
Click here to secure: http://fake-bank.com
```

## How to Verify Suspicious Emails

### Step 1: Don't Click Anything
- Don't click links or download attachments
- Don't reply to the email
- Don't forward to others

### Step 2: Check the Sender
- Hover over sender name to see actual email address
- Look for domain misspellings
- Verify sender through separate communication channel

### Step 3: Analyze the Content
- Check for urgency tactics
- Look for grammar/spelling errors
- Verify any claims through official channels

### Step 4: Verify Links
- Hover over links to see actual destination
- Don't trust shortened URLs
- Type URLs manually instead of clicking

## What to Do if You Receive Phishing

### Immediate Actions
1. **Don't interact** with the email
2. **Report to security team**: Forward to security@company.com
3. **Delete the email** after reporting
4. **Warn colleagues** if it's a targeted attack

### If You Already Clicked
1. **Don't enter any information** on suspicious sites
2. **Close browser immediately**
3. **Run antivirus scan**
4. **Report to security team immediately**
5. **Change passwords** if you entered any credentials

### If You Entered Information
1. **Change passwords immediately** on affected accounts
2. **Enable MFA** if not already active
3. **Monitor accounts** for suspicious activity
4. **Report to security team** and potentially law enforcement
5. **Consider credit monitoring** if financial information was compromised

## Prevention Tips

### Email Security
- Use email filters and spam protection
- Keep email client updated
- Be skeptical of unexpected emails
- Verify requests through separate channels

### Browser Security
- Keep browser updated
- Use reputable antivirus software
- Enable browser security features
- Be cautious with downloads

### Education
- Stay informed about latest phishing techniques
- Participate in security awareness training
- Practice identifying suspicious emails
- Share knowledge with colleagues

## Reporting Phishing

### Internal Reporting
- **Email**: security@company.com
- **Subject**: "PHISHING REPORT"
- **Include**: Original email as attachment

### External Reporting
- **Anti-Phishing Working Group**: reportphishing@apwg.org
- **FTC**: reportfraud.ftc.gov
- **FBI IC3**: ic3.gov

## Related Articles
- [Email Security Settings](kb-007)
- [Social Engineering Awareness](kb-008)
- [Incident Reporting Procedures](kb-001)
                    """,
                    "content_type": ContentType.TUTORIAL,
                    "status": ContentStatus.PUBLISHED,
                    "access_level": AccessLevel.INTERNAL,
                    "author": "security-team",
                    "review_frequency_days": 180,
                    "tags": ["phishing", "email-security", "awareness"],
                    "categories": ["email-security", "awareness"],
                    "related_articles": ["kb-007", "kb-008", "kb-001"],
                    "attachments": [],
                    "view_count": 0,
                    "rating": 0.0,
                    "rating_count": 0,
                    "search_keywords": ["phishing", "email", "scam", "fraud", "suspicious", "malicious"]
                }
            ]
            
            for article_data in default_articles:
                article_data['created_date'] = datetime.now()
                article_data['last_updated'] = datetime.now()
                article_data['last_reviewed'] = datetime.now()
                
                article = KnowledgeArticle(**article_data)
                self.articles[article.id] = article
            
            self._save_knowledge_base_data()
    
    def _build_search_index(self):
        """Build search index for all articles"""
        for article_id, article in self.articles.items():
            if article.status == ContentStatus.PUBLISHED:
                self._index_article(article)
    
    def _index_article(self, article: KnowledgeArticle):
        """Index an article for search"""
        # Extract keywords from content
        content_words = self._extract_keywords(article.content)
        title_words = self._extract_keywords(article.title)
        
        # Combine all searchable text
        all_keywords = list(set(
            content_words + title_words + 
            article.tags + article.categories + 
            article.search_keywords
        ))
        
        index_entry = SearchIndex(
            article_id=article.id,
            title=article.title,
            content=article.content,
            tags=article.tags,
            categories=article.categories,
            keywords=all_keywords,
            last_indexed=datetime.now()
        )
        
        self.search_index[article.id] = index_entry
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction - remove common words and extract meaningful terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
        
        # Extract words (alphanumeric sequences)
        words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        
        # Filter out stop words and short words
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return list(set(keywords))  # Remove duplicates
    
    def create_article(self, article_data: Dict[str, Any]) -> str:
        """Create a new knowledge base article"""
        try:
            article_id = article_data.get('id', str(uuid.uuid4()))
            article_data['id'] = article_id
            article_data['created_date'] = datetime.now()
            article_data['last_updated'] = datetime.now()
            article_data['last_reviewed'] = datetime.now()
            article_data['view_count'] = 0
            article_data['rating'] = 0.0
            article_data['rating_count'] = 0
            
            # Convert string enums to enum objects
            article_data['content_type'] = ContentType(article_data['content_type'])
            article_data['status'] = ContentStatus(article_data.get('status', 'draft'))
            article_data['access_level'] = AccessLevel(article_data.get('access_level', 'internal'))
            
            # Set default values
            article_data.setdefault('related_articles', [])
            article_data.setdefault('attachments', [])
            article_data.setdefault('search_keywords', [])
            article_data.setdefault('review_frequency_days', 365)
            
            article = KnowledgeArticle(**article_data)
            self.articles[article_id] = article
            
            # Index if published
            if article.status == ContentStatus.PUBLISHED:
                self._index_article(article)
            
            self._save_knowledge_base_data()
            logger.info(f"Created knowledge base article: {article_id}")
            return article_id
            
        except Exception as e:
            logger.error(f"Failed to create article: {str(e)}")
            raise
    
    def update_article(self, article_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing article"""
        try:
            if article_id not in self.articles:
                raise ValueError(f"Article {article_id} not found")
            
            article = self.articles[article_id]
            
            # Update fields
            for field, value in updates.items():
                if hasattr(article, field):
                    # Handle enum conversions
                    if field == 'content_type' and isinstance(value, str):
                        value = ContentType(value)
                    elif field == 'status' and isinstance(value, str):
                        value = ContentStatus(value)
                    elif field == 'access_level' and isinstance(value, str):
                        value = AccessLevel(value)
                    
                    setattr(article, field, value)
            
            article.last_updated = datetime.now()
            
            # Re-index if published
            if article.status == ContentStatus.PUBLISHED:
                self._index_article(article)
            elif article_id in self.search_index:
                # Remove from index if no longer published
                del self.search_index[article_id]
            
            self._save_knowledge_base_data()
            logger.info(f"Updated knowledge base article: {article_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update article: {str(e)}")
            return False
    
    def search_articles(self, query: str, user_id: str = None, 
                       access_level: AccessLevel = AccessLevel.INTERNAL) -> List[Dict[str, Any]]:
        """Search knowledge base articles"""
        try:
            # Log search query
            if user_id:
                query_id = str(uuid.uuid4())
                search_query = SearchQuery(
                    id=query_id,
                    query=query,
                    user_id=user_id,
                    timestamp=datetime.now(),
                    results_count=0,
                    clicked_articles=[]
                )
                self.search_queries[query_id] = search_query
            
            # Extract search terms
            search_terms = self._extract_keywords(query.lower())
            
            # Score articles based on relevance
            scored_articles = []
            
            for article_id, index_entry in self.search_index.items():
                article = self.articles[article_id]
                
                # Check access level
                if not self._check_access(article.access_level, access_level):
                    continue
                
                # Calculate relevance score
                score = self._calculate_relevance_score(search_terms, index_entry, article)
                
                if score > 0:
                    scored_articles.append({
                        'article_id': article_id,
                        'title': article.title,
                        'summary': article.summary,
                        'content_type': article.content_type.value,
                        'tags': article.tags,
                        'categories': article.categories,
                        'rating': article.rating,
                        'view_count': article.view_count,
                        'last_updated': article.last_updated.isoformat(),
                        'relevance_score': score
                    })
            
            # Sort by relevance score
            scored_articles.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Update search query results count
            if user_id and query_id in self.search_queries:
                self.search_queries[query_id].results_count = len(scored_articles)
                self._save_knowledge_base_data()
            
            return scored_articles[:20]  # Return top 20 results
            
        except Exception as e:
            logger.error(f"Failed to search articles: {str(e)}")
            return []
    
    def _check_access(self, article_access: AccessLevel, user_access: AccessLevel) -> bool:
        """Check if user has access to article based on access levels"""
        access_hierarchy = {
            AccessLevel.PUBLIC: 0,
            AccessLevel.INTERNAL: 1,
            AccessLevel.RESTRICTED: 2,
            AccessLevel.CONFIDENTIAL: 3
        }
        
        return access_hierarchy[user_access] >= access_hierarchy[article_access]
    
    def _calculate_relevance_score(self, search_terms: List[str], 
                                  index_entry: SearchIndex, article: KnowledgeArticle) -> float:
        """Calculate relevance score for search results"""
        score = 0.0
        
        # Title matches (highest weight)
        title_words = self._extract_keywords(index_entry.title.lower())
        title_matches = sum(1 for term in search_terms if term in title_words)
        score += title_matches * 10.0
        
        # Tag matches (high weight)
        tag_matches = sum(1 for term in search_terms if term in [tag.lower() for tag in index_entry.tags])
        score += tag_matches * 5.0
        
        # Category matches (medium weight)
        category_matches = sum(1 for term in search_terms if term in [cat.lower() for cat in index_entry.categories])
        score += category_matches * 3.0
        
        # Keyword matches (medium weight)
        keyword_matches = sum(1 for term in search_terms if term in [kw.lower() for kw in index_entry.keywords])
        score += keyword_matches * 2.0
        
        # Content matches (lower weight)
        content_words = self._extract_keywords(index_entry.content.lower())
        content_matches = sum(1 for term in search_terms if term in content_words)
        score += content_matches * 1.0
        
        # Boost score based on article quality metrics
        if article.rating > 0:
            score *= (1 + (article.rating / 5.0) * 0.2)  # Up to 20% boost for 5-star articles
        
        # Boost for popular articles
        if article.view_count > 0:
            score *= (1 + min(article.view_count / 1000.0, 0.1))  # Up to 10% boost for popular articles
        
        return score
    
    def get_article(self, article_id: str, user_id: str = None) -> Optional[Dict[str, Any]]:
        """Get a specific article and increment view count"""
        if article_id not in self.articles:
            return None
        
        article = self.articles[article_id]
        
        # Increment view count
        article.view_count += 1
        
        # Track article access
        if user_id:
            # Could track user reading patterns here
            pass
        
        self._save_knowledge_base_data()
        
        return {
            'id': article.id,
            'title': article.title,
            'summary': article.summary,
            'content': article.content,
            'content_type': article.content_type.value,
            'status': article.status.value,
            'access_level': article.access_level.value,
            'author': article.author,
            'created_date': article.created_date.isoformat(),
            'last_updated': article.last_updated.isoformat(),
            'last_reviewed': article.last_reviewed.isoformat(),
            'tags': article.tags,
            'categories': article.categories,
            'related_articles': article.related_articles,
            'attachments': article.attachments,
            'view_count': article.view_count,
            'rating': article.rating,
            'rating_count': article.rating_count
        }
    
    def submit_feedback(self, article_id: str, user_id: str, rating: int, 
                       feedback_text: str = "", helpful: bool = True) -> str:
        """Submit user feedback for an article"""
        try:
            if article_id not in self.articles:
                raise ValueError(f"Article {article_id} not found")
            
            if not 1 <= rating <= 5:
                raise ValueError("Rating must be between 1 and 5")
            
            feedback_id = str(uuid.uuid4())
            feedback = UserFeedback(
                id=feedback_id,
                article_id=article_id,
                user_id=user_id,
                rating=rating,
                feedback_text=feedback_text,
                helpful=helpful,
                created_date=datetime.now()
            )
            
            self.feedback[feedback_id] = feedback
            
            # Update article rating
            article = self.articles[article_id]
            total_rating = article.rating * article.rating_count + rating
            article.rating_count += 1
            article.rating = total_rating / article.rating_count
            
            self._save_knowledge_base_data()
            logger.info(f"Submitted feedback for article {article_id}: {rating} stars")
            return feedback_id
            
        except Exception as e:
            logger.error(f"Failed to submit feedback: {str(e)}")
            raise
    
    def get_popular_articles(self, limit: int = 10, 
                           access_level: AccessLevel = AccessLevel.INTERNAL) -> List[Dict[str, Any]]:
        """Get most popular articles"""
        popular_articles = []
        
        for article in self.articles.values():
            if article.status == ContentStatus.PUBLISHED and self._check_access(article.access_level, access_level):
                popular_articles.append({
                    'id': article.id,
                    'title': article.title,
                    'summary': article.summary,
                    'content_type': article.content_type.value,
                    'view_count': article.view_count,
                    'rating': article.rating,
                    'rating_count': article.rating_count,
                    'tags': article.tags,
                    'categories': article.categories
                })
        
        # Sort by view count
        popular_articles.sort(key=lambda x: x['view_count'], reverse=True)
        
        return popular_articles[:limit]
    
    def get_articles_by_category(self, category: str, 
                                access_level: AccessLevel = AccessLevel.INTERNAL) -> List[Dict[str, Any]]:
        """Get articles by category"""
        category_articles = []
        
        for article in self.articles.values():
            if (article.status == ContentStatus.PUBLISHED and 
                category.lower() in [cat.lower() for cat in article.categories] and
                self._check_access(article.access_level, access_level)):
                
                category_articles.append({
                    'id': article.id,
                    'title': article.title,
                    'summary': article.summary,
                    'content_type': article.content_type.value,
                    'rating': article.rating,
                    'view_count': article.view_count,
                    'last_updated': article.last_updated.isoformat(),
                    'tags': article.tags
                })
        
        # Sort by rating and view count
        category_articles.sort(key=lambda x: (x['rating'], x['view_count']), reverse=True)
        
        return category_articles
    
    def generate_knowledge_base_report(self) -> Dict[str, Any]:
        """Generate comprehensive knowledge base analytics report"""
        total_articles = len(self.articles)
        published_articles = sum(1 for a in self.articles.values() if a.status == ContentStatus.PUBLISHED)
        draft_articles = sum(1 for a in self.articles.values() if a.status == ContentStatus.DRAFT)
        
        # Content type breakdown
        content_type_breakdown = {}
        for content_type in ContentType:
            content_type_breakdown[content_type.value] = sum(1 for a in self.articles.values() 
                                                           if a.content_type == content_type)
        
        # Category breakdown
        all_categories = set()
        for article in self.articles.values():
            all_categories.update(article.categories)
        
        category_breakdown = {}
        for category in all_categories:
            category_breakdown[category] = sum(1 for a in self.articles.values() 
                                             if category in a.categories)
        
        # Usage statistics
        total_views = sum(a.view_count for a in self.articles.values())
        total_searches = len(self.search_queries)
        total_feedback = len(self.feedback)
        
        # Average rating
        rated_articles = [a for a in self.articles.values() if a.rating_count > 0]
        avg_rating = sum(a.rating for a in rated_articles) / len(rated_articles) if rated_articles else 0
        
        # Articles needing review
        now = datetime.now()
        articles_for_review = []
        for article in self.articles.values():
            next_review = article.last_reviewed + timedelta(days=article.review_frequency_days)
            if next_review <= now:
                articles_for_review.append(article.id)
        
        return {
            "total_articles": total_articles,
            "published_articles": published_articles,
            "draft_articles": draft_articles,
            "articles_needing_review": len(articles_for_review),
            "content_type_breakdown": content_type_breakdown,
            "category_breakdown": category_breakdown,
            "total_views": total_views,
            "total_searches": total_searches,
            "total_feedback_submissions": total_feedback,
            "average_rating": round(avg_rating, 2),
            "search_index_size": len(self.search_index),
            "report_generated": datetime.now().isoformat()
        }