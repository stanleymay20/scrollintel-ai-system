"""
Security Policy Management System
Provides comprehensive policy management with version control and approval workflows
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class PolicyStatus(Enum):
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"

class PolicyType(Enum):
    SECURITY = "security"
    PRIVACY = "privacy"
    COMPLIANCE = "compliance"
    OPERATIONAL = "operational"
    TECHNICAL = "technical"

class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"

@dataclass
class PolicyDocument:
    """Security policy document"""
    id: str
    title: str
    description: str
    policy_type: PolicyType
    version: str
    status: PolicyStatus
    content: str
    author: str
    owner: str
    created_date: datetime
    last_updated: datetime
    effective_date: Optional[datetime]
    review_date: datetime
    expiry_date: Optional[datetime]
    tags: List[str]
    related_policies: List[str]
    compliance_frameworks: List[str]
    approval_required: bool
    auto_review: bool
    review_frequency_days: int

@dataclass
class PolicyVersion:
    """Policy version tracking"""
    id: str
    policy_id: str
    version: str
    content: str
    changes_summary: str
    author: str
    created_date: datetime
    change_type: str  # major, minor, patch
    previous_version: Optional[str]

@dataclass
class ApprovalWorkflow:
    """Policy approval workflow"""
    id: str
    policy_id: str
    version: str
    requested_by: str
    request_date: datetime
    approval_steps: List[Dict[str, Any]]
    current_step: int
    status: ApprovalStatus
    comments: List[Dict[str, Any]]
    completed_date: Optional[datetime]

@dataclass
class PolicyReview:
    """Policy review record"""
    id: str
    policy_id: str
    reviewer: str
    review_date: datetime
    review_type: str  # scheduled, ad_hoc, compliance
    findings: List[str]
    recommendations: List[str]
    compliance_status: str
    next_review_date: datetime

class SecurityPolicyManagementSystem:
    """Comprehensive security policy management system"""
    
    def __init__(self, policy_path: str = "security/policies"):
        self.policy_path = Path(policy_path)
        self.policy_path.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.policies: Dict[str, PolicyDocument] = {}
        self.versions: Dict[str, PolicyVersion] = {}
        self.workflows: Dict[str, ApprovalWorkflow] = {}
        self.reviews: Dict[str, PolicyReview] = {}
        
        self._load_policy_data()
        self._initialize_default_policies()
    
    def _load_policy_data(self):
        """Load policy data from storage"""
        # Load policies
        policies_file = self.policy_path / "policies.json"
        if policies_file.exists():
            with open(policies_file, 'r') as f:
                data = json.load(f)
                for policy_id, policy_data in data.items():
                    # Convert datetime strings
                    for date_field in ['created_date', 'last_updated', 'effective_date', 'review_date', 'expiry_date']:
                        if policy_data.get(date_field):
                            policy_data[date_field] = datetime.fromisoformat(policy_data[date_field])
                    
                    # Convert enums
                    policy_data['policy_type'] = PolicyType(policy_data['policy_type'])
                    policy_data['status'] = PolicyStatus(policy_data['status'])
                    
                    self.policies[policy_id] = PolicyDocument(**policy_data)
        
        # Load versions
        versions_file = self.policy_path / "versions.json"
        if versions_file.exists():
            with open(versions_file, 'r') as f:
                data = json.load(f)
                for version_id, version_data in data.items():
                    if version_data.get('created_date'):
                        version_data['created_date'] = datetime.fromisoformat(version_data['created_date'])
                    self.versions[version_id] = PolicyVersion(**version_data)
        
        # Load workflows
        workflows_file = self.policy_path / "workflows.json"
        if workflows_file.exists():
            with open(workflows_file, 'r') as f:
                data = json.load(f)
                for workflow_id, workflow_data in data.items():
                    for date_field in ['request_date', 'completed_date']:
                        if workflow_data.get(date_field):
                            workflow_data[date_field] = datetime.fromisoformat(workflow_data[date_field])
                    
                    workflow_data['status'] = ApprovalStatus(workflow_data['status'])
                    self.workflows[workflow_id] = ApprovalWorkflow(**workflow_data)
        
        # Load reviews
        reviews_file = self.policy_path / "reviews.json"
        if reviews_file.exists():
            with open(reviews_file, 'r') as f:
                data = json.load(f)
                for review_id, review_data in data.items():
                    for date_field in ['review_date', 'next_review_date']:
                        if review_data.get(date_field):
                            review_data[date_field] = datetime.fromisoformat(review_data[date_field])
                    self.reviews[review_id] = PolicyReview(**review_data)
    
    def _save_policy_data(self):
        """Save policy data to storage"""
        # Save policies
        policies_data = {}
        for policy_id, policy in self.policies.items():
            policy_data = asdict(policy)
            
            # Convert datetime objects to strings
            for date_field in ['created_date', 'last_updated', 'effective_date', 'review_date', 'expiry_date']:
                if policy_data.get(date_field) and policy_data[date_field] is not None:
                    policy_data[date_field] = policy_data[date_field].isoformat()
            
            # Convert enums
            policy_data['policy_type'] = policy_data['policy_type'].value
            policy_data['status'] = policy_data['status'].value
            
            policies_data[policy_id] = policy_data
        
        with open(self.policy_path / "policies.json", 'w') as f:
            json.dump(policies_data, f, indent=2)
        
        # Save versions
        versions_data = {}
        for version_id, version in self.versions.items():
            version_data = asdict(version)
            if version_data.get('created_date'):
                version_data['created_date'] = version_data['created_date'].isoformat()
            versions_data[version_id] = version_data
        
        with open(self.policy_path / "versions.json", 'w') as f:
            json.dump(versions_data, f, indent=2)
        
        # Save workflows
        workflows_data = {}
        for workflow_id, workflow in self.workflows.items():
            workflow_data = asdict(workflow)
            
            for date_field in ['request_date', 'completed_date']:
                if workflow_data.get(date_field) and workflow_data[date_field] is not None:
                    workflow_data[date_field] = workflow_data[date_field].isoformat()
            
            workflow_data['status'] = workflow_data['status'].value
            workflows_data[workflow_id] = workflow_data
        
        with open(self.policy_path / "workflows.json", 'w') as f:
            json.dump(workflows_data, f, indent=2)
        
        # Save reviews
        reviews_data = {}
        for review_id, review in self.reviews.items():
            review_data = asdict(review)
            
            for date_field in ['review_date', 'next_review_date']:
                if review_data.get(date_field):
                    review_data[date_field] = review_data[date_field].isoformat()
            
            reviews_data[review_id] = review_data
        
        with open(self.policy_path / "reviews.json", 'w') as f:
            json.dump(reviews_data, f, indent=2)
    
    def _initialize_default_policies(self):
        """Initialize default security policies"""
        if not self.policies:
            default_policies = [
                {
                    "id": "pol-sec-001",
                    "title": "Information Security Policy",
                    "description": "Comprehensive information security policy covering all aspects of data protection",
                    "policy_type": PolicyType.SECURITY,
                    "version": "1.0",
                    "status": PolicyStatus.ACTIVE,
                    "content": """
# Information Security Policy

## 1. Purpose
This policy establishes the framework for protecting information assets and ensuring confidentiality, integrity, and availability of data.

## 2. Scope
This policy applies to all employees, contractors, and third parties with access to company information systems.

## 3. Policy Statements

### 3.1 Access Control
- All users must be authenticated before accessing information systems
- Access rights must follow the principle of least privilege
- User access must be reviewed quarterly

### 3.2 Data Classification
- All data must be classified according to sensitivity levels
- Handling procedures must match classification levels
- Data retention policies must be followed

### 3.3 Incident Response
- All security incidents must be reported immediately
- Incident response procedures must be followed
- Lessons learned must be documented

## 4. Compliance
Violation of this policy may result in disciplinary action up to and including termination.

## 5. Review
This policy will be reviewed annually or as needed.
                    """,
                    "author": "security-team",
                    "owner": "ciso",
                    "effective_date": datetime.now(),
                    "review_date": datetime.now() + timedelta(days=365),
                    "expiry_date": None,
                    "tags": ["security", "access-control", "data-protection"],
                    "related_policies": [],
                    "compliance_frameworks": ["ISO27001", "SOC2", "GDPR"],
                    "approval_required": True,
                    "auto_review": True,
                    "review_frequency_days": 365
                },
                {
                    "id": "pol-priv-001",
                    "title": "Data Privacy Policy",
                    "description": "Policy governing the collection, use, and protection of personal data",
                    "policy_type": PolicyType.PRIVACY,
                    "version": "1.0",
                    "status": PolicyStatus.ACTIVE,
                    "content": """
# Data Privacy Policy

## 1. Purpose
This policy ensures compliance with data protection regulations and protects individual privacy rights.

## 2. Scope
This policy applies to all processing of personal data by the organization.

## 3. Policy Statements

### 3.1 Data Collection
- Personal data must be collected for specified, explicit, and legitimate purposes
- Data subjects must be informed about data collection
- Consent must be obtained where required

### 3.2 Data Processing
- Personal data must be processed lawfully, fairly, and transparently
- Data must be adequate, relevant, and limited to what is necessary
- Data must be accurate and kept up to date

### 3.3 Data Security
- Appropriate technical and organizational measures must be implemented
- Data breaches must be reported within 72 hours
- Privacy by design principles must be followed

### 3.4 Individual Rights
- Data subjects have the right to access their personal data
- Data subjects can request correction or deletion of their data
- Data portability must be supported where applicable

## 4. Compliance
This policy ensures compliance with GDPR, CCPA, and other applicable privacy laws.

## 5. Review
This policy will be reviewed annually or when regulations change.
                    """,
                    "author": "privacy-officer",
                    "owner": "privacy-officer",
                    "effective_date": datetime.now(),
                    "review_date": datetime.now() + timedelta(days=365),
                    "expiry_date": None,
                    "tags": ["privacy", "gdpr", "personal-data"],
                    "related_policies": ["pol-sec-001"],
                    "compliance_frameworks": ["GDPR", "CCPA", "PIPEDA"],
                    "approval_required": True,
                    "auto_review": True,
                    "review_frequency_days": 365
                },
                {
                    "id": "pol-comp-001",
                    "title": "Compliance Management Policy",
                    "description": "Policy for managing regulatory compliance and audit requirements",
                    "policy_type": PolicyType.COMPLIANCE,
                    "version": "1.0",
                    "status": PolicyStatus.ACTIVE,
                    "content": """
# Compliance Management Policy

## 1. Purpose
This policy establishes the framework for managing regulatory compliance and audit requirements.

## 2. Scope
This policy applies to all compliance activities and regulatory requirements.

## 3. Policy Statements

### 3.1 Compliance Framework
- A comprehensive compliance program must be maintained
- Compliance requirements must be identified and tracked
- Regular compliance assessments must be conducted

### 3.2 Audit Management
- Internal audits must be conducted annually
- External audit requirements must be met
- Audit findings must be remediated promptly

### 3.3 Documentation
- All compliance activities must be documented
- Evidence must be maintained for audit purposes
- Documentation must be regularly updated

### 3.4 Training
- Compliance training must be provided to all staff
- Training records must be maintained
- Specialized training must be provided for key roles

## 4. Responsibilities
- Compliance Officer: Overall compliance program management
- Department Heads: Departmental compliance implementation
- All Staff: Adherence to compliance requirements

## 5. Review
This policy will be reviewed annually or when regulations change.
                    """,
                    "author": "compliance-officer",
                    "owner": "compliance-officer",
                    "effective_date": datetime.now(),
                    "review_date": datetime.now() + timedelta(days=365),
                    "expiry_date": None,
                    "tags": ["compliance", "audit", "regulatory"],
                    "related_policies": ["pol-sec-001", "pol-priv-001"],
                    "compliance_frameworks": ["SOC2", "ISO27001", "HIPAA"],
                    "approval_required": True,
                    "auto_review": True,
                    "review_frequency_days": 365
                }
            ]
            
            for policy_data in default_policies:
                policy_data['created_date'] = datetime.now()
                policy_data['last_updated'] = datetime.now()
                
                policy = PolicyDocument(**policy_data)
                self.policies[policy.id] = policy
                
                # Create initial version
                version = PolicyVersion(
                    id=str(uuid.uuid4()),
                    policy_id=policy.id,
                    version=policy.version,
                    content=policy.content,
                    changes_summary="Initial version",
                    author=policy.author,
                    created_date=datetime.now(),
                    change_type="major",
                    previous_version=None
                )
                self.versions[version.id] = version
            
            self._save_policy_data()
    
    def create_policy(self, policy_data: Dict[str, Any]) -> str:
        """Create a new security policy"""
        try:
            policy_id = policy_data.get('id', str(uuid.uuid4()))
            policy_data['id'] = policy_id
            policy_data['created_date'] = datetime.now()
            policy_data['last_updated'] = datetime.now()
            policy_data['status'] = PolicyStatus.DRAFT
            policy_data['version'] = "1.0"
            
            # Convert string enums to enum objects
            policy_data['policy_type'] = PolicyType(policy_data['policy_type'])
            
            # Set default review date if not provided
            if not policy_data.get('review_date'):
                policy_data['review_date'] = datetime.now() + timedelta(days=365)
            
            policy = PolicyDocument(**policy_data)
            self.policies[policy_id] = policy
            
            # Create initial version
            version = PolicyVersion(
                id=str(uuid.uuid4()),
                policy_id=policy_id,
                version=policy.version,
                content=policy.content,
                changes_summary="Initial version",
                author=policy.author,
                created_date=datetime.now(),
                change_type="major",
                previous_version=None
            )
            self.versions[version.id] = version
            
            self._save_policy_data()
            logger.info(f"Created policy: {policy_id}")
            return policy_id
            
        except Exception as e:
            logger.error(f"Failed to create policy: {str(e)}")
            raise
    
    def update_policy(self, policy_id: str, content: str, changes_summary: str, 
                     author: str, change_type: str = "minor") -> str:
        """Update an existing policy"""
        try:
            if policy_id not in self.policies:
                raise ValueError(f"Policy {policy_id} not found")
            
            policy = self.policies[policy_id]
            previous_version = policy.version
            
            # Update version number
            version_parts = policy.version.split('.')
            if change_type == "major":
                version_parts[0] = str(int(version_parts[0]) + 1)
                version_parts[1] = "0"
            elif change_type == "minor":
                version_parts[1] = str(int(version_parts[1]) + 1)
            elif change_type == "patch":
                if len(version_parts) < 3:
                    version_parts.append("1")
                else:
                    version_parts[2] = str(int(version_parts[2]) + 1)
            
            new_version = '.'.join(version_parts)
            
            # Update policy
            policy.content = content
            policy.version = new_version
            policy.last_updated = datetime.now()
            policy.status = PolicyStatus.DRAFT  # Reset to draft for review
            
            # Create new version record
            version = PolicyVersion(
                id=str(uuid.uuid4()),
                policy_id=policy_id,
                version=new_version,
                content=content,
                changes_summary=changes_summary,
                author=author,
                created_date=datetime.now(),
                change_type=change_type,
                previous_version=previous_version
            )
            self.versions[version.id] = version
            
            self._save_policy_data()
            logger.info(f"Updated policy: {policy_id} to version {new_version}")
            return version.id
            
        except Exception as e:
            logger.error(f"Failed to update policy: {str(e)}")
            raise
    
    def submit_for_approval(self, policy_id: str, requested_by: str, 
                           approval_steps: List[Dict[str, Any]]) -> str:
        """Submit policy for approval workflow"""
        try:
            if policy_id not in self.policies:
                raise ValueError(f"Policy {policy_id} not found")
            
            policy = self.policies[policy_id]
            
            # Create approval workflow
            workflow_id = str(uuid.uuid4())
            workflow = ApprovalWorkflow(
                id=workflow_id,
                policy_id=policy_id,
                version=policy.version,
                requested_by=requested_by,
                request_date=datetime.now(),
                approval_steps=approval_steps,
                current_step=0,
                status=ApprovalStatus.PENDING,
                comments=[],
                completed_date=None
            )
            
            self.workflows[workflow_id] = workflow
            
            # Update policy status
            policy.status = PolicyStatus.REVIEW
            policy.last_updated = datetime.now()
            
            self._save_policy_data()
            logger.info(f"Submitted policy {policy_id} for approval: {workflow_id}")
            return workflow_id
            
        except Exception as e:
            logger.error(f"Failed to submit policy for approval: {str(e)}")
            raise
    
    def approve_policy_step(self, workflow_id: str, approver: str, 
                           approved: bool, comments: str = "") -> bool:
        """Approve or reject a policy approval step"""
        try:
            if workflow_id not in self.workflows:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            workflow = self.workflows[workflow_id]
            policy = self.policies[workflow.policy_id]
            
            if workflow.status != ApprovalStatus.PENDING:
                raise ValueError(f"Workflow {workflow_id} is not pending approval")
            
            # Add comment
            comment = {
                "approver": approver,
                "timestamp": datetime.now().isoformat(),
                "approved": approved,
                "comments": comments,
                "step": workflow.current_step
            }
            workflow.comments.append(comment)
            
            if not approved:
                # Rejection - workflow fails
                workflow.status = ApprovalStatus.REJECTED
                workflow.completed_date = datetime.now()
                policy.status = PolicyStatus.DRAFT
                
                logger.info(f"Policy approval rejected by {approver}: {workflow_id}")
                self._save_policy_data()
                return False
            
            # Approval - move to next step
            workflow.current_step += 1
            
            if workflow.current_step >= len(workflow.approval_steps):
                # All steps completed - approve policy
                workflow.status = ApprovalStatus.APPROVED
                workflow.completed_date = datetime.now()
                policy.status = PolicyStatus.APPROVED
                policy.effective_date = datetime.now()
                
                logger.info(f"Policy fully approved: {workflow.policy_id}")
            else:
                logger.info(f"Policy approval step {workflow.current_step} completed: {workflow_id}")
            
            self._save_policy_data()
            return True
            
        except Exception as e:
            logger.error(f"Failed to approve policy step: {str(e)}")
            raise
    
    def activate_policy(self, policy_id: str) -> bool:
        """Activate an approved policy"""
        try:
            if policy_id not in self.policies:
                raise ValueError(f"Policy {policy_id} not found")
            
            policy = self.policies[policy_id]
            
            if policy.status != PolicyStatus.APPROVED:
                raise ValueError(f"Policy {policy_id} is not approved")
            
            policy.status = PolicyStatus.ACTIVE
            policy.last_updated = datetime.now()
            
            if not policy.effective_date:
                policy.effective_date = datetime.now()
            
            self._save_policy_data()
            logger.info(f"Activated policy: {policy_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to activate policy: {str(e)}")
            return False
    
    def schedule_policy_review(self, policy_id: str, reviewer: str, 
                              review_type: str = "scheduled") -> str:
        """Schedule a policy review"""
        try:
            if policy_id not in self.policies:
                raise ValueError(f"Policy {policy_id} not found")
            
            policy = self.policies[policy_id]
            
            review_id = str(uuid.uuid4())
            review = PolicyReview(
                id=review_id,
                policy_id=policy_id,
                reviewer=reviewer,
                review_date=datetime.now(),
                review_type=review_type,
                findings=[],
                recommendations=[],
                compliance_status="pending",
                next_review_date=datetime.now() + timedelta(days=policy.review_frequency_days)
            )
            
            self.reviews[review_id] = review
            
            # Update policy review date
            policy.review_date = review.next_review_date
            policy.last_updated = datetime.now()
            
            self._save_policy_data()
            logger.info(f"Scheduled policy review: {review_id}")
            return review_id
            
        except Exception as e:
            logger.error(f"Failed to schedule policy review: {str(e)}")
            raise
    
    def complete_policy_review(self, review_id: str, findings: List[str], 
                              recommendations: List[str], compliance_status: str) -> bool:
        """Complete a policy review"""
        try:
            if review_id not in self.reviews:
                raise ValueError(f"Review {review_id} not found")
            
            review = self.reviews[review_id]
            review.findings = findings
            review.recommendations = recommendations
            review.compliance_status = compliance_status
            
            # If review identifies issues, may need policy update
            if compliance_status == "non_compliant" or recommendations:
                policy = self.policies[review.policy_id]
                # Could automatically create update task or notification
                logger.warning(f"Policy {review.policy_id} review identified issues")
            
            self._save_policy_data()
            logger.info(f"Completed policy review: {review_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete policy review: {str(e)}")
            return False
    
    def get_policies_for_review(self) -> List[str]:
        """Get policies that need review"""
        policies_for_review = []
        now = datetime.now()
        
        for policy_id, policy in self.policies.items():
            if policy.auto_review and policy.review_date <= now:
                policies_for_review.append(policy_id)
        
        return policies_for_review
    
    def get_policy_compliance_status(self, policy_id: str) -> Dict[str, Any]:
        """Get compliance status for a policy"""
        if policy_id not in self.policies:
            raise ValueError(f"Policy {policy_id} not found")
        
        policy = self.policies[policy_id]
        
        # Get latest review
        policy_reviews = [r for r in self.reviews.values() if r.policy_id == policy_id]
        latest_review = max(policy_reviews, key=lambda x: x.review_date) if policy_reviews else None
        
        # Check if review is overdue
        review_overdue = datetime.now() > policy.review_date
        
        return {
            "policy_id": policy_id,
            "title": policy.title,
            "status": policy.status.value,
            "version": policy.version,
            "effective_date": policy.effective_date.isoformat() if policy.effective_date else None,
            "review_date": policy.review_date.isoformat(),
            "review_overdue": review_overdue,
            "compliance_frameworks": policy.compliance_frameworks,
            "latest_review": {
                "review_date": latest_review.review_date.isoformat() if latest_review else None,
                "compliance_status": latest_review.compliance_status if latest_review else "not_reviewed",
                "findings_count": len(latest_review.findings) if latest_review else 0,
                "recommendations_count": len(latest_review.recommendations) if latest_review else 0
            } if latest_review else None
        }
    
    def generate_policy_report(self) -> Dict[str, Any]:
        """Generate comprehensive policy management report"""
        total_policies = len(self.policies)
        active_policies = sum(1 for p in self.policies.values() if p.status == PolicyStatus.ACTIVE)
        draft_policies = sum(1 for p in self.policies.values() if p.status == PolicyStatus.DRAFT)
        policies_for_review = len(self.get_policies_for_review())
        
        # Policy type breakdown
        type_breakdown = {}
        for policy_type in PolicyType:
            type_breakdown[policy_type.value] = sum(1 for p in self.policies.values() if p.policy_type == policy_type)
        
        # Compliance framework coverage
        framework_coverage = {}
        all_frameworks = set()
        for policy in self.policies.values():
            all_frameworks.update(policy.compliance_frameworks)
        
        for framework in all_frameworks:
            framework_coverage[framework] = sum(1 for p in self.policies.values() 
                                              if framework in p.compliance_frameworks)
        
        # Approval workflow status
        pending_approvals = sum(1 for w in self.workflows.values() if w.status == ApprovalStatus.PENDING)
        
        return {
            "total_policies": total_policies,
            "active_policies": active_policies,
            "draft_policies": draft_policies,
            "policies_needing_review": policies_for_review,
            "pending_approvals": pending_approvals,
            "policy_type_breakdown": type_breakdown,
            "compliance_framework_coverage": framework_coverage,
            "total_versions": len(self.versions),
            "total_reviews": len(self.reviews),
            "report_generated": datetime.now().isoformat()
        }