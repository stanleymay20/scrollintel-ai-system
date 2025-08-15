"""
Marketplace of Verifiers for ScrollIntel-G6.
Allows third parties to publish verifiers and provides bounty program for finding failures.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import importlib.util
import inspect

from .proof_of_workflow import create_workflow_attestation
from .transparency_ledger import add_incident, Severity

logger = logging.getLogger(__name__)


class VerifierCategory(Enum):
    SECURITY = "security"
    ACCESSIBILITY = "accessibility"
    LICENSE = "license"
    ESG = "esg"
    PERFORMANCE = "performance"
    BIAS = "bias"
    SAFETY = "safety"
    COMPLIANCE = "compliance"


class VerifierStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"


@dataclass
class VerifierMetadata:
    """Metadata for a verifier."""
    
    id: str
    name: str
    description: str
    category: VerifierCategory
    version: str
    author: str
    author_email: str
    license: str
    created_at: datetime
    updated_at: datetime
    status: VerifierStatus
    weight: float
    execution_count: int
    success_rate: float
    avg_execution_time: float
    tags: List[str]
    dependencies: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['category'] = self.category.value
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VerifierMetadata':
        """Create from dictionary."""
        data['category'] = VerifierCategory(data['category'])
        data['status'] = VerifierStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class VerificationResult:
    """Result of running a verifier."""
    
    verifier_id: str
    timestamp: datetime
    success: bool
    score: float
    issues: List[str]
    recommendations: List[str]
    execution_time: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class BountySubmission:
    """Bounty submission for finding failures."""
    
    id: str
    submitter: str
    submitter_email: str
    title: str
    description: str
    failure_type: str
    reproduction_steps: List[str]
    evidence: Dict[str, Any]
    severity: Severity
    component: str
    submitted_at: datetime
    status: str  # pending, accepted, rejected, duplicate
    reward_amount: float
    reviewer_notes: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['severity'] = self.severity.value
        data['submitted_at'] = self.submitted_at.isoformat()
        return data


class BaseVerifier:
    """Base class for all verifiers."""
    
    def __init__(self, metadata: VerifierMetadata):
        self.metadata = metadata
    
    async def verify(self, content: Any, context: Dict[str, Any]) -> VerificationResult:
        """Verify content and return result."""
        raise NotImplementedError("Verifiers must implement the verify method")
    
    def get_metadata(self) -> VerifierMetadata:
        """Get verifier metadata."""
        return self.metadata


class AccessibilityVerifier(BaseVerifier):
    """Built-in accessibility verifier."""
    
    def __init__(self):
        metadata = VerifierMetadata(
            id="accessibility_v1",
            name="Accessibility Verifier",
            description="Checks content for accessibility compliance (WCAG 2.1)",
            category=VerifierCategory.ACCESSIBILITY,
            version="1.0.0",
            author="ScrollIntel Team",
            author_email="team@scrollintel.com",
            license="MIT",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            status=VerifierStatus.APPROVED,
            weight=0.8,
            execution_count=0,
            success_rate=0.95,
            avg_execution_time=0.5,
            tags=["accessibility", "wcag", "compliance"],
            dependencies=[]
        )
        super().__init__(metadata)
    
    async def verify(self, content: Any, context: Dict[str, Any]) -> VerificationResult:
        """Verify accessibility compliance."""
        start_time = datetime.utcnow()
        
        issues = []
        recommendations = []
        score = 1.0
        
        # Check if content is text-based
        if isinstance(content, str):
            # Check for alt text mentions in HTML-like content
            if '<img' in content and 'alt=' not in content:
                issues.append("Images without alt text detected")
                score -= 0.3
                recommendations.append("Add alt text to all images")
            
            # Check for heading structure
            if '<h1' not in content and len(content) > 500:
                issues.append("No main heading (h1) found in long content")
                score -= 0.2
                recommendations.append("Add proper heading structure")
            
            # Check for color-only information
            color_words = ['red', 'green', 'blue', 'yellow']
            if any(word in content.lower() for word in color_words):
                if 'color' in content.lower() and 'text' not in content.lower():
                    issues.append("Potential color-only information")
                    score -= 0.1
                    recommendations.append("Ensure information is not conveyed by color alone")
        
        # Check for UI components
        if isinstance(content, dict) and 'ui_components' in context:
            components = context['ui_components']
            
            for component in components:
                if component.get('type') == 'button' and not component.get('aria_label'):
                    issues.append(f"Button without aria-label: {component.get('id', 'unknown')}")
                    score -= 0.2
                    recommendations.append("Add aria-label to all interactive elements")
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        return VerificationResult(
            verifier_id=self.metadata.id,
            timestamp=datetime.utcnow(),
            success=len(issues) == 0,
            score=max(0.0, score),
            issues=issues,
            recommendations=recommendations,
            execution_time=execution_time,
            metadata={
                "wcag_level": "AA",
                "checks_performed": ["alt_text", "heading_structure", "color_dependency", "aria_labels"]
            }
        )


class LicenseVerifier(BaseVerifier):
    """Built-in license compliance verifier."""
    
    def __init__(self):
        metadata = VerifierMetadata(
            id="license_v1",
            name="License Compliance Verifier",
            description="Checks for license compliance and attribution requirements",
            category=VerifierCategory.LICENSE,
            version="1.0.0",
            author="ScrollIntel Team",
            author_email="team@scrollintel.com",
            license="MIT",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            status=VerifierStatus.APPROVED,
            weight=0.9,
            execution_count=0,
            success_rate=0.98,
            avg_execution_time=0.3,
            tags=["license", "compliance", "attribution"],
            dependencies=[]
        )
        super().__init__(metadata)
        
        # Known license patterns
        self.license_patterns = {
            'MIT': r'MIT License|Permission is hereby granted',
            'Apache-2.0': r'Apache License, Version 2\.0|Licensed under the Apache License',
            'GPL-3.0': r'GNU General Public License v3|GPL-3\.0',
            'BSD-3-Clause': r'BSD 3-Clause|Redistribution and use in source and binary forms',
            'Creative Commons': r'Creative Commons|CC BY|CC-BY'
        }
    
    async def verify(self, content: Any, context: Dict[str, Any]) -> VerificationResult:
        """Verify license compliance."""
        start_time = datetime.utcnow()
        
        issues = []
        recommendations = []
        score = 1.0
        detected_licenses = []
        
        # Check content for license information
        if isinstance(content, str):
            import re
            
            for license_name, pattern in self.license_patterns.items():
                if re.search(pattern, content, re.IGNORECASE):
                    detected_licenses.append(license_name)
            
            # Check for copyright notices
            copyright_pattern = r'Copyright \(c\) \d{4}|© \d{4}'
            has_copyright = bool(re.search(copyright_pattern, content, re.IGNORECASE))
            
            # Check for attribution requirements
            if 'attribution' in context:
                required_attributions = context['attribution']
                for attribution in required_attributions:
                    if attribution not in content:
                        issues.append(f"Missing required attribution: {attribution}")
                        score -= 0.3
                        recommendations.append(f"Add attribution: {attribution}")
            
            # Check for proper license headers in code
            if context.get('content_type') == 'code':
                if not detected_licenses and len(content) > 100:
                    issues.append("No license header found in code file")
                    score -= 0.4
                    recommendations.append("Add appropriate license header")
                
                if not has_copyright and len(content) > 100:
                    issues.append("No copyright notice found")
                    score -= 0.2
                    recommendations.append("Add copyright notice")
        
        # Check dependencies for license compatibility
        if 'dependencies' in context:
            dependencies = context['dependencies']
            incompatible_licenses = []
            
            for dep in dependencies:
                dep_license = dep.get('license', 'Unknown')
                if dep_license == 'GPL-3.0' and 'MIT' in detected_licenses:
                    incompatible_licenses.append(f"{dep['name']} ({dep_license})")
            
            if incompatible_licenses:
                issues.append(f"License incompatibility detected: {', '.join(incompatible_licenses)}")
                score -= 0.5
                recommendations.append("Review license compatibility and consider alternatives")
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        return VerificationResult(
            verifier_id=self.metadata.id,
            timestamp=datetime.utcnow(),
            success=len(issues) == 0,
            score=max(0.0, score),
            issues=issues,
            recommendations=recommendations,
            execution_time=execution_time,
            metadata={
                "detected_licenses": detected_licenses,
                "has_copyright": has_copyright if 'has_copyright' in locals() else False,
                "checks_performed": ["license_detection", "copyright_notice", "attribution", "compatibility"]
            }
        )


class VerifierRegistry:
    """Registry for managing verifiers."""
    
    def __init__(self):
        self.verifiers: Dict[str, BaseVerifier] = {}
        self.governance_council = GovernanceCouncil()
        self._load_builtin_verifiers()
    
    def _load_builtin_verifiers(self) -> None:
        """Load built-in verifiers."""
        builtin_verifiers = [
            AccessibilityVerifier(),
            LicenseVerifier()
        ]
        
        for verifier in builtin_verifiers:
            self.verifiers[verifier.metadata.id] = verifier
            logger.info(f"Loaded built-in verifier: {verifier.metadata.name}")
    
    async def register_verifier(
        self,
        verifier_code: str,
        metadata: Dict[str, Any],
        submitter: str
    ) -> Tuple[bool, str]:
        """Register a new third-party verifier."""
        
        try:
            # Create metadata object
            verifier_metadata = VerifierMetadata.from_dict(metadata)
            verifier_metadata.status = VerifierStatus.PENDING
            
            # Validate verifier code
            validation_result = await self._validate_verifier_code(verifier_code, verifier_metadata)
            if not validation_result["valid"]:
                return False, f"Validation failed: {validation_result['error']}"
            
            # Submit to governance council for approval
            approval_result = await self.governance_council.review_verifier(
                verifier_code, verifier_metadata, submitter
            )
            
            if approval_result["approved"]:
                # Load and register the verifier
                verifier = await self._load_verifier_from_code(verifier_code, verifier_metadata)
                self.verifiers[verifier.metadata.id] = verifier
                
                logger.info(f"Registered new verifier: {verifier.metadata.name}")
                return True, "Verifier registered successfully"
            else:
                return False, f"Approval denied: {approval_result['reason']}"
                
        except Exception as e:
            logger.error(f"Error registering verifier: {e}")
            return False, f"Registration error: {str(e)}"
    
    async def _validate_verifier_code(
        self,
        code: str,
        metadata: VerifierMetadata
    ) -> Dict[str, Any]:
        """Validate verifier code for security and compliance."""
        
        validation_result = {"valid": True, "error": None}
        
        # Check for dangerous imports
        dangerous_imports = ['os', 'subprocess', 'sys', 'eval', 'exec', '__import__']
        for dangerous in dangerous_imports:
            if f"import {dangerous}" in code or f"from {dangerous}" in code:
                validation_result["valid"] = False
                validation_result["error"] = f"Dangerous import detected: {dangerous}"
                return validation_result
        
        # Check for required methods
        if "async def verify(" not in code:
            validation_result["valid"] = False
            validation_result["error"] = "Verifier must implement async verify method"
            return validation_result
        
        # Check code size limits
        if len(code) > 50000:  # 50KB limit
            validation_result["valid"] = False
            validation_result["error"] = "Verifier code exceeds size limit"
            return validation_result
        
        # Try to compile the code
        try:
            compile(code, '<verifier>', 'exec')
        except SyntaxError as e:
            validation_result["valid"] = False
            validation_result["error"] = f"Syntax error: {str(e)}"
            return validation_result
        
        return validation_result
    
    async def _load_verifier_from_code(
        self,
        code: str,
        metadata: VerifierMetadata
    ) -> BaseVerifier:
        """Load verifier from code string."""
        
        # Create a module from the code
        spec = importlib.util.spec_from_loader("custom_verifier", loader=None)
        module = importlib.util.module_from_spec(spec)
        
        # Execute the code in the module namespace
        exec(code, module.__dict__)
        
        # Find the verifier class
        verifier_class = None
        for name, obj in module.__dict__.items():
            if (inspect.isclass(obj) and 
                issubclass(obj, BaseVerifier) and 
                obj != BaseVerifier):
                verifier_class = obj
                break
        
        if not verifier_class:
            raise ValueError("No verifier class found in code")
        
        # Instantiate the verifier
        return verifier_class(metadata)
    
    def get_verifier(self, verifier_id: str) -> Optional[BaseVerifier]:
        """Get a verifier by ID."""
        return self.verifiers.get(verifier_id)
    
    def list_verifiers(
        self,
        category: Optional[VerifierCategory] = None,
        status: Optional[VerifierStatus] = None
    ) -> List[VerifierMetadata]:
        """List available verifiers."""
        
        verifiers = []
        for verifier in self.verifiers.values():
            metadata = verifier.get_metadata()
            
            if category and metadata.category != category:
                continue
            
            if status and metadata.status != status:
                continue
            
            verifiers.append(metadata)
        
        return verifiers
    
    async def run_verifier(
        self,
        verifier_id: str,
        content: Any,
        context: Dict[str, Any]
    ) -> Optional[VerificationResult]:
        """Run a specific verifier."""
        
        verifier = self.get_verifier(verifier_id)
        if not verifier:
            logger.error(f"Verifier not found: {verifier_id}")
            return None
        
        try:
            result = await verifier.verify(content, context)
            
            # Update verifier statistics
            verifier.metadata.execution_count += 1
            verifier.metadata.avg_execution_time = (
                (verifier.metadata.avg_execution_time * (verifier.metadata.execution_count - 1) + 
                 result.execution_time) / verifier.metadata.execution_count
            )
            
            if result.success:
                verifier.metadata.success_rate = (
                    (verifier.metadata.success_rate * (verifier.metadata.execution_count - 1) + 1.0) /
                    verifier.metadata.execution_count
                )
            else:
                verifier.metadata.success_rate = (
                    (verifier.metadata.success_rate * (verifier.metadata.execution_count - 1) + 0.0) /
                    verifier.metadata.execution_count
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error running verifier {verifier_id}: {e}")
            return None


class GovernanceCouncil:
    """Governance council for reviewing and approving verifiers."""
    
    def __init__(self):
        self.council_members = [
            "security_expert",
            "accessibility_expert", 
            "legal_expert",
            "technical_lead"
        ]
        self.approval_threshold = 0.75  # 75% approval required
    
    async def review_verifier(
        self,
        code: str,
        metadata: VerifierMetadata,
        submitter: str
    ) -> Dict[str, Any]:
        """Review a verifier for approval."""
        
        # Simulate council review process
        reviews = []
        
        for member in self.council_members:
            review = await self._simulate_member_review(member, code, metadata)
            reviews.append(review)
        
        # Calculate approval score
        approval_scores = [r["score"] for r in reviews]
        avg_score = sum(approval_scores) / len(approval_scores)
        
        approved = avg_score >= self.approval_threshold
        
        # Compile feedback
        feedback = []
        for review in reviews:
            if review["feedback"]:
                feedback.extend(review["feedback"])
        
        return {
            "approved": approved,
            "score": avg_score,
            "reviews": reviews,
            "feedback": feedback,
            "reason": "Approved by governance council" if approved else "Insufficient approval score"
        }
    
    async def _simulate_member_review(
        self,
        member: str,
        code: str,
        metadata: VerifierMetadata
    ) -> Dict[str, Any]:
        """Simulate a council member's review."""
        
        # Simulate different review criteria based on member expertise
        if member == "security_expert":
            # Focus on security aspects
            score = 0.9 if "import os" not in code else 0.3
            feedback = ["Security review passed"] if score > 0.5 else ["Security concerns identified"]
            
        elif member == "accessibility_expert":
            # Focus on accessibility aspects
            score = 0.8 if metadata.category == VerifierCategory.ACCESSIBILITY else 0.7
            feedback = ["Accessibility considerations reviewed"]
            
        elif member == "legal_expert":
            # Focus on legal compliance
            score = 0.85 if metadata.license in ["MIT", "Apache-2.0"] else 0.6
            feedback = ["Legal compliance reviewed"]
            
        else:  # technical_lead
            # Focus on technical quality
            score = 0.8 if len(code) < 10000 else 0.6
            feedback = ["Technical implementation reviewed"]
        
        return {
            "member": member,
            "score": score,
            "feedback": feedback,
            "timestamp": datetime.utcnow().isoformat()
        }


class BountyProgram:
    """Bounty program for finding failures and vulnerabilities."""
    
    def __init__(self):
        self.submissions: List[BountySubmission] = []
        self.reward_schedule = {
            Severity.CRITICAL: 5000.0,
            Severity.HIGH: 2000.0,
            Severity.MEDIUM: 500.0,
            Severity.LOW: 100.0,
            Severity.INFO: 25.0
        }
        self.total_paid = 0.0
    
    def submit_bounty(
        self,
        submitter: str,
        submitter_email: str,
        title: str,
        description: str,
        failure_type: str,
        reproduction_steps: List[str],
        evidence: Dict[str, Any],
        severity: Severity,
        component: str
    ) -> BountySubmission:
        """Submit a bounty for a found failure."""
        
        import uuid
        
        submission = BountySubmission(
            id=str(uuid.uuid4()),
            submitter=submitter,
            submitter_email=submitter_email,
            title=title,
            description=description,
            failure_type=failure_type,
            reproduction_steps=reproduction_steps,
            evidence=evidence,
            severity=severity,
            component=component,
            submitted_at=datetime.utcnow(),
            status="pending",
            reward_amount=self.reward_schedule.get(severity, 0.0),
            reviewer_notes=""
        )
        
        self.submissions.append(submission)
        
        # Create incident in transparency ledger
        add_incident(
            component=component,
            severity=severity,
            title=f"Bounty Submission: {title}",
            description=description,
            metadata={
                "bounty_id": submission.id,
                "submitter": submitter,
                "failure_type": failure_type
            }
        )
        
        logger.info(f"Bounty submission received: {title} (${submission.reward_amount})")
        return submission
    
    async def review_submission(
        self,
        submission_id: str,
        reviewer: str,
        status: str,
        notes: str
    ) -> bool:
        """Review a bounty submission."""
        
        submission = next((s for s in self.submissions if s.id == submission_id), None)
        if not submission:
            return False
        
        submission.status = status
        submission.reviewer_notes = notes
        
        if status == "accepted":
            # Process payment (simulated)
            self.total_paid += submission.reward_amount
            
            # Create regression test
            await self._create_regression_test(submission)
            
            logger.info(f"Bounty accepted: {submission.title} (${submission.reward_amount})")
        
        return True
    
    async def _create_regression_test(self, submission: BountySubmission) -> None:
        """Create a regression test from accepted bounty submission."""
        
        # This would create an actual test case
        test_case = {
            "id": f"regression_{submission.id}",
            "title": f"Regression test for: {submission.title}",
            "description": submission.description,
            "steps": submission.reproduction_steps,
            "expected_failure": True,
            "component": submission.component,
            "created_from_bounty": submission.id
        }
        
        # Store test case (in practice, this would integrate with test framework)
        logger.info(f"Created regression test: {test_case['id']}")
    
    def get_submissions(
        self,
        status: Optional[str] = None,
        severity: Optional[Severity] = None,
        component: Optional[str] = None
    ) -> List[BountySubmission]:
        """Get bounty submissions with optional filters."""
        
        filtered = self.submissions
        
        if status:
            filtered = [s for s in filtered if s.status == status]
        
        if severity:
            filtered = [s for s in filtered if s.severity == severity]
        
        if component:
            filtered = [s for s in filtered if s.component == component]
        
        return filtered
    
    def get_bounty_stats(self) -> Dict[str, Any]:
        """Get bounty program statistics."""
        
        return {
            "total_submissions": len(self.submissions),
            "accepted_submissions": len([s for s in self.submissions if s.status == "accepted"]),
            "total_paid": self.total_paid,
            "avg_reward": self.total_paid / max(1, len([s for s in self.submissions if s.status == "accepted"])),
            "by_severity": {
                severity.value: len([s for s in self.submissions if s.severity == severity])
                for severity in Severity
            },
            "by_status": {
                status: len([s for s in self.submissions if s.status == status])
                for status in ["pending", "accepted", "rejected", "duplicate"]
            }
        }


class MarketplaceOfVerifiers:
    """Main marketplace service for verifiers."""
    
    def __init__(self):
        self.registry = VerifierRegistry()
        self.bounty_program = BountyProgram()
        self.verification_history: List[VerificationResult] = []
    
    async def run_verification_suite(
        self,
        content: Any,
        context: Dict[str, Any],
        verifier_ids: Optional[List[str]] = None,
        user_id: str = "system"
    ) -> Dict[str, Any]:
        """Run a suite of verifiers on content."""
        
        if verifier_ids is None:
            # Run all approved verifiers
            approved_verifiers = self.registry.list_verifiers(status=VerifierStatus.APPROVED)
            verifier_ids = [v.id for v in approved_verifiers]
        
        results = {}
        total_score = 0.0
        total_weight = 0.0
        all_issues = []
        all_recommendations = []
        
        # Run each verifier
        for verifier_id in verifier_ids:
            result = await self.registry.run_verifier(verifier_id, content, context)
            
            if result:
                results[verifier_id] = result
                self.verification_history.append(result)
                
                # Calculate weighted score
                verifier = self.registry.get_verifier(verifier_id)
                if verifier:
                    weight = verifier.metadata.weight
                    total_score += result.score * weight
                    total_weight += weight
                    
                    # Collect issues and recommendations
                    all_issues.extend([f"[{verifier.metadata.name}] {issue}" for issue in result.issues])
                    all_recommendations.extend([f"[{verifier.metadata.name}] {rec}" for rec in result.recommendations])
        
        # Calculate overall score
        overall_score = total_score / total_weight if total_weight > 0 else 0.0
        
        # Create summary
        summary = {
            "overall_score": overall_score,
            "verifiers_run": len(results),
            "total_issues": len(all_issues),
            "total_recommendations": len(all_recommendations),
            "all_passed": all(r.success for r in results.values()),
            "results": {vid: r.to_dict() for vid, r in results.items()},
            "issues": all_issues,
            "recommendations": all_recommendations,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Create workflow attestation
        create_workflow_attestation(
            action_type="verification_suite",
            agent_id="marketplace_verifiers",
            user_id=user_id,
            prompt=f"Verification suite with {len(verifier_ids)} verifiers",
            tools_used=["verifier_registry"] + verifier_ids,
            datasets_used=[],
            model_version="marketplace-v1.0",
            verifier_evidence={
                "overall_score": overall_score,
                "verifiers_run": len(results),
                "all_passed": summary["all_passed"]
            },
            content=summary
        )
        
        logger.info(f"Verification suite completed: {overall_score:.2f} score, {len(all_issues)} issues")
        return summary
    
    def get_marketplace_stats(self) -> Dict[str, Any]:
        """Get marketplace statistics."""
        
        verifiers = self.registry.list_verifiers()
        
        return {
            "total_verifiers": len(verifiers),
            "by_category": {
                category.value: len([v for v in verifiers if v.category == category])
                for category in VerifierCategory
            },
            "by_status": {
                status.value: len([v for v in verifiers if v.status == status])
                for status in VerifierStatus
            },
            "total_verifications": len(self.verification_history),
            "avg_score": sum(r.score for r in self.verification_history) / max(1, len(self.verification_history)),
            "bounty_stats": self.bounty_program.get_bounty_stats()
        }


# Global marketplace instance
marketplace = MarketplaceOfVerifiers()


async def run_verification_suite(
    content: Any,
    context: Dict[str, Any],
    verifier_ids: Optional[List[str]] = None,
    user_id: str = "system"
) -> Dict[str, Any]:
    """Run verification suite (convenience function)."""
    return await marketplace.run_verification_suite(content, context, verifier_ids, user_id)


async def register_verifier(
    verifier_code: str,
    metadata: Dict[str, Any],
    submitter: str
) -> Tuple[bool, str]:
    """Register a new verifier (convenience function)."""
    return await marketplace.registry.register_verifier(verifier_code, metadata, submitter)


def submit_bounty(
    submitter: str,
    submitter_email: str,
    title: str,
    description: str,
    failure_type: str,
    reproduction_steps: List[str],
    evidence: Dict[str, Any],
    severity: Severity,
    component: str
) -> BountySubmission:
    """Submit a bounty (convenience function)."""
    return marketplace.bounty_program.submit_bounty(
        submitter, submitter_email, title, description, failure_type,
        reproduction_steps, evidence, severity, component
    )


def get_marketplace_stats() -> Dict[str, Any]:
    """Get marketplace statistics (convenience function)."""
    return marketplace.get_marketplace_stats()