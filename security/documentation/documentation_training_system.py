"""
Integrated Documentation and Training System
Main system that integrates all documentation, training, awareness, and knowledge management components
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from .documentation_manager import SecurityDocumentationManager
from ..training.training_system import SecurityTrainingSystem
from ..awareness.phishing_simulator import SecurityAwarenessSystem
from ..incident_response.playbook_system import IncidentResponsePlaybookSystem
from ..policy.policy_management_system import SecurityPolicyManagementSystem
from ..knowledge_base.knowledge_base_system import SecurityKnowledgeBaseSystem

logger = logging.getLogger(__name__)

class IntegratedDocumentationTrainingSystem:
    """
    Comprehensive documentation and training system that integrates all security
    documentation, training, awareness, and knowledge management components
    """
    
    def __init__(self, base_path: str = "security"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize all subsystems
        self.documentation_manager = SecurityDocumentationManager(
            docs_path=str(self.base_path / "docs"),
            templates_path=str(self.base_path / "templates")
        )
        
        self.training_system = SecurityTrainingSystem(
            training_path=str(self.base_path / "training")
        )
        
        self.awareness_system = SecurityAwarenessSystem(
            awareness_path=str(self.base_path / "awareness")
        )
        
        self.playbook_system = IncidentResponsePlaybookSystem(
            playbook_path=str(self.base_path / "incident_response")
        )
        
        self.policy_system = SecurityPolicyManagementSystem(
            policy_path=str(self.base_path / "policies")
        )
        
        self.knowledge_base = SecurityKnowledgeBaseSystem(
            kb_path=str(self.base_path / "knowledge_base")
        )
        
        # Integration tracking
        self.integration_metrics = {}
        self._initialize_integration()
    
    def _initialize_integration(self):
        """Initialize cross-system integration"""
        # Create cross-references between systems
        self._create_documentation_training_links()
        self._setup_automated_updates()
        self._initialize_metrics_tracking()
    
    def _create_documentation_training_links(self):
        """Create links between documentation and training content"""
        # Link security policies to training modules
        for policy_id, policy in self.policy_system.policies.items():
            if policy.status.value == "active":
                # Find related training modules
                related_modules = []
                for module_id, module in self.training_system.modules.items():
                    if any(tag in policy.tags for tag in module.learning_objectives):
                        related_modules.append(module_id)
                
                # Create knowledge base articles for policies
                if not any(article.title == f"Policy Guide: {policy.title}" 
                          for article in self.knowledge_base.articles.values()):
                    self._create_policy_knowledge_article(policy)
    
    def _create_policy_knowledge_article(self, policy):
        """Create knowledge base article for a policy"""
        article_data = {
            "title": f"Policy Guide: {policy.title}",
            "summary": f"Implementation guide for {policy.title}",
            "content": f"""
# {policy.title} Implementation Guide

## Policy Overview
{policy.description}

## Key Requirements
{self._extract_policy_requirements(policy.content)}

## Implementation Steps
{self._generate_implementation_steps(policy)}

## Compliance Checklist
{self._generate_compliance_checklist(policy)}

## Related Training
{self._get_related_training_modules(policy)}

## Related Documents
- [Full Policy Document](policy:{policy.id})
- [Compliance Framework](compliance:{policy.compliance_frameworks})

## Questions and Support
For questions about this policy, contact the policy owner: {policy.owner}
            """,
            "content_type": "reference",
            "access_level": "internal",
            "author": "system-integration",
            "tags": policy.tags + ["policy-guide", "implementation"],
            "categories": ["policies", "compliance"],
            "search_keywords": policy.tags + [policy.title.lower()]
        }
        
        self.knowledge_base.create_article(article_data)
    
    def _extract_policy_requirements(self, policy_content: str) -> str:
        """Extract key requirements from policy content"""
        # Simple extraction - in practice, this would use NLP
        lines = policy_content.split('\n')
        requirements = []
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['must', 'shall', 'required', 'mandatory']):
                requirements.append(f"- {line.strip()}")
        
        return '\n'.join(requirements[:10])  # Top 10 requirements
    
    def _generate_implementation_steps(self, policy) -> str:
        """Generate implementation steps for a policy"""
        return f"""
1. Review policy requirements and scope
2. Identify affected systems and processes
3. Develop implementation plan
4. Assign responsibilities to team members
5. Implement required controls and procedures
6. Test and validate implementation
7. Train staff on new procedures
8. Monitor compliance and effectiveness
9. Schedule regular reviews
10. Document lessons learned
        """
    
    def _generate_compliance_checklist(self, policy) -> str:
        """Generate compliance checklist for a policy"""
        return f"""
- [ ] Policy has been reviewed and understood
- [ ] Required controls have been implemented
- [ ] Staff have been trained on policy requirements
- [ ] Monitoring and reporting mechanisms are in place
- [ ] Regular review schedule has been established
- [ ] Compliance documentation is maintained
- [ ] Incident response procedures are defined
- [ ] Audit trail is maintained
        """
    
    def _get_related_training_modules(self, policy) -> str:
        """Get related training modules for a policy"""
        related_modules = []
        for module_id, module in self.training_system.modules.items():
            if any(tag in policy.tags for tag in [obj.lower() for obj in module.learning_objectives]):
                related_modules.append(f"- [{module.title}](training:{module_id})")
        
        return '\n'.join(related_modules) if related_modules else "No specific training modules found."
    
    def _setup_automated_updates(self):
        """Setup automated updates between systems"""
        # This would set up triggers for cross-system updates
        # For example, when a policy is updated, related training should be reviewed
        pass
    
    def _initialize_metrics_tracking(self):
        """Initialize metrics tracking across all systems"""
        self.integration_metrics = {
            "last_updated": datetime.now(),
            "cross_references_created": 0,
            "automated_updates_processed": 0,
            "user_engagement_score": 0.0
        }
    
    def create_comprehensive_security_program(self, program_name: str, 
                                            requirements: List[str]) -> Dict[str, str]:
        """Create a comprehensive security program with all components"""
        try:
            program_ids = {}
            
            # 1. Create foundational policy
            policy_data = {
                "title": f"{program_name} Security Policy",
                "description": f"Comprehensive security policy for {program_name}",
                "policy_type": "security",
                "content": self._generate_policy_content(program_name, requirements),
                "author": "security-team",
                "owner": "ciso",
                "tags": [program_name.lower().replace(' ', '-'), "security", "policy"],
                "compliance_frameworks": ["ISO27001", "SOC2"],
                "approval_required": True
            }
            program_ids['policy'] = self.policy_system.create_policy(policy_data)
            
            # 2. Create training modules
            training_modules = self._create_training_modules(program_name, requirements)
            program_ids['training_modules'] = []
            for module_data in training_modules:
                module_id = self.training_system.create_training_module(module_data)
                program_ids['training_modules'].append(module_id)
            
            # 3. Create awareness campaign
            awareness_campaign = self._create_awareness_campaign(program_name, requirements)
            program_ids['awareness_campaign'] = self.awareness_system.create_phishing_campaign(awareness_campaign)
            
            # 4. Create incident response playbook
            playbook_data = self._create_incident_playbook(program_name, requirements)
            program_ids['incident_playbook'] = playbook_data['id']
            self.playbook_system.playbooks[playbook_data['id']] = playbook_data
            
            # 5. Create knowledge base articles
            kb_articles = self._create_knowledge_base_articles(program_name, requirements)
            program_ids['knowledge_articles'] = []
            for article_data in kb_articles:
                article_id = self.knowledge_base.create_article(article_data)
                program_ids['knowledge_articles'].append(article_id)
            
            # 6. Create documentation templates
            doc_templates = self._create_documentation_templates(program_name, requirements)
            program_ids['documentation'] = []
            for doc_data in doc_templates:
                doc_id = self.documentation_manager.create_document(
                    doc_data['id'], doc_data['title'], doc_data['content'],
                    doc_data.get('classification', 'internal'), doc_data.get('tags', [])
                )
                if doc_id:
                    program_ids['documentation'].append(doc_id)
            
            logger.info(f"Created comprehensive security program: {program_name}")
            return program_ids
            
        except Exception as e:
            logger.error(f"Failed to create security program: {str(e)}")
            raise
    
    def _generate_policy_content(self, program_name: str, requirements: List[str]) -> str:
        """Generate policy content based on program requirements"""
        return f"""
# {program_name} Security Policy

## 1. Purpose
This policy establishes the security framework for {program_name} to ensure the confidentiality, integrity, and availability of information assets.

## 2. Scope
This policy applies to all systems, processes, and personnel involved in {program_name}.

## 3. Policy Statements

### 3.1 Security Requirements
{chr(10).join(f"- {req}" for req in requirements)}

### 3.2 Responsibilities
- Management: Provide resources and support for security implementation
- Security Team: Develop and maintain security controls
- All Personnel: Follow security procedures and report incidents

### 3.3 Compliance
- All activities must comply with applicable laws and regulations
- Regular audits will be conducted to ensure compliance
- Non-compliance may result in disciplinary action

## 4. Implementation
This policy will be implemented through:
- Technical controls and security measures
- Training and awareness programs
- Regular monitoring and assessment
- Incident response procedures

## 5. Review
This policy will be reviewed annually or as needed based on changes in requirements or threats.
        """
    
    def _create_training_modules(self, program_name: str, requirements: List[str]) -> List[Dict[str, Any]]:
        """Create training modules for the security program"""
        modules = [
            {
                "title": f"{program_name} Security Fundamentals",
                "description": f"Basic security principles for {program_name}",
                "type": "general",
                "difficulty": "beginner",
                "duration_minutes": 45,
                "prerequisites": [],
                "learning_objectives": [
                    f"Understand {program_name} security requirements",
                    "Identify security threats and risks",
                    "Apply security best practices"
                ],
                "content_sections": [
                    {
                        "title": "Introduction",
                        "type": "video",
                        "content": f"{program_name}_intro.mp4",
                        "duration": 15
                    },
                    {
                        "title": "Security Requirements",
                        "type": "interactive",
                        "content": f"{program_name}_requirements.html",
                        "duration": 20
                    },
                    {
                        "title": "Best Practices",
                        "type": "document",
                        "content": f"{program_name}_practices.pdf",
                        "duration": 10
                    }
                ],
                "assessment_questions": [
                    {
                        "question": f"What is the primary goal of {program_name} security?",
                        "type": "multiple_choice",
                        "options": [
                            "Prevent all access",
                            "Protect information assets",
                            "Comply with regulations",
                            "Reduce costs"
                        ],
                        "correct_answer": 1,
                        "points": 10
                    }
                ],
                "passing_score": 80,
                "certification_points": 15,
                "mandatory": True,
                "frequency_days": 365
            }
        ]
        
        return modules
    
    def _create_awareness_campaign(self, program_name: str, requirements: List[str]) -> Dict[str, Any]:
        """Create awareness campaign for the security program"""
        return {
            "name": f"{program_name} Security Awareness Campaign",
            "description": f"Security awareness campaign for {program_name}",
            "template_ids": ["phish-001", "phish-002"],  # Use existing templates
            "target_groups": ["all_employees"],
            "start_date": datetime.now(),
            "end_date": datetime.now() + timedelta(days=90),
            "frequency": "monthly",
            "created_by": "security-team"
        }
    
    def _create_incident_playbook(self, program_name: str, requirements: List[str]) -> Dict[str, Any]:
        """Create incident response playbook for the security program"""
        from ..incident_response.playbook_system import IncidentPlaybook, PlaybookAction, ActionType, PlaybookStatus, IncidentSeverity
        
        actions = [
            PlaybookAction(
                id="action-001",
                name="Assess Incident Scope",
                description=f"Determine the scope and impact of the {program_name} security incident",
                action_type=ActionType.MANUAL,
                required=True,
                timeout_minutes=30,
                dependencies=[],
                automation_script=None,
                notification_targets=["security-team"],
                escalation_conditions={"timeout": True},
                validation_criteria={"scope_assessed": True},
                order=1
            ),
            PlaybookAction(
                id="action-002",
                name="Contain Incident",
                description="Implement containment measures to prevent further damage",
                action_type=ActionType.AUTOMATED,
                required=True,
                timeout_minutes=15,
                dependencies=["action-001"],
                automation_script="contain_incident.py",
                notification_targets=["security-team", "it-ops"],
                escalation_conditions={"timeout": True},
                validation_criteria={"incident_contained": True},
                order=2
            )
        ]
        
        return {
            "id": f"pb-{program_name.lower().replace(' ', '-')}-001",
            "name": f"{program_name} Incident Response",
            "description": f"Incident response procedures for {program_name}",
            "incident_types": [program_name.lower().replace(' ', '_')],
            "severity_levels": [IncidentSeverity.MEDIUM, IncidentSeverity.HIGH, IncidentSeverity.CRITICAL],
            "trigger_conditions": {"indicators": [f"{program_name.lower()}_incident"]},
            "actions": actions,
            "estimated_duration_minutes": 120,
            "required_roles": ["security-analyst", "incident-commander"],
            "escalation_path": ["security-manager", "ciso"],
            "success_criteria": [
                "Incident contained and resolved",
                "No data loss or unauthorized access",
                "Normal operations restored"
            ],
            "status": PlaybookStatus.ACTIVE,
            "version": "1.0",
            "created_date": datetime.now(),
            "last_updated": datetime.now(),
            "created_by": "security-team"
        }
    
    def _create_knowledge_base_articles(self, program_name: str, requirements: List[str]) -> List[Dict[str, Any]]:
        """Create knowledge base articles for the security program"""
        articles = [
            {
                "title": f"{program_name} Quick Start Guide",
                "summary": f"Quick start guide for implementing {program_name} security",
                "content": f"""
# {program_name} Quick Start Guide

## Overview
This guide provides step-by-step instructions for getting started with {program_name} security.

## Prerequisites
- Access to {program_name} systems
- Completion of security training
- Understanding of security policies

## Getting Started

### Step 1: Initial Setup
1. Review security requirements
2. Configure access controls
3. Enable security monitoring

### Step 2: Daily Operations
1. Follow security procedures
2. Monitor for security events
3. Report any incidents immediately

### Step 3: Regular Maintenance
1. Review security logs
2. Update security configurations
3. Participate in security training

## Troubleshooting
Common issues and solutions:
- Access denied: Check permissions and contact IT
- Security alert: Follow incident response procedures
- System unavailable: Check system status and escalate if needed

## Support
For additional support:
- Email: security@company.com
- Phone: +1-555-SECURITY
- Internal chat: #security-support
                """,
                "content_type": "tutorial",
                "access_level": "internal",
                "author": "security-team",
                "tags": [program_name.lower().replace(' ', '-'), "quick-start", "guide"],
                "categories": ["getting-started", "tutorials"],
                "search_keywords": [program_name.lower(), "setup", "guide", "tutorial"]
            },
            {
                "title": f"{program_name} Troubleshooting Guide",
                "summary": f"Common issues and solutions for {program_name}",
                "content": f"""
# {program_name} Troubleshooting Guide

## Common Issues

### Issue 1: Access Problems
**Symptoms:** Cannot access {program_name} systems
**Causes:** 
- Expired credentials
- Insufficient permissions
- Network connectivity issues

**Solutions:**
1. Check credential expiration
2. Verify permissions with administrator
3. Test network connectivity
4. Contact IT support if issues persist

### Issue 2: Security Alerts
**Symptoms:** Receiving security alerts or warnings
**Causes:**
- Suspicious activity detected
- Policy violations
- System anomalies

**Solutions:**
1. Do not ignore alerts
2. Follow incident response procedures
3. Document all actions taken
4. Report to security team immediately

### Issue 3: Performance Issues
**Symptoms:** Slow system performance
**Causes:**
- High security scanning load
- Network congestion
- Resource constraints

**Solutions:**
1. Check system resources
2. Review security scan schedules
3. Optimize configurations
4. Escalate to technical team

## Getting Help
- Check this knowledge base first
- Contact IT support for technical issues
- Contact security team for security concerns
- Use internal chat for quick questions
                """,
                "content_type": "troubleshooting",
                "access_level": "internal",
                "author": "security-team",
                "tags": [program_name.lower().replace(' ', '-'), "troubleshooting", "support"],
                "categories": ["troubleshooting", "support"],
                "search_keywords": [program_name.lower(), "problems", "issues", "help"]
            }
        ]
        
        return articles
    
    def _create_documentation_templates(self, program_name: str, requirements: List[str]) -> List[Dict[str, Any]]:
        """Create documentation templates for the security program"""
        templates = [
            {
                "id": f"doc-{program_name.lower().replace(' ', '-')}-implementation",
                "title": f"{program_name} Implementation Guide",
                "content": f"""
# {program_name} Implementation Guide

## Document Information
- **Document ID**: {{{{ document_id }}}}
- **Version**: {{{{ version }}}}
- **Date**: {{{{ current_date }}}}
- **Author**: {{{{ author }}}}

## Implementation Overview
This document provides detailed implementation guidance for {program_name} security requirements.

## Requirements Mapping
{chr(10).join(f"- {req}" for req in requirements)}

## Implementation Steps
1. **Planning Phase**
   - Review requirements
   - Assess current state
   - Develop implementation plan

2. **Implementation Phase**
   - Deploy security controls
   - Configure monitoring
   - Test functionality

3. **Validation Phase**
   - Verify requirements compliance
   - Conduct security testing
   - Document results

## Compliance Verification
- [ ] All requirements implemented
- [ ] Security controls tested
- [ ] Documentation completed
- [ ] Training provided
- [ ] Monitoring enabled

## Maintenance
- Regular reviews scheduled
- Update procedures documented
- Contact information current
                """,
                "classification": "internal",
                "tags": [program_name.lower().replace(' ', '-'), "implementation", "guide"]
            },
            {
                "id": f"doc-{program_name.lower().replace(' ', '-')}-procedures",
                "title": f"{program_name} Operating Procedures",
                "content": f"""
# {program_name} Operating Procedures

## Document Information
- **Document ID**: {{{{ document_id }}}}
- **Version**: {{{{ version }}}}
- **Date**: {{{{ current_date }}}}
- **Author**: {{{{ author }}}}

## Purpose
This document defines the standard operating procedures for {program_name}.

## Daily Operations
### Morning Checklist
- [ ] Review security alerts
- [ ] Check system status
- [ ] Verify backup completion
- [ ] Review access logs

### During Operations
- [ ] Monitor security events
- [ ] Respond to alerts promptly
- [ ] Follow change procedures
- [ ] Document all activities

### End of Day
- [ ] Review daily activities
- [ ] Update documentation
- [ ] Prepare for next day
- [ ] Secure systems

## Incident Response
1. **Detection**: Monitor for security events
2. **Assessment**: Evaluate severity and impact
3. **Response**: Follow incident response procedures
4. **Recovery**: Restore normal operations
5. **Review**: Document lessons learned

## Contacts
- Security Team: security@company.com
- IT Support: support@company.com
- Emergency: +1-555-EMERGENCY
                """,
                "classification": "internal",
                "tags": [program_name.lower().replace(' ', '-'), "procedures", "operations"]
            }
        ]
        
        return templates
    
    def get_user_security_dashboard(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive security dashboard for a user"""
        try:
            # Get training status
            training_status = self.training_system.get_user_training_status(user_id)
            
            # Get awareness score
            awareness_score = self.awareness_system.get_user_awareness_score(user_id)
            
            # Get recent knowledge base activity
            kb_activity = self._get_user_kb_activity(user_id)
            
            # Get policy compliance status
            policy_compliance = self._get_user_policy_compliance(user_id)
            
            # Calculate overall security score
            overall_score = self._calculate_overall_security_score(
                training_status, awareness_score, kb_activity, policy_compliance
            )
            
            return {
                "user_id": user_id,
                "overall_security_score": overall_score,
                "training_status": training_status,
                "awareness_score": awareness_score,
                "knowledge_base_activity": kb_activity,
                "policy_compliance": policy_compliance,
                "recommendations": self._generate_user_recommendations(
                    training_status, awareness_score, kb_activity, policy_compliance
                ),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate user dashboard: {str(e)}")
            return {"error": str(e)}
    
    def _get_user_kb_activity(self, user_id: str) -> Dict[str, Any]:
        """Get user knowledge base activity"""
        # Count articles viewed, searches performed, feedback submitted
        user_searches = [q for q in self.knowledge_base.search_queries.values() if q.user_id == user_id]
        user_feedback = [f for f in self.knowledge_base.feedback.values() if f.user_id == user_id]
        
        return {
            "searches_performed": len(user_searches),
            "articles_viewed": sum(len(q.clicked_articles) for q in user_searches),
            "feedback_submitted": len(user_feedback),
            "average_rating_given": sum(f.rating for f in user_feedback) / len(user_feedback) if user_feedback else 0,
            "last_activity": max([q.timestamp for q in user_searches] + [f.created_date for f in user_feedback]).isoformat() if (user_searches or user_feedback) else None
        }
    
    def _get_user_policy_compliance(self, user_id: str) -> Dict[str, Any]:
        """Get user policy compliance status"""
        # This would integrate with actual compliance tracking
        active_policies = [p for p in self.policy_system.policies.values() if p.status.value == "active"]
        
        return {
            "total_policies": len(active_policies),
            "acknowledged_policies": len(active_policies),  # Simplified
            "compliance_rate": 100.0,  # Simplified
            "last_acknowledgment": datetime.now().isoformat()
        }
    
    def _calculate_overall_security_score(self, training_status: Dict, awareness_score: Dict, 
                                        kb_activity: Dict, policy_compliance: Dict) -> float:
        """Calculate overall security score for a user"""
        # Weight different components
        training_weight = 0.4
        awareness_weight = 0.3
        knowledge_weight = 0.2
        policy_weight = 0.1
        
        # Calculate component scores (0-100)
        training_score = 100 if training_status.get('mandatory_compliance', False) else 50
        awareness_component = awareness_score.get('awareness_score', 50)
        knowledge_component = min(kb_activity.get('searches_performed', 0) * 10, 100)
        policy_component = policy_compliance.get('compliance_rate', 0)
        
        overall_score = (
            training_score * training_weight +
            awareness_component * awareness_weight +
            knowledge_component * knowledge_weight +
            policy_component * policy_weight
        )
        
        return round(overall_score, 1)
    
    def _generate_user_recommendations(self, training_status: Dict, awareness_score: Dict,
                                     kb_activity: Dict, policy_compliance: Dict) -> List[str]:
        """Generate personalized security recommendations for a user"""
        recommendations = []
        
        # Training recommendations
        if not training_status.get('mandatory_compliance', False):
            missing_training = training_status.get('missing_mandatory_training', [])
            if missing_training:
                recommendations.append(f"Complete mandatory training: {', '.join(missing_training)}")
        
        # Awareness recommendations
        if awareness_score.get('awareness_score', 100) < 80:
            improvement_areas = awareness_score.get('improvement_areas', [])
            if improvement_areas:
                recommendations.append(f"Improve security awareness in: {', '.join(improvement_areas)}")
        
        # Knowledge base recommendations
        if kb_activity.get('searches_performed', 0) < 5:
            recommendations.append("Explore the security knowledge base to learn best practices")
        
        # Policy recommendations
        if policy_compliance.get('compliance_rate', 100) < 100:
            recommendations.append("Review and acknowledge all active security policies")
        
        if not recommendations:
            recommendations.append("Great job! Keep up the excellent security practices")
        
        return recommendations
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive report across all systems"""
        try:
            # Get individual system reports
            doc_report = self.documentation_manager.generate_documentation_report()
            training_report = self.training_system.get_training_analytics()
            awareness_report = self.awareness_system.generate_awareness_report()
            playbook_report = self.playbook_system.generate_incident_report()
            policy_report = self.policy_system.generate_policy_report()
            kb_report = self.knowledge_base.generate_knowledge_base_report()
            
            # Calculate integration metrics
            integration_score = self._calculate_integration_score()
            
            return {
                "report_type": "comprehensive_security_documentation_training",
                "generated_date": datetime.now().isoformat(),
                "integration_score": integration_score,
                "documentation": doc_report,
                "training": training_report,
                "awareness": awareness_report,
                "incident_response": playbook_report,
                "policies": policy_report,
                "knowledge_base": kb_report,
                "summary": {
                    "total_documents": doc_report.get("total_documents", 0),
                    "total_training_modules": len(self.training_system.modules),
                    "total_policies": policy_report.get("total_policies", 0),
                    "total_kb_articles": kb_report.get("total_articles", 0),
                    "total_playbooks": playbook_report.get("total_playbooks", 0),
                    "overall_compliance_rate": training_report.get("overall_compliance_rate", 0),
                    "security_awareness_score": awareness_report.get("overall_click_rate", 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate comprehensive report: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_integration_score(self) -> float:
        """Calculate how well integrated the systems are"""
        # This would measure cross-references, automated updates, etc.
        # Simplified calculation for now
        return 85.0
    
    def perform_automated_maintenance(self) -> Dict[str, Any]:
        """Perform automated maintenance across all systems"""
        try:
            maintenance_results = {}
            
            # Update documentation
            updated_docs = self.documentation_manager.auto_update_documents()
            maintenance_results['documentation_updates'] = len(updated_docs)
            
            # Check for policies needing review
            policies_for_review = self.policy_system.get_policies_for_review()
            maintenance_results['policies_needing_review'] = len(policies_for_review)
            
            # Update knowledge base index
            self.knowledge_base._build_search_index()
            maintenance_results['kb_index_updated'] = True
            
            # Check training compliance
            # This would identify users who need training reminders
            maintenance_results['training_reminders_sent'] = 0  # Placeholder
            
            # Update integration metrics
            self.integration_metrics['last_updated'] = datetime.now()
            self.integration_metrics['automated_updates_processed'] += len(updated_docs)
            
            logger.info("Automated maintenance completed successfully")
            return maintenance_results
            
        except Exception as e:
            logger.error(f"Automated maintenance failed: {str(e)}")
            return {"error": str(e)}