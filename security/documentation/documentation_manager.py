"""
Security Documentation Management System
Provides automated documentation generation, version control, and maintenance
"""

import os
import json
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import markdown
import jinja2
from git import Repo
import logging

logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Metadata for security documents"""
    title: str
    version: str
    author: str
    created_date: datetime
    last_updated: datetime
    review_date: datetime
    status: str  # draft, review, approved, archived
    classification: str  # public, internal, confidential, restricted
    tags: List[str]
    dependencies: List[str]
    approval_required: bool
    auto_update: bool

@dataclass
class DocumentTemplate:
    """Template for generating security documents"""
    name: str
    template_path: str
    output_format: str
    variables: Dict[str, Any]
    auto_generate: bool
    update_frequency: str  # daily, weekly, monthly, on-change

class SecurityDocumentationManager:
    """Manages security documentation with automated updates and version control"""
    
    def __init__(self, docs_path: str = "security/docs", templates_path: str = "security/templates"):
        self.docs_path = Path(docs_path)
        self.templates_path = Path(templates_path)
        self.docs_path.mkdir(parents=True, exist_ok=True)
        self.templates_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize Jinja2 environment
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.templates_path)),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Document registry
        self.documents: Dict[str, DocumentMetadata] = {}
        self.templates: Dict[str, DocumentTemplate] = {}
        
        self._load_document_registry()
        self._load_templates()
    
    def _load_document_registry(self):
        """Load document registry from file"""
        registry_path = self.docs_path / "registry.json"
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                data = json.load(f)
                for doc_id, doc_data in data.items():
                    # Convert datetime strings back to datetime objects
                    for date_field in ['created_date', 'last_updated', 'review_date']:
                        if doc_data.get(date_field):
                            doc_data[date_field] = datetime.fromisoformat(doc_data[date_field])
                    self.documents[doc_id] = DocumentMetadata(**doc_data)
    
    def _save_document_registry(self):
        """Save document registry to file"""
        registry_path = self.docs_path / "registry.json"
        data = {}
        for doc_id, metadata in self.documents.items():
            doc_data = asdict(metadata)
            # Convert datetime objects to strings
            for date_field in ['created_date', 'last_updated', 'review_date']:
                if doc_data.get(date_field):
                    doc_data[date_field] = doc_data[date_field].isoformat()
            data[doc_id] = doc_data
        
        with open(registry_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_templates(self):
        """Load document templates"""
        templates_config = self.templates_path / "templates.yaml"
        if templates_config.exists():
            with open(templates_config, 'r') as f:
                data = yaml.safe_load(f)
                for template_name, template_data in data.items():
                    self.templates[template_name] = DocumentTemplate(
                        name=template_name,
                        **template_data
                    )
    
    def create_document(self, doc_id: str, title: str, content: str, 
                       classification: str = "internal", tags: List[str] = None,
                       template_name: str = None) -> bool:
        """Create a new security document"""
        try:
            # Create document metadata
            metadata = DocumentMetadata(
                title=title,
                version="1.0",
                author="Security Team",
                created_date=datetime.now(),
                last_updated=datetime.now(),
                review_date=datetime.now() + timedelta(days=90),
                status="draft",
                classification=classification,
                tags=tags or [],
                dependencies=[],
                approval_required=classification in ["confidential", "restricted"],
                auto_update=False
            )
            
            # Generate content from template if specified
            if template_name and template_name in self.templates:
                template = self.templates[template_name]
                jinja_template = self.jinja_env.get_template(template.template_path)
                content = jinja_template.render(
                    title=title,
                    metadata=metadata,
                    **template.variables
                )
            
            # Save document
            doc_path = self.docs_path / f"{doc_id}.md"
            with open(doc_path, 'w') as f:
                f.write(content)
            
            # Update registry
            self.documents[doc_id] = metadata
            self._save_document_registry()
            
            logger.info(f"Created security document: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create document {doc_id}: {str(e)}")
            return False
    
    def update_document(self, doc_id: str, content: str, version_increment: str = "minor") -> bool:
        """Update an existing document"""
        try:
            if doc_id not in self.documents:
                raise ValueError(f"Document {doc_id} not found")
            
            metadata = self.documents[doc_id]
            
            # Update version
            version_parts = metadata.version.split('.')
            if version_increment == "major":
                version_parts[0] = str(int(version_parts[0]) + 1)
                version_parts[1] = "0"
            elif version_increment == "minor":
                version_parts[1] = str(int(version_parts[1]) + 1)
            
            metadata.version = '.'.join(version_parts)
            metadata.last_updated = datetime.now()
            metadata.status = "draft"  # Reset to draft for review
            
            # Save updated content
            doc_path = self.docs_path / f"{doc_id}.md"
            with open(doc_path, 'w') as f:
                f.write(content)
            
            # Update registry
            self.documents[doc_id] = metadata
            self._save_document_registry()
            
            logger.info(f"Updated security document: {doc_id} to version {metadata.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {str(e)}")
            return False
    
    def auto_update_documents(self) -> List[str]:
        """Automatically update documents that have auto-update enabled"""
        updated_docs = []
        
        for doc_id, metadata in self.documents.items():
            if not metadata.auto_update:
                continue
            
            try:
                # Check if update is needed based on frequency
                if self._should_update_document(metadata):
                    # Generate updated content from template
                    if doc_id in self.templates:
                        template = self.templates[doc_id]
                        jinja_template = self.jinja_env.get_template(template.template_path)
                        
                        # Get current system data for template variables
                        template_vars = self._get_template_variables(doc_id)
                        content = jinja_template.render(**template_vars)
                        
                        if self.update_document(doc_id, content, "minor"):
                            updated_docs.append(doc_id)
                            
            except Exception as e:
                logger.error(f"Failed to auto-update document {doc_id}: {str(e)}")
        
        return updated_docs
    
    def _should_update_document(self, metadata: DocumentMetadata) -> bool:
        """Check if document should be updated based on frequency"""
        now = datetime.now()
        last_updated = metadata.last_updated
        
        # This would be expanded based on actual update frequency logic
        return (now - last_updated).days >= 7  # Weekly updates for now
    
    def _get_template_variables(self, doc_id: str) -> Dict[str, Any]:
        """Get current system data for template variables"""
        # This would integrate with actual system monitoring and security data
        return {
            "current_date": datetime.now().strftime("%Y-%m-%d"),
            "security_metrics": self._get_security_metrics(),
            "compliance_status": self._get_compliance_status(),
            "incident_summary": self._get_recent_incidents()
        }
    
    def _get_security_metrics(self) -> Dict[str, Any]:
        """Get current security metrics"""
        # Placeholder - would integrate with actual security monitoring
        return {
            "threats_detected": 42,
            "vulnerabilities_patched": 15,
            "compliance_score": 95.2
        }
    
    def _get_compliance_status(self) -> Dict[str, str]:
        """Get current compliance status"""
        return {
            "SOC2": "Compliant",
            "GDPR": "Compliant", 
            "HIPAA": "In Progress",
            "ISO27001": "Compliant"
        }
    
    def _get_recent_incidents(self) -> List[Dict[str, Any]]:
        """Get recent security incidents"""
        return [
            {
                "id": "INC-2024-001",
                "severity": "Medium",
                "status": "Resolved",
                "date": "2024-01-15"
            }
        ]
    
    def approve_document(self, doc_id: str, approver: str) -> bool:
        """Approve a document for publication"""
        try:
            if doc_id not in self.documents:
                raise ValueError(f"Document {doc_id} not found")
            
            metadata = self.documents[doc_id]
            metadata.status = "approved"
            metadata.last_updated = datetime.now()
            
            # Log approval
            logger.info(f"Document {doc_id} approved by {approver}")
            
            self._save_document_registry()
            return True
            
        except Exception as e:
            logger.error(f"Failed to approve document {doc_id}: {str(e)}")
            return False
    
    def get_documents_for_review(self) -> List[str]:
        """Get documents that need review"""
        review_needed = []
        now = datetime.now()
        
        for doc_id, metadata in self.documents.items():
            if metadata.review_date <= now or metadata.status == "draft":
                review_needed.append(doc_id)
        
        return review_needed
    
    def generate_documentation_report(self) -> Dict[str, Any]:
        """Generate comprehensive documentation status report"""
        total_docs = len(self.documents)
        approved_docs = sum(1 for m in self.documents.values() if m.status == "approved")
        draft_docs = sum(1 for m in self.documents.values() if m.status == "draft")
        review_needed = len(self.get_documents_for_review())
        
        return {
            "total_documents": total_docs,
            "approved_documents": approved_docs,
            "draft_documents": draft_docs,
            "documents_needing_review": review_needed,
            "compliance_coverage": self._calculate_compliance_coverage(),
            "last_updated": datetime.now().isoformat()
        }
    
    def _calculate_compliance_coverage(self) -> Dict[str, float]:
        """Calculate documentation coverage for compliance frameworks"""
        # Placeholder - would analyze actual compliance requirements
        return {
            "SOC2": 95.0,
            "GDPR": 88.0,
            "HIPAA": 92.0,
            "ISO27001": 97.0
        }