"""
API Routes for Documentation and Training System
Provides REST API endpoints for all documentation and training functionality
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from ...documentation.documentation_training_system import IntegratedDocumentationTrainingSystem

logger = logging.getLogger(__name__)

# Initialize the integrated system
doc_training_system = IntegratedDocumentationTrainingSystem()

router = APIRouter(prefix="/api/v1/security/documentation-training", tags=["Security Documentation & Training"])

# Documentation Management Routes
@router.post("/documents", response_model=Dict[str, str])
async def create_document(
    title: str,
    content: str,
    classification: str = "internal",
    tags: List[str] = None
):
    """Create a new security document"""
    try:
        doc_id = doc_training_system.documentation_manager.create_document(
            doc_id=None,  # Auto-generate
            title=title,
            content=content,
            classification=classification,
            tags=tags or []
        )
        return {"document_id": doc_id, "status": "created"}
    except Exception as e:
        logger.error(f"Failed to create document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/documents/{doc_id}")
async def update_document(
    doc_id: str,
    content: str,
    version_increment: str = "minor"
):
    """Update an existing document"""
    try:
        success = doc_training_system.documentation_manager.update_document(
            doc_id=doc_id,
            content=content,
            version_increment=version_increment
        )
        if success:
            return {"status": "updated"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(f"Failed to update document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents/review")
async def get_documents_for_review():
    """Get documents that need review"""
    try:
        docs_for_review = doc_training_system.documentation_manager.get_documents_for_review()
        return {"documents_for_review": docs_for_review}
    except Exception as e:
        logger.error(f"Failed to get documents for review: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/documents/{doc_id}/approve")
async def approve_document(doc_id: str, approver: str):
    """Approve a document for publication"""
    try:
        success = doc_training_system.documentation_manager.approve_document(doc_id, approver)
        if success:
            return {"status": "approved"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(f"Failed to approve document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Training System Routes
@router.post("/training/modules")
async def create_training_module(module_data: Dict[str, Any]):
    """Create a new training module"""
    try:
        module_id = doc_training_system.training_system.create_training_module(module_data)
        return {"module_id": module_id, "status": "created"}
    except Exception as e:
        logger.error(f"Failed to create training module: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/training/start")
async def start_training(user_id: str, module_id: str):
    """Start training for a user"""
    try:
        success = doc_training_system.training_system.start_training(user_id, module_id)
        if success:
            return {"status": "training_started"}
        else:
            return {"status": "training_already_completed_recently"}
    except Exception as e:
        logger.error(f"Failed to start training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/training/complete-assessment")
async def complete_assessment(user_id: str, module_id: str, answers: List[int]):
    """Complete training assessment"""
    try:
        result = doc_training_system.training_system.complete_assessment(user_id, module_id, answers)
        return result
    except Exception as e:
        logger.error(f"Failed to complete assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/training/status/{user_id}")
async def get_training_status(user_id: str):
    """Get training status for a user"""
    try:
        status = doc_training_system.training_system.get_user_training_status(user_id)
        return status
    except Exception as e:
        logger.error(f"Failed to get training status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/training/analytics")
async def get_training_analytics():
    """Get training analytics and reporting"""
    try:
        analytics = doc_training_system.training_system.get_training_analytics()
        return analytics
    except Exception as e:
        logger.error(f"Failed to get training analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Security Awareness Routes
@router.post("/awareness/campaigns")
async def create_phishing_campaign(campaign_data: Dict[str, Any]):
    """Create a new phishing simulation campaign"""
    try:
        campaign_id = doc_training_system.awareness_system.create_phishing_campaign(campaign_data)
        return {"campaign_id": campaign_id, "status": "created"}
    except Exception as e:
        logger.error(f"Failed to create phishing campaign: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/awareness/campaigns/{campaign_id}/launch")
async def launch_phishing_campaign(campaign_id: str, target_users: List[Dict[str, str]]):
    """Launch a phishing simulation campaign"""
    try:
        success = doc_training_system.awareness_system.launch_phishing_campaign(campaign_id, target_users)
        if success:
            return {"status": "campaign_launched"}
        else:
            raise HTTPException(status_code=500, detail="Failed to launch campaign")
    except Exception as e:
        logger.error(f"Failed to launch phishing campaign: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/awareness/campaigns/{campaign_id}/results")
async def get_campaign_results(campaign_id: str):
    """Get results for a phishing campaign"""
    try:
        results = doc_training_system.awareness_system.get_campaign_results(campaign_id)
        return results
    except Exception as e:
        logger.error(f"Failed to get campaign results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/awareness/user-score/{user_id}")
async def get_user_awareness_score(user_id: str):
    """Get security awareness score for a user"""
    try:
        score = doc_training_system.awareness_system.get_user_awareness_score(user_id)
        return score
    except Exception as e:
        logger.error(f"Failed to get user awareness score: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/awareness/report")
async def get_awareness_report():
    """Generate comprehensive security awareness report"""
    try:
        report = doc_training_system.awareness_system.generate_awareness_report()
        return report
    except Exception as e:
        logger.error(f"Failed to generate awareness report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Incident Response Playbook Routes
@router.post("/incidents")
async def create_incident(incident_data: Dict[str, Any]):
    """Create a new security incident"""
    try:
        incident_id = doc_training_system.playbook_system.create_incident(incident_data)
        return {"incident_id": incident_id, "status": "created"}
    except Exception as e:
        logger.error(f"Failed to create incident: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/incidents/{incident_id}/execute-playbook")
async def execute_playbook(incident_id: str, executed_by: str):
    """Execute incident response playbook"""
    try:
        execution_id = doc_training_system.playbook_system.execute_playbook(incident_id, executed_by)
        return {"execution_id": execution_id, "status": "playbook_started"}
    except Exception as e:
        logger.error(f"Failed to execute playbook: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/incidents/{incident_id}/status")
async def get_incident_status(incident_id: str):
    """Get incident status"""
    try:
        status = doc_training_system.playbook_system.get_incident_status(incident_id)
        return status
    except Exception as e:
        logger.error(f"Failed to get incident status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/incidents/report")
async def get_incident_report():
    """Generate incident response report"""
    try:
        report = doc_training_system.playbook_system.generate_incident_report()
        return report
    except Exception as e:
        logger.error(f"Failed to generate incident report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Policy Management Routes
@router.post("/policies")
async def create_policy(policy_data: Dict[str, Any]):
    """Create a new security policy"""
    try:
        policy_id = doc_training_system.policy_system.create_policy(policy_data)
        return {"policy_id": policy_id, "status": "created"}
    except Exception as e:
        logger.error(f"Failed to create policy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/policies/{policy_id}")
async def update_policy(
    policy_id: str,
    content: str,
    changes_summary: str,
    author: str,
    change_type: str = "minor"
):
    """Update an existing policy"""
    try:
        version_id = doc_training_system.policy_system.update_policy(
            policy_id, content, changes_summary, author, change_type
        )
        return {"version_id": version_id, "status": "updated"}
    except Exception as e:
        logger.error(f"Failed to update policy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/policies/{policy_id}/submit-approval")
async def submit_policy_for_approval(
    policy_id: str,
    requested_by: str,
    approval_steps: List[Dict[str, Any]]
):
    """Submit policy for approval workflow"""
    try:
        workflow_id = doc_training_system.policy_system.submit_for_approval(
            policy_id, requested_by, approval_steps
        )
        return {"workflow_id": workflow_id, "status": "submitted_for_approval"}
    except Exception as e:
        logger.error(f"Failed to submit policy for approval: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/policies/workflows/{workflow_id}/approve")
async def approve_policy_step(
    workflow_id: str,
    approver: str,
    approved: bool,
    comments: str = ""
):
    """Approve or reject a policy approval step"""
    try:
        success = doc_training_system.policy_system.approve_policy_step(
            workflow_id, approver, approved, comments
        )
        return {"status": "approved" if success else "rejected"}
    except Exception as e:
        logger.error(f"Failed to approve policy step: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/policies/review")
async def get_policies_for_review():
    """Get policies that need review"""
    try:
        policies = doc_training_system.policy_system.get_policies_for_review()
        return {"policies_for_review": policies}
    except Exception as e:
        logger.error(f"Failed to get policies for review: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/policies/{policy_id}/compliance")
async def get_policy_compliance_status(policy_id: str):
    """Get compliance status for a policy"""
    try:
        status = doc_training_system.policy_system.get_policy_compliance_status(policy_id)
        return status
    except Exception as e:
        logger.error(f"Failed to get policy compliance status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/policies/report")
async def get_policy_report():
    """Generate policy management report"""
    try:
        report = doc_training_system.policy_system.generate_policy_report()
        return report
    except Exception as e:
        logger.error(f"Failed to generate policy report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Knowledge Base Routes
@router.post("/knowledge-base/articles")
async def create_kb_article(article_data: Dict[str, Any]):
    """Create a new knowledge base article"""
    try:
        article_id = doc_training_system.knowledge_base.create_article(article_data)
        return {"article_id": article_id, "status": "created"}
    except Exception as e:
        logger.error(f"Failed to create knowledge base article: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/knowledge-base/articles/{article_id}")
async def update_kb_article(article_id: str, updates: Dict[str, Any]):
    """Update an existing knowledge base article"""
    try:
        success = doc_training_system.knowledge_base.update_article(article_id, updates)
        if success:
            return {"status": "updated"}
        else:
            raise HTTPException(status_code=404, detail="Article not found")
    except Exception as e:
        logger.error(f"Failed to update knowledge base article: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/knowledge-base/search")
async def search_knowledge_base(
    query: str = Query(..., description="Search query"),
    user_id: Optional[str] = Query(None, description="User ID for tracking"),
    access_level: str = Query("internal", description="User access level")
):
    """Search knowledge base articles"""
    try:
        from ...knowledge_base.knowledge_base_system import AccessLevel
        access_enum = AccessLevel(access_level)
        
        results = doc_training_system.knowledge_base.search_articles(query, user_id, access_enum)
        return {"results": results}
    except Exception as e:
        logger.error(f"Failed to search knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/knowledge-base/articles/{article_id}")
async def get_kb_article(article_id: str, user_id: Optional[str] = Query(None)):
    """Get a specific knowledge base article"""
    try:
        article = doc_training_system.knowledge_base.get_article(article_id, user_id)
        if article:
            return article
        else:
            raise HTTPException(status_code=404, detail="Article not found")
    except Exception as e:
        logger.error(f"Failed to get knowledge base article: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/knowledge-base/articles/{article_id}/feedback")
async def submit_kb_feedback(
    article_id: str,
    user_id: str,
    rating: int,
    feedback_text: str = "",
    helpful: bool = True
):
    """Submit feedback for a knowledge base article"""
    try:
        feedback_id = doc_training_system.knowledge_base.submit_feedback(
            article_id, user_id, rating, feedback_text, helpful
        )
        return {"feedback_id": feedback_id, "status": "submitted"}
    except Exception as e:
        logger.error(f"Failed to submit knowledge base feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/knowledge-base/popular")
async def get_popular_articles(
    limit: int = Query(10, description="Number of articles to return"),
    access_level: str = Query("internal", description="User access level")
):
    """Get most popular knowledge base articles"""
    try:
        from ...knowledge_base.knowledge_base_system import AccessLevel
        access_enum = AccessLevel(access_level)
        
        articles = doc_training_system.knowledge_base.get_popular_articles(limit, access_enum)
        return {"articles": articles}
    except Exception as e:
        logger.error(f"Failed to get popular articles: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/knowledge-base/categories/{category}")
async def get_articles_by_category(
    category: str,
    access_level: str = Query("internal", description="User access level")
):
    """Get articles by category"""
    try:
        from ...knowledge_base.knowledge_base_system import AccessLevel
        access_enum = AccessLevel(access_level)
        
        articles = doc_training_system.knowledge_base.get_articles_by_category(category, access_enum)
        return {"articles": articles}
    except Exception as e:
        logger.error(f"Failed to get articles by category: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/knowledge-base/report")
async def get_kb_report():
    """Generate knowledge base analytics report"""
    try:
        report = doc_training_system.knowledge_base.generate_knowledge_base_report()
        return report
    except Exception as e:
        logger.error(f"Failed to generate knowledge base report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Integrated System Routes
@router.post("/programs")
async def create_security_program(
    program_name: str,
    requirements: List[str]
):
    """Create a comprehensive security program with all components"""
    try:
        program_ids = doc_training_system.create_comprehensive_security_program(
            program_name, requirements
        )
        return {"program_ids": program_ids, "status": "created"}
    except Exception as e:
        logger.error(f"Failed to create security program: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard/{user_id}")
async def get_user_security_dashboard(user_id: str):
    """Get comprehensive security dashboard for a user"""
    try:
        dashboard = doc_training_system.get_user_security_dashboard(user_id)
        return dashboard
    except Exception as e:
        logger.error(f"Failed to get user security dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/report/comprehensive")
async def get_comprehensive_report():
    """Generate comprehensive report across all systems"""
    try:
        report = doc_training_system.generate_comprehensive_report()
        return report
    except Exception as e:
        logger.error(f"Failed to generate comprehensive report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/maintenance")
async def perform_automated_maintenance():
    """Perform automated maintenance across all systems"""
    try:
        results = doc_training_system.perform_automated_maintenance()
        return {"maintenance_results": results, "status": "completed"}
    except Exception as e:
        logger.error(f"Failed to perform automated maintenance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))