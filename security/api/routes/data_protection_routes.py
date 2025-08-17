"""
Data Protection API Routes

REST API endpoints for the Advanced Data Protection Engine.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from ...data_protection.data_protection_engine import (
    DataProtectionEngine, DataProtectionConfig, ProtectionPolicy
)
from ...data_protection.data_masking import UserContext, UserRole
from ...data_protection.data_classifier import DataSensitivityLevel, DataCategory

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/data-protection", tags=["data-protection"])
security = HTTPBearer()

# Global engine instance
protection_engine = DataProtectionEngine()

# Pydantic models
class ClassificationRequest(BaseModel):
    data: str
    context: Optional[Dict[str, Any]] = None

class ClassificationResponse(BaseModel):
    sensitivity_level: str
    categories: List[str]
    confidence_score: float
    detected_patterns: List[str]
    recommendations: List[str]

class ProtectionRequest(BaseModel):
    data: str
    field_name: str
    user_context: Optional[Dict[str, Any]] = None
    custom_config: Optional[Dict[str, Any]] = None

class ProtectionResponse(BaseModel):
    original_data: str
    encrypted_data: Optional[str]
    masked_data: Optional[str]
    key_id: Optional[str]
    protection_applied: List[str]
    classification: Dict[str, Any]

class BatchProtectionRequest(BaseModel):
    data_dict: Dict[str, str]
    user_context: Optional[Dict[str, Any]] = None

class UnprotectionRequest(BaseModel):
    protected_data: str
    field_name: str
    key_id: str
    user_context: Dict[str, Any]

class DeletionRequest(BaseModel):
    data_identifiers: List[str]
    encryption_key_ids: List[str]
    delay_hours: int = 0

class TrainingRequest(BaseModel):
    training_data: List[Dict[str, Any]]

class PolicyUpdateRequest(BaseModel):
    policy: str

class ValidationRequest(BaseModel):
    test_data: Dict[str, str]

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from token (simplified for demo)"""
    # In production, validate JWT token and extract user info
    return {
        'user_id': 'demo_user',
        'role': 'analyst',
        'department': 'Security',
        'clearance_level': 3
    }

def create_user_context(user_data: Dict[str, Any]) -> UserContext:
    """Create UserContext from user data"""
    return UserContext(
        user_id=user_data.get('user_id', 'unknown'),
        role=UserRole(user_data.get('role', 'viewer')),
        department=user_data.get('department', 'General'),
        clearance_level=user_data.get('clearance_level', 1),
        access_purpose=user_data.get('access_purpose', 'general'),
        session_risk_score=user_data.get('session_risk_score', 0.5)
    )

@router.post("/classify", response_model=ClassificationResponse)
async def classify_data(request: ClassificationRequest, 
                       current_user: dict = Depends(get_current_user)):
    """Classify data sensitivity and categories"""
    try:
        result = protection_engine.classifier.classify_data(request.data, request.context)
        
        return ClassificationResponse(
            sensitivity_level=result.sensitivity_level.value,
            categories=[cat.value for cat in result.categories],
            confidence_score=result.confidence_score,
            detected_patterns=result.detected_patterns,
            recommendations=result.recommendations
        )
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/protect", response_model=ProtectionResponse)
async def protect_data(request: ProtectionRequest,
                      current_user: dict = Depends(get_current_user)):
    """Apply comprehensive data protection"""
    try:
        user_context = None
        if request.user_context:
            user_context = create_user_context(request.user_context)
        
        result = protection_engine.protect_data(
            data=request.data,
            field_name=request.field_name,
            user_context=user_context,
            custom_config=request.custom_config
        )
        
        classification_dict = {}
        if result.classified_data:
            classification_dict = {
                'sensitivity_level': result.classified_data.sensitivity_level.value,
                'categories': [cat.value for cat in result.classified_data.categories],
                'confidence_score': result.classified_data.confidence_score,
                'recommendations': result.classified_data.recommendations
            }
        
        return ProtectionResponse(
            original_data=result.original_data,
            encrypted_data=result.encrypted_data,
            masked_data=result.masked_data,
            key_id=result.key_id,
            protection_applied=result.protection_applied,
            classification=classification_dict
        )
    except Exception as e:
        logger.error(f"Protection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/protect/batch")
async def batch_protect_data(request: BatchProtectionRequest,
                           current_user: dict = Depends(get_current_user)):
    """Batch protect multiple data fields"""
    try:
        user_context = None
        if request.user_context:
            user_context = create_user_context(request.user_context)
        
        results = protection_engine.batch_protect_data(
            data_dict=request.data_dict,
            user_context=user_context
        )
        
        # Convert results to serializable format
        serialized_results = {}
        for field_name, result in results.items():
            classification_dict = {}
            if result.classified_data:
                classification_dict = {
                    'sensitivity_level': result.classified_data.sensitivity_level.value,
                    'categories': [cat.value for cat in result.classified_data.categories],
                    'confidence_score': result.classified_data.confidence_score
                }
            
            serialized_results[field_name] = {
                'encrypted_data': result.encrypted_data,
                'masked_data': result.masked_data,
                'key_id': result.key_id,
                'protection_applied': result.protection_applied,
                'classification': classification_dict
            }
        
        return serialized_results
    except Exception as e:
        logger.error(f"Batch protection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/unprotect")
async def unprotect_data(request: UnprotectionRequest,
                        current_user: dict = Depends(get_current_user)):
    """Decrypt and unmask protected data"""
    try:
        user_context = create_user_context(request.user_context)
        
        result = protection_engine.unprotect_data(
            protected_data=request.protected_data,
            field_name=request.field_name,
            key_id=request.key_id,
            user_context=user_context
        )
        
        return {"unprotected_data": result}
    except Exception as e:
        logger.error(f"Unprotection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/deletion/schedule")
async def schedule_deletion(request: DeletionRequest,
                          background_tasks: BackgroundTasks,
                          current_user: dict = Depends(get_current_user)):
    """Schedule secure data deletion"""
    try:
        request_id = protection_engine.schedule_secure_deletion(
            data_identifiers=request.data_identifiers,
            encryption_key_ids=request.encryption_key_ids,
            delay_hours=request.delay_hours
        )
        
        return {"deletion_request_id": request_id}
    except Exception as e:
        logger.error(f"Deletion scheduling error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/deletion/status/{request_id}")
async def get_deletion_status(request_id: str,
                            current_user: dict = Depends(get_current_user)):
    """Get deletion request status"""
    try:
        status = protection_engine.deletion_engine.get_deletion_status(request_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Deletion request not found")
        
        return {
            "request_id": status.request_id,
            "status": status.status.value,
            "requested_at": status.requested_at.isoformat(),
            "scheduled_for": status.scheduled_for.isoformat() if status.scheduled_for else None,
            "data_count": len(status.data_identifiers)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deletion status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/classifier/train")
async def train_classifier(request: TrainingRequest,
                          current_user: dict = Depends(get_current_user)):
    """Train the ML classifier"""
    try:
        # Convert training data format
        training_data = []
        for item in request.training_data:
            text = item.get('text', '')
            sensitivity = DataSensitivityLevel(item.get('sensitivity_level', 'public'))
            training_data.append((text, sensitivity))
        
        accuracy = protection_engine.train_classifier(training_data)
        
        return {
            "training_completed": True,
            "accuracy": accuracy,
            "training_samples": len(training_data)
        }
    except Exception as e:
        logger.error(f"Classifier training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/policy")
async def update_policy(request: PolicyUpdateRequest,
                       current_user: dict = Depends(get_current_user)):
    """Update data protection policy"""
    try:
        policy = ProtectionPolicy(request.policy)
        protection_engine.update_protection_policy(policy)
        
        return {"policy_updated": True, "new_policy": policy.value}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid policy: {e}")
    except Exception as e:
        logger.error(f"Policy update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics")
async def get_statistics(current_user: dict = Depends(get_current_user)):
    """Get comprehensive protection statistics"""
    try:
        stats = protection_engine.get_protection_statistics()
        return stats
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/audit-trail")
async def get_audit_trail(start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         current_user: dict = Depends(get_current_user)):
    """Export audit trail for compliance"""
    try:
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None
        
        audit_trail = protection_engine.export_audit_trail(start_dt, end_dt)
        
        return {
            "audit_entries": audit_trail,
            "total_entries": len(audit_trail),
            "date_range": {
                "start": start_date,
                "end": end_date
            }
        }
    except Exception as e:
        logger.error(f"Audit trail error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate")
async def validate_protection(request: ValidationRequest,
                            current_user: dict = Depends(get_current_user)):
    """Validate data protection implementation"""
    try:
        validation_results = protection_engine.validate_data_protection(request.test_data)
        return validation_results
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/keys/cleanup")
async def cleanup_expired_keys(current_user: dict = Depends(get_current_user)):
    """Clean up expired encryption keys"""
    try:
        cleaned_count = protection_engine.cleanup_expired_keys()
        return {"keys_cleaned": cleaned_count}
    except Exception as e:
        logger.error(f"Key cleanup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/masking/preview/{field_name}")
async def preview_masking(field_name: str, data: str,
                         current_user: dict = Depends(get_current_user)):
    """Preview masking for different user roles"""
    try:
        preview = protection_engine.masking_engine.get_masking_preview(data, field_name)
        
        # Convert enum keys to strings
        preview_dict = {role.value: masked_data for role, masked_data in preview.items()}
        
        return {
            "field_name": field_name,
            "original_data": data,
            "masking_preview": preview_dict
        }
    except Exception as e:
        logger.error(f"Masking preview error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        stats = protection_engine.get_protection_statistics()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "classifier": "trained" if stats.get('classification', {}).get('model_trained') else "not_trained",
                "key_manager": "active",
                "encryption_engine": "active",
                "masking_engine": "active",
                "deletion_engine": "active"
            }
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }