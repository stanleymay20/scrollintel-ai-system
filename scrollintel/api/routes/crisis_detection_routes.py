"""
API routes for Crisis Detection and Assessment Engine
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from scrollintel.engines.crisis_detection_engine import (
    CrisisDetectionEngine,
    Crisis,
    CrisisType,
    SeverityLevel,
    CrisisStatus,
    Signal
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/crisis-detection", tags=["crisis-detection"])

# Global crisis detection engine instance
crisis_engine = CrisisDetectionEngine()


@router.post("/detect", response_model=Dict[str, Any])
async def detect_crises(background_tasks: BackgroundTasks):
    """
    Trigger crisis detection and assessment process
    """
    try:
        # Run detection in background
        background_tasks.add_task(run_crisis_detection)
        
        return {
            "status": "success",
            "message": "Crisis detection process initiated",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error initiating crisis detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_crisis_detection():
    """Background task for crisis detection"""
    try:
        crises = await crisis_engine.detect_and_assess_crises()
        logger.info(f"Detected {len(crises)} new crises")
    except Exception as e:
        logger.error(f"Error in background crisis detection: {str(e)}")


@router.get("/active", response_model=List[Dict[str, Any]])
async def get_active_crises():
    """
    Get all currently active crises
    """
    try:
        crises = await crisis_engine.get_active_crises()
        
        return [
            {
                "id": crisis.id,
                "crisis_type": crisis.crisis_type.value,
                "severity_level": crisis.severity_level.name,
                "status": crisis.current_status.value,
                "start_time": crisis.start_time.isoformat(),
                "affected_areas": crisis.affected_areas,
                "stakeholders_impacted": crisis.stakeholders_impacted,
                "classification": {
                    "confidence": crisis.classification.confidence if crisis.classification else 0,
                    "sub_categories": crisis.classification.sub_categories if crisis.classification else [],
                    "rationale": crisis.classification.classification_rationale if crisis.classification else ""
                } if crisis.classification else None,
                "impact_assessment": {
                    "financial_impact": crisis.impact_assessment.financial_impact if crisis.impact_assessment else {},
                    "operational_impact": crisis.impact_assessment.operational_impact if crisis.impact_assessment else {},
                    "mitigation_urgency": crisis.impact_assessment.mitigation_urgency.name if crisis.impact_assessment else "MEDIUM"
                } if crisis.impact_assessment else None
            }
            for crisis in crises
        ]
    except Exception as e:
        logger.error(f"Error getting active crises: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/crisis/{crisis_id}", response_model=Dict[str, Any])
async def get_crisis_details(crisis_id: str):
    """
    Get detailed information about a specific crisis
    """
    try:
        crisis = await crisis_engine.get_crisis_by_id(crisis_id)
        
        if not crisis:
            raise HTTPException(status_code=404, detail="Crisis not found")
        
        return {
            "id": crisis.id,
            "crisis_type": crisis.crisis_type.value,
            "severity_level": crisis.severity_level.name,
            "status": crisis.current_status.value,
            "start_time": crisis.start_time.isoformat(),
            "resolution_time": crisis.resolution_time.isoformat() if crisis.resolution_time else None,
            "affected_areas": crisis.affected_areas,
            "stakeholders_impacted": crisis.stakeholders_impacted,
            "signals": [
                {
                    "source": signal.source,
                    "signal_type": signal.signal_type,
                    "value": signal.value,
                    "timestamp": signal.timestamp.isoformat(),
                    "confidence": signal.confidence,
                    "metadata": signal.metadata
                }
                for signal in crisis.signals
            ],
            "classification": {
                "crisis_type": crisis.classification.crisis_type.value,
                "severity_level": crisis.classification.severity_level.name,
                "confidence": crisis.classification.confidence,
                "sub_categories": crisis.classification.sub_categories,
                "related_crises": crisis.classification.related_crises,
                "rationale": crisis.classification.classification_rationale
            } if crisis.classification else None,
            "impact_assessment": {
                "financial_impact": crisis.impact_assessment.financial_impact,
                "operational_impact": crisis.impact_assessment.operational_impact,
                "reputation_impact": crisis.impact_assessment.reputation_impact,
                "stakeholder_impact": crisis.impact_assessment.stakeholder_impact,
                "timeline_impact": {
                    k: v.total_seconds() for k, v in crisis.impact_assessment.timeline_impact.items()
                },
                "recovery_estimate_seconds": crisis.impact_assessment.recovery_estimate.total_seconds(),
                "cascading_risks": crisis.impact_assessment.cascading_risks,
                "mitigation_urgency": crisis.impact_assessment.mitigation_urgency.name
            } if crisis.impact_assessment else None,
            "escalation_history": crisis.escalation_history
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting crisis details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/crisis/{crisis_id}/resolve", response_model=Dict[str, Any])
async def resolve_crisis(crisis_id: str):
    """
    Mark a crisis as resolved
    """
    try:
        success = await crisis_engine.resolve_crisis(crisis_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Crisis not found")
        
        return {
            "status": "success",
            "message": f"Crisis {crisis_id} marked as resolved",
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving crisis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/signals/submit", response_model=Dict[str, Any])
async def submit_signal(signal_data: Dict[str, Any]):
    """
    Submit a manual signal for crisis detection
    """
    try:
        # Validate signal data
        required_fields = ['source', 'signal_type', 'value', 'confidence']
        for field in required_fields:
            if field not in signal_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Create signal
        signal = Signal(
            source=signal_data['source'],
            signal_type=signal_data['signal_type'],
            value=signal_data['value'],
            timestamp=datetime.now(),
            confidence=float(signal_data['confidence']),
            metadata=signal_data.get('metadata', {})
        )
        
        # Process signal through early warning system
        potential_crises = await crisis_engine.early_warning_system.detect_potential_crises([signal])
        
        # Convert high-probability potential crises to actual crises
        new_crises = []
        for potential_crisis in potential_crises:
            if potential_crisis.probability > 0.7:
                crisis = crisis_engine._create_crisis_from_potential(potential_crisis)
                processed_crisis = await crisis_engine._process_crisis(crisis)
                crisis_engine.active_crises[crisis.id] = processed_crisis
                new_crises.append(processed_crisis)
        
        return {
            "status": "success",
            "message": "Signal processed successfully",
            "signal_id": f"signal_{datetime.now().timestamp()}",
            "potential_crises_detected": len(potential_crises),
            "new_crises_created": len(new_crises),
            "new_crisis_ids": [crisis.id for crisis in new_crises]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting signal: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=Dict[str, Any])
async def get_crisis_metrics():
    """
    Get crisis detection and management metrics
    """
    try:
        metrics = await crisis_engine.get_crisis_metrics()
        
        return {
            "status": "success",
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting crisis metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/types", response_model=List[Dict[str, str]])
async def get_crisis_types():
    """
    Get available crisis types
    """
    try:
        return [
            {
                "value": crisis_type.value,
                "name": crisis_type.name,
                "description": _get_crisis_type_description(crisis_type)
            }
            for crisis_type in CrisisType
        ]
    except Exception as e:
        logger.error(f"Error getting crisis types: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/severity-levels", response_model=List[Dict[str, Any]])
async def get_severity_levels():
    """
    Get available severity levels
    """
    try:
        return [
            {
                "value": level.value,
                "name": level.name,
                "description": _get_severity_description(level)
            }
            for level in SeverityLevel
        ]
    except Exception as e:
        logger.error(f"Error getting severity levels: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test/simulate-crisis", response_model=Dict[str, Any])
async def simulate_crisis(crisis_data: Dict[str, Any]):
    """
    Simulate a crisis for testing purposes
    """
    try:
        # Validate crisis data
        if 'crisis_type' not in crisis_data:
            raise HTTPException(status_code=400, detail="Missing crisis_type")
        
        # Create test signals
        test_signals = []
        crisis_type_value = crisis_data['crisis_type']
        
        if crisis_type_value == 'system_outage':
            test_signals = [
                Signal(
                    source="system_monitor",
                    signal_type="high_cpu_usage",
                    value=95.0,
                    timestamp=datetime.now(),
                    confidence=0.9,
                    metadata={"test": True}
                ),
                Signal(
                    source="application_monitor",
                    signal_type="high_error_rate",
                    value=15.0,
                    timestamp=datetime.now(),
                    confidence=0.85,
                    metadata={"test": True}
                )
            ]
        elif crisis_type_value == 'security_breach':
            test_signals = [
                Signal(
                    source="security_monitor",
                    signal_type="high_failed_logins",
                    value=500,
                    timestamp=datetime.now(),
                    confidence=0.9,
                    metadata={"test": True}
                )
            ]
        
        # Process signals
        potential_crises = await crisis_engine.early_warning_system.detect_potential_crises(test_signals)
        
        # Create and process crisis
        new_crises = []
        for potential_crisis in potential_crises:
            crisis = crisis_engine._create_crisis_from_potential(potential_crisis)
            processed_crisis = await crisis_engine._process_crisis(crisis)
            crisis_engine.active_crises[crisis.id] = processed_crisis
            new_crises.append(processed_crisis)
        
        return {
            "status": "success",
            "message": "Crisis simulation completed",
            "simulated_crises": len(new_crises),
            "crisis_ids": [crisis.id for crisis in new_crises],
            "crisis_details": [
                {
                    "id": crisis.id,
                    "type": crisis.crisis_type.value,
                    "severity": crisis.severity_level.name,
                    "status": crisis.current_status.value
                }
                for crisis in new_crises
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error simulating crisis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_crisis_type_description(crisis_type: CrisisType) -> str:
    """Get human-readable description of crisis type"""
    descriptions = {
        CrisisType.SYSTEM_OUTAGE: "Critical system failures affecting service availability",
        CrisisType.SECURITY_BREACH: "Unauthorized access or data compromise incidents",
        CrisisType.DATA_LOSS: "Loss or corruption of critical business data",
        CrisisType.FINANCIAL_CRISIS: "Severe financial difficulties threatening business continuity",
        CrisisType.REGULATORY_VIOLATION: "Violations of regulatory requirements or compliance",
        CrisisType.REPUTATION_DAMAGE: "Events causing significant brand or reputation harm",
        CrisisType.PERSONNEL_CRISIS: "Critical staffing issues or workplace incidents",
        CrisisType.SUPPLY_CHAIN_DISRUPTION: "Disruptions to critical supply chains or vendors",
        CrisisType.MARKET_VOLATILITY: "Extreme market conditions affecting business operations",
        CrisisType.NATURAL_DISASTER: "Natural disasters impacting business operations",
        CrisisType.CYBER_ATTACK: "Malicious cyber attacks on systems or data",
        CrisisType.LEGAL_ISSUE: "Significant legal challenges or litigation"
    }
    
    return descriptions.get(crisis_type, "Unknown crisis type")


def _get_severity_description(severity: SeverityLevel) -> str:
    """Get human-readable description of severity level"""
    descriptions = {
        SeverityLevel.LOW: "Minor impact, routine response procedures",
        SeverityLevel.MEDIUM: "Moderate impact, standard escalation procedures",
        SeverityLevel.HIGH: "Significant impact, immediate attention required",
        SeverityLevel.CRITICAL: "Severe impact, executive involvement required",
        SeverityLevel.CATASTROPHIC: "Extreme impact, all-hands emergency response"
    }
    
    return descriptions.get(severity, "Unknown severity level")