"""API routes for compliance reporting functionality."""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
import logging

from ...engines.compliance_reporter import ComplianceReporter, ComplianceReporterError
from ...models.governance_models import ComplianceReport
from ..middleware.auth import get_current_user
from ..models.responses import APIResponse


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/compliance", tags=["compliance"])


# Request/Response Models
class ComplianceReportRequest(BaseModel):
    framework: str = Field(..., regex="^(GDPR|CCPA|SOX|HIPAA)$")
    scope: Optional[List[str]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    generated_by: Optional[str] = None


class ComplianceDashboardRequest(BaseModel):
    frameworks: Optional[List[str]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


class DataClassificationValidationRequest(BaseModel):
    dataset_id: str


class AccessComplianceAuditRequest(BaseModel):
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


class AuditTrailReportRequest(BaseModel):
    resource_id: Optional[str] = None
    user_id: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    event_types: Optional[List[str]] = None


# Initialize service
compliance_reporter = ComplianceReporter()


@router.post("/report")
async def generate_compliance_report(
    request: ComplianceReportRequest,
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Generate comprehensive compliance report."""
    try:
        generated_by = request.generated_by or current_user.get('id', 'unknown')
        
        report = compliance_reporter.generate_compliance_report(
            framework=request.framework,
            scope=request.scope,
            start_date=request.start_date,
            end_date=request.end_date,
            generated_by=generated_by
        )
        
        # Convert to serializable format
        report_data = {
            'id': report.id,
            'report_type': report.report_type,
            'scope': report.scope,
            'compliance_score': report.compliance_score,
            'violations': report.violations,
            'recommendations': report.recommendations,
            'assessment_criteria': report.assessment_criteria,
            'generated_by': report.generated_by,
            'generated_at': report.generated_at,
            'period_start': report.period_start,
            'period_end': report.period_end
        }
        
        return APIResponse(
            success=True,
            message=f"{request.framework} compliance report generated successfully",
            data=report_data
        )
        
    except ComplianceReporterError as e:
        logger.error(f"Compliance reporter error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating compliance report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate compliance report: {str(e)}")


@router.post("/dashboard")
async def get_compliance_dashboard(
    request: ComplianceDashboardRequest,
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Get compliance dashboard with key metrics."""
    try:
        dashboard = compliance_reporter.get_compliance_dashboard(
            frameworks=request.frameworks,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        return APIResponse(
            success=True,
            message="Compliance dashboard retrieved successfully",
            data=dashboard
        )
        
    except ComplianceReporterError as e:
        logger.error(f"Compliance reporter error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting compliance dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get compliance dashboard: {str(e)}")


@router.post("/validate-classification")
async def validate_data_classification_compliance(
    request: DataClassificationValidationRequest,
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Validate data classification compliance for a dataset."""
    try:
        validation_result = compliance_reporter.validate_data_classification_compliance(
            dataset_id=request.dataset_id
        )
        
        return APIResponse(
            success=True,
            message="Data classification validation completed",
            data=validation_result
        )
        
    except ComplianceReporterError as e:
        logger.error(f"Compliance reporter error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error validating data classification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to validate data classification: {str(e)}")


@router.post("/audit-access")
async def audit_access_compliance(
    request: AccessComplianceAuditRequest,
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Audit access compliance across the system."""
    try:
        audit_result = compliance_reporter.audit_access_compliance(
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        return APIResponse(
            success=True,
            message="Access compliance audit completed",
            data=audit_result
        )
        
    except ComplianceReporterError as e:
        logger.error(f"Compliance reporter error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error auditing access compliance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to audit access compliance: {str(e)}")


@router.post("/audit-trail-report")
async def generate_audit_trail_report(
    request: AuditTrailReportRequest,
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Generate comprehensive audit trail report."""
    try:
        report = compliance_reporter.generate_audit_trail_report(
            resource_id=request.resource_id,
            user_id=request.user_id,
            start_date=request.start_date,
            end_date=request.end_date,
            event_types=request.event_types
        )
        
        return APIResponse(
            success=True,
            message="Audit trail report generated successfully",
            data=report
        )
        
    except ComplianceReporterError as e:
        logger.error(f"Compliance reporter error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating audit trail report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate audit trail report: {str(e)}")


@router.get("/frameworks")
async def get_supported_frameworks(
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Get list of supported compliance frameworks."""
    try:
        frameworks = compliance_reporter.compliance_frameworks
        
        framework_list = [
            {
                'code': code,
                'name': info['name'],
                'requirements': info['requirements']
            }
            for code, info in frameworks.items()
        ]
        
        return APIResponse(
            success=True,
            message="Supported compliance frameworks retrieved successfully",
            data={
                'frameworks': framework_list,
                'total_frameworks': len(framework_list)
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting supported frameworks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get supported frameworks: {str(e)}")


@router.get("/framework/{framework}/requirements")
async def get_framework_requirements(
    framework: str,
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Get requirements for a specific compliance framework."""
    try:
        if framework not in compliance_reporter.compliance_frameworks:
            raise HTTPException(status_code=404, detail=f"Framework {framework} not supported")
        
        framework_info = compliance_reporter.compliance_frameworks[framework]
        
        return APIResponse(
            success=True,
            message=f"{framework} requirements retrieved successfully",
            data={
                'framework': framework,
                'name': framework_info['name'],
                'requirements': framework_info['requirements']
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting framework requirements: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get framework requirements: {str(e)}")


@router.get("/reports")
async def list_compliance_reports(
    framework: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    limit: int = Query(50, ge=1, le=1000),
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """List compliance reports with filters."""
    try:
        from ...models.database import get_db_session
        from ...models.governance_database import ComplianceReportModel
        
        with get_db_session() as session:
            query = session.query(ComplianceReportModel)
            
            if framework:
                query = query.filter(ComplianceReportModel.report_type == framework)
            
            if start_date:
                query = query.filter(ComplianceReportModel.generated_at >= start_date)
            
            if end_date:
                query = query.filter(ComplianceReportModel.generated_at <= end_date)
            
            reports = query.order_by(ComplianceReportModel.generated_at.desc()).limit(limit).all()
            
            report_list = [
                {
                    'id': str(report.id),
                    'report_type': report.report_type,
                    'compliance_score': report.compliance_score,
                    'violations_count': len(report.violations) if report.violations else 0,
                    'generated_by': report.generated_by,
                    'generated_at': report.generated_at,
                    'period_start': report.period_start,
                    'period_end': report.period_end
                }
                for report in reports
            ]
            
            return APIResponse(
                success=True,
                message="Compliance reports retrieved successfully",
                data={
                    'reports': report_list,
                    'total_reports': len(report_list),
                    'filters': {
                        'framework': framework,
                        'start_date': start_date,
                        'end_date': end_date,
                        'limit': limit
                    }
                }
            )
        
    except Exception as e:
        logger.error(f"Error listing compliance reports: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list compliance reports: {str(e)}")


@router.get("/report/{report_id}")
async def get_compliance_report(
    report_id: str,
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Get detailed compliance report by ID."""
    try:
        from ...models.database import get_db_session
        from ...models.governance_database import ComplianceReportModel
        
        with get_db_session() as session:
            report = session.query(ComplianceReportModel).filter(
                ComplianceReportModel.id == report_id
            ).first()
            
            if not report:
                raise HTTPException(status_code=404, detail=f"Compliance report {report_id} not found")
            
            report_data = {
                'id': str(report.id),
                'report_type': report.report_type,
                'scope': report.scope,
                'compliance_score': report.compliance_score,
                'violations': report.violations,
                'recommendations': report.recommendations,
                'assessment_criteria': report.assessment_criteria,
                'generated_by': report.generated_by,
                'generated_at': report.generated_at,
                'period_start': report.period_start,
                'period_end': report.period_end
            }
            
            return APIResponse(
                success=True,
                message="Compliance report retrieved successfully",
                data=report_data
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting compliance report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get compliance report: {str(e)}")


@router.get("/metrics/summary")
async def get_compliance_metrics_summary(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Get compliance metrics summary."""
    try:
        from ...models.database import get_db_session
        from ...models.governance_database import ComplianceReportModel
        from sqlalchemy import func
        
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        with get_db_session() as session:
            # Get compliance metrics
            reports = session.query(ComplianceReportModel).filter(
                ComplianceReportModel.generated_at >= start_date,
                ComplianceReportModel.generated_at <= end_date
            ).all()
            
            if not reports:
                summary = {
                    'period': {'start_date': start_date, 'end_date': end_date},
                    'total_reports': 0,
                    'average_score': 0,
                    'total_violations': 0,
                    'framework_breakdown': {},
                    'trend': 'stable'
                }
            else:
                # Calculate summary metrics
                total_reports = len(reports)
                average_score = sum(report.compliance_score for report in reports) / total_reports
                total_violations = sum(len(report.violations) for report in reports if report.violations)
                
                # Framework breakdown
                framework_breakdown = {}
                for report in reports:
                    framework = report.report_type
                    if framework not in framework_breakdown:
                        framework_breakdown[framework] = {
                            'count': 0,
                            'average_score': 0,
                            'violations': 0
                        }
                    
                    framework_breakdown[framework]['count'] += 1
                    framework_breakdown[framework]['average_score'] += report.compliance_score
                    framework_breakdown[framework]['violations'] += len(report.violations) if report.violations else 0
                
                # Calculate averages
                for framework_data in framework_breakdown.values():
                    if framework_data['count'] > 0:
                        framework_data['average_score'] /= framework_data['count']
                
                # Calculate trend (simplified)
                sorted_reports = sorted(reports, key=lambda x: x.generated_at)
                if len(sorted_reports) >= 2:
                    first_half = sorted_reports[:len(sorted_reports)//2]
                    second_half = sorted_reports[len(sorted_reports)//2:]
                    
                    first_avg = sum(r.compliance_score for r in first_half) / len(first_half)
                    second_avg = sum(r.compliance_score for r in second_half) / len(second_half)
                    
                    if second_avg > first_avg * 1.05:
                        trend = 'improving'
                    elif second_avg < first_avg * 0.95:
                        trend = 'declining'
                    else:
                        trend = 'stable'
                else:
                    trend = 'stable'
                
                summary = {
                    'period': {'start_date': start_date, 'end_date': end_date},
                    'total_reports': total_reports,
                    'average_score': round(average_score, 2),
                    'total_violations': total_violations,
                    'framework_breakdown': framework_breakdown,
                    'trend': trend
                }
            
            return APIResponse(
                success=True,
                message="Compliance metrics summary retrieved successfully",
                data=summary
            )
        
    except Exception as e:
        logger.error(f"Error getting compliance metrics summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get compliance metrics summary: {str(e)}")


@router.get("/health")
async def health_check() -> APIResponse:
    """Health check endpoint for compliance service."""
    try:
        # Test basic functionality
        frameworks = compliance_reporter.compliance_frameworks
        
        return APIResponse(
            success=True,
            message="Compliance service is healthy",
            data={
                'service': 'compliance',
                'status': 'healthy',
                'supported_frameworks': len(frameworks),
                'timestamp': datetime.utcnow()
            }
        )
        
    except Exception as e:
        logger.error(f"Compliance service health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Compliance service unhealthy: {str(e)}")