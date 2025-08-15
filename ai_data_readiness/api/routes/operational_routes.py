"""API routes for operational dashboards and alerting."""

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

from ...core.operational_dashboard import get_dashboard_generator
from ...core.alerting_system import get_alerting_system, AlertRule, AlertContact, AlertChannel, AlertSeverity, EscalationLevel
from ...core.capacity_planner import get_capacity_planner, ResourceComponent
from ...models.monitoring_models import MonitoringDashboard

router = APIRouter(prefix="/operational", tags=["operational"])
logger = logging.getLogger(__name__)


# Dashboard endpoints
@router.get("/dashboards/templates")
async def get_dashboard_templates():
    """Get available dashboard templates."""
    try:
        generator = get_dashboard_generator()
        templates = list(generator.dashboard_templates.keys())
        
        template_info = {}
        for template_name in templates:
            template = generator.dashboard_templates[template_name]
            template_info[template_name] = {
                'name': template.name,
                'description': template.description,
                'widget_count': len(template.widgets),
                'refresh_interval': template.refresh_interval
            }
        
        return {
            'templates': template_info,
            'total_count': len(templates)
        }
    except Exception as e:
        logger.error(f"Error getting dashboard templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dashboards/create")
async def create_dashboard(
    template_name: str = Query(..., description="Dashboard template name"),
    custom_config: Optional[Dict[str, Any]] = None
):
    """Create a new dashboard from template."""
    try:
        generator = get_dashboard_generator()
        dashboard = generator.create_dashboard(template_name, custom_config)
        
        return {
            'dashboard_id': dashboard.id,
            'name': dashboard.name,
            'description': dashboard.description,
            'created_at': dashboard.created_at,
            'layout': dashboard.layout
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboards/{template_name}/html")
async def get_dashboard_html(
    template_name: str,
    time_range_hours: int = Query(default=24, ge=1, le=168, description="Time range for data")
):
    """Get dashboard as HTML page."""
    try:
        generator = get_dashboard_generator()
        dashboard = generator.create_dashboard(template_name)
        html_content = generator.generate_dashboard_html(dashboard, time_range_hours)
        
        return HTMLResponse(content=html_content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating dashboard HTML: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboards/{template_name}/data")
async def get_dashboard_data(
    template_name: str,
    time_range_hours: int = Query(default=24, ge=1, le=168, description="Time range for data")
):
    """Get dashboard data as JSON."""
    try:
        generator = get_dashboard_generator()
        dashboard = generator.create_dashboard(template_name)
        data = generator.get_dashboard_data(dashboard, time_range_hours)
        
        return {
            'dashboard_name': dashboard.name,
            'time_range_hours': time_range_hours,
            'generated_at': datetime.utcnow().isoformat(),
            'data': data
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dashboards/export")
async def export_dashboard_config(
    template_name: str = Query(..., description="Dashboard template name"),
    filepath: str = Query(..., description="Export file path")
):
    """Export dashboard configuration to file."""
    try:
        generator = get_dashboard_generator()
        dashboard = generator.create_dashboard(template_name)
        generator.export_dashboard_config(dashboard, filepath)
        
        return {
            'message': f"Dashboard configuration exported to {filepath}",
            'dashboard_id': dashboard.id,
            'exported_at': datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error exporting dashboard config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Alerting endpoints
@router.get("/alerts/system/status")
async def get_alerting_system_status():
    """Get alerting system status."""
    try:
        alerting = get_alerting_system()
        stats = alerting.get_alerting_statistics()
        
        return {
            'alerting_active': stats['alerting_active'],
            'statistics': stats,
            'status': 'operational' if stats['alerting_active'] else 'stopped'
        }
    except Exception as e:
        logger.error(f"Error getting alerting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/system/start")
async def start_alerting_system(
    check_interval_seconds: int = Query(default=30, ge=10, le=300, description="Alert check interval")
):
    """Start the alerting system."""
    try:
        alerting = get_alerting_system()
        alerting.start_alerting(check_interval_seconds)
        
        return {
            'message': 'Alerting system started',
            'check_interval_seconds': check_interval_seconds,
            'started_at': datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error starting alerting system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/system/stop")
async def stop_alerting_system():
    """Stop the alerting system."""
    try:
        alerting = get_alerting_system()
        alerting.stop_alerting()
        
        return {
            'message': 'Alerting system stopped',
            'stopped_at': datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error stopping alerting system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/rules")
async def get_alert_rules():
    """Get all alert rules."""
    try:
        alerting = get_alerting_system()
        rules = alerting.get_alert_rules()
        
        return {
            'alert_rules': rules,
            'total_count': len(rules)
        }
    except Exception as e:
        logger.error(f"Error getting alert rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/rules")
async def create_alert_rule(
    name: str = Query(..., description="Alert rule name"),
    description: str = Query(..., description="Alert rule description"),
    metric_name: str = Query(..., description="Metric to monitor"),
    condition: str = Query(..., description="Condition (>, <, >=, <=, ==, !=)"),
    threshold: float = Query(..., description="Alert threshold"),
    severity: AlertSeverity = Query(..., description="Alert severity"),
    duration_minutes: int = Query(default=5, description="Duration condition must persist"),
    cooldown_minutes: int = Query(default=30, description="Cooldown between alerts"),
    channels: List[AlertChannel] = Query(default=[AlertChannel.EMAIL], description="Alert channels")
):
    """Create a new alert rule."""
    try:
        alerting = get_alerting_system()
        
        rule = AlertRule(
            id=f"rule_{name.lower().replace(' ', '_')}_{int(datetime.utcnow().timestamp())}",
            name=name,
            description=description,
            metric_name=metric_name,
            condition=condition,
            threshold=threshold,
            severity=severity,
            duration_minutes=duration_minutes,
            cooldown_minutes=cooldown_minutes,
            channels=channels
        )
        
        alerting.add_alert_rule(rule)
        
        return {
            'message': 'Alert rule created successfully',
            'rule_id': rule.id,
            'rule': rule.to_dict()
        }
    except Exception as e:
        logger.error(f"Error creating alert rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/alerts/rules/{rule_id}")
async def delete_alert_rule(rule_id: str):
    """Delete an alert rule."""
    try:
        alerting = get_alerting_system()
        alerting.remove_alert_rule(rule_id)
        
        return {
            'message': f'Alert rule {rule_id} deleted successfully',
            'deleted_at': datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error deleting alert rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/contacts")
async def get_alert_contacts():
    """Get all alert contacts."""
    try:
        alerting = get_alerting_system()
        contacts = alerting.get_contacts()
        
        return {
            'contacts': contacts,
            'total_count': len(contacts)
        }
    except Exception as e:
        logger.error(f"Error getting alert contacts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/contacts")
async def create_alert_contact(
    name: str = Query(..., description="Contact name"),
    email: Optional[str] = Query(default=None, description="Email address"),
    phone: Optional[str] = Query(default=None, description="Phone number"),
    escalation_level: EscalationLevel = Query(default=EscalationLevel.L1, description="Escalation level"),
    channels: List[AlertChannel] = Query(default=[AlertChannel.EMAIL], description="Preferred channels")
):
    """Create a new alert contact."""
    try:
        alerting = get_alerting_system()
        
        contact = AlertContact(
            id=f"contact_{name.lower().replace(' ', '_')}_{int(datetime.utcnow().timestamp())}",
            name=name,
            email=email,
            phone=phone,
            escalation_level=escalation_level,
            channels=channels
        )
        
        alerting.add_contact(contact)
        
        return {
            'message': 'Alert contact created successfully',
            'contact_id': contact.id,
            'contact': contact.to_dict()
        }
    except Exception as e:
        logger.error(f"Error creating alert contact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/alerts/contacts/{contact_id}")
async def delete_alert_contact(contact_id: str):
    """Delete an alert contact."""
    try:
        alerting = get_alerting_system()
        alerting.remove_contact(contact_id)
        
        return {
            'message': f'Alert contact {contact_id} deleted successfully',
            'deleted_at': datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error deleting alert contact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/incidents/active")
async def get_active_incidents():
    """Get all active incidents."""
    try:
        alerting = get_alerting_system()
        incidents = alerting.get_active_incidents()
        
        return {
            'active_incidents': incidents,
            'total_count': len(incidents)
        }
    except Exception as e:
        logger.error(f"Error getting active incidents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/incidents/history")
async def get_incident_history(
    hours: int = Query(default=24, ge=1, le=168, description="Hours of history to retrieve")
):
    """Get incident history."""
    try:
        alerting = get_alerting_system()
        incidents = alerting.get_incident_history(hours=hours)
        
        return {
            'incident_history': incidents,
            'total_count': len(incidents),
            'period_hours': hours
        }
    except Exception as e:
        logger.error(f"Error getting incident history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/incidents/{incident_id}/acknowledge")
async def acknowledge_incident(
    incident_id: str,
    acknowledged_by: str = Query(..., description="User acknowledging the incident")
):
    """Acknowledge an incident."""
    try:
        alerting = get_alerting_system()
        alerting.acknowledge_incident(incident_id, acknowledged_by)
        
        return {
            'message': f'Incident {incident_id} acknowledged',
            'acknowledged_by': acknowledged_by,
            'acknowledged_at': datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error acknowledging incident: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/incidents/{incident_id}/resolve")
async def resolve_incident(
    incident_id: str,
    resolved_by: str = Query(..., description="User resolving the incident")
):
    """Resolve an incident."""
    try:
        alerting = get_alerting_system()
        alerting.resolve_incident(incident_id, resolved_by)
        
        return {
            'message': f'Incident {incident_id} resolved',
            'resolved_by': resolved_by,
            'resolved_at': datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error resolving incident: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Capacity planning endpoints
@router.get("/capacity/status")
async def get_capacity_status():
    """Get current capacity status."""
    try:
        planner = get_capacity_planner()
        status = planner.get_current_capacity_status()
        
        return {
            'capacity_status': status,
            'generated_at': datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting capacity status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/capacity/plan/generate")
async def generate_capacity_plan(
    time_horizon_days: int = Query(default=30, ge=7, le=365, description="Planning time horizon in days"),
    background_tasks: BackgroundTasks = None
):
    """Generate capacity planning report."""
    try:
        planner = get_capacity_planner()
        
        # Generate the plan (this might take some time)
        report = planner.generate_capacity_plan(time_horizon_days)
        
        return {
            'message': 'Capacity planning report generated successfully',
            'report': report.to_dict()
        }
    except Exception as e:
        logger.error(f"Error generating capacity plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/capacity/plan/export")
async def export_capacity_plan(
    time_horizon_days: int = Query(default=30, ge=7, le=365, description="Planning time horizon in days"),
    filepath: str = Query(..., description="Export file path")
):
    """Generate and export capacity planning report."""
    try:
        planner = get_capacity_planner()
        
        # Generate the plan
        report = planner.generate_capacity_plan(time_horizon_days)
        
        # Export to file
        planner.export_capacity_plan(report, filepath)
        
        return {
            'message': f'Capacity planning report exported to {filepath}',
            'time_horizon_days': time_horizon_days,
            'exported_at': datetime.utcnow().isoformat(),
            'summary': report.executive_summary
        }
    except Exception as e:
        logger.error(f"Error exporting capacity plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/capacity/components")
async def get_capacity_components():
    """Get available resource components for capacity planning."""
    try:
        components = [
            {
                'name': component.value,
                'display_name': component.value.replace('_', ' ').title(),
                'description': f"{component.value.replace('_', ' ').title()} resource utilization"
            }
            for component in ResourceComponent
        ]
        
        return {
            'components': components,
            'total_count': len(components)
        }
    except Exception as e:
        logger.error(f"Error getting capacity components: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@router.get("/health")
async def operational_health_check():
    """Health check for operational systems."""
    try:
        # Check dashboard generator
        generator = get_dashboard_generator()
        dashboard_status = "operational"
        
        # Check alerting system
        alerting = get_alerting_system()
        alerting_stats = alerting.get_alerting_statistics()
        alerting_status = "operational" if alerting_stats['alerting_active'] else "stopped"
        
        # Check capacity planner
        planner = get_capacity_planner()
        capacity_status = "operational"
        
        overall_status = "healthy"
        if alerting_status == "stopped":
            overall_status = "degraded"
        
        return {
            'overall_status': overall_status,
            'components': {
                'dashboard_generator': dashboard_status,
                'alerting_system': alerting_status,
                'capacity_planner': capacity_status
            },
            'alerting_statistics': alerting_stats,
            'checked_at': datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in operational health check: {e}")
        return JSONResponse(
            status_code=503,
            content={
                'overall_status': 'unhealthy',
                'error': str(e),
                'checked_at': datetime.utcnow().isoformat()
            }
        )


# Statistics endpoint
@router.get("/statistics")
async def get_operational_statistics():
    """Get comprehensive operational statistics."""
    try:
        alerting = get_alerting_system()
        planner = get_capacity_planner()
        
        # Get alerting statistics
        alerting_stats = alerting.get_alerting_statistics()
        
        # Get capacity status
        capacity_status = planner.get_current_capacity_status()
        
        # Get active incidents
        active_incidents = alerting.get_active_incidents()
        
        return {
            'alerting': alerting_stats,
            'capacity': {
                'overall_status': capacity_status.get('overall', {}).get('status', 'unknown'),
                'components_at_risk': len([
                    comp for comp in capacity_status.values()
                    if isinstance(comp, dict) and comp.get('status') in ['warning', 'critical']
                ])
            },
            'incidents': {
                'active_count': len(active_incidents),
                'critical_count': len([
                    i for i in active_incidents
                    if i.get('alert', {}).get('severity') == 'critical'
                ])
            },
            'generated_at': datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting operational statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))