"""
Webhook Event Handlers for Advanced Analytics Dashboard API

This module provides event handlers that trigger webhooks when
specific events occur in the analytics dashboard system.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .webhook_manager import WebhookManager, WebhookEventType

logger = logging.getLogger(__name__)

class WebhookHandlers:
    """Event handlers for triggering webhooks"""
    
    def __init__(self, webhook_manager: Optional[WebhookManager] = None):
        self.webhook_manager = webhook_manager or WebhookManager()
    
    # Dashboard Event Handlers
    
    async def on_dashboard_created(self, dashboard_data: Dict[str, Any]):
        """Handle dashboard creation event"""
        try:
            await self.webhook_manager.trigger_event(
                event_type=WebhookEventType.DASHBOARD_CREATED.value,
                data={
                    "dashboard_id": dashboard_data.get("id"),
                    "name": dashboard_data.get("name"),
                    "type": dashboard_data.get("type"),
                    "owner": dashboard_data.get("owner"),
                    "created_at": dashboard_data.get("created_at"),
                    "widget_count": len(dashboard_data.get("widgets", []))
                },
                metadata={
                    "source": "dashboard_manager",
                    "action": "create"
                }
            )
            logger.debug(f"Dashboard created webhook triggered: {dashboard_data.get('id')}")
        except Exception as e:
            logger.error(f"Error triggering dashboard created webhook: {str(e)}")
    
    async def on_dashboard_updated(self, dashboard_data: Dict[str, Any], changes: Dict[str, Any]):
        """Handle dashboard update event"""
        try:
            await self.webhook_manager.trigger_event(
                event_type=WebhookEventType.DASHBOARD_UPDATED.value,
                data={
                    "dashboard_id": dashboard_data.get("id"),
                    "name": dashboard_data.get("name"),
                    "type": dashboard_data.get("type"),
                    "owner": dashboard_data.get("owner"),
                    "updated_at": dashboard_data.get("updated_at"),
                    "changes": changes
                },
                metadata={
                    "source": "dashboard_manager",
                    "action": "update"
                }
            )
            logger.debug(f"Dashboard updated webhook triggered: {dashboard_data.get('id')}")
        except Exception as e:
            logger.error(f"Error triggering dashboard updated webhook: {str(e)}")
    
    async def on_dashboard_deleted(self, dashboard_id: str, dashboard_name: str, owner: str):
        """Handle dashboard deletion event"""
        try:
            await self.webhook_manager.trigger_event(
                event_type=WebhookEventType.DASHBOARD_DELETED.value,
                data={
                    "dashboard_id": dashboard_id,
                    "name": dashboard_name,
                    "owner": owner,
                    "deleted_at": datetime.utcnow().isoformat()
                },
                metadata={
                    "source": "dashboard_manager",
                    "action": "delete"
                }
            )
            logger.debug(f"Dashboard deleted webhook triggered: {dashboard_id}")
        except Exception as e:
            logger.error(f"Error triggering dashboard deleted webhook: {str(e)}")
    
    # Widget Event Handlers
    
    async def on_widget_added(self, dashboard_id: str, widget_data: Dict[str, Any]):
        """Handle widget addition event"""
        try:
            await self.webhook_manager.trigger_event(
                event_type=WebhookEventType.WIDGET_ADDED.value,
                data={
                    "dashboard_id": dashboard_id,
                    "widget_id": widget_data.get("id"),
                    "widget_type": widget_data.get("type"),
                    "title": widget_data.get("title"),
                    "position": widget_data.get("position"),
                    "added_at": datetime.utcnow().isoformat()
                },
                metadata={
                    "source": "dashboard_manager",
                    "action": "add_widget"
                }
            )
            logger.debug(f"Widget added webhook triggered: {widget_data.get('id')}")
        except Exception as e:
            logger.error(f"Error triggering widget added webhook: {str(e)}")
    
    async def on_widget_updated(self, dashboard_id: str, widget_data: Dict[str, Any], changes: Dict[str, Any]):
        """Handle widget update event"""
        try:
            await self.webhook_manager.trigger_event(
                event_type=WebhookEventType.WIDGET_UPDATED.value,
                data={
                    "dashboard_id": dashboard_id,
                    "widget_id": widget_data.get("id"),
                    "widget_type": widget_data.get("type"),
                    "title": widget_data.get("title"),
                    "changes": changes,
                    "updated_at": datetime.utcnow().isoformat()
                },
                metadata={
                    "source": "dashboard_manager",
                    "action": "update_widget"
                }
            )
            logger.debug(f"Widget updated webhook triggered: {widget_data.get('id')}")
        except Exception as e:
            logger.error(f"Error triggering widget updated webhook: {str(e)}")
    
    async def on_widget_removed(self, dashboard_id: str, widget_id: str, widget_type: str):
        """Handle widget removal event"""
        try:
            await self.webhook_manager.trigger_event(
                event_type=WebhookEventType.WIDGET_REMOVED.value,
                data={
                    "dashboard_id": dashboard_id,
                    "widget_id": widget_id,
                    "widget_type": widget_type,
                    "removed_at": datetime.utcnow().isoformat()
                },
                metadata={
                    "source": "dashboard_manager",
                    "action": "remove_widget"
                }
            )
            logger.debug(f"Widget removed webhook triggered: {widget_id}")
        except Exception as e:
            logger.error(f"Error triggering widget removed webhook: {str(e)}")
    
    # Insight Event Handlers
    
    async def on_insight_generated(self, insight_data: Dict[str, Any]):
        """Handle insight generation event"""
        try:
            await self.webhook_manager.trigger_event(
                event_type=WebhookEventType.INSIGHT_GENERATED.value,
                data={
                    "insight_id": insight_data.get("id"),
                    "type": insight_data.get("type"),
                    "title": insight_data.get("title"),
                    "significance": insight_data.get("significance"),
                    "confidence": insight_data.get("confidence"),
                    "dashboard_id": insight_data.get("dashboard_id"),
                    "generated_at": insight_data.get("created_at"),
                    "recommendations_count": len(insight_data.get("recommendations", []))
                },
                metadata={
                    "source": "insight_generator",
                    "action": "generate"
                }
            )
            logger.debug(f"Insight generated webhook triggered: {insight_data.get('id')}")
        except Exception as e:
            logger.error(f"Error triggering insight generated webhook: {str(e)}")
    
    async def on_insight_updated(self, insight_data: Dict[str, Any], changes: Dict[str, Any]):
        """Handle insight update event"""
        try:
            await self.webhook_manager.trigger_event(
                event_type=WebhookEventType.INSIGHT_UPDATED.value,
                data={
                    "insight_id": insight_data.get("id"),
                    "type": insight_data.get("type"),
                    "title": insight_data.get("title"),
                    "significance": insight_data.get("significance"),
                    "confidence": insight_data.get("confidence"),
                    "changes": changes,
                    "updated_at": datetime.utcnow().isoformat()
                },
                metadata={
                    "source": "insight_generator",
                    "action": "update"
                }
            )
            logger.debug(f"Insight updated webhook triggered: {insight_data.get('id')}")
        except Exception as e:
            logger.error(f"Error triggering insight updated webhook: {str(e)}")
    
    # Forecast Event Handlers
    
    async def on_forecast_created(self, forecast_data: Dict[str, Any]):
        """Handle forecast creation event"""
        try:
            await self.webhook_manager.trigger_event(
                event_type=WebhookEventType.FORECAST_CREATED.value,
                data={
                    "forecast_id": forecast_data.get("id"),
                    "metric": forecast_data.get("metric"),
                    "horizon": forecast_data.get("horizon"),
                    "model": forecast_data.get("model"),
                    "confidence": forecast_data.get("confidence"),
                    "accuracy": forecast_data.get("accuracy"),
                    "created_at": forecast_data.get("generated_at"),
                    "predictions_count": len(forecast_data.get("predictions", []))
                },
                metadata={
                    "source": "predictive_engine",
                    "action": "create"
                }
            )
            logger.debug(f"Forecast created webhook triggered: {forecast_data.get('id')}")
        except Exception as e:
            logger.error(f"Error triggering forecast created webhook: {str(e)}")
    
    async def on_forecast_updated(self, forecast_data: Dict[str, Any], changes: Dict[str, Any]):
        """Handle forecast update event"""
        try:
            await self.webhook_manager.trigger_event(
                event_type=WebhookEventType.FORECAST_UPDATED.value,
                data={
                    "forecast_id": forecast_data.get("id"),
                    "metric": forecast_data.get("metric"),
                    "horizon": forecast_data.get("horizon"),
                    "model": forecast_data.get("model"),
                    "confidence": forecast_data.get("confidence"),
                    "changes": changes,
                    "updated_at": datetime.utcnow().isoformat()
                },
                metadata={
                    "source": "predictive_engine",
                    "action": "update"
                }
            )
            logger.debug(f"Forecast updated webhook triggered: {forecast_data.get('id')}")
        except Exception as e:
            logger.error(f"Error triggering forecast updated webhook: {str(e)}")
    
    # ROI Event Handlers
    
    async def on_roi_analysis_completed(self, roi_data: Dict[str, Any]):
        """Handle ROI analysis completion event"""
        try:
            await self.webhook_manager.trigger_event(
                event_type=WebhookEventType.ROI_ANALYSIS_COMPLETED.value,
                data={
                    "analysis_id": roi_data.get("id"),
                    "project_id": roi_data.get("project_id"),
                    "project_name": roi_data.get("project_name"),
                    "roi_percentage": roi_data.get("roi_percentage"),
                    "payback_period": roi_data.get("payback_period"),
                    "npv": roi_data.get("npv"),
                    "irr": roi_data.get("irr"),
                    "total_investment": roi_data.get("total_investment"),
                    "total_benefits": roi_data.get("total_benefits"),
                    "completed_at": roi_data.get("analysis_date")
                },
                metadata={
                    "source": "roi_calculator",
                    "action": "complete_analysis"
                }
            )
            logger.debug(f"ROI analysis completed webhook triggered: {roi_data.get('id')}")
        except Exception as e:
            logger.error(f"Error triggering ROI analysis completed webhook: {str(e)}")
    
    # Data Source Event Handlers
    
    async def on_data_source_connected(self, source_data: Dict[str, Any]):
        """Handle data source connection event"""
        try:
            await self.webhook_manager.trigger_event(
                event_type=WebhookEventType.DATA_SOURCE_CONNECTED.value,
                data={
                    "source_id": source_data.get("id"),
                    "name": source_data.get("name"),
                    "type": source_data.get("type"),
                    "status": source_data.get("status"),
                    "connected_at": datetime.utcnow().isoformat(),
                    "record_count": source_data.get("record_count", 0)
                },
                metadata={
                    "source": "data_connector",
                    "action": "connect"
                }
            )
            logger.debug(f"Data source connected webhook triggered: {source_data.get('id')}")
        except Exception as e:
            logger.error(f"Error triggering data source connected webhook: {str(e)}")
    
    async def on_data_source_disconnected(self, source_id: str, source_name: str, reason: str):
        """Handle data source disconnection event"""
        try:
            await self.webhook_manager.trigger_event(
                event_type=WebhookEventType.DATA_SOURCE_DISCONNECTED.value,
                data={
                    "source_id": source_id,
                    "name": source_name,
                    "reason": reason,
                    "disconnected_at": datetime.utcnow().isoformat()
                },
                metadata={
                    "source": "data_connector",
                    "action": "disconnect"
                }
            )
            logger.debug(f"Data source disconnected webhook triggered: {source_id}")
        except Exception as e:
            logger.error(f"Error triggering data source disconnected webhook: {str(e)}")
    
    async def on_data_sync_completed(self, sync_data: Dict[str, Any]):
        """Handle data sync completion event"""
        try:
            await self.webhook_manager.trigger_event(
                event_type=WebhookEventType.DATA_SYNC_COMPLETED.value,
                data={
                    "source_id": sync_data.get("source_id"),
                    "source_name": sync_data.get("source_name"),
                    "records_processed": sync_data.get("records_processed"),
                    "records_added": sync_data.get("records_added"),
                    "records_updated": sync_data.get("records_updated"),
                    "records_deleted": sync_data.get("records_deleted"),
                    "sync_duration": sync_data.get("sync_duration"),
                    "completed_at": sync_data.get("completed_at")
                },
                metadata={
                    "source": "data_connector",
                    "action": "sync_complete"
                }
            )
            logger.debug(f"Data sync completed webhook triggered: {sync_data.get('source_id')}")
        except Exception as e:
            logger.error(f"Error triggering data sync completed webhook: {str(e)}")
    
    async def on_data_sync_failed(self, source_id: str, source_name: str, error: str):
        """Handle data sync failure event"""
        try:
            await self.webhook_manager.trigger_event(
                event_type=WebhookEventType.DATA_SYNC_FAILED.value,
                data={
                    "source_id": source_id,
                    "source_name": source_name,
                    "error": error,
                    "failed_at": datetime.utcnow().isoformat()
                },
                metadata={
                    "source": "data_connector",
                    "action": "sync_failed"
                }
            )
            logger.debug(f"Data sync failed webhook triggered: {source_id}")
        except Exception as e:
            logger.error(f"Error triggering data sync failed webhook: {str(e)}")
    
    # Alert Event Handlers
    
    async def on_alert_triggered(self, alert_data: Dict[str, Any]):
        """Handle alert trigger event"""
        try:
            await self.webhook_manager.trigger_event(
                event_type=WebhookEventType.ALERT_TRIGGERED.value,
                data={
                    "alert_id": alert_data.get("id"),
                    "type": alert_data.get("type"),
                    "severity": alert_data.get("severity"),
                    "title": alert_data.get("title"),
                    "message": alert_data.get("message"),
                    "dashboard_id": alert_data.get("dashboard_id"),
                    "widget_id": alert_data.get("widget_id"),
                    "metric": alert_data.get("metric"),
                    "threshold": alert_data.get("threshold"),
                    "current_value": alert_data.get("current_value"),
                    "triggered_at": alert_data.get("triggered_at")
                },
                metadata={
                    "source": "alerting_system",
                    "action": "trigger"
                }
            )
            logger.debug(f"Alert triggered webhook triggered: {alert_data.get('id')}")
        except Exception as e:
            logger.error(f"Error triggering alert webhook: {str(e)}")
    
    # Report Event Handlers
    
    async def on_report_generated(self, report_data: Dict[str, Any]):
        """Handle report generation event"""
        try:
            await self.webhook_manager.trigger_event(
                event_type=WebhookEventType.REPORT_GENERATED.value,
                data={
                    "report_id": report_data.get("id"),
                    "title": report_data.get("title"),
                    "type": report_data.get("type"),
                    "format": report_data.get("format"),
                    "file_size": report_data.get("file_size"),
                    "dashboard_id": report_data.get("dashboard_id"),
                    "generated_at": report_data.get("generated_at"),
                    "download_url": report_data.get("download_url")
                },
                metadata={
                    "source": "reporting_engine",
                    "action": "generate"
                }
            )
            logger.debug(f"Report generated webhook triggered: {report_data.get('id')}")
        except Exception as e:
            logger.error(f"Error triggering report generated webhook: {str(e)}")
    
    async def on_schedule_executed(self, schedule_data: Dict[str, Any], execution_data: Dict[str, Any]):
        """Handle schedule execution event"""
        try:
            await self.webhook_manager.trigger_event(
                event_type=WebhookEventType.SCHEDULE_EXECUTED.value,
                data={
                    "schedule_id": schedule_data.get("id"),
                    "schedule_name": schedule_data.get("name"),
                    "report_type": schedule_data.get("report_type"),
                    "execution_id": execution_data.get("id"),
                    "status": execution_data.get("status"),
                    "duration": execution_data.get("duration"),
                    "executed_at": execution_data.get("executed_at"),
                    "report_id": execution_data.get("report_id"),
                    "recipients": schedule_data.get("recipients", [])
                },
                metadata={
                    "source": "report_scheduler",
                    "action": "execute"
                }
            )
            logger.debug(f"Schedule executed webhook triggered: {schedule_data.get('id')}")
        except Exception as e:
            logger.error(f"Error triggering schedule executed webhook: {str(e)}")
    
    # Threshold and Anomaly Event Handlers
    
    async def on_threshold_breached(self, threshold_data: Dict[str, Any]):
        """Handle threshold breach event"""
        try:
            await self.webhook_manager.trigger_event(
                event_type=WebhookEventType.THRESHOLD_BREACHED.value,
                data={
                    "threshold_id": threshold_data.get("id"),
                    "metric": threshold_data.get("metric"),
                    "threshold_value": threshold_data.get("threshold_value"),
                    "current_value": threshold_data.get("current_value"),
                    "breach_type": threshold_data.get("breach_type"),  # "above" or "below"
                    "dashboard_id": threshold_data.get("dashboard_id"),
                    "widget_id": threshold_data.get("widget_id"),
                    "severity": threshold_data.get("severity"),
                    "breached_at": threshold_data.get("breached_at")
                },
                metadata={
                    "source": "monitoring_system",
                    "action": "threshold_breach"
                }
            )
            logger.debug(f"Threshold breached webhook triggered: {threshold_data.get('id')}")
        except Exception as e:
            logger.error(f"Error triggering threshold breached webhook: {str(e)}")
    
    async def on_anomaly_detected(self, anomaly_data: Dict[str, Any]):
        """Handle anomaly detection event"""
        try:
            await self.webhook_manager.trigger_event(
                event_type=WebhookEventType.ANOMALY_DETECTED.value,
                data={
                    "anomaly_id": anomaly_data.get("id"),
                    "metric": anomaly_data.get("metric"),
                    "expected_value": anomaly_data.get("expected_value"),
                    "actual_value": anomaly_data.get("actual_value"),
                    "anomaly_score": anomaly_data.get("anomaly_score"),
                    "confidence": anomaly_data.get("confidence"),
                    "dashboard_id": anomaly_data.get("dashboard_id"),
                    "widget_id": anomaly_data.get("widget_id"),
                    "detected_at": anomaly_data.get("detected_at"),
                    "possible_causes": anomaly_data.get("possible_causes", [])
                },
                metadata={
                    "source": "anomaly_detector",
                    "action": "detect"
                }
            )
            logger.debug(f"Anomaly detected webhook triggered: {anomaly_data.get('id')}")
        except Exception as e:
            logger.error(f"Error triggering anomaly detected webhook: {str(e)}")
    
    # Utility Methods
    
    async def trigger_custom_event(self, event_type: str, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        """Trigger a custom webhook event"""
        try:
            await self.webhook_manager.trigger_event(
                event_type=event_type,
                data=data,
                metadata=metadata or {"source": "custom", "action": "trigger"}
            )
            logger.debug(f"Custom webhook triggered: {event_type}")
        except Exception as e:
            logger.error(f"Error triggering custom webhook: {str(e)}")
    
    async def get_supported_events(self) -> List[str]:
        """Get list of supported webhook events"""
        return [event.value for event in WebhookEventType]