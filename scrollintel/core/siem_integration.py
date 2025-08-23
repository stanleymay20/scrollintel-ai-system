"""
SIEM Integration System
Handles integration with Splunk, ELK, and other SIEM platforms
"""
import json
import asyncio
import aiohttp
import ssl
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from sqlalchemy.orm import Session

from ..models.security_audit_models import (
    SIEMIntegration, SIEMPlatform, SecurityAuditLog, 
    SIEMEventPayload, SIEMIntegrationCreate
)
from ..core.database_connection_manager import get_sync_session
from ..core.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class SIEMConnectionConfig:
    """SIEM connection configuration"""
    endpoint_url: str
    api_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    certificate_path: Optional[str] = None
    timeout: int = 30
    retry_count: int = 3
    batch_size: int = 100

class BaseSIEMConnector(ABC):
    """Base class for SIEM connectors"""
    
    def __init__(self, config: SIEMConnectionConfig):
        self.config = config
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def send_events(self, events: List[SIEMEventPayload]) -> bool:
        """Send events to SIEM platform"""
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test connection to SIEM platform"""
        pass
    
    @abstractmethod
    def format_event(self, audit_log: SecurityAuditLog) -> Dict[str, Any]:
        """Format audit log for SIEM platform"""
        pass

class SplunkConnector(BaseSIEMConnector):
    """Splunk SIEM connector"""
    
    async def send_events(self, events: List[SIEMEventPayload]) -> bool:
        """Send events to Splunk via HTTP Event Collector"""
        try:
            headers = {
                "Authorization": f"Splunk {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            # Format events for Splunk
            splunk_events = []
            for event in events:
                splunk_event = {
                    "time": int(event.timestamp.timestamp()),
                    "event": self._format_splunk_event(event),
                    "source": "scrollintel",
                    "sourcetype": "security_audit",
                    "index": "security"
                }
                splunk_events.append(splunk_event)
            
            # Send in batches
            for i in range(0, len(splunk_events), self.config.batch_size):
                batch = splunk_events[i:i + self.config.batch_size]
                payload = "\n".join([json.dumps(event) for event in batch])
                
                async with self.session.post(
                    f"{self.config.endpoint_url}/services/collector/event",
                    headers=headers,
                    data=payload
                ) as response:
                    if response.status != 200:
                        logger.error(f"Splunk event submission failed: {response.status}")
                        return False
            
            logger.info(f"Successfully sent {len(events)} events to Splunk")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send events to Splunk: {str(e)}")
            return False
    
    async def test_connection(self) -> bool:
        """Test Splunk connection"""
        try:
            headers = {
                "Authorization": f"Splunk {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            async with self.session.get(
                f"{self.config.endpoint_url}/services/collector/health",
                headers=headers
            ) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Splunk connection test failed: {str(e)}")
            return False
    
    def format_event(self, audit_log: SecurityAuditLog) -> Dict[str, Any]:
        """Format event for Splunk"""
        return self._format_splunk_event(SIEMEventPayload(
            timestamp=audit_log.timestamp,
            event_id=audit_log.id,
            event_type=audit_log.event_type,
            severity=audit_log.severity,
            source=audit_log.source_ip or "unknown",
            user=audit_log.user_id,
            action=audit_log.action,
            outcome=audit_log.outcome,
            message=f"{audit_log.action} {audit_log.outcome}",
            details=audit_log.details or {},
            risk_score=audit_log.risk_score,
            tags=[audit_log.event_type, audit_log.severity]
        ))
    
    def _format_splunk_event(self, event: SIEMEventPayload) -> Dict[str, Any]:
        """Format event for Splunk format"""
        return {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "severity": event.severity,
            "source_ip": event.source,
            "destination": event.destination,
            "user": event.user,
            "action": event.action,
            "outcome": event.outcome,
            "message": event.message,
            "risk_score": event.risk_score,
            "tags": event.tags,
            **event.details
        }

class ELKConnector(BaseSIEMConnector):
    """Elasticsearch (ELK Stack) connector"""
    
    async def send_events(self, events: List[SIEMEventPayload]) -> bool:
        """Send events to Elasticsearch"""
        try:
            headers = {"Content-Type": "application/json"}
            
            if self.config.username and self.config.password:
                auth = aiohttp.BasicAuth(self.config.username, self.config.password)
            else:
                auth = None
            
            # Prepare bulk index operations
            bulk_data = []
            for event in events:
                index_meta = {
                    "index": {
                        "_index": f"security-audit-{datetime.now().strftime('%Y.%m')}",
                        "_type": "_doc"
                    }
                }
                bulk_data.append(json.dumps(index_meta))
                bulk_data.append(json.dumps(self._format_elk_event(event)))
            
            payload = "\n".join(bulk_data) + "\n"
            
            async with self.session.post(
                f"{self.config.endpoint_url}/_bulk",
                headers=headers,
                data=payload,
                auth=auth
            ) as response:
                if response.status not in [200, 201]:
                    logger.error(f"ELK event submission failed: {response.status}")
                    return False
                
                result = await response.json()
                if result.get("errors"):
                    logger.error(f"ELK bulk operation had errors: {result}")
                    return False
            
            logger.info(f"Successfully sent {len(events)} events to ELK")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send events to ELK: {str(e)}")
            return False
    
    async def test_connection(self) -> bool:
        """Test ELK connection"""
        try:
            auth = None
            if self.config.username and self.config.password:
                auth = aiohttp.BasicAuth(self.config.username, self.config.password)
            
            async with self.session.get(
                f"{self.config.endpoint_url}/_cluster/health",
                auth=auth
            ) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"ELK connection test failed: {str(e)}")
            return False
    
    def format_event(self, audit_log: SecurityAuditLog) -> Dict[str, Any]:
        """Format event for ELK"""
        return self._format_elk_event(SIEMEventPayload(
            timestamp=audit_log.timestamp,
            event_id=audit_log.id,
            event_type=audit_log.event_type,
            severity=audit_log.severity,
            source=audit_log.source_ip or "unknown",
            user=audit_log.user_id,
            action=audit_log.action,
            outcome=audit_log.outcome,
            message=f"{audit_log.action} {audit_log.outcome}",
            details=audit_log.details or {},
            risk_score=audit_log.risk_score,
            tags=[audit_log.event_type, audit_log.severity]
        ))
    
    def _format_elk_event(self, event: SIEMEventPayload) -> Dict[str, Any]:
        """Format event for ELK format"""
        return {
            "@timestamp": event.timestamp.isoformat(),
            "event_id": event.event_id,
            "event_type": event.event_type,
            "severity": event.severity,
            "source": {
                "ip": event.source,
                "user": event.user
            },
            "destination": event.destination,
            "action": event.action,
            "outcome": event.outcome,
            "message": event.message,
            "risk_score": event.risk_score,
            "tags": event.tags,
            "details": event.details,
            "scrollintel": {
                "version": "1.0",
                "component": "security_audit"
            }
        }

class QRadarConnector(BaseSIEMConnector):
    """IBM QRadar connector"""
    
    async def send_events(self, events: List[SIEMEventPayload]) -> bool:
        """Send events to QRadar via REST API"""
        try:
            headers = {
                "SEC": self.config.api_key,
                "Content-Type": "application/json",
                "Version": "12.0"
            }
            
            # QRadar expects events in specific format
            qradar_events = []
            for event in events:
                qradar_event = self._format_qradar_event(event)
                qradar_events.append(qradar_event)
            
            async with self.session.post(
                f"{self.config.endpoint_url}/api/siem/events",
                headers=headers,
                json=qradar_events
            ) as response:
                if response.status not in [200, 201]:
                    logger.error(f"QRadar event submission failed: {response.status}")
                    return False
            
            logger.info(f"Successfully sent {len(events)} events to QRadar")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send events to QRadar: {str(e)}")
            return False
    
    async def test_connection(self) -> bool:
        """Test QRadar connection"""
        try:
            headers = {
                "SEC": self.config.api_key,
                "Version": "12.0"
            }
            
            async with self.session.get(
                f"{self.config.endpoint_url}/api/system/about",
                headers=headers
            ) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"QRadar connection test failed: {str(e)}")
            return False
    
    def format_event(self, audit_log: SecurityAuditLog) -> Dict[str, Any]:
        """Format event for QRadar"""
        return self._format_qradar_event(SIEMEventPayload(
            timestamp=audit_log.timestamp,
            event_id=audit_log.id,
            event_type=audit_log.event_type,
            severity=audit_log.severity,
            source=audit_log.source_ip or "unknown",
            user=audit_log.user_id,
            action=audit_log.action,
            outcome=audit_log.outcome,
            message=f"{audit_log.action} {audit_log.outcome}",
            details=audit_log.details or {},
            risk_score=audit_log.risk_score,
            tags=[audit_log.event_type, audit_log.severity]
        ))
    
    def _format_qradar_event(self, event: SIEMEventPayload) -> Dict[str, Any]:
        """Format event for QRadar format"""
        return {
            "events": [{
                "qid": 28250001,  # Custom event ID for ScrollIntel
                "source_ip": event.source,
                "destination_ip": event.destination,
                "start_time": int(event.timestamp.timestamp() * 1000),
                "event_count": 1,
                "magnitude": event.risk_score,
                "username": event.user,
                "properties": [
                    {"name": "EventID", "value": event.event_id},
                    {"name": "EventType", "value": event.event_type},
                    {"name": "Severity", "value": event.severity},
                    {"name": "Action", "value": event.action},
                    {"name": "Outcome", "value": event.outcome},
                    {"name": "Message", "value": event.message},
                    {"name": "Details", "value": json.dumps(event.details)}
                ]
            }]
        }

class SIEMIntegrationManager:
    """Manages SIEM integrations and event forwarding"""
    
    def __init__(self):
        self.connectors = {
            SIEMPlatform.SPLUNK: SplunkConnector,
            SIEMPlatform.ELK_STACK: ELKConnector,
            SIEMPlatform.QRADAR: QRadarConnector
        }
        self.active_integrations = {}
    
    async def create_integration(self, integration_data: SIEMIntegrationCreate) -> str:
        """Create new SIEM integration"""
        try:
            integration_id = str(uuid.uuid4())
            
            integration = SIEMIntegration(
                id=integration_id,
                name=integration_data.name,
                platform=integration_data.platform.value,
                endpoint_url=integration_data.endpoint_url,
                api_key=integration_data.api_key,
                username=integration_data.username,
                password=integration_data.password,
                certificate_path=integration_data.certificate_path,
                config=integration_data.config or {}
            )
            
            with get_sync_session() as db:
                db.add(integration)
                db.commit()
            
            # Test the integration
            if await self.test_integration(integration_id):
                logger.info(f"SIEM integration created successfully: {integration_id}")
                return integration_id
            else:
                logger.error(f"SIEM integration test failed: {integration_id}")
                raise Exception("Integration test failed")
                
        except Exception as e:
            logger.error(f"Failed to create SIEM integration: {str(e)}")
            raise
    
    async def test_integration(self, integration_id: str) -> bool:
        """Test SIEM integration connection"""
        try:
            with get_sync_session() as db:
                integration = db.query(SIEMIntegration).filter(
                    SIEMIntegration.id == integration_id
                ).first()
                
                if not integration:
                    return False
                
                config = SIEMConnectionConfig(
                    endpoint_url=integration.endpoint_url,
                    api_key=integration.api_key,
                    username=integration.username,
                    password=integration.password,
                    certificate_path=integration.certificate_path
                )
                
                connector_class = self.connectors.get(SIEMPlatform(integration.platform))
                if not connector_class:
                    return False
                
                async with connector_class(config) as connector:
                    return await connector.test_connection()
                    
        except Exception as e:
            logger.error(f"SIEM integration test failed: {str(e)}")
            return False
    
    async def forward_events_to_siem(
        self,
        events: List[SecurityAuditLog],
        integration_ids: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """Forward security events to SIEM platforms"""
        results = {}
        
        try:
            with get_sync_session() as db:
                query = db.query(SIEMIntegration).filter(SIEMIntegration.is_active == True)
                
                if integration_ids:
                    query = query.filter(SIEMIntegration.id.in_(integration_ids))
                
                integrations = query.all()
            
            for integration in integrations:
                try:
                    config = SIEMConnectionConfig(
                        endpoint_url=integration.endpoint_url,
                        api_key=integration.api_key,
                        username=integration.username,
                        password=integration.password,
                        certificate_path=integration.certificate_path
                    )
                    
                    connector_class = self.connectors.get(SIEMPlatform(integration.platform))
                    if not connector_class:
                        results[integration.id] = False
                        continue
                    
                    # Convert audit logs to SIEM events
                    siem_events = []
                    async with connector_class(config) as connector:
                        for event in events:
                            siem_event = SIEMEventPayload(
                                timestamp=event.timestamp,
                                event_id=event.id,
                                event_type=event.event_type,
                                severity=event.severity,
                                source=event.source_ip or "unknown",
                                user=event.user_id,
                                action=event.action,
                                outcome=event.outcome,
                                message=f"{event.action} {event.outcome}",
                                details=event.details or {},
                                risk_score=event.risk_score,
                                tags=[event.event_type, event.severity]
                            )
                            siem_events.append(siem_event)
                        
                        success = await connector.send_events(siem_events)
                        results[integration.id] = success
                        
                        # Update integration status
                        with get_sync_session() as db:
                            db.query(SIEMIntegration).filter(
                                SIEMIntegration.id == integration.id
                            ).update({
                                "last_sync": datetime.utcnow(),
                                "sync_status": "success" if success else "failed",
                                "error_count": 0 if success else integration.error_count + 1
                            })
                            db.commit()
                
                except Exception as e:
                    logger.error(f"Failed to forward events to {integration.id}: {str(e)}")
                    results[integration.id] = False
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to forward events to SIEM: {str(e)}")
            return {}

# Global SIEM integration manager
siem_manager = SIEMIntegrationManager()