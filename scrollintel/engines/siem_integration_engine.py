"""
SIEM Integration Engine
Handles integration with various SIEM platforms for security event forwarding
"""

import json
import requests
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session
from scrollintel.models.security_audit_models import AuditLog, SIEMIntegration, SecurityAlert
from scrollintel.core.config import get_database_session
import logging

logger = logging.getLogger(__name__)

class SIEMIntegrationEngine:
    """Engine for integrating with SIEM platforms"""
    
    def __init__(self):
        self.supported_platforms = {
            'splunk': SplunkConnector,
            'elk': ELKConnector,
            'qradar': QRadarConnector,
            'sentinel': SentinelConnector,
            'sumo_logic': SumoLogicConnector
        }
        self.active_integrations = {}
        self._load_integrations()
    
    def _load_integrations(self):
        """Load active SIEM integrations from database"""
        try:
            with get_database_session() as db:
                integrations = db.query(SIEMIntegration).filter(
                    SIEMIntegration.is_active == True
                ).all()
                
                for integration in integrations:
                    if integration.platform_type in self.supported_platforms:
                        connector_class = self.supported_platforms[integration.platform_type]
                        self.active_integrations[integration.id] = connector_class(integration)
                        
        except Exception as e:
            logger.error(f"Error loading SIEM integrations: {str(e)}")
    
    async def send_security_event(self, audit_log: AuditLog):
        """Send security event to all active SIEM platforms"""
        tasks = []
        for integration_id, connector in self.active_integrations.items():
            task = asyncio.create_task(
                connector.send_event(audit_log)
            )
            tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results
        return []
    
    async def batch_send_events(self, events: List[AuditLog]):
        """Send multiple events in batch to SIEM platforms"""
        tasks = []
        for integration_id, connector in self.active_integrations.items():
            task = asyncio.create_task(
                connector.batch_send_events(events)
            )
            tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results
        return []
    
    def create_integration(self, config: Dict[str, Any]) -> int:
        """Create new SIEM integration"""
        try:
            with get_database_session() as db:
                integration = SIEMIntegration(
                    name=config['name'],
                    platform_type=config['platform_type'],
                    endpoint_url=config['endpoint_url'],
                    authentication_config=config['authentication_config'],
                    event_filters=config.get('event_filters', {}),
                    batch_size=config.get('batch_size', 100),
                    send_interval=config.get('send_interval', 60)
                )
                
                db.add(integration)
                db.commit()
                db.refresh(integration)
                
                # Initialize connector
                if integration.platform_type in self.supported_platforms:
                    connector_class = self.supported_platforms[integration.platform_type]
                    self.active_integrations[integration.id] = connector_class(integration)
                
                return integration.id
                
        except Exception as e:
            logger.error(f"Error creating SIEM integration: {str(e)}")
            raise

class BaseSIEMConnector:
    """Base class for SIEM connectors"""
    
    def __init__(self, integration: SIEMIntegration):
        self.integration = integration
        self.config = integration.authentication_config
        self.endpoint_url = integration.endpoint_url
        self.event_filters = integration.event_filters or {}
    
    def should_send_event(self, audit_log: AuditLog) -> bool:
        """Check if event should be sent based on filters"""
        if not self.event_filters:
            return True
        
        # Apply severity filter
        if 'min_severity' in self.event_filters:
            severity_levels = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
            min_level = severity_levels.get(self.event_filters['min_severity'], 1)
            event_level = severity_levels.get(audit_log.severity, 1)
            if event_level < min_level:
                return False
        
        # Apply event type filter
        if 'event_types' in self.event_filters:
            if audit_log.event_type not in self.event_filters['event_types']:
                return False
        
        return True
    
    def format_event(self, audit_log: AuditLog) -> Dict[str, Any]:
        """Format audit log for SIEM platform"""
        return {
            'event_id': audit_log.event_id,
            'timestamp': audit_log.timestamp.isoformat(),
            'event_type': audit_log.event_type,
            'severity': audit_log.severity,
            'source_system': audit_log.source_system,
            'user_id': audit_log.user_id,
            'ip_address': audit_log.ip_address,
            'action': audit_log.action_performed,
            'result': audit_log.result_status,
            'details': audit_log.details,
            'risk_score': audit_log.risk_score
        }
    
    async def send_event(self, audit_log: AuditLog):
        """Send single event to SIEM platform"""
        raise NotImplementedError
    
    async def batch_send_events(self, events: List[AuditLog]):
        """Send multiple events to SIEM platform"""
        raise NotImplementedError

class SplunkConnector(BaseSIEMConnector):
    """Splunk SIEM connector"""
    
    async def send_event(self, audit_log: AuditLog):
        """Send event to Splunk"""
        if not self.should_send_event(audit_log):
            return
        
        try:
            event_data = self.format_event(audit_log)
            
            headers = {
                'Authorization': f"Splunk {self.config['token']}",
                'Content-Type': 'application/json'
            }
            
            payload = {
                'event': event_data,
                'source': 'scrollintel',
                'sourcetype': 'security_audit'
            }
            
            async with requests.post(
                f"{self.endpoint_url}/services/collector/event",
                headers=headers,
                json=payload,
                timeout=30
            ) as response:
                response.raise_for_status()
                return True
                
        except Exception as e:
            logger.error(f"Error sending event to Splunk: {str(e)}")
            return False
    
    async def batch_send_events(self, events: List[AuditLog]):
        """Send batch of events to Splunk"""
        filtered_events = [e for e in events if self.should_send_event(e)]
        if not filtered_events:
            return
        
        try:
            headers = {
                'Authorization': f"Splunk {self.config['token']}",
                'Content-Type': 'application/json'
            }
            
            batch_data = []
            for event in filtered_events:
                event_data = self.format_event(event)
                batch_data.append({
                    'event': event_data,
                    'source': 'scrollintel',
                    'sourcetype': 'security_audit'
                })
            
            # Send in chunks
            chunk_size = self.integration.batch_size
            for i in range(0, len(batch_data), chunk_size):
                chunk = batch_data[i:i + chunk_size]
                
                async with requests.post(
                    f"{self.endpoint_url}/services/collector/event",
                    headers=headers,
                    json=chunk,
                    timeout=60
                ) as response:
                    response.raise_for_status()
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending batch events to Splunk: {str(e)}")
            return False

class ELKConnector(BaseSIEMConnector):
    """Elasticsearch/ELK Stack connector"""
    
    async def send_event(self, audit_log: AuditLog):
        """Send event to Elasticsearch"""
        if not self.should_send_event(audit_log):
            return
        
        try:
            event_data = self.format_event(audit_log)
            
            headers = {
                'Content-Type': 'application/json'
            }
            
            # Add authentication if configured
            if 'username' in self.config and 'password' in self.config:
                import base64
                credentials = base64.b64encode(
                    f"{self.config['username']}:{self.config['password']}".encode()
                ).decode()
                headers['Authorization'] = f'Basic {credentials}'
            
            index_name = f"scrollintel-security-{datetime.now().strftime('%Y-%m')}"
            
            async with requests.post(
                f"{self.endpoint_url}/{index_name}/_doc",
                headers=headers,
                json=event_data,
                timeout=30
            ) as response:
                response.raise_for_status()
                return True
                
        except Exception as e:
            logger.error(f"Error sending event to ELK: {str(e)}")
            return False
    
    async def batch_send_events(self, events: List[AuditLog]):
        """Send batch of events to Elasticsearch using bulk API"""
        filtered_events = [e for e in events if self.should_send_event(e)]
        if not filtered_events:
            return
        
        try:
            headers = {
                'Content-Type': 'application/x-ndjson'
            }
            
            # Add authentication if configured
            if 'username' in self.config and 'password' in self.config:
                import base64
                credentials = base64.b64encode(
                    f"{self.config['username']}:{self.config['password']}".encode()
                ).decode()
                headers['Authorization'] = f'Basic {credentials}'
            
            index_name = f"scrollintel-security-{datetime.now().strftime('%Y-%m')}"
            
            # Build bulk request body
            bulk_body = []
            for event in filtered_events:
                event_data = self.format_event(event)
                
                # Index action
                bulk_body.append(json.dumps({"index": {"_index": index_name}}))
                # Document
                bulk_body.append(json.dumps(event_data))
            
            bulk_data = '\n'.join(bulk_body) + '\n'
            
            async with requests.post(
                f"{self.endpoint_url}/_bulk",
                headers=headers,
                data=bulk_data,
                timeout=60
            ) as response:
                response.raise_for_status()
                return True
                
        except Exception as e:
            logger.error(f"Error sending batch events to ELK: {str(e)}")
            return False

class QRadarConnector(BaseSIEMConnector):
    """IBM QRadar connector"""
    
    async def send_event(self, audit_log: AuditLog):
        """Send event to QRadar"""
        if not self.should_send_event(audit_log):
            return
        
        try:
            event_data = self.format_event(audit_log)
            
            headers = {
                'SEC': self.config['sec_token'],
                'Content-Type': 'application/json'
            }
            
            async with requests.post(
                f"{self.endpoint_url}/api/siem/events",
                headers=headers,
                json=event_data,
                timeout=30
            ) as response:
                response.raise_for_status()
                return True
                
        except Exception as e:
            logger.error(f"Error sending event to QRadar: {str(e)}")
            return False
    
    async def batch_send_events(self, events: List[AuditLog]):
        """Send batch of events to QRadar"""
        # QRadar typically handles events individually
        results = []
        for event in events:
            result = await self.send_event(event)
            results.append(result)
        return all(results)

class SentinelConnector(BaseSIEMConnector):
    """Microsoft Sentinel connector"""
    
    async def send_event(self, audit_log: AuditLog):
        """Send event to Microsoft Sentinel"""
        if not self.should_send_event(audit_log):
            return
        
        try:
            event_data = self.format_event(audit_log)
            
            headers = {
                'Authorization': f"Bearer {self.config['access_token']}",
                'Content-Type': 'application/json',
                'Log-Type': 'ScrollIntelSecurity'
            }
            
            async with requests.post(
                f"{self.endpoint_url}/api/logs",
                headers=headers,
                json=[event_data],
                timeout=30
            ) as response:
                response.raise_for_status()
                return True
                
        except Exception as e:
            logger.error(f"Error sending event to Sentinel: {str(e)}")
            return False
    
    async def batch_send_events(self, events: List[AuditLog]):
        """Send batch of events to Microsoft Sentinel"""
        filtered_events = [e for e in events if self.should_send_event(e)]
        if not filtered_events:
            return
        
        try:
            headers = {
                'Authorization': f"Bearer {self.config['access_token']}",
                'Content-Type': 'application/json',
                'Log-Type': 'ScrollIntelSecurity'
            }
            
            batch_data = [self.format_event(event) for event in filtered_events]
            
            async with requests.post(
                f"{self.endpoint_url}/api/logs",
                headers=headers,
                json=batch_data,
                timeout=60
            ) as response:
                response.raise_for_status()
                return True
                
        except Exception as e:
            logger.error(f"Error sending batch events to Sentinel: {str(e)}")
            return False

class SumoLogicConnector(BaseSIEMConnector):
    """Sumo Logic connector"""
    
    async def send_event(self, audit_log: AuditLog):
        """Send event to Sumo Logic"""
        if not self.should_send_event(audit_log):
            return
        
        try:
            event_data = self.format_event(audit_log)
            
            headers = {
                'Content-Type': 'application/json'
            }
            
            # Use HTTP collector endpoint
            async with requests.post(
                self.endpoint_url,  # This should be the collector URL
                headers=headers,
                json=event_data,
                timeout=30
            ) as response:
                response.raise_for_status()
                return True
                
        except Exception as e:
            logger.error(f"Error sending event to Sumo Logic: {str(e)}")
            return False
    
    async def batch_send_events(self, events: List[AuditLog]):
        """Send batch of events to Sumo Logic"""
        filtered_events = [e for e in events if self.should_send_event(e)]
        if not filtered_events:
            return
        
        try:
            headers = {
                'Content-Type': 'application/json'
            }
            
            batch_data = [self.format_event(event) for event in filtered_events]
            
            async with requests.post(
                self.endpoint_url,
                headers=headers,
                json=batch_data,
                timeout=60
            ) as response:
                response.raise_for_status()
                return True
                
        except Exception as e:
            logger.error(f"Error sending batch events to Sumo Logic: {str(e)}")
            return False