"""
AI-Enhanced Security Operations Center Orchestrator
Coordinates all AI SOC components for comprehensive security operations
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json

from .ml_siem_engine import MLSIEMEngine, SecurityEvent, ThreatAlert
from .threat_correlation_system import ThreatCorrelationSystem, CorrelationResult
from .incident_response_orchestrator import IncidentResponseOrchestrator, SecurityIncident
from .behavioral_analytics_engine import BehavioralAnalyticsEngine, BehaviorAnomaly
from .predictive_security_analytics import PredictiveSecurityAnalytics, RiskForecast, ThreatPrediction

logger = logging.getLogger(__name__)

@dataclass
class SOCMetrics:
    events_processed: int = 0
    alerts_generated: int = 0
    incidents_created: int = 0
    incidents_resolved: int = 0
    false_positive_rate: float = 0.0
    mean_time_to_detection: float = 0.0
    mean_time_to_response: float = 0.0
    automation_rate: float = 0.0
    threat_hunting_hits: int = 0
    forecasts_accuracy: float = 0.0

@dataclass
class SOCDashboard:
    timestamp: datetime
    overall_risk_score: float
    active_incidents: int
    recent_alerts: List[ThreatAlert]
    top_threats: List[str]
    performance_metrics: SOCMetrics
    system_health: Dict[str, str]
    recommendations: List[str]

class AISOCOrchestrator:
    """
    Main orchestrator for AI-Enhanced Security Operations Center
    Coordinates all AI SOC components for comprehensive security operations
    """
    
    def __init__(self):
        # Initialize all AI SOC components
        self.ml_siem = MLSIEMEngine()
        self.correlation_system = ThreatCorrelationSystem()
        self.incident_orchestrator = IncidentResponseOrchestrator()
        self.behavioral_analytics = BehavioralAnalyticsEngine()
        self.predictive_analytics = PredictiveSecurityAnalytics()
        
        # SOC state
        self.is_initialized = False
        self.active_alerts: Dict[str, ThreatAlert] = {}
        self.active_incidents: Dict[str, SecurityIncident] = {}
        self.soc_metrics = SOCMetrics()
        
        # Performance tracking
        self.processing_start_times: Dict[str, datetime] = {}
        
        # Configuration
        self.config = {
            'auto_incident_creation': True,
            'auto_response_enabled': True,
            'threat_hunting_interval': 3600,  # 1 hour
            'forecasting_interval': 86400,    # 24 hours
            'alert_retention_days': 30,
            'max_concurrent_incidents': 100
        }
    
    async def initialize(self):
        """Initialize all AI SOC components"""
        try:
            logger.info("Initializing AI-Enhanced Security Operations Center...")
            
            # Initialize components in parallel
            init_tasks = [
                self.ml_siem.initialize(),
                self.behavioral_analytics.initialize(),
                self.predictive_analytics.initialize()
            ]
            
            await asyncio.gather(*init_tasks)
            
            # Start background tasks
            asyncio.create_task(self._threat_hunting_loop())
            asyncio.create_task(self._forecasting_loop())
            asyncio.create_task(self._metrics_update_loop())
            asyncio.create_task(self._cleanup_loop())
            
            self.is_initialized = True
            logger.info("AI SOC initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI SOC: {e}")
            raise
    
    async def process_security_event(self, event: SecurityEvent) -> Dict[str, Any]:
        """
        Process security event through the complete AI SOC pipeline
        """
        if not self.is_initialized:
            await self.initialize()
        
        processing_id = f"proc_{event.event_id}_{int(datetime.now().timestamp())}"
        self.processing_start_times[processing_id] = datetime.now()
        
        try:
            results = {
                'event_id': event.event_id,
                'processing_id': processing_id,
                'alerts': [],
                'correlations': [],
                'incidents': [],
                'anomalies': [],
                'actions_taken': []
            }
            
            # Step 1: ML SIEM Analysis
            alert = await self.ml_siem.analyze_event(event)
            if alert:
                self.active_alerts[alert.alert_id] = alert
                results['alerts'].append(alert)
                results['actions_taken'].append('ML SIEM alert generated')
                self.soc_metrics.alerts_generated += 1
            
            # Step 2: Threat Correlation
            correlations = await self.correlation_system.add_event(event)
            if correlations:
                results['correlations'].extend(correlations)
                results['actions_taken'].append(f'{len(correlations)} correlations found')
            
            # Step 3: Behavioral Analysis
            anomaly = await self.behavioral_analytics.analyze_user_behavior(event)
            if anomaly:
                results['anomalies'].append(anomaly)
                results['actions_taken'].append('Behavioral anomaly detected')
            
            # Step 4: Incident Creation (if configured)
            if self.config['auto_incident_creation']:
                incidents = await self._create_incidents_from_results(alert, correlations, anomaly)
                if incidents:
                    results['incidents'].extend(incidents)
                    results['actions_taken'].append(f'{len(incidents)} incidents created')
            
            # Step 5: Automated Response (if configured)
            if self.config['auto_response_enabled'] and results['incidents']:
                for incident in results['incidents']:
                    response = await self.incident_orchestrator.execute_incident_response(incident)
                    results['actions_taken'].append(f'Automated response executed for incident {incident.incident_id}')
            
            # Update metrics
            self.soc_metrics.events_processed += 1
            self._update_processing_metrics(processing_id)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing security event {event.event_id}: {e}")
            return {
                'event_id': event.event_id,
                'processing_id': processing_id,
                'error': str(e),
                'actions_taken': ['Error occurred during processing']
            }
    
    async def _create_incidents_from_results(self, alert: Optional[ThreatAlert], 
                                           correlations: List[CorrelationResult],
                                           anomaly: Optional[BehaviorAnomaly]) -> List[SecurityIncident]:
        """Create incidents from analysis results"""
        incidents = []
        
        # Create incident from high-severity alert
        if alert and alert.severity in ['HIGH', 'CRITICAL']:
            incident = await self.incident_orchestrator.create_incident_from_alert(alert)
            incidents.append(incident)
            self.active_incidents[incident.incident_id] = incident
            self.soc_metrics.incidents_created += 1
        
        # Create incidents from correlations
        for correlation in correlations:
            if correlation.rule.severity in ['HIGH', 'CRITICAL']:
                incident = await self.incident_orchestrator.create_incident_from_correlation(correlation)
                incidents.append(incident)
                self.active_incidents[incident.incident_id] = incident
                self.soc_metrics.incidents_created += 1
        
        # Create incident from high-severity anomaly
        if anomaly and anomaly.severity in ['HIGH', 'CRITICAL']:
            # Convert anomaly to alert for incident creation
            alert_from_anomaly = ThreatAlert(
                alert_id=f"anomaly_{anomaly.anomaly_id}",
                event=anomaly.event,
                threat_type=f"Behavioral Anomaly: {anomaly.anomaly_type}",
                severity=anomaly.severity,
                confidence=anomaly.confidence,
                recommended_actions=[
                    "Investigate user behavior",
                    "Review recent activities",
                    "Consider access restrictions"
                ],
                created_at=anomaly.detected_at,
                correlation_events=[]
            )
            
            incident = await self.incident_orchestrator.create_incident_from_alert(alert_from_anomaly)
            incidents.append(incident)
            self.active_incidents[incident.incident_id] = incident
            self.soc_metrics.incidents_created += 1
        
        return incidents
    
    async def _threat_hunting_loop(self):
        """Background threat hunting loop"""
        while True:
            try:
                await asyncio.sleep(self.config['threat_hunting_interval'])
                
                # Execute threat hunting
                hunting_results = await self.behavioral_analytics.execute_threat_hunting()
                
                if hunting_results:
                    self.soc_metrics.threat_hunting_hits += len(hunting_results)
                    logger.info(f"Threat hunting found {len(hunting_results)} potential threats")
                    
                    # Create incidents from high-confidence hunting results
                    for result in hunting_results:
                        if result.get('confidence', 0) > 0.8:
                            # Create synthetic alert for hunting result
                            hunting_alert = ThreatAlert(
                                alert_id=f"hunt_{result.get('query_id')}_{int(datetime.now().timestamp())}",
                                event=SecurityEvent(
                                    event_id=f"hunt_event_{int(datetime.now().timestamp())}",
                                    timestamp=datetime.now(),
                                    event_type='ANOMALOUS_BEHAVIOR',
                                    source_ip='hunting_system',
                                    user_id=None,
                                    resource='threat_hunting',
                                    raw_data=result,
                                    risk_score=0.8
                                ),
                                threat_type=f"Threat Hunting: {result.get('query_name', 'Unknown')}",
                                severity='HIGH',
                                confidence=result.get('confidence', 0.8),
                                recommended_actions=['Investigate hunting findings', 'Validate threat indicators'],
                                created_at=datetime.now(),
                                correlation_events=[]
                            )
                            
                            incident = await self.incident_orchestrator.create_incident_from_alert(hunting_alert)
                            self.active_incidents[incident.incident_id] = incident
                
            except Exception as e:
                logger.error(f"Error in threat hunting loop: {e}")
    
    async def _forecasting_loop(self):
        """Background forecasting loop"""
        while True:
            try:
                await asyncio.sleep(self.config['forecasting_interval'])
                
                # Generate risk forecasts for high-risk entities
                high_risk_entities = await self._identify_high_risk_entities()
                
                for entity_id in high_risk_entities:
                    forecast = await self.predictive_analytics.generate_risk_forecast(
                        'user', entity_id, 30
                    )
                    
                    # Create preventive incidents for very high predicted risk
                    if forecast.predicted_risk_score > 0.8:
                        preventive_alert = ThreatAlert(
                            alert_id=f"forecast_{forecast.forecast_id}",
                            event=SecurityEvent(
                                event_id=f"forecast_event_{int(datetime.now().timestamp())}",
                                timestamp=datetime.now(),
                                event_type='PREDICTIVE_ALERT',
                                source_ip='forecasting_system',
                                user_id=entity_id,
                                resource='risk_forecast',
                                raw_data={'forecast': forecast.__dict__},
                                risk_score=forecast.predicted_risk_score
                            ),
                            threat_type="High Risk Forecast",
                            severity='HIGH',
                            confidence=0.7,
                            recommended_actions=forecast.recommendations,
                            created_at=datetime.now(),
                            correlation_events=[]
                        )
                        
                        incident = await self.incident_orchestrator.create_incident_from_alert(preventive_alert)
                        self.active_incidents[incident.incident_id] = incident
                
                # Generate threat predictions
                threat_types = ['data_breach', 'insider_threat', 'malware', 'phishing']
                for threat_type in threat_types:
                    prediction = await self.predictive_analytics.predict_threat_likelihood(threat_type)
                    
                    if prediction.probability > 0.7:
                        logger.warning(f"High probability {threat_type} predicted: {prediction.probability:.2f}")
                
            except Exception as e:
                logger.error(f"Error in forecasting loop: {e}")
    
    async def _identify_high_risk_entities(self) -> List[str]:
        """Identify entities with highest risk scores"""
        # This would typically query the behavioral analytics for high-risk users
        # For demo, return a sample list
        return ['user_1', 'user_5', 'user_12', 'user_23', 'user_34']
    
    async def _metrics_update_loop(self):
        """Background metrics update loop"""
        while True:
            try:
                await asyncio.sleep(300)  # Update every 5 minutes
                
                # Update SOC metrics
                await self._update_soc_metrics()
                
            except Exception as e:
                logger.error(f"Error in metrics update loop: {e}")
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while True:
            try:
                await asyncio.sleep(3600)  # Cleanup every hour
                
                # Clean up old alerts
                cutoff_time = datetime.now() - timedelta(days=self.config['alert_retention_days'])
                old_alerts = [alert_id for alert_id, alert in self.active_alerts.items() 
                             if alert.created_at < cutoff_time]
                
                for alert_id in old_alerts:
                    del self.active_alerts[alert_id]
                
                # Clean up resolved incidents
                resolved_incidents = [incident_id for incident_id, incident in self.active_incidents.items()
                                    if incident.status == 'CLOSED']
                
                for incident_id in resolved_incidents:
                    del self.active_incidents[incident_id]
                
                logger.info(f"Cleaned up {len(old_alerts)} old alerts and {len(resolved_incidents)} resolved incidents")
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def _update_processing_metrics(self, processing_id: str):
        """Update processing time metrics"""
        if processing_id in self.processing_start_times:
            processing_time = (datetime.now() - self.processing_start_times[processing_id]).total_seconds()
            
            # Update mean time to detection (simplified)
            if self.soc_metrics.mean_time_to_detection == 0:
                self.soc_metrics.mean_time_to_detection = processing_time
            else:
                self.soc_metrics.mean_time_to_detection = (
                    self.soc_metrics.mean_time_to_detection * 0.9 + processing_time * 0.1
                )
            
            del self.processing_start_times[processing_id]
    
    async def _update_soc_metrics(self):
        """Update comprehensive SOC metrics"""
        # Get metrics from all components
        ml_siem_metrics = self.ml_siem.get_performance_metrics()
        correlation_metrics = self.correlation_system.get_performance_metrics()
        incident_metrics = self.incident_orchestrator.get_performance_metrics()
        behavioral_metrics = self.behavioral_analytics.get_performance_metrics()
        predictive_metrics = self.predictive_analytics.get_performance_metrics()
        
        # Update SOC metrics
        self.soc_metrics.false_positive_rate = ml_siem_metrics.get('false_positive_rate', 0.0)
        self.soc_metrics.automation_rate = incident_metrics.get('auto_resolution_rate', 0.0)
        self.soc_metrics.forecasts_accuracy = predictive_metrics.get('forecast_accuracy', 0.0)
        
        # Count resolved incidents
        resolved_count = sum(1 for incident in self.active_incidents.values() 
                           if incident.status in ['RECOVERED', 'CLOSED'])
        self.soc_metrics.incidents_resolved = resolved_count
    
    async def get_soc_dashboard(self) -> SOCDashboard:
        """Generate comprehensive SOC dashboard"""
        # Calculate overall risk score
        overall_risk = await self._calculate_overall_risk_score()
        
        # Get recent alerts
        recent_alerts = sorted(
            [alert for alert in self.active_alerts.values()],
            key=lambda a: a.created_at,
            reverse=True
        )[:10]
        
        # Identify top threats
        top_threats = await self._identify_top_threats()
        
        # Get system health
        system_health = await self._get_system_health()
        
        # Generate recommendations
        recommendations = await self._generate_soc_recommendations()
        
        return SOCDashboard(
            timestamp=datetime.now(),
            overall_risk_score=overall_risk,
            active_incidents=len(self.active_incidents),
            recent_alerts=recent_alerts,
            top_threats=top_threats,
            performance_metrics=self.soc_metrics,
            system_health=system_health,
            recommendations=recommendations
        )
    
    async def _calculate_overall_risk_score(self) -> float:
        """Calculate overall organizational risk score"""
        risk_factors = []
        
        # Factor in active incidents
        if self.active_incidents:
            incident_risk = len([i for i in self.active_incidents.values() 
                               if i.severity in ['HIGH', 'CRITICAL']]) / len(self.active_incidents)
            risk_factors.append(incident_risk * 0.4)
        
        # Factor in recent alerts
        if self.active_alerts:
            alert_risk = len([a for a in self.active_alerts.values() 
                            if a.severity in ['HIGH', 'CRITICAL']]) / len(self.active_alerts)
            risk_factors.append(alert_risk * 0.3)
        
        # Factor in false positive rate (inverse)
        fp_risk = max(0, 1 - self.soc_metrics.false_positive_rate * 10)
        risk_factors.append(fp_risk * 0.2)
        
        # Factor in threat hunting hits
        hunting_risk = min(1.0, self.soc_metrics.threat_hunting_hits / 10) * 0.1
        risk_factors.append(hunting_risk)
        
        return sum(risk_factors) if risk_factors else 0.3
    
    async def _identify_top_threats(self) -> List[str]:
        """Identify top current threats"""
        threat_counts = {}
        
        # Count threat types from active alerts
        for alert in self.active_alerts.values():
            threat_type = alert.threat_type
            threat_counts[threat_type] = threat_counts.get(threat_type, 0) + 1
        
        # Sort by frequency
        top_threats = sorted(threat_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [threat for threat, count in top_threats[:5]]
    
    async def _get_system_health(self) -> Dict[str, str]:
        """Get system health status"""
        return {
            'ml_siem': 'healthy' if self.ml_siem.is_trained else 'degraded',
            'correlation_system': 'healthy',
            'incident_orchestrator': 'healthy',
            'behavioral_analytics': 'healthy',
            'predictive_analytics': 'healthy' if self.predictive_analytics.is_trained else 'degraded',
            'overall': 'healthy'
        }
    
    async def _generate_soc_recommendations(self) -> List[str]:
        """Generate SOC operational recommendations"""
        recommendations = []
        
        # Check false positive rate
        if self.soc_metrics.false_positive_rate > 0.1:
            recommendations.append("Consider tuning ML models to reduce false positives")
        
        # Check automation rate
        if self.soc_metrics.automation_rate < 0.5:
            recommendations.append("Review incident response playbooks for automation opportunities")
        
        # Check active incidents
        if len(self.active_incidents) > 50:
            recommendations.append("High incident volume - consider additional analyst resources")
        
        # Check threat hunting
        if self.soc_metrics.threat_hunting_hits == 0:
            recommendations.append("Review threat hunting queries for effectiveness")
        
        # Check forecasting accuracy
        if self.soc_metrics.forecasts_accuracy < 0.7:
            recommendations.append("Retrain predictive models with recent data")
        
        if not recommendations:
            recommendations.append("SOC operating within normal parameters")
        
        return recommendations
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from all SOC components"""
        return {
            'soc_metrics': {
                'events_processed': self.soc_metrics.events_processed,
                'alerts_generated': self.soc_metrics.alerts_generated,
                'incidents_created': self.soc_metrics.incidents_created,
                'incidents_resolved': self.soc_metrics.incidents_resolved,
                'false_positive_rate': self.soc_metrics.false_positive_rate,
                'automation_rate': self.soc_metrics.automation_rate,
                'threat_hunting_hits': self.soc_metrics.threat_hunting_hits
            },
            'ml_siem': self.ml_siem.get_performance_metrics(),
            'correlation_system': self.correlation_system.get_performance_metrics(),
            'incident_orchestrator': self.incident_orchestrator.get_performance_metrics(),
            'behavioral_analytics': self.behavioral_analytics.get_performance_metrics(),
            'predictive_analytics': self.predictive_analytics.get_performance_metrics(),
            'active_alerts': len(self.active_alerts),
            'active_incidents': len(self.active_incidents)
        }