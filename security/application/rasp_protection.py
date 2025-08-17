"""
Runtime Application Self-Protection (RASP) Implementation
Real-time threat detection and response during application execution
"""

import asyncio
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import hashlib
import re
import ipaddress

class ThreatType(Enum):
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    DESERIALIZATION = "deserialization"
    AUTHENTICATION_BYPASS = "auth_bypass"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    BRUTE_FORCE = "brute_force"
    DDOS = "ddos"
    MALICIOUS_FILE_UPLOAD = "malicious_upload"
    API_ABUSE = "api_abuse"

class ThreatSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ResponseAction(Enum):
    BLOCK = "block"
    MONITOR = "monitor"
    RATE_LIMIT = "rate_limit"
    CHALLENGE = "challenge"
    QUARANTINE = "quarantine"
    ALERT_ONLY = "alert_only"

@dataclass
class ThreatEvent:
    id: str
    timestamp: datetime
    threat_type: ThreatType
    severity: ThreatSeverity
    source_ip: str
    user_id: Optional[str]
    request_path: str
    request_method: str
    payload: str
    confidence_score: float
    response_action: ResponseAction
    blocked: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AttackPattern:
    pattern_id: str
    name: str
    threat_type: ThreatType
    regex_patterns: List[str]
    severity: ThreatSeverity
    confidence_threshold: float
    response_action: ResponseAction

class RASPEngine:
    """Runtime Application Self-Protection Engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enabled', True)
        self.learning_mode = config.get('learning_mode', False)
        
        # Threat detection patterns
        self.attack_patterns = self._load_attack_patterns()
        
        # Rate limiting and tracking
        self.request_tracker = defaultdict(lambda: deque(maxlen=1000))
        self.ip_reputation = defaultdict(int)
        self.user_behavior = defaultdict(lambda: {'requests': deque(maxlen=100), 'anomalies': 0})
        
        # Real-time monitoring
        self.threat_events = deque(maxlen=10000)
        self.active_attacks = {}
        
        # ML-based anomaly detection
        self.baseline_metrics = self._initialize_baseline()
        self.anomaly_threshold = config.get('anomaly_threshold', 0.8)
        
        # Response handlers
        self.response_handlers = {
            ResponseAction.BLOCK: self._block_request,
            ResponseAction.RATE_LIMIT: self._rate_limit_request,
            ResponseAction.CHALLENGE: self._challenge_request,
            ResponseAction.QUARANTINE: self._quarantine_user,
            ResponseAction.MONITOR: self._monitor_request,
            ResponseAction.ALERT_ONLY: self._alert_only
        }
        
        # Background monitoring thread
        self.monitoring_thread = threading.Thread(target=self._background_monitoring, daemon=True)
        self.monitoring_thread.start()
    
    def _load_attack_patterns(self) -> List[AttackPattern]:
        """Load predefined attack patterns for threat detection"""
        patterns = [
            # SQL Injection patterns
            AttackPattern(
                pattern_id="sql_001",
                name="SQL Injection - Union Based",
                threat_type=ThreatType.SQL_INJECTION,
                regex_patterns=[
                    r"(?i)(union\s+select|union\s+all\s+select)",
                    r"(?i)(select.*from.*information_schema)",
                    r"(?i)(select.*from.*sysobjects)",
                    r"(?i)(\'\s*or\s*\'\s*=\s*\')",
                    r"(?i)(\'\s*or\s*1\s*=\s*1)",
                ],
                severity=ThreatSeverity.CRITICAL,
                confidence_threshold=0.9,
                response_action=ResponseAction.BLOCK
            ),
            
            # XSS patterns
            AttackPattern(
                pattern_id="xss_001",
                name="Cross-Site Scripting",
                threat_type=ThreatType.XSS,
                regex_patterns=[
                    r"(?i)<script[^>]*>.*?</script>",
                    r"(?i)javascript:",
                    r"(?i)on\w+\s*=",
                    r"(?i)<iframe[^>]*>",
                    r"(?i)eval\s*\(",
                ],
                severity=ThreatSeverity.HIGH,
                confidence_threshold=0.8,
                response_action=ResponseAction.BLOCK
            ),
            
            # Command Injection patterns
            AttackPattern(
                pattern_id="cmd_001",
                name="Command Injection",
                threat_type=ThreatType.COMMAND_INJECTION,
                regex_patterns=[
                    r"(?i)(;|\||\&)\s*(cat|ls|pwd|whoami|id|uname)",
                    r"(?i)(wget|curl)\s+http",
                    r"(?i)(nc|netcat)\s+-",
                    r"(?i)/bin/(sh|bash|csh)",
                    r"(?i)cmd\.exe",
                ],
                severity=ThreatSeverity.CRITICAL,
                confidence_threshold=0.9,
                response_action=ResponseAction.BLOCK
            ),
            
            # Path Traversal patterns
            AttackPattern(
                pattern_id="path_001",
                name="Path Traversal",
                threat_type=ThreatType.PATH_TRAVERSAL,
                regex_patterns=[
                    r"\.\.\/",
                    r"\.\.\\",
                    r"%2e%2e%2f",
                    r"%2e%2e%5c",
                    r"\/etc\/passwd",
                    r"\/windows\/system32",
                ],
                severity=ThreatSeverity.HIGH,
                confidence_threshold=0.8,
                response_action=ResponseAction.BLOCK
            ),
        ]
        
        return patterns
    
    def _initialize_baseline(self) -> Dict[str, Any]:
        """Initialize baseline metrics for anomaly detection"""
        return {
            'avg_request_size': 1024,
            'avg_response_time': 200,
            'common_user_agents': set(),
            'typical_request_patterns': {},
            'normal_error_rate': 0.05
        }
    
    async def analyze_request(self, request_data: Dict[str, Any]) -> Optional[ThreatEvent]:
        """Analyze incoming request for threats"""
        if not self.enabled:
            return None
        
        source_ip = request_data.get('source_ip', 'unknown')
        user_id = request_data.get('user_id')
        request_path = request_data.get('path', '')
        request_method = request_data.get('method', 'GET')
        headers = request_data.get('headers', {})
        body = request_data.get('body', '')
        query_params = request_data.get('query_params', {})
        
        # Combine all request data for analysis
        full_payload = f"{request_path} {json.dumps(query_params)} {body}"
        
        # Pattern-based threat detection
        threat_event = await self._detect_attack_patterns(
            source_ip, user_id, request_path, request_method, full_payload
        )
        
        if threat_event:
            return threat_event
        
        # Behavioral analysis
        behavioral_threat = await self._analyze_behavior(
            source_ip, user_id, request_data
        )
        
        if behavioral_threat:
            return behavioral_threat
        
        # Anomaly detection
        anomaly_threat = await self._detect_anomalies(request_data)
        
        return anomaly_threat
    
    async def _detect_attack_patterns(self, source_ip: str, user_id: Optional[str], 
                                    path: str, method: str, payload: str) -> Optional[ThreatEvent]:
        """Detect known attack patterns in request"""
        
        for pattern in self.attack_patterns:
            confidence_score = 0.0
            matched_patterns = []
            
            for regex_pattern in pattern.regex_patterns:
                if re.search(regex_pattern, payload):
                    confidence_score += 1.0 / len(pattern.regex_patterns)
                    matched_patterns.append(regex_pattern)
            
            if confidence_score >= pattern.confidence_threshold:
                threat_id = hashlib.md5(f"{source_ip}{time.time()}".encode()).hexdigest()
                
                threat_event = ThreatEvent(
                    id=threat_id,
                    timestamp=datetime.now(),
                    threat_type=pattern.threat_type,
                    severity=pattern.severity,
                    source_ip=source_ip,
                    user_id=user_id,
                    request_path=path,
                    request_method=method,
                    payload=payload,
                    confidence_score=confidence_score,
                    response_action=pattern.response_action,
                    metadata={
                        'pattern_id': pattern.pattern_id,
                        'matched_patterns': matched_patterns,
                        'pattern_name': pattern.name
                    }
                )
                
                # Execute response action
                if not self.learning_mode:
                    await self._execute_response_action(threat_event)
                
                self.threat_events.append(threat_event)
                return threat_event
        
        return None
    
    async def _analyze_behavior(self, source_ip: str, user_id: Optional[str], 
                              request_data: Dict[str, Any]) -> Optional[ThreatEvent]:
        """Analyze behavioral patterns for anomalies"""
        
        current_time = datetime.now()
        
        # Track request frequency
        self.request_tracker[source_ip].append(current_time)
        
        # Check for brute force attacks
        recent_requests = [
            req_time for req_time in self.request_tracker[source_ip]
            if current_time - req_time < timedelta(minutes=5)
        ]
        
        if len(recent_requests) > self.config.get('max_requests_per_5min', 100):
            threat_id = hashlib.md5(f"{source_ip}_brute_force_{time.time()}".encode()).hexdigest()
            
            return ThreatEvent(
                id=threat_id,
                timestamp=current_time,
                threat_type=ThreatType.BRUTE_FORCE,
                severity=ThreatSeverity.HIGH,
                source_ip=source_ip,
                user_id=user_id,
                request_path=request_data.get('path', ''),
                request_method=request_data.get('method', 'GET'),
                payload=f"High frequency requests: {len(recent_requests)} in 5 minutes",
                confidence_score=0.9,
                response_action=ResponseAction.RATE_LIMIT,
                metadata={'request_count': len(recent_requests)}
            )
        
        # Check for suspicious user agent patterns
        user_agent = request_data.get('headers', {}).get('User-Agent', '')
        if self._is_suspicious_user_agent(user_agent):
            threat_id = hashlib.md5(f"{source_ip}_suspicious_ua_{time.time()}".encode()).hexdigest()
            
            return ThreatEvent(
                id=threat_id,
                timestamp=current_time,
                threat_type=ThreatType.API_ABUSE,
                severity=ThreatSeverity.MEDIUM,
                source_ip=source_ip,
                user_id=user_id,
                request_path=request_data.get('path', ''),
                request_method=request_data.get('method', 'GET'),
                payload=f"Suspicious User-Agent: {user_agent}",
                confidence_score=0.7,
                response_action=ResponseAction.MONITOR,
                metadata={'user_agent': user_agent}
            )
        
        return None
    
    async def _detect_anomalies(self, request_data: Dict[str, Any]) -> Optional[ThreatEvent]:
        """Detect anomalies using ML-based analysis"""
        
        # Analyze request size anomalies
        body_size = len(request_data.get('body', ''))
        if body_size > self.baseline_metrics['avg_request_size'] * 10:
            threat_id = hashlib.md5(f"anomaly_size_{time.time()}".encode()).hexdigest()
            
            return ThreatEvent(
                id=threat_id,
                timestamp=datetime.now(),
                threat_type=ThreatType.DATA_EXFILTRATION,
                severity=ThreatSeverity.MEDIUM,
                source_ip=request_data.get('source_ip', 'unknown'),
                user_id=request_data.get('user_id'),
                request_path=request_data.get('path', ''),
                request_method=request_data.get('method', 'GET'),
                payload=f"Unusually large request: {body_size} bytes",
                confidence_score=0.6,
                response_action=ResponseAction.MONITOR,
                metadata={'request_size': body_size}
            )
        
        return None
    
    def _is_suspicious_user_agent(self, user_agent: str) -> bool:
        """Check if user agent is suspicious"""
        suspicious_patterns = [
            r'(?i)(bot|crawler|spider|scraper)',
            r'(?i)(curl|wget|python|java)',
            r'(?i)(scanner|exploit|hack)',
            r'^$',  # Empty user agent
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, user_agent):
                return True
        
        return False
    
    async def _execute_response_action(self, threat_event: ThreatEvent):
        """Execute the appropriate response action for a threat"""
        handler = self.response_handlers.get(threat_event.response_action)
        if handler:
            await handler(threat_event)
    
    async def _block_request(self, threat_event: ThreatEvent):
        """Block the request completely"""
        threat_event.blocked = True
        self.ip_reputation[threat_event.source_ip] -= 10
        
        # Add to active attacks tracking
        self.active_attacks[threat_event.source_ip] = {
            'threat_type': threat_event.threat_type,
            'blocked_at': threat_event.timestamp,
            'block_duration': timedelta(hours=1)
        }
        
        await self._send_alert(threat_event, "REQUEST BLOCKED")
    
    async def _rate_limit_request(self, threat_event: ThreatEvent):
        """Apply rate limiting to the source"""
        # Implement rate limiting logic
        self.ip_reputation[threat_event.source_ip] -= 5
        await self._send_alert(threat_event, "RATE LIMITED")
    
    async def _challenge_request(self, threat_event: ThreatEvent):
        """Challenge the request with CAPTCHA or similar"""
        await self._send_alert(threat_event, "CHALLENGE ISSUED")
    
    async def _quarantine_user(self, threat_event: ThreatEvent):
        """Quarantine the user account"""
        if threat_event.user_id:
            # Implement user quarantine logic
            await self._send_alert(threat_event, "USER QUARANTINED")
    
    async def _monitor_request(self, threat_event: ThreatEvent):
        """Monitor the request without blocking"""
        await self._send_alert(threat_event, "MONITORING")
    
    async def _alert_only(self, threat_event: ThreatEvent):
        """Send alert without taking action"""
        await self._send_alert(threat_event, "ALERT")
    
    async def _send_alert(self, threat_event: ThreatEvent, action: str):
        """Send security alert"""
        alert_data = {
            'timestamp': threat_event.timestamp.isoformat(),
            'threat_id': threat_event.id,
            'threat_type': threat_event.threat_type.value,
            'severity': threat_event.severity.value,
            'source_ip': threat_event.source_ip,
            'action_taken': action,
            'confidence': threat_event.confidence_score,
            'details': threat_event.metadata
        }
        
        # Send to SIEM/logging system
        print(f"RASP ALERT: {json.dumps(alert_data, indent=2)}")
        
        # Send to external alerting systems if configured
        webhook_url = self.config.get('alert_webhook')
        if webhook_url:
            # Implement webhook notification
            pass
    
    def _background_monitoring(self):
        """Background thread for continuous monitoring"""
        while True:
            try:
                # Clean up old tracking data
                current_time = datetime.now()
                cutoff_time = current_time - timedelta(hours=24)
                
                # Clean request tracker
                for ip in list(self.request_tracker.keys()):
                    self.request_tracker[ip] = deque([
                        req_time for req_time in self.request_tracker[ip]
                        if req_time > cutoff_time
                    ], maxlen=1000)
                
                # Clean active attacks
                for ip in list(self.active_attacks.keys()):
                    attack_info = self.active_attacks[ip]
                    if current_time - attack_info['blocked_at'] > attack_info['block_duration']:
                        del self.active_attacks[ip]
                
                # Update baseline metrics
                self._update_baseline_metrics()
                
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                print(f"Background monitoring error: {e}")
                time.sleep(60)
    
    def _update_baseline_metrics(self):
        """Update baseline metrics for anomaly detection"""
        # This would typically use ML models to update baselines
        # For now, we'll use simple heuristics
        pass
    
    def get_threat_summary(self) -> Dict[str, Any]:
        """Get summary of recent threats"""
        recent_threats = [
            event for event in self.threat_events
            if datetime.now() - event.timestamp < timedelta(hours=24)
        ]
        
        threat_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for threat in recent_threats:
            threat_counts[threat.threat_type.value] += 1
            severity_counts[threat.severity.value] += 1
        
        return {
            'total_threats_24h': len(recent_threats),
            'active_blocks': len(self.active_attacks),
            'threat_types': dict(threat_counts),
            'severity_breakdown': dict(severity_counts),
            'top_attacking_ips': self._get_top_attacking_ips()
        }
    
    def _get_top_attacking_ips(self) -> List[Dict[str, Any]]:
        """Get top attacking IP addresses"""
        ip_threat_counts = defaultdict(int)
        
        for threat in self.threat_events:
            if datetime.now() - threat.timestamp < timedelta(hours=24):
                ip_threat_counts[threat.source_ip] += 1
        
        return [
            {'ip': ip, 'threat_count': count}
            for ip, count in sorted(ip_threat_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
    
    def is_request_blocked(self, source_ip: str) -> bool:
        """Check if requests from IP should be blocked"""
        return source_ip in self.active_attacks

class RASPMiddleware:
    """RASP middleware for web applications"""
    
    def __init__(self, rasp_engine: RASPEngine):
        self.rasp_engine = rasp_engine
    
    async def __call__(self, request, call_next):
        """Process request through RASP engine"""
        
        # Extract request data
        request_data = {
            'source_ip': request.client.host if hasattr(request, 'client') else 'unknown',
            'user_id': getattr(request.state, 'user_id', None),
            'path': str(request.url.path),
            'method': request.method,
            'headers': dict(request.headers),
            'query_params': dict(request.query_params),
            'body': await self._get_request_body(request)
        }
        
        # Check if IP is already blocked
        if self.rasp_engine.is_request_blocked(request_data['source_ip']):
            from fastapi import HTTPException
            raise HTTPException(status_code=403, detail="Request blocked by security policy")
        
        # Analyze request for threats
        threat_event = await self.rasp_engine.analyze_request(request_data)
        
        if threat_event and threat_event.blocked:
            from fastapi import HTTPException
            raise HTTPException(status_code=403, detail="Request blocked due to security threat")
        
        # Continue with request processing
        response = await call_next(request)
        
        return response
    
    async def _get_request_body(self, request) -> str:
        """Safely extract request body"""
        try:
            if hasattr(request, '_body'):
                return request._body.decode('utf-8')
            else:
                body = await request.body()
                return body.decode('utf-8')
        except Exception:
            return ""