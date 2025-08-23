"""
Threat Intelligence Integration and Custom Intelligence Correlation
Integrates multiple threat feeds and performs advanced correlation analysis
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import re
from collections import defaultdict
import requests
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class IndicatorType(Enum):
    IP_ADDRESS = "ip_address"
    DOMAIN = "domain"
    URL = "url"
    FILE_HASH = "file_hash"
    EMAIL = "email"
    USER_AGENT = "user_agent"
    REGISTRY_KEY = "registry_key"
    MUTEX = "mutex"

class ThreatSource(Enum):
    MITRE_ATTACK = "mitre_attack"
    NIST_CVE = "nist_cve"
    ALIENVAULT_OTX = "alienvault_otx"
    IBM_XFORCE = "ibm_xforce"
    RECORDED_FUTURE = "recorded_future"
    CROWDSTRIKE = "crowdstrike"
    FIREEYE = "fireeye"
    INTERNAL = "internal"
    OSINT = "osint"

@dataclass
class ThreatIndicator:
    indicator_id: str
    value: str
    indicator_type: IndicatorType
    threat_type: str
    severity: str
    confidence: float
    source: ThreatSource
    first_seen: datetime
    last_seen: datetime
    tags: List[str]
    context: Dict[str, Any]
    ttl: Optional[datetime] = None

@dataclass
class ThreatCampaign:
    campaign_id: str
    name: str
    threat_actor: str
    start_date: datetime
    end_date: Optional[datetime]
    indicators: List[str]
    tactics: List[str]
    techniques: List[str]
    targets: List[str]
    confidence: float
    description: str

@dataclass
class CorrelationResult:
    correlation_id: str
    matched_indicators: List[str]
    correlation_score: float
    threat_campaigns: List[str]
    risk_assessment: str
    recommended_actions: List[str]
    timestamp: datetime

class ThreatIntelligenceCorrelator:
    """Advanced threat intelligence correlation engine"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or "sqlite:///threat_intelligence.db"
        self.threat_feeds = {}
        self.indicators = {}
        self.campaigns = {}
        self.correlation_rules = []
        self.feed_configs = self._initialize_feed_configs()
        
    async def initialize(self):
        """Initialize threat intelligence system"""
        await self._setup_database()
        await self._load_correlation_rules()
        await self._initialize_threat_feeds()
        
    def _initialize_feed_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize threat feed configurations"""
        return {
            "mitre_attack": {
                "url": "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json",
                "update_interval": 86400,  # 24 hours
                "enabled": True,
                "priority": 9
            },
            "alienvault_otx": {
                "url": "https://otx.alienvault.com/api/v1/indicators/export",
                "update_interval": 3600,  # 1 hour
                "enabled": True,
                "priority": 8
            },
            "internal_feeds": {
                "update_interval": 300,  # 5 minutes
                "enabled": True,
                "priority": 10
            }
        }
        
    async def _setup_database(self):
        """Setup threat intelligence database"""
        # Database setup would go here
        logger.info("Setting up threat intelligence database")
        
    async def _load_correlation_rules(self):
        """Load threat correlation rules"""
        self.correlation_rules = [
            {
                "rule_id": "ip_domain_correlation",
                "description": "Correlate suspicious IPs with malicious domains",
                "conditions": ["ip_address", "domain"],
                "threshold": 0.7,
                "action": "alert"
            },
            {
                "rule_id": "hash_campaign_correlation",
                "description": "Correlate file hashes with known campaigns",
                "conditions": ["file_hash", "campaign_indicators"],
                "threshold": 0.8,
                "action": "block"
            },
            {
                "rule_id": "multi_indicator_correlation",
                "description": "Correlate multiple indicators from same source",
                "conditions": ["multiple_indicators"],
                "threshold": 0.6,
                "action": "investigate"
            }
        ]
        
    async def _initialize_threat_feeds(self):
        """Initialize and start threat feed collection"""
        for feed_name, config in self.feed_configs.items():
            if config.get("enabled", False):
                self.threat_feeds[feed_name] = {
                    "config": config,
                    "last_update": None,
                    "status": "initialized",
                    "indicators_count": 0
                }
                
    async def collect_threat_intelligence(self) -> Dict[str, Any]:
        """Collect threat intelligence from all configured feeds"""
        collection_results = {}
        
        for feed_name, feed_info in self.threat_feeds.items():
            try:
                if feed_name == "mitre_attack":
                    indicators = await self._collect_mitre_attack()
                elif feed_name == "alienvault_otx":
                    indicators = await self._collect_alienvault_otx()
                elif feed_name == "internal_feeds":
                    indicators = await self._collect_internal_feeds()
                else:
                    indicators = []
                    
                # Store indicators
                for indicator in indicators:
                    self.indicators[indicator.indicator_id] = indicator
                    
                collection_results[feed_name] = {
                    "status": "success",
                    "indicators_collected": len(indicators),
                    "last_update": datetime.now().isoformat()
                }
                
                self.threat_feeds[feed_name]["last_update"] = datetime.now()
                self.threat_feeds[feed_name]["indicators_count"] = len(indicators)
                
            except Exception as e:
                logger.error(f"Error collecting from {feed_name}: {str(e)}")
                collection_results[feed_name] = {
                    "status": "error",
                    "error": str(e),
                    "last_update": None
                }
                
        return collection_results
        
    async def _collect_mitre_attack(self) -> List[ThreatIndicator]:
        """Collect MITRE ATT&CK framework data"""
        indicators = []
        
        # Simulate MITRE ATT&CK data collection
        techniques = [
            "T1566.001", "T1059.001", "T1055", "T1003.001", "T1083",
            "T1082", "T1057", "T1012", "T1016", "T1033"
        ]
        
        for i, technique in enumerate(techniques):
            indicator = ThreatIndicator(
                indicator_id=f"mitre_{technique}_{int(datetime.now().timestamp())}",
                value=technique,
                indicator_type=IndicatorType.REGISTRY_KEY,
                threat_type="technique",
                severity="medium",
                confidence=0.9,
                source=ThreatSource.MITRE_ATTACK,
                first_seen=datetime.now() - timedelta(days=np.random.randint(1, 30)),
                last_seen=datetime.now(),
                tags=["mitre", "attack", "technique"],
                context={
                    "tactic": "initial_access" if i < 3 else "execution",
                    "platform": "windows",
                    "data_sources": ["process_monitoring", "file_monitoring"]
                }
            )
            indicators.append(indicator)
            
        return indicators
        
    async def _collect_alienvault_otx(self) -> List[ThreatIndicator]:
        """Collect AlienVault OTX threat intelligence"""
        indicators = []
        
        # Simulate OTX data collection
        sample_ips = [
            "192.168.1.100", "10.0.0.50", "172.16.0.25",
            "203.0.113.10", "198.51.100.20"
        ]
        
        sample_domains = [
            "malicious-domain.com", "phishing-site.net", "c2-server.org",
            "fake-bank.com", "trojan-host.info"
        ]
        
        # Generate IP indicators
        for ip in sample_ips:
            indicator = ThreatIndicator(
                indicator_id=f"otx_ip_{hashlib.md5(ip.encode()).hexdigest()[:8]}",
                value=ip,
                indicator_type=IndicatorType.IP_ADDRESS,
                threat_type="malicious_ip",
                severity=np.random.choice(["low", "medium", "high"]),
                confidence=np.random.uniform(0.6, 0.95),
                source=ThreatSource.ALIENVAULT_OTX,
                first_seen=datetime.now() - timedelta(days=np.random.randint(1, 7)),
                last_seen=datetime.now(),
                tags=["otx", "malicious", "ip"],
                context={
                    "country": np.random.choice(["CN", "RU", "KP", "IR"]),
                    "asn": f"AS{np.random.randint(1000, 9999)}",
                    "reputation": "malicious"
                }
            )
            indicators.append(indicator)
            
        # Generate domain indicators
        for domain in sample_domains:
            indicator = ThreatIndicator(
                indicator_id=f"otx_domain_{hashlib.md5(domain.encode()).hexdigest()[:8]}",
                value=domain,
                indicator_type=IndicatorType.DOMAIN,
                threat_type="malicious_domain",
                severity=np.random.choice(["medium", "high"]),
                confidence=np.random.uniform(0.7, 0.95),
                source=ThreatSource.ALIENVAULT_OTX,
                first_seen=datetime.now() - timedelta(days=np.random.randint(1, 14)),
                last_seen=datetime.now(),
                tags=["otx", "malicious", "domain"],
                context={
                    "category": np.random.choice(["phishing", "malware", "c2"]),
                    "registrar": "unknown",
                    "creation_date": "2023-01-01"
                }
            )
            indicators.append(indicator)
            
        return indicators
        
    async def _collect_internal_feeds(self) -> List[ThreatIndicator]:
        """Collect internal threat intelligence"""
        indicators = []
        
        # Simulate internal threat intelligence
        internal_hashes = [
            "d41d8cd98f00b204e9800998ecf8427e",
            "5d41402abc4b2a76b9719d911017c592",
            "098f6bcd4621d373cade4e832627b4f6"
        ]
        
        for hash_value in internal_hashes:
            indicator = ThreatIndicator(
                indicator_id=f"internal_hash_{hash_value[:8]}",
                value=hash_value,
                indicator_type=IndicatorType.FILE_HASH,
                threat_type="malware",
                severity="high",
                confidence=0.95,
                source=ThreatSource.INTERNAL,
                first_seen=datetime.now() - timedelta(hours=np.random.randint(1, 24)),
                last_seen=datetime.now(),
                tags=["internal", "malware", "hash"],
                context={
                    "file_type": "executable",
                    "detection_method": "sandbox_analysis",
                    "family": "trojan"
                }
            )
            indicators.append(indicator)
            
        return indicators
        
    def correlate_indicators(self, target_indicators: List[str]) -> List[CorrelationResult]:
        """Correlate target indicators against threat intelligence"""
        correlations = []
        
        for target in target_indicators:
            correlation_result = self._perform_correlation(target)
            if correlation_result:
                correlations.append(correlation_result)
                
        return correlations
        
    def _perform_correlation(self, target_indicator: str) -> Optional[CorrelationResult]:
        """Perform correlation analysis for a single indicator"""
        matched_indicators = []
        correlation_score = 0.0
        threat_campaigns = []
        
        # Direct indicator matching
        for indicator_id, indicator in self.indicators.items():
            if self._indicators_match(target_indicator, indicator.value):
                matched_indicators.append(indicator_id)
                correlation_score += indicator.confidence * 0.8
                
        # Campaign correlation
        for campaign_id, campaign in self.campaigns.items():
            if target_indicator in campaign.indicators:
                threat_campaigns.append(campaign_id)
                correlation_score += 0.6
                
        # Pattern-based correlation
        pattern_matches = self._find_pattern_matches(target_indicator)
        correlation_score += len(pattern_matches) * 0.3
        
        if matched_indicators or threat_campaigns or pattern_matches:
            risk_assessment = self._assess_risk(correlation_score)
            recommended_actions = self._generate_recommendations(risk_assessment, matched_indicators)
            
            return CorrelationResult(
                correlation_id=f"corr_{int(datetime.now().timestamp())}_{hashlib.md5(target_indicator.encode()).hexdigest()[:8]}",
                matched_indicators=matched_indicators,
                correlation_score=min(correlation_score, 1.0),
                threat_campaigns=threat_campaigns,
                risk_assessment=risk_assessment,
                recommended_actions=recommended_actions,
                timestamp=datetime.now()
            )
            
        return None
        
    def _indicators_match(self, target: str, indicator_value: str) -> bool:
        """Check if indicators match"""
        # Exact match
        if target == indicator_value:
            return True
            
        # Fuzzy matching for domains
        if self._is_domain(target) and self._is_domain(indicator_value):
            return self._domain_similarity(target, indicator_value) > 0.8
            
        # IP range matching
        if self._is_ip(target) and self._is_ip(indicator_value):
            return self._ip_similarity(target, indicator_value) > 0.9
            
        return False
        
    def _is_domain(self, value: str) -> bool:
        """Check if value is a domain"""
        domain_pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'
        return bool(re.match(domain_pattern, value))
        
    def _is_ip(self, value: str) -> bool:
        """Check if value is an IP address"""
        ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        return bool(re.match(ip_pattern, value))
        
    def _domain_similarity(self, domain1: str, domain2: str) -> float:
        """Calculate domain similarity"""
        # Simple Levenshtein distance-based similarity
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
                
            return previous_row[-1]
            
        distance = levenshtein_distance(domain1, domain2)
        max_len = max(len(domain1), len(domain2))
        return 1 - (distance / max_len) if max_len > 0 else 0
        
    def _ip_similarity(self, ip1: str, ip2: str) -> float:
        """Calculate IP address similarity"""
        octets1 = ip1.split('.')
        octets2 = ip2.split('.')
        
        if len(octets1) != 4 or len(octets2) != 4:
            return 0
            
        matches = sum(1 for o1, o2 in zip(octets1, octets2) if o1 == o2)
        return matches / 4
        
    def _find_pattern_matches(self, target_indicator: str) -> List[str]:
        """Find pattern-based matches"""
        patterns = []
        
        # Check for suspicious patterns
        if self._is_domain(target_indicator):
            # Check for DGA-like domains
            if self._is_dga_domain(target_indicator):
                patterns.append("dga_domain")
                
            # Check for typosquatting
            if self._is_typosquatting(target_indicator):
                patterns.append("typosquatting")
                
        elif self._is_ip(target_indicator):
            # Check for suspicious IP ranges
            if self._is_suspicious_ip_range(target_indicator):
                patterns.append("suspicious_ip_range")
                
        return patterns
        
    def _is_dga_domain(self, domain: str) -> bool:
        """Check if domain matches DGA patterns"""
        # Simple heuristics for DGA detection
        domain_part = domain.split('.')[0]
        
        # Check for high entropy (randomness)
        if len(domain_part) > 10:
            vowels = sum(1 for c in domain_part.lower() if c in 'aeiou')
            consonants = len(domain_part) - vowels
            if vowels > 0 and consonants / vowels > 3:
                return True
                
        return False
        
    def _is_typosquatting(self, domain: str) -> bool:
        """Check if domain is typosquatting legitimate domains"""
        legitimate_domains = [
            "google.com", "microsoft.com", "amazon.com", "facebook.com",
            "apple.com", "netflix.com", "paypal.com", "ebay.com"
        ]
        
        for legit_domain in legitimate_domains:
            if self._domain_similarity(domain, legit_domain) > 0.7 and domain != legit_domain:
                return True
                
        return False
        
    def _is_suspicious_ip_range(self, ip: str) -> bool:
        """Check if IP is in suspicious ranges"""
        suspicious_ranges = [
            "192.168.", "10.", "172.16.", "127."  # Private/local ranges used maliciously
        ]
        
        return any(ip.startswith(range_prefix) for range_prefix in suspicious_ranges)
        
    def _assess_risk(self, correlation_score: float) -> str:
        """Assess risk level based on correlation score"""
        if correlation_score >= 0.8:
            return "CRITICAL"
        elif correlation_score >= 0.6:
            return "HIGH"
        elif correlation_score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
            
    def _generate_recommendations(self, risk_level: str, matched_indicators: List[str]) -> List[str]:
        """Generate recommended actions based on correlation results"""
        recommendations = []
        
        if risk_level == "CRITICAL":
            recommendations.extend([
                "Immediately block all traffic to/from this indicator",
                "Initiate incident response procedures",
                "Perform forensic analysis of affected systems",
                "Notify security team and management"
            ])
        elif risk_level == "HIGH":
            recommendations.extend([
                "Block indicator and monitor for related activity",
                "Review logs for historical connections",
                "Increase monitoring for related indicators",
                "Consider threat hunting activities"
            ])
        elif risk_level == "MEDIUM":
            recommendations.extend([
                "Add to watchlist for monitoring",
                "Review context and validate threat",
                "Consider additional intelligence gathering"
            ])
        else:
            recommendations.append("Continue monitoring - low confidence threat")
            
        if matched_indicators:
            recommendations.append(f"Review {len(matched_indicators)} related indicators")
            
        return recommendations
        
    def get_threat_intelligence_summary(self) -> Dict[str, Any]:
        """Get comprehensive threat intelligence summary"""
        return {
            "feed_status": {
                feed_name: {
                    "status": feed_info.get("status", "unknown"),
                    "last_update": feed_info.get("last_update"),
                    "indicators_count": feed_info.get("indicators_count", 0)
                }
                for feed_name, feed_info in self.threat_feeds.items()
            },
            "total_indicators": len(self.indicators),
            "indicator_breakdown": self._get_indicator_breakdown(),
            "recent_correlations": self._get_recent_correlations(),
            "threat_landscape": self._analyze_threat_landscape(),
            "feed_health": self._assess_feed_health()
        }
        
    def _get_indicator_breakdown(self) -> Dict[str, int]:
        """Get breakdown of indicators by type"""
        breakdown = defaultdict(int)
        
        for indicator in self.indicators.values():
            breakdown[indicator.indicator_type.value] += 1
            breakdown[f"severity_{indicator.severity}"] += 1
            breakdown[f"source_{indicator.source.value}"] += 1
            
        return dict(breakdown)
        
    def _get_recent_correlations(self) -> List[Dict[str, Any]]:
        """Get recent correlation results"""
        # This would typically query a database
        # For now, return sample data
        return [
            {
                "correlation_id": "corr_123456",
                "timestamp": datetime.now().isoformat(),
                "risk_level": "HIGH",
                "matched_indicators": 3,
                "threat_campaigns": 1
            }
        ]
        
    def _analyze_threat_landscape(self) -> Dict[str, Any]:
        """Analyze current threat landscape"""
        return {
            "trending_threats": ["Ransomware", "Phishing", "Supply Chain"],
            "active_campaigns": len(self.campaigns),
            "threat_actor_activity": "Moderate",
            "geographic_distribution": {
                "CN": 35,
                "RU": 25,
                "KP": 15,
                "IR": 10,
                "Other": 15
            }
        }
        
    def _assess_feed_health(self) -> Dict[str, str]:
        """Assess health of threat intelligence feeds"""
        health_status = {}
        
        for feed_name, feed_info in self.threat_feeds.items():
            last_update = feed_info.get("last_update")
            if last_update:
                time_since_update = datetime.now() - last_update
                if time_since_update.total_seconds() < 3600:  # 1 hour
                    health_status[feed_name] = "Healthy"
                elif time_since_update.total_seconds() < 86400:  # 24 hours
                    health_status[feed_name] = "Warning"
                else:
                    health_status[feed_name] = "Critical"
            else:
                health_status[feed_name] = "Unknown"
                
        return health_status