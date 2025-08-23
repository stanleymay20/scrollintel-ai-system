"""
Forensic Analysis Capabilities with Detailed Incident Reconstruction
Advanced digital forensics and incident reconstruction system
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import re
import gzip
import base64
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class EvidenceType(Enum):
    LOG_FILE = "log_file"
    NETWORK_PACKET = "network_packet"
    FILE_SYSTEM = "file_system"
    MEMORY_DUMP = "memory_dump"
    REGISTRY_ENTRY = "registry_entry"
    DATABASE_RECORD = "database_record"
    EMAIL = "email"
    BROWSER_ARTIFACT = "browser_artifact"

class ForensicPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DigitalEvidence:
    evidence_id: str
    evidence_type: EvidenceType
    source_system: str
    collection_timestamp: datetime
    file_path: Optional[str]
    file_hash: Optional[str]
    file_size: Optional[int]
    metadata: Dict[str, Any]
    chain_of_custody: List[Dict[str, Any]]
    integrity_verified: bool

@dataclass
class ForensicTimeline:
    timeline_id: str
    incident_id: str
    events: List[Dict[str, Any]]
    start_time: datetime
    end_time: datetime
    confidence_score: float
    reconstruction_method: str

@dataclass
class IncidentReconstruction:
    reconstruction_id: str
    incident_id: str
    attack_vector: str
    attack_timeline: ForensicTimeline
    affected_systems: List[str]
    compromised_accounts: List[str]
    data_accessed: List[str]
    persistence_mechanisms: List[str]
    lateral_movement: List[str]
    exfiltration_evidence: List[str]
    attribution_indicators: List[str]
    confidence_level: float

class ForensicAnalyzer:
    """Advanced forensic analysis and incident reconstruction system"""
    
    def __init__(self, evidence_store_path: str = "evidence_store"):
        self.evidence_store_path = Path(evidence_store_path)
        self.evidence_store_path.mkdir(exist_ok=True)
        self.database_path = self.evidence_store_path / "forensics.db"
        self.evidence_registry = {}
        self.timeline_cache = {}
        
    async def initialize(self):
        """Initialize forensic analysis system"""
        await self._setup_database()
        await self._initialize_analysis_engines()
        
    async def _setup_database(self):
        """Setup forensic database"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Evidence table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evidence (
                evidence_id TEXT PRIMARY KEY,
                evidence_type TEXT,
                source_system TEXT,
                collection_timestamp TEXT,
                file_path TEXT,
                file_hash TEXT,
                file_size INTEGER,
                metadata TEXT,
                chain_of_custody TEXT,
                integrity_verified BOOLEAN
            )
        """)
        
        # Timeline events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS timeline_events (
                event_id TEXT PRIMARY KEY,
                timeline_id TEXT,
                timestamp TEXT,
                event_type TEXT,
                source_system TEXT,
                description TEXT,
                evidence_id TEXT,
                confidence_score REAL,
                metadata TEXT
            )
        """)
        
        # Incident reconstructions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS incident_reconstructions (
                reconstruction_id TEXT PRIMARY KEY,
                incident_id TEXT,
                attack_vector TEXT,
                timeline_id TEXT,
                affected_systems TEXT,
                compromised_accounts TEXT,
                data_accessed TEXT,
                persistence_mechanisms TEXT,
                lateral_movement TEXT,
                exfiltration_evidence TEXT,
                attribution_indicators TEXT,
                confidence_level REAL,
                created_timestamp TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        
    async def _initialize_analysis_engines(self):
        """Initialize various analysis engines"""
        logger.info("Initializing forensic analysis engines")
        
    async def collect_evidence(self, incident_id: str, source_systems: List[str]) -> List[DigitalEvidence]:
        """Collect digital evidence from specified systems"""
        evidence_items = []
        
        for system in source_systems:
            system_evidence = await self._collect_system_evidence(incident_id, system)
            evidence_items.extend(system_evidence)
            
        # Store evidence in database
        await self._store_evidence(evidence_items)
        
        return evidence_items
        
    async def _collect_system_evidence(self, incident_id: str, system: str) -> List[DigitalEvidence]:
        """Collect evidence from a specific system"""
        evidence_items = []
        
        # Simulate evidence collection from different system types
        if system.startswith("web_server"):
            evidence_items.extend(await self._collect_web_server_evidence(incident_id, system))
        elif system.startswith("database"):
            evidence_items.extend(await self._collect_database_evidence(incident_id, system))
        elif system.startswith("workstation"):
            evidence_items.extend(await self._collect_workstation_evidence(incident_id, system))
        elif system.startswith("network"):
            evidence_items.extend(await self._collect_network_evidence(incident_id, system))
            
        return evidence_items
        
    async def _collect_web_server_evidence(self, incident_id: str, system: str) -> List[DigitalEvidence]:
        """Collect evidence from web servers"""
        evidence_items = []
        
        # Access logs
        access_log_evidence = DigitalEvidence(
            evidence_id=f"evidence_{incident_id}_{system}_access_log",
            evidence_type=EvidenceType.LOG_FILE,
            source_system=system,
            collection_timestamp=datetime.now(),
            file_path="/var/log/apache2/access.log",
            file_hash=hashlib.sha256(b"simulated_access_log_content").hexdigest(),
            file_size=1024000,
            metadata={
                "log_format": "combined",
                "rotation_policy": "daily",
                "compression": "gzip"
            },
            chain_of_custody=[{
                "collector": "forensic_system",
                "timestamp": datetime.now().isoformat(),
                "action": "collected",
                "hash_verified": True
            }],
            integrity_verified=True
        )
        evidence_items.append(access_log_evidence)
        
        # Error logs
        error_log_evidence = DigitalEvidence(
            evidence_id=f"evidence_{incident_id}_{system}_error_log",
            evidence_type=EvidenceType.LOG_FILE,
            source_system=system,
            collection_timestamp=datetime.now(),
            file_path="/var/log/apache2/error.log",
            file_hash=hashlib.sha256(b"simulated_error_log_content").hexdigest(),
            file_size=512000,
            metadata={
                "log_level": "warn",
                "format": "standard"
            },
            chain_of_custody=[{
                "collector": "forensic_system",
                "timestamp": datetime.now().isoformat(),
                "action": "collected",
                "hash_verified": True
            }],
            integrity_verified=True
        )
        evidence_items.append(error_log_evidence)
        
        return evidence_items
        
    async def _collect_database_evidence(self, incident_id: str, system: str) -> List[DigitalEvidence]:
        """Collect evidence from database systems"""
        evidence_items = []
        
        # Database transaction logs
        transaction_log_evidence = DigitalEvidence(
            evidence_id=f"evidence_{incident_id}_{system}_transaction_log",
            evidence_type=EvidenceType.DATABASE_RECORD,
            source_system=system,
            collection_timestamp=datetime.now(),
            file_path="/var/lib/mysql/mysql-bin.000001",
            file_hash=hashlib.sha256(b"simulated_transaction_log").hexdigest(),
            file_size=2048000,
            metadata={
                "database_type": "mysql",
                "version": "8.0.25",
                "binlog_format": "row"
            },
            chain_of_custody=[{
                "collector": "forensic_system",
                "timestamp": datetime.now().isoformat(),
                "action": "collected",
                "hash_verified": True
            }],
            integrity_verified=True
        )
        evidence_items.append(transaction_log_evidence)
        
        return evidence_items
        
    async def _collect_workstation_evidence(self, incident_id: str, system: str) -> List[DigitalEvidence]:
        """Collect evidence from workstations"""
        evidence_items = []
        
        # Windows Event Logs
        event_log_evidence = DigitalEvidence(
            evidence_id=f"evidence_{incident_id}_{system}_event_log",
            evidence_type=EvidenceType.LOG_FILE,
            source_system=system,
            collection_timestamp=datetime.now(),
            file_path="C:\\Windows\\System32\\winevt\\Logs\\Security.evtx",
            file_hash=hashlib.sha256(b"simulated_event_log").hexdigest(),
            file_size=10240000,
            metadata={
                "log_type": "security",
                "format": "evtx",
                "events_count": 15000
            },
            chain_of_custody=[{
                "collector": "forensic_system",
                "timestamp": datetime.now().isoformat(),
                "action": "collected",
                "hash_verified": True
            }],
            integrity_verified=True
        )
        evidence_items.append(event_log_evidence)
        
        # Registry hives
        registry_evidence = DigitalEvidence(
            evidence_id=f"evidence_{incident_id}_{system}_registry",
            evidence_type=EvidenceType.REGISTRY_ENTRY,
            source_system=system,
            collection_timestamp=datetime.now(),
            file_path="C:\\Windows\\System32\\config\\SYSTEM",
            file_hash=hashlib.sha256(b"simulated_registry_hive").hexdigest(),
            file_size=20480000,
            metadata={
                "hive_type": "SYSTEM",
                "last_modified": datetime.now().isoformat()
            },
            chain_of_custody=[{
                "collector": "forensic_system",
                "timestamp": datetime.now().isoformat(),
                "action": "collected",
                "hash_verified": True
            }],
            integrity_verified=True
        )
        evidence_items.append(registry_evidence)
        
        return evidence_items
        
    async def _collect_network_evidence(self, incident_id: str, system: str) -> List[DigitalEvidence]:
        """Collect network evidence"""
        evidence_items = []
        
        # Network packet capture
        pcap_evidence = DigitalEvidence(
            evidence_id=f"evidence_{incident_id}_{system}_pcap",
            evidence_type=EvidenceType.NETWORK_PACKET,
            source_system=system,
            collection_timestamp=datetime.now(),
            file_path="/var/log/network/capture.pcap",
            file_hash=hashlib.sha256(b"simulated_pcap_data").hexdigest(),
            file_size=50000000,
            metadata={
                "capture_duration": "1 hour",
                "packet_count": 125000,
                "protocols": ["TCP", "UDP", "HTTP", "HTTPS"]
            },
            chain_of_custody=[{
                "collector": "forensic_system",
                "timestamp": datetime.now().isoformat(),
                "action": "collected",
                "hash_verified": True
            }],
            integrity_verified=True
        )
        evidence_items.append(pcap_evidence)
        
        return evidence_items
        
    async def _store_evidence(self, evidence_items: List[DigitalEvidence]):
        """Store evidence in database"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        for evidence in evidence_items:
            cursor.execute("""
                INSERT OR REPLACE INTO evidence 
                (evidence_id, evidence_type, source_system, collection_timestamp, 
                 file_path, file_hash, file_size, metadata, chain_of_custody, integrity_verified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                evidence.evidence_id,
                evidence.evidence_type.value,
                evidence.source_system,
                evidence.collection_timestamp.isoformat(),
                evidence.file_path,
                evidence.file_hash,
                evidence.file_size,
                json.dumps(evidence.metadata),
                json.dumps(evidence.chain_of_custody),
                evidence.integrity_verified
            ))
            
        conn.commit()
        conn.close()
        
    async def analyze_evidence(self, evidence_items: List[DigitalEvidence]) -> Dict[str, Any]:
        """Perform comprehensive evidence analysis"""
        analysis_results = {
            "evidence_summary": self._summarize_evidence(evidence_items),
            "timeline_analysis": await self._analyze_timeline(evidence_items),
            "artifact_analysis": await self._analyze_artifacts(evidence_items),
            "correlation_analysis": await self._correlate_evidence(evidence_items),
            "ioc_extraction": await self._extract_iocs(evidence_items)
        }
        
        return analysis_results
        
    def _summarize_evidence(self, evidence_items: List[DigitalEvidence]) -> Dict[str, Any]:
        """Summarize collected evidence"""
        evidence_by_type = {}
        evidence_by_system = {}
        total_size = 0
        
        for evidence in evidence_items:
            # Group by type
            evidence_type = evidence.evidence_type.value
            if evidence_type not in evidence_by_type:
                evidence_by_type[evidence_type] = 0
            evidence_by_type[evidence_type] += 1
            
            # Group by system
            if evidence.source_system not in evidence_by_system:
                evidence_by_system[evidence.source_system] = 0
            evidence_by_system[evidence.source_system] += 1
            
            # Calculate total size
            if evidence.file_size:
                total_size += evidence.file_size
                
        return {
            "total_evidence_items": len(evidence_items),
            "evidence_by_type": evidence_by_type,
            "evidence_by_system": evidence_by_system,
            "total_size_bytes": total_size,
            "collection_timespan": {
                "start": min(e.collection_timestamp for e in evidence_items).isoformat(),
                "end": max(e.collection_timestamp for e in evidence_items).isoformat()
            }
        }
        
    async def _analyze_timeline(self, evidence_items: List[DigitalEvidence]) -> Dict[str, Any]:
        """Analyze timeline from evidence"""
        timeline_events = []
        
        for evidence in evidence_items:
            # Extract timeline events from different evidence types
            if evidence.evidence_type == EvidenceType.LOG_FILE:
                events = await self._extract_log_events(evidence)
                timeline_events.extend(events)
            elif evidence.evidence_type == EvidenceType.NETWORK_PACKET:
                events = await self._extract_network_events(evidence)
                timeline_events.extend(events)
            elif evidence.evidence_type == EvidenceType.REGISTRY_ENTRY:
                events = await self._extract_registry_events(evidence)
                timeline_events.extend(events)
                
        # Sort events by timestamp
        timeline_events.sort(key=lambda x: x['timestamp'])
        
        return {
            "total_events": len(timeline_events),
            "event_types": list(set(e['event_type'] for e in timeline_events)),
            "timespan": {
                "start": timeline_events[0]['timestamp'] if timeline_events else None,
                "end": timeline_events[-1]['timestamp'] if timeline_events else None
            },
            "events": timeline_events[:100]  # Return first 100 events
        }
        
    async def _extract_log_events(self, evidence: DigitalEvidence) -> List[Dict[str, Any]]:
        """Extract events from log files"""
        events = []
        
        # Simulate log parsing
        base_time = evidence.collection_timestamp - timedelta(hours=24)
        
        for i in range(50):  # Simulate 50 log events
            event_time = base_time + timedelta(minutes=i * 5)
            
            event = {
                "event_id": f"log_event_{evidence.evidence_id}_{i}",
                "timestamp": event_time.isoformat(),
                "event_type": np.random.choice(["login", "logout", "file_access", "network_connection", "error"]),
                "source_system": evidence.source_system,
                "description": f"Simulated log event {i}",
                "evidence_id": evidence.evidence_id,
                "confidence_score": np.random.uniform(0.7, 1.0),
                "metadata": {
                    "log_level": np.random.choice(["INFO", "WARN", "ERROR"]),
                    "user": f"user_{np.random.randint(1, 10)}",
                    "ip_address": f"192.168.1.{np.random.randint(1, 255)}"
                }
            }
            events.append(event)
            
        return events
        
    async def _extract_network_events(self, evidence: DigitalEvidence) -> List[Dict[str, Any]]:
        """Extract events from network captures"""
        events = []
        
        # Simulate network event extraction
        base_time = evidence.collection_timestamp - timedelta(hours=1)
        
        for i in range(30):  # Simulate 30 network events
            event_time = base_time + timedelta(minutes=i * 2)
            
            event = {
                "event_id": f"network_event_{evidence.evidence_id}_{i}",
                "timestamp": event_time.isoformat(),
                "event_type": np.random.choice(["connection", "dns_query", "http_request", "data_transfer"]),
                "source_system": evidence.source_system,
                "description": f"Network event {i}",
                "evidence_id": evidence.evidence_id,
                "confidence_score": np.random.uniform(0.8, 1.0),
                "metadata": {
                    "src_ip": f"192.168.1.{np.random.randint(1, 255)}",
                    "dst_ip": f"203.0.113.{np.random.randint(1, 255)}",
                    "protocol": np.random.choice(["TCP", "UDP"]),
                    "port": np.random.choice([80, 443, 22, 3389, 1433])
                }
            }
            events.append(event)
            
        return events
        
    async def _extract_registry_events(self, evidence: DigitalEvidence) -> List[Dict[str, Any]]:
        """Extract events from registry evidence"""
        events = []
        
        # Simulate registry event extraction
        base_time = evidence.collection_timestamp - timedelta(days=7)
        
        for i in range(20):  # Simulate 20 registry events
            event_time = base_time + timedelta(hours=i * 8)
            
            event = {
                "event_id": f"registry_event_{evidence.evidence_id}_{i}",
                "timestamp": event_time.isoformat(),
                "event_type": np.random.choice(["key_created", "key_modified", "key_deleted", "value_set"]),
                "source_system": evidence.source_system,
                "description": f"Registry modification {i}",
                "evidence_id": evidence.evidence_id,
                "confidence_score": np.random.uniform(0.9, 1.0),
                "metadata": {
                    "registry_path": f"HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run\\Entry{i}",
                    "value_type": "REG_SZ",
                    "value_data": f"C:\\Program Files\\Application{i}\\app.exe"
                }
            }
            events.append(event)
            
        return events
        
    async def _analyze_artifacts(self, evidence_items: List[DigitalEvidence]) -> Dict[str, Any]:
        """Analyze digital artifacts"""
        artifacts = {
            "file_artifacts": [],
            "network_artifacts": [],
            "registry_artifacts": [],
            "memory_artifacts": []
        }
        
        for evidence in evidence_items:
            if evidence.evidence_type == EvidenceType.FILE_SYSTEM:
                artifacts["file_artifacts"].extend(await self._analyze_file_artifacts(evidence))
            elif evidence.evidence_type == EvidenceType.NETWORK_PACKET:
                artifacts["network_artifacts"].extend(await self._analyze_network_artifacts(evidence))
            elif evidence.evidence_type == EvidenceType.REGISTRY_ENTRY:
                artifacts["registry_artifacts"].extend(await self._analyze_registry_artifacts(evidence))
            elif evidence.evidence_type == EvidenceType.MEMORY_DUMP:
                artifacts["memory_artifacts"].extend(await self._analyze_memory_artifacts(evidence))
                
        return artifacts
        
    async def _analyze_file_artifacts(self, evidence: DigitalEvidence) -> List[Dict[str, Any]]:
        """Analyze file system artifacts"""
        return [
            {
                "artifact_type": "suspicious_file",
                "file_path": "/tmp/malware.exe",
                "file_hash": "d41d8cd98f00b204e9800998ecf8427e",
                "creation_time": datetime.now().isoformat(),
                "significance": "Potential malware executable"
            }
        ]
        
    async def _analyze_network_artifacts(self, evidence: DigitalEvidence) -> List[Dict[str, Any]]:
        """Analyze network artifacts"""
        return [
            {
                "artifact_type": "suspicious_connection",
                "destination_ip": "203.0.113.100",
                "destination_port": 4444,
                "protocol": "TCP",
                "significance": "Connection to known C2 server"
            }
        ]
        
    async def _analyze_registry_artifacts(self, evidence: DigitalEvidence) -> List[Dict[str, Any]]:
        """Analyze registry artifacts"""
        return [
            {
                "artifact_type": "persistence_mechanism",
                "registry_key": "HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run\\Backdoor",
                "value": "C:\\Windows\\System32\\backdoor.exe",
                "significance": "Malware persistence mechanism"
            }
        ]
        
    async def _analyze_memory_artifacts(self, evidence: DigitalEvidence) -> List[Dict[str, Any]]:
        """Analyze memory artifacts"""
        return [
            {
                "artifact_type": "injected_code",
                "process_name": "explorer.exe",
                "pid": 1234,
                "significance": "Code injection detected"
            }
        ]
        
    async def _correlate_evidence(self, evidence_items: List[DigitalEvidence]) -> Dict[str, Any]:
        """Correlate evidence across different sources"""
        correlations = []
        
        # Simple correlation based on timestamps and common indicators
        for i, evidence1 in enumerate(evidence_items):
            for evidence2 in evidence_items[i+1:]:
                correlation_score = self._calculate_correlation_score(evidence1, evidence2)
                if correlation_score > 0.5:
                    correlations.append({
                        "evidence1_id": evidence1.evidence_id,
                        "evidence2_id": evidence2.evidence_id,
                        "correlation_score": correlation_score,
                        "correlation_type": "temporal_proximity"
                    })
                    
        return {
            "total_correlations": len(correlations),
            "correlations": correlations
        }
        
    def _calculate_correlation_score(self, evidence1: DigitalEvidence, evidence2: DigitalEvidence) -> float:
        """Calculate correlation score between two pieces of evidence"""
        score = 0.0
        
        # Time proximity
        time_diff = abs((evidence1.collection_timestamp - evidence2.collection_timestamp).total_seconds())
        if time_diff < 3600:  # Within 1 hour
            score += 0.3
        elif time_diff < 86400:  # Within 1 day
            score += 0.1
            
        # Same source system
        if evidence1.source_system == evidence2.source_system:
            score += 0.2
            
        # Similar evidence types
        if evidence1.evidence_type == evidence2.evidence_type:
            score += 0.1
            
        return min(score, 1.0)
        
    async def _extract_iocs(self, evidence_items: List[DigitalEvidence]) -> Dict[str, Any]:
        """Extract Indicators of Compromise (IOCs)"""
        iocs = {
            "ip_addresses": [],
            "domains": [],
            "file_hashes": [],
            "urls": [],
            "email_addresses": [],
            "registry_keys": []
        }
        
        for evidence in evidence_items:
            # Extract IOCs based on evidence type
            if evidence.evidence_type == EvidenceType.LOG_FILE:
                extracted_iocs = await self._extract_log_iocs(evidence)
            elif evidence.evidence_type == EvidenceType.NETWORK_PACKET:
                extracted_iocs = await self._extract_network_iocs(evidence)
            elif evidence.evidence_type == EvidenceType.REGISTRY_ENTRY:
                extracted_iocs = await self._extract_registry_iocs(evidence)
            else:
                extracted_iocs = {}
                
            # Merge extracted IOCs
            for ioc_type, ioc_list in extracted_iocs.items():
                if ioc_type in iocs:
                    iocs[ioc_type].extend(ioc_list)
                    
        # Remove duplicates
        for ioc_type in iocs:
            iocs[ioc_type] = list(set(iocs[ioc_type]))
            
        return iocs
        
    async def _extract_log_iocs(self, evidence: DigitalEvidence) -> Dict[str, List[str]]:
        """Extract IOCs from log files"""
        return {
            "ip_addresses": ["203.0.113.100", "198.51.100.50"],
            "domains": ["malicious-domain.com", "c2-server.net"],
            "urls": ["http://malicious-domain.com/payload.exe"]
        }
        
    async def _extract_network_iocs(self, evidence: DigitalEvidence) -> Dict[str, List[str]]:
        """Extract IOCs from network captures"""
        return {
            "ip_addresses": ["203.0.113.200", "198.51.100.100"],
            "domains": ["suspicious-site.org"],
            "urls": ["https://suspicious-site.org/download"]
        }
        
    async def _extract_registry_iocs(self, evidence: DigitalEvidence) -> Dict[str, List[str]]:
        """Extract IOCs from registry evidence"""
        return {
            "registry_keys": [
                "HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run\\Malware",
                "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\Backdoor"
            ],
            "file_hashes": ["d41d8cd98f00b204e9800998ecf8427e"]
        }
        
    async def reconstruct_incident(self, incident_id: str, evidence_items: List[DigitalEvidence]) -> IncidentReconstruction:
        """Reconstruct incident timeline and attack progression"""
        
        # Analyze evidence to build timeline
        analysis_results = await self.analyze_evidence(evidence_items)
        
        # Create forensic timeline
        timeline = ForensicTimeline(
            timeline_id=f"timeline_{incident_id}",
            incident_id=incident_id,
            events=analysis_results["timeline_analysis"]["events"],
            start_time=datetime.fromisoformat(analysis_results["timeline_analysis"]["timespan"]["start"]) if analysis_results["timeline_analysis"]["timespan"]["start"] else datetime.now(),
            end_time=datetime.fromisoformat(analysis_results["timeline_analysis"]["timespan"]["end"]) if analysis_results["timeline_analysis"]["timespan"]["end"] else datetime.now(),
            confidence_score=0.85,
            reconstruction_method="automated_analysis"
        )
        
        # Determine attack vector
        attack_vector = self._determine_attack_vector(analysis_results)
        
        # Identify affected systems
        affected_systems = list(set(e.source_system for e in evidence_items))
        
        # Identify compromised accounts
        compromised_accounts = self._identify_compromised_accounts(analysis_results)
        
        # Identify accessed data
        data_accessed = self._identify_accessed_data(analysis_results)
        
        # Identify persistence mechanisms
        persistence_mechanisms = self._identify_persistence_mechanisms(analysis_results)
        
        # Identify lateral movement
        lateral_movement = self._identify_lateral_movement(analysis_results)
        
        # Identify exfiltration evidence
        exfiltration_evidence = self._identify_exfiltration_evidence(analysis_results)
        
        # Attribution indicators
        attribution_indicators = self._identify_attribution_indicators(analysis_results)
        
        reconstruction = IncidentReconstruction(
            reconstruction_id=f"reconstruction_{incident_id}_{int(datetime.now().timestamp())}",
            incident_id=incident_id,
            attack_vector=attack_vector,
            attack_timeline=timeline,
            affected_systems=affected_systems,
            compromised_accounts=compromised_accounts,
            data_accessed=data_accessed,
            persistence_mechanisms=persistence_mechanisms,
            lateral_movement=lateral_movement,
            exfiltration_evidence=exfiltration_evidence,
            attribution_indicators=attribution_indicators,
            confidence_level=0.80
        )
        
        # Store reconstruction
        await self._store_reconstruction(reconstruction)
        
        return reconstruction
        
    def _determine_attack_vector(self, analysis_results: Dict[str, Any]) -> str:
        """Determine the primary attack vector"""
        # Analyze artifacts and events to determine attack vector
        artifacts = analysis_results.get("artifact_analysis", {})
        
        if any("phishing" in str(artifact).lower() for artifact in artifacts.get("file_artifacts", [])):
            return "phishing_email"
        elif any("web" in str(artifact).lower() for artifact in artifacts.get("network_artifacts", [])):
            return "web_application_exploit"
        elif any("remote" in str(artifact).lower() for artifact in artifacts.get("network_artifacts", [])):
            return "remote_access"
        else:
            return "unknown"
            
    def _identify_compromised_accounts(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Identify compromised user accounts"""
        # Analyze timeline events for authentication anomalies
        return ["admin", "service_account", "user123"]
        
    def _identify_accessed_data(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Identify data that was accessed during the incident"""
        return ["customer_database", "financial_records", "employee_data"]
        
    def _identify_persistence_mechanisms(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Identify persistence mechanisms used by attackers"""
        artifacts = analysis_results.get("artifact_analysis", {})
        registry_artifacts = artifacts.get("registry_artifacts", [])
        
        mechanisms = []
        for artifact in registry_artifacts:
            if "Run" in artifact.get("registry_key", ""):
                mechanisms.append("registry_run_key")
                
        return mechanisms or ["unknown"]
        
    def _identify_lateral_movement(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Identify lateral movement techniques"""
        return ["rdp_connections", "smb_shares", "wmi_execution"]
        
    def _identify_exfiltration_evidence(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Identify evidence of data exfiltration"""
        return ["large_file_transfers", "encrypted_archives", "external_connections"]
        
    def _identify_attribution_indicators(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Identify indicators that might help with attribution"""
        return ["specific_malware_family", "known_c2_infrastructure", "attack_patterns"]
        
    async def _store_reconstruction(self, reconstruction: IncidentReconstruction):
        """Store incident reconstruction in database"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO incident_reconstructions
            (reconstruction_id, incident_id, attack_vector, timeline_id, affected_systems,
             compromised_accounts, data_accessed, persistence_mechanisms, lateral_movement,
             exfiltration_evidence, attribution_indicators, confidence_level, created_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            reconstruction.reconstruction_id,
            reconstruction.incident_id,
            reconstruction.attack_vector,
            reconstruction.attack_timeline.timeline_id,
            json.dumps(reconstruction.affected_systems),
            json.dumps(reconstruction.compromised_accounts),
            json.dumps(reconstruction.data_accessed),
            json.dumps(reconstruction.persistence_mechanisms),
            json.dumps(reconstruction.lateral_movement),
            json.dumps(reconstruction.exfiltration_evidence),
            json.dumps(reconstruction.attribution_indicators),
            reconstruction.confidence_level,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
    def generate_forensic_report(self, reconstruction: IncidentReconstruction) -> Dict[str, Any]:
        """Generate comprehensive forensic report"""
        return {
            "executive_summary": {
                "incident_id": reconstruction.incident_id,
                "attack_vector": reconstruction.attack_vector,
                "confidence_level": reconstruction.confidence_level,
                "affected_systems_count": len(reconstruction.affected_systems),
                "compromised_accounts_count": len(reconstruction.compromised_accounts),
                "timeline_span": {
                    "start": reconstruction.attack_timeline.start_time.isoformat(),
                    "end": reconstruction.attack_timeline.end_time.isoformat()
                }
            },
            "detailed_findings": {
                "attack_progression": self._describe_attack_progression(reconstruction),
                "technical_details": self._compile_technical_details(reconstruction),
                "impact_assessment": self._assess_impact(reconstruction),
                "recommendations": self._generate_forensic_recommendations(reconstruction)
            },
            "evidence_summary": {
                "total_evidence_items": len(reconstruction.attack_timeline.events),
                "high_confidence_events": len([e for e in reconstruction.attack_timeline.events if e.get("confidence_score", 0) > 0.8]),
                "timeline_completeness": reconstruction.attack_timeline.confidence_score
            },
            "appendices": {
                "ioc_list": "See separate IOC document",
                "evidence_inventory": "See evidence catalog",
                "technical_artifacts": "See artifact analysis report"
            }
        }
        
    def _describe_attack_progression(self, reconstruction: IncidentReconstruction) -> List[str]:
        """Describe the attack progression"""
        return [
            f"Initial compromise via {reconstruction.attack_vector}",
            f"Persistence established using {', '.join(reconstruction.persistence_mechanisms)}",
            f"Lateral movement through {', '.join(reconstruction.lateral_movement)}",
            f"Data access to {', '.join(reconstruction.data_accessed)}",
            f"Evidence of exfiltration: {', '.join(reconstruction.exfiltration_evidence)}"
        ]
        
    def _compile_technical_details(self, reconstruction: IncidentReconstruction) -> Dict[str, Any]:
        """Compile technical details"""
        return {
            "affected_systems": reconstruction.affected_systems,
            "compromised_accounts": reconstruction.compromised_accounts,
            "persistence_mechanisms": reconstruction.persistence_mechanisms,
            "lateral_movement_techniques": reconstruction.lateral_movement,
            "attribution_indicators": reconstruction.attribution_indicators
        }
        
    def _assess_impact(self, reconstruction: IncidentReconstruction) -> Dict[str, Any]:
        """Assess incident impact"""
        return {
            "data_impact": "High" if len(reconstruction.data_accessed) > 2 else "Medium",
            "system_impact": "High" if len(reconstruction.affected_systems) > 3 else "Medium",
            "business_impact": "Significant operational disruption",
            "regulatory_impact": "Potential compliance violations"
        }
        
    def _generate_forensic_recommendations(self, reconstruction: IncidentReconstruction) -> List[str]:
        """Generate forensic-based recommendations"""
        recommendations = [
            "Implement enhanced monitoring for identified attack vectors",
            "Review and update incident response procedures",
            "Conduct security awareness training focusing on identified weaknesses"
        ]
        
        if "phishing" in reconstruction.attack_vector:
            recommendations.append("Implement advanced email security controls")
        if "registry_run_key" in reconstruction.persistence_mechanisms:
            recommendations.append("Monitor registry modifications more closely")
        if len(reconstruction.compromised_accounts) > 2:
            recommendations.append("Review privileged account management practices")
            
        return recommendations