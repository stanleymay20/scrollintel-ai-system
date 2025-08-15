"""
Transparency Ledger for ScrollIntel-G6.
Public verifiable changelog of model/router/policy versions, eval scores, and incidents.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

from .proof_of_workflow import create_workflow_attestation

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    MODEL_UPDATE = "model_update"
    ROUTER_UPDATE = "router_update"
    POLICY_UPDATE = "policy_update"
    EVAL_SCORE = "eval_score"
    INCIDENT = "incident"
    SECURITY_PATCH = "security_patch"
    CONFIGURATION_CHANGE = "configuration_change"


class Severity(Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LedgerEntry:
    """Entry in the transparency ledger."""
    
    id: str
    timestamp: datetime
    change_type: ChangeType
    severity: Severity
    title: str
    description: str
    component: str
    version_before: Optional[str]
    version_after: Optional[str]
    metadata: Dict[str, Any]
    public_hash: str
    verification_signature: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['change_type'] = self.change_type.value
        data['severity'] = self.severity.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LedgerEntry':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['change_type'] = ChangeType(data['change_type'])
        data['severity'] = Severity(data['severity'])
        return cls(**data)


class PublicVerifier:
    """Public verifier for transparency ledger entries."""
    
    def __init__(self):
        self.verification_rules = self._load_verification_rules()
    
    def _load_verification_rules(self) -> Dict[str, Any]:
        """Load verification rules for different entry types."""
        return {
            ChangeType.MODEL_UPDATE: {
                "required_fields": ["component", "version_before", "version_after"],
                "metadata_requirements": ["model_size", "training_data_hash", "eval_scores"]
            },
            ChangeType.EVAL_SCORE: {
                "required_fields": ["component", "metadata"],
                "metadata_requirements": ["benchmark_name", "score", "baseline_comparison"]
            },
            ChangeType.INCIDENT: {
                "required_fields": ["severity", "description"],
                "metadata_requirements": ["impact_scope", "resolution_time", "root_cause"]
            }
        }
    
    def verify_entry(self, entry: LedgerEntry) -> Dict[str, Any]:
        """Verify a ledger entry against rules."""
        verification_result = {
            "valid": True,
            "issues": [],
            "score": 1.0
        }
        
        # Check required fields
        rules = self.verification_rules.get(entry.change_type, {})
        required_fields = rules.get("required_fields", [])
        
        for field in required_fields:
            if not getattr(entry, field, None):
                verification_result["issues"].append(f"Missing required field: {field}")
                verification_result["valid"] = False
        
        # Check metadata requirements
        metadata_requirements = rules.get("metadata_requirements", [])
        for requirement in metadata_requirements:
            if requirement not in entry.metadata:
                verification_result["issues"].append(f"Missing metadata: {requirement}")
                verification_result["score"] -= 0.1
        
        # Verify hash integrity
        if not self._verify_hash(entry):
            verification_result["issues"].append("Hash verification failed")
            verification_result["valid"] = False
        
        # Calculate final score
        verification_result["score"] = max(0.0, verification_result["score"])
        
        return verification_result
    
    def _verify_hash(self, entry: LedgerEntry) -> bool:
        """Verify the public hash of an entry."""
        # Create canonical representation
        data = entry.to_dict()
        data.pop('public_hash', None)
        data.pop('verification_signature', None)
        
        canonical = json.dumps(data, sort_keys=True)
        expected_hash = hashlib.sha256(canonical.encode()).hexdigest()
        
        return entry.public_hash == expected_hash


class TransparencyLedger:
    """Main transparency ledger service."""
    
    def __init__(self):
        self.entries: List[LedgerEntry] = []
        self.verifier = PublicVerifier()
        self.public_api_enabled = True
        self.storage_path = ".scrollintel/transparency_ledger.json"
        self._load_existing_entries()
    
    def _load_existing_entries(self) -> None:
        """Load existing entries from storage."""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.entries = [LedgerEntry.from_dict(entry) for entry in data]
            logger.info(f"Loaded {len(self.entries)} transparency ledger entries")
        except FileNotFoundError:
            logger.info("No existing transparency ledger found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading transparency ledger: {e}")
    
    def _save_entries(self) -> None:
        """Save entries to storage."""
        try:
            import os
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            with open(self.storage_path, 'w') as f:
                json.dump([entry.to_dict() for entry in self.entries], f, indent=2)
            
            logger.debug("Transparency ledger saved to storage")
        except Exception as e:
            logger.error(f"Error saving transparency ledger: {e}")
    
    def add_entry(
        self,
        change_type: ChangeType,
        severity: Severity,
        title: str,
        description: str,
        component: str,
        version_before: Optional[str] = None,
        version_after: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: str = "system"
    ) -> LedgerEntry:
        """Add a new entry to the transparency ledger."""
        
        import uuid
        
        if metadata is None:
            metadata = {}
        
        # Create entry
        entry = LedgerEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            change_type=change_type,
            severity=severity,
            title=title,
            description=description,
            component=component,
            version_before=version_before,
            version_after=version_after,
            metadata=metadata,
            public_hash=""
        )
        
        # Generate public hash
        entry.public_hash = self._generate_public_hash(entry)
        
        # Verify entry
        verification = self.verifier.verify_entry(entry)
        if not verification["valid"]:
            logger.warning(f"Transparency ledger entry has issues: {verification['issues']}")
        
        # Add to ledger
        self.entries.append(entry)
        self._save_entries()
        
        # Create workflow attestation
        create_workflow_attestation(
            action_type="transparency_ledger_entry",
            agent_id="transparency_ledger",
            user_id=user_id,
            prompt=f"Ledger entry: {title}",
            tools_used=["transparency_ledger", "public_verifier"],
            datasets_used=[],
            model_version="ledger-v1.0",
            verifier_evidence={
                "verification_valid": verification["valid"],
                "verification_score": verification["score"],
                "entry_hash": entry.public_hash
            },
            content=entry.to_dict()
        )
        
        logger.info(f"Added transparency ledger entry: {title} ({change_type.value})")
        return entry
    
    def _generate_public_hash(self, entry: LedgerEntry) -> str:
        """Generate public hash for an entry."""
        # Create canonical representation without hash
        data = entry.to_dict()
        data.pop('public_hash', None)
        data.pop('verification_signature', None)
        
        canonical = json.dumps(data, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()
    
    def get_entries(
        self,
        change_type: Optional[ChangeType] = None,
        component: Optional[str] = None,
        severity: Optional[Severity] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[LedgerEntry]:
        """Get filtered entries from the ledger."""
        
        filtered_entries = self.entries
        
        # Apply filters
        if change_type:
            filtered_entries = [e for e in filtered_entries if e.change_type == change_type]
        
        if component:
            filtered_entries = [e for e in filtered_entries if e.component == component]
        
        if severity:
            filtered_entries = [e for e in filtered_entries if e.severity == severity]
        
        if start_date:
            filtered_entries = [e for e in filtered_entries if e.timestamp >= start_date]
        
        if end_date:
            filtered_entries = [e for e in filtered_entries if e.timestamp <= end_date]
        
        # Sort by timestamp (newest first) and limit
        filtered_entries.sort(key=lambda e: e.timestamp, reverse=True)
        return filtered_entries[:limit]
    
    def get_public_changelog(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get public changelog (sanitized for public consumption)."""
        
        entries = self.get_entries(limit=limit)
        public_entries = []
        
        for entry in entries:
            # Create sanitized version for public consumption
            public_entry = {
                "id": entry.id,
                "timestamp": entry.timestamp.isoformat(),
                "change_type": entry.change_type.value,
                "severity": entry.severity.value,
                "title": entry.title,
                "description": self._sanitize_description(entry.description),
                "component": entry.component,
                "version_before": entry.version_before,
                "version_after": entry.version_after,
                "public_hash": entry.public_hash,
                "metadata": self._sanitize_metadata(entry.metadata)
            }
            
            public_entries.append(public_entry)
        
        return public_entries
    
    def _sanitize_description(self, description: str) -> str:
        """Sanitize description for public consumption."""
        # Remove sensitive information
        sensitive_patterns = [
            r'password[s]?[:\s=]+\S+',
            r'key[s]?[:\s=]+\S+',
            r'token[s]?[:\s=]+\S+',
            r'secret[s]?[:\s=]+\S+',
        ]
        
        sanitized = description
        for pattern in sensitive_patterns:
            import re
            sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata for public consumption."""
        sanitized = {}
        
        # Allow certain metadata fields
        allowed_fields = [
            'model_size', 'benchmark_name', 'score', 'baseline_comparison',
            'impact_scope', 'resolution_time', 'component_version',
            'performance_improvement', 'bug_fixes_count'
        ]
        
        for key, value in metadata.items():
            if key in allowed_fields:
                sanitized[key] = value
            elif 'public_' in key:
                sanitized[key] = value
        
        return sanitized
    
    def verify_ledger_integrity(self) -> Dict[str, Any]:
        """Verify the integrity of the entire ledger."""
        
        verification_result = {
            "valid": True,
            "total_entries": len(self.entries),
            "verified_entries": 0,
            "failed_entries": 0,
            "issues": []
        }
        
        for entry in self.entries:
            entry_verification = self.verifier.verify_entry(entry)
            
            if entry_verification["valid"]:
                verification_result["verified_entries"] += 1
            else:
                verification_result["failed_entries"] += 1
                verification_result["issues"].extend([
                    f"Entry {entry.id}: {issue}" for issue in entry_verification["issues"]
                ])
        
        if verification_result["failed_entries"] > 0:
            verification_result["valid"] = False
        
        return verification_result
    
    def get_component_history(self, component: str) -> List[Dict[str, Any]]:
        """Get version history for a specific component."""
        
        component_entries = self.get_entries(component=component)
        
        history = []
        for entry in component_entries:
            if entry.change_type in [ChangeType.MODEL_UPDATE, ChangeType.ROUTER_UPDATE, ChangeType.POLICY_UPDATE]:
                history.append({
                    "timestamp": entry.timestamp.isoformat(),
                    "version_before": entry.version_before,
                    "version_after": entry.version_after,
                    "title": entry.title,
                    "description": entry.description,
                    "metadata": entry.metadata
                })
        
        return history
    
    def get_eval_scores_timeline(self, component: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get timeline of evaluation scores."""
        
        eval_entries = self.get_entries(change_type=ChangeType.EVAL_SCORE, component=component)
        
        timeline = []
        for entry in eval_entries:
            if "score" in entry.metadata and "benchmark_name" in entry.metadata:
                timeline.append({
                    "timestamp": entry.timestamp.isoformat(),
                    "component": entry.component,
                    "benchmark": entry.metadata["benchmark_name"],
                    "score": entry.metadata["score"],
                    "baseline_comparison": entry.metadata.get("baseline_comparison"),
                    "metadata": entry.metadata
                })
        
        return timeline
    
    def get_incident_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get incident summary for the last N days."""
        
        from datetime import timedelta
        start_date = datetime.utcnow() - timedelta(days=days)
        
        incidents = self.get_entries(
            change_type=ChangeType.INCIDENT,
            start_date=start_date
        )
        
        summary = {
            "total_incidents": len(incidents),
            "by_severity": {},
            "by_component": {},
            "avg_resolution_time": 0.0,
            "incidents": []
        }
        
        resolution_times = []
        
        for incident in incidents:
            # Count by severity
            severity = incident.severity.value
            summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1
            
            # Count by component
            component = incident.component
            summary["by_component"][component] = summary["by_component"].get(component, 0) + 1
            
            # Track resolution time
            if "resolution_time" in incident.metadata:
                resolution_times.append(incident.metadata["resolution_time"])
            
            # Add to incidents list
            summary["incidents"].append({
                "id": incident.id,
                "timestamp": incident.timestamp.isoformat(),
                "severity": severity,
                "component": component,
                "title": incident.title,
                "resolved": "resolution_time" in incident.metadata
            })
        
        # Calculate average resolution time
        if resolution_times:
            summary["avg_resolution_time"] = sum(resolution_times) / len(resolution_times)
        
        return summary


# Global transparency ledger instance
transparency_ledger = TransparencyLedger()


def add_model_update(
    component: str,
    version_before: str,
    version_after: str,
    title: str,
    description: str,
    metadata: Optional[Dict[str, Any]] = None,
    user_id: str = "system"
) -> LedgerEntry:
    """Add model update entry (convenience function)."""
    return transparency_ledger.add_entry(
        change_type=ChangeType.MODEL_UPDATE,
        severity=Severity.MEDIUM,
        title=title,
        description=description,
        component=component,
        version_before=version_before,
        version_after=version_after,
        metadata=metadata or {},
        user_id=user_id
    )


def add_eval_score(
    component: str,
    benchmark_name: str,
    score: float,
    baseline_comparison: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    user_id: str = "system"
) -> LedgerEntry:
    """Add evaluation score entry (convenience function)."""
    eval_metadata = {
        "benchmark_name": benchmark_name,
        "score": score,
        "baseline_comparison": baseline_comparison
    }
    if metadata:
        eval_metadata.update(metadata)
    
    return transparency_ledger.add_entry(
        change_type=ChangeType.EVAL_SCORE,
        severity=Severity.INFO,
        title=f"Evaluation: {benchmark_name}",
        description=f"Score: {score}" + (f" (vs baseline: {baseline_comparison})" if baseline_comparison else ""),
        component=component,
        metadata=eval_metadata,
        user_id=user_id
    )


def add_incident(
    component: str,
    severity: Severity,
    title: str,
    description: str,
    metadata: Optional[Dict[str, Any]] = None,
    user_id: str = "system"
) -> LedgerEntry:
    """Add incident entry (convenience function)."""
    return transparency_ledger.add_entry(
        change_type=ChangeType.INCIDENT,
        severity=severity,
        title=title,
        description=description,
        component=component,
        metadata=metadata or {},
        user_id=user_id
    )


def get_public_changelog(limit: int = 50) -> List[Dict[str, Any]]:
    """Get public changelog (convenience function)."""
    return transparency_ledger.get_public_changelog(limit)


def verify_ledger_integrity() -> Dict[str, Any]:
    """Verify ledger integrity (convenience function)."""
    return transparency_ledger.verify_ledger_integrity()