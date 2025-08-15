"""
Proof-of-Workflow (PoWf) attestation service for ScrollIntel-G6.
Provides cryptographic attestation of all production actions with hash chaining.
"""

import hashlib
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import logging

logger = logging.getLogger(__name__)


@dataclass
class WorkflowAttestation:
    """Cryptographic attestation of a workflow action."""
    
    id: str
    timestamp: datetime
    action_type: str
    agent_id: str
    user_id: str
    prompt_hash: str
    tools_used: List[str]
    datasets_used: List[str]
    model_version: str
    verifier_evidence: Dict[str, Any]
    parent_hash: Optional[str]
    content_hash: str
    signature: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowAttestation':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class ProofOfWorkflowService:
    """Service for creating and verifying workflow attestations."""
    
    def __init__(self):
        self.private_key = self._generate_or_load_key()
        self.public_key = self.private_key.public_key()
        self.attestation_chain: List[WorkflowAttestation] = []
        self.worm_storage = WORMStorage()
    
    def _generate_or_load_key(self) -> rsa.RSAPrivateKey:
        """Generate or load RSA private key for signing."""
        try:
            # Try to load existing key
            with open('.scrollintel/powf_private_key.pem', 'rb') as f:
                private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None,
                    backend=default_backend()
                )
            logger.info("Loaded existing PoWf private key")
            return private_key
        except FileNotFoundError:
            # Generate new key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            
            # Save private key
            import os
            os.makedirs('.scrollintel', exist_ok=True)
            with open('.scrollintel/powf_private_key.pem', 'wb') as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            # Save public key
            with open('.scrollintel/powf_public_key.pem', 'wb') as f:
                f.write(self.public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))
            
            logger.info("Generated new PoWf key pair")
            return private_key
    
    def create_attestation(
        self,
        action_type: str,
        agent_id: str,
        user_id: str,
        prompt: str,
        tools_used: List[str],
        datasets_used: List[str],
        model_version: str,
        verifier_evidence: Dict[str, Any],
        content: Any
    ) -> WorkflowAttestation:
        """Create a new workflow attestation."""
        
        # Generate hashes
        prompt_hash = self._hash_content(prompt)
        content_hash = self._hash_content(json.dumps(content, sort_keys=True))
        
        # Get parent hash from last attestation
        parent_hash = None
        if self.attestation_chain:
            parent_hash = self._hash_attestation(self.attestation_chain[-1])
        
        # Create attestation
        attestation = WorkflowAttestation(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            action_type=action_type,
            agent_id=agent_id,
            user_id=user_id,
            prompt_hash=prompt_hash,
            tools_used=tools_used,
            datasets_used=datasets_used,
            model_version=model_version,
            verifier_evidence=verifier_evidence,
            parent_hash=parent_hash,
            content_hash=content_hash
        )
        
        # Sign attestation
        attestation.signature = self._sign_attestation(attestation)
        
        # Add to chain
        self.attestation_chain.append(attestation)
        
        # Store in WORM
        self.worm_storage.store_attestation(attestation)
        
        logger.info(f"Created PoWf attestation {attestation.id} for {action_type}")
        return attestation
    
    def _hash_content(self, content: str) -> str:
        """Create SHA-256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _hash_attestation(self, attestation: WorkflowAttestation) -> str:
        """Create hash of attestation for chaining."""
        # Create canonical representation without signature
        data = attestation.to_dict()
        data.pop('signature', None)
        canonical = json.dumps(data, sort_keys=True)
        return self._hash_content(canonical)
    
    def _sign_attestation(self, attestation: WorkflowAttestation) -> str:
        """Sign attestation with private key."""
        # Create canonical representation without signature
        data = attestation.to_dict()
        data.pop('signature', None)
        canonical = json.dumps(data, sort_keys=True)
        
        # Sign
        signature = self.private_key.sign(
            canonical.encode('utf-8'),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return signature.hex()
    
    def verify_attestation(self, attestation: WorkflowAttestation) -> bool:
        """Verify attestation signature."""
        try:
            # Create canonical representation without signature
            data = attestation.to_dict()
            signature_hex = data.pop('signature', '')
            canonical = json.dumps(data, sort_keys=True)
            
            # Verify signature
            signature = bytes.fromhex(signature_hex)
            self.public_key.verify(
                signature,
                canonical.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception as e:
            logger.error(f"Attestation verification failed: {e}")
            return False
    
    def verify_chain_integrity(self) -> bool:
        """Verify the integrity of the entire attestation chain."""
        if not self.attestation_chain:
            return True
        
        # Verify first attestation has no parent
        if self.attestation_chain[0].parent_hash is not None:
            logger.error("First attestation should have no parent hash")
            return False
        
        # Verify chain links
        for i in range(1, len(self.attestation_chain)):
            current = self.attestation_chain[i]
            previous = self.attestation_chain[i-1]
            
            expected_parent_hash = self._hash_attestation(previous)
            if current.parent_hash != expected_parent_hash:
                logger.error(f"Chain break at attestation {i}: expected {expected_parent_hash}, got {current.parent_hash}")
                return False
            
            # Verify signature
            if not self.verify_attestation(current):
                logger.error(f"Invalid signature for attestation {i}")
                return False
        
        logger.info(f"Chain integrity verified for {len(self.attestation_chain)} attestations")
        return True
    
    def get_public_key_pem(self) -> str:
        """Get public key in PEM format for third-party verification."""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')
    
    def export_attestations(self, start_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Export attestations for public verification."""
        attestations = self.attestation_chain
        
        if start_date:
            attestations = [a for a in attestations if a.timestamp >= start_date]
        
        return [a.to_dict() for a in attestations]


class WORMStorage:
    """Write-Once-Read-Many storage for attestations."""
    
    def __init__(self, storage_path: str = '.scrollintel/worm_storage'):
        self.storage_path = storage_path
        import os
        os.makedirs(storage_path, exist_ok=True)
    
    def store_attestation(self, attestation: WorkflowAttestation) -> None:
        """Store attestation in WORM storage."""
        filename = f"{attestation.timestamp.strftime('%Y%m%d_%H%M%S')}_{attestation.id}.json"
        filepath = f"{self.storage_path}/{filename}"
        
        # Check if file already exists (WORM property)
        if os.path.exists(filepath):
            raise ValueError(f"Attestation {attestation.id} already exists in WORM storage")
        
        # Write attestation
        with open(filepath, 'w') as f:
            json.dump(attestation.to_dict(), f, indent=2)
        
        # Make file read-only
        import stat
        os.chmod(filepath, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        
        logger.debug(f"Stored attestation {attestation.id} in WORM storage")
    
    def load_attestations(self) -> List[WorkflowAttestation]:
        """Load all attestations from WORM storage."""
        attestations = []
        
        for filename in sorted(os.listdir(self.storage_path)):
            if filename.endswith('.json'):
                filepath = f"{self.storage_path}/{filename}"
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    attestations.append(WorkflowAttestation.from_dict(data))
        
        return attestations


# Global PoWf service instance
powf_service = ProofOfWorkflowService()


def create_workflow_attestation(
    action_type: str,
    agent_id: str,
    user_id: str,
    prompt: str,
    tools_used: List[str],
    datasets_used: List[str],
    model_version: str,
    verifier_evidence: Dict[str, Any],
    content: Any
) -> WorkflowAttestation:
    """Create a workflow attestation (convenience function)."""
    return powf_service.create_attestation(
        action_type=action_type,
        agent_id=agent_id,
        user_id=user_id,
        prompt=prompt,
        tools_used=tools_used,
        datasets_used=datasets_used,
        model_version=model_version,
        verifier_evidence=verifier_evidence,
        content=content
    )


def verify_workflow_attestation(attestation: WorkflowAttestation) -> bool:
    """Verify a workflow attestation (convenience function)."""
    return powf_service.verify_attestation(attestation)


def get_attestation_chain() -> List[WorkflowAttestation]:
    """Get the current attestation chain."""
    return powf_service.attestation_chain.copy()


def export_public_verifier_data() -> Dict[str, Any]:
    """Export data needed for public verification."""
    return {
        'public_key': powf_service.get_public_key_pem(),
        'attestations': powf_service.export_attestations(),
        'chain_valid': powf_service.verify_chain_integrity()
    }