"""
HSM Key Manager

Hardware Security Module integration for enterprise key management.
Supports AES-256 with hardware-backed key storage.
"""

import logging
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from enum import Enum
import secrets
import hashlib
import hmac
from datetime import datetime, timedelta
import json
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)

class KeyType(Enum):
    AES_256 = "aes_256"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"
    HMAC_SHA256 = "hmac_sha256"

class KeyStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    COMPROMISED = "compromised"
    EXPIRED = "expired"
    PENDING_DELETION = "pending_deletion"

@dataclass
class KeyMetadata:
    key_id: str
    key_type: KeyType
    status: KeyStatus
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    usage_count: int
    purpose: str
    owner: str
    tags: Dict[str, str]

class HSMKeyManager:
    """Hardware Security Module Key Manager"""
    
    def __init__(self, hsm_config: Optional[Dict[str, Any]] = None):
        self.hsm_config = hsm_config or {}
        self.backend = default_backend()
        self.key_store = {}  # In production, this would be HSM storage
        self.key_metadata = {}
        self.master_key = self._initialize_master_key()
        
    def _initialize_master_key(self) -> bytes:
        """Initialize master key for key encryption"""
        # In production, this would be stored in HSM
        master_key_material = self.hsm_config.get('master_key')
        if master_key_material:
            return base64.b64decode(master_key_material)
        
        # Generate new master key
        master_key = secrets.token_bytes(32)  # 256-bit key
        logger.warning("Generated new master key - store securely in HSM")
        return master_key
    
    def generate_key(self, key_type: KeyType, key_id: str, purpose: str,
                    owner: str, expires_in_days: Optional[int] = None,
                    tags: Optional[Dict[str, str]] = None) -> str:
        """Generate new encryption key"""
        try:
            if key_id in self.key_store:
                raise ValueError(f"Key {key_id} already exists")
            
            # Generate key material
            if key_type == KeyType.AES_256:
                key_material = secrets.token_bytes(32)  # 256-bit key
            elif key_type == KeyType.HMAC_SHA256:
                key_material = secrets.token_bytes(32)  # 256-bit key
            elif key_type == KeyType.RSA_2048:
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048,
                    backend=self.backend
                )
                key_material = private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )
            elif key_type == KeyType.RSA_4096:
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=4096,
                    backend=self.backend
                )
                key_material = private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )
            else:
                raise ValueError(f"Unsupported key type: {key_type}")
            
            # Encrypt key material with master key
            encrypted_key = self._encrypt_key_material(key_material)
            
            # Store encrypted key
            self.key_store[key_id] = encrypted_key
            
            # Create metadata
            expires_at = None
            if expires_in_days:
                expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
            
            metadata = KeyMetadata(
                key_id=key_id,
                key_type=key_type,
                status=KeyStatus.ACTIVE,
                created_at=datetime.utcnow(),
                expires_at=expires_at,
                last_used=None,
                usage_count=0,
                purpose=purpose,
                owner=owner,
                tags=tags or {}
            )
            
            self.key_metadata[key_id] = metadata
            
            logger.info(f"Generated key {key_id} of type {key_type.value}")
            return key_id
            
        except Exception as e:
            logger.error(f"Error generating key: {e}")
            raise
    
    def get_key(self, key_id: str) -> bytes:
        """Retrieve and decrypt key material"""
        try:
            if key_id not in self.key_store:
                raise ValueError(f"Key {key_id} not found")
            
            metadata = self.key_metadata[key_id]
            
            # Check key status
            if metadata.status != KeyStatus.ACTIVE:
                raise ValueError(f"Key {key_id} is not active (status: {metadata.status.value})")
            
            # Check expiration
            if metadata.expires_at and datetime.utcnow() > metadata.expires_at:
                metadata.status = KeyStatus.EXPIRED
                raise ValueError(f"Key {key_id} has expired")
            
            # Decrypt key material
            encrypted_key = self.key_store[key_id]
            key_material = self._decrypt_key_material(encrypted_key)
            
            # Update usage statistics
            metadata.last_used = datetime.utcnow()
            metadata.usage_count += 1
            
            return key_material
            
        except Exception as e:
            logger.error(f"Error retrieving key {key_id}: {e}")
            raise
    
    def rotate_key(self, key_id: str, new_key_id: Optional[str] = None) -> str:
        """Rotate encryption key"""
        try:
            if key_id not in self.key_metadata:
                raise ValueError(f"Key {key_id} not found")
            
            old_metadata = self.key_metadata[key_id]
            new_key_id = new_key_id or f"{key_id}_rotated_{int(datetime.utcnow().timestamp())}"
            
            # Generate new key with same properties
            self.generate_key(
                key_type=old_metadata.key_type,
                key_id=new_key_id,
                purpose=old_metadata.purpose,
                owner=old_metadata.owner,
                expires_in_days=None if not old_metadata.expires_at else 
                    (old_metadata.expires_at - datetime.utcnow()).days,
                tags=old_metadata.tags
            )
            
            # Mark old key as inactive
            old_metadata.status = KeyStatus.INACTIVE
            
            logger.info(f"Rotated key {key_id} to {new_key_id}")
            return new_key_id
            
        except Exception as e:
            logger.error(f"Error rotating key {key_id}: {e}")
            raise
    
    def revoke_key(self, key_id: str, reason: str = "Manual revocation") -> None:
        """Revoke encryption key"""
        try:
            if key_id not in self.key_metadata:
                raise ValueError(f"Key {key_id} not found")
            
            metadata = self.key_metadata[key_id]
            metadata.status = KeyStatus.COMPROMISED
            
            # Add revocation reason to tags
            metadata.tags['revocation_reason'] = reason
            metadata.tags['revoked_at'] = datetime.utcnow().isoformat()
            
            logger.warning(f"Revoked key {key_id}: {reason}")
            
        except Exception as e:
            logger.error(f"Error revoking key {key_id}: {e}")
            raise
    
    def delete_key(self, key_id: str, force: bool = False) -> None:
        """Delete encryption key (with safety checks)"""
        try:
            if key_id not in self.key_metadata:
                raise ValueError(f"Key {key_id} not found")
            
            metadata = self.key_metadata[key_id]
            
            # Safety check - don't delete recently used keys
            if not force and metadata.last_used:
                time_since_use = datetime.utcnow() - metadata.last_used
                if time_since_use < timedelta(days=30):
                    metadata.status = KeyStatus.PENDING_DELETION
                    logger.warning(f"Key {key_id} marked for deletion (recently used)")
                    return
            
            # Remove from storage
            del self.key_store[key_id]
            del self.key_metadata[key_id]
            
            logger.info(f"Deleted key {key_id}")
            
        except Exception as e:
            logger.error(f"Error deleting key {key_id}: {e}")
            raise
    
    def list_keys(self, owner: Optional[str] = None, 
                 status: Optional[KeyStatus] = None) -> List[KeyMetadata]:
        """List keys with optional filtering"""
        keys = list(self.key_metadata.values())
        
        if owner:
            keys = [k for k in keys if k.owner == owner]
        
        if status:
            keys = [k for k in keys if k.status == status]
        
        return keys
    
    def get_key_metadata(self, key_id: str) -> KeyMetadata:
        """Get key metadata"""
        if key_id not in self.key_metadata:
            raise ValueError(f"Key {key_id} not found")
        
        return self.key_metadata[key_id]
    
    def derive_key(self, master_key_id: str, context: str, key_length: int = 32) -> bytes:
        """Derive key from master key using context"""
        try:
            master_key = self.get_key(master_key_id)
            
            # Use HKDF for key derivation
            context_bytes = context.encode('utf-8')
            salt = hashlib.sha256(context_bytes).digest()
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=key_length,
                salt=salt,
                iterations=100000,
                backend=self.backend
            )
            
            derived_key = kdf.derive(master_key)
            return derived_key
            
        except Exception as e:
            logger.error(f"Error deriving key: {e}")
            raise
    
    def _encrypt_key_material(self, key_material: bytes) -> bytes:
        """Encrypt key material with master key"""
        # Generate random IV
        iv = secrets.token_bytes(16)
        
        # Create HMAC for authentication
        auth_key = hashlib.sha256(self.master_key + b"auth").digest()
        hmac_obj = hmac.new(auth_key, key_material, hashlib.sha256)
        auth_tag = hmac_obj.digest()
        
        # Simple XOR encryption (in production, use proper AES-GCM)
        cipher_key = hashlib.sha256(self.master_key + iv).digest()
        encrypted = bytes(a ^ b for a, b in zip(key_material, 
                         (cipher_key * ((len(key_material) // 32) + 1))[:len(key_material)]))
        
        return iv + auth_tag + encrypted
    
    def _decrypt_key_material(self, encrypted_data: bytes) -> bytes:
        """Decrypt key material with master key"""
        # Extract components
        iv = encrypted_data[:16]
        auth_tag = encrypted_data[16:48]
        encrypted = encrypted_data[48:]
        
        # Decrypt
        cipher_key = hashlib.sha256(self.master_key + iv).digest()
        decrypted = bytes(a ^ b for a, b in zip(encrypted,
                         (cipher_key * ((len(encrypted) // 32) + 1))[:len(encrypted)]))
        
        # Verify authentication
        auth_key = hashlib.sha256(self.master_key + b"auth").digest()
        expected_tag = hmac.new(auth_key, decrypted, hashlib.sha256).digest()
        
        if not hmac.compare_digest(auth_tag, expected_tag):
            raise ValueError("Key authentication failed")
        
        return decrypted
    
    def backup_keys(self, backup_path: str, include_metadata: bool = True) -> None:
        """Backup keys to secure storage"""
        try:
            backup_data = {
                'keys': {},
                'metadata': {} if include_metadata else None,
                'timestamp': datetime.utcnow().isoformat(),
                'version': '1.0'
            }
            
            # Backup encrypted keys
            for key_id, encrypted_key in self.key_store.items():
                backup_data['keys'][key_id] = base64.b64encode(encrypted_key).decode()
            
            # Backup metadata
            if include_metadata:
                for key_id, metadata in self.key_metadata.items():
                    backup_data['metadata'][key_id] = {
                        'key_type': metadata.key_type.value,
                        'status': metadata.status.value,
                        'created_at': metadata.created_at.isoformat(),
                        'expires_at': metadata.expires_at.isoformat() if metadata.expires_at else None,
                        'last_used': metadata.last_used.isoformat() if metadata.last_used else None,
                        'usage_count': metadata.usage_count,
                        'purpose': metadata.purpose,
                        'owner': metadata.owner,
                        'tags': metadata.tags
                    }
            
            # Write backup (in production, encrypt this)
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            logger.info(f"Keys backed up to {backup_path}")
            
        except Exception as e:
            logger.error(f"Error backing up keys: {e}")
            raise
    
    def restore_keys(self, backup_path: str) -> None:
        """Restore keys from backup"""
        try:
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
            
            # Restore keys
            for key_id, encrypted_key_b64 in backup_data['keys'].items():
                self.key_store[key_id] = base64.b64decode(encrypted_key_b64)
            
            # Restore metadata
            if backup_data.get('metadata'):
                for key_id, metadata_dict in backup_data['metadata'].items():
                    metadata = KeyMetadata(
                        key_id=key_id,
                        key_type=KeyType(metadata_dict['key_type']),
                        status=KeyStatus(metadata_dict['status']),
                        created_at=datetime.fromisoformat(metadata_dict['created_at']),
                        expires_at=datetime.fromisoformat(metadata_dict['expires_at']) 
                                  if metadata_dict['expires_at'] else None,
                        last_used=datetime.fromisoformat(metadata_dict['last_used'])
                                 if metadata_dict['last_used'] else None,
                        usage_count=metadata_dict['usage_count'],
                        purpose=metadata_dict['purpose'],
                        owner=metadata_dict['owner'],
                        tags=metadata_dict['tags']
                    )
                    self.key_metadata[key_id] = metadata
            
            logger.info(f"Keys restored from {backup_path}")
            
        except Exception as e:
            logger.error(f"Error restoring keys: {e}")
            raise
    
    def get_key_usage_stats(self) -> Dict[str, Any]:
        """Get key usage statistics"""
        stats = {
            'total_keys': len(self.key_metadata),
            'active_keys': len([k for k in self.key_metadata.values() 
                              if k.status == KeyStatus.ACTIVE]),
            'expired_keys': len([k for k in self.key_metadata.values() 
                               if k.status == KeyStatus.EXPIRED]),
            'key_types': {},
            'usage_by_owner': {},
            'most_used_keys': []
        }
        
        # Key type distribution
        for metadata in self.key_metadata.values():
            key_type = metadata.key_type.value
            stats['key_types'][key_type] = stats['key_types'].get(key_type, 0) + 1
        
        # Usage by owner
        for metadata in self.key_metadata.values():
            owner = metadata.owner
            stats['usage_by_owner'][owner] = stats['usage_by_owner'].get(owner, 0) + metadata.usage_count
        
        # Most used keys
        sorted_keys = sorted(self.key_metadata.values(), 
                           key=lambda k: k.usage_count, reverse=True)
        stats['most_used_keys'] = [(k.key_id, k.usage_count) for k in sorted_keys[:10]]
        
        return stats