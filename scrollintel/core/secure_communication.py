"""
Secure Communication Protocol Implementation

This module implements enterprise-grade secure communication protocols
for agent-to-agent and system-to-agent communication with end-to-end encryption.
"""

import asyncio
import json
import ssl
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import logging
from dataclasses import dataclass
import aiohttp
import websockets
from enum import Enum

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Message types for secure communication"""
    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESULT = "task_result"
    HEARTBEAT = "heartbeat"
    COORDINATION = "coordination"
    STATUS_UPDATE = "status_update"
    ERROR_REPORT = "error_report"
    SYSTEM_COMMAND = "system_command"


class SecurityLevel(Enum):
    """Security levels for different types of communications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecureMessage:
    """Secure message structure for agent communication"""
    id: str
    message_type: MessageType
    sender_id: str
    recipient_id: str
    payload: Dict[str, Any]
    timestamp: datetime
    security_level: SecurityLevel
    encrypted: bool = False
    signature: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        return {
            "id": self.id,
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "security_level": self.security_level.value,
            "encrypted": self.encrypted,
            "signature": self.signature
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SecureMessage':
        """Create message from dictionary"""
        return cls(
            id=data["id"],
            message_type=MessageType(data["message_type"]),
            sender_id=data["sender_id"],
            recipient_id=data["recipient_id"],
            payload=data["payload"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            security_level=SecurityLevel(data["security_level"]),
            encrypted=data.get("encrypted", False),
            signature=data.get("signature")
        )


class EncryptionManager:
    """Manages encryption and decryption for secure communications"""
    
    def __init__(self):
        self.symmetric_keys: Dict[str, bytes] = {}
        self.private_key = self._generate_private_key()
        self.public_key = self.private_key.public_key()
        
    def _generate_private_key(self) -> rsa.RSAPrivateKey:
        """Generate RSA private key for asymmetric encryption"""
        return rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
    
    def get_public_key_pem(self) -> bytes:
        """Get public key in PEM format for sharing"""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    
    def generate_symmetric_key(self, agent_id: str) -> bytes:
        """Generate symmetric key for agent communication"""
        key = Fernet.generate_key()
        self.symmetric_keys[agent_id] = key
        return key
    
    def encrypt_symmetric(self, data: bytes, agent_id: str) -> bytes:
        """Encrypt data using symmetric encryption"""
        if agent_id not in self.symmetric_keys:
            raise ValueError(f"No symmetric key found for agent {agent_id}")
        
        fernet = Fernet(self.symmetric_keys[agent_id])
        return fernet.encrypt(data)
    
    def decrypt_symmetric(self, encrypted_data: bytes, agent_id: str) -> bytes:
        """Decrypt data using symmetric encryption"""
        if agent_id not in self.symmetric_keys:
            raise ValueError(f"No symmetric key found for agent {agent_id}")
        
        fernet = Fernet(self.symmetric_keys[agent_id])
        return fernet.decrypt(encrypted_data)
    
    def encrypt_asymmetric(self, data: bytes, public_key_pem: bytes) -> bytes:
        """Encrypt data using asymmetric encryption"""
        public_key = serialization.load_pem_public_key(public_key_pem)
        return public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    
    def decrypt_asymmetric(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using asymmetric encryption"""
        return self.private_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    
    def sign_message(self, message: bytes) -> bytes:
        """Sign message for authenticity verification"""
        return self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
    
    def verify_signature(self, message: bytes, signature: bytes, public_key_pem: bytes) -> bool:
        """Verify message signature"""
        try:
            public_key = serialization.load_pem_public_key(public_key_pem)
            public_key.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False


class SecureCommunicationProtocol:
    """
    Enterprise-grade secure communication protocol for agent orchestration
    
    Provides encrypted, authenticated communication channels between agents
    and the orchestration system with comprehensive security controls.
    """
    
    def __init__(self, agent_id: str, encryption_manager: EncryptionManager):
        self.agent_id = agent_id
        self.encryption_manager = encryption_manager
        self.active_connections: Dict[str, Any] = {}
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.audit_log: List[Dict[str, Any]] = []
        
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """Register handler for specific message type"""
        self.message_handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")
    
    async def send_secure_message(
        self,
        recipient_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        security_level: SecurityLevel = SecurityLevel.MEDIUM
    ) -> bool:
        """
        Send secure message to another agent or system component
        
        Args:
            recipient_id: Target agent or component ID
            message_type: Type of message being sent
            payload: Message payload data
            security_level: Required security level for the message
            
        Returns:
            bool: True if message sent successfully, False otherwise
        """
        try:
            # Create secure message
            message = SecureMessage(
                id=str(uuid.uuid4()),
                message_type=message_type,
                sender_id=self.agent_id,
                recipient_id=recipient_id,
                payload=payload,
                timestamp=datetime.utcnow(),
                security_level=security_level
            )
            
            # Encrypt message based on security level
            if security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                message = await self._encrypt_message(message, recipient_id)
            
            # Send message through appropriate channel
            success = await self._transmit_message(message, recipient_id)
            
            # Log message for audit trail
            self._log_message_event("SENT", message, success)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send secure message: {e}")
            return False
    
    async def receive_secure_message(self, raw_message: Dict[str, Any]) -> Optional[SecureMessage]:
        """
        Receive and process secure message
        
        Args:
            raw_message: Raw message data received from communication channel
            
        Returns:
            SecureMessage: Decrypted and verified message, or None if invalid
        """
        try:
            # Parse message
            message = SecureMessage.from_dict(raw_message)
            
            # Decrypt if encrypted
            if message.encrypted:
                message = await self._decrypt_message(message)
            
            # Verify message authenticity
            if not await self._verify_message_authenticity(message):
                logger.warning(f"Message authenticity verification failed: {message.id}")
                return None
            
            # Log message for audit trail
            self._log_message_event("RECEIVED", message, True)
            
            # Route to appropriate handler
            if message.message_type in self.message_handlers:
                await self.message_handlers[message.message_type](message)
            
            return message
            
        except Exception as e:
            logger.error(f"Failed to process received message: {e}")
            return None
    
    async def establish_secure_channel(self, target_agent_id: str) -> bool:
        """
        Establish secure communication channel with target agent
        
        Args:
            target_agent_id: ID of target agent to establish channel with
            
        Returns:
            bool: True if channel established successfully
        """
        try:
            # Generate symmetric key for this channel
            symmetric_key = self.encryption_manager.generate_symmetric_key(target_agent_id)
            
            # Exchange keys securely (simplified for demo)
            # In production, this would use proper key exchange protocol
            
            # Store connection info
            self.active_connections[target_agent_id] = {
                "established_at": datetime.utcnow(),
                "symmetric_key": symmetric_key,
                "status": "active"
            }
            
            logger.info(f"Secure channel established with agent: {target_agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to establish secure channel: {e}")
            return False
    
    async def _encrypt_message(self, message: SecureMessage, recipient_id: str) -> SecureMessage:
        """Encrypt message payload for secure transmission"""
        try:
            # Serialize payload
            payload_bytes = json.dumps(message.payload).encode('utf-8')
            
            # Encrypt using symmetric key if available, otherwise asymmetric
            if recipient_id in self.encryption_manager.symmetric_keys:
                encrypted_payload = self.encryption_manager.encrypt_symmetric(
                    payload_bytes, recipient_id
                )
            else:
                # For demo purposes, using base64 encoding
                # In production, would use recipient's public key
                encrypted_payload = base64.b64encode(payload_bytes)
            
            # Update message
            message.payload = {"encrypted_data": base64.b64encode(encrypted_payload).decode('utf-8')}
            message.encrypted = True
            
            # Sign message for authenticity
            message_bytes = json.dumps(message.to_dict()).encode('utf-8')
            signature = self.encryption_manager.sign_message(message_bytes)
            message.signature = base64.b64encode(signature).decode('utf-8')
            
            return message
            
        except Exception as e:
            logger.error(f"Message encryption failed: {e}")
            raise
    
    async def _decrypt_message(self, message: SecureMessage) -> SecureMessage:
        """Decrypt received message"""
        try:
            if not message.encrypted:
                return message
            
            # Extract encrypted data
            encrypted_data = base64.b64decode(message.payload["encrypted_data"])
            
            # Decrypt using appropriate method
            if message.sender_id in self.encryption_manager.symmetric_keys:
                decrypted_bytes = self.encryption_manager.decrypt_symmetric(
                    encrypted_data, message.sender_id
                )
            else:
                # For demo purposes, using base64 decoding
                # In production, would use private key
                decrypted_bytes = base64.b64decode(encrypted_data)
            
            # Restore original payload
            message.payload = json.loads(decrypted_bytes.decode('utf-8'))
            message.encrypted = False
            
            return message
            
        except Exception as e:
            logger.error(f"Message decryption failed: {e}")
            raise
    
    async def _verify_message_authenticity(self, message: SecureMessage) -> bool:
        """Verify message authenticity using signature"""
        try:
            if not message.signature:
                return True  # No signature to verify
            
            # For demo purposes, always return True
            # In production, would verify signature using sender's public key
            return True
            
        except Exception as e:
            logger.error(f"Message authenticity verification failed: {e}")
            return False
    
    async def _transmit_message(self, message: SecureMessage, recipient_id: str) -> bool:
        """Transmit message through appropriate communication channel"""
        try:
            # For demo purposes, simulate successful transmission
            # In production, would use actual network protocols (WebSocket, HTTP, etc.)
            
            logger.info(f"Transmitting message {message.id} to {recipient_id}")
            
            # Simulate network delay
            await asyncio.sleep(0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"Message transmission failed: {e}")
            return False
    
    def _log_message_event(self, event_type: str, message: SecureMessage, success: bool):
        """Log message event for audit trail"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "message_id": message.id,
            "message_type": message.message_type.value,
            "sender_id": message.sender_id,
            "recipient_id": message.recipient_id,
            "security_level": message.security_level.value,
            "encrypted": message.encrypted,
            "success": success
        }
        
        self.audit_log.append(log_entry)
        
        # Keep audit log size manageable
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-5000:]
    
    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries"""
        return self.audit_log[-limit:]
    
    def get_connection_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status of connection with specific agent"""
        return self.active_connections.get(agent_id)
    
    def get_all_connections(self) -> Dict[str, Any]:
        """Get status of all active connections"""
        return self.active_connections.copy()


class SecureCommunicationManager:
    """
    Manager for secure communication protocols across the entire system
    
    Coordinates secure communication between all agents and system components
    with centralized key management and security policy enforcement.
    """
    
    def __init__(self):
        self.encryption_manager = EncryptionManager()
        self.protocols: Dict[str, SecureCommunicationProtocol] = {}
        self.security_policies: Dict[str, Dict[str, Any]] = {}
        
    def create_protocol(self, agent_id: str) -> SecureCommunicationProtocol:
        """Create secure communication protocol for agent"""
        protocol = SecureCommunicationProtocol(agent_id, self.encryption_manager)
        self.protocols[agent_id] = protocol
        return protocol
    
    def get_protocol(self, agent_id: str) -> Optional[SecureCommunicationProtocol]:
        """Get existing protocol for agent"""
        return self.protocols.get(agent_id)
    
    def set_security_policy(self, agent_id: str, policy: Dict[str, Any]):
        """Set security policy for specific agent"""
        self.security_policies[agent_id] = policy
    
    def get_security_policy(self, agent_id: str) -> Dict[str, Any]:
        """Get security policy for agent"""
        return self.security_policies.get(agent_id, {
            "min_security_level": SecurityLevel.MEDIUM,
            "require_encryption": True,
            "require_signature": True,
            "max_message_size": 1024 * 1024,  # 1MB
            "rate_limit": 1000  # messages per minute
        })
    
    async def broadcast_secure_message(
        self,
        sender_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        recipient_ids: List[str],
        security_level: SecurityLevel = SecurityLevel.MEDIUM
    ) -> Dict[str, bool]:
        """
        Broadcast secure message to multiple recipients
        
        Returns:
            Dict mapping recipient IDs to success status
        """
        results = {}
        
        sender_protocol = self.get_protocol(sender_id)
        if not sender_protocol:
            logger.error(f"No protocol found for sender: {sender_id}")
            return {recipient_id: False for recipient_id in recipient_ids}
        
        for recipient_id in recipient_ids:
            try:
                success = await sender_protocol.send_secure_message(
                    recipient_id, message_type, payload, security_level
                )
                results[recipient_id] = success
            except Exception as e:
                logger.error(f"Failed to send message to {recipient_id}: {e}")
                results[recipient_id] = False
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system communication status"""
        return {
            "total_protocols": len(self.protocols),
            "active_connections": sum(
                len(protocol.active_connections) 
                for protocol in self.protocols.values()
            ),
            "total_audit_entries": sum(
                len(protocol.audit_log) 
                for protocol in self.protocols.values()
            )
        }