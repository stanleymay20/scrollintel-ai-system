"""
Multi-Factor Authentication System
Implements TOTP, SMS, and biometric authentication options
"""

import pyotp
import qrcode
import io
import base64
import hashlib
import hmac
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import secrets
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MFAMethod(Enum):
    TOTP = "totp"
    SMS = "sms"
    BIOMETRIC = "biometric"
    BACKUP_CODES = "backup_codes"

@dataclass
class MFAChallenge:
    challenge_id: str
    user_id: str
    method: MFAMethod
    challenge_data: Dict[str, Any]
    expires_at: datetime
    attempts: int = 0
    max_attempts: int = 3

@dataclass
class BiometricTemplate:
    template_id: str
    user_id: str
    template_type: str  # fingerprint, face, voice
    template_data: bytes
    created_at: datetime
    last_used: datetime

class MFASystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_challenges: Dict[str, MFAChallenge] = {}
        self.biometric_templates: Dict[str, List[BiometricTemplate]] = {}
        self.backup_codes: Dict[str, List[str]] = {}
        self.totp_secrets: Dict[str, str] = {}
        
    def setup_totp(self, user_id: str, issuer_name: str = "ScrollIntel") -> Tuple[str, str]:
        """Setup TOTP for a user and return secret and QR code"""
        try:
            # Generate secret
            secret = pyotp.random_base32()
            self.totp_secrets[user_id] = secret
            
            # Create TOTP URI
            totp = pyotp.TOTP(secret)
            provisioning_uri = totp.provisioning_uri(
                name=user_id,
                issuer_name=issuer_name
            )
            
            # Generate QR code
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(provisioning_uri)
            qr.make(fit=True)
            
            img = qr.make_image(fill_color="black", back_color="white")
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            qr_code_b64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            logger.info(f"TOTP setup completed for user {user_id}")
            return secret, qr_code_b64
            
        except Exception as e:
            logger.error(f"TOTP setup failed for user {user_id}: {str(e)}")
            raise
    
    def verify_totp(self, user_id: str, token: str, window: int = 1) -> bool:
        """Verify TOTP token"""
        try:
            if user_id not in self.totp_secrets:
                logger.warning(f"No TOTP secret found for user {user_id}")
                return False
            
            secret = self.totp_secrets[user_id]
            totp = pyotp.TOTP(secret)
            
            # Verify with time window for clock drift
            is_valid = totp.verify(token, valid_window=window)
            
            if is_valid:
                logger.info(f"TOTP verification successful for user {user_id}")
            else:
                logger.warning(f"TOTP verification failed for user {user_id}")
                
            return is_valid
            
        except Exception as e:
            logger.error(f"TOTP verification error for user {user_id}: {str(e)}")
            return False
    
    def initiate_sms_challenge(self, user_id: str, phone_number: str) -> str:
        """Initiate SMS-based MFA challenge"""
        try:
            # Generate 6-digit code
            code = f"{secrets.randbelow(1000000):06d}"
            
            # Create challenge
            challenge_id = secrets.token_urlsafe(32)
            challenge = MFAChallenge(
                challenge_id=challenge_id,
                user_id=user_id,
                method=MFAMethod.SMS,
                challenge_data={
                    "code": code,
                    "phone_number": phone_number,
                    "code_hash": hashlib.sha256(code.encode()).hexdigest()
                },
                expires_at=datetime.utcnow() + timedelta(minutes=5)
            )
            
            self.active_challenges[challenge_id] = challenge
            
            # In production, integrate with SMS service (Twilio, AWS SNS, etc.)
            logger.info(f"SMS challenge initiated for user {user_id}, phone {phone_number}")
            logger.debug(f"SMS code for testing: {code}")  # Remove in production
            
            return challenge_id
            
        except Exception as e:
            logger.error(f"SMS challenge initiation failed for user {user_id}: {str(e)}")
            raise
    
    def verify_sms_challenge(self, challenge_id: str, provided_code: str) -> bool:
        """Verify SMS challenge code"""
        try:
            if challenge_id not in self.active_challenges:
                logger.warning(f"SMS challenge not found: {challenge_id}")
                return False
            
            challenge = self.active_challenges[challenge_id]
            
            # Check expiration
            if datetime.utcnow() > challenge.expires_at:
                logger.warning(f"SMS challenge expired: {challenge_id}")
                del self.active_challenges[challenge_id]
                return False
            
            # Check attempts
            if challenge.attempts >= challenge.max_attempts:
                logger.warning(f"SMS challenge max attempts exceeded: {challenge_id}")
                del self.active_challenges[challenge_id]
                return False
            
            challenge.attempts += 1
            
            # Verify code using constant-time comparison
            expected_hash = challenge.challenge_data["code_hash"]
            provided_hash = hashlib.sha256(provided_code.encode()).hexdigest()
            
            is_valid = hmac.compare_digest(expected_hash, provided_hash)
            
            if is_valid:
                logger.info(f"SMS challenge verification successful: {challenge_id}")
                del self.active_challenges[challenge_id]
            else:
                logger.warning(f"SMS challenge verification failed: {challenge_id}")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"SMS challenge verification error: {str(e)}")
            return False
    
    def register_biometric_template(self, user_id: str, template_type: str, 
                                  template_data: bytes) -> str:
        """Register biometric template for user"""
        try:
            template_id = secrets.token_urlsafe(32)
            template = BiometricTemplate(
                template_id=template_id,
                user_id=user_id,
                template_type=template_type,
                template_data=template_data,
                created_at=datetime.utcnow(),
                last_used=datetime.utcnow()
            )
            
            if user_id not in self.biometric_templates:
                self.biometric_templates[user_id] = []
            
            self.biometric_templates[user_id].append(template)
            
            logger.info(f"Biometric template registered for user {user_id}, type {template_type}")
            return template_id
            
        except Exception as e:
            logger.error(f"Biometric template registration failed: {str(e)}")
            raise
    
    def verify_biometric(self, user_id: str, template_type: str, 
                        biometric_data: bytes, threshold: float = 0.8) -> bool:
        """Verify biometric authentication"""
        try:
            if user_id not in self.biometric_templates:
                logger.warning(f"No biometric templates found for user {user_id}")
                return False
            
            user_templates = self.biometric_templates[user_id]
            matching_templates = [t for t in user_templates if t.template_type == template_type]
            
            if not matching_templates:
                logger.warning(f"No {template_type} templates found for user {user_id}")
                return False
            
            # Simulate biometric matching (in production, use specialized biometric SDK)
            for template in matching_templates:
                similarity_score = self._calculate_biometric_similarity(
                    template.template_data, biometric_data
                )
                
                if similarity_score >= threshold:
                    template.last_used = datetime.utcnow()
                    logger.info(f"Biometric verification successful for user {user_id}")
                    return True
            
            logger.warning(f"Biometric verification failed for user {user_id}")
            return False
            
        except Exception as e:
            logger.error(f"Biometric verification error: {str(e)}")
            return False
    
    def generate_backup_codes(self, user_id: str, count: int = 10) -> List[str]:
        """Generate backup codes for user"""
        try:
            codes = []
            for _ in range(count):
                # Generate 8-character alphanumeric code
                code = secrets.token_hex(4).upper()
                codes.append(code)
            
            # Store hashed versions
            hashed_codes = [hashlib.sha256(code.encode()).hexdigest() for code in codes]
            self.backup_codes[user_id] = hashed_codes
            
            logger.info(f"Generated {count} backup codes for user {user_id}")
            return codes
            
        except Exception as e:
            logger.error(f"Backup code generation failed: {str(e)}")
            raise
    
    def verify_backup_code(self, user_id: str, provided_code: str) -> bool:
        """Verify and consume backup code"""
        try:
            if user_id not in self.backup_codes:
                logger.warning(f"No backup codes found for user {user_id}")
                return False
            
            provided_hash = hashlib.sha256(provided_code.encode()).hexdigest()
            user_codes = self.backup_codes[user_id]
            
            if provided_hash in user_codes:
                # Remove used code
                user_codes.remove(provided_hash)
                logger.info(f"Backup code verification successful for user {user_id}")
                return True
            
            logger.warning(f"Backup code verification failed for user {user_id}")
            return False
            
        except Exception as e:
            logger.error(f"Backup code verification error: {str(e)}")
            return False
    
    def get_user_mfa_methods(self, user_id: str) -> List[MFAMethod]:
        """Get available MFA methods for user"""
        methods = []
        
        if user_id in self.totp_secrets:
            methods.append(MFAMethod.TOTP)
        
        if user_id in self.biometric_templates:
            methods.append(MFAMethod.BIOMETRIC)
        
        if user_id in self.backup_codes and self.backup_codes[user_id]:
            methods.append(MFAMethod.BACKUP_CODES)
        
        # SMS is always available if phone number is provided
        methods.append(MFAMethod.SMS)
        
        return methods
    
    def _calculate_biometric_similarity(self, template1: bytes, template2: bytes) -> float:
        """Calculate similarity between biometric templates (simplified implementation)"""
        # In production, use specialized biometric matching algorithms
        # This is a simplified implementation for demonstration
        
        if len(template1) != len(template2):
            return 0.0
        
        matching_bytes = sum(1 for a, b in zip(template1, template2) if a == b)
        similarity = matching_bytes / len(template1)
        
        return similarity
    
    def cleanup_expired_challenges(self):
        """Remove expired challenges"""
        current_time = datetime.utcnow()
        expired_challenges = [
            challenge_id for challenge_id, challenge in self.active_challenges.items()
            if current_time > challenge.expires_at
        ]
        
        for challenge_id in expired_challenges:
            del self.active_challenges[challenge_id]
        
        if expired_challenges:
            logger.info(f"Cleaned up {len(expired_challenges)} expired challenges")