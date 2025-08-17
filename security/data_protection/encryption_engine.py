"""
Format-Preserving Encryption Engine

Maintains data utility for analytics while ensuring protection.
Implements AES-256 with HSM key management.
"""

import logging
from typing import Dict, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import hashlib
import hmac
import secrets
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64
import re

logger = logging.getLogger(__name__)

class EncryptionMode(Enum):
    AES_256_GCM = "aes_256_gcm"
    FORMAT_PRESERVING = "format_preserving"
    DETERMINISTIC = "deterministic"

@dataclass
class EncryptionConfig:
    mode: EncryptionMode
    key_id: str
    preserve_format: bool = True
    allow_analytics: bool = True

class FormatPreservingEncryption:
    """Format-preserving encryption maintaining data utility"""
    
    def __init__(self, key_manager):
        self.key_manager = key_manager
        self.backend = default_backend()
        self.format_patterns = self._initialize_format_patterns()
        
    def _initialize_format_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize format patterns for different data types"""
        return {
            'credit_card': re.compile(r'^\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}$'),
            'ssn': re.compile(r'^\d{3}-\d{2}-\d{4}$'),
            'phone': re.compile(r'^\d{3}-\d{3}-\d{4}$'),
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'date': re.compile(r'^\d{2}/\d{2}/\d{4}$'),
            'numeric': re.compile(r'^\d+$'),
            'alphanumeric': re.compile(r'^[a-zA-Z0-9]+$')
        }
    
    def encrypt_field(self, data: str, field_type: str, key_id: str, 
                     preserve_format: bool = True) -> str:
        """Encrypt field while preserving format"""
        try:
            if not data:
                return data
            
            # Get encryption key
            encryption_key = self.key_manager.get_key(key_id)
            
            if preserve_format:
                return self._format_preserving_encrypt(data, field_type, encryption_key)
            else:
                return self._standard_encrypt(data, encryption_key)
                
        except Exception as e:
            logger.error(f"Error encrypting field: {e}")
            raise
    
    def decrypt_field(self, encrypted_data: str, field_type: str, key_id: str,
                     preserve_format: bool = True) -> str:
        """Decrypt field"""
        try:
            if not encrypted_data:
                return encrypted_data
            
            # Get decryption key
            decryption_key = self.key_manager.get_key(key_id)
            
            if preserve_format:
                return self._format_preserving_decrypt(encrypted_data, field_type, decryption_key)
            else:
                return self._standard_decrypt(encrypted_data, decryption_key)
                
        except Exception as e:
            logger.error(f"Error decrypting field: {e}")
            raise
    
    def _format_preserving_encrypt(self, data: str, field_type: str, key: bytes) -> str:
        """Format-preserving encryption for specific field types"""
        if field_type == "credit_card":
            return self._encrypt_credit_card(data, key)
        elif field_type == "ssn":
            return self._encrypt_ssn(data, key)
        elif field_type == "phone":
            return self._encrypt_phone(data, key)
        elif field_type == "email":
            return self._encrypt_email(data, key)
        elif field_type == "numeric":
            return self._encrypt_numeric(data, key)
        else:
            return self._encrypt_generic_format_preserving(data, key)
    
    def _format_preserving_decrypt(self, encrypted_data: str, field_type: str, key: bytes) -> str:
        """Format-preserving decryption for specific field types"""
        if field_type == "credit_card":
            return self._decrypt_credit_card(encrypted_data, key)
        elif field_type == "ssn":
            return self._decrypt_ssn(encrypted_data, key)
        elif field_type == "phone":
            return self._decrypt_phone(encrypted_data, key)
        elif field_type == "email":
            return self._decrypt_email(encrypted_data, key)
        elif field_type == "numeric":
            return self._decrypt_numeric(encrypted_data, key)
        else:
            return self._decrypt_generic_format_preserving(encrypted_data, key)
    
    def _encrypt_credit_card(self, cc_number: str, key: bytes) -> str:
        """Encrypt credit card while preserving format"""
        # Remove separators
        clean_cc = re.sub(r'[-\s]', '', cc_number)
        
        # Keep first 4 and last 4 digits, encrypt middle
        if len(clean_cc) >= 8:
            prefix = clean_cc[:4]
            suffix = clean_cc[-4:]
            middle = clean_cc[4:-4]
            
            # Encrypt middle digits
            encrypted_middle = self._encrypt_numeric_sequence(middle, key)
            
            # Reconstruct with original format
            if '-' in cc_number:
                return f"{prefix}-{encrypted_middle[:4]}-{encrypted_middle[4:]}-{suffix}"
            elif ' ' in cc_number:
                return f"{prefix} {encrypted_middle[:4]} {encrypted_middle[4:]} {suffix}"
            else:
                return f"{prefix}{encrypted_middle}{suffix}"
        
        return self._encrypt_numeric_sequence(clean_cc, key)
    
    def _decrypt_credit_card(self, encrypted_cc: str, key: bytes) -> str:
        """Decrypt credit card"""
        # Remove separators
        clean_cc = re.sub(r'[-\s]', '', encrypted_cc)
        
        if len(clean_cc) >= 8:
            prefix = clean_cc[:4]
            suffix = clean_cc[-4:]
            middle = clean_cc[4:-4]
            
            # Decrypt middle digits
            decrypted_middle = self._decrypt_numeric_sequence(middle, key)
            
            # Reconstruct with original format
            if '-' in encrypted_cc:
                return f"{prefix}-{decrypted_middle[:4]}-{decrypted_middle[4:]}-{suffix}"
            elif ' ' in encrypted_cc:
                return f"{prefix} {decrypted_middle[:4]} {decrypted_middle[4:]} {suffix}"
            else:
                return f"{prefix}{decrypted_middle}{suffix}"
        
        return self._decrypt_numeric_sequence(clean_cc, key)
    
    def _encrypt_ssn(self, ssn: str, key: bytes) -> str:
        """Encrypt SSN while preserving format"""
        parts = ssn.split('-')
        if len(parts) == 3:
            encrypted_parts = [self._encrypt_numeric_sequence(part, key) for part in parts]
            return '-'.join(encrypted_parts)
        return self._encrypt_numeric_sequence(ssn.replace('-', ''), key)
    
    def _decrypt_ssn(self, encrypted_ssn: str, key: bytes) -> str:
        """Decrypt SSN"""
        parts = encrypted_ssn.split('-')
        if len(parts) == 3:
            decrypted_parts = [self._decrypt_numeric_sequence(part, key) for part in parts]
            return '-'.join(decrypted_parts)
        return self._decrypt_numeric_sequence(encrypted_ssn.replace('-', ''), key)
    
    def _encrypt_phone(self, phone: str, key: bytes) -> str:
        """Encrypt phone while preserving format"""
        parts = phone.split('-')
        if len(parts) == 3:
            encrypted_parts = [self._encrypt_numeric_sequence(part, key) for part in parts]
            return '-'.join(encrypted_parts)
        return self._encrypt_numeric_sequence(phone.replace('-', ''), key)
    
    def _decrypt_phone(self, encrypted_phone: str, key: bytes) -> str:
        """Decrypt phone"""
        parts = encrypted_phone.split('-')
        if len(parts) == 3:
            decrypted_parts = [self._decrypt_numeric_sequence(part, key) for part in parts]
            return '-'.join(decrypted_parts)
        return self._decrypt_numeric_sequence(encrypted_phone.replace('-', ''), key)
    
    def _encrypt_email(self, email: str, key: bytes) -> str:
        """Encrypt email while preserving format"""
        local, domain = email.split('@', 1)
        encrypted_local = self._encrypt_alphanumeric_sequence(local, key)
        return f"{encrypted_local}@{domain}"
    
    def _decrypt_email(self, encrypted_email: str, key: bytes) -> str:
        """Decrypt email"""
        local, domain = encrypted_email.split('@', 1)
        decrypted_local = self._decrypt_alphanumeric_sequence(local, key)
        return f"{decrypted_local}@{domain}"
    
    def _encrypt_numeric(self, data: str, key: bytes) -> str:
        """Encrypt numeric data while preserving format"""
        return self._encrypt_numeric_sequence(data, key)
    
    def _decrypt_numeric(self, encrypted_data: str, key: bytes) -> str:
        """Decrypt numeric data"""
        return self._decrypt_numeric_sequence(encrypted_data, key)
    
    def _encrypt_numeric_sequence(self, sequence: str, key: bytes) -> str:
        """Encrypt numeric sequence using format-preserving encryption"""
        if not sequence.isdigit():
            return sequence
        
        # Convert to integer
        num = int(sequence)
        length = len(sequence)
        
        # Use Feistel network for format-preserving encryption
        encrypted_num = self._feistel_encrypt(num, key, 10)  # Base 10 for digits
        
        # Ensure result fits in original length
        max_val = 10 ** length - 1
        encrypted_num = encrypted_num % max_val
        
        # Pad to original length
        return str(encrypted_num).zfill(length)
    
    def _decrypt_numeric_sequence(self, encrypted_sequence: str, key: bytes) -> str:
        """Decrypt numeric sequence"""
        if not encrypted_sequence.isdigit():
            return encrypted_sequence
        
        num = int(encrypted_sequence)
        length = len(encrypted_sequence)
        
        # Use Feistel network for decryption
        decrypted_num = self._feistel_decrypt(num, key, 10)
        
        # Ensure result fits in original length
        max_val = 10 ** length - 1
        decrypted_num = decrypted_num % max_val
        
        return str(decrypted_num).zfill(length)
    
    def _encrypt_alphanumeric_sequence(self, sequence: str, key: bytes) -> str:
        """Encrypt alphanumeric sequence"""
        if not sequence.isalnum():
            return sequence
            
        try:
            # Convert to base-36 number
            num = int(sequence, 36)
            encrypted_num = self._feistel_encrypt(num, key, 36)
            
            # Ensure result fits in original length
            max_val = 36 ** len(sequence) - 1
            encrypted_num = encrypted_num % max_val
            
            # Convert back to base-36 string
            if encrypted_num == 0:
                return "0" * len(sequence)
                
            result = ""
            temp_num = encrypted_num
            while temp_num > 0:
                result = "0123456789abcdefghijklmnopqrstuvwxyz"[temp_num % 36] + result
                temp_num //= 36
            
            return result.zfill(len(sequence))
        except ValueError:
            # Fallback for non-alphanumeric
            return sequence
    
    def _decrypt_alphanumeric_sequence(self, encrypted_sequence: str, key: bytes) -> str:
        """Decrypt alphanumeric sequence"""
        if not encrypted_sequence.isalnum():
            return encrypted_sequence
            
        try:
            # Convert from base-36
            num = int(encrypted_sequence, 36)
            decrypted_num = self._feistel_decrypt(num, key, 36)
            
            # Ensure result fits in original length
            max_val = 36 ** len(encrypted_sequence) - 1
            decrypted_num = decrypted_num % max_val
            
            # Convert back to base-36 string
            if decrypted_num == 0:
                return "0" * len(encrypted_sequence)
                
            result = ""
            temp_num = decrypted_num
            while temp_num > 0:
                result = "0123456789abcdefghijklmnopqrstuvwxyz"[temp_num % 36] + result
                temp_num //= 36
            
            return result.zfill(len(encrypted_sequence))
        except ValueError:
            return encrypted_sequence
    
    def _feistel_encrypt(self, plaintext: int, key: bytes, base: int) -> int:
        """Feistel network encryption for format-preserving encryption"""
        # Split plaintext into left and right halves
        str_plain = str(plaintext)
        mid = len(str_plain) // 2
        
        if mid == 0:
            return plaintext
        
        left = int(str_plain[:mid]) if str_plain[:mid] else 0
        right = int(str_plain[mid:]) if str_plain[mid:] else 0
        
        # Perform Feistel rounds
        for round_num in range(8):  # 8 rounds for security
            # Generate round key
            round_key = self._generate_round_key(key, round_num)
            
            # Feistel function
            f_output = self._feistel_function(right, round_key, base)
            
            # Feistel step
            new_left = right
            new_right = (left + f_output) % (base ** len(str_plain[mid:]))
            
            left, right = new_left, new_right
        
        # Combine halves
        left_str = str(left).zfill(len(str_plain[:mid]))
        right_str = str(right).zfill(len(str_plain[mid:]))
        
        return int(left_str + right_str)
    
    def _feistel_decrypt(self, ciphertext: int, key: bytes, base: int) -> int:
        """Feistel network decryption"""
        str_cipher = str(ciphertext)
        mid = len(str_cipher) // 2
        
        if mid == 0:
            return ciphertext
        
        left = int(str_cipher[:mid]) if str_cipher[:mid] else 0
        right = int(str_cipher[mid:]) if str_cipher[mid:] else 0
        
        # Reverse Feistel rounds
        for round_num in range(7, -1, -1):  # Reverse order
            round_key = self._generate_round_key(key, round_num)
            f_output = self._feistel_function(left, round_key, base)
            
            new_right = left
            new_left = (right - f_output) % (base ** len(str_cipher[mid:]))
            
            left, right = new_left, new_right
        
        # Combine halves
        left_str = str(left).zfill(len(str_cipher[:mid]))
        right_str = str(right).zfill(len(str_cipher[mid:]))
        
        return int(left_str + right_str)
    
    def _feistel_function(self, input_val: int, round_key: bytes, base: int) -> int:
        """Feistel function using HMAC"""
        input_bytes = str(input_val).encode()
        hmac_result = hmac.new(round_key, input_bytes, hashlib.sha256).digest()
        return int.from_bytes(hmac_result[:4], 'big') % (base ** 4)
    
    def _generate_round_key(self, master_key: bytes, round_num: int) -> bytes:
        """Generate round key for Feistel network"""
        round_bytes = round_num.to_bytes(4, 'big')
        return hashlib.sha256(master_key + round_bytes).digest()[:16]
    
    def _encrypt_generic_format_preserving(self, data: str, key: bytes) -> str:
        """Generic format-preserving encryption"""
        if data.isdigit():
            return self._encrypt_numeric_sequence(data, key)
        elif data.isalnum():
            return self._encrypt_alphanumeric_sequence(data, key)
        else:
            # For complex formats, use standard encryption
            return self._standard_encrypt(data, key)
    
    def _decrypt_generic_format_preserving(self, encrypted_data: str, key: bytes) -> str:
        """Generic format-preserving decryption"""
        if encrypted_data.isdigit():
            return self._decrypt_numeric_sequence(encrypted_data, key)
        elif encrypted_data.isalnum():
            return self._decrypt_alphanumeric_sequence(encrypted_data, key)
        else:
            return self._standard_decrypt(encrypted_data, key)
    
    def _standard_encrypt(self, data: str, key: bytes) -> str:
        """Standard AES-256-GCM encryption"""
        # Generate random IV
        iv = secrets.token_bytes(12)  # 96-bit IV for GCM
        
        # Create cipher
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        
        # Encrypt data
        ciphertext = encryptor.update(data.encode()) + encryptor.finalize()
        
        # Combine IV, tag, and ciphertext
        encrypted_data = iv + encryptor.tag + ciphertext
        
        # Return base64 encoded
        return base64.b64encode(encrypted_data).decode()
    
    def _standard_decrypt(self, encrypted_data: str, key: bytes) -> str:
        """Standard AES-256-GCM decryption"""
        # Decode base64
        encrypted_bytes = base64.b64decode(encrypted_data.encode())
        
        # Extract components
        iv = encrypted_bytes[:12]
        tag = encrypted_bytes[12:28]
        ciphertext = encrypted_bytes[28:]
        
        # Create cipher
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=self.backend)
        decryptor = cipher.decryptor()
        
        # Decrypt data
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        return plaintext.decode()
    
    def encrypt_for_analytics(self, data: str, key_id: str) -> str:
        """Encrypt data while preserving analytical properties"""
        # Use deterministic encryption for analytics
        key = self.key_manager.get_key(key_id)
        
        # Create deterministic encryption using HMAC
        hmac_result = hmac.new(key, data.encode(), hashlib.sha256).hexdigest()
        
        # Preserve some statistical properties
        if data.isdigit():
            # For numeric data, preserve order relationships
            num_val = int(data)
            encrypted_num = (num_val + int(hmac_result[:8], 16)) % (10 ** len(data))
            return str(encrypted_num).zfill(len(data))
        
        return hmac_result[:len(data) * 2]  # Preserve length relationship
    
    def batch_encrypt(self, data_dict: Dict[str, str], field_types: Dict[str, str], 
                     key_id: str) -> Dict[str, str]:
        """Batch encrypt multiple fields"""
        encrypted_data = {}
        
        for field_name, field_value in data_dict.items():
            field_type = field_types.get(field_name, 'generic')
            encrypted_data[field_name] = self.encrypt_field(
                field_value, field_type, key_id
            )
        
        return encrypted_data
    
    def batch_decrypt(self, encrypted_dict: Dict[str, str], field_types: Dict[str, str],
                     key_id: str) -> Dict[str, str]:
        """Batch decrypt multiple fields"""
        decrypted_data = {}
        
        for field_name, encrypted_value in encrypted_dict.items():
            field_type = field_types.get(field_name, 'generic')
            decrypted_data[field_name] = self.decrypt_field(
                encrypted_value, field_type, key_id
            )
        
        return decrypted_data