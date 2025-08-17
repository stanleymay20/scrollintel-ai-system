"""
Secure Secrets Management System
Integration with HashiCorp Vault and AWS Secrets Manager
"""

import asyncio
import json
import base64
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import hmac
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
# import hvac  # HashiCorp Vault client
# import boto3  # AWS SDK
# from botocore.exceptions import ClientError

class SecretType(Enum):
    DATABASE_CREDENTIALS = "database_credentials"
    API_KEYS = "api_keys"
    ENCRYPTION_KEYS = "encryption_keys"
    CERTIFICATES = "certificates"
    OAUTH_TOKENS = "oauth_tokens"
    WEBHOOK_SECRETS = "webhook_secrets"
    SERVICE_ACCOUNTS = "service_accounts"
    CONFIGURATION = "configuration"

class SecretProvider(Enum):
    VAULT = "hashicorp_vault"
    AWS_SECRETS_MANAGER = "aws_secrets_manager"
    AZURE_KEY_VAULT = "azure_key_vault"
    LOCAL_ENCRYPTED = "local_encrypted"

@dataclass
class SecretMetadata:
    secret_id: str
    name: str
    secret_type: SecretType
    provider: SecretProvider
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    rotation_interval: Optional[int] = None  # days
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    tags: Dict[str, str] = field(default_factory=dict)
    permissions: List[str] = field(default_factory=list)

@dataclass
class SecretValue:
    value: Union[str, Dict[str, Any]]
    metadata: SecretMetadata
    version: str = "1"

class SecretRotationPolicy:
    """Policy for automatic secret rotation"""
    
    def __init__(self, secret_type: SecretType, rotation_interval: int, 
                 rotation_function: Optional[callable] = None):
        self.secret_type = secret_type
        self.rotation_interval = rotation_interval  # days
        self.rotation_function = rotation_function
        self.last_rotation = None
        self.next_rotation = None

class SecretsManager:
    """Enterprise secrets management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers = {}
        self.local_encryption_key = None
        self.secret_cache = {}
        self.cache_ttl = config.get('cache_ttl', 300)  # 5 minutes default
        self.rotation_policies = {}
        
        # Initialize providers
        self._initialize_providers()
        
        # Initialize local encryption
        self._initialize_local_encryption()
        
        # Start background tasks
        asyncio.create_task(self._background_tasks())
    
    def _initialize_providers(self):
        """Initialize secret providers"""
        
        # HashiCorp Vault
        vault_config = self.config.get('vault', {})
        if vault_config.get('enabled', False):
            self.providers[SecretProvider.VAULT] = VaultProvider(vault_config)
        
        # AWS Secrets Manager
        aws_config = self.config.get('aws_secrets_manager', {})
        if aws_config.get('enabled', False):
            self.providers[SecretProvider.AWS_SECRETS_MANAGER] = AWSSecretsProvider(aws_config)
        
        # Azure Key Vault
        azure_config = self.config.get('azure_key_vault', {})
        if azure_config.get('enabled', False):
            self.providers[SecretProvider.AZURE_KEY_VAULT] = AzureKeyVaultProvider(azure_config)
        
        # Local encrypted storage (fallback)
        self.providers[SecretProvider.LOCAL_ENCRYPTED] = LocalEncryptedProvider(
            self.config.get('local_storage', {})
        )
    
    def _initialize_local_encryption(self):
        """Initialize local encryption for caching and fallback storage"""
        
        # Get or generate encryption key
        key_material = self.config.get('encryption_key')
        if not key_material:
            # Generate from password and salt
            password = self.config.get('master_password', 'default-password').encode()
            salt = self.config.get('salt', b'default-salt-16b')
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
        else:
            key = key_material.encode()
        
        self.local_encryption_key = Fernet(key)
    
    async def store_secret(self, name: str, value: Union[str, Dict[str, Any]], 
                          secret_type: SecretType, provider: Optional[SecretProvider] = None,
                          expires_at: Optional[datetime] = None,
                          tags: Optional[Dict[str, str]] = None) -> str:
        """Store a secret"""
        
        # Choose provider
        if provider is None:
            provider = self._choose_provider(secret_type)
        
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not available")
        
        # Generate secret ID
        secret_id = self._generate_secret_id(name, secret_type)
        
        # Create metadata
        metadata = SecretMetadata(
            secret_id=secret_id,
            name=name,
            secret_type=secret_type,
            provider=provider,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            expires_at=expires_at,
            tags=tags or {}
        )
        
        # Store secret
        await self.providers[provider].store_secret(secret_id, value, metadata)
        
        # Clear cache
        self._clear_cache(secret_id)
        
        return secret_id
    
    async def get_secret(self, secret_id: str, use_cache: bool = True) -> Optional[SecretValue]:
        """Retrieve a secret"""
        
        # Check cache first
        if use_cache and secret_id in self.secret_cache:
            cached_entry = self.secret_cache[secret_id]
            if time.time() - cached_entry['timestamp'] < self.cache_ttl:
                # Update access tracking
                cached_entry['secret'].metadata.last_accessed = datetime.now()
                cached_entry['secret'].metadata.access_count += 1
                return cached_entry['secret']
        
        # Try each provider until found
        for provider_type, provider in self.providers.items():
            try:
                secret = await provider.get_secret(secret_id)
                if secret:
                    # Cache the secret
                    if use_cache:
                        self.secret_cache[secret_id] = {
                            'secret': secret,
                            'timestamp': time.time()
                        }
                    
                    # Update access tracking
                    secret.metadata.last_accessed = datetime.now()
                    secret.metadata.access_count += 1
                    
                    return secret
            except Exception as e:
                print(f"Error retrieving secret from {provider_type}: {e}")
                continue
        
        return None
    
    async def update_secret(self, secret_id: str, value: Union[str, Dict[str, Any]]) -> bool:
        """Update a secret value"""
        
        # Get current secret to find provider
        current_secret = await self.get_secret(secret_id, use_cache=False)
        if not current_secret:
            return False
        
        provider = self.providers[current_secret.metadata.provider]
        
        # Update metadata
        current_secret.metadata.updated_at = datetime.now()
        
        # Update secret
        success = await provider.update_secret(secret_id, value, current_secret.metadata)
        
        if success:
            # Clear cache
            self._clear_cache(secret_id)
        
        return success
    
    async def delete_secret(self, secret_id: str) -> bool:
        """Delete a secret"""
        
        # Get current secret to find provider
        current_secret = await self.get_secret(secret_id, use_cache=False)
        if not current_secret:
            return False
        
        provider = self.providers[current_secret.metadata.provider]
        
        # Delete secret
        success = await provider.delete_secret(secret_id)
        
        if success:
            # Clear cache
            self._clear_cache(secret_id)
        
        return success
    
    async def list_secrets(self, secret_type: Optional[SecretType] = None,
                          tags: Optional[Dict[str, str]] = None) -> List[SecretMetadata]:
        """List secrets with optional filtering"""
        
        all_secrets = []
        
        for provider in self.providers.values():
            try:
                secrets = await provider.list_secrets()
                all_secrets.extend(secrets)
            except Exception as e:
                print(f"Error listing secrets from provider: {e}")
                continue
        
        # Apply filters
        filtered_secrets = all_secrets
        
        if secret_type:
            filtered_secrets = [s for s in filtered_secrets if s.secret_type == secret_type]
        
        if tags:
            filtered_secrets = [
                s for s in filtered_secrets
                if all(s.tags.get(k) == v for k, v in tags.items())
            ]
        
        return filtered_secrets
    
    async def rotate_secret(self, secret_id: str, new_value: Optional[Union[str, Dict[str, Any]]] = None) -> bool:
        """Rotate a secret (update with new value)"""
        
        current_secret = await self.get_secret(secret_id, use_cache=False)
        if not current_secret:
            return False
        
        # Generate new value if not provided
        if new_value is None:
            rotation_policy = self.rotation_policies.get(current_secret.metadata.secret_type)
            if rotation_policy and rotation_policy.rotation_function:
                new_value = await rotation_policy.rotation_function(current_secret)
            else:
                # Default rotation for different secret types
                new_value = await self._generate_rotated_value(current_secret)
        
        # Update secret
        success = await self.update_secret(secret_id, new_value)
        
        if success:
            # Update rotation tracking
            if current_secret.metadata.secret_type in self.rotation_policies:
                policy = self.rotation_policies[current_secret.metadata.secret_type]
                policy.last_rotation = datetime.now()
                policy.next_rotation = datetime.now() + timedelta(days=policy.rotation_interval)
        
        return success
    
    def add_rotation_policy(self, policy: SecretRotationPolicy):
        """Add automatic rotation policy"""
        self.rotation_policies[policy.secret_type] = policy
    
    async def _generate_rotated_value(self, current_secret: SecretValue) -> Union[str, Dict[str, Any]]:
        """Generate new rotated value based on secret type"""
        
        if current_secret.metadata.secret_type == SecretType.API_KEYS:
            # Generate new API key
            import secrets
            return secrets.token_urlsafe(32)
        
        elif current_secret.metadata.secret_type == SecretType.DATABASE_CREDENTIALS:
            # Generate new password, keep username
            if isinstance(current_secret.value, dict):
                new_value = current_secret.value.copy()
                new_value['password'] = self._generate_secure_password()
                return new_value
            else:
                return self._generate_secure_password()
        
        elif current_secret.metadata.secret_type == SecretType.ENCRYPTION_KEYS:
            # Generate new encryption key
            return Fernet.generate_key().decode()
        
        else:
            # Default: generate secure random string
            import secrets
            return secrets.token_urlsafe(32)
    
    def _generate_secure_password(self, length: int = 16) -> str:
        """Generate secure password"""
        import secrets
        import string
        
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        password = ''.join(secrets.choice(alphabet) for _ in range(length))
        return password
    
    def _choose_provider(self, secret_type: SecretType) -> SecretProvider:
        """Choose appropriate provider based on secret type and availability"""
        
        # Preference order based on secret type
        preferences = {
            SecretType.DATABASE_CREDENTIALS: [SecretProvider.VAULT, SecretProvider.AWS_SECRETS_MANAGER],
            SecretType.API_KEYS: [SecretProvider.VAULT, SecretProvider.AWS_SECRETS_MANAGER],
            SecretType.ENCRYPTION_KEYS: [SecretProvider.VAULT, SecretProvider.AZURE_KEY_VAULT],
            SecretType.CERTIFICATES: [SecretProvider.VAULT, SecretProvider.AZURE_KEY_VAULT],
            SecretType.OAUTH_TOKENS: [SecretProvider.AWS_SECRETS_MANAGER, SecretProvider.VAULT],
        }
        
        preferred_providers = preferences.get(secret_type, [SecretProvider.VAULT])
        
        # Return first available preferred provider
        for provider in preferred_providers:
            if provider in self.providers:
                return provider
        
        # Fallback to any available provider
        if self.providers:
            return next(iter(self.providers.keys()))
        
        raise ValueError("No secret providers available")
    
    def _generate_secret_id(self, name: str, secret_type: SecretType) -> str:
        """Generate unique secret ID"""
        content = f"{name}:{secret_type.value}:{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _clear_cache(self, secret_id: str):
        """Clear cached secret"""
        if secret_id in self.secret_cache:
            del self.secret_cache[secret_id]
    
    async def _background_tasks(self):
        """Background tasks for maintenance"""
        while True:
            try:
                # Check for expired secrets
                await self._check_expired_secrets()
                
                # Check for secrets needing rotation
                await self._check_rotation_needed()
                
                # Clean cache
                await self._clean_cache()
                
                # Wait before next check
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                print(f"Background task error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _check_expired_secrets(self):
        """Check and handle expired secrets"""
        try:
            secrets = await self.list_secrets()
            current_time = datetime.now()
            
            for secret_metadata in secrets:
                if secret_metadata.expires_at and current_time > secret_metadata.expires_at:
                    print(f"Secret {secret_metadata.name} has expired")
                    # Could automatically delete or alert
        except Exception as e:
            print(f"Error checking expired secrets: {e}")
    
    async def _check_rotation_needed(self):
        """Check if any secrets need rotation"""
        try:
            for secret_type, policy in self.rotation_policies.items():
                if policy.next_rotation and datetime.now() > policy.next_rotation:
                    # Find secrets of this type that need rotation
                    secrets = await self.list_secrets(secret_type=secret_type)
                    
                    for secret_metadata in secrets:
                        if (not secret_metadata.updated_at or 
                            datetime.now() - secret_metadata.updated_at > timedelta(days=policy.rotation_interval)):
                            
                            print(f"Rotating secret {secret_metadata.name}")
                            await self.rotate_secret(secret_metadata.secret_id)
        except Exception as e:
            print(f"Error checking rotation: {e}")
    
    async def _clean_cache(self):
        """Clean expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.secret_cache.items()
            if current_time - entry['timestamp'] > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.secret_cache[key]

class VaultProvider:
    """HashiCorp Vault provider"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = hvac.Client(
            url=config.get('url', 'http://localhost:8200'),
            token=config.get('token')
        )
        self.mount_point = config.get('mount_point', 'secret')
    
    async def store_secret(self, secret_id: str, value: Union[str, Dict[str, Any]], 
                          metadata: SecretMetadata) -> bool:
        """Store secret in Vault"""
        try:
            secret_data = {
                'value': value,
                'metadata': {
                    'secret_type': metadata.secret_type.value,
                    'created_at': metadata.created_at.isoformat(),
                    'tags': metadata.tags
                }
            }
            
            self.client.secrets.kv.v2.create_or_update_secret(
                path=secret_id,
                secret=secret_data,
                mount_point=self.mount_point
            )
            
            return True
        except Exception as e:
            print(f"Vault store error: {e}")
            return False
    
    async def get_secret(self, secret_id: str) -> Optional[SecretValue]:
        """Get secret from Vault"""
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=secret_id,
                mount_point=self.mount_point
            )
            
            secret_data = response['data']['data']
            metadata_data = secret_data.get('metadata', {})
            
            metadata = SecretMetadata(
                secret_id=secret_id,
                name=secret_id,  # Vault doesn't store separate name
                secret_type=SecretType(metadata_data.get('secret_type', 'configuration')),
                provider=SecretProvider.VAULT,
                created_at=datetime.fromisoformat(metadata_data.get('created_at', datetime.now().isoformat())),
                updated_at=datetime.now(),
                tags=metadata_data.get('tags', {})
            )
            
            return SecretValue(
                value=secret_data['value'],
                metadata=metadata,
                version=str(response['data']['metadata']['version'])
            )
            
        except Exception as e:
            print(f"Vault get error: {e}")
            return None
    
    async def update_secret(self, secret_id: str, value: Union[str, Dict[str, Any]], 
                           metadata: SecretMetadata) -> bool:
        """Update secret in Vault"""
        return await self.store_secret(secret_id, value, metadata)
    
    async def delete_secret(self, secret_id: str) -> bool:
        """Delete secret from Vault"""
        try:
            self.client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=secret_id,
                mount_point=self.mount_point
            )
            return True
        except Exception as e:
            print(f"Vault delete error: {e}")
            return False
    
    async def list_secrets(self) -> List[SecretMetadata]:
        """List secrets from Vault"""
        try:
            response = self.client.secrets.kv.v2.list_secrets(
                path='',
                mount_point=self.mount_point
            )
            
            secrets = []
            for secret_name in response['data']['keys']:
                secret = await self.get_secret(secret_name)
                if secret:
                    secrets.append(secret.metadata)
            
            return secrets
        except Exception as e:
            print(f"Vault list error: {e}")
            return []

class AWSSecretsProvider:
    """AWS Secrets Manager provider"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = boto3.client(
            'secretsmanager',
            region_name=config.get('region', 'us-east-1'),
            aws_access_key_id=config.get('access_key_id'),
            aws_secret_access_key=config.get('secret_access_key')
        )
    
    async def store_secret(self, secret_id: str, value: Union[str, Dict[str, Any]], 
                          metadata: SecretMetadata) -> bool:
        """Store secret in AWS Secrets Manager"""
        try:
            secret_string = json.dumps(value) if isinstance(value, dict) else str(value)
            
            self.client.create_secret(
                Name=secret_id,
                SecretString=secret_string,
                Description=f"Secret of type {metadata.secret_type.value}",
                Tags=[
                    {'Key': k, 'Value': v} for k, v in metadata.tags.items()
                ]
            )
            
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceExistsException':
                # Update existing secret
                return await self.update_secret(secret_id, value, metadata)
            else:
                print(f"AWS Secrets Manager store error: {e}")
                return False
    
    async def get_secret(self, secret_id: str) -> Optional[SecretValue]:
        """Get secret from AWS Secrets Manager"""
        try:
            response = self.client.get_secret_value(SecretId=secret_id)
            
            secret_string = response['SecretString']
            try:
                value = json.loads(secret_string)
            except json.JSONDecodeError:
                value = secret_string
            
            # Get metadata
            describe_response = self.client.describe_secret(SecretId=secret_id)
            
            tags = {tag['Key']: tag['Value'] for tag in describe_response.get('Tags', [])}
            
            metadata = SecretMetadata(
                secret_id=secret_id,
                name=describe_response.get('Name', secret_id),
                secret_type=SecretType.CONFIGURATION,  # Default, could be in tags
                provider=SecretProvider.AWS_SECRETS_MANAGER,
                created_at=describe_response['CreatedDate'],
                updated_at=describe_response['LastChangedDate'],
                tags=tags
            )
            
            return SecretValue(
                value=value,
                metadata=metadata,
                version=response['VersionId']
            )
            
        except ClientError as e:
            print(f"AWS Secrets Manager get error: {e}")
            return None
    
    async def update_secret(self, secret_id: str, value: Union[str, Dict[str, Any]], 
                           metadata: SecretMetadata) -> bool:
        """Update secret in AWS Secrets Manager"""
        try:
            secret_string = json.dumps(value) if isinstance(value, dict) else str(value)
            
            self.client.update_secret(
                SecretId=secret_id,
                SecretString=secret_string
            )
            
            return True
        except ClientError as e:
            print(f"AWS Secrets Manager update error: {e}")
            return False
    
    async def delete_secret(self, secret_id: str) -> bool:
        """Delete secret from AWS Secrets Manager"""
        try:
            self.client.delete_secret(
                SecretId=secret_id,
                ForceDeleteWithoutRecovery=True
            )
            return True
        except ClientError as e:
            print(f"AWS Secrets Manager delete error: {e}")
            return False
    
    async def list_secrets(self) -> List[SecretMetadata]:
        """List secrets from AWS Secrets Manager"""
        try:
            response = self.client.list_secrets()
            
            secrets = []
            for secret_info in response['SecretList']:
                secret = await self.get_secret(secret_info['ARN'])
                if secret:
                    secrets.append(secret.metadata)
            
            return secrets
        except ClientError as e:
            print(f"AWS Secrets Manager list error: {e}")
            return []

class AzureKeyVaultProvider:
    """Azure Key Vault provider (placeholder)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Initialize Azure Key Vault client
    
    async def store_secret(self, secret_id: str, value: Union[str, Dict[str, Any]], 
                          metadata: SecretMetadata) -> bool:
        # Implement Azure Key Vault storage
        return False
    
    async def get_secret(self, secret_id: str) -> Optional[SecretValue]:
        # Implement Azure Key Vault retrieval
        return None
    
    async def update_secret(self, secret_id: str, value: Union[str, Dict[str, Any]], 
                           metadata: SecretMetadata) -> bool:
        # Implement Azure Key Vault update
        return False
    
    async def delete_secret(self, secret_id: str) -> bool:
        # Implement Azure Key Vault deletion
        return False
    
    async def list_secrets(self) -> List[SecretMetadata]:
        # Implement Azure Key Vault listing
        return []

class LocalEncryptedProvider:
    """Local encrypted storage provider (fallback)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage_path = config.get('storage_path', '/tmp/secrets.json')
        self.encryption_key = None
        self._initialize_encryption()
        self.secrets_db = self._load_secrets_db()
    
    def _initialize_encryption(self):
        """Initialize local encryption"""
        key_material = self.config.get('encryption_key', 'default-key')
        password = key_material.encode()
        salt = b'local-storage-salt-16b'
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        self.encryption_key = Fernet(key)
    
    def _load_secrets_db(self) -> Dict[str, Any]:
        """Load secrets database from disk"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'rb') as f:
                    encrypted_data = f.read()
                    decrypted_data = self.encryption_key.decrypt(encrypted_data)
                    return json.loads(decrypted_data.decode())
        except Exception as e:
            print(f"Error loading secrets database: {e}")
        
        return {}
    
    def _save_secrets_db(self):
        """Save secrets database to disk"""
        try:
            data = json.dumps(self.secrets_db).encode()
            encrypted_data = self.encryption_key.encrypt(data)
            
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'wb') as f:
                f.write(encrypted_data)
        except Exception as e:
            print(f"Error saving secrets database: {e}")
    
    async def store_secret(self, secret_id: str, value: Union[str, Dict[str, Any]], 
                          metadata: SecretMetadata) -> bool:
        """Store secret locally"""
        try:
            self.secrets_db[secret_id] = {
                'value': value,
                'metadata': {
                    'secret_type': metadata.secret_type.value,
                    'created_at': metadata.created_at.isoformat(),
                    'updated_at': metadata.updated_at.isoformat(),
                    'tags': metadata.tags
                }
            }
            
            self._save_secrets_db()
            return True
        except Exception as e:
            print(f"Local storage error: {e}")
            return False
    
    async def get_secret(self, secret_id: str) -> Optional[SecretValue]:
        """Get secret from local storage"""
        try:
            if secret_id not in self.secrets_db:
                return None
            
            secret_data = self.secrets_db[secret_id]
            metadata_data = secret_data['metadata']
            
            metadata = SecretMetadata(
                secret_id=secret_id,
                name=secret_id,
                secret_type=SecretType(metadata_data['secret_type']),
                provider=SecretProvider.LOCAL_ENCRYPTED,
                created_at=datetime.fromisoformat(metadata_data['created_at']),
                updated_at=datetime.fromisoformat(metadata_data['updated_at']),
                tags=metadata_data.get('tags', {})
            )
            
            return SecretValue(
                value=secret_data['value'],
                metadata=metadata
            )
            
        except Exception as e:
            print(f"Local get error: {e}")
            return None
    
    async def update_secret(self, secret_id: str, value: Union[str, Dict[str, Any]], 
                           metadata: SecretMetadata) -> bool:
        """Update secret in local storage"""
        metadata.updated_at = datetime.now()
        return await self.store_secret(secret_id, value, metadata)
    
    async def delete_secret(self, secret_id: str) -> bool:
        """Delete secret from local storage"""
        try:
            if secret_id in self.secrets_db:
                del self.secrets_db[secret_id]
                self._save_secrets_db()
                return True
            return False
        except Exception as e:
            print(f"Local delete error: {e}")
            return False
    
    async def list_secrets(self) -> List[SecretMetadata]:
        """List secrets from local storage"""
        secrets = []
        for secret_id in self.secrets_db.keys():
            secret = await self.get_secret(secret_id)
            if secret:
                secrets.append(secret.metadata)
        return secrets