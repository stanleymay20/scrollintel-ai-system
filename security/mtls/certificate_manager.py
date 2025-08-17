"""
Mutual TLS (mTLS) Certificate Manager
Implements certificate generation, rotation, and management for service-to-service communication
"""

import os
import ssl
import socket
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from cryptography import x509
from cryptography.x509.oid import NameOID, ExtensionOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
import ipaddress

logger = logging.getLogger(__name__)

@dataclass
class CertificateInfo:
    common_name: str
    subject_alt_names: List[str]
    not_before: datetime
    not_after: datetime
    serial_number: int
    issuer: str
    fingerprint: str

@dataclass
class ServiceIdentity:
    service_name: str
    namespace: str
    cluster: str
    environment: str

class CertificateAuthority:
    """Certificate Authority for issuing service certificates"""
    
    def __init__(self, ca_cert_path: str, ca_key_path: str, ca_passphrase: Optional[str] = None):
        self.ca_cert = self._load_ca_certificate(ca_cert_path)
        self.ca_key = self._load_ca_private_key(ca_key_path, ca_passphrase)
        
    def _load_ca_certificate(self, cert_path: str) -> x509.Certificate:
        """Load CA certificate from file"""
        try:
            with open(cert_path, 'rb') as f:
                cert_data = f.read()
            return x509.load_pem_x509_certificate(cert_data, default_backend())
        except Exception as e:
            logger.error(f"Failed to load CA certificate: {e}")
            raise
    
    def _load_ca_private_key(self, key_path: str, passphrase: Optional[str]) -> rsa.RSAPrivateKey:
        """Load CA private key from file"""
        try:
            with open(key_path, 'rb') as f:
                key_data = f.read()
            
            password = passphrase.encode() if passphrase else None
            return serialization.load_pem_private_key(key_data, password, default_backend())
        except Exception as e:
            logger.error(f"Failed to load CA private key: {e}")
            raise
    
    def issue_certificate(self, service_identity: ServiceIdentity, 
                         validity_days: int = 90) -> Tuple[bytes, bytes]:
        """Issue a certificate for a service"""
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        # Create certificate
        subject = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ScrollIntel"),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, service_identity.namespace),
            x509.NameAttribute(NameOID.COMMON_NAME, self._generate_spiffe_id(service_identity)),
        ])
        
        # Subject Alternative Names
        san_list = [
            x509.DNSName(service_identity.service_name),
            x509.DNSName(f"{service_identity.service_name}.{service_identity.namespace}"),
            x509.DNSName(f"{service_identity.service_name}.{service_identity.namespace}.svc"),
            x509.DNSName(f"{service_identity.service_name}.{service_identity.namespace}.svc.cluster.local"),
            x509.UniformResourceIdentifier(self._generate_spiffe_id(service_identity))
        ]
        
        # Add localhost for development
        san_list.extend([
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            x509.IPAddress(ipaddress.IPv6Address("::1"))
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            self.ca_cert.subject
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=validity_days)
        ).add_extension(
            x509.SubjectAlternativeName(san_list),
            critical=False,
        ).add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                content_commitment=False,
                data_encipherment=False,
                encipher_only=False,
                decipher_only=False
            ),
            critical=True,
        ).add_extension(
            x509.ExtendedKeyUsage([
                x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
                x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
            ]),
            critical=True,
        ).sign(self.ca_key, hashes.SHA256(), default_backend())
        
        # Serialize certificate and private key
        cert_pem = cert.public_bytes(serialization.Encoding.PEM)
        key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        return cert_pem, key_pem
    
    def _generate_spiffe_id(self, service_identity: ServiceIdentity) -> str:
        """Generate SPIFFE ID for service"""
        return f"spiffe://{service_identity.cluster}/ns/{service_identity.namespace}/sa/{service_identity.service_name}"

class CertificateManager:
    """Manages certificates for mTLS communication"""
    
    def __init__(self, ca: CertificateAuthority, cert_store_path: str):
        self.ca = ca
        self.cert_store_path = cert_store_path
        self.certificates = {}
        self._ensure_cert_store()
        
    def _ensure_cert_store(self):
        """Ensure certificate store directory exists"""
        os.makedirs(self.cert_store_path, exist_ok=True)
        
    def get_or_create_certificate(self, service_identity: ServiceIdentity) -> Tuple[str, str]:
        """Get existing certificate or create new one"""
        cert_key = f"{service_identity.service_name}.{service_identity.namespace}"
        
        # Check if certificate exists and is valid
        cert_path = os.path.join(self.cert_store_path, f"{cert_key}.crt")
        key_path = os.path.join(self.cert_store_path, f"{cert_key}.key")
        
        if os.path.exists(cert_path) and os.path.exists(key_path):
            if self._is_certificate_valid(cert_path):
                return cert_path, key_path
        
        # Create new certificate
        cert_pem, key_pem = self.ca.issue_certificate(service_identity)
        
        # Save to files
        with open(cert_path, 'wb') as f:
            f.write(cert_pem)
        
        with open(key_path, 'wb') as f:
            f.write(key_pem)
        
        # Set appropriate permissions
        os.chmod(cert_path, 0o644)
        os.chmod(key_path, 0o600)
        
        logger.info(f"Created certificate for {cert_key}")
        return cert_path, key_path
    
    def _is_certificate_valid(self, cert_path: str, buffer_days: int = 30) -> bool:
        """Check if certificate is valid and not expiring soon"""
        try:
            with open(cert_path, 'rb') as f:
                cert_data = f.read()
            
            cert = x509.load_pem_x509_certificate(cert_data, default_backend())
            
            # Check if certificate is not expired and has buffer time
            now = datetime.utcnow()
            expiry_buffer = now + timedelta(days=buffer_days)
            
            return cert.not_valid_after > expiry_buffer
            
        except Exception as e:
            logger.error(f"Error validating certificate {cert_path}: {e}")
            return False
    
    def rotate_certificate(self, service_identity: ServiceIdentity) -> Tuple[str, str]:
        """Force rotation of certificate"""
        cert_key = f"{service_identity.service_name}.{service_identity.namespace}"
        
        # Remove existing certificate
        cert_path = os.path.join(self.cert_store_path, f"{cert_key}.crt")
        key_path = os.path.join(self.cert_store_path, f"{cert_key}.key")
        
        for path in [cert_path, key_path]:
            if os.path.exists(path):
                os.remove(path)
        
        # Create new certificate
        return self.get_or_create_certificate(service_identity)
    
    def get_certificate_info(self, cert_path: str) -> Optional[CertificateInfo]:
        """Get certificate information"""
        try:
            with open(cert_path, 'rb') as f:
                cert_data = f.read()
            
            cert = x509.load_pem_x509_certificate(cert_data, default_backend())
            
            # Extract SAN
            san_list = []
            try:
                san_ext = cert.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
                san_list = [name.value for name in san_ext.value]
            except x509.ExtensionNotFound:
                pass
            
            return CertificateInfo(
                common_name=cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value,
                subject_alt_names=san_list,
                not_before=cert.not_valid_before,
                not_after=cert.not_valid_after,
                serial_number=cert.serial_number,
                issuer=cert.issuer.rfc4514_string(),
                fingerprint=cert.fingerprint(hashes.SHA256()).hex()
            )
            
        except Exception as e:
            logger.error(f"Error getting certificate info: {e}")
            return None
    
    def list_certificates(self) -> List[Dict[str, any]]:
        """List all managed certificates"""
        certificates = []
        
        for file in os.listdir(self.cert_store_path):
            if file.endswith('.crt'):
                cert_path = os.path.join(self.cert_store_path, file)
                cert_info = self.get_certificate_info(cert_path)
                
                if cert_info:
                    certificates.append({
                        'file': file,
                        'path': cert_path,
                        'common_name': cert_info.common_name,
                        'expires': cert_info.not_after,
                        'days_until_expiry': (cert_info.not_after - datetime.utcnow()).days,
                        'fingerprint': cert_info.fingerprint
                    })
        
        return certificates

class MTLSClient:
    """Client for making mTLS connections"""
    
    def __init__(self, cert_path: str, key_path: str, ca_cert_path: str):
        self.cert_path = cert_path
        self.key_path = key_path
        self.ca_cert_path = ca_cert_path
        
    def create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for mTLS"""
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        
        # Load CA certificate
        context.load_verify_locations(self.ca_cert_path)
        
        # Load client certificate and key
        context.load_cert_chain(self.cert_path, self.key_path)
        
        # Require certificate verification
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        
        return context
    
    def connect(self, hostname: str, port: int) -> ssl.SSLSocket:
        """Create mTLS connection to service"""
        context = self.create_ssl_context()
        
        sock = socket.create_connection((hostname, port))
        ssl_sock = context.wrap_socket(sock, server_hostname=hostname)
        
        return ssl_sock

class MTLSServer:
    """Server for accepting mTLS connections"""
    
    def __init__(self, cert_path: str, key_path: str, ca_cert_path: str):
        self.cert_path = cert_path
        self.key_path = key_path
        self.ca_cert_path = ca_cert_path
        
    def create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for mTLS server"""
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        
        # Load server certificate and key
        context.load_cert_chain(self.cert_path, self.key_path)
        
        # Load CA certificate for client verification
        context.load_verify_locations(self.ca_cert_path)
        
        # Require client certificates
        context.verify_mode = ssl.CERT_REQUIRED
        
        return context
    
    def wrap_socket(self, sock: socket.socket) -> ssl.SSLSocket:
        """Wrap socket with mTLS"""
        context = self.create_ssl_context()
        return context.wrap_socket(sock, server_side=True)

# Certificate rotation scheduler
class CertificateRotationScheduler:
    """Schedules automatic certificate rotation"""
    
    def __init__(self, cert_manager: CertificateManager):
        self.cert_manager = cert_manager
        
    def check_and_rotate_certificates(self, rotation_threshold_days: int = 30):
        """Check and rotate certificates that are expiring soon"""
        certificates = self.cert_manager.list_certificates()
        
        for cert in certificates:
            if cert['days_until_expiry'] <= rotation_threshold_days:
                logger.info(f"Rotating certificate {cert['file']} (expires in {cert['days_until_expiry']} days)")
                
                # Extract service identity from filename
                service_name = cert['file'].replace('.crt', '')
                parts = service_name.split('.')
                
                if len(parts) >= 2:
                    service_identity = ServiceIdentity(
                        service_name=parts[0],
                        namespace=parts[1],
                        cluster="scrollintel-cluster",
                        environment="production"
                    )
                    
                    try:
                        self.cert_manager.rotate_certificate(service_identity)
                        logger.info(f"Successfully rotated certificate for {service_name}")
                    except Exception as e:
                        logger.error(f"Failed to rotate certificate for {service_name}: {e}")

# Example usage
if __name__ == "__main__":
    # Initialize CA
    ca = CertificateAuthority(
        ca_cert_path="./certs/ca.crt",
        ca_key_path="./certs/ca.key"
    )
    
    # Initialize certificate manager
    cert_manager = CertificateManager(ca, "./certs/services")
    
    # Create service identity
    api_service = ServiceIdentity(
        service_name="scrollintel-api",
        namespace="scrollintel",
        cluster="production",
        environment="prod"
    )
    
    # Get or create certificate
    cert_path, key_path = cert_manager.get_or_create_certificate(api_service)
    print(f"Certificate: {cert_path}")
    print(f"Private Key: {key_path}")
    
    # List all certificates
    certificates = cert_manager.list_certificates()
    for cert in certificates:
        print(f"Certificate: {cert['common_name']}, Expires: {cert['expires']}")