"""
Immutable Audit Logging System with Blockchain-based Integrity Verification
Implements tamper-proof audit trails with cryptographic verification
"""

import hashlib
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import sqlite3
import threading
from pathlib import Path


@dataclass
class AuditEvent:
    """Immutable audit event structure"""
    event_id: str
    timestamp: float
    event_type: str
    user_id: Optional[str]
    resource: str
    action: str
    outcome: str
    details: Dict[str, Any]
    source_ip: Optional[str]
    user_agent: Optional[str]
    session_id: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass
class BlockchainBlock:
    """Blockchain block for audit trail integrity"""
    index: int
    timestamp: float
    events: List[AuditEvent]
    previous_hash: str
    nonce: int
    hash: str
    
    def calculate_hash(self) -> str:
        """Calculate block hash"""
        block_string = json.dumps({
            'index': self.index,
            'timestamp': self.timestamp,
            'events': [event.to_dict() for event in self.events],
            'previous_hash': self.previous_hash,
            'nonce': self.nonce
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()


class ImmutableAuditLogger:
    """
    Immutable audit logging system with blockchain-based integrity verification
    """
    
    def __init__(self, db_path: str = "security/audit_blockchain.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.pending_events: List[AuditEvent] = []
        self.block_size = 100  # Events per block
        self.difficulty = 4  # Mining difficulty
        
        # Initialize database
        self._init_database()
        
        # Generate or load cryptographic keys
        self._init_crypto_keys()
        
        # Initialize genesis block if needed
        self._init_genesis_block()
    
    def _init_database(self):
        """Initialize SQLite database for blockchain storage"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS blockchain (
                    block_index INTEGER PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    events_json TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    nonce INTEGER NOT NULL,
                    block_hash TEXT NOT NULL UNIQUE,
                    signature TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    block_index INTEGER,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    user_id TEXT,
                    resource TEXT NOT NULL,
                    action TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    source_ip TEXT,
                    user_agent TEXT,
                    session_id TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (block_index) REFERENCES blockchain (block_index)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_events(user_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_events(resource)
            """)
    
    def _init_crypto_keys(self):
        """Initialize cryptographic keys for signing"""
        key_path = Path("security/audit_private_key.pem")
        
        if key_path.exists():
            with open(key_path, 'rb') as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(), password=None, backend=default_backend()
                )
        else:
            # Generate new key pair
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            
            # Save private key
            key_path.parent.mkdir(parents=True, exist_ok=True)
            with open(key_path, 'wb') as f:
                f.write(self.private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
        
        self.public_key = self.private_key.public_key()
    
    def _init_genesis_block(self):
        """Initialize genesis block if blockchain is empty"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM blockchain")
            if cursor.fetchone()[0] == 0:
                genesis_block = BlockchainBlock(
                    index=0,
                    timestamp=time.time(),
                    events=[],
                    previous_hash="0",
                    nonce=0,
                    hash=""
                )
                genesis_block.hash = genesis_block.calculate_hash()
                self._store_block(genesis_block)
    
    def log_event(self, event: AuditEvent) -> str:
        """
        Log an immutable audit event
        Returns the event ID for tracking
        """
        with self.lock:
            # Add to pending events
            self.pending_events.append(event)
            
            # Create new block if we have enough events
            if len(self.pending_events) >= self.block_size:
                self._create_new_block()
            
            return event.event_id
    
    def _create_new_block(self):
        """Create and mine a new blockchain block"""
        if not self.pending_events:
            return
        
        # Get previous block hash
        previous_hash = self._get_latest_block_hash()
        
        # Create new block
        new_block = BlockchainBlock(
            index=self._get_next_block_index(),
            timestamp=time.time(),
            events=self.pending_events.copy(),
            previous_hash=previous_hash,
            nonce=0,
            hash=""
        )
        
        # Mine the block (proof of work)
        new_block = self._mine_block(new_block)
        
        # Store the block
        self._store_block(new_block)
        
        # Clear pending events
        self.pending_events.clear()
    
    def _mine_block(self, block: BlockchainBlock) -> BlockchainBlock:
        """Mine block using proof of work"""
        target = "0" * self.difficulty
        
        while not block.hash.startswith(target):
            block.nonce += 1
            block.hash = block.calculate_hash()
        
        return block
    
    def _store_block(self, block: BlockchainBlock):
        """Store block in database with digital signature"""
        # Sign the block hash
        signature = self.private_key.sign(
            block.hash.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        with sqlite3.connect(self.db_path) as conn:
            # Store block
            conn.execute("""
                INSERT INTO blockchain 
                (block_index, timestamp, events_json, previous_hash, nonce, block_hash, signature)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                block.index,
                block.timestamp,
                json.dumps([event.to_dict() for event in block.events]),
                block.previous_hash,
                block.nonce,
                block.hash,
                signature.hex()
            ))
            
            # Store individual events
            for event in block.events:
                conn.execute("""
                    INSERT INTO audit_events 
                    (event_id, block_index, timestamp, event_type, user_id, resource, 
                     action, outcome, details_json, source_ip, user_agent, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    block.index,
                    event.timestamp,
                    event.event_type,
                    event.user_id,
                    event.resource,
                    event.action,
                    event.outcome,
                    json.dumps(event.details),
                    event.source_ip,
                    event.user_agent,
                    event.session_id
                ))
    
    def _get_latest_block_hash(self) -> str:
        """Get hash of the latest block"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT block_hash FROM blockchain 
                ORDER BY block_index DESC LIMIT 1
            """)
            result = cursor.fetchone()
            return result[0] if result else "0"
    
    def _get_next_block_index(self) -> int:
        """Get the next block index"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT MAX(block_index) FROM blockchain
            """)
            result = cursor.fetchone()
            return (result[0] + 1) if result[0] is not None else 0
    
    def verify_integrity(self) -> Dict[str, Any]:
        """
        Verify the integrity of the entire audit trail
        Returns verification results
        """
        results = {
            'valid': True,
            'total_blocks': 0,
            'total_events': 0,
            'verification_errors': [],
            'verification_timestamp': datetime.utcnow().isoformat()
        }
        
        with sqlite3.connect(self.db_path) as conn:
            # Get all blocks
            cursor = conn.execute("""
                SELECT block_index, timestamp, events_json, previous_hash, 
                       nonce, block_hash, signature
                FROM blockchain ORDER BY block_index
            """)
            
            blocks = cursor.fetchall()
            results['total_blocks'] = len(blocks)
            
            previous_hash = "0"
            
            for block_data in blocks:
                block_index, timestamp, events_json, prev_hash, nonce, block_hash, signature = block_data
                
                # Verify previous hash linkage
                if prev_hash != previous_hash:
                    results['valid'] = False
                    results['verification_errors'].append(
                        f"Block {block_index}: Invalid previous hash linkage"
                    )
                
                # Reconstruct and verify block hash
                events = json.loads(events_json)
                reconstructed_block = BlockchainBlock(
                    index=block_index,
                    timestamp=timestamp,
                    events=[AuditEvent(**event) for event in events],
                    previous_hash=prev_hash,
                    nonce=nonce,
                    hash=""
                )
                
                calculated_hash = reconstructed_block.calculate_hash()
                if calculated_hash != block_hash:
                    results['valid'] = False
                    results['verification_errors'].append(
                        f"Block {block_index}: Hash verification failed"
                    )
                
                # Verify digital signature
                try:
                    self.public_key.verify(
                        bytes.fromhex(signature),
                        block_hash.encode(),
                        padding.PSS(
                            mgf=padding.MGF1(hashes.SHA256()),
                            salt_length=padding.PSS.MAX_LENGTH
                        ),
                        hashes.SHA256()
                    )
                except Exception as e:
                    results['valid'] = False
                    results['verification_errors'].append(
                        f"Block {block_index}: Signature verification failed: {str(e)}"
                    )
                
                results['total_events'] += len(events)
                previous_hash = block_hash
        
        return results
    
    def get_audit_trail(self, 
                       start_time: Optional[float] = None,
                       end_time: Optional[float] = None,
                       user_id: Optional[str] = None,
                       resource: Optional[str] = None,
                       event_type: Optional[str] = None,
                       limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Retrieve audit trail with filtering options
        """
        query = """
            SELECT event_id, timestamp, event_type, user_id, resource, 
                   action, outcome, details_json, source_ip, user_agent, session_id
            FROM audit_events WHERE 1=1
        """
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if resource:
            query += " AND resource LIKE ?"
            params.append(f"%{resource}%")
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            events = []
            
            for row in cursor.fetchall():
                event_data = {
                    'event_id': row[0],
                    'timestamp': row[1],
                    'event_type': row[2],
                    'user_id': row[3],
                    'resource': row[4],
                    'action': row[5],
                    'outcome': row[6],
                    'details': json.loads(row[7]),
                    'source_ip': row[8],
                    'user_agent': row[9],
                    'session_id': row[10]
                }
                events.append(event_data)
            
            return events
    
    def force_block_creation(self):
        """Force creation of a new block with pending events"""
        with self.lock:
            if self.pending_events:
                self._create_new_block()
    
    def get_blockchain_stats(self) -> Dict[str, Any]:
        """Get blockchain statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_blocks,
                    MIN(timestamp) as first_block_time,
                    MAX(timestamp) as latest_block_time
                FROM blockchain
            """)
            block_stats = cursor.fetchone()
            
            cursor = conn.execute("SELECT COUNT(*) FROM audit_events")
            total_events = cursor.fetchone()[0]
            
            return {
                'total_blocks': block_stats[0],
                'total_events': total_events,
                'first_block_time': block_stats[1],
                'latest_block_time': block_stats[2],
                'pending_events': len(self.pending_events)
            }