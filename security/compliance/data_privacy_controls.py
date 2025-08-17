"""
Data Privacy Controls with Automated Data Subject Request Handling
Implements GDPR, CCPA, and other privacy regulation compliance
"""

import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from pathlib import Path
import hashlib
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
# Email imports removed for demo simplicity


class RequestType(Enum):
    ACCESS = "access"
    RECTIFICATION = "rectification"
    ERASURE = "erasure"
    PORTABILITY = "portability"
    RESTRICTION = "restriction"
    OBJECTION = "objection"
    WITHDRAW_CONSENT = "withdraw_consent"


class RequestStatus(Enum):
    RECEIVED = "received"
    VERIFIED = "verified"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"
    EXPIRED = "expired"


class DataCategory(Enum):
    PERSONAL_IDENTIFIABLE = "pii"
    SENSITIVE_PERSONAL = "sensitive_pii"
    FINANCIAL = "financial"
    HEALTH = "health"
    BIOMETRIC = "biometric"
    BEHAVIORAL = "behavioral"
    LOCATION = "location"
    COMMUNICATION = "communication"


@dataclass
class DataSubjectRequest:
    """Data subject request record"""
    request_id: str
    request_type: RequestType
    status: RequestStatus
    data_subject_email: str
    data_subject_name: Optional[str]
    verification_method: str
    verification_status: str
    requested_data_categories: List[DataCategory]
    specific_data_items: List[str]
    reason: Optional[str]
    submitted_at: datetime
    verified_at: Optional[datetime]
    completed_at: Optional[datetime]
    due_date: datetime
    assigned_to: Optional[str]
    processing_notes: List[str]
    evidence_of_identity: Dict[str, Any]
    response_data: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['request_type'] = self.request_type.value
        data['status'] = self.status.value
        data['requested_data_categories'] = [cat.value for cat in self.requested_data_categories]
        data['submitted_at'] = self.submitted_at.isoformat()
        data['verified_at'] = self.verified_at.isoformat() if self.verified_at else None
        data['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        data['due_date'] = self.due_date.isoformat()
        return data


@dataclass
class DataInventoryItem:
    """Data inventory item for privacy mapping"""
    item_id: str
    data_category: DataCategory
    data_type: str
    storage_location: str
    retention_period_days: int
    legal_basis: str
    purpose: str
    data_controller: str
    data_processor: Optional[str]
    encryption_status: bool
    access_controls: List[str]
    sharing_agreements: List[str]
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['data_category'] = self.data_category.value
        data['last_updated'] = self.last_updated.isoformat()
        return data


class DataPrivacyControls:
    """
    Comprehensive data privacy controls and automated request handling
    """
    
    def __init__(self, db_path: str = "security/privacy_controls.db"):
        self.db_path = db_path
        self.request_queue = queue.Queue()
        self.processing_thread = None
        self.running = False
        
        self._init_database()
        self._init_data_inventory()
        self._init_processing_workflows()
        self._init_notification_system()
    
    def _init_database(self):
        """Initialize privacy controls database"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_subject_requests (
                    request_id TEXT PRIMARY KEY,
                    request_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    data_subject_email TEXT NOT NULL,
                    data_subject_name TEXT,
                    verification_method TEXT NOT NULL,
                    verification_status TEXT NOT NULL,
                    requested_data_categories_json TEXT NOT NULL,
                    specific_data_items_json TEXT NOT NULL,
                    reason TEXT,
                    submitted_at DATETIME NOT NULL,
                    verified_at DATETIME,
                    completed_at DATETIME,
                    due_date DATETIME NOT NULL,
                    assigned_to TEXT,
                    processing_notes_json TEXT NOT NULL,
                    evidence_of_identity_json TEXT NOT NULL,
                    response_data_json TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_inventory (
                    item_id TEXT PRIMARY KEY,
                    data_category TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    storage_location TEXT NOT NULL,
                    retention_period_days INTEGER NOT NULL,
                    legal_basis TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    data_controller TEXT NOT NULL,
                    data_processor TEXT,
                    encryption_status BOOLEAN NOT NULL,
                    access_controls_json TEXT NOT NULL,
                    sharing_agreements_json TEXT NOT NULL,
                    last_updated DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS consent_records (
                    consent_id TEXT PRIMARY KEY,
                    data_subject_email TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    legal_basis TEXT NOT NULL,
                    consent_given BOOLEAN NOT NULL,
                    consent_date DATETIME NOT NULL,
                    consent_method TEXT NOT NULL,
                    withdrawn_date DATETIME,
                    withdrawal_method TEXT,
                    data_categories_json TEXT NOT NULL,
                    retention_period_days INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS privacy_breach_log (
                    breach_id TEXT PRIMARY KEY,
                    breach_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    affected_data_subjects INTEGER NOT NULL,
                    data_categories_json TEXT NOT NULL,
                    breach_date DATETIME NOT NULL,
                    discovered_date DATETIME NOT NULL,
                    reported_date DATETIME,
                    description TEXT NOT NULL,
                    containment_measures TEXT,
                    notification_required BOOLEAN NOT NULL,
                    regulatory_notification_sent BOOLEAN DEFAULT FALSE,
                    affected_individuals_notified BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def _init_data_inventory(self):
        """Initialize data inventory with common data types"""
        inventory_items = [
            DataInventoryItem(
                item_id="user_profiles",
                data_category=DataCategory.PERSONAL_IDENTIFIABLE,
                data_type="User Profile Data",
                storage_location="primary_database.users",
                retention_period_days=2555,  # 7 years
                legal_basis="Contract",
                purpose="Service provision and account management",
                data_controller="Company",
                data_processor=None,
                encryption_status=True,
                access_controls=["role_based", "mfa_required"],
                sharing_agreements=[],
                last_updated=datetime.utcnow()
            ),
            DataInventoryItem(
                item_id="payment_data",
                data_category=DataCategory.FINANCIAL,
                data_type="Payment Information",
                storage_location="payment_processor.stripe",
                retention_period_days=2555,
                legal_basis="Contract",
                purpose="Payment processing and fraud prevention",
                data_controller="Company",
                data_processor="Stripe Inc.",
                encryption_status=True,
                access_controls=["pci_compliant", "tokenized"],
                sharing_agreements=["stripe_dpa"],
                last_updated=datetime.utcnow()
            ),
            DataInventoryItem(
                item_id="analytics_data",
                data_category=DataCategory.BEHAVIORAL,
                data_type="Usage Analytics",
                storage_location="analytics_database.events",
                retention_period_days=730,  # 2 years
                legal_basis="Legitimate Interest",
                purpose="Service improvement and analytics",
                data_controller="Company",
                data_processor=None,
                encryption_status=True,
                access_controls=["pseudonymized", "aggregated"],
                sharing_agreements=[],
                last_updated=datetime.utcnow()
            )
        ]
        
        for item in inventory_items:
            self._store_inventory_item(item)
    
    def _init_processing_workflows(self):
        """Initialize request processing workflows"""
        self.processing_workflows = {
            RequestType.ACCESS: {
                'sla_days': 30,
                'verification_required': True,
                'automated_steps': ['verify_identity', 'collect_data', 'format_response'],
                'manual_review_required': False
            },
            RequestType.RECTIFICATION: {
                'sla_days': 30,
                'verification_required': True,
                'automated_steps': ['verify_identity', 'validate_correction'],
                'manual_review_required': True
            },
            RequestType.ERASURE: {
                'sla_days': 30,
                'verification_required': True,
                'automated_steps': ['verify_identity', 'check_legal_obligations'],
                'manual_review_required': True
            },
            RequestType.PORTABILITY: {
                'sla_days': 30,
                'verification_required': True,
                'automated_steps': ['verify_identity', 'collect_data', 'format_portable'],
                'manual_review_required': False
            }
        }
    
    def _init_notification_system(self):
        """Initialize notification system"""
        self.notification_config = {
            'smtp_server': 'localhost',
            'smtp_port': 587,
            'sender_email': 'privacy@company.com',
            'privacy_team_email': 'privacy-team@company.com',
            'dpo_email': 'dpo@company.com'
        }
    
    def start_processing(self):
        """Start automated request processing"""
        if self.running:
            return
        
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop_processing(self):
        """Stop automated request processing"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
    
    def submit_data_subject_request(self, 
                                  request_type: RequestType,
                                  data_subject_email: str,
                                  data_subject_name: Optional[str] = None,
                                  requested_categories: Optional[List[DataCategory]] = None,
                                  specific_items: Optional[List[str]] = None,
                                  reason: Optional[str] = None,
                                  evidence_of_identity: Optional[Dict[str, Any]] = None) -> str:
        """
        Submit a new data subject request
        Returns request ID
        """
        request_id = str(uuid.uuid4())
        
        # Default to all categories if none specified
        if not requested_categories:
            requested_categories = [DataCategory.PERSONAL_IDENTIFIABLE]
        
        if not specific_items:
            specific_items = []
        
        if not evidence_of_identity:
            evidence_of_identity = {}
        
        # Calculate due date based on request type
        workflow = self.processing_workflows.get(request_type, {'sla_days': 30})
        due_date = datetime.utcnow() + timedelta(days=workflow['sla_days'])
        
        request = DataSubjectRequest(
            request_id=request_id,
            request_type=request_type,
            status=RequestStatus.RECEIVED,
            data_subject_email=data_subject_email,
            data_subject_name=data_subject_name,
            verification_method="email",
            verification_status="pending",
            requested_data_categories=requested_categories,
            specific_data_items=specific_items,
            reason=reason,
            submitted_at=datetime.utcnow(),
            verified_at=None,
            completed_at=None,
            due_date=due_date,
            assigned_to=None,
            processing_notes=[],
            evidence_of_identity=evidence_of_identity,
            response_data=None,
            metadata={'source': 'web_form', 'ip_address': '127.0.0.1'}
        )
        
        # Store request
        self._store_request(request)
        
        # Add to processing queue
        self.request_queue.put(request)
        
        # Send confirmation email
        self._send_request_confirmation(request)
        
        return request_id
    
    def _processing_worker(self):
        """Background worker for request processing"""
        while self.running:
            try:
                request = self.request_queue.get(timeout=5)
                self._process_request(request)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing privacy request: {str(e)}")
    
    def _process_request(self, request: DataSubjectRequest):
        """Process a data subject request"""
        try:
            workflow = self.processing_workflows.get(request.request_type)
            if not workflow:
                self._update_request_status(request.request_id, RequestStatus.REJECTED, 
                                          "Unsupported request type")
                return
            
            # Step 1: Verify identity if required
            if workflow['verification_required']:
                if not self._verify_identity(request):
                    self._update_request_status(request.request_id, RequestStatus.REJECTED,
                                              "Identity verification failed")
                    return
                
                self._update_request_status(request.request_id, RequestStatus.VERIFIED)
            
            # Step 2: Execute automated steps
            self._update_request_status(request.request_id, RequestStatus.IN_PROGRESS)
            
            for step in workflow['automated_steps']:
                success = self._execute_processing_step(request, step)
                if not success:
                    self._update_request_status(request.request_id, RequestStatus.REJECTED,
                                              f"Failed at step: {step}")
                    return
            
            # Step 3: Manual review if required
            if workflow['manual_review_required']:
                self._assign_for_manual_review(request)
            else:
                self._complete_request(request)
        
        except Exception as e:
            print(f"Error processing request {request.request_id}: {str(e)}")
            self._update_request_status(request.request_id, RequestStatus.REJECTED, str(e))
    
    def _verify_identity(self, request: DataSubjectRequest) -> bool:
        """Verify data subject identity"""
        # In real implementation, this would use various verification methods
        # For now, we'll simulate email verification
        
        if request.verification_method == "email":
            # Send verification email (simulated)
            verification_token = hashlib.sha256(
                f"{request.request_id}{request.data_subject_email}{time.time()}".encode()
            ).hexdigest()[:16]
            
            print(f"Sending verification email to {request.data_subject_email} with token {verification_token}")
            
            # In real implementation, user would click link to verify
            # For simulation, we'll mark as verified
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE data_subject_requests 
                    SET verification_status = 'verified', verified_at = ?
                    WHERE request_id = ?
                """, (datetime.utcnow(), request.request_id))
            
            return True
        
        return False
    
    def _execute_processing_step(self, request: DataSubjectRequest, step: str) -> bool:
        """Execute a processing step"""
        if step == 'verify_identity':
            return self._verify_identity(request)
        elif step == 'collect_data':
            return self._collect_subject_data(request)
        elif step == 'format_response':
            return self._format_response_data(request)
        elif step == 'format_portable':
            return self._format_portable_data(request)
        elif step == 'validate_correction':
            return self._validate_correction_request(request)
        elif step == 'check_legal_obligations':
            return self._check_erasure_obligations(request)
        
        return False
    
    def _collect_subject_data(self, request: DataSubjectRequest) -> bool:
        """Collect all data for the data subject"""
        try:
            collected_data = {}
            
            # Get relevant inventory items
            inventory_items = self._get_inventory_by_categories(request.requested_data_categories)
            
            for item in inventory_items:
                # Simulate data collection from various sources
                data = self._extract_data_from_source(
                    request.data_subject_email, 
                    item.storage_location,
                    item.data_type
                )
                
                if data:
                    collected_data[item.data_type] = {
                        'data': data,
                        'source': item.storage_location,
                        'category': item.data_category.value,
                        'retention_period': item.retention_period_days,
                        'legal_basis': item.legal_basis
                    }
            
            # Store collected data
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE data_subject_requests 
                    SET response_data_json = ?
                    WHERE request_id = ?
                """, (json.dumps(collected_data), request.request_id))
            
            return True
        
        except Exception as e:
            print(f"Error collecting data for {request.request_id}: {str(e)}")
            return False
    
    def _extract_data_from_source(self, email: str, source: str, data_type: str) -> Optional[Dict[str, Any]]:
        """Extract data from specific source (mock implementation)"""
        # In real implementation, this would query actual data sources
        mock_data = {
            'primary_database.users': {
                'user_id': '12345',
                'email': email,
                'name': 'John Doe',
                'created_at': '2023-01-15T10:30:00Z',
                'last_login': '2024-03-15T14:22:00Z'
            },
            'analytics_database.events': {
                'total_sessions': 156,
                'last_activity': '2024-03-15T14:22:00Z',
                'preferred_features': ['dashboard', 'reports']
            },
            'payment_processor.stripe': {
                'customer_id': 'cus_12345',
                'payment_methods': ['card_ending_1234'],
                'total_transactions': 23
            }
        }
        
        return mock_data.get(source)
    
    def _format_response_data(self, request: DataSubjectRequest) -> bool:
        """Format response data for access request"""
        try:
            # Get collected data
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT response_data_json FROM data_subject_requests 
                    WHERE request_id = ?
                """, (request.request_id,))
                
                result = cursor.fetchone()
                if not result or not result[0]:
                    return False
                
                collected_data = json.loads(result[0])
            
            # Format for human readability
            formatted_response = {
                'request_id': request.request_id,
                'data_subject': request.data_subject_email,
                'request_date': request.submitted_at.isoformat(),
                'data_categories': [cat.value for cat in request.requested_data_categories],
                'personal_data': collected_data,
                'data_sources': list(set(
                    item['source'] for item in collected_data.values()
                )),
                'retention_information': {
                    data_type: {
                        'retention_period_days': item['retention_period'],
                        'legal_basis': item['legal_basis']
                    }
                    for data_type, item in collected_data.items()
                }
            }
            
            # Update response data
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE data_subject_requests 
                    SET response_data_json = ?
                    WHERE request_id = ?
                """, (json.dumps(formatted_response), request.request_id))
            
            return True
        
        except Exception as e:
            print(f"Error formatting response for {request.request_id}: {str(e)}")
            return False
    
    def _format_portable_data(self, request: DataSubjectRequest) -> bool:
        """Format data for portability request"""
        try:
            # Get collected data and format in machine-readable format
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT response_data_json FROM data_subject_requests 
                    WHERE request_id = ?
                """, (request.request_id,))
                
                result = cursor.fetchone()
                if not result or not result[0]:
                    return False
                
                collected_data = json.loads(result[0])
            
            # Format for portability (structured JSON)
            portable_data = {
                'export_info': {
                    'request_id': request.request_id,
                    'export_date': datetime.utcnow().isoformat(),
                    'format': 'JSON',
                    'data_subject': request.data_subject_email
                },
                'data': {}
            }
            
            for data_type, item in collected_data.items():
                portable_data['data'][data_type] = item['data']
            
            # Update response data
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE data_subject_requests 
                    SET response_data_json = ?
                    WHERE request_id = ?
                """, (json.dumps(portable_data), request.request_id))
            
            return True
        
        except Exception as e:
            print(f"Error formatting portable data for {request.request_id}: {str(e)}")
            return False
    
    def _validate_correction_request(self, request: DataSubjectRequest) -> bool:
        """Validate rectification request"""
        # In real implementation, this would validate the correction request
        # and check if the data can be legally corrected
        return True
    
    def _check_erasure_obligations(self, request: DataSubjectRequest) -> bool:
        """Check legal obligations for erasure request"""
        # In real implementation, this would check if data can be legally erased
        # considering retention requirements, legal holds, etc.
        return True
    
    def _assign_for_manual_review(self, request: DataSubjectRequest):
        """Assign request for manual review"""
        # In real implementation, this would integrate with task management system
        print(f"Assigning request {request.request_id} for manual review")
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE data_subject_requests 
                SET assigned_to = 'privacy_team'
                WHERE request_id = ?
            """, (request.request_id,))
    
    def _complete_request(self, request: DataSubjectRequest):
        """Complete the request processing"""
        self._update_request_status(request.request_id, RequestStatus.COMPLETED)
        self._send_completion_notification(request)
    
    def _store_request(self, request: DataSubjectRequest):
        """Store data subject request in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO data_subject_requests 
                (request_id, request_type, status, data_subject_email, data_subject_name,
                 verification_method, verification_status, requested_data_categories_json,
                 specific_data_items_json, reason, submitted_at, due_date,
                 processing_notes_json, evidence_of_identity_json, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                request.request_id,
                request.request_type.value,
                request.status.value,
                request.data_subject_email,
                request.data_subject_name,
                request.verification_method,
                request.verification_status,
                json.dumps([cat.value for cat in request.requested_data_categories]),
                json.dumps(request.specific_data_items),
                request.reason,
                request.submitted_at,
                request.due_date,
                json.dumps(request.processing_notes),
                json.dumps(request.evidence_of_identity),
                json.dumps(request.metadata)
            ))
    
    def _store_inventory_item(self, item: DataInventoryItem):
        """Store data inventory item"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO data_inventory 
                (item_id, data_category, data_type, storage_location, retention_period_days,
                 legal_basis, purpose, data_controller, data_processor, encryption_status,
                 access_controls_json, sharing_agreements_json, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item.item_id,
                item.data_category.value,
                item.data_type,
                item.storage_location,
                item.retention_period_days,
                item.legal_basis,
                item.purpose,
                item.data_controller,
                item.data_processor,
                item.encryption_status,
                json.dumps(item.access_controls),
                json.dumps(item.sharing_agreements),
                item.last_updated
            ))
    
    def _get_inventory_by_categories(self, categories: List[DataCategory]) -> List[DataInventoryItem]:
        """Get inventory items by data categories"""
        with sqlite3.connect(self.db_path) as conn:
            placeholders = ','.join('?' for _ in categories)
            cursor = conn.execute(f"""
                SELECT item_id, data_category, data_type, storage_location, retention_period_days,
                       legal_basis, purpose, data_controller, data_processor, encryption_status,
                       access_controls_json, sharing_agreements_json, last_updated
                FROM data_inventory 
                WHERE data_category IN ({placeholders})
            """, [cat.value for cat in categories])
            
            items = []
            for row in cursor.fetchall():
                item = DataInventoryItem(
                    item_id=row[0],
                    data_category=DataCategory(row[1]),
                    data_type=row[2],
                    storage_location=row[3],
                    retention_period_days=row[4],
                    legal_basis=row[5],
                    purpose=row[6],
                    data_controller=row[7],
                    data_processor=row[8],
                    encryption_status=bool(row[9]),
                    access_controls=json.loads(row[10]),
                    sharing_agreements=json.loads(row[11]),
                    last_updated=datetime.fromisoformat(row[12])
                )
                items.append(item)
            
            return items
    
    def _update_request_status(self, request_id: str, status: RequestStatus, notes: Optional[str] = None):
        """Update request status"""
        with sqlite3.connect(self.db_path) as conn:
            completed_at = datetime.utcnow() if status == RequestStatus.COMPLETED else None
            
            if notes:
                # Add note to processing notes
                cursor = conn.execute("""
                    SELECT processing_notes_json FROM data_subject_requests 
                    WHERE request_id = ?
                """, (request_id,))
                
                result = cursor.fetchone()
                if result:
                    existing_notes = json.loads(result[0])
                    existing_notes.append(f"{datetime.utcnow().isoformat()}: {notes}")
                    
                    conn.execute("""
                        UPDATE data_subject_requests 
                        SET status = ?, completed_at = ?, processing_notes_json = ?, updated_at = ?
                        WHERE request_id = ?
                    """, (status.value, completed_at, json.dumps(existing_notes), datetime.utcnow(), request_id))
            else:
                conn.execute("""
                    UPDATE data_subject_requests 
                    SET status = ?, completed_at = ?, updated_at = ?
                    WHERE request_id = ?
                """, (status.value, completed_at, datetime.utcnow(), request_id))
    
    def _send_request_confirmation(self, request: DataSubjectRequest):
        """Send request confirmation email"""
        try:
            subject = f"Privacy Request Confirmation - {request.request_id}"
            body = f"""
            Dear {request.data_subject_name or 'Data Subject'},
            
            We have received your privacy request with the following details:
            
            Request ID: {request.request_id}
            Request Type: {request.request_type.value.replace('_', ' ').title()}
            Submitted: {request.submitted_at.strftime('%Y-%m-%d %H:%M:%S')}
            Due Date: {request.due_date.strftime('%Y-%m-%d')}
            
            We will process your request within the required timeframe and notify you once completed.
            
            If you have any questions, please contact our privacy team at privacy@company.com
            
            Best regards,
            Privacy Team
            """
            
            # In real implementation, this would send actual email
            print(f"Confirmation email sent to {request.data_subject_email}")
            
        except Exception as e:
            print(f"Error sending confirmation email: {str(e)}")
    
    def _send_completion_notification(self, request: DataSubjectRequest):
        """Send completion notification"""
        try:
            subject = f"Privacy Request Completed - {request.request_id}"
            body = f"""
            Dear {request.data_subject_name or 'Data Subject'},
            
            Your privacy request has been completed:
            
            Request ID: {request.request_id}
            Request Type: {request.request_type.value.replace('_', ' ').title()}
            Completed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}
            
            """
            
            if request.request_type in [RequestType.ACCESS, RequestType.PORTABILITY]:
                body += "Your data has been prepared and will be sent to you separately.\n\n"
            
            body += """
            If you have any questions about this response, please contact our privacy team.
            
            Best regards,
            Privacy Team
            """
            
            # In real implementation, this would send actual email
            print(f"Completion notification sent to {request.data_subject_email}")
            
        except Exception as e:
            print(f"Error sending completion notification: {str(e)}")
    
    def get_privacy_dashboard(self) -> Dict[str, Any]:
        """Get privacy controls dashboard data"""
        with sqlite3.connect(self.db_path) as conn:
            # Request metrics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_requests,
                    SUM(CASE WHEN status = 'received' THEN 1 ELSE 0 END) as pending_requests,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_requests,
                    SUM(CASE WHEN due_date < datetime('now') AND status != 'completed' THEN 1 ELSE 0 END) as overdue_requests
                FROM data_subject_requests
                WHERE submitted_at > datetime('now', '-30 days')
            """)
            
            metrics = cursor.fetchone()
            
            # Requests by type
            cursor = conn.execute("""
                SELECT request_type, COUNT(*) as count
                FROM data_subject_requests
                WHERE submitted_at > datetime('now', '-30 days')
                GROUP BY request_type
            """)
            
            by_type = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Recent requests
            cursor = conn.execute("""
                SELECT request_id, request_type, status, data_subject_email, submitted_at, due_date
                FROM data_subject_requests
                ORDER BY submitted_at DESC
                LIMIT 10
            """)
            
            recent_requests = [
                {
                    'request_id': row[0],
                    'request_type': row[1],
                    'status': row[2],
                    'data_subject_email': row[3],
                    'submitted_at': row[4],
                    'due_date': row[5]
                }
                for row in cursor.fetchall()
            ]
            
            return {
                'metrics': {
                    'total_requests': metrics[0],
                    'pending_requests': metrics[1],
                    'completed_requests': metrics[2],
                    'overdue_requests': metrics[3]
                },
                'by_type': by_type,
                'recent_requests': recent_requests
            }
    
    def record_consent(self, 
                      data_subject_email: str,
                      purpose: str,
                      legal_basis: str,
                      consent_given: bool,
                      consent_method: str,
                      data_categories: List[DataCategory],
                      retention_period_days: Optional[int] = None) -> str:
        """Record consent for data processing"""
        consent_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO consent_records 
                (consent_id, data_subject_email, purpose, legal_basis, consent_given,
                 consent_date, consent_method, data_categories_json, retention_period_days)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                consent_id,
                data_subject_email,
                purpose,
                legal_basis,
                consent_given,
                datetime.utcnow(),
                consent_method,
                json.dumps([cat.value for cat in data_categories]),
                retention_period_days
            ))
        
        return consent_id
    
    def withdraw_consent(self, consent_id: str, withdrawal_method: str) -> bool:
        """Withdraw consent"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                UPDATE consent_records 
                SET consent_given = FALSE, withdrawn_date = ?, withdrawal_method = ?
                WHERE consent_id = ? AND consent_given = TRUE
            """, (datetime.utcnow(), withdrawal_method, consent_id))
            
            return cursor.rowcount > 0